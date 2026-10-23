"""Durable deployment state on PostgreSQL with an explicit fake container boundary."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch
from uuid import UUID, uuid4

from job_fixture import JobAPITest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from vf_api.deployment_manager import DeploymentManager
from vf_api.deployment_runtime import DeploymentUnavailable, DockerRuntime
from vf_api.models import Deployment, Model, TrainingJob
from vf_api.routers.deployments import get_manager, operation_lock
from vf_api.settings import DeploymentSettings
from vf_api.upload_dependencies import get_artifacts


class FakeRuntime:
    healthy = staticmethod(DockerRuntime.healthy)
    endpoint = staticmethod(DockerRuntime.endpoint)

    def __init__(self):
        self.containers = {}
        self.created_specs = []
        self.fail_start = False

    def image(self):
        return "sha256:" + "1" * 64

    def route_ready(self, info, spec):
        return True

    def inspect(self, spec):
        return self.containers.get(spec["id"])

    def create(self, spec, bundle, image):
        self.created_specs.append(spec.copy())
        self.containers[spec["id"]] = {
            "Id": uuid4().hex,
            "State": {"Running": False, "Health": {"Status": "starting"}},
            "NetworkSettings": {
                "Ports": {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "32001"}]}
            },
        }

    def start(self, spec):
        if self.fail_start:
            raise DeploymentUnavailable("private-runtime-sentinel")
        self.containers[spec["id"]]["State"] = {
            "Running": True,
            "Health": {"Status": "healthy"},
        }

    def remove(self, spec):
        self.containers.pop(spec["id"], None)


class FakeArtifacts:
    def download(self, key, output, size, sha256):
        data = b"test graph" if key.endswith("model.onnx") else b"{}"
        assert len(data) == size and hashlib.sha256(data).hexdigest() == sha256
        output.write(data)


class DeploymentTest(JobAPITest):
    def setUp(self):
        super().setUp()
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.runtime = FakeRuntime()
        self.manager = DeploymentManager(
            DeploymentSettings(
                deployment_work_dir=Path(temporary.name) / "deployments",
                deployment_ready_seconds=5,
            ),
            self.runtime,
        )
        self.store = FakeArtifacts()
        self.app.dependency_overrides[get_manager] = lambda: self.manager
        self.app.dependency_overrides[get_artifacts] = lambda: self.store
        self.addCleanup(self.app.dependency_overrides.clear)
        with Session(self.engine) as session:
            job = TrainingJob(
                project_id=self.project_id,
                dataset_version_id=self.version_id,
                model_arch="resnet18",
                hyperparams={},
                status="succeeded",
            )
            session.add(job)
            session.flush()
            prefix = f"{self.project_id}/jobs/{job.id}/{uuid4()}"
            index = {
                name: {
                    "key": f"{prefix}/{name}",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in (
                    ("onnx/model.onnx", b"test graph"),
                    ("onnx/metadata.json", b"{}"),
                )
            }
            model = Model(
                project_id=self.project_id,
                job_id=job.id,
                version=1,
                input_size=224,
                class_map={"ok": 0, "bad": 1},
                metrics={"artifacts": index},
                weights_key=f"{prefix}/run/best.pt",
                onnx_key=index["onnx/model.onnx"]["key"],
            )
            session.add(model)
            session.flush()
            self.model_id = model.id
            session.commit()
        self.addCleanup(self.clear_deployments)
        self.url = f"/api/v1/models/{self.model_id}/deployments"

    def clear_deployments(self):
        with Session(self.engine) as session:
            session.execute(
                delete(Deployment).where(Deployment.model_id == self.model_id)
            )
            session.commit()

    def deploy(self, **body):
        return self.client.post(self.url, headers=self.headers, json=body)

    def get(self, identifier, headers=None):
        return self.client.get(
            f"/api/v1/deployments/{identifier}",
            headers=self.headers if headers is None else headers,
        )

    def stop(self, identifier, headers=None):
        return self.client.delete(
            f"/api/v1/deployments/{identifier}",
            headers=self.headers if headers is None else headers,
        )

    def listing(self, **params):
        return self.client.get(
            f"/api/v1/projects/{self.project_id}/deployments",
            headers=self.headers,
            params=params,
        )

    def test_create_read_list_stop_and_one_time_key(self):
        response = self.deploy(slug="casting-test")
        self.assertEqual(response.status_code, 201, response.text)
        created = response.json()
        identifier = UUID(created["id"])
        key = created["api_key"]
        self.assertEqual(created["status"], "running")
        self.assertEqual(
            created["endpoint_url"], "http://127.0.0.1:32001/predict/casting-test"
        )
        self.assertTrue(created["runtime_checked"])
        with Session(self.engine) as session:
            row = session.get(Deployment, identifier)
            self.assertEqual(row.api_key_hash, hashlib.sha256(key.encode()).hexdigest())
            self.assertNotEqual(row.api_key_hash, key)
            self.assertFalse(session.get(Model, self.model_id).is_promoted)
        for result in (self.get(identifier), self.listing()):
            self.assertEqual(result.status_code, 200, result.text)
            self.assertNotIn(key, result.text)
            self.assertNotIn("api_key", result.text)
            self.assertNotIn("container_id", result.text)
            self.assertEqual(result.headers["cache-control"], "no-store")
        self.assertFalse(self.listing().json()["items"][0]["runtime_checked"])
        self.assertIsNone(self.listing().json()["items"][0]["endpoint_url"])
        self.assertEqual(
            (self.manager.root / str(identifier) / "model.onnx").read_bytes(),
            b"test graph",
        )
        stopped = self.stop(identifier)
        self.assertEqual(stopped.status_code, 200, stopped.text)
        self.assertEqual(stopped.json()["status"], "stopped")
        self.assertEqual(self.stop(identifier).json(), stopped.json())
        self.assertFalse((self.manager.root / str(identifier)).exists())
        self.assertFalse(self.runtime.containers)

    def test_capacity_slug_uniqueness_and_validation(self):
        first = self.deploy(slug="first").json()
        self.assertEqual(self.deploy(slug="first").status_code, 409)
        self.assertEqual(self.deploy(slug="second").status_code, 201)
        self.assertEqual(self.deploy().status_code, 409)
        self.stop(first["id"])
        self.assertEqual(self.deploy().status_code, 201)
        for body in (
            {"slug": "../invalid"},
            {"image": "arbitrary"},
            {"confidence_threshold": "0.5"},
            {"confidence_threshold": 2},
        ):
            self.assertEqual(self.deploy(**body).status_code, 422)
        self.assertEqual(self.listing(page_size=1).json()["total"], 3)
        self.assertEqual(len(self.listing(page=2, page_size=1).json()["items"]), 1)

    def test_owner_and_access_checks_precede_container_operations(self):
        identifier = self.deploy().json()["id"]
        with (
            patch.object(self.runtime, "image") as image,
            patch.object(self.runtime, "inspect") as inspect,
            patch.object(self.runtime, "remove") as remove,
        ):
            for headers, status in (
                ({}, 401),
                (self.auth(self.user_id, "refresh"), 401),
                (self.auth(self.other_id), 404),
            ):
                self.assertEqual(
                    self.client.post(self.url, headers=headers, json={}).status_code,
                    status,
                )
                self.assertEqual(self.get(identifier, headers).status_code, status)
                self.assertEqual(self.stop(identifier, headers).status_code, status)
            self.assertEqual(self.get(uuid4()).status_code, 404)
            image.assert_not_called()
            inspect.assert_not_called()
            remove.assert_not_called()

    def test_failed_startup_is_recorded_only_after_confirmed_cleanup(self):
        self.runtime.fail_start = True
        response = self.deploy()
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(response.json()["status"], "failed")
        self.assertIsNone(response.json()["api_key"])
        self.assertNotIn("private-runtime-sentinel", response.text)
        self.assertFalse(self.runtime.containers)
        self.assertFalse((self.manager.root / response.json()["id"]).exists())

    def test_uncertain_cleanup_keeps_capacity_reserved_until_reconciled(self):
        self.runtime.fail_start = True
        with patch.object(
            self.runtime,
            "remove",
            side_effect=DeploymentUnavailable("private-runtime-sentinel"),
        ):
            response = self.deploy()
        self.assertEqual(response.status_code, 503)
        row = self.listing().json()["items"][0]
        self.assertEqual(row["status"], "starting")
        with Session(self.engine) as session:
            session.get(Deployment, UUID(row["id"])).created_at -= timedelta(minutes=10)
            session.commit()
        recovered = self.get(row["id"])
        self.assertEqual(recovered.json()["status"], "failed")
        self.assertFalse(self.runtime.containers)

    def test_status_read_recovers_ready_start_and_detects_missing_container(self):
        identifier = UUID(self.deploy().json()["id"])
        with Session(self.engine) as session:
            row = session.get(Deployment, identifier)
            row.status = "starting"
            row.container_id = None
            session.commit()
        self.assertEqual(self.get(identifier).json()["status"], "running")
        self.runtime.containers.pop(identifier)
        self.assertEqual(self.get(identifier).json()["status"], "failed")

    def test_read_and_stop_during_launch_do_not_race_container_creation(self):
        started, release = Event(), Event()
        original = self.manager.launch

        def paused(*args):
            started.set()
            release.wait(10)
            return original(*args)

        with patch.object(self.manager, "launch", side_effect=paused):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.deploy)
                try:
                    self.assertTrue(started.wait(5))
                    row = self.listing().json()["items"][0]
                    self.assertEqual(self.get(row["id"]).json()["status"], "starting")
                    self.assertFalse(self.get(row["id"]).json()["runtime_checked"])
                    self.assertEqual(self.stop(row["id"]).status_code, 409)
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=10).json()["status"], "running")

    def test_stop_does_not_claim_success_when_docker_is_unavailable(self):
        identifier = self.deploy().json()["id"]
        with patch.object(
            self.runtime,
            "remove",
            side_effect=DeploymentUnavailable("private-runtime-sentinel"),
        ):
            response = self.stop(identifier)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private-runtime-sentinel", response.text)
        self.assertEqual(self.listing().json()["items"][0]["status"], "running")

    def test_gateway_outage_keeps_healthy_model_running(self):
        identifier = UUID(self.deploy().json()["id"])
        self.runtime.containers[identifier]["Config"] = {
            "Labels": {"traefik.enable": "true"}
        }
        with patch.object(self.runtime, "route_ready", return_value=False):
            response = self.get(identifier)
        self.assertEqual(response.status_code, 503)
        self.assertIn(identifier, self.runtime.containers)
        self.assertEqual(self.listing().json()["items"][0]["status"], "running")

    def test_reservation_gap_keeps_starting_and_concurrent_stops_are_safe(self):
        identifier = UUID(self.deploy().json()["id"])
        self.runtime.containers.pop(identifier)
        with Session(self.engine) as session:
            row = session.get(Deployment, identifier)
            row.status = "starting"
            row.container_id = None
            session.commit()
        self.assertEqual(self.get(identifier).json()["status"], "starting")
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(self.stop, [identifier, identifier]))
        self.assertTrue(all(r.status_code in (200, 409) for r in responses))
        self.assertEqual(self.stop(identifier).json()["status"], "stopped")

    def test_capacity_reservation_lock_prevents_parallel_overcommit(self):
        with Session(self.engine) as session:
            self.assertTrue(operation_lock(session, "capacity"))
            self.assertEqual(self.deploy().status_code, 409)
        self.assertEqual(self.listing().json()["total"], 0)
