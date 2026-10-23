"""Opt-in checks against local Docker, PostgreSQL and private S3 storage."""

import io
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import UUID, uuid4

from bundle_fixture import make_bundle
import httpx2 as httpx
from job_fixture import JobAPITest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from vf_api.deployment_manager import DeploymentManager
from vf_api.database import make_session_factory
from vf_api.models import Deployment, Model, TrainingJob
from vf_api.routers.deployments import get_manager
from vf_api.settings import DeploymentSettings, StorageSettings
from vf_api.storage import ArtifactStore, PredictionStore
from vf_api.upload_dependencies import get_artifacts, get_predictions
from vf_worker.predictions import collect_predictions


class DockerDeploymentTest(JobAPITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.store = ArtifactStore(
            StorageSettings(s3_artifacts_bucket="vf-test-deploy-" + uuid4().hex)
        )
        cls.addClassCleanup(cls.store.close)
        cls.store.client.create_bucket(Bucket=cls.store.bucket)
        cls.addClassCleanup(cls.drop_bucket)
        cls.predictions = PredictionStore(
            StorageSettings(s3_predictions_bucket=cls.store.bucket)
        )
        cls.addClassCleanup(cls.predictions.close)

    @classmethod
    def drop_bucket(cls):
        for page in cls.store.client.get_paginator("list_objects_v2").paginate(
            Bucket=cls.store.bucket
        ):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if objects:
                cls.store.client.delete_objects(
                    Bucket=cls.store.bucket, Delete={"Objects": objects}
                )
        cls.store.client.delete_bucket(Bucket=cls.store.bucket)

    def setUp(self):
        super().setUp()
        Path("artifacts").mkdir(exist_ok=True)
        temporary = TemporaryDirectory(dir="artifacts", prefix="deployment-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).absolute()
        self.manager = DeploymentManager(
            DeploymentSettings(
                deployment_work_dir=self.directory / "deployments",
                deployment_namespace="vftest-" + uuid4().hex[:12],
                prediction_log_dir=self.directory / "predictions",
            )
        )
        self.app.dependency_overrides[get_manager] = lambda: self.manager
        self.app.dependency_overrides[get_artifacts] = lambda: self.store
        self.app.dependency_overrides[get_predictions] = lambda: self.predictions
        self.addCleanup(self.app.dependency_overrides.clear)
        self.addCleanup(self.clean_deployments)

    def clean_deployments(self):
        with Session(self.engine) as session:
            rows = session.scalars(
                select(Deployment)
                .join(Model)
                .where(Model.project_id == self.project_id)
            ).all()
            for row in rows:
                self.manager.cleanup(
                    {
                        "id": row.id,
                        "model_id": row.model_id,
                        "project_id": self.project_id,
                    }
                )
                row.status = "stopped"
            session.commit()

    def publish(self, bundle=None):
        if bundle is None:
            bundle = self.directory / "source"
            bundle.mkdir()
            make_bundle(bundle)
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
                f"onnx/{name}": self.store.put_file_once(
                    f"{prefix}/onnx/{name}", bundle / name
                )
                for name in ("model.onnx", "metadata.json")
            }
            model = Model(
                project_id=self.project_id,
                job_id=job.id,
                version=1,
                input_size=32,
                class_map={"red": 0, "green": 1},
                metrics={"artifacts": index},
                weights_key=f"{prefix}/run/best.pt",
                onnx_key=index["onnx/model.onnx"]["key"],
            )
            session.add(model)
            session.flush()
            self.model_id = model.id
            session.commit()

    def deploy(self):
        response = self.client.post(
            f"/api/v1/models/{self.model_id}/deployments", headers=self.headers, json={}
        )
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()

    def test_real_prediction_security_and_stop(self):
        self.publish()
        created = self.deploy()
        self.assertEqual(created["status"], "running")
        spec = {
            "id": UUID(created["id"]),
            "model_id": self.model_id,
            "project_id": self.project_id,
        }
        info = self.manager.runtime.inspect(spec)
        self.assertEqual(info["Config"]["User"], "10001:10001")
        self.assertTrue(info["HostConfig"]["ReadonlyRootfs"])
        self.assertEqual(info["HostConfig"]["Memory"], 1536 * 1024**2)
        self.assertEqual(info["HostConfig"]["CapDrop"], ["ALL"])
        self.assertIn("no-new-privileges", info["HostConfig"]["SecurityOpt"])
        self.assertEqual(
            {m["Destination"]: m["RW"] for m in info["Mounts"]},
            {"/bundle": False, "/outbox": True},
        )
        self.assertNotIn(created["api_key"], "\n".join(info["Config"]["Env"]))
        self.assertFalse(info["HostConfig"]["PortBindings"])
        self.assertEqual(
            created["endpoint_url"],
            f"{str(self.manager.settings.deployment_gateway_url).rstrip('/')}/predict/{created['slug']}",
        )
        output = io.BytesIO()
        Image.new("RGB", (48, 24), "red").save(output, format="PNG")
        with httpx.Client(timeout=30, trust_env=False) as client:
            self.assertEqual(client.post(created["endpoint_url"]).status_code, 401)
            preflight = client.options(
                created["endpoint_url"],
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "X-API-Key,Content-Type",
                },
            )
            self.assertEqual(preflight.status_code, 200)
            self.assertEqual(
                preflight.headers["access-control-allow-origin"],
                "http://localhost:3000",
            )
            rejected = client.options(
                created["endpoint_url"],
                headers={
                    "Origin": "https://untrusted.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            self.assertNotIn("access-control-allow-origin", rejected.headers)
            self.assertEqual(
                client.post(created["endpoint_url"] + "/extra").status_code, 404
            )
            self.assertEqual(
                client.get(
                    str(self.manager.settings.deployment_gateway_url).rstrip("/")
                    + "/docs"
                ).status_code,
                404,
            )
            prediction = client.post(
                created["endpoint_url"],
                headers={
                    "X-API-Key": created["api_key"],
                    "Origin": "http://localhost:3000",
                },
                files={"image": ("red.png", output.getvalue(), "image/png")},
            )
        self.assertEqual(prediction.status_code, 200, prediction.text)
        self.assertEqual(
            prediction.headers["access-control-allow-origin"], "http://localhost:3000"
        )
        self.assertEqual(prediction.json()["label"], "red")
        self.assertEqual(prediction.json()["model_id"], str(self.model_id))
        prediction_id = prediction.json()["prediction_id"]
        event_path = (
            self.manager.settings.prediction_log_dir
            / created["id"]
            / f"{prediction_id}.json"
        )
        self.assertTrue(event_path.exists())
        path = f"/api/v1/deployments/{created['id']}"
        self.assertEqual(
            self.client.get(path, headers=self.headers).json()["endpoint_url"],
            created["endpoint_url"],
        )
        self.assertEqual(
            self.client.delete(path, headers=self.headers).json()["status"], "stopped"
        )
        self.assertIsNone(self.manager.runtime.inspect(spec))
        self.assertFalse((self.manager.root / created["id"]).exists())
        # Stopping the model must retain pending logs until the CPU worker commits them.
        self.assertTrue(event_path.exists())
        self.assertEqual(
            collect_predictions(
                self.manager.settings.prediction_log_dir,
                make_session_factory(self.engine),
                self.predictions,
            )["collected"],
            1,
        )
        self.assertFalse(event_path.exists())
        stats = self.client.get(path + "/stats", headers=self.headers)
        self.assertEqual(stats.json()["requests"], 1)
        self.assertEqual(stats.json()["class_distribution"]["red"], 1)
        history = self.client.get(path + "/predictions", headers=self.headers).json()
        self.assertEqual(history["items"][0]["id"], prediction_id)
        with httpx.Client(timeout=5, trust_env=False) as client:
            self.assertEqual(
                client.get(history["items"][0]["image_url"]).content, output.getvalue()
            )
        with httpx.Client(timeout=5, trust_env=False) as client:
            deadline = time.monotonic() + 10
            while client.post(created["endpoint_url"]).status_code != 404:
                self.assertLess(
                    time.monotonic(), deadline, "Stopped route was not removed"
                )
                time.sleep(0.2)

    def test_two_routes_keep_keys_separate_and_stopping_one_preserves_other(self):
        self.publish()
        first, second = self.deploy(), self.deploy()
        self.assertEqual(first["status"], "running")
        self.assertEqual(second["status"], "running")
        image = io.BytesIO()
        Image.new("RGB", (32, 32), "green").save(image, format="PNG")
        with httpx.Client(timeout=30, trust_env=False) as client:
            self.assertEqual(
                client.post(
                    first["endpoint_url"], headers={"X-API-Key": second["api_key"]}
                ).status_code,
                401,
            )
            self.assertEqual(
                client.post(
                    second["endpoint_url"], headers={"X-API-Key": first["api_key"]}
                ).status_code,
                401,
            )
            stopped = self.client.delete(
                f"/api/v1/deployments/{first['id']}", headers=self.headers
            )
            self.assertEqual(stopped.json()["status"], "stopped")
            result = client.post(
                second["endpoint_url"],
                headers={"X-API-Key": second["api_key"]},
                files={"image": ("green.png", image.getvalue(), "image/png")},
            )
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()["label"], "green")

    def test_container_exit_reconciles_to_failed(self):
        self.publish()
        created = self.deploy()
        self.assertEqual(created["status"], "running")
        spec = {
            "id": UUID(created["id"]),
            "model_id": self.model_id,
            "project_id": self.project_id,
        }
        info = self.manager.runtime.inspect(spec)
        self.manager.runtime.command("container", "stop", "--time", "5", info["Id"])
        response = self.client.get(
            f"/api/v1/deployments/{created['id']}", headers=self.headers
        )
        self.assertEqual(response.json()["status"], "failed")
        self.assertIsNone(self.manager.runtime.inspect(spec))

    def test_invalid_bundle_cleans_failed_startup(self):
        bundle = self.directory / "invalid"
        bundle.mkdir()
        make_bundle(bundle)
        (bundle / "metadata.json").write_text("{}")
        self.publish(bundle)
        created = self.deploy()
        self.assertEqual(created["status"], "failed")
        self.assertIsNone(created["api_key"])
        self.assertFalse((self.manager.root / created["id"]).exists())

    def test_changed_storage_bytes_never_reach_docker_create(self):
        self.publish()
        with Session(self.engine) as session:
            key = session.get(Model, self.model_id).onnx_key
        self.store.client.put_object(Bucket=self.store.bucket, Key=key, Body=b"changed")
        with patch.object(self.manager.runtime, "create") as create:
            result = self.deploy()
            create.assert_not_called()
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["api_key"])
        self.assertFalse(list(self.manager.root.iterdir()))
