"""Worker state and storage boundaries; the controlled executor does not train ML."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch
from uuid import UUID, uuid4

from botocore.exceptions import ClientError
from job_fixture import JobAPITest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_dataset_snapshots import snapshot_fixture

from vf_api.database import make_session_factory
from vf_api.dataset_snapshots import build_snapshot
from vf_api.models import DatasetVersion, Model, Project, TrainingJob, TrainingMetric
from vf_api.settings import StorageSettings
from vf_api.storage import ArtifactStore, DatasetStore, ObjectStore
from vf_api.version_schemas import VersionCreate
from vf_worker.settings import WorkerSettings
from vf_worker.training import (
    Cancelled,
    WorkerBusy,
    execute,
    gpu_slot,
    record_history,
    run_job,
)


def fake_pipeline(spec, settings, log, check, progress):
    """Only synthesize the trusted subprocess's contract for orchestration tests."""
    work = Path(spec["work"])
    (work / "run").mkdir()
    (work / "onnx").mkdir()
    (work / "run/best.pt").write_bytes(b"test weights, not a model")
    (work / "onnx/model.onnx").write_bytes(b"test graph, not ONNX")
    manifest = json.loads((work / "manifest.json").read_text())
    manifest_sha = hashlib.sha256((work / "manifest.json").read_bytes()).hexdigest()
    weight_sha = hashlib.sha256((work / "run/best.pt").read_bytes()).hexdigest()
    onnx_sha = hashlib.sha256((work / "onnx/model.onnx").read_bytes()).hexdigest()
    provenance = dict(
        manifest_sha256=manifest_sha,
        checkpoint_sha256=weight_sha,
        task_type=manifest["task_type"],
    )
    files = {
        "run/run.json": {"status": "completed"},
        "run/history.json": [
            {
                "epoch": 1,
                "train": {"loss": 1.0, "accuracy": 0.5},
                "val": {"loss": 0.9, "accuracy": 0.6},
                "learning_rate": 0.0003,
            }
        ],
        "evaluation.json": {
            **provenance,
            "split": "test",
            "metrics": {"accuracy": 0.6},
        },
        "onnx/metadata.json": {
            **provenance,
            "model_sha256": onnx_sha,
            "class_map": manifest["class_map"],
            "opset": 17,
            "input": {"shape": ["batch", 3, 224, 224]},
        },
        "onnx/verification.json": {
            "model_sha256": onnx_sha,
            "latency": {"mean_ms": 1},
            "parity": {
                "passed": True,
                "meets_32_test_image_check": True,
                "split": "test",
                "samples": 32,
                "batch_sizes": [1, 8],
                "absolute_tolerance": 0.001,
                "relative_tolerance": 0,
                "max_absolute_error": 0.00001,
                "image_keys": [
                    sample["image_key"]
                    for sample in manifest["samples"]
                    if sample["split"] == "test"
                ],
            },
        },
    }
    if manifest["task_type"] == "detection":
        files["run/history.json"] = [
            {
                "epoch": 1,
                "fitness": 0.4,
                "metrics": {
                    "train/box_loss": 0.8,
                    "val/box_loss": 0.7,
                    "metrics/mAP50(B)": 0.5,
                    "lr/pg0": 0.01,
                },
            }
        ]
        files["evaluation.json"]["metrics"] = {"map50": 0.5}
        files["onnx/metadata.json"]["input"]["shape"] = ["batch", 3, 640, 640]
        files["onnx/verification.json"]["parity"].update(
            absolute_tolerance=0.002,
            box_absolute_tolerance=0.002,
            score_absolute_tolerance=0.001,
            max_box_absolute_error=0.0015,
            max_score_absolute_error=0.00001,
        )
    for name, value in files.items():
        (work / name).write_text(json.dumps(value))
    check()
    progress()


class TrainingWorkerTest(JobAPITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        suffix = uuid4().hex
        cls.storage_settings = StorageSettings(
            s3_images_bucket="vf-test-images-" + suffix,
            s3_datasets_bucket="vf-test-data-" + suffix,
            s3_artifacts_bucket="vf-test-artifacts-" + suffix,
        )
        cls.images, cls.datasets, cls.artifacts = [
            store(cls.storage_settings)
            for store in (ObjectStore, DatasetStore, ArtifactStore)
        ]
        for store in (cls.images, cls.datasets, cls.artifacts):
            cls.addClassCleanup(store.close)
            store.client.create_bucket(Bucket=store.bucket)
            cls.addClassCleanup(cls.drop_bucket, store)
        cls.sessions = make_session_factory(cls.engine)

    @classmethod
    def drop_bucket(cls, store):
        for page in store.client.get_paginator("list_objects_v2").paginate(
            Bucket=store.bucket
        ):
            objects = [{"Key": row["Key"]} for row in page.get("Contents", [])]
            if objects:
                store.client.delete_objects(
                    Bucket=store.bucket, Delete={"Objects": objects}
                )
        store.client.delete_bucket(Bucket=store.bucket)

    def setUp(self):
        super().setUp()
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.configuration = WorkerSettings(
            worker_ml_python=Path(sys.executable),
            worker_work_dir=root / "jobs",
            worker_cache_dir=root / "cache",
            worker_gpu_id=0,
            worker_device="cpu",
        )
        self.make_dataset()

    def make_dataset(self, task="classification", number=2):
        snapshot, objects = snapshot_fixture(task=task, count=40)
        for row in snapshot.images:
            old = row["image_key"]
            row["image_key"] = old.replace("project/", f"{self.project_id}/", 1)
            self.images.put(row["image_key"], objects[old], "image/png")
        self.first_key = snapshot.images[0]["image_key"]
        options = VersionCreate(split_ratios={"train": 0.1, "val": 0.1, "test": 0.8})
        manifest, audit, counts = build_snapshot(snapshot, options, self.images)
        self.manifest_key = f"{self.project_id}/v{number}/{uuid4()}/manifest.json"
        counts["audit_key"] = self.manifest_key.replace("manifest.json", "audit.json")
        self.datasets.put_once(self.manifest_key, manifest)
        self.datasets.put_once(counts["audit_key"], audit)
        with Session(self.engine) as session:
            session.get(Project, self.project_id).task_type = task
            version = DatasetVersion(
                project_id=self.project_id,
                version=number,
                manifest_key=self.manifest_key,
                split_ratios=options.split_ratios.model_dump(),
                counts=counts,
            )
            session.add(version)
            session.flush()
            self.version_id = version.id
            session.commit()

    def submit(self, model_arch="resnet18"):
        response = self.create(
            model_arch=model_arch, hyperparams={"epochs": 1, "batch_size": 2}
        )
        self.assertEqual(response.status_code, 201, response.text)
        identifier = UUID(response.json()["id"])
        with Session(self.engine) as session:
            task_id = session.get(TrainingJob, identifier).celery_task_id
        return identifier, task_id

    def run_job(self, identifier, task_id, executor=fake_pipeline):
        return run_job(
            identifier,
            task_id,
            self.sessions,
            self.configuration,
            self.datasets,
            self.images,
            self.artifacts,
            executor=executor,
        )

    def models(self):
        with Session(self.engine) as session:
            return list(
                session.scalars(
                    select(Model).where(Model.project_id == self.project_id)
                )
            )

    def test_claim_metrics_private_artifacts_and_single_model_registration(self):
        identifier, task_id = self.submit()
        result = self.run_job(identifier, task_id)
        self.assertIsNotNone(result)
        status = self.read(identifier).json()
        self.assertEqual(status["status"], "succeeded", status)
        self.assertEqual(status["model_id"], result)
        model = self.models()[0]
        self.assertEqual(model.class_map, {"ok": 0, "defect": 1})
        self.assertEqual(model.version, 1)
        self.assertEqual(model.input_size, 224)
        self.assertEqual(model.metrics["test"], {"accuracy": 0.6})
        self.assertEqual(len(model.metrics["artifacts"]), 9)
        for artifact in model.metrics["artifacts"].values():
            data = self.artifacts.read(artifact["key"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), artifact["sha256"])
        with self.assertRaises(ClientError) as error:
            path = Path(self.temporary.name) / "other.json"
            path.write_text("{}")
            self.artifacts.put_file_once(model.onnx_key, path)
        self.assertEqual(
            error.exception.response["ResponseMetadata"]["HTTPStatusCode"], 412
        )
        self.assertIsNone(self.run_job(identifier, task_id))
        self.assertEqual(len(self.models()), 1)
        metrics = self.client.get(
            f"/api/v1/jobs/{identifier}/metrics", headers=self.headers
        ).json()
        self.assertEqual(metrics["total"], 5)
        self.assertEqual(self.cancel(identifier).json(), status)
        self.assertEqual(
            self.client.get(self.path, headers=self.headers).json()["items"][0][
                "model_id"
            ],
            result,
        )
        self.assertEqual(
            list(self.configuration.worker_work_dir.glob(f"{identifier}-*")), []
        )

    def test_cancelled_and_wrong_task_messages_do_not_claim_or_download(self):
        identifier, task_id = self.submit()
        with patch.object(self.datasets, "read") as read:
            self.assertIsNone(self.run_job(identifier, str(uuid4())))
            self.cancel(identifier)
            self.assertIsNone(self.run_job(identifier, task_id))
            read.assert_not_called()
        self.assertEqual(self.models(), [])

    def test_concurrent_redelivery_cannot_execute_twice(self):
        identifier, task_id = self.submit()
        entered, resume = Event(), Event()

        def waiting(*args):
            entered.set()
            if not resume.wait(10):
                raise RuntimeError("Executor not resumed")
            fake_pipeline(*args)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.run_job, identifier, task_id, waiting)
            try:
                self.assertTrue(entered.wait(10))
                self.assertIsNone(
                    pool.submit(self.run_job, identifier, task_id).result(5)
                )
            finally:
                resume.set()
            self.assertIsNotNone(first.result(10))
        self.assertEqual(len(self.models()), 1)

    def test_gpu_lock_does_not_claim_a_second_job(self):
        identifier, task_id = self.submit()
        with gpu_slot(self.configuration):
            with self.assertRaises(WorkerBusy):
                self.run_job(identifier, task_id)
        self.assertEqual(self.read(identifier).json()["status"], "queued")

    def test_detector_registration_keeps_separate_box_and_score_gates(self):
        self.make_dataset(task="detection", number=3)
        identifier, task_id = self.submit("yolov8s")
        self.assertIsNotNone(self.run_job(identifier, task_id))
        self.assertEqual(self.models()[0].input_size, 640)
        for field, value in (
            ("max_box_absolute_error", 0.0021),
            ("max_score_absolute_error", 0.0011),
        ):

            def bad_parity(*args):
                fake_pipeline(*args)
                path = Path(args[0]["work"]) / "onnx/verification.json"
                report = json.loads(path.read_text())
                report["parity"][field] = value
                path.write_text(json.dumps(report))

            identifier, task_id = self.submit("yolov8s")
            self.assertIsNone(self.run_job(identifier, task_id, bad_parity))
            self.assertEqual(self.read(identifier).json()["status"], "failed")
        self.assertEqual(len(self.models()), 1)

    def test_concurrent_jobs_allocate_distinct_model_versions(self):
        jobs = [self.submit(), self.submit()]
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    run_job,
                    identifier,
                    task_id,
                    self.sessions,
                    self.configuration.model_copy(update={"worker_gpu_id": index}),
                    self.datasets,
                    self.images,
                    self.artifacts,
                    executor=fake_pipeline,
                )
                for index, (identifier, task_id) in enumerate(jobs)
            ]
            self.assertTrue(all(future.result(15) is not None for future in futures))
        self.assertEqual(sorted(model.version for model in self.models()), [1, 2])

    def test_changed_original_or_manifest_fails_before_ml_and_hides_exception_details(
        self,
    ):
        for store, key in (
            (self.images, self.first_key),
            (self.datasets, self.manifest_key),
        ):
            original = store.read(key)
            store.put(key, b"corrupted", "application/octet-stream")
            identifier, task_id = self.submit()
            try:
                with patch(__name__ + ".fake_pipeline") as executor:
                    self.run_job(identifier, task_id, executor)
                    executor.assert_not_called()
                response = self.read(identifier).json()
                self.assertEqual(response["status"], "failed")
                self.assertEqual(response["error"], "Dataset verification failed")
            finally:
                store.put(key, original, "application/octet-stream")
        self.assertEqual(self.models(), [])

    def test_cancellation_during_training_acknowledges_only_after_executor_stops(self):
        identifier, task_id = self.submit()

        def cancelling(spec, settings, log, check, progress):
            response = self.cancel(identifier).json()
            self.assertEqual(response["status"], "running")
            self.assertIsNone(response["finished_at"])
            check()

        self.run_job(identifier, task_id, cancelling)
        self.assertEqual(self.read(identifier).json()["status"], "cancelled")
        self.assertEqual(self.models(), [])

    def test_failed_export_and_partial_upload_never_register_models(self):
        def bad_export(*args):
            fake_pipeline(*args)
            report = Path(args[0]["work"]) / "onnx/verification.json"
            value = json.loads(report.read_text())
            value["parity"]["passed"] = False
            report.write_text(json.dumps(value))

        identifier, task_id = self.submit()
        self.run_job(identifier, task_id, bad_export)
        self.assertEqual(self.read(identifier).json()["status"], "failed")
        identifier, task_id = self.submit()
        original = self.artifacts.put_file_once

        def fail_weights(key, path):
            if key.endswith("best.pt"):
                raise RuntimeError("private-error-details")
            return original(key, path)

        with patch.object(self.artifacts, "put_file_once", side_effect=fail_weights):
            self.run_job(identifier, task_id)
        response = self.read(identifier).json()
        self.assertEqual(response["error"], "Artifact publication failed")
        self.assertEqual(self.models(), [])

    def test_late_cancel_during_upload_prevents_registration(self):
        identifier, task_id = self.submit()
        original = self.artifacts.put_file_once

        def cancel_after_write(key, path):
            result = original(key, path)
            self.cancel(identifier)
            return result

        with patch.object(
            self.artifacts, "put_file_once", side_effect=cancel_after_write
        ):
            self.run_job(identifier, task_id)
        self.assertEqual(self.read(identifier).json()["status"], "cancelled")
        self.assertEqual(self.models(), [])

    def test_detector_epoch_metrics_are_durable_and_repeat_safe(self):
        identifier, _ = self.submit()
        with Session(self.engine) as session:
            session.get(TrainingJob, identifier).status = "running"
            session.commit()
        path = Path(self.temporary.name) / "history.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "epoch": 1,
                        "fitness": 0.4,
                        "metrics": {
                            "train/box_loss": 0.8,
                            "val/box_loss": 0.7,
                            "metrics/mAP50(B)": 0.5,
                            "lr/pg0": 0.01,
                        },
                    }
                ]
            )
        )
        for _ in range(2):
            self.assertEqual(record_history(self.sessions, identifier, path, 0), 1)
        with Session(self.engine) as session:
            rows = list(
                session.scalars(
                    select(TrainingMetric).where(TrainingMetric.job_id == identifier)
                )
            )
            self.assertEqual(len(rows), 5)
            self.assertIn(
                ("val", "mAP50(B)", 0.5),
                [(row.split, row.name, row.value) for row in rows],
            )

    def test_supervisor_kills_real_child_on_cancellation(self):
        original = subprocess.Popen
        processes = []

        def sleeping(*args, **kwargs):
            self.assertEqual(
                args[0][0], str(self.configuration.worker_ml_python.absolute())
            )
            process = original(
                [sys.executable, "-c", "import time; time.sleep(30)"], **kwargs
            )
            processes.append(process)
            return process

        checks = 0

        def cancel_after_spawn():
            nonlocal checks
            checks += 1
            if checks > 1:
                raise Cancelled()

        root = Path(self.temporary.name)
        # Virtualenv executables are symlinks; following them loses the venv packages.
        interpreter = root / "venv-python"
        interpreter.symlink_to(sys.executable)
        self.configuration.worker_ml_python = interpreter
        with patch("vf_worker.training.subprocess.Popen", side_effect=sleeping):
            with self.assertRaises(Cancelled):
                execute(
                    {"work": str(root)},
                    self.configuration,
                    root / "process.log",
                    cancel_after_spawn,
                    lambda: None,
                )
        self.assertLess(processes[0].returncode, 0)
