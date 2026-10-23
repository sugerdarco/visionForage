"""Real private S3 downloads and typed reports; fixture bytes are not ML models."""

import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from botocore.exceptions import EndpointConnectionError
import httpx2 as httpx
from job_fixture import JobAPITest
from sqlalchemy.orm import Session

from vf_api.models import Model, Project, TrainingJob
from vf_api.settings import StorageSettings
from vf_api.storage import ArtifactStore
from vf_api.upload_dependencies import get_artifacts


class ModelArtifactTest(JobAPITest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.store = ArtifactStore(
            StorageSettings(s3_artifacts_bucket="vf-test-models-" + uuid4().hex)
        )
        cls.addClassCleanup(cls.store.close)
        cls.store.client.create_bucket(Bucket=cls.store.bucket)
        cls.addClassCleanup(cls.drop_bucket)
        cls.app.dependency_overrides[get_artifacts] = lambda: cls.store

    @classmethod
    def drop_bucket(cls):
        for page in cls.store.client.get_paginator("list_objects_v2").paginate(
            Bucket=cls.store.bucket
        ):
            objects = [{"Key": row["Key"]} for row in page.get("Contents", [])]
            if objects:
                cls.store.client.delete_objects(
                    Bucket=cls.store.bucket, Delete={"Objects": objects}
                )
        cls.store.client.delete_bucket(Bucket=cls.store.bucket)

    def publish(self, *, detection=False):
        task = "detection" if detection else "classification"
        with Session(self.engine) as session:
            session.get(Project, self.project_id).task_type = task
            job = TrainingJob(
                project_id=self.project_id,
                dataset_version_id=self.version_id,
                model_arch="yolov8s" if detection else "resnet18",
                hyperparams={},
                status="succeeded",
            )
            session.add(job)
            session.flush()
            job_id = job.id
            session.commit()
        class_map = {"ok": 0, "defect": 1}
        weights, graph, manifest = (
            b"fixture weights",
            b"fixture ONNX",
            b"fixture manifest",
        )

        def digest(data):
            return hashlib.sha256(data).hexdigest()

        counts = dict(
            true_positives=0,
            false_positives=0,
            false_negatives=0,
            support=0,
            predictions=0,
            precision=0.0,
            recall=0.0,
        )
        scores = dict(support=16, precision=1.0, recall=1.0, f1=1.0, roc_auc=None)
        metrics = {
            "samples": 32,
            "accuracy": 1.0,
            "loss": 0.1,
            "macro_f1": 1.0,
            "roc_auc": None,
            "roc_auc_method": "binary",
            "positive_class": "defect",
            "zero_division": 0,
            "per_class": {name: scores for name in class_map},
            "confusion_matrix": {
                "class_names": list(class_map),
                "rows": "true",
                "columns": "predicted",
                "counts": [[16, 0], [0, 16]],
            },
            "confident_mistakes": [{"image_key": "private-report-sentinel"}],
        }
        if detection:
            metrics = {
                "map50": None,
                "map50_95": None,
                "fixed_threshold_micro": counts,
                "per_class": [
                    {
                        "class_name": name,
                        "class_index": index,
                        "ap50": None,
                        "ap50_95": None,
                        **counts,
                    }
                    for name, index in class_map.items()
                ],
            }
        self.evaluation = {
            "task_type": task,
            "split": "test",
            "checkpoint_epoch": 3,
            "manifest_sha256": digest(manifest),
            "checkpoint_sha256": digest(weights),
            "selection_val_loss": 0.1,
            "selection_split": "val",
            "class_map": class_map,
            "metrics": metrics,
            "runtime": {"private": "private-report-sentinel"},
            "loader_audit": {"private": "private-report-sentinel"},
            "metric_settings": {
                "ap_confidence_floor": 0.001,
                "ap_iou_thresholds": [0.5, 0.95],
                "precision_recall_confidence": 0.25,
                "precision_recall_iou": 0.5,
                "nms_iou": 0.7,
                "max_detections_per_image": 300,
            },
        }
        self.verification = {
            "model_sha256": digest(graph),
            "parity": {
                "passed": True,
                "split": "test",
                "samples": 32,
                "batch_sizes": [1, 8],
                "absolute_tolerance": 0.002 if detection else 0.001,
                "relative_tolerance": 0.0,
                "max_absolute_error": 0.0001,
                "image_keys": ["private-report-sentinel"],
            },
            "latency": {
                "batch_size": 1,
                "warmup_runs": 10,
                "measured_runs": 200,
                "mean_ms": 20.0,
                "p95_ms": 22.0,
                "private": "private-report-sentinel",
            },
        }
        if detection:
            self.verification["parity"].update(
                box_absolute_tolerance=0.002,
                score_absolute_tolerance=0.001,
                max_box_absolute_error=0.0001,
                max_score_absolute_error=0.000001,
            )
        files = {
            "run/best.pt": weights,
            "onnx/model.onnx": graph,
            "manifest.json": manifest,
            "onnx/metadata.json": json.dumps({"task_type": task, "opset": 17}).encode(),
            "evaluation.json": json.dumps(self.evaluation).encode(),
            "onnx/verification.json": json.dumps(self.verification).encode(),
        }
        self.files = files
        self.index = {}
        prefix = f"{self.project_id}/jobs/{job_id}/{uuid4()}"
        with TemporaryDirectory() as directory:
            for name, data in files.items():
                path = Path(directory) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                self.index[name] = self.store.put_file_once(f"{prefix}/{name}", path)
        with Session(self.engine) as session:
            row = Model(
                project_id=self.project_id,
                job_id=job_id,
                version=1,
                input_size=640 if detection else 224,
                class_map=class_map,
                metrics={"artifacts": self.index},
                weights_key=self.index["run/best.pt"]["key"],
                onnx_key=self.index["onnx/model.onnx"]["key"],
            )
            session.add(row)
            session.flush()
            self.model_id = row.id
            session.commit()
        self.path = f"/api/v1/models/{self.model_id}"

    def get(self, suffix, headers=None):
        return self.client.get(
            self.path + suffix, headers=self.headers if headers is None else headers
        )

    def replace_report(self, name, report):
        data = json.dumps(report).encode()
        self.store.put(self.index[name]["key"], data, "application/json")
        self.index[name].update(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)
        )
        self.save_index(self.index)

    def save_index(self, index):
        with Session(self.engine) as session:
            session.get(Model, self.model_id).metrics = {
                "artifacts": copy.deepcopy(index)
            }
            session.commit()

    def test_classification_report_is_typed_and_redacts_private_fields(self):
        self.publish()
        response = self.get("/evaluation")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(
            data["metrics"]["confusion_matrix"]["counts"], [[16, 0], [0, 16]]
        )
        self.assertIsNone(data["metrics"]["roc_auc"])
        self.assertEqual(data["checkpoint_epoch"], 3)
        self.assertEqual(data["latency_scope"], "worker_cpu_model_only")
        self.assertNotIn("private-report-sentinel", response.text)
        self.assertNotIn("runtime", data)
        self.assertNotIn("image_keys", data["parity"])
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_detection_report_preserves_undefined_ap_and_thresholds(self):
        self.publish(detection=True)
        response = self.get("/evaluation")
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertIsNone(data["metrics"]["map50"])
        self.assertIsNone(data["metrics"]["per_class"][0]["ap50"])
        self.assertEqual(data["metric_settings"]["precision_recall_confidence"], 0.25)
        self.assertEqual(data["parity"]["box_absolute_tolerance"], 0.002)

    def test_signed_downloads_match_registered_bytes_and_remain_private(self):
        self.publish()
        listing = self.get("/artifacts")
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertEqual(
            [r["name"] for r in listing.json()], ["weights", "onnx", "metadata"]
        )
        self.assertNotIn('"key"', listing.text)
        self.assertNotIn("download_url", listing.text)
        for name, source in (
            ("weights", "run/best.pt"),
            ("onnx", "onnx/model.onnx"),
            ("metadata", "onnx/metadata.json"),
        ):
            response = self.get("/artifacts/" + name)
            self.assertEqual(response.status_code, 200, response.text)
            value = response.json()
            self.assertEqual(value["expires_in"], 300)
            with httpx.Client(trust_env=False) as client:
                download = client.get(value["download_url"])
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.content, self.files[source])
                self.assertEqual(
                    hashlib.sha256(download.content).hexdigest(), value["sha256"]
                )
                self.assertIn("attachment;", download.headers["content-disposition"])
                self.assertEqual(download.headers["cache-control"], "private, no-store")
                url = urlsplit(value["download_url"])
                unsigned = urlunsplit((url.scheme, url.netloc, url.path, "", ""))
                self.assertEqual(client.get(unsigned).status_code, 403)
                tampered = value["download_url"].replace(
                    "X-Amz-Expires=300", "X-Amz-Expires=600"
                )
                self.assertEqual(client.get(tampered).status_code, 403)
        self.assertEqual(self.get("/artifacts/raw-log").status_code, 404)

    def test_owner_checks_precede_storage_reads_or_signing(self):
        self.publish()
        with (
            patch.object(self.store, "read") as read,
            patch.object(self.store.client, "head_object") as head,
            patch.object(self.store, "download_url") as sign,
        ):
            for suffix in ("/evaluation", "/artifacts", "/artifacts/weights"):
                self.assertEqual(self.get(suffix, {}).status_code, 401)
                self.assertEqual(
                    self.get(suffix, self.auth(self.user_id, "refresh")).status_code,
                    401,
                )
                self.assertEqual(
                    self.get(suffix, self.auth(self.other_id)).status_code, 404
                )
                missing = self.client.get(
                    f"/api/v1/models/{uuid4()}" + suffix, headers=self.headers
                )
                self.assertEqual(missing.status_code, 404)
            read.assert_not_called()
            head.assert_not_called()
            sign.assert_not_called()

    def test_cross_job_and_malformed_artifact_indexes_cannot_be_signed(self):
        self.publish()
        for field, value in (
            ("key", f"{uuid4()}/jobs/{uuid4()}/{uuid4()}/run/best.pt"),
            (
                "key",
                self.index["run/best.pt"]["key"].replace(
                    "/run/best.pt", "/../run/best.pt"
                ),
            ),
            ("sha256", "bad"),
            ("bytes", True),
            ("bytes", 0),
        ):
            index = copy.deepcopy(self.index)
            index["run/best.pt"][field] = value
            self.save_index(index)
            with patch.object(self.store, "download_url") as sign:
                self.assertEqual(self.get("/artifacts/weights").status_code, 409)
                sign.assert_not_called()

    def test_missing_and_corrupt_objects_fail_safely(self):
        self.publish()
        self.store.put(
            self.index["evaluation.json"]["key"], b"broken", "application/json"
        )
        response = self.get("/evaluation")
        self.assertEqual(response.status_code, 409)
        self.assertNotIn(self.index["evaluation.json"]["key"], response.text)
        self.store.client.delete_object(
            Bucket=self.store.bucket, Key=self.index["run/best.pt"]["key"]
        )
        self.assertEqual(self.get("/artifacts/weights").status_code, 409)
        self.store.client.delete_object(
            Bucket=self.store.bucket, Key=self.index["evaluation.json"]["key"]
        )
        self.assertEqual(self.get("/evaluation").status_code, 409)

    def test_wrong_provenance_schema_and_gate_are_not_presented_as_valid_reports(self):
        self.publish()
        for field, value in (
            ("task_type", "detection"),
            ("split", "val"),
            ("checkpoint_sha256", "0" * 64),
            ("checkpoint_epoch", "private-report-sentinel"),
            ("smoke_run", True),
        ):
            report = copy.deepcopy(self.evaluation)
            report[field] = value
            self.replace_report("evaluation.json", report)
            response = self.get("/evaluation")
            self.assertEqual(response.status_code, 409)
            self.assertNotIn("private-report-sentinel", response.text)
        self.replace_report("evaluation.json", self.evaluation)
        report = copy.deepcopy(self.verification)
        report["parity"]["max_absolute_error"] = 0.1
        self.replace_report("onnx/verification.json", report)
        self.assertEqual(self.get("/evaluation").status_code, 409)

    def test_storage_outage_returns_generic_service_error(self):
        self.publish()
        error = EndpointConnectionError(
            endpoint_url="https://private-report-sentinel.invalid"
        )
        with patch.object(self.store, "read", side_effect=error):
            response = self.get("/evaluation")
            self.assertEqual(response.status_code, 503)
            self.assertNotIn("private-report-sentinel", response.text)
        with patch.object(self.store.client, "head_object", side_effect=error):
            self.assertEqual(self.get("/artifacts/onnx").status_code, 503)

    def test_report_checksum_and_size_bounds_precede_publication(self):
        self.publish()
        original = self.files["evaluation.json"]
        self.store.put(
            self.index["evaluation.json"]["key"],
            b" " + original[1:],
            "application/json",
        )
        self.assertEqual(self.get("/evaluation").status_code, 409)
        index = copy.deepcopy(self.index)
        index["evaluation.json"]["bytes"] = 32 * 1024 * 1024 + 1
        self.save_index(index)
        with patch.object(self.store, "read") as read:
            self.assertEqual(self.get("/evaluation").status_code, 409)
            read.assert_not_called()
        self.replace_report("evaluation.json", ["invalid report shape"])
        self.assertEqual(self.get("/evaluation").status_code, 409)

    def test_download_rejects_changed_object_size_before_signing(self):
        self.publish()
        self.store.put(
            self.index["onnx/model.onnx"]["key"], b"changed", "application/octet-stream"
        )
        with patch.object(self.store, "download_url") as sign:
            self.assertEqual(self.get("/artifacts/onnx").status_code, 409)
            sign.assert_not_called()
