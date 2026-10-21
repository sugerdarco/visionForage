"""Real PostgreSQL checks in a fresh disposable database, never the app database."""

from uuid import UUID, uuid4

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from database_fixture import DisposableDatabaseTest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from vf_api.models import (
    Annotation,
    Base,
    Class,
    DatasetVersion,
    Deployment,
    Image,
    Model,
    Prediction,
    Project,
    TrainingJob,
    TrainingMetric,
    User,
)


class DatabaseTest(DisposableDatabaseTest):
    def setUp(self):
        self.session = Session(self.engine)
        self.addCleanup(self.session.close)

    def row(self, model, **values):
        row = model(**values)
        self.session.add(row)
        self.session.flush()
        return row

    def fixture(self):
        user = self.row(User, email="owner@example.test", password_hash="test-hash")
        project = self.row(
            Project, owner_id=user.id, name="Steel", task_type="detection"
        )
        label = self.row(
            Class, project_id=project.id, name="scratch", index=0, color="#00aaff"
        )
        image = self.row(
            Image,
            project_id=project.id,
            storage_key="project/image.jpg",
            width=200,
            height=200,
            sha256="a" * 64,
        )
        version = self.row(
            DatasetVersion,
            project_id=project.id,
            version=1,
            manifest_key="project/v1/manifest.json",
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            counts={"train": 7, "val": 2, "test": 1},
        )
        job = self.row(
            TrainingJob,
            project_id=project.id,
            dataset_version_id=version.id,
            model_arch="yolov8s",
            hyperparams={"epochs": 100},
        )
        return user, project, label, image, version, job

    def rejects(self, model, **values):
        with self.assertRaises(IntegrityError), self.session.begin_nested():
            self.row(model, **values)

    def test_upgrade_downgrade_and_metadata_agreement(self):
        expected = {
            "users",
            "projects",
            "classes",
            "images",
            "annotations",
            "dataset_versions",
            "training_jobs",
            "training_metrics",
            "models",
            "deployments",
            "predictions",
            "alembic_version",
        }
        self.assertEqual(set(sa.inspect(self.engine).get_table_names()), expected)
        self.migrate("downgrade", "base")
        self.assertEqual(sa.inspect(self.engine).get_table_names(), ["alembic_version"])
        with self.engine.connect() as connection:
            functions = connection.execute(
                sa.text(
                    "SELECT proname FROM pg_proc JOIN pg_namespace n ON n.oid = pronamespace "
                    "WHERE n.nspname = 'public' AND proname LIKE 'vf_%'"
                )
            ).all()
            self.assertEqual(functions, [])
        self.migrate("upgrade", "head")
        self.migrate("upgrade", "head")
        with self.engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"compare_server_default": True}
            )
            self.assertEqual(compare_metadata(context, Base.metadata), [])
            self.assertEqual(
                connection.scalar(sa.text("SELECT version_num FROM alembic_version")),
                "0002",
            )

    def test_complete_record_chain_and_server_defaults(self):
        user, project, label, image, version, job = self.fixture()
        annotation = self.row(
            Annotation, image_id=image.id, class_id=label.id, bbox=[0.1, 0.2, 0.3, 0.4]
        )
        metric = self.row(
            TrainingMetric,
            job_id=job.id,
            epoch=1,
            split="val",
            name="map50",
            value=0.74,
        )
        model = self.row(
            Model,
            project_id=project.id,
            job_id=job.id,
            version=1,
            metrics={"map50": 0.74},
            weights_key="job/best.pt",
            onnx_key="job/model.onnx",
            input_size=640,
            class_map={"scratch": 0},
        )
        deployment = self.row(
            Deployment, model_id=model.id, slug="steel-v1", api_key_hash="test-hash"
        )
        prediction = self.row(
            Prediction,
            deployment_id=deployment.id,
            image_key="prediction.jpg",
            output={"boxes": []},
            max_confidence=0.1,
            latency_ms=180,
        )
        for row in (
            user,
            project,
            label,
            image,
            version,
            job,
            annotation,
            metric,
            model,
            deployment,
            prediction,
        ):
            self.assertIsInstance(row.id, UUID)
            self.assertIsNotNone(row.created_at.tzinfo)
        self.assertEqual(image.status, "unlabeled")
        self.assertEqual(job.status, "queued")
        self.assertEqual(annotation.source, "human")
        self.assertEqual(deployment.status, "starting")
        self.assertAlmostEqual(deployment.confidence_threshold, 0.6)
        self.assertFalse(prediction.sent_to_review)
        self.session.expire_all()
        self.assertEqual(annotation.bbox, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(version.counts["train"], 7)
        self.assertEqual(model.class_map, {"scratch": 0})

    def test_uniqueness_and_project_scoped_image_deduplication(self):
        user, project, label, image, version, job = self.fixture()
        self.rejects(User, email=user.email, password_hash="different-hash")
        self.rejects(
            Class, project_id=project.id, name="other", index=0, color="#ffffff"
        )
        self.rejects(
            Class, project_id=project.id, name=label.name, index=1, color="#ffffff"
        )
        self.rejects(
            Image,
            project_id=project.id,
            storage_key="different.jpg",
            width=200,
            height=200,
            sha256=image.sha256,
        )
        self.rejects(
            DatasetVersion,
            project_id=project.id,
            version=1,
            manifest_key="v1-copy.json",
            split_ratios={},
            counts={},
        )
        other = self.row(
            Project, owner_id=user.id, name="Other", task_type="classification"
        )
        self.row(
            Image,
            project_id=other.id,
            storage_key="other/image.jpg",
            width=200,
            height=200,
            sha256=image.sha256,
        )

    def test_foreign_keys_and_cross_project_job_model_links(self):
        user, project, label, image, version, job = self.fixture()
        self.rejects(
            Project, owner_id=uuid4(), name="Missing owner", task_type="detection"
        )
        other = self.row(Project, owner_id=user.id, name="Other", task_type="detection")
        self.rejects(
            TrainingJob,
            project_id=other.id,
            dataset_version_id=version.id,
            model_arch="yolov8s",
        )
        self.rejects(
            Model,
            project_id=other.id,
            job_id=job.id,
            version=1,
            metrics={},
            weights_key="best.pt",
            onnx_key="model.onnx",
            input_size=640,
            class_map={},
        )
        with self.assertRaises(IntegrityError), self.session.begin_nested():
            self.session.execute(sa.delete(Project).where(Project.id == project.id))

    def test_boxes_confidence_status_and_finite_metrics(self):
        _, project, label, image, _, job = self.fixture()
        for box in (
            [],
            [0.1, 0.2, 0.3],
            [-0.1, 0, 0.2, 0.2],
            [0, 0, 0, 1],
            [0.9, 0, 0.2, 1],
            [None, 0, 1, 1],
            [float("nan"), 0, 1, 1],
            [[0, 0], [1, 1]],
        ):
            with self.subTest(box=box):
                self.rejects(Annotation, image_id=image.id, class_id=label.id, bbox=box)
        self.row(Annotation, image_id=image.id, class_id=label.id, bbox=[0, 0, 1, 1])
        mixed = self.row(
            Annotation, image_id=image.id, class_id=label.id, bbox=[0, 0.1, 1, 0.9]
        )
        self.session.refresh(mixed)
        self.assertEqual(mixed.bbox, [0.0, 0.1, 1.0, 0.9])
        self.row(Annotation, image_id=image.id, class_id=label.id, bbox=None)
        self.rejects(Annotation, image_id=image.id, class_id=label.id, confidence=1.1)
        self.rejects(
            TrainingJob,
            project_id=project.id,
            dataset_version_id=job.dataset_version_id,
            model_arch="yolov8s",
            status="typo",
        )
        for value in (float("nan"), float("inf"), float("-inf")):
            self.rejects(
                TrainingMetric,
                job_id=job.id,
                epoch=1,
                split="val",
                name="loss",
                value=value,
            )
        self.row(
            TrainingMetric, job_id=job.id, epoch=1, split="val", name="loss", value=0.1
        )
        self.rejects(
            TrainingMetric, job_id=job.id, epoch=1, split="val", name="loss", value=0.2
        )

    def test_one_model_per_job_and_one_promoted_model_per_project(self):
        _, project, _, _, version, job = self.fixture()
        values = dict(
            project_id=project.id,
            metrics={},
            weights_key="best.pt",
            onnx_key="model.onnx",
            input_size=640,
            class_map={},
        )
        first = self.row(Model, job_id=job.id, version=1, is_promoted=True, **values)
        self.rejects(Model, job_id=job.id, version=2, **values)
        another_job = self.row(
            TrainingJob,
            project_id=project.id,
            dataset_version_id=version.id,
            model_arch="yolov8s",
        )
        self.rejects(
            Model, job_id=another_job.id, version=2, is_promoted=True, **values
        )
        second = self.row(Model, job_id=another_job.id, version=2, **values)
        first.is_promoted = False
        self.session.flush()
        second.is_promoted = True
        self.session.flush()

    def test_raw_sql_timestamp_updates_and_frozen_version(self):
        _, project, _, _, version, _ = self.fixture()
        original = project.updated_at
        self.session.execute(
            sa.text("UPDATE projects SET name = 'Renamed' WHERE id = :id"),
            {"id": project.id},
        )
        self.session.refresh(project)
        self.assertGreater(project.updated_at, original)
        with self.assertRaises(IntegrityError), self.session.begin_nested():
            self.session.execute(
                sa.text("UPDATE dataset_versions SET counts = '{}' WHERE id = :id"),
                {"id": version.id},
            )
