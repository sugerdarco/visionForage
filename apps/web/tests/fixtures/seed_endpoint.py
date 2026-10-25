"""Seed a deployable RGB-mean graph for an explicitly owned browser-test project.

This is an integration fixture, never a trained model or quality measurement.
"""

import argparse
import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

from bundle_fixture import make_bundle
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_model_artifacts import ModelArtifactTest

from vf_api.database import make_engine
from vf_api.models import Annotation, Class, Image, Model, Project, TrainingJob, User
from vf_api.settings import Settings, StorageSettings
from vf_api.storage import ArtifactStore, ObjectStore

parser = argparse.ArgumentParser()
parser.add_argument("project_id", type=UUID)
parser.add_argument("email")
args = parser.parse_args()
if not args.email.startswith("e2e-endpoint-") or not args.email.endswith(
    "@example.com"
):
    parser.error("Only a generated endpoint-test account is allowed")
engine = make_engine(Settings())
store = ArtifactStore(StorageSettings())
images = ObjectStore(StorageSettings())
try:
    with Session(engine) as session, TemporaryDirectory() as directory:
        project = session.scalar(
            select(Project)
            .join(User)
            .where(Project.id == args.project_id, User.email == args.email)
        )
        if project is None or project.task_type != "classification" or project.deleting:
            parser.error("The test account must own an active classification project")
        classes = session.scalars(
            select(Class).where(Class.project_id == project.id)
        ).all()
        for cls in classes:
            for shade in range(3):
                data = io.BytesIO()
                color = (
                    (200 + shade, 0, 0) if cls.name == "red" else (0, 200 + shade, 0)
                )
                PILImage.new("RGB", (32, 32), color).save(data, format="PNG")
                raw = data.getvalue()
                sha = hashlib.sha256(raw).hexdigest()
                key = f"{project.id}/originals/{sha}.png"
                images.put(key, raw, "image/png")
                image = Image(
                    project_id=project.id,
                    storage_key=key,
                    sha256=sha,
                    width=32,
                    height=32,
                    status="labeled",
                )
                session.add(image)
                session.flush()
                session.add(
                    Annotation(image_id=image.id, class_id=cls.id, source="human")
                )
        version = ModelArtifactTest().add_version(session, project.id)
        job = TrainingJob(
            project_id=project.id,
            dataset_version_id=version.id,
            model_arch="resnet18",
            hyperparams={},
            status="succeeded",
        )
        session.add(job)
        session.flush()
        root = Path(directory)
        make_bundle(root)
        prefix = f"{project.id}/jobs/{job.id}/{uuid4()}"
        index = {
            f"onnx/{name}": store.put_file_once(f"{prefix}/onnx/{name}", root / name)
            for name in ("model.onnx", "metadata.json")
        }
        model = Model(
            project_id=project.id,
            job_id=job.id,
            version=1,
            input_size=32,
            class_map={"red": 0, "green": 1},
            metrics={"artifacts": index},
            weights_key=f"{prefix}/run/best.pt",
            onnx_key=index["onnx/model.onnx"]["key"],
        )
        session.add(model)
        session.commit()
        print(model.id)
finally:
    store.close()
    images.close()
    engine.dispose()
