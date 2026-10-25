"""Seed detection predictions for the real review API, without running inference."""

import argparse
import hashlib
import io
from uuid import UUID, uuid4

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_model_artifacts import ModelArtifactTest

from vf_api.database import make_engine
from vf_api.models import Deployment, Prediction, Project, User
from vf_api.settings import Settings, StorageSettings
from vf_api.storage import ArtifactStore, PredictionStore

parser = argparse.ArgumentParser()
parser.add_argument("project_id", type=UUID)
parser.add_argument("email")
args = parser.parse_args()
if not args.email.startswith("e2e-review-") or not args.email.endswith("@example.com"):
    parser.error("Only a generated review-test account is allowed")
fixture = ModelArtifactTest()
fixture.engine = make_engine(Settings())
fixture.store = ArtifactStore(StorageSettings())
store = PredictionStore(StorageSettings())
try:
    with Session(fixture.engine) as session:
        project = session.scalar(
            select(Project)
            .join(User)
            .where(Project.id == args.project_id, User.email == args.email)
        )
        if project is None or project.task_type != "detection" or project.deleting:
            parser.error("The test account must own an active detection project")
        fixture.project_id = project.id
        fixture.version_id = fixture.add_version(session, project.id).id
        session.commit()
    fixture.publish(detection=True)
    with Session(fixture.engine) as session:
        endpoint = Deployment(
            model_id=fixture.model_id,
            slug="review-" + uuid4().hex,
            status="stopped",
            api_key_hash="a" * 64,
            confidence_threshold=0.6,
        )
        session.add(endpoint)
        session.flush()
        for index in range(2):
            raw = io.BytesIO()
            Image.new("RGB", (200, 100), (100, 120, 130 + index)).save(
                raw, format="PNG"
            )
            data = raw.getvalue()
            identifier = uuid4()
            key = f"{endpoint.id}/{identifier}.png"
            store.put_once(key, data, "image/png", hashlib.sha256(data).hexdigest())
            session.add(
                Prediction(
                    id=identifier,
                    deployment_id=endpoint.id,
                    image_key=key,
                    max_confidence=0.2 + index * 0.1,
                    latency_ms=10,
                    sent_to_review=True,
                    output={
                        "boxes": [
                            {
                                "label": "defect",
                                "confidence": 0.2 + index * 0.1,
                                "x": 0.1,
                                "y": 0.2,
                                "w": 0.3,
                                "h": 0.4,
                            }
                        ]
                    },
                )
            )
        session.commit()
finally:
    store.close()
    fixture.store.close()
    fixture.engine.dispose()
