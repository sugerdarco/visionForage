"""Publish synthetic report/download fixtures into an explicitly owned E2E project.

No training runs. Reuses the API integration fixture; its bytes are not ML models.
The browser test deletes the project through the API when it finishes.
"""

import argparse
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from test_model_artifacts import ModelArtifactTest
from vf_api.database import make_engine
from vf_api.models import Project, User
from vf_api.settings import Settings, StorageSettings
from vf_api.storage import ArtifactStore

parser = argparse.ArgumentParser()
parser.add_argument("project_id", type=UUID)
parser.add_argument("email")
parser.add_argument("task", choices=("classification", "detection"))
parser.add_argument("--version-count", type=int, choices=(1, 2), default=1)
args = parser.parse_args()
if not args.email.startswith("e2e-model-") or not args.email.endswith("@example.com"):
    parser.error("Only a generated model-test account is allowed")
fixture = ModelArtifactTest()
fixture.engine = make_engine(Settings())
fixture.store = ArtifactStore(StorageSettings())
try:
    with Session(fixture.engine) as session:
        project = session.scalar(
            select(Project)
            .join(User)
            .where(Project.id == args.project_id, User.email == args.email)
        )
        if project is None or project.task_type != args.task or project.deleting:
            parser.error(
                "The test account must own an active project of the requested task"
            )
        fixture.project_id = project.id
        for number in range(1, args.version_count + 1):
            fixture.version_id = fixture.add_version(
                session, project.id, number=number
            ).id
        session.commit()
    fixture.publish(detection=args.task == "detection")
    print(fixture.model_id)
finally:
    fixture.store.close()
    fixture.engine.dispose()
