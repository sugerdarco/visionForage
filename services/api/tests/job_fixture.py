"""Frozen metadata fixtures; job submission must not need object storage or Redis."""

from uuid import uuid4

from api_fixture import APITest
from sqlalchemy.orm import Session

from vf_api.models import DatasetVersion, Project, User
from vf_api.security import issue_token


class JobAPITest(APITest):
    def setUp(self):
        with Session(self.engine) as session:
            owner = User(email=f"{uuid4().hex}@example.com", password_hash="unused")
            other = User(email=f"{uuid4().hex}@example.com", password_hash="unused")
            session.add_all([owner, other])
            session.flush()
            self.user_id, self.other_id = owner.id, other.id
            self.project = Project(
                owner_id=owner.id, name="Training", task_type="classification"
            )
            self.other_project = Project(
                owner_id=other.id, name="Other", task_type="detection"
            )
            session.add_all([self.project, self.other_project])
            session.flush()
            self.project_id, self.other_project_id = (
                self.project.id,
                self.other_project.id,
            )
            self.version = self.add_version(session, self.project.id)
            self.other_version = self.add_version(session, self.other_project.id)
            self.version_id, self.other_version_id = (
                self.version.id,
                self.other_version.id,
            )
            self.counts = dict(self.version.counts)
            session.commit()
        self.headers = self.auth(self.user_id)
        self.path = f"/api/v1/projects/{self.project_id}/jobs"

    def add_version(self, session, project_id, *, number=1, counts=None):
        row = DatasetVersion(
            project_id=project_id,
            version=number,
            manifest_key=f"{project_id}/v{number}/{uuid4()}/manifest.json",
            split_ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            counts=counts
            if counts is not None
            else {
                "images": {"train": 100, "val": 32, "test": 32},
                "manifest_sha256": "a" * 64,
                "audit_sha256": "b" * 64,
                "audit_key": f"{project_id}/v{number}/audit.json",
            },
        )
        session.add(row)
        session.flush()
        return row

    def auth(self, identifier, kind="access"):
        token, _ = issue_token(identifier, kind, self.settings)
        return {"Authorization": f"Bearer {token}"}

    def create(self, **overrides):
        return self.client.post(
            self.path,
            headers=self.headers,
            json={
                "dataset_version_id": str(self.version_id),
                "model_arch": "resnet18",
                **overrides,
            },
        )

    def read(self, identifier):
        return self.client.get(f"/api/v1/jobs/{identifier}", headers=self.headers)

    def cancel(self, identifier):
        return self.client.post(
            f"/api/v1/jobs/{identifier}/cancel", headers=self.headers
        )
