"""Small registry records for logging tests, without running model training."""

from job_fixture import JobAPITest
from sqlalchemy.orm import Session

from vf_api.models import Deployment, Model, TrainingJob


class PredictionAPITest(JobAPITest):
    def setUp(self):
        super().setUp()
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
            model = Model(
                project_id=self.project_id,
                job_id=job.id,
                version=1,
                input_size=32,
                class_map={"red": 0, "green": 1},
                metrics={},
                weights_key="unused",
                onnx_key="unused",
            )
            session.add(model)
            session.flush()
            deployment = Deployment(
                model_id=model.id,
                slug=f"test-{model.id.hex}",
                api_key_hash="unused",
                status="running",
            )
            session.add(deployment)
            session.flush()
            self.model_id, self.deployment_id = model.id, deployment.id
            session.commit()
        self.base = f"/api/v1/deployments/{self.deployment_id}"
