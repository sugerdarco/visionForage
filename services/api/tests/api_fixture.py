"""Run the HTTP application against the test class's disposable database."""

import secrets

from database_fixture import DisposableDatabaseTest
from fastapi.testclient import TestClient

from vf_api.main import create_app
from vf_api.settings import APISettings, Settings


class APITest(DisposableDatabaseTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.settings = APISettings(
            **Settings().model_dump(),
            jwt_secret=secrets.token_hex(32),
            access_token_minutes=15,
            refresh_token_days=7,
        ).model_copy(update={"postgres_db": cls.name})
        cls.app = create_app(cls.settings)
        cls.client = TestClient(cls.app)
        cls.client.__enter__()
        cls.addClassCleanup(cls.client.__exit__, None, None, None)
