"""Each integration test class owns and removes a randomly named database."""

from pathlib import Path
import unittest
from uuid import uuid4

from alembic import command
from alembic.config import Config
import sqlalchemy as sa

from vf_api.settings import Settings


class DisposableDatabaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        url = Settings().database_url
        cls.name = "vf_test_" + uuid4().hex
        cls.admin = sa.create_engine(
            url.set(database="postgres"),
            isolation_level="AUTOCOMMIT",
            hide_parameters=True,
            connect_args={"connect_timeout": 10},
        )
        cls.addClassCleanup(cls.admin.dispose)
        with cls.admin.connect() as connection:
            connection.execute(sa.text(f'CREATE DATABASE "{cls.name}"'))
        cls.addClassCleanup(cls.drop_database)
        cls.engine = sa.create_engine(url.set(database=cls.name), hide_parameters=True)
        cls.addClassCleanup(cls.engine.dispose)
        cls.config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
        cls.migrate("upgrade", "head")

    @classmethod
    def drop_database(cls):
        with cls.admin.connect() as connection:
            connection.execute(sa.text(f'DROP DATABASE "{cls.name}" WITH (FORCE)'))

    @classmethod
    def migrate(cls, action, revision):
        with cls.engine.begin() as connection:
            cls.config.attributes["connection"] = connection
            try:
                getattr(command, action)(cls.config, revision)
            finally:
                cls.config.attributes.pop("connection")
