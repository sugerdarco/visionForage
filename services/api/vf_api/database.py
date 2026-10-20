"""Explicit database setup; importing this module opens no connections."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from vf_api.settings import Settings


def make_engine(settings: Settings) -> Engine:
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={"connect_timeout": 10},
    )


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
