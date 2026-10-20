"""Use the application's configuration without writing secrets to alembic.ini."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy.engine import Connection

from vf_api.database import make_engine
from vf_api.models import Base
from vf_api.settings import Settings

config = context.config
if config.config_file_name:
    fileConfig(config.config_file_name, disable_existing_loggers=False)


def run_migrations(connection: Connection | None = None) -> None:
    options = {"target_metadata": Base.metadata, "compare_server_default": True}
    if connection is None:
        context.configure(url=Settings().database_url, literal_binds=True, **options)
    else:
        context.configure(connection=connection, **options)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations()
elif config.attributes.get("connection") is not None:
    # Tests supply a connection to their disposable database.
    run_migrations(config.attributes["connection"])
else:
    engine = make_engine(Settings())
    try:
        with engine.connect() as connection:
            run_migrations(connection)
    finally:
        engine.dispose()
