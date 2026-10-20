PYTHON ?= python3
UV ?= uv
RUFF ?= $(UV) tool run --from ruff==0.14.1 ruff
COMPOSE = docker compose --env-file .env -f infra/compose.yml
API_PYTHON ?= .venv/bin/python
ALEMBIC = $(API_PYTHON) -m alembic -c services/api/alembic.ini
.DEFAULT_GOAL := lint

.PHONY: lint test env up down status check-infra api-env api migrate db-current test-db storage-config worker beat test-uploads

api-env:
	@test -x $(API_PYTHON) || $(UV) venv --python 3.11.16 .venv
	$(UV) pip sync --python $(API_PYTHON) requirements/api.txt

api:
	PYTHONPATH=services/api $(API_PYTHON) -m uvicorn vf_api.main:create_app --factory --host 127.0.0.1 --port 8000

storage-config:
	PYTHONPATH=services/api $(API_PYTHON) -m vf_api.storage

worker:
	PYTHONPATH=services/api:services/worker $(API_PYTHON) -m celery -A vf_worker.app:app worker --loglevel=INFO --queues=cpu --concurrency=2

beat:
	@mkdir -p artifacts/celery
	PYTHONPATH=services/api:services/worker $(API_PYTHON) -m celery -A vf_worker.app:app beat --loglevel=INFO --schedule artifacts/celery/schedule

test-uploads:
	PYTHONPATH=services/api:services/worker:services/api/tests $(API_PYTHON) -m unittest discover -s services/api/integration_tests -v

migrate:
	$(ALEMBIC) upgrade head

db-current:
	$(ALEMBIC) current

test-db:
	PYTHONPATH=services/api $(API_PYTHON) -m unittest discover -s services/api/tests -v

env:
	$(PYTHON) scripts/setup_local_env.py

up:
	$(COMPOSE) up -d --build --wait --wait-timeout 120 postgres redis minio
	$(COMPOSE) run --rm --no-deps storage-init

down:
	$(COMPOSE) down

status:
	$(COMPOSE) ps

check-infra:
	bash scripts/check_local_services.sh

lint:
	$(RUFF) check scripts tests packages services
	@for script in scripts/*.sh; do bash -n "$$script" || exit; done

test:
	PYTHONPATH=packages $(PYTHON) -m unittest discover -s tests -v
