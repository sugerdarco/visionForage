PYTHON ?= python3
UV ?= uv
RUFF ?= $(UV) tool run --from ruff==0.14.1 ruff
COMPOSE = docker compose --env-file .env -f infra/compose.yml
.DEFAULT_GOAL := lint

.PHONY: lint test env up down status check-infra

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
	$(RUFF) check scripts tests packages
	@for script in scripts/*.sh; do bash -n "$$script" || exit; done

test:
	PYTHONPATH=packages $(PYTHON) -m unittest discover -s tests -v
