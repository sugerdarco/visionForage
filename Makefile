PYTHON ?= python3
UV ?= uv
RUFF ?= $(UV) tool run --from ruff==0.14.1 ruff

.PHONY: lint test

lint:
	$(RUFF) check scripts tests
	@for script in scripts/*.sh; do bash -n "$$script" || exit; done

test:
	$(PYTHON) -m unittest discover -s tests -v
