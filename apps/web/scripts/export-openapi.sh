#!/usr/bin/env bash
# Export the API's OpenAPI document from the checked-in FastAPI source without
# starting its lifespan, reading the private .env, or contacting any service.
# API_PYTHON must point at an interpreter with requirements/api.txt installed.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
python="${API_PYTHON:-$repo/.venv/bin/python}"
cd "$repo/services/api"
"$python" -I - "$here/openapi/openapi.json" <<'PY'
import json
import sys

sys.path.insert(0, ".")
from vf_api.main import create_app
from vf_api.settings import APISettings

# Placeholder values satisfy settings validation; nothing connects or signs tokens.
settings = APISettings(_env_file=None, postgres_password="unused", jwt_secret="0" * 64)
with open(sys.argv[1], "w", encoding="utf-8") as output:
    json.dump(create_app(settings).openapi(), output, indent=2, sort_keys=True)
    output.write("\n")
PY
echo "Wrote openapi/openapi.json"
