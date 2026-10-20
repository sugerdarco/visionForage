# Accounts and authentication

FastAPI now provides the first HTTP layer over the PostgreSQL schema. Registration
stores a bcrypt password hash; login verifies that hash and signs access and
refresh tokens. Protected requests verify the access token and load its user
from PostgreSQL. The resulting user ID scopes [projects and their classes](PROJECTS.md).

This step runs Uvicorn in the local API environment and PostgreSQL in Docker.
It needs no GPU, sudo, or remote connection. Redis and MinIO support uploads and
thumbnail jobs; authentication uses only PostgreSQL.

## Run locally

From the repository root, after [local service setup](LOCAL.md):

```bash
make api-env UV="$HOME/.local/share/visionforge/tools/bin/uv"
python3 scripts/setup_local_env.py --add-api-secret
make migrate
make test-db
make api
```

For a new checkout, `make env` generates all passwords and `JWT_SECRET`.
For an older `.env`, `--add-api-secret` appends a missing random signing key and
preserves existing credentials. It never replaces an existing key, even an empty
one. The API requires at least 64 characters; the helper generates 32 random
bytes encoded as 64 hexadecimal characters. Keep the key stable across restarts.
Environment variables override `.env`; migrations still need only database settings.

The server binds to `127.0.0.1:8000`. Open
[interactive API documentation](http://127.0.0.1:8000/docs) or download
[`/openapi.json`](http://127.0.0.1:8000/openapi.json). Stop the foreground server
with Ctrl+C. The app does not apply migrations or create accounts at startup.
Compose packaging for the API is still a separate task.

## HTTP contract

All account endpoints use JSON under `/api/v1`.

| Method and path | Request | Success response |
| --- | --- | --- |
| `POST /auth/register` | `email`, `password` | 201: user `id`, normalized `email`, `created_at` |
| `POST /auth/login` | `email`, `password` | 200: `access_token`, `refresh_token`, `token_type`, `expires_in`, `refresh_expires_in` |
| `POST /auth/refresh` | `refresh_token` | 200: new `access_token`, `token_type`, `expires_in` |
| `GET /auth/me` | `Authorization: Bearer <access_token>` | 200: current user's public fields |
| `GET /healthz` | None; outside `/api/v1` | 200: process is responding |
| `GET /readyz` | None; outside `/api/v1` | 200: PostgreSQL and the user table are accessible; otherwise 503 |

Register first, then log in. In `/docs`, paste the returned **access** token into
the Authorize dialog before calling `/auth/me`. Send refresh credentials only in
the refresh request body. Lifetimes are seconds in responses: access defaults to
900 seconds and refresh to 604800 seconds. `ACCESS_TOKEN_MINUTES` permits 1–60
(default 15); `REFRESH_TOKEN_DAYS` permits 1–30 (default 7).

Email validation and lowercasing apply on both registration and login, including
the local part before `@`. A concurrent duplicate registration is resolved by the
database uniqueness constraint and returns 409. Direct SQL inserts must follow
the same normalization policy; the existing database constraint is case-sensitive.
Passwords require 12 or more characters and at most 72 UTF-8 bytes. Unicode is
allowed within that byte limit; spaces are preserved. There is no truncation.

Malformed payloads return 422 without rejected values or validation context.
Invalid credentials, wrong token types, expired tokens and tokens for deleted
users return 401. A wrong password and an unknown account share the same response;
unknown accounts still run bcrypt verification. Account responses use `no-store`
cache headers, and response schemas never expose password hashes.

## Choices and limits

- **FastAPI + Pydantic:** typed request/response schemas produce OpenAPI for the
  later generated web client. Database and bcrypt operations use synchronous
  handlers so blocking work runs in FastAPI's thread pool.
- **bcrypt, cost 12:** follows the build plan. Argon2 would be another reasonable
  choice, but changing algorithms is unnecessary for this increment. Bcrypt's
  byte limit is checked explicitly before hashing; see the
  [bcrypt project documentation](https://pypi.org/project/bcrypt/).
- **PyJWT with HS256:** a single API service can sign and verify with one private
  key. Verification fixes the algorithm and requires issuer, audience, expiry,
  issue time, not-before time, subject, token ID and token type. Access and refresh
  tokens cannot substitute for one another. The key must stay inside the API;
  separate verifying services may justify asymmetric keys later. See
  [PyJWT verification options](https://pyjwt.readthedocs.io/en/stable/api.html).
- **Stateless refresh:** no session table is added to the plan's 11 tables.
  Refresh issues only an access token and never extends the original refresh
  expiry. The refresh token is reusable until expiry: there is no rotation,
  replay detection or individual session revocation. Deleting a user rejects
  both token kinds; rotating the signing key invalidates every user's tokens.
- **A separate API environment:** no PyTorch, timm or Ultralytics imports. Tests
  use FastAPI's TestClient and the HTTPX2 dependency supported by the pinned
  [Starlette client](https://github.com/Kludex/starlette/blob/1.9.0/starlette/testclient.py).

`/auth/me` and the two health endpoints are small additions to the plan's route
list. They make identity and service checks available before project routes.
This is a local development auth foundation. Public deployment still needs
HTTPS, login throttling, a browser token-storage decision and account recovery;
there are no logout, email verification or password-reset endpoints yet.

## Verification

`make test-db` runs database, configuration and HTTP tests in uniquely named
disposable PostgreSQL databases, then removes them. It never creates test accounts
in the application database. Checks cover the register/login/refresh/me flow,
bcrypt storage, case normalization, simultaneous registration, Unicode and byte
limits, credential redaction, malformed/expired/forged tokens, deleted users,
OpenAPI, readiness and database outages.

Verified on 2026-10-20: 21 API/database/configuration tests and 110 existing
ML/setup tests pass, together with lint. A real Uvicorn process also returned
200 for `/healthz`, `/readyz`, `/docs` and `/openapi.json` against local PostgreSQL.

Project and class endpoints now use the current-user dependency; see
[PROJECTS.md](PROJECTS.md) for their request formats and ownership rules.
Image upload, gallery reads and thumbnail processing are now available; see
[UPLOADS.md](UPLOADS.md). [Annotation reads and saves](ANNOTATIONS.md) now support
classification labels, detection boxes and reviewed negative images.
[Dataset imports](IMPORTS.md) add signed ZIP uploads for ImageFolder, YOLO and COCO.
