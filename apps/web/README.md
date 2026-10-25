# VisionForge web app

The browser workspace for VisionForge. Users register, create a classification or detection
project, upload or import images, label them, freeze dataset versions, submit training runs,
watch them live, compare recorded runs, inspect evaluation reports, download artifacts and
select a preferred model. Every screen talks to the existing FastAPI service;
this app adds no backend of its own.

## Stack and versions

| Choice                          | Version                                    | Why                                                                |
| ------------------------------- | ------------------------------------------ | ------------------------------------------------------------------ |
| Node.js                         | 22.12 or newer (verified on 22.23.1)       | Required by Next 16, Vitest 5 and jsdom 30                         |
| Next.js (App Router, Turbopack) | 16.4.3                                     | Patched release selected after the integration dependency audit    |
| React / React DOM               | 19.3.1                                     | Required by react-konva 19.3                                       |
| TypeScript                      | 5.9.3 (strict, `noUncheckedIndexedAccess`) | openapi-typescript 7 supports TypeScript 5 only                    |
| Tailwind CSS                    | 4.3.3                                      | Styling through a small local component set in `src/components/ui` |
| TanStack Query                  | 5.104.5                                    | Server-state cache; cleared on every sign-in and sign-out          |
| Konva / react-konva             | 10.9.0 / 19.3.2                            | Detection box editor, loaded client-side only                      |
| Recharts                        | 3.10.1                                     | Epoch curves, each with a data-table alternative                   |
| openapi-typescript              | 7.13.0                                     | Generates request/response types from the API schema               |
| Vitest + jsdom / Playwright     | 5.0.5 + 30.1.3 / 1.64.1                    | Unit and browser tests                                             |

All versions are pinned exactly in `package.json`, and `package-lock.json` is the single lockfile.
Next 16 compatibility notes are in `node_modules/next/dist/docs/01-app/02-guides/upgrading/version-16.md`.
`agentRules: false` in `next.config.ts` stops Next from generating agent instruction files.

## Run locally

For the complete application, run `make up` from the repository root and open
`http://127.0.0.1:3000`. This builds the production web image and starts the API,
CPU worker, scheduler, storage and gateway. Do not also start native copies of
these services. The remote GPU worker is needed only to execute training runs.
See [backend setup](../../docs/BACKEND.md) for first-time storage prerequisites.

For frontend development, stop only the Compose web service before starting the
development server on the same port:

```bash
# From the repository root:
docker compose --env-file .env -f infra/compose.yml stop web
```

```bash
cd apps/web
cp .env.example .env.local   # NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
npm ci
npm run dev                  # http://127.0.0.1:3000
```

Use `http://127.0.0.1:3000` consistently. The API and MinIO CORS lists allow both
`http://127.0.0.1:3000` and `http://localhost:3000`. `NEXT_PUBLIC_API_BASE_URL` holds only the
public API origin; `src/lib/config.ts` appends `/api/v1` in one place. Never put database,
Redis, MinIO or JWT secrets in `NEXT_PUBLIC_*` variables, and do not load the root `.env` here.

| Command                                       | Purpose                                                 |
| --------------------------------------------- | ------------------------------------------------------- |
| `npm run dev` / `npm run build` / `npm start` | Develop, build and serve the production bundle          |
| `npm run typecheck`                           | `next typegen` then `tsc --noEmit`                      |
| `npm run lint` / `npm run format:check`       | ESLint (Next core-web-vitals + TypeScript) and Prettier |
| `npm test`                                    | Unit tests (Vitest)                                     |
| `npm run test:e2e`                            | Browser tests against the running API (see below)       |
| `npm run openapi:export`                      | Export `openapi/openapi.json` from the API source       |
| `npm run openapi:types`                       | Regenerate `src/lib/api/schema.d.ts` from that file     |

### API types

`openapi/openapi.json` is exported from the checked-in FastAPI source by calling
`create_app().openapi()` with placeholder settings. It does not start the app lifespan, read the
private `.env`, or contact PostgreSQL, Redis or MinIO. It only needs the API Python environment:

```bash
API_PYTHON=../../.venv/bin/python npm run openapi:export   # default path shown
npm run openapi:types
```

Commit both files together. Only frontend-only view state and the untyped `counts` JSON of a
dataset version (`VersionCounts` in `src/lib/api/types.ts`) are handwritten.

## Layout

```text
src/app/                 routes (all screens are client components)
src/components/ui/       buttons, fields, dialogs, badges, states, pagination
src/features/auth/       in-memory session, login/register forms
src/features/projects/   dashboard, project frame/sidebar, settings
src/features/images/     upload pipeline, ZIP import, gallery
src/features/annotations/ classification + detection editors, save queue, label queue
src/features/datasets/   version list/create/detail
src/features/training/   run list, submit form, live detail, comparison
src/features/models/     registry, test evaluation, artifact downloads and promotion
src/features/deployments/ endpoint controls, prediction tester and monitoring
src/features/review/     classification/box corrections and human acceptance
src/lib/api/             generated schema, typed client, endpoints, signed-storage PUT
src/lib/events/          SSE parser and reconnecting job stream
src/lib/geometry/        image/normalized/screen coordinate transforms
tests/unit/              Vitest behaviour tests
tests/e2e/               Playwright browser tests
tests/fixtures/          deterministic PNG/ZIP fixture generator
```

## Screens and the API routes they use

All paths are under `/api/v1`.

| Screen             | Browser route                                                   | API                                                                                                                                                                  |
| ------------------ | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Register / sign in | `/register`, `/login`                                           | `POST /auth/register`, `POST /auth/login`, `POST /auth/refresh`, `GET /auth/me`                                                                                      |
| Projects           | `/projects`                                                     | `GET/POST /projects`                                                                                                                                                 |
| Settings           | `/projects/[id]/settings`                                       | `GET/PATCH/DELETE /projects/{id}`, `POST /projects/{id}/classes`                                                                                                     |
| Images             | `/projects/[id]/images`                                         | `POST …/uploads/presign`, signed storage PUT, `POST …/uploads/complete`, `GET /projects/{id}/images`, `GET /images/{id}`, `POST …/imports/presign`, `POST …/imports` |
| Labeling           | `/projects/[id]/label/[imageId]`                                | `GET /images/{id}`, `GET/PUT /images/{id}/annotations`, `GET /projects/{id}/images`                                                                                  |
| Dataset versions   | `/projects/[id]/versions[/versionId]`                           | `GET/POST /projects/{id}/dataset-versions`, `GET …/dataset-versions/{versionId}`                                                                                     |
| Training           | `/projects/[id]/training`, `/training/new`, `/training/[jobId]` | `GET/POST /projects/{id}/jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/cancel`, `GET /jobs/{id}/metrics`, `GET /jobs/{id}/events`                                        |
| Compare runs       | `/projects/[id]/compare`                                        | `GET /jobs/{id}`, `GET /jobs/{id}/metrics` (all pages)                                                                                                               |
| Models             | `/projects/[id]/models`                                         | `GET /projects/{id}/models`, `GET /models/{id}`, `GET /models/{id}/evaluation`, `GET /models/{id}/artifacts[/{artifact}]`, `POST /models/{id}/promote`               |
| Deployments        | `/projects/[id]/deployments`                                    | `POST /models/{id}/deployments`, `GET /projects/{id}/deployments`, `GET/DELETE /deployments/{id}`; keyed prediction through the gateway                              |
| Monitoring         | `/projects/[id]/monitoring`                                     | `GET /deployments/{id}/stats`, `GET /deployments/{id}/predictions`                                                                                                   |
| Review             | `/projects/[id]/review`                                         | `GET /projects/{id}/review-queue`, `POST /review/{id}/accept`                                                                                                        |

## Behaviour worth knowing

- **Sessions are in memory only.** Access and refresh tokens are never written to storage,
  cookies or URLs. Navigation inside the app keeps the session; a reload or new tab requires
  signing in again (the login form returns to the requested page). Concurrent 401s share one
  refresh request; a rejected refresh clears the session, cached data and open streams and
  returns to the login page. Sign-out is local; the API has no logout endpoint.
- **Bearer tokens go only to the API.** Signed storage PUTs use `src/lib/api/storage.ts`, which
  sends the raw `File` with exactly the returned headers and no `Authorization`.
- **Uploads** hash the exact bytes with SHA-256, presign in batches of at most 20, PUT, then
  complete each file separately (201 new, 200 duplicate). Each file reports its own state.
  Stopping an upload only stops browser work; registered images stay (there is no delete API).
- **Imports** are synchronous. The UI shows “Validating and importing” rather than fake progress.
  If the completion response is lost, “Check again” repeats completion with the same grant, which
  the API documents as safe.
- **Annotations** are saved as the complete set for the image, one request at a time, so an
  older draft can never overwrite a newer one. Failures keep the draft and block navigation
  until the user retries or explicitly discards. The API is last-save-wins with no revision
  check, so editing the same image in two tabs is not supported.
- **Boxes** stay in normalized top-left `[x, y, width, height]` (not YOLO centers). The canvas
  draws in original-image pixels inside a zoom/pan group; resizes fold the transform scale back
  into width/height before committing.
- **Version and job creation are not idempotent.** The app never retries them. If a response is
  lost, it refreshes the list and requires the user to confirm before submitting again.
- **Live progress** uses `fetch` streaming (not `EventSource`) with the bearer header. The epoch
  cursor advances only after an epoch is applied; replays are upserted by epoch so points never
  duplicate; an invalid cursor (422) resets and replays history; 429 waits for `Retry-After`
  when readable; streams stop after `done`. Finished runs load every REST metrics page instead.
  The progress panel is best-effort and not replayed.
- Missing metrics display “Not available”, never zero. Accuracy/mAP values stay on their raw
  0–1 scale.

## Verification

Run these from `apps/web` with `make up` running. Playwright reuses the Compose web
server when available; otherwise it starts a temporary development server:

```bash
npm run typecheck && npm run lint && npm run format:check && npm test && npm run build
npm run test:e2e
```

Browser tests use the installed Google Chrome (`channel: "chrome"`); set
`E2E_CHANNEL=bundled` after `npx playwright install chromium` to use Playwright's build. They
create uniquely named accounts in the local application database and never delete other data.

| Suite                              | Kind                                   | Covers                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/*`                     | Mocked                                 | Coordinate transforms and the 200×100 example, stale-save ordering, refresh single-flight and failed refresh, SSE framing (split chunks/UTF-8, CRLF, comments, retry, multiline), replay cursors, duplicate-free upserts, 422 cursor reset, 429/`Retry-After`, terminal shutdown, abort, ambiguous POST outcomes                                                                                                                                                                         |
| `tests/e2e/classification.spec.ts` | **Real API, MinIO, Redis, CPU worker** | Register/login, project create/rename/append class, signed browser PUTs and CORS, invalid file rejection, dedupe, thumbnails, keyboard labeling, gallery filters, ImageFolder import, version validation error, version creation with stored counts, immutability after a label edit, job submit (validation, defaults), real SSE connection, queued cancellation, reload reconstruction, comparison, populated-project deletion, account isolation, asynchronous empty-project deletion |
| `tests/e2e/detection.spec.ts`      | **Real API**                           | Known box `[0.1,0.2,0.3,0.4]` at (20,20) 60×40, drawing with the mouse, zoom/resize without drift, keyboard select/move/reclassify/delete/undo, persistence after reload, unlabeled vs reviewed negative, YOLO and COCO imports                                                                                                                                                                                                                                                          |
| `tests/e2e/failures.spec.ts`       | Real accounts + mocked failures        | API unavailable, token expiry/refresh and refresh failure, failed annotation save, partial storage-PUT failure, expired signed URL, version validation, uncertain POST, SSE disconnect + `Last-Event-ID` replay + terminal stop, pending cancellation, unknown resources                                                                                                                                                                                                                 |
| `tests/e2e/accessibility.spec.ts`  | Real API                               | Keyboard-only sign-in, inline errors, dialog focus restoration, visible focus, phone-width layout                                                                                                                                                                                                                                                                                                                                                                                        |

The standard browser tests cancel submitted jobs while queued. The separately opted-in
`live-training.spec.ts` performs real training, review and redeployment; see [the walkthrough](../../docs/DEMO.md).

## Known limitations and missing backend contracts

- **Session persistence:** reload signs out (by design for this increment). Persistent sessions
  need an agreed HttpOnly-cookie or BFF contract.
- **Deployments:** create/stop endpoints, copy the one-time key and test JPEG/PNG predictions.
- **Monitoring:** request counts, latency percentiles/history plot, class distribution,
  confidence histogram and private prediction history.
- **Review:** correct classes or boxes, accept reviewed negatives, then freeze a new
  version for retraining. Failed acceptance retains the draft for an explicit retry.
- **Deletion:** populated projects can be deleted. The UI shows pending cleanup, disables
  editing, and polls until removal. Individual images and frozen versions have no delete API.

- **Desktop-first labeling:** the editors are designed for 1280 px and wider. Auth, project and
  list screens work at phone width.
- **Browser support:** current Chrome, Edge and Firefox (streaming `fetch`, `<dialog>`,
  `crypto.subtle`). Signed uploads and `crypto.subtle` need a secure context
  (`localhost`/`127.0.0.1` or HTTPS).

## Integration verification

The application uses generated types from the current backend, including asynchronous
project deletion. The API exposes `Retry-After` to browser clients. Pending requests cannot
cross an account switch or expire a newer session. Signed downloads request a new link on
each click and never send the API bearer token to storage.

The dependency audit found advisories in the supplied Next.js version. The application now
pins [Next.js 16.4.3](https://github.com/vercel/next.js/releases/tag/v16.4.3), above the
[patched version for the reported advisory](https://github.com/advisories/GHSA-2464-fhr4-w9xw).
`npm audit` reported zero known vulnerabilities after the update; this is not a guarantee
against undiscovered issues.

`tests/e2e/models.spec.ts` checks both report types, promotion, real private downloads and
missing-artifact errors. It uses the root `.venv` and API integration fixtures to seed synthetic
reports into a newly created, explicitly owned test project; these are not trained models or
quality measurements. Cleanup requests delete only those test projects. Other browser tests
create isolated accounts and sample images. No GPU worker should consume the test job queue during the normal regression suite.
`deployments.spec.ts` uses a tiny RGB-mean ONNX fixture and real Docker inference,
collection, review and version creation. `review.spec.ts` checks actual acceptance
of corrected detection boxes and reviewed negatives using seeded predictions.

`live-training.spec.ts` is separately opt-in: set `E2E_REAL_TRAINING=1`,
`E2E_CASTING_ARCHIVE` (240 labeled casting images in ImageFolder ZIP form), and
`E2E_CASTING_IMAGE` (a different, known defective image). Start an allocated GPU worker
and run `npm run test:e2e -- live-training.spec.ts`. It runs two real two-epoch jobs,
checks ONNX parity reports, deploys both resulting models, accepts the correction
and verifies the next version contains it. It records a local video and result JSON.
`E2E_KEEP_DEMO=1` retains the demo project; both endpoints are stopped at the end.
These short jobs verify integration, not baseline model quality. The training form loads
a directly requested version even if its cached list is older, and resets its selection
when the requested version changes. `training-version.spec.ts` covers that regression.

The production image uses a digest-pinned Node 22 base, a standalone Next.js build and a
non-root runtime with a read-only filesystem. `WEB_API_BASE_URL` is a public, build-time URL;
changing it requires rebuilding `web`. The browser cannot resolve Compose's internal `api`
hostname. `WEB_PORT` defaults to 3000; if changing the browser origin, update API, MinIO and
inference CORS settings to match. No service credentials enter the frontend build context.

Verification completed on 2026-10-25: type checking, lint, formatting, 30 unit tests,
production builds and Compose health checks passed. The 19 standard browser scenarios
and the separate real-GPU scenario passed, including the stale-version regression,
actual inference/collection, both review editors and retraining/redeployment.
The backend suite passed 100 tests. Production dependency audit reported zero known
vulnerabilities. See the [recorded workflow](../../docs/results/browser-workflow-20261025.json)
and [walkthrough](../../docs/DEMO.md). Raw traces, video and access details stay local.
