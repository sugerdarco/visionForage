import { expect, test, type Page, type Route } from "@playwright/test";
import {
  API,
  apiAs,
  fixture,
  openAuthed,
  PASSWORD,
  register,
  signInHere,
  uniqueEmail,
} from "./helpers";

/** Mocked API responses need the same CORS header the real API sends to the app origin. */
const CORS = { "Access-Control-Allow-Origin": "http://127.0.0.1:3000" };

/** A real account with one classification project. */
async function seed(page: Page, tag: string, opts: { upload?: boolean } = {}) {
  const email = uniqueEmail(tag);
  await register(page, email);
  const api = await apiAs(page, email);
  const created = await api.post("/projects", {
    name: `Failure paths ${tag}`,
    task_type: "classification",
    classes: [
      { name: "ok", color: "#16a34a" },
      { name: "defect", color: "#dc2626" },
    ],
  });
  const project = await created.json();
  return { email, api, projectId: project.id as string, opts };
}

async function login(page: Page, email: string) {
  await page.goto("/login");
  await signInHere(page, email);
  await expect(page.getByRole("heading", { name: "Projects", exact: true })).toBeVisible();
}

test("API unavailable: login and list screens explain it and recover", async ({ page }) => {
  const { email } = await seed(page, "down");
  await page.route(`${API}/auth/login`, (route) => route.abort("connectionrefused"));
  await page.goto("/login");
  await signInHere(page, email);
  await expect(page.getByText(/could not be reached/)).toBeVisible();
  await page.unroute(`${API}/auth/login`);

  await page.route(`${API}/projects?**`, (route) => route.abort("connectionrefused"));
  await signInHere(page, email);
  await expect(page.getByRole("heading", { name: "API unavailable" })).toBeVisible({
    timeout: 30_000,
  });
  await page.unroute(`${API}/projects?**`);
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByRole("link", { name: /Failure paths down/ })).toBeVisible();
});

test("expired access token refreshes once; a rejected refresh signs out", async ({ page }) => {
  const { email } = await seed(page, "auth");
  let refreshes = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/auth/refresh")) refreshes++;
  });
  // Every list request is rejected as expired until the client has refreshed its token.
  await page.route(`${API}/projects?**`, (route) =>
    refreshes === 0
      ? route.fulfill({ status: 401, headers: CORS, json: { detail: "Token expired" } })
      : route.continue(),
  );
  await login(page, email);
  await expect(page.getByRole("link", { name: /Failure paths auth/ })).toBeVisible();
  expect(refreshes).toBe(1);
  await page.unroute(`${API}/projects?**`);

  await page.route(`${API}/auth/refresh`, (route) =>
    route.fulfill({ status: 401, headers: CORS, json: { detail: "Invalid token" } }),
  );
  await page.route(`${API}/projects/*`, (route) =>
    route.fulfill({ status: 401, headers: CORS, json: { detail: "Token expired" } }),
  );
  await page.getByRole("link", { name: /Failure paths auth/ }).click();
  await expect(page).toHaveURL(/\/login\?expired=1/);
  await expect(page.getByText("Your session ended")).toBeVisible();
});

test("failed annotation save keeps the draft and blocks silent navigation", async ({ page }) => {
  const { email, projectId, api } = await seed(page, "save");
  await login(page, email);
  await page.getByRole("link", { name: /Failure paths save/ }).click();
  await page.getByRole("button", { name: "Upload images" }).click();
  await page
    .getByLabel("Choose images to upload")
    .setInputFiles([fixture("upload/part-1.png"), fixture("upload/part-2.png")]);
  await expect(page.getByText("2 added")).toBeVisible({ timeout: 60_000 });

  await page.route(`${API}/images/*/annotations`, (route) =>
    route.request().method() === "PUT"
      ? route.fulfill({ status: 503, headers: CORS, json: { detail: "Database unavailable" } })
      : route.continue(),
  );
  await page.getByRole("link", { name: "Start labeling these" }).click();
  await expect(page.getByRole("radiogroup", { name: "Image class" })).toBeVisible();
  const imageId = page.url().match(/label\/([0-9a-f-]+)/)![1]!;
  await page.keyboard.press("2");
  await expect(page.getByText("Not saved: Database unavailable")).toBeVisible();
  await expect(page.getByRole("radio", { name: /defect/ })).toHaveAttribute("aria-checked", "true");
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Unsaved changes" })).toBeVisible();
  await page.getByRole("button", { name: "Stay and retry" }).click();
  await expect(page).toHaveURL(new RegExp(imageId));

  await page.unroute(`${API}/images/*/annotations`);
  await page.getByRole("button", { name: "Retry save" }).click();
  await expect(page.getByText("All changes saved")).toBeVisible();
  const saved = await api.get(`/images/${imageId}/annotations`);
  expect(saved.status).toBe("labeled");
  expect(projectId).toBeTruthy();
});

test("one failed storage PUT does not affect the other files; it can be retried", async ({
  page,
}) => {
  const { email } = await seed(page, "upload");
  await login(page, email);
  await page.getByRole("link", { name: /Failure paths upload/ }).click();
  let puts = 0;
  await page.route("http://127.0.0.1:9000/**", (route: Route) => {
    if (route.request().method() === "PUT" && puts++ === 0)
      return route.fulfill({ status: 500, body: "boom" });
    return route.continue();
  });
  await page.getByRole("button", { name: "Upload images" }).click();
  await page
    .getByLabel("Choose images to upload")
    .setInputFiles([fixture("upload/part-1.png"), fixture("upload/part-2.png")]);
  await expect(page.getByText("1 added · 0 already present · 1 failed")).toBeVisible({
    timeout: 60_000,
  });
  await page.getByRole("button", { name: "Retry 1 failed" }).click();
  await expect(page.getByText("2 added · 0 already present · 0 failed")).toBeVisible({
    timeout: 60_000,
  });
});

test("an expired signed thumbnail URL triggers fresh metadata", async ({ page }) => {
  const { email } = await seed(page, "expiry");
  await login(page, email);
  await page.getByRole("link", { name: /Failure paths expiry/ }).click();
  await page.getByRole("button", { name: "Upload images" }).click();
  await page.getByLabel("Choose images to upload").setInputFiles([fixture("upload/part-3.png")]);
  await expect(page.getByText("1 added")).toBeVisible({ timeout: 60_000 });
  await expect(page.locator('img[src*="thumbs"]')).toHaveCount(1, { timeout: 60_000 });

  // Serve one gallery response whose signature storage rejects, as an expired URL would be.
  let lists = 0;
  await page.route(`${API}/projects/*/images?**`, async (route) => {
    lists++;
    if (lists > 1) return route.continue();
    const response = await route.fetch();
    const body = await response.json();
    for (const item of body.items) {
      if (item.thumbnail_url)
        item.thumbnail_url = item.thumbnail_url.replace(
          /X-Amz-Signature=[0-9a-f]+/,
          "X-Amz-Signature=" + "0".repeat(64),
        );
    }
    return route.fulfill({ response, json: body });
  });
  await page.getByLabel("Status").selectOption("unlabeled");
  await expect.poll(() => lists, { timeout: 30_000 }).toBeGreaterThanOrEqual(2);
  await expect(page.locator('img[src*="thumbs"]')).not.toHaveAttribute(
    "src",
    /X-Amz-Signature=0{64}/,
  );
  const loaded = await page
    .locator('img[src*="thumbs"]')
    .evaluate((img: HTMLImageElement) => img.complete && img.naturalWidth > 0);
  expect(loaded).toBe(true);
});

test("dataset version: validation errors and an uncertain outcome", async ({ page }) => {
  const { email } = await seed(page, "version");
  await login(page, email);
  await page.getByRole("link", { name: /Failure paths version/ }).click();
  await page
    .getByRole("navigation", { name: "Project" })
    .getByRole("link", { name: /^Dataset versions/ })
    .click();
  await page.getByLabel("Validation").fill("20");
  await page.getByRole("button", { name: "Create version" }).click();
  await expect(page.getByText(/must add up to 100%/)).toBeVisible();
  await page.getByLabel("Validation").fill("15");
  await page.getByLabel("Seed").fill("-1");
  await page.getByRole("button", { name: "Create version" }).click();
  await expect(page.getByText(/Use a whole number from 0/)).toBeVisible();
  await page.getByLabel("Seed").fill("42");
  // Real API rejection: no labeled images.
  await page.getByRole("button", { name: "Create version" }).click();
  await expect(page.getByText("Version not created")).toBeVisible();

  // A lost response is never retried automatically.
  let posts = 0;
  await page.route(`${API}/projects/*/dataset-versions`, (route) => {
    if (route.request().method() === "POST") {
      posts++;
      return route.abort("connectionreset");
    }
    return route.continue();
  });
  await page.getByRole("button", { name: "Create version" }).click();
  await expect(page.getByText("Result unknown")).toBeVisible();
  const button = page.getByRole("button", { name: "Create version" });
  await expect(button).toBeDisabled();
  expect(posts).toBe(1);
  await page.getByLabel("I checked the list and want to create another version").check();
  await expect(button).toBeEnabled();
});

const JOB = "9f0c5a52-4f7e-4b7a-9c1e-0a0b0c0d0e0f";

function jobBody(projectId: string, patch: Record<string, unknown> = {}) {
  return {
    id: JOB,
    project_id: projectId,
    dataset_version_id: "1a2b3c4d-0000-4000-8000-000000000001",
    model_arch: "resnet18",
    hyperparams: {
      epochs: 5,
      batch_size: 64,
      patience: 5,
      seed: 42,
      learning_rate: 0.0003,
      weight_decay: 0.01,
      label_smoothing: 0.1,
      horizontal_flip: false,
      vertical_flip: false,
    },
    status: "running",
    gpu_id: 1,
    dispatched_at: "2026-10-22T10:00:00Z",
    cancel_requested_at: null,
    started_at: "2026-10-22T10:00:05Z",
    finished_at: null,
    error: null,
    model_id: null,
    created_at: "2026-10-22T10:00:00Z",
    updated_at: "2026-10-22T10:00:05Z",
    ...patch,
  };
}

function frame(event: string, data: Record<string, unknown>, id?: string) {
  const payload = JSON.stringify({ type: event, job_id: JOB, ts: "2026-10-22T10:01:00Z", ...data });
  return `${id ? `id: ${id}\n` : ""}event: ${event}\ndata: ${payload}\n\n`;
}

test("live stream: disconnect, replay with Last-Event-ID, no duplicate points, terminal stop", async ({
  page,
}) => {
  const { email, projectId } = await seed(page, "sse");
  await page.route(`${API}/jobs/${JOB}`, (route) =>
    route.fulfill({ headers: CORS, json: jobBody(projectId) }),
  );
  const cursors: (string | null)[] = [];
  await page.route(`${API}/jobs/${JOB}/events`, (route) => {
    const cursor = route.request().headers()["last-event-id"] ?? null;
    cursors.push(cursor);
    const status = frame("status", { status: "running", gpu_id: 1 });
    const body =
      cursors.length === 1
        ? // Stream drops after epoch 2 without a reconnect or done event.
          "retry: 50\n\n" +
          status +
          frame(
            "epoch",
            {
              epoch: 1,
              metrics: { train_loss: 0.9, val_loss: 1.0, val_accuracy: 0.5, lr: 0.0003 },
            },
            `epoch:${JOB}:1`,
          ) +
          frame("log", { line: "Running training, test evaluation and ONNX verification" }) +
          frame(
            "epoch",
            {
              epoch: 2,
              metrics: { train_loss: 0.6, val_loss: 0.8, val_accuracy: 0.7, lr: 0.0002 },
            },
            `epoch:${JOB}:2`,
          )
        : status +
          frame(
            "epoch",
            {
              epoch: 2,
              metrics: { train_loss: 0.6, val_loss: 0.8, val_accuracy: 0.7, lr: 0.0002 },
            },
            `epoch:${JOB}:2`,
          ) +
          frame(
            "epoch",
            {
              epoch: 3,
              metrics: { train_loss: 0.4, val_loss: 0.7, val_accuracy: 0.8, lr: 0.0001 },
            },
            `epoch:${JOB}:3`,
          ) +
          frame("done", {
            status: "succeeded",
            model_id: "7e7e7e7e-0000-4000-8000-000000000007",
            finished_at: "2026-10-22T10:05:00Z",
          });
    return route.fulfill({
      status: 200,
      headers: { ...CORS, "Content-Type": "text/event-stream" },
      body,
    });
  });
  await openAuthed(page, `/projects/${projectId}/training/${JOB}`, email);
  await expect(page.getByText("Training succeeded")).toBeVisible({ timeout: 30_000 });
  expect(cursors).toEqual([null, `epoch:${JOB}:2`]);
  await expect(page.getByText("3 epochs recorded (up to 5 requested)")).toBeVisible();
  await expect(page.getByText(/Early stopping ended training after 3 of 5 epochs/)).toBeVisible();
  const loss = page.getByRole("figure").filter({ hasText: "Loss" });
  await loss.getByText("Show data table").click();
  await expect(loss.locator("tbody tr")).toHaveCount(3);
  await expect(
    page.getByText("Running training, test evaluation and ONNX verification"),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /View registered model/ })).toBeVisible();
  await page.waitForTimeout(1500);
  expect(cursors).toHaveLength(2);
});

test("cancelling a running job shows a pending request, not success", async ({ page }) => {
  const { email, projectId } = await seed(page, "cancel");
  let cancelled = false;
  await page.route(`${API}/jobs/${JOB}`, (route) =>
    route.fulfill({
      headers: CORS,
      json: jobBody(projectId, cancelled ? { cancel_requested_at: "2026-10-22T10:02:00Z" } : {}),
    }),
  );
  await page.route(`${API}/jobs/${JOB}/cancel`, (route) => {
    cancelled = true;
    return route.fulfill({
      headers: CORS,
      json: jobBody(projectId, { cancel_requested_at: "2026-10-22T10:02:00Z" }),
    });
  });
  await page.route(`${API}/jobs/${JOB}/events`, (route) =>
    route.fulfill({
      status: 200,
      headers: { ...CORS, "Content-Type": "text/event-stream" },
      body:
        "retry: 1000\n\n" +
        frame("status", {
          status: "running",
          gpu_id: 1,
          cancel_requested_at: cancelled ? "2026-10-22T10:02:00Z" : null,
        }) +
        frame("reconnect", { reason: "stream_window_ended" }),
    }),
  );
  await openAuthed(page, `/projects/${projectId}/training/${JOB}`, email);
  await page.getByRole("button", { name: "Cancel run" }).click();
  await expect(page.getByRole("dialog")).toContainText("remains running until the worker confirms");
  await page.getByRole("dialog").getByRole("button", { name: "Cancel run" }).click();
  await expect(page.getByText("Cancellation requested").first()).toBeVisible();
  await expect(page.getByText(/stays “running” until the worker confirms/)).toBeVisible();
  await expect(page.getByText("Running", { exact: true })).toBeVisible();
  await expect(page.getByText("Cancelled", { exact: true })).toHaveCount(0);
});

test("unknown resources show a neutral not-found state", async ({ page }) => {
  const { email, projectId } = await seed(page, "missing");
  await openAuthed(page, `/projects/7b7b7b7b-0000-4000-8000-000000000000/images`, email);
  await expect(page.getByRole("heading", { name: "Not found" })).toBeVisible();
  await expect(page.getByText("not found or is not available to your account")).toBeVisible();
  await page.goto(`/projects/${projectId}/training/7b7b7b7b-0000-4000-8000-000000000001`);
  await signInHere(page, email);
  await expect(page.getByRole("heading", { name: "Not found" })).toBeVisible();
  expect(PASSWORD).toBeTruthy();
});
