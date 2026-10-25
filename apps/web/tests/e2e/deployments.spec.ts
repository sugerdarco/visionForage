import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { apiAs, createProject, fixture, nav, registerAndLogin, uniqueEmail } from "./helpers";

test("deploy, predict, monitor, correct a label, freeze a new version and stop", async ({
  page,
}) => {
  test.setTimeout(240_000);
  const email = uniqueEmail("endpoint-flow");
  await registerAndLogin(page, email);
  const projectId = await createProject(page, "Endpoint workflow", "Classification", [
    "red",
    "green",
  ]);
  const api = await apiAs(page, email);
  const root = resolve("../..");
  try {
    execFileSync(
      resolve(root, ".venv/bin/python"),
      [resolve("tests/fixtures/seed_endpoint.py"), projectId, email],
      {
        cwd: root,
        env: {
          ...process.env,
          PYTHONPATH:
            "services/api:services/api/tests:services/api/integration_tests:services/inference/tests",
        },
        encoding: "utf8",
      },
    );
    await nav(page, "Deployments");
    await page.getByLabel("Review confidence threshold").fill("1");
    await page.getByRole("button", { name: "Deploy model", exact: true }).click();
    const keyField = page.getByLabel("New endpoint API key", { exact: true });
    await expect(keyField).toBeVisible({ timeout: 90_000 });
    expect(await keyField.getAttribute("type")).toBe("password");
    const key = await keyField.inputValue();
    const deploymentRow = (await api.get(`/projects/${projectId}/deployments`)).items[0];
    const deployment = await api.get(`/deployments/${deploymentRow.id}`);
    await expect(page.getByText("Status: running", { exact: true })).toBeVisible();
    await page.getByLabel("Prediction image").setInputFiles(fixture("upload/wide-200x100.png"));
    await page.getByLabel("Endpoint API key", { exact: true }).fill("wrong-key");
    await page.getByRole("button", { name: "Predict", exact: true }).click();
    await expect(page.getByText("The endpoint API key was rejected.")).toBeVisible();
    await page.getByLabel("Endpoint API key", { exact: true }).fill(key);
    const request = page.waitForRequest(
      (r) => r.url() === deployment.endpoint_url && r.method() === "POST",
    );
    await page.getByRole("button", { name: "Predict", exact: true }).click();
    const headers = (await request).headers();
    expect(headers["x-api-key"]).toBe(key);
    expect(headers.authorization).toBeUndefined();
    await expect(page.getByTestId("prediction-result")).toContainText("prediction_id");
    await nav(page, "Monitoring");
    await expect(page.getByTestId("recorded-requests")).toHaveText("1", { timeout: 90_000 });
    await page.getByText("View confidence histogram table", { exact: true }).click();
    await expect(page.getByRole("table", { name: "Confidence histogram" })).toBeVisible();
    await nav(page, "Review");
    await page.getByRole("button", { name: "Review image" }).click();
    const project = await api.get(`/projects/${projectId}`);
    const green = project.classes.find((c: { name: string }) => c.name === "green");
    await page.getByLabel("Correct class").selectOption(green.id);
    // A failed acceptance must keep the correction available for an explicit retry.
    await page.route(
      "**/review/*/accept",
      (route) =>
        route.fulfill({
          status: 503,
          contentType: "application/json",
          body: JSON.stringify({ detail: "Temporary collection outage" }),
        }),
      { times: 1 },
    );
    await page
      .getByRole("checkbox", { name: "I reviewed this image and corrected its labels" })
      .check();
    await page.getByRole("button", { name: "Accept reviewed image" }).click();
    await expect(page.getByText(/Your draft is retained/)).toBeVisible();
    await expect(page.getByLabel("Correct class")).toHaveValue(green.id);
    await page.getByRole("button", { name: "Accept reviewed image" }).click();
    await expect(page.getByText("Reviewed image accepted.", { exact: false })).toBeVisible();
    expect((await api.get(`/projects/${projectId}/review-queue`)).total).toBe(0);
    const history = await api.get(`/deployments/${deployment.id}/predictions`);
    expect(history.items[0].reviewed_at).toBeTruthy();
    const annotations = await api.get(`/images/${history.items[0].review_image_id}/annotations`);
    expect(JSON.stringify(annotations)).toContain(green.id);
    await nav(page, "Dataset versions");
    await page.getByRole("button", { name: "Create version", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Dataset version 2" })).toBeVisible();
    const versions = await api.get(`/projects/${projectId}/dataset-versions`);
    const version = versions.items.find((v: { version: number }) => v.version === 2);
    expect(Object.values(version.counts.images).reduce((a: number, b) => a + Number(b), 0)).toBe(7);
    await nav(page, "Deployments");
    await expect(page.getByLabel("New endpoint API key", { exact: true })).toHaveCount(0);
    await page.getByRole("button", { name: deployment.slug, exact: true }).click();
    await expect(page.getByLabel("Endpoint API key", { exact: true })).toHaveValue("");
    await page.getByRole("button", { name: "Stop endpoint", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Stop this endpoint?" });
    await dialog.getByRole("button", { name: "Keep running" }).click();
    expect((await api.get(`/deployments/${deployment.id}`)).status).toBe("running");
    await page.getByRole("button", { name: "Stop endpoint", exact: true }).click();
    await dialog.getByRole("button", { name: "Stop endpoint", exact: true }).click();
    await expect(page.getByText("Status: stopped", { exact: true })).toBeVisible();
    expect((await api.get(`/deployments/${deployment.id}/stats`)).requests).toBe(1);
  } finally {
    await api.remove(`/projects/${projectId}`);
  }
});
