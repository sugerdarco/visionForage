import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { apiAs, createProject, nav, PASSWORD, registerAndLogin, uniqueEmail } from "./helpers";

test.use({ video: { mode: "on", size: { width: 1366, height: 860 } } });
test.skip(process.env.E2E_REAL_TRAINING !== "1", "Needs an allocated live GPU worker");

// Opt in only with an allocated GPU worker and a prepared casting ImageFolder ZIP.
// Normal browser regression tests deliberately cancel jobs before training.
test("live casting: import, train, deploy, review, retrain and redeploy", async ({ page }) => {
  test.setTimeout(1_800_000);
  const archive = process.env.E2E_CASTING_ARCHIVE;
  const sample = process.env.E2E_CASTING_IMAGE;
  if (!archive || !sample)
    throw new Error(
      "Set E2E_CASTING_ARCHIVE and E2E_CASTING_IMAGE (a separate known defective image)",
    );
  const email = uniqueEmail("live-casting");
  await registerAndLogin(page, email);
  const projectId = await createProject(page, "Casting review and retraining", "Classification", [
    "ok",
    "defective",
  ]);
  const api = await apiAs(page, email);
  const evidence: Record<string, unknown> = {
    projectId,
    protocol:
      "240 public casting images, two epochs per run; workflow verification, not a model quality benchmark",
    runs: [],
    deployments: [],
  };
  const root = resolve("../..");
  await writeFile(
    resolve(root, "artifacts/demo-access.json"),
    JSON.stringify({ email, password: PASSWORD, projectId }, null, 2),
    { mode: 0o600 },
  );
  try {
    await page.getByRole("button", { name: "Import labeled ZIP" }).click();
    await page.getByLabel("Choose ZIP archive").setInputFiles(archive);
    await page.getByRole("button", { name: "Import", exact: true }).click();
    await expect(page.getByText("Import complete")).toBeVisible({ timeout: 180_000 });
    const endpoints: { id: string; model: string }[] = [];
    for (const version of [1, 2]) {
      await nav(page, "Dataset versions");
      await page.getByRole("button", { name: "Create version", exact: true }).click();
      await expect(page.getByRole("heading", { name: `Dataset version ${version}` })).toBeVisible({
        timeout: 120_000,
      });
      const datasetId = page.url().split("/").pop()!;
      const dataset = await api.get(`/projects/${projectId}/dataset-versions/${datasetId}`);
      expect(Object.values(dataset.counts.images).reduce((a: number, b) => a + Number(b), 0)).toBe(
        version === 1 ? 240 : 241,
      );
      await page.getByRole("link", { name: "Train on this version" }).click();
      await expect(page.getByLabel("Dataset version", { exact: true })).toHaveValue(datasetId);
      await page.getByLabel("Epochs").fill("2");
      await page.getByRole("button", { name: "Start training" }).click();
      await expect(page).toHaveURL(/training\/[0-9a-f-]+$/);
      const jobId = page.url().split("/").pop()!;
      expect((await api.get(`/jobs/${jobId}`)).dataset_version_id).toBe(datasetId);
      await expect(page.getByText("Live updates connected")).toBeVisible();
      await expect
        .poll(async () => (await api.get(`/jobs/${jobId}`)).status, {
          timeout: 600_000,
          intervals: [3000],
        })
        .toBe("succeeded");
      const job = await api.get(`/jobs/${jobId}`);
      expect(job.dataset_version_id).toBe(datasetId);
      (evidence.runs as unknown[]).push({
        jobId,
        datasetId,
        status: job.status,
        counts: dataset.counts.images,
      });
      await nav(page, "Models");
      await page.getByRole("link", { name: `Model v${version}`, exact: true }).click();
      await expect(page.getByRole("heading", { name: "Held-out test evaluation" })).toBeVisible();
      await expect(page.getByText("Passed on 32 test images")).toBeVisible();
      await page.getByRole("button", { name: "Use as preferred model" }).click();
      await expect(
        page.getByRole("button", { name: "Preferred model", exact: true }),
      ).toBeDisabled();
      await page.screenshot({
        path: resolve(root, `artifacts/demo-model-v${version}.png`),
        fullPage: true,
      });
      await page.getByRole("link", { name: "Deploy this model" }).click();
      await page.getByLabel("Review confidence threshold").fill("1");
      await page.getByRole("button", { name: "Deploy model", exact: true }).click();
      await expect(page.getByLabel("New endpoint API key", { exact: true })).toBeVisible({
        timeout: 120_000,
      });
      await page.getByLabel("Prediction image").setInputFiles(sample);
      await page.getByRole("button", { name: "Predict", exact: true }).click();
      await expect(page.getByTestId("prediction-result")).toContainText("prediction_id");
      const prediction = JSON.parse((await page.getByTestId("prediction-result").textContent())!);
      const list = await api.get(`/projects/${projectId}/deployments`);
      const endpoint = list.items.find(
        (d: { model_version: number }) => d.model_version === version,
      );
      expect(endpoint.model_id).toBe(job.model_id);
      expect(prediction.model_id).toBe(endpoint.model_id);
      endpoints.push({ id: endpoint.id, model: endpoint.model_id });
      (evidence.deployments as unknown[]).push({
        id: endpoint.id,
        modelId: endpoint.model_id,
        predictionId: prediction.prediction_id,
      });
      await nav(page, "Monitoring");
      await page.getByLabel("Monitored endpoint").selectOption(endpoint.id);
      await expect(page.getByTestId("recorded-requests")).toHaveText("1", { timeout: 90_000 });
      await page.screenshot({
        path: resolve(root, `artifacts/demo-monitoring-v${version}.png`),
        fullPage: true,
      });
      if (version === 1) {
        await nav(page, "Review");
        await page.getByRole("button", { name: "Review image" }).click();
        await page.getByLabel("Correct class").selectOption({ label: "defective" });
        await page
          .getByRole("checkbox", { name: "I reviewed this image and corrected its labels" })
          .check();
        await page.screenshot({ path: resolve(root, "artifacts/demo-review.png"), fullPage: true });
        await page.getByRole("button", { name: "Accept reviewed image" }).click();
        await expect(page.getByText("Reviewed image accepted.", { exact: false })).toBeVisible();
      }
    }
    expect(endpoints[0]!.model).not.toBe(endpoints[1]!.model);
    const firstDataset = (evidence.runs as { datasetId: string }[])[0]!.datasetId;
    const unchanged = await api.get(`/projects/${projectId}/dataset-versions/${firstDataset}`);
    expect(Object.values(unchanged.counts.images).reduce((a: number, b) => a + Number(b), 0)).toBe(
      240,
    );
    await nav(page, "Deployments");
    for (const endpoint of endpoints) {
      const detail = await api.get(`/deployments/${endpoint.id}`);
      const item = page
        .getByRole("listitem")
        .filter({ has: page.getByRole("button", { name: detail.slug, exact: true }) });
      await item.getByRole("button", { name: "Stop endpoint", exact: true }).click();
      await page
        .getByRole("dialog")
        .getByRole("button", { name: "Stop endpoint", exact: true })
        .click();
      await expect
        .poll(async () => (await api.get(`/deployments/${endpoint.id}`)).status)
        .toBe("stopped");
    }
    evidence.passed = true;
  } finally {
    await writeFile(
      resolve(root, "artifacts/live-browser-result.json"),
      JSON.stringify(evidence, null, 2),
    );
    if (process.env.E2E_KEEP_DEMO !== "1") await api.remove(`/projects/${projectId}`);
  }
});
