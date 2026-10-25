import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { apiAs, createProject, nav, registerAndLogin, uniqueEmail } from "./helpers";

test("revisiting training honors the requested version instead of a preserved selection", async ({
  page,
}) => {
  const email = uniqueEmail("model-training-version");
  await registerAndLogin(page, email);
  const projectId = await createProject(page, "Training version selection", "Classification", [
    "ok",
    "defect",
  ]);
  const api = await apiAs(page, email);
  const root = resolve("../..");
  try {
    execFileSync(
      resolve(root, ".venv/bin/python"),
      [
        resolve("tests/fixtures/seed_model.py"),
        projectId,
        email,
        "classification",
        "--version-count",
        "2",
      ],
      {
        cwd: root,
        env: {
          ...process.env,
          PYTHONPATH: "services/api:services/api/tests:services/api/integration_tests",
        },
        encoding: "utf8",
      },
    );
    const versions = await api.get(`/projects/${projectId}/dataset-versions`);
    // Simulate a cached version index from before v2 existed. A direct version
    // link must load v2 even while that index still contains only v1.
    await page.route(
      /\/dataset-versions\?.*page_size=100/,
      async (route) => {
        const response = await route.fetch();
        const data = await response.json();
        await route.fulfill({
          response,
          json: {
            ...data,
            items: data.items.filter((v: { version: number }) => v.version === 1),
            total: 1,
          },
        });
      },
      { times: 1 },
    );
    for (const number of [1, 2]) {
      const version = versions.items.find((v: { version: number }) => v.version === number);
      await nav(page, "Dataset versions");
      await page.locator(`a[href="/projects/${projectId}/versions/${version.id}"]`).click();
      await page.getByRole("link", { name: "Train on this version" }).click();
      await expect(page.getByLabel("Dataset version", { exact: true })).toHaveValue(version.id);
      await page.getByLabel("Epochs").fill("2");
      if (number === 2) {
        await page.getByRole("button", { name: "Start training" }).click();
        await expect(page).toHaveURL(/training\/[0-9a-f-]+$/);
        const jobId = page.url().split("/").pop()!;
        expect((await api.get(`/jobs/${jobId}`)).dataset_version_id).toBe(version.id);
        const cancelled = await api.post(`/jobs/${jobId}/cancel`);
        expect(cancelled.ok()).toBe(true);
      }
    }
  } finally {
    await api.remove(`/projects/${projectId}`);
  }
});
