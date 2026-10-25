import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { apiAs, createProject, nav, registerAndLogin, uniqueEmail } from "./helpers";

for (const task of ["classification", "detection"] as const) {
  test(`${task} model: reports, private artifact downloads and promotion`, async ({ page }) => {
    const email = uniqueEmail("model-" + task);
    await registerAndLogin(page, email);
    const projectId = await createProject(
      page,
      "Model test " + task,
      task === "classification" ? "Classification" : "Object detection",
      ["ok", "defect"],
    );
    const api = await apiAs(page, email);
    const root = resolve("../..");
    try {
      const modelId = execFileSync(
        resolve(root, ".venv/bin/python"),
        [resolve("tests/fixtures/seed_model.py"), projectId, email, task],
        {
          cwd: root,
          env: {
            ...process.env,
            PYTHONPATH: "services/api:services/api/tests:services/api/integration_tests",
          },
          encoding: "utf8",
        },
      ).trim();
      await nav(page, "Models");
      await page.getByRole("link", { name: "Model v1" }).click();
      await expect(page.getByRole("heading", { name: "Held-out test evaluation" })).toBeVisible();
      await expect(page.getByRole("table", { name: "Per-class results" })).toContainText("defect");
      await expect(page.getByText("Passed on 32 test images")).toBeVisible();
      await expect(page.getByText("Mean 20.00 ms · p95 22.00 ms")).toBeVisible();
      if (task === "classification") {
        await expect(page.getByRole("table", { name: /Confusion matrix/ })).toContainText("16");
      } else {
        await expect(page.getByText("Not available", { exact: true })).toHaveCount(4);
      }
      await page.getByRole("button", { name: "Use as preferred model" }).click();
      await expect(
        page.getByRole("button", { name: "Preferred model", exact: true }),
      ).toBeDisabled();
      expect((await api.get(`/models/${modelId}`)).is_promoted).toBe(true);
      const downloaded = page.waitForEvent("download");
      await page.getByRole("button", { name: "Download weights" }).click();
      const download = await downloaded;
      expect(download.suggestedFilename()).toBe("best.pt");
      expect(await readFile((await download.path())!, "utf8")).toBe("fixture weights");
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({
        path: resolve(root, "artifacts", `frontend-${task}-model.png`),
        fullPage: true,
      });
      // Losing storage access is shown as an error without hiding other model controls.
      await page.route(`**/models/${modelId}/artifacts/onnx`, (route) =>
        route.fulfill({
          status: 409,
          headers: { "Access-Control-Allow-Origin": "http://127.0.0.1:3000" },
          contentType: "application/json",
          body: JSON.stringify({ detail: "Registered model artifact unavailable" }),
        }),
      );
      await page.getByRole("button", { name: "Download onnx" }).click();
      await expect(
        page.getByRole("alert").filter({ hasText: "Registered model artifact unavailable" }),
      ).toBeVisible();
    } finally {
      await api.remove(`/projects/${projectId}`);
    }
  });
}
