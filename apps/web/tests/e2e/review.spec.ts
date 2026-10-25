import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";
import { apiAs, createProject, nav, registerAndLogin, uniqueEmail } from "./helpers";

test("detection review corrects normalized boxes and accepts an explicit negative", async ({
  page,
}) => {
  const email = uniqueEmail("review-detection");
  await registerAndLogin(page, email);
  const projectId = await createProject(page, "Detection review", "Object detection", [
    "ok",
    "defect",
  ]);
  const api = await apiAs(page, email);
  const root = resolve("../..");
  try {
    execFileSync(
      resolve(root, ".venv/bin/python"),
      [resolve("tests/fixtures/seed_review.py"), projectId, email],
      {
        cwd: root,
        env: {
          ...process.env,
          PYTHONPATH: "services/api:services/api/tests:services/api/integration_tests",
        },
        encoding: "utf8",
      },
    );
    const queue = await api.get(`/projects/${projectId}/review-queue`);
    await nav(page, "Review");
    await page.getByRole("button", { name: "Review image" }).first().click();
    await expect(page.getByTestId("review-canvas").locator("canvas").first()).toBeVisible();
    await page.getByLabel("Box 1 width", { exact: true }).fill("1");
    await expect(page.getByText(/positive sizes that fit/)).toBeVisible();
    await expect(page.getByRole("button", { name: "Accept reviewed image" })).toBeDisabled();
    await page.getByLabel("Box 1 width", { exact: true }).fill("0.5");
    // Cancelling internal navigation retains the unsaved box correction.
    page.once("dialog", (dialog) => dialog.dismiss());
    await nav(page, "Images");
    await expect(page.getByLabel("Box 1 width", { exact: true })).toHaveValue("0.5");
    await page
      .getByRole("checkbox", { name: "I reviewed this image and corrected its labels" })
      .check();
    await page.getByRole("button", { name: "Accept reviewed image" }).click();
    await expect(page.getByText("Reviewed image accepted.", { exact: false })).toBeVisible();
    const history = await api.get(`/deployments/${queue.items[0].deployment_id}/predictions`);
    const accepted = history.items.find((p: { id: string }) => p.id === queue.items[0].id);
    const annotations = await api.get(`/images/${accepted.review_image_id}/annotations`);
    expect(annotations.annotations[0].bbox).toEqual([0.1, 0.2, 0.5, 0.4]);
    expect(annotations.annotations[0].source).toBe("human");
    await page.getByRole("button", { name: "Review image" }).click();
    await page.getByRole("button", { name: "Remove box 1" }).click();
    await expect(page.getByRole("button", { name: "Accept reviewed image" })).toBeDisabled();
    await page
      .getByRole("checkbox", { name: "I reviewed this image and confirmed there are no defects" })
      .check();
    await page.getByRole("button", { name: "Accept reviewed image" }).click();
    await expect(page.getByText("No predictions awaiting review")).toBeVisible();
    const after = await api.get(`/deployments/${queue.items[0].deployment_id}/predictions`);
    const negative = after.items.find((p: { id: string }) => p.id === queue.items[1].id);
    expect(await api.get(`/images/${negative.review_image_id}/annotations`)).toMatchObject({
      status: "labeled",
      annotations: [],
    });
  } finally {
    await api.remove(`/projects/${projectId}`);
  }
});
