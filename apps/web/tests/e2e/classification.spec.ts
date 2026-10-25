import { expect, test } from "@playwright/test";
import {
  apiAs,
  createProject,
  fixture,
  nav,
  registerAndLogin,
  signInHere,
  uniqueEmail,
} from "./helpers";

test.describe.configure({ mode: "serial" });

const email = uniqueEmail("cls");
let projectId = "";

test("classification workflow: project → upload → label → import → version → train → compare", async ({
  page,
}) => {
  test.setTimeout(300_000);
  await registerAndLogin(page, email);
  projectId = await createProject(page, "Casting line", "Classification", ["ok", "defect"]);
  const api = await apiAs(page, email);

  // Settings: rename, append a class, reject a duplicate class.
  await nav(page, "Settings");
  await page.getByLabel("Name", { exact: true }).fill("Casting line 3");
  await page.getByRole("button", { name: "Rename" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Saved." })).toBeVisible();
  await expect(page.getByRole("complementary").getByText("Casting line 3")).toBeVisible();
  await page.getByRole("button", { name: "Add class" }).click();
  await page.getByLabel("Name of class 1").fill("ok");
  await page.getByRole("button", { name: "Append class" }).click();
  await expect(page.getByText("This class name is already used.")).toBeVisible();
  await page.getByLabel("Name of class 1").fill("porosity");
  await page.getByRole("button", { name: "Append class" }).click();
  await expect(page.getByRole("cell", { name: "porosity" })).toBeVisible();
  const project = await api.get(`/projects/${projectId}`);
  expect(project.name).toBe("Casting line 3");
  expect(project.classes.map((c: { name: string; index: number }) => [c.name, c.index])).toEqual([
    ["ok", 0],
    ["defect", 1],
    ["porosity", 2],
  ]);

  // Upload: two good files and one invalid file; the bad one does not erase the others.
  await nav(page, "Images");
  await page.getByRole("button", { name: "Upload images" }).click();
  await page
    .getByLabel("Choose images to upload")
    .setInputFiles([
      fixture("upload/part-1.png"),
      fixture("upload/part-2.png"),
      fixture("upload/not-an-image.png"),
    ]);
  await expect(page.getByText("2 added · 0 already present · 1 failed")).toBeVisible({
    timeout: 60_000,
  });
  // Re-uploading identical bytes is reported as a duplicate, not a new image.
  await page.getByRole("button", { name: "Clear list" }).click();
  await page.getByLabel("Choose images to upload").setInputFiles([fixture("upload/part-1.png")]);
  await expect(page.getByText("0 added · 1 already present · 0 failed")).toBeVisible({
    timeout: 60_000,
  });
  await expect(page.getByRole("button", { name: /Retry/ })).toHaveCount(0);
  await expect(page.getByRole("link", { name: /Label image/ })).toHaveCount(2);
  await expect(page.locator('img[src*="thumbs"]')).toHaveCount(2, { timeout: 60_000 });

  // Label both images with keyboard shortcuts.
  await page.getByLabel("Status").selectOption("unlabeled");
  await expect(page).toHaveURL(/status=unlabeled/);
  await page.getByRole("link", { name: "Start labeling these" }).click();
  await expect(page.getByRole("radiogroup", { name: "Image class" })).toBeVisible();
  const firstId = page.url().match(/label\/([0-9a-f-]+)/)![1]!;
  await page.keyboard.press("2");
  await expect(page.getByText("All changes saved")).toBeVisible();
  await expect(page.getByRole("radio", { name: /defect/ })).toHaveAttribute("aria-checked", "true");
  await page.keyboard.press("ArrowRight");
  await expect(page).not.toHaveURL(new RegExp(firstId));
  const secondId = page.url().match(/label\/([0-9a-f-]+)/)![1]!;
  await expect(page.getByText(`Image ${secondId.slice(0, 8)}`)).toBeVisible();
  await page.keyboard.press("1");
  await expect(page.getByRole("radio", { name: /^ok/ })).toHaveAttribute("aria-checked", "true");
  await expect(page.getByText("All changes saved")).toBeVisible();
  const byName = Object.fromEntries(
    project.classes.map((c: { name: string; id: string }) => [c.name, c.id]),
  );
  await expect
    .poll(async () => (await api.get(`/images/${secondId}/annotations`)).annotations)
    .toMatchObject([{ class_id: byName.ok }]);
  const first = await api.get(`/images/${firstId}/annotations`);
  expect(first).toMatchObject({
    status: "labeled",
    annotations: [{ class_id: byName.defect, bbox: null }],
  });
  // Back to the first image: the saved label is restored from the server.
  await page.getByRole("button", { name: "Previous" }).click();
  await expect(page.getByRole("radio", { name: /defect/ })).toHaveAttribute("aria-checked", "true");
  await page.getByRole("button", { name: "← Images" }).click();
  await expect(page).toHaveURL(/images\?status=unlabeled/);
  await page.getByLabel("Status").selectOption("labeled");
  await expect(page.getByRole("link", { name: /Label image/ })).toHaveCount(2);
  await page.getByLabel("Class").selectOption({ label: "defect" });
  await expect(page.getByRole("link", { name: /Label image/ })).toHaveCount(1);

  // Import an ImageFolder ZIP large enough for a trainable version.
  await page.getByRole("button", { name: "Import labeled ZIP" }).click();
  await page.getByLabel("Choose ZIP archive").setInputFiles(fixture("imagefolder.zip"));
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(page.getByText("Import complete")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText("New images: 230")).toBeVisible();

  // Freeze a version. The porosity class has no examples, so creation is rejected with the API's reason.
  await nav(page, "Dataset versions");
  await page.getByRole("button", { name: "Create version" }).click();
  await expect(page.getByText("Version not created")).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/training example/)).toBeVisible();
  // Label one porosity example per split need: give porosity three distinct images via upload + labeling.
  await nav(page, "Images");
  await page.getByRole("button", { name: "Upload images" }).click();
  await page
    .getByLabel("Choose images to upload")
    .setInputFiles([fixture("upload/part-3.png"), fixture("upload/wide-200x100.png")]);
  await expect(page.getByText("2 added · 0 already present · 0 failed")).toBeVisible({
    timeout: 60_000,
  });
  await page.getByLabel("Status").selectOption("unlabeled");
  await expect(page).toHaveURL(/status=unlabeled/);
  await page.getByRole("link", { name: "Start labeling these" }).click();
  await expect(page.getByRole("radiogroup", { name: "Image class" })).toBeVisible();
  await page.keyboard.press("3");
  await expect(page.getByRole("radio", { name: /porosity/ })).toHaveAttribute(
    "aria-checked",
    "true",
  );
  await expect(page.getByText("All changes saved")).toBeVisible();
  const porosityFirst = page.url();
  await page.getByRole("button", { name: "Next image →" }).click();
  await expect(page).not.toHaveURL(porosityFirst);
  await expect(page.getByRole("radio", { name: /porosity/ })).toHaveAttribute(
    "aria-checked",
    "false",
  );
  await page.keyboard.press("3");
  await expect(page.getByRole("radio", { name: /porosity/ })).toHaveAttribute(
    "aria-checked",
    "true",
  );
  await expect(page.getByText("All changes saved")).toBeVisible();
  // A third distinct porosity example comes from relabeling an uploaded image.
  const imgs = await api.get(
    `/projects/${projectId}/images?status=labeled&class=${byName.ok}&page_size=1`,
  );
  await api.put(`/images/${imgs.items[0].id}/annotations`, {
    annotations: [{ class_id: byName.porosity }],
  });

  await page.getByRole("button", { name: "← Images" }).click();
  await nav(page, "Dataset versions");
  const create = page.getByRole("button", { name: "Create version" });
  await create.click();
  // Duplicate clicks are blocked while the build runs.
  await expect(page.getByRole("button", { name: "Building version…" })).toBeDisabled();
  await expect(page).toHaveURL(/versions\/[0-9a-f-]+$/, { timeout: 120_000 });
  const versionId = page.url().split("/").pop()!;
  const version = await api.get(`/projects/${projectId}/dataset-versions/${versionId}`);
  await expect(
    page.getByRole("heading", { name: `Dataset version ${version.version}` }),
  ).toBeVisible();
  for (const [label, split] of [
    ["Train", "train"],
    ["Validation", "val"],
    ["Test", "test"],
  ] as const) {
    await expect(
      page
        .getByRole("row", { name: new RegExp(`^${label} `) })
        .getByRole("cell")
        .nth(2),
    ).toHaveText(String(version.counts.images[split]));
  }
  expect(version.counts.images.test).toBeGreaterThanOrEqual(32);
  expect(
    await page.getByRole("link", { name: "Download manifest" }).getAttribute("href"),
  ).toContain("X-Amz-Signature");

  // A later label change does not alter the frozen version.
  await api.put(`/images/${firstId}/annotations`, { annotations: [{ class_id: byName.ok }] });
  const unchanged = await api.get(`/projects/${projectId}/dataset-versions/${versionId}`);
  expect(unchanged.counts).toEqual(version.counts);

  // Submit one training run, watch its queued state, then cancel it before any GPU claims it.
  await page.getByRole("link", { name: "Train on this version" }).click();
  await expect(page.getByLabel("Epochs")).toHaveValue("20");
  await expect(page.getByLabel("Learning rate")).toHaveValue("0.0003");
  await page.getByLabel("Epochs").fill("0");
  await page.getByRole("button", { name: "Start training" }).click();
  await expect(page.getByText("Use a value from 1 to 1000.")).toBeVisible();
  await page.getByLabel("Epochs").fill("2");
  await page.getByRole("button", { name: "Start training" }).click();
  await expect(page).toHaveURL(/training\/[0-9a-f-]+$/);
  const jobId = page.url().split("/").pop()!;
  // The real SSE endpoint accepts the bearer stream and reports the queued state.
  await expect(page.getByText("Live updates connected")).toBeVisible();
  await expect(page.getByText("Waiting for a GPU worker")).toBeVisible();
  await page.getByRole("button", { name: "Cancel run" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Cancel run" }).click();
  await expect(page.getByText("This run was cancelled")).toBeVisible();
  // The stream delivers the terminal state and stops reconnecting.
  await expect(page.getByText("Live updates connected")).toHaveCount(0);
  const job = await api.get(`/jobs/${jobId}`);
  expect(job).toMatchObject({
    status: "cancelled",
    model_arch: "resnet18",
    dataset_version_id: versionId,
  });
  expect(job.hyperparams).toMatchObject({ epochs: 2, batch_size: 64, learning_rate: 0.0003 });
  const jobs = await api.get(`/projects/${projectId}/jobs`);
  expect(jobs.total).toBe(1);

  // Reload reconstructs the job from the server (after signing in again: sessions are in-memory).
  await page.reload();
  await expect(page).toHaveURL(/\/login\?next=/);
  await signInHere(page, email);
  await expect(page.getByText("This run was cancelled")).toBeVisible();
  await expect(page.locator("div.contents", { hasText: "Epochs" })).toContainText("2");

  // Compare view lists the run's settings.
  await nav(page, "Compare runs");
  await page.getByRole("checkbox").first().click();
  await expect(page.getByRole("checkbox").first()).toBeChecked();
  await expect(page.getByRole("rowheader", { name: "Learning rate" })).toBeVisible();
  await expect(page.getByText("None of these runs has recorded epoch metrics.")).toBeVisible();
});

test("accounts are isolated and empty projects can be deleted", async ({ page }) => {
  const other = uniqueEmail("other");
  await registerAndLogin(page, other);
  await expect(page.getByText("No projects yet")).toBeVisible();
  // Another account's project looks exactly like a missing one.
  await page.goto(`/projects/${projectId}/images`);
  await signInHere(page, other);
  await expect(page.getByRole("heading", { name: "Not found" })).toBeVisible();
  await page.getByRole("link", { name: "Back to projects" }).click();

  await page.getByRole("button", { name: "Create your first project" }).click();
  const dialog = page.getByRole("dialog", { name: "New project" });
  await dialog.getByLabel("Project name").fill("Scratch project");
  await dialog.getByLabel("Name of class 1").fill("a");
  await dialog.getByRole("button", { name: "Create project" }).click();
  await expect(page).toHaveURL(/images/);
  await nav(page, "Settings");
  await page.getByRole("button", { name: "Delete project" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Delete" }).click();
  await expect(page.getByRole("heading", { name: "Deleting project" })).toBeVisible();
  await page.getByRole("link", { name: "Back to projects" }).click();
  await expect(page.getByText("No projects yet")).toBeVisible({ timeout: 30000 });

  // Signing out and in as the first account shows only that account's data.
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/login$/);
  await signInHere(page, email);
  await expect(page.getByRole("link", { name: /Casting line 3/ })).toBeVisible();
  await expect(page.getByText("Scratch project")).toHaveCount(0);
  await page.getByRole("link", { name: /Casting line 3/ }).click();
  await nav(page, "Settings");
  await page.getByRole("button", { name: "Delete project" }).click();
  await expect(page.getByRole("dialog")).toContainText("all its data");
  await page.getByRole("dialog").getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Deleting project" })).toBeVisible();
  await page.getByRole("link", { name: "Back to projects" }).click();
  await expect(page.getByText("No projects yet")).toBeVisible({ timeout: 30000 });
});
