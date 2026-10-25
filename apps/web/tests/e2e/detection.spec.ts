import { expect, test, type Page } from "@playwright/test";
import { apiAs, createProject, fixture, nav, registerAndLogin, uniqueEmail } from "./helpers";

/** Screen position of an image-space point, mirroring fitView() for the current stage size. */
async function screenPoint(
  page: Page,
  image: { width: number; height: number },
  x: number,
  y: number,
) {
  const box = (await page.getByTestId("detection-stage").boundingBox())!;
  const scale = Math.min((box.width - 32) / image.width, (box.height - 32) / image.height);
  const offsetX = (box.width - image.width * scale) / 2;
  const offsetY = (box.height - image.height * scale) / 2;
  return { x: box.x + offsetX + x * scale, y: box.y + offsetY + y * scale };
}

async function savedBoxes(api: Awaited<ReturnType<typeof apiAs>>, imageId: string) {
  return (await api.get(`/images/${imageId}/annotations`)) as {
    status: string;
    annotations: { class_id: string; bbox: number[] }[];
  };
}

test("detection: coordinates, drawing, editing, persistence, negatives and imports", async ({
  page,
}) => {
  test.setTimeout(240_000);
  const email = uniqueEmail("det");
  await registerAndLogin(page, email);
  const projectId = await createProject(page, "Steel surface", "Object detection", [
    "scratch",
    "dent",
  ]);
  const api = await apiAs(page, email);
  const classes = (await api.get(`/projects/${projectId}`)).classes as {
    id: string;
    name: string;
  }[];
  const scratch = classes.find((c) => c.name === "scratch")!.id;
  const dent = classes.find((c) => c.name === "dent")!.id;

  await page.getByRole("button", { name: "Upload images" }).click();
  await page
    .getByLabel("Choose images to upload")
    .setInputFiles([fixture("upload/wide-200x100.png")]);
  await expect(page.getByText("1 added")).toBeVisible({ timeout: 60_000 });
  const imageId = (await api.get(`/projects/${projectId}/images`)).items[0].id as string;

  // The documented example: [0.1, 0.2, 0.3, 0.4] on 200×100 starts at (20, 20) and is 60×40.
  await api.put(`/images/${imageId}/annotations`, {
    annotations: [{ class_id: scratch, bbox: [0.1, 0.2, 0.3, 0.4] }],
  });
  await page.getByRole("link", { name: /Label image/ }).click();
  await expect(page.getByTestId("detection-stage")).toBeVisible();
  await expect(page.getByTestId("box-pixels").first()).toHaveText("x 20.0, y 20.0, 60.0 × 40.0 px");
  await page.waitForTimeout(500);
  await page.screenshot({ path: "test-results/screens/detection-fit.png" });

  // Draw a second box from image (100, 50) to (160, 90) with the dent class.
  await page.getByLabel("Class for new boxes").selectOption(dent);
  await page.getByRole("button", { name: "Draw" }).click();
  const start = await screenPoint(page, { width: 200, height: 100 }, 100, 50);
  const end = await screenPoint(page, { width: 200, height: 100 }, 160, 90);
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move((start.x + end.x) / 2, (start.y + end.y) / 2, { steps: 4 });
  await page.mouse.move(end.x, end.y, { steps: 4 });
  await page.mouse.up();
  await expect(page.getByTestId("box-row")).toHaveCount(2);
  await expect(page.getByText("All changes saved")).toBeVisible();
  await expect.poll(async () => (await savedBoxes(api, imageId)).annotations.length).toBe(2);
  const drawn = (await savedBoxes(api, imageId)).annotations.find((a) => a.class_id === dent)!;
  [0.5, 0.5, 0.3, 0.4].forEach((value, i) => expect(drawn.bbox[i]).toBeCloseTo(value, 1));

  // Zooming and panning change the view only, never the stored geometry.
  const before = await page.getByTestId("box-pixels").allTextContents();
  await page.getByRole("button", { name: "Zoom in" }).click();
  await page.getByRole("button", { name: "Zoom in" }).click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: "test-results/screens/detection-zoomed.png" });
  await page.setViewportSize({ width: 1100, height: 760 });
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: "Fit" }).click();
  await page.waitForTimeout(300);
  await page.screenshot({ path: "test-results/screens/detection-resized.png" });
  await expect(page.getByTestId("box-pixels")).toHaveText(before);
  await page.setViewportSize({ width: 1366, height: 860 });

  // Keyboard-only editing: select from the list, nudge, reclassify, delete, undo.
  await page.getByRole("button", { name: /^Box 1:/ }).click();
  await page.locator("body").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByTestId("box-pixels").first()).toHaveText("x 21.0, y 20.0, 60.0 × 40.0 px");
  await page.keyboard.press("2");
  await expect(page.getByRole("button", { name: /^Box 1: dent/ })).toBeVisible();
  await page.keyboard.press("Delete");
  await expect(page.getByTestId("box-row")).toHaveCount(1);
  await page.keyboard.press("Control+z");
  await expect(page.getByTestId("box-row")).toHaveCount(2);
  await expect(page.getByText("All changes saved")).toBeVisible({ timeout: 10_000 });
  await expect
    .poll(async () =>
      (await savedBoxes(api, imageId)).annotations.map((a) => a.bbox[0]!.toFixed(3)).sort(),
    )
    .toEqual(["0.105", drawn.bbox[0]!.toFixed(3)].sort());

  // Leaving and returning reloads exactly what the server stored.
  const stored = await page.getByTestId("box-pixels").allTextContents();
  await page.getByRole("button", { name: "← Images" }).click();
  await page.getByRole("link", { name: /Label image/ }).click();
  // The API orders annotation rows by UUID, not by their geometry. Compare the
  // complete box set on both sides without imposing a visual row order.
  await expect
    .poll(async () => (await page.getByTestId("box-pixels").allTextContents()).sort())
    .toEqual([...stored].sort());

  // Remove all boxes: unlabeled. Then mark as a reviewed negative: labeled with no boxes.
  await page.getByRole("button", { name: "Delete box 2" }).click();
  await page.getByRole("button", { name: "Delete box 1" }).click();
  await expect(page.getByTestId("label-summary")).toHaveText("Unlabeled");
  await expect.poll(async () => (await savedBoxes(api, imageId)).status).toBe("unlabeled");
  await page.getByLabel("Reviewed — no defects in this image").check();
  await expect
    .poll(async () => savedBoxes(api, imageId))
    .toMatchObject({ status: "labeled", annotations: [] });
  await page.getByRole("button", { name: "← Images" }).click();
  await page.getByLabel("Status").selectOption("labeled");
  await expect(page.getByRole("link", { name: /Label image .* \(Labeled\)/ })).toHaveCount(1);
  await page.getByLabel("Status").selectOption("");

  // YOLO import: the missing-label notice is shown before importing.
  await page.getByRole("button", { name: "Import labeled ZIP" }).click();
  await expect(page.getByText("Missing label files become reviewed negatives")).toBeVisible();
  await page.getByLabel("Choose ZIP archive").setInputFiles(fixture("yolo.zip"));
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(page.getByText("Import complete")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("New images: 4")).toBeVisible();
  await expect(page.getByText("New classes: 0")).toBeVisible();

  // COCO import adds the new "pit" class.
  await page.getByLabel("Format").selectOption("coco");
  await expect(page.getByLabel("Annotations file")).toHaveValue("annotations.json");
  await page.getByLabel("Choose ZIP archive").setInputFiles(fixture("coco.zip"));
  await page.getByRole("button", { name: "Import", exact: true }).click();
  await expect(page.getByText("Import complete")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByText("New images: 3")).toBeVisible();
  await expect(page.getByText("New classes: 1")).toBeVisible();
  await expect(page.getByLabel("Class")).toContainText("pit");
  const coco = await api.get(`/projects/${projectId}/images?page_size=100`);
  expect(coco.total).toBe(8);

  await nav(page, "Settings");
  await expect(page.getByRole("cell", { name: "pit" })).toBeVisible();
});
