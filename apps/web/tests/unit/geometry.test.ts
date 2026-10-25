import { describe, expect, it } from "vitest";
import {
  commitTransform,
  fitView,
  imageToNormalized,
  normalizedToImage,
  rectFromPoints,
  toImagePoint,
  toScreenPoint,
  translateBox,
  zoomAt,
  type NormalizedBox,
} from "@/lib/geometry/boxes";

const image = { width: 200, height: 100 };

describe("box coordinates", () => {
  it("maps the documented 200×100 example to pixels", () => {
    expect(normalizedToImage([0.1, 0.2, 0.3, 0.4], image)).toEqual({
      x: 20,
      y: 20,
      width: 60,
      height: 40,
    });
  });

  it("round-trips through any zoom and pan without drift", () => {
    const box: NormalizedBox = [0.1, 0.2, 0.3, 0.4];
    for (const view of [
      { scale: 1, offsetX: 0, offsetY: 0 },
      { scale: 3.7, offsetX: -125.5, offsetY: 42 },
      { scale: 0.25, offsetX: 300, offsetY: -12 },
    ]) {
      const rect = normalizedToImage(box, image);
      const topLeft = toScreenPoint({ x: rect.x, y: rect.y }, view);
      const bottomRight = toScreenPoint({ x: rect.x + rect.width, y: rect.y + rect.height }, view);
      const back = imageToNormalized(
        rectFromPoints(toImagePoint(topLeft, view), toImagePoint(bottomRight, view)),
        image,
      );
      back!.forEach((value, i) => expect(value).toBeCloseTo(box[i]!, 12));
    }
  });

  it("clamps to the image and rejects zero-area boxes", () => {
    const clamped = imageToNormalized({ x: -20, y: 90, width: 60, height: 40 }, image)!;
    [0, 0.9, 0.2, 0.1].forEach((value, i) => expect(clamped[i]).toBeCloseTo(value, 12));
    expect(imageToNormalized({ x: 10, y: 10, width: 0, height: 30 }, image)).toBeNull();
    expect(imageToNormalized({ x: 250, y: 10, width: 20, height: 30 }, image)).toBeNull();
  });

  it("normalizes negative drags and never exceeds the right edge", () => {
    const box = imageToNormalized({ x: 200, y: 100, width: -200, height: -100 }, image)!;
    expect(box).toEqual([0, 0, 1, 1]);
    const edge = imageToNormalized({ x: 133.3333, y: 0, width: 80, height: 10 }, image)!;
    expect(edge[0] + edge[2]).toBeLessThanOrEqual(1);
  });

  it("removes the transform scale when committing a resize", () => {
    expect(
      commitTransform({ x: 20, y: 20, width: 60, height: 40, scaleX: 1.5, scaleY: 0.5 }),
    ).toEqual({ x: 20, y: 20, width: 90, height: 20 });
  });

  it("keeps translated boxes inside the image", () => {
    expect(translateBox([0.8, 0.8, 0.3, 0.1], 0.5, -1)).toEqual([0.7, 0, 0.3, 0.1]);
  });

  it("fits and zooms around an anchor point", () => {
    const view = fitView(image, { width: 432, height: 232 });
    expect(view.scale).toBe(2);
    expect(toScreenPoint({ x: 0, y: 0 }, view)).toEqual({ x: 16, y: 16 });
    const anchor = { x: 100, y: 50 };
    const before = toImagePoint(anchor, view);
    const after = toImagePoint(anchor, zoomAt(view, 2.5, anchor));
    expect(after.x).toBeCloseTo(before.x, 10);
    expect(after.y).toBeCloseTo(before.y, 10);
  });
});
