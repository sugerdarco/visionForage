/**
 * Coordinate spaces used by the detection editor:
 * - normalized: the API box `[x, y, width, height]`, top-left origin, fractions of the
 *   original image (not YOLO centers, not x1/y1/x2/y2);
 * - image: pixels of the original image;
 * - screen: canvas pixels after zoom (`scale`) and pan (`offsetX`, `offsetY`).
 * Canonical editor data stays normalized so viewport changes never move a box.
 */

export type NormalizedBox = [number, number, number, number];

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface View {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export const MIN_SCALE = 0.02;
export const MAX_SCALE = 40;

export function toImagePoint(screen: Point, view: View): Point {
  return { x: (screen.x - view.offsetX) / view.scale, y: (screen.y - view.offsetY) / view.scale };
}

export function toScreenPoint(image: Point, view: View): Point {
  return { x: image.x * view.scale + view.offsetX, y: image.y * view.scale + view.offsetY };
}

export function normalizedToImage(box: NormalizedBox, image: Size): Rect {
  const [x, y, width, height] = box;
  return {
    x: x * image.width,
    y: y * image.height,
    width: width * image.width,
    height: height * image.height,
  };
}

/** Clamp an image-space rectangle to the image and normalize it, or null if it has no area. */
export function imageToNormalized(rect: Rect, image: Size): NormalizedBox | null {
  let { x, y, width, height } = rect;
  if (width < 0) {
    x += width;
    width = -width;
  }
  if (height < 0) {
    y += height;
    height = -height;
  }
  const left = Math.min(Math.max(x, 0), image.width);
  const top = Math.min(Math.max(y, 0), image.height);
  const right = Math.min(Math.max(x + width, 0), image.width);
  const bottom = Math.min(Math.max(y + height, 0), image.height);
  if (right - left < MIN_BOX_PIXELS || bottom - top < MIN_BOX_PIXELS) return null;
  const nx = left / image.width;
  const ny = top / image.height;
  // Compute extents from clamped edges so x + width never exceeds 1 by rounding.
  const nw = Math.min(right / image.width, 1) - nx;
  const nh = Math.min(bottom / image.height, 1) - ny;
  if (!(nw > 0 && nh > 0)) return null;
  return [nx, ny, nw, nh];
}

/** Smallest box edge, in original-image pixels, that the editor commits. */
export const MIN_BOX_PIXELS = 1;

/** Rectangle spanned by two image-space points (drag start and current pointer). */
export function rectFromPoints(a: Point, b: Point): Rect {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(a.x - b.x),
    height: Math.abs(a.y - b.y),
  };
}

/**
 * Konva resizes by changing node scale. Fold that scale back into width/height
 * before committing so the stored box reflects the visible extent.
 */
export function commitTransform(node: {
  x: number;
  y: number;
  width: number;
  height: number;
  scaleX: number;
  scaleY: number;
}): Rect {
  return {
    x: node.x,
    y: node.y,
    width: node.width * node.scaleX,
    height: node.height * node.scaleY,
  };
}

/** Move a box by an image-space delta, keeping its size and staying inside the image. */
export function translateBox(box: NormalizedBox, dx: number, dy: number): NormalizedBox {
  const [x, y, w, h] = box;
  return [clamp(x + dx, 0, 1 - w), clamp(y + dy, 0, 1 - h), w, h];
}

export function fitView(image: Size, viewport: Size, padding = 16): View {
  const availableW = Math.max(viewport.width - padding * 2, 1);
  const availableH = Math.max(viewport.height - padding * 2, 1);
  const scale = clamp(
    Math.min(availableW / image.width, availableH / image.height),
    MIN_SCALE,
    MAX_SCALE,
  );
  return {
    scale,
    offsetX: (viewport.width - image.width * scale) / 2,
    offsetY: (viewport.height - image.height * scale) / 2,
  };
}

/** Zoom by `factor` keeping the screen point `anchor` over the same image point. */
export function zoomAt(view: View, factor: number, anchor: Point): View {
  const scale = clamp(view.scale * factor, MIN_SCALE, MAX_SCALE);
  const imagePoint = toImagePoint(anchor, view);
  return {
    scale,
    offsetX: anchor.x - imagePoint.x * scale,
    offsetY: anchor.y - imagePoint.y * scale,
  };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}
