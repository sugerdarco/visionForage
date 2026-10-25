"use client";

import type Konva from "konva";
import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Group, Image as KonvaImage, Layer, Rect, Stage, Transformer } from "react-konva";
import {
  commitTransform,
  fitView,
  imageToNormalized,
  normalizedToImage,
  rectFromPoints,
  toImagePoint,
  translateBox,
  zoomAt,
  type Point,
  type Size,
  type View,
} from "@/lib/geometry/boxes";
import { newBoxKey, type DraftBox } from "./draft";

export type Tool = "draw" | "select" | "pan";

export interface CanvasHandle {
  fit(): void;
  zoom(factor: number): void;
}

interface Props {
  imageUrl: string;
  imageSize: Size;
  boxes: DraftBox[];
  colors: Record<string, string>;
  selectedKey: string | null;
  activeClassId: string | null;
  tool: Tool;
  onSelect(key: string | null): void;
  onBoxesChange(boxes: DraftBox[]): void;
  onImageError(): void;
  onZoomChange?(scale: number): void;
}

/**
 * Konva view of one image. Boxes are drawn inside a Group whose transform is the
 * current zoom/pan, so their geometry is in original-image pixels; every commit
 * converts back to normalized API coordinates.
 */
const DetectionCanvas = forwardRef<CanvasHandle, Props>(function DetectionCanvas(props, ref) {
  const {
    imageUrl,
    imageSize,
    boxes,
    colors,
    selectedKey,
    activeClassId,
    tool,
    onSelect,
    onBoxesChange,
  } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const rectRefs = useRef(new Map<string, Konva.Rect>());
  const [viewport, setViewport] = useState<Size>({ width: 0, height: 0 });
  const [view, setView] = useState<View | null>(null);
  const [element, setElement] = useState<HTMLImageElement | null>(null);
  const [drawing, setDrawing] = useState<{ start: Point; current: Point } | null>(null);
  const panning = useRef<{ start: Point; origin: View } | null>(null);
  const [spaceHeld, setSpaceHeld] = useState(false);

  // Load the original once per URL; a failure usually means the signed URL expired.
  const onImageError = props.onImageError;
  useEffect(() => {
    const img = new window.Image();
    img.onload = () => setElement(img);
    img.onerror = () => onImageError();
    img.src = imageUrl;
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl, onImageError]);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      setViewport({
        width: Math.floor(entry.contentRect.width),
        height: Math.floor(entry.contentRect.height),
      });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const fit = useCallback(() => {
    if (viewport.width && viewport.height) setView(fitView(imageSize, viewport));
  }, [imageSize, viewport]);

  // Fit on first layout; later container resizes keep the user's zoom but stay valid.
  useEffect(() => {
    if (!view && viewport.width && viewport.height) fit();
  }, [view, viewport, fit]);

  const onZoomChange = props.onZoomChange;
  useEffect(() => {
    if (view) onZoomChange?.(view.scale);
  }, [view, onZoomChange]);

  useImperativeHandle(
    ref,
    () => ({
      fit,
      zoom: (factor: number) =>
        setView((v) =>
          v ? zoomAt(v, factor, { x: viewport.width / 2, y: viewport.height / 2 }) : v,
        ),
    }),
    [fit, viewport],
  );

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !(e.target instanceof HTMLInputElement)) setSpaceHeld(true);
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") setSpaceHeld(false);
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  useEffect(() => {
    const transformer = transformerRef.current;
    if (!transformer) return;
    const node = selectedKey && tool === "select" ? rectRefs.current.get(selectedKey) : undefined;
    transformer.nodes(node ? [node] : []);
    transformer.getLayer()?.batchDraw();
  }, [selectedKey, boxes, tool]);

  if (!view) {
    return <div ref={containerRef} className="h-full w-full" />;
  }

  const pointer = (stage: Konva.Stage | null): Point | null => stage?.getPointerPosition() ?? null;
  const effectiveTool: Tool = spaceHeld ? "pan" : tool;

  function handleMouseDown(e: Konva.KonvaEventObject<MouseEvent>) {
    const stage = e.target.getStage();
    const position = pointer(stage);
    if (!position || !view) return;
    if (effectiveTool === "pan" || e.evt.button === 1) {
      panning.current = { start: position, origin: view };
      return;
    }
    const onBox = e.target.name() === "box" || e.target.getParent()?.className === "Transformer";
    if (effectiveTool === "draw" && !onBox && e.evt.button === 0) {
      const point = clampPoint(toImagePoint(position, view));
      setDrawing({ start: point, current: point });
      onSelect(null);
      return;
    }
    if (!onBox) onSelect(null);
  }

  function handleMouseMove(e: Konva.KonvaEventObject<MouseEvent>) {
    const position = pointer(e.target.getStage());
    if (!position || !view) return;
    if (panning.current) {
      const { start, origin } = panning.current;
      setView({
        ...origin,
        offsetX: origin.offsetX + position.x - start.x,
        offsetY: origin.offsetY + position.y - start.y,
      });
      return;
    }
    if (drawing) setDrawing({ ...drawing, current: clampPoint(toImagePoint(position, view)) });
  }

  function handleMouseUp() {
    panning.current = null;
    if (!drawing) return;
    setDrawing(null);
    if (!activeClassId) return;
    const bbox = imageToNormalized(rectFromPoints(drawing.start, drawing.current), imageSize);
    // A click without a drag (or a sliver) is not a box.
    if (!bbox || bbox[2] * imageSize.width < 3 || bbox[3] * imageSize.height < 3) return;
    const box: DraftBox = { key: newBoxKey(), classId: activeClassId, bbox };
    onBoxesChange([...boxes, box]);
    onSelect(box.key);
  }

  function clampPoint(point: Point): Point {
    return {
      x: Math.min(Math.max(point.x, 0), imageSize.width),
      y: Math.min(Math.max(point.y, 0), imageSize.height),
    };
  }

  function handleWheel(e: Konva.KonvaEventObject<WheelEvent>) {
    e.evt.preventDefault();
    const position = pointer(e.target.getStage());
    if (!position || !view) return;
    setView(zoomAt(view, e.evt.deltaY < 0 ? 1.15 : 1 / 1.15, position));
  }

  const preview = drawing ? rectFromPoints(drawing.start, drawing.current) : null;
  const cursor =
    effectiveTool === "pan"
      ? panning.current
        ? "grabbing"
        : "grab"
      : effectiveTool === "draw"
        ? "crosshair"
        : "default";

  return (
    <div
      ref={containerRef}
      className="h-full w-full"
      style={{ cursor }}
      data-testid="detection-stage"
    >
      <Stage
        width={viewport.width}
        height={viewport.height}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <Layer>
          <Group x={view.offsetX} y={view.offsetY} scaleX={view.scale} scaleY={view.scale}>
            {element && (
              <KonvaImage
                image={element}
                width={imageSize.width}
                height={imageSize.height}
                listening={false}
              />
            )}
            {boxes.map((box) => {
              const rect = normalizedToImage(box.bbox, imageSize);
              const color = colors[box.classId] ?? "#f97316";
              const selected = box.key === selectedKey;
              return (
                <Rect
                  key={box.key}
                  name="box"
                  ref={(node) => {
                    if (node) rectRefs.current.set(box.key, node);
                    else rectRefs.current.delete(box.key);
                  }}
                  x={rect.x}
                  y={rect.y}
                  width={rect.width}
                  height={rect.height}
                  stroke={color}
                  strokeWidth={selected ? 3 : 2}
                  strokeScaleEnabled={false}
                  fill={selected ? `${color}33` : `${color}14`}
                  dash={selected ? undefined : [6, 3]}
                  draggable={tool === "select" && !spaceHeld}
                  onMouseDown={(e) => {
                    if (tool === "select") {
                      e.cancelBubble = true;
                      onSelect(box.key);
                    }
                  }}
                  onDragEnd={(e) => {
                    const node = e.target;
                    const dx = node.x() / imageSize.width - box.bbox[0];
                    const dy = node.y() / imageSize.height - box.bbox[1];
                    const moved = translateBox(box.bbox, dx, dy);
                    const back = normalizedToImage(moved, imageSize);
                    node.position({ x: back.x, y: back.y });
                    onBoxesChange(
                      boxes.map((b) => (b.key === box.key ? { ...b, bbox: moved } : b)),
                    );
                  }}
                  onTransformEnd={(e) => {
                    const node = e.target as Konva.Rect;
                    const committed = commitTransform({
                      x: node.x(),
                      y: node.y(),
                      width: node.width(),
                      height: node.height(),
                      scaleX: node.scaleX(),
                      scaleY: node.scaleY(),
                    });
                    node.scale({ x: 1, y: 1 });
                    const bbox = imageToNormalized(committed, imageSize);
                    if (!bbox) {
                      const restore = normalizedToImage(box.bbox, imageSize);
                      node.setAttrs(restore);
                      return;
                    }
                    onBoxesChange(boxes.map((b) => (b.key === box.key ? { ...b, bbox } : b)));
                  }}
                />
              );
            })}
            {preview && (
              <Rect
                x={preview.x}
                y={preview.y}
                width={preview.width}
                height={preview.height}
                stroke={(activeClassId && colors[activeClassId]) || "#f97316"}
                strokeWidth={2}
                strokeScaleEnabled={false}
                dash={[4, 4]}
                listening={false}
              />
            )}
          </Group>
          <Transformer
            ref={transformerRef}
            rotateEnabled={false}
            keepRatio={false}
            ignoreStroke
            flipEnabled={false}
            anchorSize={9}
            boundBoxFunc={(oldBox, newBox) =>
              newBox.width < 3 || newBox.height < 3 ? oldBox : newBox
            }
          />
        </Layer>
      </Stage>
    </div>
  );
});

export default DetectionCanvas;
