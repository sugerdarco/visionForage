"use client";

import dynamic from "next/dynamic";
import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { inputClass } from "@/components/ui/field";
import { normalizedToImage, translateBox } from "@/lib/geometry/boxes";
import type { CanvasHandle, Tool } from "./detection-canvas";
import { detectionBody, isTypingTarget, type DraftBox } from "./draft";
import { SaveIndicator, useImageAnnotationsSeed, type EditorProps } from "./editor-common";
import { useDraftSaver } from "./use-draft-saver";

// Konva needs the browser canvas; keep it out of server rendering.
const DetectionCanvas = dynamic(() => import("./detection-canvas"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-sm text-slate-300">
      Loading editor…
    </div>
  ),
});

const TOOLS: { tool: Tool; label: string; key: string }[] = [
  { tool: "draw", label: "Draw", key: "B" },
  { tool: "select", label: "Select / move", key: "V" },
  { tool: "pan", label: "Pan", key: "H" },
];

export function DetectionEditor({
  project,
  image,
  annotations,
  nav,
  onImageExpired,
  registerGuard,
}: EditorProps) {
  const seed = useImageAnnotationsSeed(annotations);
  const saver = useDraftSaver({
    projectId: project.id,
    imageId: image.id,
    initial: seed,
    toBody: detectionBody,
    debounceMs: 700,
  });
  const { draft, change, undo, redo, flush, isDirtyNow } = saver;
  const canvas = useRef<CanvasHandle>(null);
  const [tool, setTool] = useState<Tool>("draw");
  const [selected, setSelected] = useState<string | null>(null);
  const [activeClassId, setActiveClassId] = useState<string | null>(project.classes[0]?.id ?? null);
  const [zoom, setZoom] = useState(1);
  const size = { width: image.width, height: image.height };
  const colors = Object.fromEntries(project.classes.map((c) => [c.id, c.color]));
  const names = Object.fromEntries(project.classes.map((c) => [c.id, c.name]));

  useEffect(() => {
    registerGuard({ flush, isDirty: isDirtyNow });
    return () => registerGuard(null);
  }, [registerGuard, flush, isDirtyNow]);

  const setBoxes = useCallback(
    (boxes: DraftBox[]) =>
      // Drawing a box means the image is no longer a reviewed negative.
      change({
        ...draft,
        boxes,
        reviewedNegative: boxes.length > 0 ? false : draft.reviewedNegative,
      }),
    [change, draft],
  );

  const removeBox = useCallback(
    (key: string) => {
      setBoxes(draft.boxes.filter((b) => b.key !== key));
      setSelected(null);
    },
    [draft.boxes, setBoxes],
  );

  const reclassify = useCallback(
    (key: string, classId: string) =>
      setBoxes(draft.boxes.map((b) => (b.key === key ? { ...b, classId } : b))),
    [draft.boxes, setBoxes],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
        return;
      }
      if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
        return;
      }
      if (mod && e.key.toLowerCase() === "s") {
        e.preventDefault();
        void flush();
        return;
      }
      if (mod || e.altKey) return;
      const box = draft.boxes.find((b) => b.key === selected);
      if (box && e.key.startsWith("Arrow")) {
        e.preventDefault();
        const step = e.shiftKey ? 10 : 1;
        const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
        const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
        const moved = translateBox(box.bbox, dx / image.width, dy / image.height);
        setBoxes(draft.boxes.map((b) => (b.key === box.key ? { ...b, bbox: moved } : b)));
        return;
      }
      if (/^[1-9]$/.test(e.key)) {
        const target = project.classes[Number(e.key) - 1];
        if (!target) return;
        setActiveClassId(target.id);
        if (box) reclassify(box.key, target.id);
        return;
      }
      switch (e.key) {
        case "Delete":
        case "Backspace":
          if (box) {
            e.preventDefault();
            removeBox(box.key);
          }
          break;
        case "Escape":
          setSelected(null);
          break;
        case "Tab":
          if (draft.boxes.length && document.activeElement === document.body) {
            e.preventDefault();
            const index = draft.boxes.findIndex((b) => b.key === selected);
            const next =
              draft.boxes[
                (index + (e.shiftKey ? -1 : 1) + draft.boxes.length) % draft.boxes.length
              ];
            setSelected(next?.key ?? null);
          }
          break;
        case "b":
        case "B":
          setTool("draw");
          break;
        case "v":
        case "V":
          setTool("select");
          break;
        case "h":
        case "H":
          setTool("pan");
          break;
        case "f":
        case "F":
          canvas.current?.fit();
          break;
        case "+":
        case "=":
          canvas.current?.zoom(1.25);
          break;
        case "-":
          canvas.current?.zoom(0.8);
          break;
        case "]":
          nav.goNext();
          break;
        case "[":
          nav.goPrevious?.();
          break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [
    draft,
    selected,
    project.classes,
    image.width,
    image.height,
    setBoxes,
    removeBox,
    reclassify,
    undo,
    redo,
    flush,
    nav,
  ]);

  const status =
    draft.boxes.length > 0
      ? `${draft.boxes.length} box${draft.boxes.length === 1 ? "" : "es"}`
      : draft.reviewedNegative
        ? "Reviewed: no defects"
        : "Unlabeled";

  return (
    <div className="flex h-full flex-col lg:flex-row">
      <div className="flex min-h-[360px] min-w-0 flex-1 flex-col">
        <div
          role="toolbar"
          aria-label="Editor tools"
          className="flex flex-wrap items-center gap-2 border-b border-slate-700 bg-slate-900 px-3 py-2 text-white"
        >
          {TOOLS.map((t) => (
            <Button
              key={t.tool}
              size="sm"
              variant={tool === t.tool ? "primary" : "ghost"}
              className={tool === t.tool ? "" : "text-white hover:bg-slate-700"}
              aria-pressed={tool === t.tool}
              onClick={() => setTool(t.tool)}
              title={`${t.label} (${t.key})`}
            >
              {t.label}
            </Button>
          ))}
          <span className="mx-1 h-5 w-px bg-slate-600" />
          <Button
            size="sm"
            variant="ghost"
            className="text-white hover:bg-slate-700"
            onClick={() => canvas.current?.zoom(0.8)}
            aria-label="Zoom out"
            title="Zoom out (-)"
          >
            −
          </Button>
          <span className="w-14 text-center text-xs tabular-nums" aria-live="polite">
            {Math.round(zoom * 100)}%
          </span>
          <Button
            size="sm"
            variant="ghost"
            className="text-white hover:bg-slate-700"
            onClick={() => canvas.current?.zoom(1.25)}
            aria-label="Zoom in"
            title="Zoom in (+)"
          >
            +
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="text-white hover:bg-slate-700"
            onClick={() => canvas.current?.fit()}
            title="Fit to view (F)"
          >
            Fit
          </Button>
          <span className="mx-1 h-5 w-px bg-slate-600" />
          <Button
            size="sm"
            variant="ghost"
            className="text-white hover:bg-slate-700"
            disabled={!saver.canUndo}
            onClick={undo}
            title="Undo (Ctrl+Z)"
          >
            Undo
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="text-white hover:bg-slate-700"
            disabled={!saver.canRedo}
            onClick={redo}
            title="Redo (Ctrl+Shift+Z)"
          >
            Redo
          </Button>
          <span className="ml-auto text-xs text-slate-300">Hold Space to pan · wheel to zoom</span>
        </div>
        <div className="min-h-0 flex-1 bg-slate-800">
          <DetectionCanvas
            ref={canvas}
            imageUrl={image.image_url}
            imageSize={size}
            boxes={draft.boxes}
            colors={colors}
            selectedKey={selected}
            activeClassId={activeClassId}
            tool={tool}
            onSelect={setSelected}
            onBoxesChange={setBoxes}
            onImageError={onImageExpired}
            onZoomChange={setZoom}
          />
        </div>
      </div>
      <aside className="flex w-full flex-col gap-4 overflow-y-auto border-l border-line bg-panel p-4 lg:w-80">
        <div>
          <label className="flex flex-col gap-1 text-sm">
            <span className="font-semibold">Class for new boxes</span>
            <select
              className={inputClass}
              value={activeClassId ?? ""}
              onChange={(e) => setActiveClassId(e.target.value || null)}
            >
              {project.classes.map((c, i) => (
                <option key={c.id} value={c.id}>
                  {i < 9 ? `${i + 1}. ` : ""}
                  {c.name}
                </option>
              ))}
            </select>
          </label>
          {project.classes.length === 0 && (
            <p className="mt-1 text-sm text-danger">
              Add classes in Settings before drawing boxes.
            </p>
          )}
        </div>
        <div className="flex items-center justify-between">
          <h2 className="font-semibold">Boxes</h2>
          <span className="text-sm text-muted" data-testid="label-summary">
            {status}
          </span>
        </div>
        {draft.boxes.length === 0 ? (
          <div className="flex flex-col gap-2 rounded-md border border-dashed border-line p-3 text-sm">
            <p className="text-muted">
              Drag on the image to draw a box. If you checked this image and it has no defects, mark
              it as reviewed so it is included in datasets as a clean example.
            </p>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={draft.reviewedNegative}
                onChange={(e) => change({ ...draft, reviewedNegative: e.target.checked })}
              />
              Reviewed — no defects in this image
            </label>
          </div>
        ) : (
          <ul aria-label="Boxes" className="flex flex-col gap-2">
            {draft.boxes.map((box, index) => {
              const px = normalizedToImage(box.bbox, size);
              const isSelected = box.key === selected;
              return (
                <li
                  key={box.key}
                  className={`rounded-md border p-2 text-sm ${isSelected ? "border-accent bg-blue-50" : "border-line"}`}
                  data-testid="box-row"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="h-3 w-3 shrink-0 rounded-sm"
                      style={{ background: colors[box.classId] ?? "#f97316" }}
                    />
                    <button
                      type="button"
                      className="flex-1 text-left font-medium"
                      aria-pressed={isSelected}
                      onClick={() => {
                        setSelected(isSelected ? null : box.key);
                        setTool("select");
                      }}
                    >
                      Box {index + 1}: {names[box.classId] ?? "Unknown class"}
                    </button>
                    <Button
                      size="sm"
                      variant="ghost"
                      aria-label={`Delete box ${index + 1}`}
                      onClick={() => removeBox(box.key)}
                    >
                      Delete
                    </Button>
                  </div>
                  <p className="mt-1 font-mono text-xs text-muted" data-testid="box-pixels">
                    x {px.x.toFixed(1)}, y {px.y.toFixed(1)}, {px.width.toFixed(1)} ×{" "}
                    {px.height.toFixed(1)} px
                  </p>
                  {isSelected && (
                    <label className="mt-2 flex items-center gap-2 text-xs">
                      Class
                      <select
                        className={`${inputClass} py-1`}
                        value={box.classId}
                        onChange={(e) => reclassify(box.key, e.target.value)}
                      >
                        {project.classes.map((c) => (
                          <option key={c.id} value={c.id}>
                            {c.name}
                          </option>
                        ))}
                      </select>
                    </label>
                  )}
                </li>
              );
            })}
          </ul>
        )}
        <details className="text-xs text-muted">
          <summary className="cursor-pointer">Keyboard shortcuts</summary>
          <ul className="mt-2 list-disc pl-4">
            <li>B draw · V select · H pan (or hold Space) · F fit · +/− zoom</li>
            <li>1–9 choose class (re-classifies the selected box)</li>
            <li>Tab cycles boxes · arrows move the selected box (Shift = 10 px)</li>
            <li>Delete removes the selected box · Esc deselects</li>
            <li>Ctrl+Z undo · Ctrl+Shift+Z redo · Ctrl+S save now</li>
            <li>[ previous image · ] next image</li>
          </ul>
        </details>
        <div className="mt-auto flex flex-col gap-2">
          <SaveIndicator save={saver.save} dirty={saver.dirty} onRetry={() => void saver.retry()} />
          <p className="text-xs text-muted">
            Changes save automatically. Saves replace the whole label set for this image; editing
            the same image in two tabs at once is not supported.
          </p>
          <Button variant="primary" onClick={nav.goNext}>
            Next image →
          </Button>
        </div>
      </aside>
    </div>
  );
}
