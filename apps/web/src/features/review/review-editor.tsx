"use client";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Field, inputClass } from "@/components/ui/field";
import { Notice } from "@/components/ui/states";
import type { CanvasHandle, Tool } from "@/features/annotations/detection-canvas";
import { newBoxKey, type DraftBox } from "@/features/annotations/draft";
import type { Prediction, ProjectDetail, ReviewAccept } from "@/lib/api/types";
import type { NormalizedBox } from "@/lib/geometry/boxes";
const Canvas = dynamic(() => import("@/features/annotations/detection-canvas"), { ssr: false });

export function ReviewEditor({
  prediction,
  project,
  onAccept,
  onRefresh,
  busy,
  error,
}: {
  prediction: Prediction;
  project: ProjectDetail;
  onAccept(body: ReviewAccept): void;
  onRefresh(): void;
  busy: boolean;
  error: string | null;
}) {
  const output = prediction.output;
  const [classId, setClassId] = useState(() =>
    "label" in output ? (project.classes.find((c) => c.name === output.label)?.id ?? "") : "",
  );
  const [boxes, setBoxes] = useState<DraftBox[]>(() =>
    "boxes" in prediction.output
      ? prediction.output.boxes.map((b) => ({
          key: newBoxKey(),
          classId: project.classes.find((c) => c.name === b.label)?.id ?? "",
          bbox: [b.x, b.y, Math.min(b.w, 1 - b.x), Math.min(b.h, 1 - b.y)],
        }))
      : [],
  );
  const [reviewed, setReviewed] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [imageError, setImageError] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [activeClass, setActiveClass] = useState(project.classes[0]?.id ?? "");
  const [tool, setTool] = useState<Tool>("select");
  const canvas = useRef<CanvasHandle>(null);
  const detection = project.task_type === "detection";
  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setSize({ width: img.naturalWidth, height: img.naturalHeight });
      setImageError(false);
    };
    img.onerror = () => setImageError(true);
    img.src = prediction.image_url;
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [prediction.image_url]);
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    const guard = (e: MouseEvent) => {
      const anchor = (e.target as Element).closest?.("a[href]");
      if (anchor && (busy || !window.confirm("Discard unaccepted review corrections?"))) {
        e.preventDefault();
        e.stopPropagation();
      }
    };
    document.addEventListener("click", guard, true);
    window.addEventListener("beforeunload", warn);
    return () => {
      window.removeEventListener("beforeunload", warn);
      document.removeEventListener("click", guard, true);
    };
  }, [dirty, busy]);
  function updateBoxes(next: DraftBox[]) {
    setBoxes(next);
    setDirty(true);
    setReviewed(false);
  }
  const validBoxes = boxes.every(
    (b) =>
      b.classId &&
      b.bbox.every((n) => Number.isFinite(n) && n >= 0 && n <= 1) &&
      b.bbox[2] > 0 &&
      b.bbox[3] > 0 &&
      b.bbox[0] + b.bbox[2] <= 1.000000001 &&
      b.bbox[1] + b.bbox[3] <= 1.000000001,
  );
  return (
    <fieldset disabled={busy} className="flex flex-col gap-4">
      <p className="text-sm text-muted">
        Model v{prediction.model_version} · {(prediction.max_confidence * 100).toFixed(1)}% maximum
        confidence. Correct the suggestion, then accept it as a human label.
      </p>
      {imageError ? (
        <Notice tone="error">
          The preview is unavailable or expired.{" "}
          <Button onClick={onRefresh}>Refresh preview</Button>
        </Notice>
      ) : detection ? (
        <>
          <div className="flex flex-wrap gap-2">
            <Field label="Class for new review boxes">
              {(p) => (
                <select
                  {...p}
                  className={inputClass}
                  value={activeClass}
                  onChange={(e) => setActiveClass(e.target.value)}
                >
                  {project.classes.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            {(["draw", "select", "pan"] as Tool[]).map((t) => (
              <Button key={t} aria-pressed={tool === t} onClick={() => setTool(t)}>
                {t === "draw" ? "Draw box" : t === "select" ? "Select / move" : "Pan"}
              </Button>
            ))}
            <Button onClick={() => canvas.current?.fit()}>Fit image</Button>
            <Button onClick={() => canvas.current?.zoom(1.25)}>Zoom in</Button>
            <Button onClick={() => canvas.current?.zoom(0.8)}>Zoom out</Button>
          </div>
          <div
            className="h-[440px] overflow-hidden rounded bg-slate-900"
            data-testid="review-canvas"
            style={{ pointerEvents: busy ? "none" : undefined }}
          >
            {size && (
              <Canvas
                ref={canvas}
                imageUrl={prediction.image_url}
                imageSize={size}
                boxes={boxes}
                colors={Object.fromEntries(project.classes.map((c) => [c.id, c.color]))}
                selectedKey={selected}
                activeClassId={activeClass}
                tool={tool}
                onSelect={setSelected}
                onBoxesChange={updateBoxes}
                onImageError={() => setImageError(true)}
              />
            )}
          </div>
          <Button
            onClick={() =>
              updateBoxes([
                ...boxes,
                { key: newBoxKey(), classId: activeClass, bbox: [0.1, 0.1, 0.2, 0.2] },
              ])
            }
          >
            Add box with coordinates
          </Button>
          <p className="text-xs text-muted">
            Coordinates are normalized top-left x, y, width and height. All boxes must fit inside
            the image. Remove all boxes to mark a reviewed negative.
          </p>
          {boxes.map((box, index) => (
            <fieldset disabled={busy} key={box.key} className="rounded border border-line p-3">
              <legend>Review box {index + 1}</legend>
              <div className="grid gap-2 sm:grid-cols-6">
                <select
                  aria-label={`Class for review box ${index + 1}`}
                  className={inputClass}
                  value={box.classId}
                  onChange={(e) =>
                    updateBoxes(
                      boxes.map((b) => (b.key === box.key ? { ...b, classId: e.target.value } : b)),
                    )
                  }
                >
                  <option value="">Choose class</option>
                  {project.classes.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name}
                    </option>
                  ))}
                </select>
                {["x", "y", "width", "height"].map((name, coordinate) => (
                  <label key={name} className="text-xs">
                    {name}
                    <input
                      aria-label={`Box ${index + 1} ${name}`}
                      className={inputClass}
                      type="number"
                      min="0"
                      max="1"
                      step="any"
                      value={box.bbox[coordinate]}
                      onChange={(e) =>
                        updateBoxes(
                          boxes.map((b) =>
                            b.key === box.key
                              ? {
                                  ...b,
                                  bbox: b.bbox.map((n, i) =>
                                    i === coordinate ? Number(e.target.value) : n,
                                  ) as NormalizedBox,
                                }
                              : b,
                          ),
                        )
                      }
                    />
                  </label>
                ))}
                <Button
                  variant="danger"
                  onClick={() => updateBoxes(boxes.filter((b) => b.key !== box.key))}
                >
                  Remove box {index + 1}
                </Button>
              </div>
            </fieldset>
          ))}
          {!validBoxes && (
            <Notice tone="error">
              Choose a class for every box and use positive sizes that fit inside the image.
            </Notice>
          )}
        </>
      ) : (
        <>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={prediction.image_url}
            alt="Image awaiting human review"
            referrerPolicy="no-referrer"
            className="max-h-96 w-full rounded bg-slate-100 object-contain"
          />
          <Field label="Correct class">
            {(p) => (
              <select
                {...p}
                className={inputClass}
                value={classId}
                onChange={(e) => {
                  setClassId(e.target.value);
                  setDirty(true);
                  setReviewed(false);
                }}
              >
                <option value="">Choose class</option>
                {project.classes.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name}
                  </option>
                ))}
              </select>
            )}
          </Field>
        </>
      )}
      <label className="flex gap-2 text-sm">
        <input
          type="checkbox"
          checked={reviewed}
          disabled={busy || !size || imageError}
          onChange={(e) => setReviewed(e.target.checked)}
        />
        {detection && !boxes.length
          ? "I reviewed this image and confirmed there are no defects"
          : "I reviewed this image and corrected its labels"}
      </label>
      {error && (
        <Notice tone="error">
          {error} Your draft is retained. Retrying the same acceptance is safe.
        </Notice>
      )}
      <Button
        variant="primary"
        busy={busy}
        disabled={!reviewed || !size || imageError || (detection ? !validBoxes : !classId)}
        onClick={() => {
          onAccept({
            annotations: detection
              ? boxes.map((b) => ({ class_id: b.classId, bbox: b.bbox }))
              : [{ class_id: classId }],
          });
        }}
      >
        Accept reviewed image
      </Button>
      <p className="text-xs text-muted">
        Acceptance adds this image and these labels to the live dataset. Create a new dataset
        version before retraining; existing versions stay unchanged.
      </p>
    </fieldset>
  );
}
