"use client";

import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { classificationBody, isTypingTarget } from "./draft";
import { SaveIndicator, useImageAnnotationsSeed, type EditorProps } from "./editor-common";
import { useDraftSaver } from "./use-draft-saver";

export function ClassificationEditor({
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
    toBody: classificationBody,
    debounceMs: 0,
  });
  const { draft, change } = saver;
  const { flush, isDirtyNow } = saver;

  useEffect(() => {
    registerGuard({ flush, isDirty: isDirtyNow });
    return () => registerGuard(null);
  }, [registerGuard, flush, isDirtyNow]);

  const classes = project.classes;
  const choose = (classId: string | null) => change({ ...draft, classId });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTypingTarget(e.target) || e.altKey || e.ctrlKey || e.metaKey) return;
      if (/^[1-9]$/.test(e.key)) {
        const target = classes[Number(e.key) - 1];
        if (target) {
          e.preventDefault();
          change({ ...draft, classId: target.id });
        }
      } else if (e.key === "0" || e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        change({ ...draft, classId: null });
      } else if (e.key === "ArrowRight" || e.key === "]") {
        e.preventDefault();
        nav.goNext();
      } else if (e.key === "ArrowLeft" || e.key === "[") {
        e.preventDefault();
        nav.goPrevious?.();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [classes, draft, change, nav]);

  return (
    <div className="flex h-full flex-col lg:flex-row">
      <div className="flex min-h-0 flex-1 items-center justify-center bg-slate-800 p-4">
        {/* Original image at its natural aspect ratio. */}
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={image.image_url}
          alt={`Image ${image.id} to classify`}
          onError={onImageExpired}
          className="h-full w-full object-contain"
          style={{ imageRendering: image.width < 256 ? "pixelated" : undefined }}
        />
      </div>
      <aside className="flex w-full flex-col gap-4 border-l border-line bg-panel p-4 lg:w-80">
        <div>
          <h2 className="font-semibold">Class</h2>
          <p className="text-xs text-muted">
            Pick exactly one. Number keys select, 0 clears, ←/→ move between images.
          </p>
        </div>
        {classes.length === 0 ? (
          <p className="text-sm text-muted">This project has no classes. Add them in Settings.</p>
        ) : (
          <div role="radiogroup" aria-label="Image class" className="flex flex-col gap-2">
            {classes.map((c, index) => {
              const selected = draft.classId === c.id;
              return (
                <button
                  key={c.id}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  onClick={() => choose(c.id)}
                  className={`flex items-center gap-3 rounded-md border px-3 py-2 text-left text-sm ${
                    selected
                      ? "border-ink bg-slate-900 text-white"
                      : "border-line bg-white hover:bg-slate-50"
                  }`}
                >
                  <span
                    className="h-4 w-4 shrink-0 rounded-sm border border-white/40"
                    style={{ background: c.color }}
                  />
                  <span className="flex-1 break-all">{c.name}</span>
                  {index < 9 && (
                    <kbd
                      className={`rounded border px-1.5 text-xs ${selected ? "border-white/40" : "border-line"}`}
                    >
                      {index + 1}
                    </kbd>
                  )}
                </button>
              );
            })}
          </div>
        )}
        <Button size="sm" variant="ghost" disabled={!draft.classId} onClick={() => choose(null)}>
          Clear label (mark unlabeled)
        </Button>
        <SaveIndicator save={saver.save} dirty={saver.dirty} onRetry={() => void saver.retry()} />
        <Button variant="primary" onClick={nav.goNext}>
          Next image →
        </Button>
      </aside>
    </div>
  );
}
