"use client";

import { useState } from "react";
import { Button, Spinner } from "@/components/ui/button";
import type { Image, ImageAnnotations, ProjectDetail } from "@/lib/api/types";
import { draftFromServer } from "./draft";
import type { SaveSnapshot } from "./save-queue";

export interface Navigator {
  /** Leave to `href` once pending edits are saved, or after the user confirms discarding them. */
  leave(href: string): void;
  goPrevious: (() => void) | null;
  goNext: () => void;
  position: string;
}

export interface EditorProps {
  project: ProjectDetail;
  image: Image;
  annotations: ImageAnnotations;
  nav: Navigator;
  onImageExpired(): void;
  /** Editors register their flush so navigation can wait for the latest save. */
  registerGuard(guard: Guard | null): void;
}

export interface Guard {
  flush(): Promise<void>;
  isDirty(): boolean;
}

export function SaveIndicator({
  save,
  dirty,
  onRetry,
}: {
  save: SaveSnapshot;
  dirty: boolean;
  onRetry: () => void;
}) {
  if (save.state === "error") {
    return (
      <div role="alert" className="flex items-center gap-2 text-sm text-danger">
        Not saved: {save.error?.message}
        <Button size="sm" onClick={onRetry}>
          Retry save
        </Button>
      </div>
    );
  }
  if (save.state === "saving")
    return (
      <span role="status" className="flex items-center gap-2 text-sm text-muted">
        <Spinner /> Saving…
      </span>
    );
  if (dirty)
    return (
      <span role="status" className="text-sm text-warn">
        Unsaved changes
      </span>
    );
  return (
    <span role="status" className="text-sm text-ok">
      All changes saved
    </span>
  );
}

export function useImageAnnotationsSeed(annotations: ImageAnnotations) {
  // Seed once per mount; later server responses never replace the editor draft.
  const [seed] = useState(() => draftFromServer(annotations));
  return seed;
}
