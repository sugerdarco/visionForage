"use client";

import { useQueryClient } from "@tanstack/react-query";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useSyncExternalStore,
} from "react";
import { api } from "@/lib/api/endpoints";
import type { AnnotationSave, ImageAnnotations } from "@/lib/api/types";
import { historyReducer, type Draft } from "./draft";
import { AnnotationSaveQueue } from "./save-queue";

/**
 * Draft history plus serialized autosave for one image. Server responses update
 * the cache but never replace the draft, so unsaved edits are never overwritten.
 */
export function useDraftSaver(options: {
  projectId: string;
  imageId: string;
  initial: Draft;
  toBody: (draft: Draft) => AnnotationSave;
  debounceMs: number;
}) {
  const { projectId, imageId, initial, toBody, debounceMs } = options;
  const queryClient = useQueryClient();
  const [history, dispatch] = useReducer(historyReducer, {
    past: [],
    present: initial,
    future: [],
    revision: 0,
  });
  const queue = useMemo(
    () =>
      new AnnotationSaveQueue(async (body) => {
        const saved: ImageAnnotations = await api.images.saveAnnotations(imageId, body);
        queryClient.setQueryData(["annotations", imageId], saved);
        void queryClient.invalidateQueries({ queryKey: ["images", projectId] });
        void queryClient.invalidateQueries({ queryKey: ["image", imageId] });
      }),
    [imageId, projectId, queryClient],
  );
  const snapshot = useSyncExternalStore(
    useCallback((listener) => queue.subscribe(listener), [queue]),
    () => queue.getSnapshot(),
    () => queue.getSnapshot(),
  );

  const latest = useRef(history);
  useLayoutEffect(() => {
    latest.current = history;
  });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flush = useCallback(() => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = null;
    const { revision, present } = latest.current;
    if (revision > queue.getSnapshot().savedRevision) void queue.save(revision, toBody(present));
    return queue.idle();
  }, [queue, toBody]);

  useEffect(() => {
    if (history.revision === 0) return;
    queue.markDirty(history.revision);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      timer.current = null;
      void queue.save(history.revision, toBody(history.present));
    }, debounceMs);
  }, [history.revision, history.present, queue, toBody, debounceMs]);

  // Save any debounced edit when the editor unmounts (e.g. navigation away).
  useEffect(() => () => void flush(), [flush]);

  const dirty = history.revision > snapshot.savedRevision;

  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const change = useCallback((draft: Draft) => dispatch({ type: "change", draft }), []);
  const undo = useCallback(() => dispatch({ type: "undo" }), []);
  const redo = useCallback(() => dispatch({ type: "redo" }), []);
  const retry = useCallback(() => queue.retry(), [queue]);
  const isDirtyNow = useCallback(
    () => latest.current.revision > queue.getSnapshot().savedRevision,
    [queue],
  );

  return {
    draft: history.present,
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
    change,
    undo,
    redo,
    save: snapshot,
    dirty,
    flush,
    retry,
    isDirtyNow,
  };
}

export type DraftSaver = ReturnType<typeof useDraftSaver>;
