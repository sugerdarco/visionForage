import type { NormalizedBox } from "@/lib/geometry/boxes";
import type { AnnotationSave, ImageAnnotations } from "@/lib/api/types";

export interface DraftBox {
  key: string;
  classId: string;
  bbox: NormalizedBox;
}

/** Editor-owned label state for one image. Server data seeds it once and never overwrites it. */
export interface Draft {
  /** Classification: the chosen class, or null when unlabeled. */
  classId: string | null;
  /** Detection boxes in normalized top-left coordinates. */
  boxes: DraftBox[];
  /** Detection only: an empty set that a person reviewed and found clean. */
  reviewedNegative: boolean;
}

let keyCounter = 0;
export function newBoxKey(): string {
  return `box-${++keyCounter}`;
}

export function draftFromServer(data: ImageAnnotations): Draft {
  const boxes = data.annotations
    .filter((a) => a.bbox)
    .map((a) => ({ key: newBoxKey(), classId: a.class_id, bbox: [...a.bbox!] as NormalizedBox }));
  const tag = data.annotations.find((a) => !a.bbox);
  return {
    classId: tag?.class_id ?? null,
    boxes,
    reviewedNegative: data.status === "labeled" && data.annotations.length === 0,
  };
}

/** The complete set to PUT. IDs, source, confidence and image IDs are never sent. */
export function classificationBody(draft: Draft): AnnotationSave {
  return { annotations: draft.classId ? [{ class_id: draft.classId }] : [] };
}

export function detectionBody(draft: Draft): AnnotationSave {
  if (draft.boxes.length > 0) {
    return { annotations: draft.boxes.map((b) => ({ class_id: b.classId, bbox: b.bbox })) };
  }
  return draft.reviewedNegative ? { annotations: [], status: "labeled" } : { annotations: [] };
}

export interface History {
  past: Draft[];
  present: Draft;
  future: Draft[];
  /** Increments on every change, including undo/redo; used to order saves. */
  revision: number;
}

const HISTORY_LIMIT = 200;

export type HistoryAction =
  | { type: "reset"; draft: Draft }
  | { type: "change"; draft: Draft }
  | { type: "undo" }
  | { type: "redo" };

export function historyReducer(state: History, action: HistoryAction): History {
  switch (action.type) {
    case "reset":
      return { past: [], present: action.draft, future: [], revision: state.revision };
    case "change":
      if (action.draft === state.present) return state;
      return {
        past: [...state.past, state.present].slice(-HISTORY_LIMIT),
        present: action.draft,
        future: [],
        revision: state.revision + 1,
      };
    case "undo": {
      const previous = state.past.at(-1);
      if (!previous) return state;
      return {
        past: state.past.slice(0, -1),
        present: previous,
        future: [state.present, ...state.future],
        revision: state.revision + 1,
      };
    }
    case "redo": {
      const [next, ...rest] = state.future;
      if (!next) return state;
      return {
        past: [...state.past, state.present],
        present: next,
        future: rest,
        revision: state.revision + 1,
      };
    }
  }
}

export const EMPTY_DRAFT: Draft = { classId: null, boxes: [], reviewedNegative: false };

/** True when keyboard shortcuts must not fire because the user is typing or using a control. */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}
