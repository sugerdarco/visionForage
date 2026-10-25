import { toApiError, type ApiError } from "@/lib/api/errors";
import type { AnnotationSave } from "@/lib/api/types";

export type SaveState = "saved" | "dirty" | "saving" | "error";

export interface SaveSnapshot {
  state: SaveState;
  error: ApiError | null;
  savedRevision: number;
  latestRevision: number;
}

type Send = (body: AnnotationSave) => Promise<unknown>;

/**
 * Serializes whole-set annotation saves for one image. The server is last-save-wins
 * with no revision check, so only one request is ever in flight and an older draft
 * is never sent after a newer one. Failures keep the image dirty; nothing retries
 * automatically, and a manual retry always sends the latest draft.
 */
export class AnnotationSaveQueue {
  private readonly send: Send;
  private inFlight: Promise<void> | null = null;
  private pending: { revision: number; body: AnnotationSave } | null = null;
  private latest: { revision: number; body: AnnotationSave } | null = null;
  private snapshot: SaveSnapshot = {
    state: "saved",
    error: null,
    savedRevision: 0,
    latestRevision: 0,
  };
  private readonly listeners = new Set<(snapshot: SaveSnapshot) => void>();

  constructor(send: Send, initialRevision = 0) {
    this.send = send;
    this.snapshot = {
      ...this.snapshot,
      savedRevision: initialRevision,
      latestRevision: initialRevision,
    };
  }

  subscribe(listener: (snapshot: SaveSnapshot) => void): () => void {
    this.listeners.add(listener);
    listener(this.snapshot);
    return () => this.listeners.delete(listener);
  }

  getSnapshot(): SaveSnapshot {
    return this.snapshot;
  }

  get isDirty(): boolean {
    return this.snapshot.latestRevision > this.snapshot.savedRevision;
  }

  /** Record a local edit without saving (e.g. while autosave is debounced). */
  markDirty(revision: number) {
    if (revision > this.snapshot.latestRevision) {
      this.update({ latestRevision: revision, state: this.inFlight ? "saving" : "dirty" });
    }
  }

  /** Queue the complete annotation set for `revision`. Resolves when it is no longer pending. */
  save(revision: number, body: AnnotationSave): Promise<void> {
    if (this.latest && revision < this.latest.revision) return this.idle();
    this.latest = { revision, body };
    this.pending = this.latest;
    if (revision > this.snapshot.latestRevision) this.update({ latestRevision: revision });
    if (!this.inFlight) this.inFlight = this.drain();
    return this.idle();
  }

  /** Resend the latest draft after a failure. */
  retry(): Promise<void> {
    if (!this.latest) return Promise.resolve();
    return this.save(this.latest.revision, this.latest.body);
  }

  idle(): Promise<void> {
    return this.inFlight ?? Promise.resolve();
  }

  private async drain() {
    while (this.pending) {
      const job = this.pending;
      this.pending = null;
      this.update({ state: "saving", error: null });
      try {
        await this.send(job.body);
        this.update({ savedRevision: Math.max(this.snapshot.savedRevision, job.revision) });
        if (!this.pending) {
          this.update({ state: this.isDirty ? "dirty" : "saved" });
        }
      } catch (raw) {
        // A newer queued draft supersedes this failure; otherwise surface it and stop.
        if (!this.pending) this.update({ state: "error", error: toApiError(raw) });
      }
    }
    this.inFlight = null;
  }

  private update(partial: Partial<SaveSnapshot>) {
    this.snapshot = { ...this.snapshot, ...partial };
    for (const listener of this.listeners) listener(this.snapshot);
  }
}
