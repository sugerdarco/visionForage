import type { ApiClient } from "@/lib/api/client";
import { toApiError } from "@/lib/api/errors";
import type { JobStatus } from "@/lib/api/types";
import { SseParser, type SseFrame } from "./sse-parser";

/** Public job state carried by `status` and `done` events. */
export interface StreamJobState {
  status: JobStatus;
  gpu_id?: number | null;
  cancel_requested_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  model_id?: string | null;
}

export type ConnectionState = "connecting" | "open" | "waiting" | "closed";

export type FatalReason = "signed_out" | "job_not_found";

export interface JobStreamHandlers {
  onStatus?(state: StreamJobState): void;
  onEpoch?(epoch: number, metrics: Record<string, number>): void;
  onLog?(line: string, ts: string | undefined): void;
  onDone?(state: StreamJobState): void;
  /** The server rejected the cursor; drop cached points before the full replay arrives. */
  onHistoryReset?(): void;
  onConnection?(state: ConnectionState, detail?: string): void;
  onFatal?(reason: FatalReason): void;
}

type StreamClient = Pick<ApiClient, "accessToken" | "refreshAccessToken" | "url">;

export interface JobStreamOptions {
  jobId: string;
  client: StreamClient;
  handlers: JobStreamHandlers;
  fetchImpl?: typeof fetch;
  sleep?: (ms: number, signal: AbortSignal) => Promise<void>;
  random?: () => number;
  maxBackoffMs?: number;
}

const DEFAULT_RETRY_MS = 2000;

function abortableSleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve();
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => (clearTimeout(timer), resolve()), { once: true });
  });
}

type ReadResult = "done" | "reconnect" | "retry" | "auth" | "fatal_missing" | "stopped";

/**
 * One authenticated SSE connection per job detail view. Handles replay cursors,
 * rotation, auth refresh, bounded jittered backoff and terminal shutdown.
 * Native EventSource is not used because it cannot send the bearer header.
 */
export class JobStreamController {
  private readonly options: JobStreamOptions;
  private readonly abort = new AbortController();
  private cursor: string | null = null;
  private retryMs = DEFAULT_RETRY_MS;
  private failures = 0;
  private authFailures = 0;
  private running = false;

  constructor(options: JobStreamOptions) {
    this.options = options;
  }

  /** The last applied epoch event ID, sent as Last-Event-ID on reconnect. */
  get lastEventId(): string | null {
    return this.cursor;
  }

  start(): Promise<void> {
    if (this.running) throw new Error("Stream already started");
    this.running = true;
    return this.loop();
  }

  stop() {
    this.abort.abort();
  }

  private get signal() {
    return this.abort.signal;
  }

  private emitConnection(state: ConnectionState, detail?: string) {
    this.options.handlers.onConnection?.(state, detail);
  }

  private async loop() {
    const { client, handlers } = this.options;
    try {
      while (!this.signal.aborted) {
        const token = client.accessToken;
        if (!token) {
          handlers.onFatal?.("signed_out");
          return;
        }
        this.emitConnection("connecting");
        const result = await this.connect(token);
        if (this.signal.aborted || result === "stopped") return;
        if (result === "done") return;
        if (result === "fatal_missing") {
          handlers.onFatal?.("job_not_found");
          return;
        }
        if (result === "auth") {
          // Refresh once; a second rejection before any accepted frame ends the session.
          if (this.authFailures++ > 0) {
            handlers.onFatal?.("signed_out");
            return;
          }
          try {
            await client.refreshAccessToken(token);
          } catch (error) {
            if (toApiError(error).kind === "unauthorized") {
              handlers.onFatal?.("signed_out");
              return;
            }
            await this.backoff("Could not refresh your session; retrying");
          }
          continue;
        }
        if (result === "reconnect") {
          this.failures = 0;
          continue;
        }
        await this.backoff("Connection lost; reconnecting");
      }
    } finally {
      this.emitConnection("closed");
    }
  }

  private async backoff(detail: string, minimumMs = 0) {
    const max = this.options.maxBackoffMs ?? 30_000;
    const base = Math.min(max, this.retryMs * 2 ** Math.min(this.failures, 6));
    const random = this.options.random ?? Math.random;
    const delay = Math.max(minimumMs, Math.round(base / 2 + random() * (base / 2)));
    this.failures++;
    this.emitConnection("waiting", detail);
    await (this.options.sleep ?? abortableSleep)(delay, this.signal);
  }

  private async connect(token: string): Promise<ReadResult> {
    const { client, jobId } = this.options;
    const headers: Record<string, string> = {
      Accept: "text/event-stream",
      Authorization: `Bearer ${token}`,
    };
    if (this.cursor) headers["Last-Event-ID"] = this.cursor;
    let response: Response;
    try {
      response = await (this.options.fetchImpl ?? globalThis.fetch.bind(globalThis))(
        client.url(`/jobs/${jobId}/events`),
        { headers, signal: this.signal, cache: "no-store" },
      );
    } catch {
      return this.signal.aborted ? "stopped" : "retry";
    }
    if (response.status === 401) return "auth";
    if (response.status === 404) return "fatal_missing";
    if (response.status === 422) {
      // Invalid, foreign or ahead cursor: rebuild from committed history.
      this.cursor = null;
      this.options.handlers.onHistoryReset?.();
      return "reconnect";
    }
    if (response.status === 429) {
      const header = response.headers.get("Retry-After");
      const seconds = header && /^\d+$/.test(header) ? Number(header) : 2;
      await this.backoff("Too many live connections; waiting", seconds * 1000);
      return "reconnect";
    }
    if (!response.ok || !response.body) return "retry";
    return this.read(response.body);
  }

  private async read(body: ReadableStream<Uint8Array>): Promise<ReadResult> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    const parser = new SseParser();
    // Cancelling the reader on abort also unblocks a pending read in every runtime.
    const cancel = () => reader.cancel().catch(() => undefined);
    this.signal.addEventListener("abort", cancel, { once: true });
    this.emitConnection("open");
    try {
      while (true) {
        let chunk: ReadableStreamReadResult<Uint8Array>;
        try {
          chunk = await reader.read();
        } catch {
          return this.signal.aborted ? "stopped" : "retry";
        }
        if (chunk.done) return this.signal.aborted ? "stopped" : "retry";
        const frames = parser.push(decoder.decode(chunk.value, { stream: true }));
        if (parser.retry !== null) this.retryMs = parser.retry;
        for (const frame of frames) {
          if (this.signal.aborted) return "stopped";
          const outcome = this.apply(frame);
          if (outcome) return outcome;
        }
      }
    } finally {
      this.signal.removeEventListener("abort", cancel);
      void cancel();
    }
  }

  private apply(frame: SseFrame): ReadResult | null {
    const { handlers } = this.options;
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(frame.data) as Record<string, unknown>;
    } catch {
      return null;
    }
    switch (frame.event) {
      case "status":
        this.failures = 0;
        this.authFailures = 0;
        handlers.onStatus?.(payload as unknown as StreamJobState);
        return null;
      case "epoch": {
        const epoch = payload.epoch;
        const metrics = payload.metrics;
        if (typeof epoch !== "number" || !metrics || typeof metrics !== "object") return null;
        handlers.onEpoch?.(epoch, metrics as Record<string, number>);
        // Advance the replay cursor only after the epoch has been applied.
        if (frame.id) this.cursor = frame.id;
        return null;
      }
      case "log":
        if (typeof payload.line === "string") {
          handlers.onLog?.(payload.line, typeof payload.ts === "string" ? payload.ts : undefined);
        }
        return null;
      case "done":
        handlers.onDone?.(payload as unknown as StreamJobState);
        return "done";
      case "reconnect":
        return "reconnect";
      case "error":
        switch (payload.reason) {
          case "authentication_required":
            return "auth";
          case "job_not_found":
            return "fatal_missing";
          default:
            return "retry";
        }
      default:
        return null;
    }
  }
}
