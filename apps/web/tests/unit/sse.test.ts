import { describe, expect, it } from "vitest";
import { JobStreamController, type StreamJobState } from "@/lib/events/job-stream";
import { SseParser } from "@/lib/events/sse-parser";
import { historyFromRows, upsertEpoch, type MetricHistory } from "@/lib/metrics";

describe("SseParser", () => {
  it("handles split frames, comments, retry, CRLF and multiline data", () => {
    const parser = new SseParser();
    expect(parser.push("retry: 2000\n\n: keep-alive\n\nid: epoch:j:1\nev")).toEqual([]);
    expect(parser.retry).toBe(2000);
    const frames = parser.push(
      'ent: epoch\r\ndata: {"a":\r\ndata: 1}\r\n\r\nevent: log\ndata: x\n\n',
    );
    expect(frames).toEqual([
      { event: "epoch", data: '{"a":\n1}', id: "epoch:j:1" },
      { event: "log", data: "x", id: undefined },
    ]);
  });

  it("handles a CR at a chunk boundary followed by LF", () => {
    const parser = new SseParser();
    expect(parser.push("data: a\r")).toEqual([]);
    expect(parser.push("\n\r")).toEqual([{ event: "message", data: "a", id: undefined }]);
    expect(parser.push("\ndata: b\n\n")).toEqual([{ event: "message", data: "b", id: undefined }]);
  });

  it("keeps multibyte characters split across network chunks intact", () => {
    const bytes = new TextEncoder().encode('event: log\ndata: {"line":"Prüfung ✓"}\n\n');
    const decoder = new TextDecoder();
    const parser = new SseParser();
    const frames = [];
    for (let i = 0; i < bytes.length; i += 3) {
      frames.push(...parser.push(decoder.decode(bytes.slice(i, i + 3), { stream: true })));
    }
    expect(JSON.parse(frames[0]!.data).line).toBe("Prüfung ✓");
  });
});

const JOB = "11111111-1111-4111-8111-111111111111";

function frame(event: string, data: object, id?: string) {
  return `${id ? `id: ${id}\n` : ""}event: ${event}\ndata: ${JSON.stringify({ type: event, job_id: JOB, ts: "t", ...data })}\n\n`;
}

function streamResponse(chunks: string[], { close = true } = {}) {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
        if (close) controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } },
  );
}

function harness(responses: ((headers: Record<string, string>) => Response)[]) {
  const requests: Record<string, string>[] = [];
  let token = "t1";
  let refreshes = 0;
  const client = {
    get accessToken() {
      return token;
    },
    url: (path: string) => `http://api.test/api/v1${path}`,
    refreshAccessToken: async () => {
      refreshes++;
      token = `t${refreshes + 1}`;
      return token;
    },
  };
  const fetchImpl = (async (_url: RequestInfo | URL, init?: RequestInit) => {
    const headers = init?.headers as Record<string, string>;
    requests.push(headers);
    const next = responses.shift();
    if (!next) {
      controller.stop();
      throw new Error("no more responses");
    }
    return next(headers);
  }) as typeof fetch;
  let history: MetricHistory = new Map();
  const statuses: StreamJobState[] = [];
  const events: string[] = [];
  const controller = new JobStreamController({
    jobId: JOB,
    client: client as never,
    fetchImpl,
    sleep: async () => undefined,
    random: () => 0.5,
    handlers: {
      onStatus: (s) => statuses.push(s),
      onEpoch: (epoch, metrics) => (history = upsertEpoch(history, epoch, metrics)),
      onDone: (s) => events.push(`done:${s.status}:${s.model_id}`),
      onHistoryReset: () => {
        history = new Map();
        events.push("reset");
      },
      onFatal: (reason) => events.push(`fatal:${reason}`),
    },
  });
  return {
    controller,
    requests,
    statuses,
    events,
    getHistory: () => history,
    getRefreshes: () => refreshes,
  };
}

describe("JobStreamController", () => {
  it("resumes after rotation with the last applied epoch and does not duplicate points", async () => {
    const h = harness([
      () =>
        streamResponse([
          "retry: 2000\n\n",
          frame("status", { status: "running" }),
          frame("epoch", { epoch: 1, metrics: { train_loss: 0.9 } }, `epoch:${JOB}:1`),
          frame("log", { line: "Running training" }),
          frame("epoch", { epoch: 2, metrics: { train_loss: 0.7 } }, `epoch:${JOB}:2`),
          frame("reconnect", { reason: "stream_window_ended" }),
        ]),
      // A transport drop mid-stream.
      () => streamResponse([frame("status", { status: "running" })]),
      () =>
        streamResponse([
          frame("status", { status: "running" }),
          frame(
            "epoch",
            { epoch: 3, metrics: { train_loss: 0.5, "val_mAP50(B)": 0.4 } },
            `epoch:${JOB}:3`,
          ),
          frame("done", { status: "succeeded", model_id: "m1" }),
        ]),
    ]);
    await h.controller.start();
    expect(h.requests.map((r) => r["Last-Event-ID"])).toEqual([
      undefined,
      `epoch:${JOB}:2`,
      `epoch:${JOB}:2`,
    ]);
    expect([...h.getHistory().keys()]).toEqual([1, 2, 3]);
    expect(h.getHistory().get(3)).toEqual({ train_loss: 0.5, "val_mAP50(B)": 0.4 });
    expect(h.events).toEqual(["done:succeeded:m1"]);
  });

  it("replays history idempotently when the same epochs arrive again", () => {
    let history = historyFromRows([
      { epoch: 1, split: "train", name: "loss", value: 0.9 },
      { epoch: 1, split: "train", name: "learning_rate", value: 0.0003 },
    ]);
    history = upsertEpoch(history, 1, { train_loss: 0.9, lr: 0.0003 });
    expect(history.size).toBe(1);
    expect(history.get(1)).toEqual({ train_loss: 0.9, lr: 0.0003 });
  });

  it("clears an invalid cursor and rebuilds from history", async () => {
    const h = harness([
      () => streamResponse([frame("epoch", { epoch: 1, metrics: { a: 1 } }, `epoch:${JOB}:1`)]),
      () => new Response(JSON.stringify({ detail: "bad cursor" }), { status: 422 }),
      () =>
        streamResponse([
          frame("epoch", { epoch: 1, metrics: { a: 1 } }, `epoch:${JOB}:1`),
          frame("done", { status: "failed", error: "Training failed" }),
        ]),
    ]);
    await h.controller.start();
    expect(h.requests[1]!["Last-Event-ID"]).toBe(`epoch:${JOB}:1`);
    expect(h.requests[2]!["Last-Event-ID"]).toBeUndefined();
    expect(h.events).toEqual(["reset", "done:failed:undefined"]);
  });

  it("refreshes authentication once, then gives up if still rejected", async () => {
    const h = harness([
      () => new Response(null, { status: 401 }),
      () => streamResponse([frame("error", { reason: "authentication_required" })]),
    ]);
    await h.controller.start();
    expect(h.getRefreshes()).toBe(1);
    expect(h.requests.map((r) => r.Authorization)).toEqual(["Bearer t1", "Bearer t2"]);
    expect(h.events).toEqual(["fatal:signed_out"]);
  });

  it("stops on a missing job and on terminal done without reconnecting", async () => {
    const missing = harness([() => new Response(null, { status: 404 })]);
    await missing.controller.start();
    expect(missing.events).toEqual(["fatal:job_not_found"]);

    const done = harness([
      () => streamResponse([frame("done", { status: "cancelled" })], { close: false }),
    ]);
    await done.controller.start();
    expect(done.requests).toHaveLength(1);
    expect(done.events).toEqual(["done:cancelled:undefined"]);
  });

  it("waits for Retry-After on 429 and retries service errors with backoff", async () => {
    const waits: number[] = [];
    const h = harness([
      () => new Response(null, { status: 429, headers: { "Retry-After": "7" } }),
      () => streamResponse([frame("error", { reason: "database_unavailable" })]),
      () => streamResponse([frame("done", { status: "succeeded", model_id: "m" })]),
    ]);
    // Re-create with a recording sleep.
    const controller = new JobStreamController({
      jobId: JOB,
      client: (h.controller as unknown as { options: { client: never } }).options.client,
      fetchImpl: (h.controller as unknown as { options: { fetchImpl: typeof fetch } }).options
        .fetchImpl,
      sleep: async (ms) => void waits.push(ms),
      random: () => 0,
      handlers: { onDone: (s) => h.events.push(`done:${s.status}`) },
    });
    await controller.start();
    expect(waits[0]).toBe(7000);
    expect(waits).toHaveLength(2);
    expect(h.events).toEqual(["done:succeeded"]);
  });

  it("aborts the reader when stopped", async () => {
    const h = harness([
      (headers) => {
        void headers;
        return streamResponse([frame("status", { status: "running" })], { close: false });
      },
    ]);
    const finished = h.controller.start();
    await new Promise((r) => setTimeout(r, 10));
    h.controller.stop();
    await finished;
    expect(h.requests).toHaveLength(1);
  });
});
