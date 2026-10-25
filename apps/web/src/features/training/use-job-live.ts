"use client";

import { useEffect, useState } from "react";
import { useSession } from "@/features/auth/session";
import { apiClient } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import { isTerminal, type Job } from "@/lib/api/types";
import {
  JobStreamController,
  type ConnectionState,
  type StreamJobState,
} from "@/lib/events/job-stream";
import { historyFromRows, upsertEpoch, type MetricHistory } from "@/lib/metrics";

export const MAX_PROGRESS_LINES = 200;

export interface ProgressLine {
  id: number;
  ts: string | undefined;
  line: string;
}

export interface JobLive {
  state: StreamJobState | null;
  history: MetricHistory;
  historyComplete: boolean;
  progress: ProgressLine[];
  connection: { state: ConnectionState | "idle"; detail?: string };
  missing: boolean;
  historyError: string | null;
}

let lineId = 0;

/**
 * Terminal jobs: read every REST metrics page once. Active jobs: one SSE stream
 * that replays history (no cursor on a fresh view), then follows live updates
 * until `done`. Points are upserted by epoch, so replays never duplicate.
 */
export function useJobLive(job: Job | undefined): JobLive {
  const { signOut } = useSession();
  const [state, setState] = useState<StreamJobState | null>(null);
  const [history, setHistory] = useState<MetricHistory>(() => new Map());
  const [historyComplete, setHistoryComplete] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressLine[]>([]);
  const [connection, setConnection] = useState<JobLive["connection"]>({ state: "idle" });
  const [missing, setMissing] = useState(false);
  const jobId = job?.id;
  // Decide the data source once, from the status when the page opened.
  const [mode, setMode] = useState<"rest" | "stream" | null>(null);
  if (job && mode === null) setMode(isTerminal(job.status) ? "rest" : "stream");

  useEffect(() => {
    if (!jobId || mode !== "rest") return;
    const controller = new AbortController();
    api.jobs
      .allMetrics(jobId, controller.signal)
      .then((rows) => {
        setHistory(historyFromRows(rows));
        setHistoryComplete(true);
      })
      .catch((error) => {
        if (!controller.signal.aborted) setHistoryError(toApiError(error).message);
      });
    return () => controller.abort();
  }, [jobId, mode]);

  useEffect(() => {
    if (!jobId || mode !== "stream") return;
    const controller = new JobStreamController({
      jobId,
      client: apiClient,
      handlers: {
        onStatus: setState,
        onEpoch: (epoch, metrics) => setHistory((h) => upsertEpoch(h, epoch, metrics)),
        onLog: (line, ts) =>
          setProgress((lines) => [...lines, { id: ++lineId, ts, line }].slice(-MAX_PROGRESS_LINES)),
        onDone: (final) => {
          setState(final);
          setHistoryComplete(true);
        },
        onHistoryReset: () => setHistory(new Map()),
        onConnection: (next, detail) => setConnection({ state: next, detail }),
        onFatal: (reason) => {
          if (reason === "signed_out") signOut("expired");
          else setMissing(true);
        },
      },
    });
    void controller.start();
    return () => controller.stop();
  }, [jobId, mode, signOut]);

  return { state, history, historyComplete, progress, connection, missing, historyError };
}
