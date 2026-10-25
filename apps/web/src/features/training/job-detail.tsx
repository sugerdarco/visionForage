"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Card, PageHeader } from "@/components/ui/page-header";
import { ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import { isTerminal, type Job } from "@/lib/api/types";
import { formatDateTime, formatDuration, shortId } from "@/lib/format";
import { ARCH_LABEL, JobStatusBadge } from "./job-status";
import { MetricPanels } from "./metric-panels";
import { PARAM_SPECS } from "./params";
import { useJobLive, MAX_PROGRESS_LINES } from "./use-job-live";
import { useVersionIndex } from "./use-version-index";

export function JobDetail() {
  const { projectId, jobId } = useParams<{ projectId: string; jobId: string }>();
  const queryClient = useQueryClient();
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: ({ signal }) => api.jobs.get(jobId, signal),
  });
  const versions = useVersionIndex(projectId);
  const live = useJobLive(job.data);

  // Fold streamed state into the cached job so every view agrees.
  const liveState = live.state;
  useEffect(() => {
    if (!liveState) return;
    queryClient.setQueryData<Job>(["job", jobId], (current) =>
      current ? { ...current, ...liveState } : current,
    );
    if (isTerminal(liveState.status))
      void queryClient.invalidateQueries({ queryKey: ["jobs", projectId] });
  }, [liveState, jobId, projectId, queryClient]);

  if (job.isPending) return <LoadingState label="Loading training run" />;
  if (job.isError || live.missing)
    return (
      <ErrorState
        error={job.error ?? toApiError(new Error("This training run is no longer available."))}
        onRetry={() => job.refetch()}
      />
    );

  const data = job.data;
  const version = versions.data?.byId.get(data.dataset_version_id);
  const epochsRecorded = live.history.size;
  const requestedEpochs =
    typeof data.hyperparams.epochs === "number" ? data.hyperparams.epochs : null;

  return (
    <>
      <nav aria-label="Breadcrumb" className="mb-2 text-sm text-muted">
        <Link href={`/projects/${projectId}/training`} className="hover:underline">
          Training
        </Link>{" "}
        / {shortId(data.id)}
      </nav>
      <PageHeader
        title={`Run ${shortId(data.id)}`}
        description={
          <span className="inline-flex flex-wrap items-center gap-2">
            <JobStatusBadge job={data} /> {ARCH_LABEL[data.model_arch] ?? data.model_arch} on{" "}
            {version ? (
              <Link className="underline" href={`/projects/${projectId}/versions/${version.id}`}>
                dataset v{version.version}
              </Link>
            ) : (
              `dataset ${shortId(data.dataset_version_id)}`
            )}
          </span>
        }
        actions={<CancelButton job={data} />}
      />
      <div className="flex flex-col gap-6">
        <StatusNotice
          job={data}
          epochsRecorded={epochsRecorded}
          requestedEpochs={requestedEpochs}
          projectId={projectId}
        />
        {!isTerminal(data.status) && <ConnectionLine connection={live.connection} />}
        <section aria-label="Training curves" className="flex flex-col gap-2">
          <div className="flex items-baseline justify-between">
            <h2 className="text-base font-semibold">Curves</h2>
            <span className="text-sm text-muted">
              {epochsRecorded} epoch{epochsRecorded === 1 ? "" : "s"} recorded
              {requestedEpochs ? ` (up to ${requestedEpochs} requested)` : ""}
            </span>
          </div>
          {live.historyError ? (
            <ErrorState error={new Error(live.historyError)} />
          ) : (
            <MetricPanels history={live.history} />
          )}
          <p className="text-xs text-muted">
            Validation metrics are measured on the validation split during training. Final test-set
            results are not available from the training-run API yet.
          </p>
        </section>
        <div className="grid gap-6 lg:grid-cols-2">
          <Card title="Effective settings">
            <Parameters job={data} />
          </Card>
          <Card title="Timeline">
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
              <dt className="text-muted">Created</dt>
              <dd>{formatDateTime(data.created_at)}</dd>
              <dt className="text-muted">Sent to worker queue</dt>
              <dd>{formatDateTime(data.dispatched_at)}</dd>
              <dt className="text-muted">Started</dt>
              <dd>{formatDateTime(data.started_at)}</dd>
              <dt className="text-muted">Finished</dt>
              <dd>{formatDateTime(data.finished_at)}</dd>
              <dt className="text-muted">Duration</dt>
              <dd>{data.started_at ? formatDuration(data.started_at, data.finished_at) : "—"}</dd>
              {data.gpu_id !== null && data.gpu_id !== undefined && (
                <>
                  <dt className="text-muted">GPU</dt>
                  <dd>{data.gpu_id}</dd>
                </>
              )}
              {data.cancel_requested_at && (
                <>
                  <dt className="text-muted">Cancellation requested</dt>
                  <dd>{formatDateTime(data.cancel_requested_at)}</dd>
                </>
              )}
            </dl>
          </Card>
        </div>
        {!isTerminal(data.status) || live.progress.length > 0 ? (
          <Card title="Progress">
            {live.progress.length === 0 ? (
              <p className="text-sm text-muted">
                Stage updates appear here while this page is open.
              </p>
            ) : (
              <ol className="max-h-60 overflow-y-auto font-mono text-xs" aria-live="polite">
                {live.progress.map((p) => (
                  <li key={p.id} className="border-b border-line py-1 last:border-0">
                    <span className="text-muted">
                      {p.ts ? new Date(p.ts).toLocaleTimeString() : ""}
                    </span>{" "}
                    {p.line}
                  </li>
                ))}
              </ol>
            )}
            <p className="mt-2 text-xs text-muted">
              Best-effort stage messages, not a complete training log. They are not replayed after a
              reload; the latest {MAX_PROGRESS_LINES} are kept.
            </p>
          </Card>
        ) : null}
      </div>
    </>
  );
}

function ConnectionLine({ connection }: { connection: { state: string; detail?: string } }) {
  const text =
    connection.state === "open"
      ? "Live updates connected"
      : connection.state === "connecting"
        ? "Connecting to live updates…"
        : connection.state === "waiting"
          ? `${connection.detail ?? "Reconnecting"}…`
          : connection.state === "closed"
            ? "Live updates stopped"
            : "";
  if (!text) return null;
  return (
    <p role="status" className="text-sm text-muted">
      <span
        className={`mr-2 inline-block h-2 w-2 rounded-full ${connection.state === "open" ? "bg-green-600" : "bg-amber-500"}`}
      />
      {text}
    </p>
  );
}

function StatusNotice({
  job,
  epochsRecorded,
  requestedEpochs,
  projectId,
}: {
  job: Job;
  epochsRecorded: number;
  requestedEpochs: number | null;
  projectId: string;
}) {
  switch (job.status) {
    case "queued":
      return (
        <Notice title="Waiting for a GPU worker">
          The run is accepted and queued. It starts when a training worker claims it
          {job.dispatched_at ? "" : "; it has not been handed to the worker queue yet"}.
        </Notice>
      );
    case "running":
      return job.cancel_requested_at ? (
        <Notice tone="warn" title="Cancellation requested">
          The worker stops at its next checkpoint. The run stays “running” until the worker confirms
          it has stopped.
        </Notice>
      ) : null;
    case "succeeded":
      return (
        <Notice tone="success" title="Training succeeded">
          <p>
            The model passed evaluation and ONNX verification and was registered.
            {requestedEpochs && epochsRecorded > 0 && epochsRecorded < requestedEpochs
              ? ` Early stopping ended training after ${epochsRecorded} of ${requestedEpochs} epochs; this is normal.`
              : ""}
          </p>
          {job.model_id && (
            <Link
              className="mt-1 inline-block underline"
              href={`/projects/${projectId}/models?model=${job.model_id}`}
            >
              View registered model {shortId(job.model_id)}
            </Link>
          )}
        </Notice>
      );
    case "failed":
      return (
        <Notice tone="error" title="Training failed">
          {job.error ?? "The run failed without a public error summary."}
          {job.cancel_requested_at ? " A cancellation had been requested." : ""}
        </Notice>
      );
    case "cancelled":
      return (
        <Notice title="Cancelled">This run was cancelled and did not register a model.</Notice>
      );
  }
}

function Parameters({ job }: { job: Job }) {
  const task = job.model_arch === "yolov8s" ? "detection" : "classification";
  const known = PARAM_SPECS[task];
  const keys = [
    ...known.map((s) => s.key),
    ...Object.keys(job.hyperparams).filter((k) => !known.some((s) => s.key === k)),
  ];
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
      {keys
        .filter((key) => key in job.hyperparams)
        .map((key) => {
          const value = job.hyperparams[key];
          return (
            <div key={key} className="contents">
              <dt className="text-muted">{known.find((s) => s.key === key)?.label ?? key}</dt>
              <dd className="font-mono">
                {typeof value === "boolean" ? (value ? "on" : "off") : String(value)}
              </dd>
            </div>
          );
        })}
    </dl>
  );
}

function CancelButton({ job }: { job: Job }) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (isTerminal(job.status)) return null;
  if (job.cancel_requested_at) return <Badge tone="amber">Cancellation requested</Badge>;

  async function cancel() {
    setBusy(true);
    setError(null);
    try {
      const updated = await api.jobs.cancel(job.id);
      queryClient.setQueryData(["job", job.id], updated);
      await queryClient.invalidateQueries({ queryKey: ["jobs", job.project_id] });
      setOpen(false);
    } catch (raw) {
      setError(toApiError(raw).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Button variant="danger" onClick={() => setOpen(true)}>
        Cancel run
      </Button>
      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        title="Cancel this run?"
        footer={
          <>
            <Button onClick={() => setOpen(false)}>Keep running</Button>
            <Button variant="danger" busy={busy} onClick={cancel}>
              Cancel run
            </Button>
          </>
        }
      >
        <p>
          {job.status === "queued"
            ? "The run has not started and will be cancelled immediately."
            : "The worker is asked to stop at its next checkpoint. The run remains running until the worker confirms."}
        </p>
        {error && (
          <Notice tone="error" className="mt-3">
            {error}
          </Notice>
        )}
      </Dialog>
    </>
  );
}
