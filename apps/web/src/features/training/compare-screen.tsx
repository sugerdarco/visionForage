"use client";

import { useQueries, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { inputClass } from "@/components/ui/field";
import { Card, PageHeader } from "@/components/ui/page-header";
import { EmptyState, ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { useProjectId } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import type { Job } from "@/lib/api/types";
import {
  formatMetric,
  historyFromRows,
  metricKeys,
  metricLabel,
  type MetricHistory,
} from "@/lib/metrics";
import { formatDateTime, shortId } from "@/lib/format";
import { ARCH_LABEL, JobStatusBadge } from "./job-status";
import { MetricChart, seriesStyle } from "./metric-chart";
import { seriesFor } from "./metric-panels";
import { PARAM_SPECS } from "./params";
import { useVersionIndex } from "./use-version-index";

export const MAX_COMPARE = 4;

export function CompareScreen() {
  const projectId = useProjectId();
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const selected = (params.get("jobs") ?? "").split(",").filter(Boolean).slice(0, MAX_COMPARE);
  const versions = useVersionIndex(projectId);
  const jobs = useQuery({
    queryKey: ["jobs", projectId, "compare-picker"],
    queryFn: ({ signal }) => api.jobs.list(projectId, 1, 100, signal),
  });

  const setSelected = (ids: string[]) => {
    const query = ids.length ? `?jobs=${ids.join(",")}` : "";
    router.replace(`${pathname}${query}`, { scroll: false });
  };

  return (
    <>
      <PageHeader
        title="Compare runs"
        description={`Pick up to ${MAX_COMPARE} runs to compare their settings and recorded epoch metrics. Comparison uses training and validation curves only.`}
      />
      <div className="grid gap-6 xl:grid-cols-[18rem_minmax(0,1fr)]">
        <Card title="Runs" className="h-fit">
          {jobs.isPending ? (
            <LoadingState />
          ) : jobs.isError ? (
            <ErrorState error={jobs.error} onRetry={() => jobs.refetch()} />
          ) : jobs.data.items.length === 0 ? (
            <p className="text-sm text-muted">No runs yet.</p>
          ) : (
            <fieldset>
              <legend className="sr-only">Runs to compare</legend>
              <ul className="flex flex-col gap-1">
                {jobs.data.items.map((job) => {
                  const checked = selected.includes(job.id);
                  const disabled = !checked && selected.length >= MAX_COMPARE;
                  const v = versions.data?.byId.get(job.dataset_version_id);
                  return (
                    <li key={job.id}>
                      <label
                        className={`flex items-start gap-2 rounded px-2 py-1.5 text-sm ${disabled ? "opacity-50" : "hover:bg-slate-50"}`}
                      >
                        <input
                          type="checkbox"
                          className="mt-1"
                          checked={checked}
                          disabled={disabled}
                          onChange={() =>
                            setSelected(
                              checked
                                ? selected.filter((id) => id !== job.id)
                                : [...selected, job.id],
                            )
                          }
                        />
                        <span>
                          <span className="font-mono">{shortId(job.id)}</span> ·{" "}
                          {v ? `v${v.version}` : "dataset ?"}
                          <span className="block">
                            <JobStatusBadge job={job} />
                          </span>
                          <span className="block text-xs text-muted">
                            {formatDateTime(job.created_at)}
                          </span>
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
              {jobs.data.total > jobs.data.items.length && (
                <p className="mt-2 text-xs text-muted">
                  Showing the newest {jobs.data.items.length} runs.
                </p>
              )}
            </fieldset>
          )}
        </Card>
        <div className="min-w-0">
          {selected.length === 0 ? (
            <EmptyState title="Select runs to compare">
              Choose two or more runs from the list.
            </EmptyState>
          ) : (
            <Comparison
              projectId={projectId}
              jobIds={selected}
              versionNumbers={versions.data?.byId}
            />
          )}
        </div>
      </div>
    </>
  );
}

function Comparison({
  projectId,
  jobIds,
  versionNumbers,
}: {
  projectId: string;
  jobIds: string[];
  versionNumbers: Map<string, { version: number }> | undefined;
}) {
  // Historical REST data for each run; comparisons never open live streams.
  const jobQueries = useQueries({
    queries: jobIds.map((id) => ({
      queryKey: ["job", id],
      queryFn: ({ signal }: { signal: AbortSignal }) => api.jobs.get(id, signal),
    })),
  });
  const metricQueries = useQueries({
    queries: jobIds.map((id) => ({
      queryKey: ["metrics", id],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        api.jobs.allMetrics(id, signal).then(historyFromRows),
    })),
  });
  const [metric, setMetric] = useState<string | null>(null);

  const failed = [...jobQueries, ...metricQueries].find((q) => q.isError);
  if (failed) return <ErrorState error={failed.error} />;
  if (jobQueries.some((q) => q.isPending) || metricQueries.some((q) => q.isPending))
    return <LoadingState label="Loading runs" />;

  const runs = jobIds.map((id, i) => ({
    job: jobQueries[i]!.data as Job,
    history: metricQueries[i]!.data as MetricHistory,
  }));
  const datasets = new Set(runs.map((r) => r.job.dataset_version_id));
  const archs = new Set(runs.map((r) => r.job.model_arch));
  const keys = [...new Set(runs.flatMap((r) => metricKeys(r.history)))];
  const chosen =
    metric && keys.includes(metric)
      ? metric
      : (keys.find((k) => k.startsWith("val_")) ?? keys[0] ?? null);
  const runLabel = (job: Job) => {
    const v = versionNumbers?.get(job.dataset_version_id);
    return `${shortId(job.id)}${v ? ` (v${v.version})` : ""}`;
  };

  const paramKeys = [
    ...new Set(
      runs.flatMap((r) => {
        const task = r.job.model_arch === "yolov8s" ? "detection" : "classification";
        return [...PARAM_SPECS[task].map((s) => s.key), ...Object.keys(r.job.hyperparams)];
      }),
    ),
  ];

  return (
    <div className="flex flex-col gap-6">
      {datasets.size > 1 && (
        <Notice tone="warn" title="Not directly comparable">
          These runs use different dataset versions, so their splits differ. Differences in their
          metrics can come from the data rather than the settings.
        </Notice>
      )}
      {archs.size > 1 && (
        <Notice tone="warn">
          These runs use different model types; their metrics are not on the same scale.
        </Notice>
      )}
      <Card title="Settings">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-muted">
              <tr>
                <th className="py-2 pr-4 font-medium">Setting</th>
                {runs.map((r, i) => (
                  <th key={r.job.id} className="py-2 pr-4 font-medium">
                    <span
                      className="inline-block h-0.5 w-5 align-middle"
                      style={{ background: seriesStyle(i).color }}
                    />{" "}
                    <Link
                      className="font-mono text-accent underline"
                      href={`/projects/${projectId}/training/${r.job.id}`}
                    >
                      {runLabel(r.job)}
                    </Link>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <CompareRow label="Status" values={runs.map((r) => r.job.status)} />
              <CompareRow
                label="Model"
                values={runs.map((r) => ARCH_LABEL[r.job.model_arch] ?? r.job.model_arch)}
              />
              <CompareRow
                label="Dataset version"
                values={runs.map((r) => {
                  const v = versionNumbers?.get(r.job.dataset_version_id);
                  return v ? `v${v.version}` : shortId(r.job.dataset_version_id);
                })}
              />
              <CompareRow
                label="Epochs recorded"
                values={runs.map((r) => String(r.history.size))}
              />
              {paramKeys.map((key) => (
                <CompareRow
                  key={key}
                  label={
                    PARAM_SPECS.classification
                      .concat(PARAM_SPECS.detection)
                      .find((s) => s.key === key)?.label ?? key
                  }
                  values={runs.map((r) =>
                    key in r.job.hyperparams ? String(r.job.hyperparams[key]) : "Not applicable",
                  )}
                />
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted">
          Rows marked “differs” have different values across the selected runs.
        </p>
      </Card>
      <section aria-label="Metric comparison" className="flex flex-col gap-3">
        {keys.length === 0 ? (
          <p className="text-sm text-muted">None of these runs has recorded epoch metrics.</p>
        ) : (
          <>
            <label className="flex max-w-sm flex-col gap-1 text-sm">
              <span className="font-medium">Metric</span>
              <select
                className={inputClass}
                value={chosen ?? ""}
                onChange={(e) => setMetric(e.target.value)}
              >
                {keys.map((k) => (
                  <option key={k} value={k}>
                    {metricLabel(k)} ({k})
                  </option>
                ))}
              </select>
            </label>
            {chosen && (
              <>
                <MetricChart
                  title={metricLabel(chosen)}
                  series={runs.map((r) => seriesFor(r.history, chosen, r.job.id, runLabel(r.job)))}
                />
                <table className="w-full max-w-2xl text-sm">
                  <thead className="text-left text-xs text-muted">
                    <tr>
                      <th className="py-1 pr-4 font-medium">Run</th>
                      <th className="py-1 pr-4 text-right font-medium">Last recorded</th>
                      <th className="py-1 pr-4 text-right font-medium">Lowest</th>
                      <th className="py-1 text-right font-medium">Highest</th>
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((r) => {
                      const points = [...seriesFor(r.history, chosen).points.entries()].sort(
                        ([a], [b]) => a - b,
                      );
                      const values = points.map(([, v]) => v);
                      return (
                        <tr key={r.job.id} className="border-t border-line">
                          <td className="py-1 pr-4 font-mono">{runLabel(r.job)}</td>
                          <td className="py-1 pr-4 text-right font-mono">
                            {values.length
                              ? `${formatMetric(values.at(-1))} (epoch ${points.at(-1)![0]})`
                              : "Not available"}
                          </td>
                          <td className="py-1 pr-4 text-right font-mono">
                            {values.length ? formatMetric(Math.min(...values)) : "Not available"}
                          </td>
                          <td className="py-1 text-right font-mono">
                            {values.length ? formatMetric(Math.max(...values)) : "Not available"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </>
            )}
          </>
        )}
        <p className="text-xs text-muted">
          Final test-set reports are not available through the training-run API yet, so this page
          does not rank runs on held-out test results.
        </p>
      </section>
    </div>
  );
}

function CompareRow({ label, values }: { label: string; values: string[] }) {
  const differs = new Set(values).size > 1;
  return (
    <tr className={`border-t border-line ${differs ? "bg-amber-50" : ""}`}>
      <th scope="row" className="py-1.5 pr-4 text-left font-normal text-muted">
        {label}
        {differs && <span className="ml-2 text-xs font-medium text-warn">differs</span>}
      </th>
      {values.map((value, i) => (
        <td key={i} className="py-1.5 pr-4 font-mono">
          {value}
        </td>
      ))}
    </tr>
  );
}
