"use client";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Field, inputClass } from "@/components/ui/field";
import { Card, PageHeader } from "@/components/ui/page-header";
import { Pagination } from "@/components/ui/pagination";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { useProjectId } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import type { DeploymentStats } from "@/lib/api/types";
import { formatDateTime } from "@/lib/format";
import { useDeployments } from "./use-deployments";

export function MonitoringScreen() {
  const projectId = useProjectId();
  const params = useSearchParams();
  const deployments = useDeployments(projectId);
  const [selected, setSelected] = useState(params.get("deployment") ?? "");
  const id = deployments.data?.some((d) => d.id === selected)
    ? selected
    : deployments.data?.[0]?.id;
  return (
    <>
      <PageHeader
        title="Monitoring"
        description="Recorded predictions, latency and confidence for each endpoint."
      />
      {deployments.isPending ? (
        <LoadingState />
      ) : deployments.isError ? (
        <ErrorState error={deployments.error} onRetry={() => deployments.refetch()} />
      ) : !deployments.data.length ? (
        <EmptyState title="No endpoints to monitor">
          <Link className="text-accent underline" href={`/projects/${projectId}/deployments`}>
            Deploy a model
          </Link>
        </EmptyState>
      ) : (
        <>
          <Field label="Monitored endpoint">
            {(p) => (
              <select
                {...p}
                className={inputClass}
                aria-label="Monitored endpoint"
                value={id}
                onChange={(e) => setSelected(e.target.value)}
              >
                {deployments.data.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.slug} · model v{d.model_version}
                  </option>
                ))}
              </select>
            )}
          </Field>
          {id && <Monitor key={id} id={id} projectId={projectId} />}
        </>
      )}
    </>
  );
}

function Monitor({ id, projectId }: { id: string; projectId: string }) {
  const [window, setWindow] = useState<DeploymentStats["window"]>("24h");
  const [page, setPage] = useState(1);
  const stats = useQuery({
    queryKey: ["deployment-stats", id, window],
    queryFn: ({ signal }) => api.deployments.stats(id, window, signal),
    refetchInterval: 10000,
  });
  const history = useQuery({
    queryKey: ["prediction-history", id, page],
    queryFn: ({ signal }) => api.deployments.predictions(id, page, signal),
    refetchInterval: 15000,
  });
  const state = useQuery({
    queryKey: ["deployment", id],
    queryFn: ({ signal }) => api.deployments.get(id, signal),
    refetchInterval: 10000,
  });
  const ms = (value: number | null) => (value == null ? "Not available" : `${value.toFixed(2)} ms`);
  return (
    <div className="mt-5 flex flex-col gap-6">
      <div className="flex flex-wrap items-end gap-4">
        <Field label="Monitoring window">
          {(p) => (
            <select
              {...p}
              className={inputClass}
              value={window}
              onChange={(e) => setWindow(e.target.value as DeploymentStats["window"])}
            >
              {["1h", "24h", "7d", "30d"].map((w) => (
                <option key={w}>{w}</option>
              ))}
            </select>
          )}
        </Field>
        <Button
          onClick={() => {
            void stats.refetch();
            void history.refetch();
            void state.refetch();
          }}
        >
          Refresh monitoring
        </Button>
        <span className="pb-2 text-sm">Endpoint: {state.data?.status ?? "Checking…"}</span>
      </div>
      {stats.isPending ? (
        <LoadingState />
      ) : stats.isError ? (
        <ErrorState error={stats.error} onRetry={() => stats.refetch()} />
      ) : (
        <>
          <dl className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            {[
              ["Recorded requests", stats.data.requests],
              ["Low confidence", stats.data.low_confidence_count],
              ["Mean latency", ms(stats.data.latency_mean_ms)],
              ["p50 latency", ms(stats.data.latency_p50_ms)],
              ["p95 latency", ms(stats.data.latency_p95_ms)],
            ].map(([label, value]) => (
              <div key={label} className="rounded border border-line bg-panel p-4">
                <dt className="text-sm text-muted">{label}</dt>
                <dd
                  className="mt-1 text-xl font-semibold"
                  data-testid={label === "Recorded requests" ? "recorded-requests" : undefined}
                >
                  {value}
                </dd>
              </div>
            ))}
          </dl>
          <p className="text-xs text-muted">
            Successful recorded predictions only; collection can lag by several seconds. Rejected
            requests are excluded. Low-confidence counts include predictions already reviewed.
          </p>
          <div className="grid gap-6 lg:grid-cols-2">
            <CountChart
              title={`Class distribution (${stats.data.class_count_unit})`}
              rows={Object.entries(stats.data.class_distribution).map(([name, count]) => ({
                name,
                count,
              }))}
            />
            <CountChart
              title="Confidence histogram"
              rows={stats.data.confidence_histogram.map((bin) => ({
                name: `${bin.lower.toFixed(1)}–${bin.upper.toFixed(1)}${bin.upper_inclusive ? " inclusive" : ""}`,
                count: bin.count,
              }))}
            />
          </div>
        </>
      )}
      <Card title="Prediction history">
        <p className="mb-3 text-sm text-muted">
          All recorded history for this endpoint, newest first. Image links refresh automatically
          and expire after five minutes.
        </p>
        {history.isPending ? (
          <LoadingState />
        ) : history.isError ? (
          <ErrorState error={history.error} onRetry={() => history.refetch()} />
        ) : !history.data.total ? (
          <EmptyState title="No recorded predictions yet">
            Send a prediction from the Deployments screen.
          </EmptyState>
        ) : (
          <>
            <h3 className="mb-2 font-medium">Latency on this history page</h3>
            <div className="mb-5 h-56" aria-hidden="true">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={[...history.data.items]
                    .reverse()
                    .map((p) => ({ time: formatDateTime(p.created_at), latency: p.latency_ms }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                  <YAxis unit=" ms" width={75} />
                  <Tooltip formatter={(value) => [`${Number(value).toFixed(2)} ms`, "Latency"]} />
                  <Line dataKey="latency" stroke="#2563eb" isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="mb-3 text-xs text-muted">
              Each point is a prediction on this page, ordered oldest to newest. Exact latency is
              also shown below.
            </p>
            <ul className="flex flex-col gap-3">
              {history.data.items.map((item) => (
                <li
                  key={item.id}
                  className="flex flex-col gap-4 rounded border border-line p-3 sm:flex-row"
                >
                  {/* Private signed URLs are not sent through Next's image optimizer. */}
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={item.image_url}
                    alt="Prediction input"
                    referrerPolicy="no-referrer"
                    className="h-28 w-40 rounded bg-slate-100 object-contain"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="text-sm">
                      {formatDateTime(item.created_at)} · model v{item.model_version}
                    </p>
                    <p className="mt-1 font-medium">
                      {"label" in item.output
                        ? item.output.label
                        : `${item.output.boxes.length} detected boxes`}{" "}
                      · {(item.max_confidence * 100).toFixed(1)}% confidence
                    </p>
                    <p className="text-sm text-muted">
                      {item.latency_ms.toFixed(2)} ms ·{" "}
                      {item.reviewed_at
                        ? "Reviewed"
                        : item.sent_to_review
                          ? "Awaiting review"
                          : "Above review threshold"}
                    </p>
                    <details className="mt-2 text-xs">
                      <summary className="cursor-pointer">Prediction details</summary>
                      <pre className="overflow-auto p-2">
                        {JSON.stringify(item.output, null, 2)}
                      </pre>
                    </details>
                    {item.sent_to_review && !item.reviewed_at && (
                      <Link
                        className="mt-2 inline-block text-sm text-accent underline"
                        href={`/projects/${projectId}/review`}
                      >
                        Review predictions
                      </Link>
                    )}
                  </div>
                </li>
              ))}
            </ul>
            <Pagination
              page={page}
              pageSize={20}
              total={history.data.total}
              label="predictions"
              onPage={setPage}
            />
          </>
        )}
      </Card>
    </div>
  );
}

function CountChart({ title, rows }: { title: string; rows: { name: string; count: number }[] }) {
  return (
    <Card title={title}>
      {!rows.length ? (
        <p className="text-sm text-muted">No recorded values.</p>
      ) : (
        <>
          <div className="h-64 min-w-0" aria-hidden="true">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rows}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#2563eb" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <details>
            <summary className="cursor-pointer text-sm text-accent">
              View {title.toLowerCase()} table
            </summary>
            <table className="mt-2 w-full text-sm">
              <caption className="sr-only">{title}</caption>
              <thead>
                <tr>
                  <th className="text-left">Category</th>
                  <th className="text-right">Count</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.name} className="border-t border-line">
                    <th scope="row" className="py-1 text-left font-normal">
                      {row.name}
                    </th>
                    <td className="text-right">{row.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        </>
      )}
    </Card>
  );
}
