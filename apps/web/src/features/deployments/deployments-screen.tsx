"use client";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Field, inputClass } from "@/components/ui/field";
import { Card, PageHeader } from "@/components/ui/page-header";
import { EmptyState, ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { useProjectId } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import { submitOnce } from "@/lib/api/outcome";
import type { Deployment, DeploymentCreated, Model } from "@/lib/api/types";
import { PredictionTester } from "./prediction-tester";
import { useDeployments } from "./use-deployments";

export function DeploymentsScreen() {
  const projectId = useProjectId();
  const params = useSearchParams();
  return (
    <DeploymentControls
      key={`${projectId}:${params.get("model") ?? "default"}`}
      projectId={projectId}
      initialModelId={params.get("model") ?? ""}
    />
  );
}

function DeploymentControls({
  projectId,
  initialModelId,
}: {
  projectId: string;
  initialModelId: string;
}) {
  const cache = useQueryClient();
  const deployments = useDeployments(projectId);
  const models = useQuery({
    queryKey: ["deployable-models", projectId],
    staleTime: 0,
    queryFn: async ({ signal }) => {
      const items: Model[] = [];
      for (let page = 1; ; page++) {
        const r = await api.models.list(projectId, page, 100, signal);
        items.push(...r.items);
        if (!r.items.length || page * r.page_size >= r.total) return items;
      }
    },
  });
  const [modelId, setModelId] = useState(initialModelId);
  const [slug, setSlug] = useState("");
  const [threshold, setThreshold] = useState("0.6");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uncertain, setUncertain] = useState(false);
  const [created, setCreated] = useState<DeploymentCreated | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [stopping, setStopping] = useState<Deployment | null>(null);
  const selected = useQuery({
    queryKey: ["deployment", selectedId],
    queryFn: ({ signal }) => api.deployments.get(selectedId!, signal),
    enabled: !!selectedId,
    refetchInterval: 5000,
  });
  const chosen = modelId
    ? models.data?.some((m) => m.id === modelId)
      ? modelId
      : undefined
    : models.data?.[0]?.id;
  async function create(event: React.FormEvent) {
    event.preventDefault();
    if (!chosen) return;
    const confidence = Number(threshold);
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      setError("Confidence threshold must be between 0 and 1.");
      return;
    }
    setBusy(true);
    setError(null);
    setCreated(null);
    const outcome = await submitOnce(() =>
      api.deployments.create(chosen, {
        confidence_threshold: confidence,
        ...(slug.trim() ? { slug: slug.trim() } : {}),
      }),
    );
    if (outcome.kind === "created") {
      setCreated(outcome.value);
      setSelectedId(outcome.value.id);
      if (outcome.value.status !== "running")
        setError("The model could not start. Check its status before creating another deployment.");
    } else {
      setError(outcome.error.message);
      setUncertain(outcome.kind === "uncertain");
    }
    await cache.invalidateQueries({ queryKey: ["deployments", projectId] });
    setBusy(false);
  }
  async function stop() {
    if (!stopping) return;
    setBusy(true);
    setError(null);
    try {
      const row = await api.deployments.stop(stopping.id);
      cache.setQueryData(["deployment", row.id], row);
      if (created?.id === row.id) setCreated(null);
      setStopping(null);
      await cache.invalidateQueries({ queryKey: ["deployments", projectId] });
    } catch (raw) {
      setError(toApiError(raw).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <PageHeader
        title="Deployments"
        description="Run a registered model as a private prediction endpoint."
      />
      {error && (
        <Notice tone="error" className="mb-4">
          {error}
        </Notice>
      )}
      {uncertain && (
        <Notice tone="warn" className="mb-4">
          The response was lost. Check the deployment list before creating another endpoint. A lost
          one-time key cannot be recovered; stop that endpoint and create another if needed.
          <Button className="mt-2" onClick={() => setUncertain(false)}>
            I checked the deployment list
          </Button>
        </Notice>
      )}
      <Card title="Deploy a model">
        {models.isPending ? (
          <LoadingState />
        ) : models.isError ? (
          <ErrorState error={models.error} onRetry={() => models.refetch()} />
        ) : !models.data.length ? (
          <EmptyState title="No models to deploy">Complete a training run first.</EmptyState>
        ) : (
          <form onSubmit={create} className="grid gap-4 md:grid-cols-3">
            <Field label="Model version">
              {(p) => (
                <select
                  {...p}
                  className={inputClass}
                  value={chosen ?? ""}
                  onChange={(e) => setModelId(e.target.value)}
                >
                  {!chosen && (
                    <option value="" disabled>
                      Choose a model
                    </option>
                  )}
                  {models.data.map((m) => (
                    <option key={m.id} value={m.id}>
                      Model v{m.version}
                      {m.is_promoted ? " · preferred" : ""}
                    </option>
                  ))}
                </select>
              )}
            </Field>
            <Field label="Endpoint name (optional)">
              {(p) => (
                <input
                  {...p}
                  className={inputClass}
                  value={slug}
                  onChange={(e) => setSlug(e.target.value)}
                  pattern="[a-z][a-z0-9-]{0,62}"
                  placeholder="casting-line"
                />
              )}
            </Field>
            <Field label="Review confidence threshold">
              {(p) => (
                <input
                  {...p}
                  type="number"
                  min="0"
                  max="1"
                  step="0.01"
                  required
                  className={inputClass}
                  value={threshold}
                  onChange={(e) => setThreshold(e.target.value)}
                />
              )}
            </Field>
            {modelId && !chosen && !models.isFetching && (
              <Notice tone="error">
                The requested model is unavailable. Choose a model from this project.
              </Notice>
            )}
            <Button type="submit" variant="primary" busy={busy} disabled={uncertain || !chosen}>
              Deploy model
            </Button>
          </form>
        )}
      </Card>
      {created?.api_key && (
        <Notice tone="success" className="mt-4">
          <p className="font-semibold">
            Save this API key now. It is shown only for this creation.
          </p>
          <Field label="New endpoint API key">
            {(p) => (
              <input
                {...p}
                type="password"
                readOnly
                autoComplete="off"
                className={inputClass}
                value={created.api_key!}
              />
            )}
          </Field>
          <Button
            className="mt-2"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(created.api_key!);
              } catch {
                setError("Clipboard unavailable. Select and copy the key from the field.");
              }
            }}
          >
            Copy API key
          </Button>
        </Notice>
      )}
      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <Card title="Project endpoints">
          {deployments.isPending ? (
            <LoadingState />
          ) : deployments.isError ? (
            <ErrorState error={deployments.error} onRetry={() => deployments.refetch()} />
          ) : !deployments.data.length ? (
            <EmptyState title="No endpoints yet" />
          ) : (
            <ul className="flex flex-col gap-3">
              {deployments.data.map((d) => (
                <li key={d.id} className="rounded border border-line p-3">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <Button onClick={() => setSelectedId(d.id)}>{d.slug}</Button>
                    <span>
                      Model v{d.model_version} ·{" "}
                      {selected.data?.id === d.id ? selected.data.status : d.status}
                    </span>
                  </div>
                  <div className="mt-2 flex gap-4 text-sm">
                    <Link
                      className="text-accent underline"
                      href={`/projects/${projectId}/monitoring?deployment=${d.id}`}
                    >
                      Monitor
                    </Link>
                    {d.status !== "stopped" && (
                      <Button variant="danger" size="sm" onClick={() => setStopping(d)}>
                        Stop endpoint
                      </Button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
        {selectedId &&
          (selected.isPending ? (
            <LoadingState />
          ) : selected.isError ? (
            <ErrorState error={selected.error} onRetry={() => selected.refetch()} />
          ) : (
            <div className="flex flex-col gap-4">
              <Card title="Endpoint details">
                <p className="break-all font-mono text-sm">
                  {selected.data.endpoint_url ?? "No active endpoint URL"}
                </p>
                <p className="mt-2">Status: {selected.data.status}</p>
                <p className="text-sm text-muted">
                  Review threshold: {selected.data.confidence_threshold}. Stopping retains recorded
                  prediction history.
                </p>
              </Card>
              <PredictionTester
                key={selectedId + (created?.id === selectedId ? "-new" : "")}
                deployment={selected.data}
                initialKey={created?.id === selectedId ? (created.api_key ?? "") : ""}
              />
            </div>
          ))}
      </div>
      <Dialog
        open={!!stopping}
        onClose={() => !busy && setStopping(null)}
        title="Stop this endpoint?"
        footer={
          <>
            <Button disabled={busy} onClick={() => setStopping(null)}>
              Keep running
            </Button>
            <Button variant="danger" busy={busy} onClick={stop}>
              Stop endpoint
            </Button>
          </>
        }
      >
        <p>
          New predictions to {stopping?.slug} will stop working. Saved prediction history and model
          artifacts remain available.
        </p>
      </Dialog>
    </>
  );
}
