"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Card, PageHeader } from "@/components/ui/page-header";
import { Pagination } from "@/components/ui/pagination";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { useProjectId } from "@/features/projects/use-project";
import { ARCH_LABEL } from "@/features/training/job-status";
import { api } from "@/lib/api/endpoints";
import { ModelActions, ModelResults } from "./model-results";
import { formatDateTime, shortId } from "@/lib/format";

const PAGE_SIZE = 20;

export function ModelsScreen() {
  const projectId = useProjectId();
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const page = Math.max(1, Number(params.get("page")) || 1);
  const selected = params.get("model");
  const models = useQuery({
    queryKey: ["models", projectId, page],
    queryFn: ({ signal }) => api.models.list(projectId, page, PAGE_SIZE, signal),
    placeholderData: keepPreviousData,
  });

  return (
    <>
      <PageHeader
        title="Models"
        description="Every successful training run registers a model version with its frozen class map."
      />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <section aria-label="Registered models" className="flex min-w-0 flex-col gap-4">
          {models.isPending ? (
            <LoadingState label="Loading models" />
          ) : models.isError ? (
            <ErrorState error={models.error} onRetry={() => models.refetch()} />
          ) : models.data.total === 0 ? (
            <EmptyState title="No registered models yet">
              Models appear here after a training run succeeds.
            </EmptyState>
          ) : (
            <>
              <div className="overflow-x-auto rounded-lg border border-line bg-panel">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 text-left text-xs text-muted">
                    <tr>
                      <th className="px-3 py-2 font-medium">Model</th>
                      <th className="px-3 py-2 font-medium">Architecture</th>
                      <th className="px-3 py-2 font-medium">Dataset</th>
                      <th className="px-3 py-2 font-medium">Run</th>
                      <th className="px-3 py-2 font-medium">Registered</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.data.items.map((m) => (
                      <tr
                        key={m.id}
                        className={`border-t border-line ${m.id === selected ? "bg-blue-50" : ""}`}
                      >
                        <td className="px-3 py-2">
                          <Link
                            className="font-medium text-accent underline"
                            href={`${pathname}?model=${m.id}${page > 1 ? `&page=${page}` : ""}`}
                          >
                            Model v{m.version}
                          </Link>{" "}
                          {m.is_promoted && <Badge tone="green">Preferred</Badge>}
                        </td>
                        <td className="px-3 py-2">{ARCH_LABEL[m.model_arch] ?? m.model_arch}</td>
                        <td className="px-3 py-2">
                          <Link
                            className="underline"
                            href={`/projects/${projectId}/versions/${m.dataset_version_id}`}
                          >
                            v{m.dataset_version}
                          </Link>
                        </td>
                        <td className="px-3 py-2">
                          <Link
                            className="font-mono underline"
                            href={`/projects/${projectId}/training/${m.job_id}`}
                          >
                            {shortId(m.job_id)}
                          </Link>
                        </td>
                        <td className="px-3 py-2 whitespace-nowrap">
                          {formatDateTime(m.created_at)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <Pagination
                page={page}
                pageSize={PAGE_SIZE}
                total={models.data.total}
                label="models"
                onPage={(next) => router.push(`${pathname}?page=${next}`)}
              />
            </>
          )}
        </section>
        {selected && <ModelDetailCard modelId={selected} />}
      </div>
      {selected && <ModelResults key={selected} modelId={selected} />}
    </>
  );
}

function ModelDetailCard({ modelId }: { modelId: string }) {
  const model = useQuery({
    queryKey: ["model", modelId],
    queryFn: ({ signal }) => api.models.get(modelId, signal),
  });
  if (model.isPending) return <LoadingState label="Loading model" />;
  if (model.isError) return <ErrorState error={model.error} onRetry={() => model.refetch()} />;
  const m = model.data;
  return (
    <Card title={`Model v${m.version}`} className="h-fit">
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted">Architecture</dt>
        <dd>{ARCH_LABEL[m.model_arch] ?? m.model_arch}</dd>
        <dt className="text-muted">Input size</dt>
        <dd>{m.input_size} px</dd>
        <dt className="text-muted">Dataset version</dt>
        <dd>v{m.dataset_version}</dd>
        <dt className="text-muted">Registered</dt>
        <dd>{formatDateTime(m.created_at)}</dd>
      </dl>
      <ModelActions key={m.id} model={m} />
      <h3 className="mt-4 mb-1 text-sm font-semibold">Frozen class map</h3>
      <table className="w-full text-sm">
        <tbody>
          {Object.entries(m.class_map)
            .sort(([, a], [, b]) => a - b)
            .map(([name, index]) => (
              <tr key={name} className="border-t border-line">
                <td className="py-1 pr-3 font-mono">{index}</td>
                <td className="py-1 break-all">{name}</td>
              </tr>
            ))}
        </tbody>
      </table>
      <h3 className="mt-4 mb-1 text-sm font-semibold">Requested training settings</h3>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 font-mono text-xs">
        {Object.entries(m.hyperparams).map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-muted">{k}</dt>
            <dd>{String(v)}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-muted">
        Early stopping can end training before the requested number of epochs.
      </p>
    </Card>
  );
}
