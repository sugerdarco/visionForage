"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { PageHeader } from "@/components/ui/page-header";
import { Pagination } from "@/components/ui/pagination";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { useProjectId } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import { isTerminal } from "@/lib/api/types";
import { formatDateTime, formatDuration, shortId } from "@/lib/format";
import { ARCH_LABEL, JobStatusBadge } from "./job-status";
import { useVersionIndex } from "./use-version-index";

const PAGE_SIZE = 20;

export function JobsScreen() {
  const projectId = useProjectId();
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const page = Math.max(1, Number(params.get("page")) || 1);
  const versions = useVersionIndex(projectId);
  const jobs = useQuery({
    queryKey: ["jobs", projectId, page],
    queryFn: ({ signal }) => api.jobs.list(projectId, page, PAGE_SIZE, signal),
    placeholderData: keepPreviousData,
    // Poll while any visible job is still active; detail pages stream instead.
    refetchInterval: (query) =>
      query.state.data?.items.some((j) => !isTerminal(j.status)) ? 5000 : false,
  });
  const newRun = (
    <Link
      href={`/projects/${projectId}/training/new`}
      className="rounded-md bg-accent px-3.5 py-2 text-sm font-medium text-white hover:bg-accent-strong"
    >
      New training run
    </Link>
  );

  return (
    <>
      <PageHeader
        title="Training"
        description="Each run trains one model on one frozen dataset version. Successful runs register a model."
        actions={
          <>
            <Link
              href={`/projects/${projectId}/compare`}
              className="rounded-md border border-line bg-white px-3.5 py-2 text-sm font-medium hover:bg-slate-50"
            >
              Compare runs
            </Link>
            {newRun}
          </>
        }
      />
      {jobs.isPending ? (
        <LoadingState label="Loading training runs" />
      ) : jobs.isError ? (
        <ErrorState error={jobs.error} onRetry={() => jobs.refetch()} />
      ) : jobs.data.total === 0 ? (
        <EmptyState title="No training runs yet" action={newRun}>
          Create a dataset version with at least 32 test images, then start a run.
        </EmptyState>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="overflow-x-auto rounded-lg border border-line bg-panel">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 text-left text-xs text-muted">
                <tr>
                  <th className="px-3 py-2 font-medium">Run</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3 py-2 font-medium">Model</th>
                  <th className="px-3 py-2 font-medium">Dataset</th>
                  <th className="px-3 py-2 font-medium">Created</th>
                  <th className="px-3 py-2 font-medium">Duration</th>
                </tr>
              </thead>
              <tbody>
                {jobs.data.items.map((job) => {
                  const version = versions.data?.byId.get(job.dataset_version_id);
                  return (
                    <tr key={job.id} className="border-t border-line">
                      <td className="px-3 py-2">
                        <Link
                          className="font-mono text-accent underline"
                          href={`/projects/${projectId}/training/${job.id}`}
                        >
                          {shortId(job.id)}
                        </Link>
                      </td>
                      <td className="px-3 py-2">
                        <JobStatusBadge job={job} />
                      </td>
                      <td className="px-3 py-2">{ARCH_LABEL[job.model_arch] ?? job.model_arch}</td>
                      <td className="px-3 py-2">
                        {version ? `v${version.version}` : shortId(job.dataset_version_id)}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        {formatDateTime(job.created_at)}
                      </td>
                      <td className="px-3 py-2 whitespace-nowrap">
                        {job.started_at ? formatDuration(job.started_at, job.finished_at) : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <Pagination
            page={page}
            pageSize={PAGE_SIZE}
            total={jobs.data.total}
            label="runs"
            onPage={(next) => router.push(`${pathname}?page=${next}`)}
          />
        </div>
      )}
    </>
  );
}
