"use client";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, PageHeader } from "@/components/ui/page-header";
import { Pagination } from "@/components/ui/pagination";
import { EmptyState, ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { useProject, useProjectId } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import type { Prediction, ReviewAccept } from "@/lib/api/types";
import { ReviewEditor } from "./review-editor";

export function ReviewScreen() {
  const projectId = useProjectId();
  const project = useProject(projectId);
  const cache = useQueryClient();
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accepted, setAccepted] = useState(false);
  const queue = useQuery({
    queryKey: ["review-queue", projectId, page],
    queryFn: ({ signal }) => api.reviews.list(projectId, page, signal),
    refetchInterval: selected ? false : 15000,
  });
  async function accept(body: ReviewAccept) {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      await api.reviews.accept(selected.id, body);
      setSelected(null);
      if (page > 1 && queue.data?.items.length === 1) setPage(page - 1);
      setAccepted(true);
      await Promise.all([
        cache.invalidateQueries({ queryKey: ["review-queue", projectId] }),
        cache.invalidateQueries({ queryKey: ["images", projectId] }),
        cache.invalidateQueries({ queryKey: ["prediction-history"] }),
      ]);
    } catch (raw) {
      setError(toApiError(raw).message);
    } finally {
      setBusy(false);
    }
  }
  async function refreshPreview() {
    const result = await queue.refetch();
    const current = result.data?.items.find((p) => p.id === selected?.id);
    if (current) setSelected(current);
    else
      setError(
        "This prediction is no longer on this queue page. Return to the queue to check its state.",
      );
  }
  return (
    <>
      <PageHeader
        title="Review predictions"
        description="Correct low-confidence images before adding them to the next training dataset."
      />
      {accepted && (
        <Notice tone="success" className="mb-4">
          Reviewed image accepted.{" "}
          <Link className="underline" href={`/projects/${projectId}/versions`}>
            Create a new dataset version
          </Link>{" "}
          to include the correction in training.
        </Notice>
      )}
      {selected && project.data ? (
        <Card title="Correct and accept">
          <ReviewEditor
            key={selected.id}
            prediction={selected}
            project={project.data}
            onAccept={accept}
            onRefresh={refreshPreview}
            busy={busy}
            error={error}
          />
          <Button
            className="mt-4"
            disabled={busy}
            onClick={() => {
              if (window.confirm("Leave this review and discard any unaccepted corrections?")) {
                setSelected(null);
                setError(null);
              }
            }}
          >
            Back to review queue
          </Button>
        </Card>
      ) : queue.isPending ? (
        <LoadingState />
      ) : queue.isError ? (
        <ErrorState error={queue.error} onRetry={() => queue.refetch()} />
      ) : queue.data.total === 0 ? (
        <EmptyState title="No predictions awaiting review">
          New low-confidence predictions appear after background collection. Accepted predictions
          remain in monitoring history.
        </EmptyState>
      ) : (
        <>
          <p className="mb-3 text-sm text-muted">
            {queue.data.total} awaiting review, lowest confidence first.
          </p>
          <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {queue.data.items.map((item) => (
              <li key={item.id} className="rounded border border-line bg-panel p-4">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={item.image_url}
                  alt="Prediction awaiting review"
                  referrerPolicy="no-referrer"
                  className="h-40 w-full rounded bg-slate-100 object-contain"
                />
                <p className="mt-2 font-medium">
                  {"label" in item.output ? item.output.label : `${item.output.boxes.length} boxes`}{" "}
                  · {(item.max_confidence * 100).toFixed(1)}%
                </p>
                <p className="text-sm text-muted">Model v{item.model_version}</p>
                <Button
                  className="mt-3"
                  onClick={() => {
                    setSelected(item);
                    setError(null);
                    setAccepted(false);
                  }}
                >
                  Review image
                </Button>
              </li>
            ))}
          </ul>
          <Pagination
            page={page}
            pageSize={20}
            total={queue.data.total}
            label="predictions awaiting review"
            onPage={setPage}
          />
        </>
      )}
    </>
  );
}
