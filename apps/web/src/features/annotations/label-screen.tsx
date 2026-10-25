"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { filterQuery, readFilters } from "@/features/images/gallery-filters";
import { STATUS_BADGE } from "@/features/images/images-screen";
import { useProject } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import { shortId } from "@/lib/format";
import { ClassificationEditor } from "./classification-editor";
import { DetectionEditor } from "./detection-editor";
import type { EditorProps, Guard, Navigator } from "./editor-common";
import { LABEL_QUEUE_PAGE_SIZE, useLabelQueue } from "./use-label-queue";

export function LabelScreen() {
  const { projectId, imageId } = useParams<{ projectId: string; imageId: string }>();
  const params = useSearchParams();
  const router = useRouter();
  const filters = readFilters(new URLSearchParams(params.toString()));
  const query = filterQuery(filters);
  const project = useProject(projectId);
  const queue = useLabelQueue(projectId, filters, imageId);
  const [guard, setGuard] = useState<Guard | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [leaving, setLeaving] = useState(false);
  const [endOfQueue, setEndOfQueue] = useState(false);

  const image = useQuery({
    queryKey: ["image", imageId],
    queryFn: ({ signal }) => api.images.get(imageId, signal),
    staleTime: 60_000,
  });
  const annotations = useQuery({
    queryKey: ["annotations", imageId],
    queryFn: ({ signal }) => api.images.annotations(imageId, signal),
    // Seed the editor from the latest server state when an image is opened.
    staleTime: 0,
    refetchOnMount: "always",
  });

  const leave = useCallback(
    async (href: string) => {
      setLeaving(true);
      try {
        if (guard) {
          await guard.flush();
          if (guard.isDirty()) {
            setPending(href);
            return;
          }
        }
        router.push(href);
      } finally {
        setLeaving(false);
      }
    },
    [guard, router],
  );

  const galleryHref = `/projects/${projectId}/images${query}`;
  const hrefFor = (id: string) => `/projects/${projectId}/label/${id}${query}`;
  const nav: Navigator = {
    leave: (href) => void leave(href),
    goPrevious: queue.previous ? () => void leave(hrefFor(queue.previous!)) : null,
    goNext: () => {
      void (async () => {
        const next = await queue.next();
        if (next) {
          setEndOfQueue(false);
          await leave(hrefFor(next));
        } else setEndOfQueue(true);
      })();
    },
    position:
      queue.index >= 0 && queue.total !== null
        ? `${(filters.page - 1) * LABEL_QUEUE_PAGE_SIZE + queue.index + 1} of ${queue.total}`
        : "",
  };

  if (project.isPending || image.isPending || annotations.isPending) {
    return (
      <div className="p-6">
        <LoadingState label="Loading image" />
      </div>
    );
  }
  const error = project.error ?? image.error ?? annotations.error;
  if (error || !project.data || !image.data || !annotations.data) {
    return (
      <div className="mx-auto max-w-3xl p-6">
        <ErrorState
          error={error}
          onRetry={() => {
            void image.refetch();
            void annotations.refetch();
          }}
        />
        <Link href={galleryHref} className="mt-4 inline-block text-sm text-accent underline">
          Back to images
        </Link>
      </div>
    );
  }

  const editorProps: EditorProps = {
    project: project.data,
    image: image.data,
    annotations: annotations.data,
    nav,
    onImageExpired: () => void image.refetch(),
    registerGuard: setGuard,
  };

  return (
    <main id="main" className="flex h-[calc(100vh-3rem)] min-h-[560px] flex-col">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-panel px-4 py-2">
        <Button size="sm" variant="ghost" onClick={() => nav.leave(galleryHref)} busy={leaving}>
          ← Images
        </Button>
        <span className="text-sm font-medium">{project.data.name}</span>
        <span className="text-sm text-muted">Image {shortId(image.data.id)}</span>
        <Badge tone={STATUS_BADGE[annotations.data.status].tone}>
          Saved status: {STATUS_BADGE[annotations.data.status].label}
        </Badge>
        <span className="text-xs text-muted">
          {image.data.width} × {image.data.height} px
        </span>
        <div className="ml-auto flex items-center gap-2">
          {nav.position && <span className="text-sm text-muted">{nav.position}</span>}
          <Button
            size="sm"
            disabled={!nav.goPrevious}
            onClick={() => nav.goPrevious?.()}
            title="Previous image ([)"
          >
            Previous
          </Button>
          <Button size="sm" onClick={nav.goNext} title="Next image (])">
            Next
          </Button>
        </div>
      </div>
      {endOfQueue && (
        <Notice tone="info" className="m-3">
          No more images in this selection.{" "}
          <button className="underline" onClick={() => nav.leave(galleryHref)}>
            Back to images
          </button>
        </Notice>
      )}
      {annotations.data.status === "in_review" && (
        <Notice tone="warn" className="m-3">
          This image is in review with model suggestions. Saving replaces them with your labels.
        </Notice>
      )}
      <div className="min-h-0 flex-1">
        {project.data.task_type === "detection" ? (
          <DetectionEditor key={image.data.id} {...editorProps} />
        ) : (
          <ClassificationEditor key={image.data.id} {...editorProps} />
        )}
      </div>
      <Dialog
        open={pending !== null}
        onClose={() => setPending(null)}
        title="Unsaved changes"
        footer={
          <>
            <Button variant="primary" onClick={() => setPending(null)}>
              Stay and retry
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                const href = pending!;
                setPending(null);
                router.push(href);
              }}
            >
              Discard and leave
            </Button>
          </>
        }
      >
        The latest labels for this image could not be saved. If you leave now, those changes are
        lost.
      </Dialog>
    </main>
  );
}
