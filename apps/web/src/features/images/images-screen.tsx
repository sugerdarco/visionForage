"use client";

import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { inputClass } from "@/components/ui/field";
import { Card, PageHeader } from "@/components/ui/page-header";
import { Pagination } from "@/components/ui/pagination";
import { EmptyState, ErrorState, LoadingState } from "@/components/ui/states";
import { useProject, useProjectId } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import type { Image, ImageStatus, ProjectDetail, TaskType } from "@/lib/api/types";
import { formatDateTime, shortId } from "@/lib/format";
import { filterQuery, labelQueueKey, readFilters, type GalleryFilters } from "./gallery-filters";
import { Importer } from "./importer";
import { Uploader } from "./uploader";

const PAGE_SIZE = 24;
const THUMBNAIL_POLLS = 20;

export const STATUS_BADGE: Record<
  ImageStatus,
  { label: string; tone: "neutral" | "green" | "amber" }
> = {
  unlabeled: { label: "Unlabeled", tone: "neutral" },
  labeled: { label: "Labeled", tone: "green" },
  in_review: { label: "In review", tone: "amber" },
};

export function ImagesScreen() {
  const projectId = useProjectId();
  const project = useProject(projectId);
  const [panel, setPanel] = useState<"upload" | "import" | null>(null);
  if (!project.data) return null;
  const task = project.data.task_type as TaskType;
  return (
    <>
      <PageHeader
        title="Images"
        description={
          task === "classification"
            ? "Give every image exactly one class. Even acceptable parts need a class such as “ok”."
            : "Draw a box around every defect. Images you checked and found clean can be marked as reviewed negatives."
        }
        actions={
          <>
            <Button
              variant={panel === "upload" ? "primary" : "secondary"}
              aria-expanded={panel === "upload"}
              onClick={() => setPanel(panel === "upload" ? null : "upload")}
            >
              Upload images
            </Button>
            <Button
              variant={panel === "import" ? "primary" : "secondary"}
              aria-expanded={panel === "import"}
              onClick={() => setPanel(panel === "import" ? null : "import")}
            >
              Import labeled ZIP
            </Button>
          </>
        }
      />
      {panel === "upload" && (
        <Card title="Upload images" className="mb-6">
          <Uploader projectId={projectId} />
        </Card>
      )}
      {panel === "import" && (
        <Card title="Import a labeled dataset" className="mb-6">
          <Importer projectId={projectId} task={task} />
        </Card>
      )}
      <Gallery project={project.data} />
    </>
  );
}

function Gallery({ project }: { project: ProjectDetail }) {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const filters = readFilters(new URLSearchParams(params.toString()));
  const polls = useRef(0);
  const [detail, setDetail] = useState<string | null>(null);
  const queryClient = useQueryClient();

  // Starting to label from the gallery always takes a fresh queue snapshot.
  useEffect(() => {
    queryClient.removeQueries({ queryKey: labelQueueKey(project.id) });
  }, [queryClient, project.id]);

  const images = useQuery({
    queryKey: ["images", project.id, filters.status, filters.classId, filters.page],
    queryFn: ({ signal }) =>
      api.images.list(
        project.id,
        {
          page: filters.page,
          pageSize: PAGE_SIZE,
          status: filters.status,
          classId: filters.classId,
        },
        signal,
      ),
    placeholderData: keepPreviousData,
    // Signed URLs last 300 s: refresh before they expire, and poll briefly for pending thumbnails.
    refetchInterval: (query) => {
      const pending = query.state.data?.items.some((i) => !i.thumbnail_url);
      if (pending && polls.current < THUMBNAIL_POLLS) {
        polls.current++;
        return 3000;
      }
      return 240_000;
    },
  });

  useEffect(() => {
    polls.current = 0;
  }, [filters.status, filters.classId, filters.page]);

  const setFilters = (next: Partial<GalleryFilters>) =>
    router.replace(`${pathname}${filterQuery({ ...filters, page: 1, ...next })}`, {
      scroll: false,
    });

  const lastRefresh = useRef(0);
  const onImageError = () => {
    // Most likely an expired signed URL: fetch fresh metadata (at most every 10 s).
    if (Date.now() - lastRefresh.current > 10_000) {
      lastRefresh.current = Date.now();
      void images.refetch();
    }
  };

  const labelQuery = filterQuery(filters);

  return (
    <section aria-label="Gallery" className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium">Status</span>
          <select
            className={inputClass}
            value={filters.status ?? ""}
            onChange={(e) => setFilters({ status: (e.target.value || null) as ImageStatus | null })}
          >
            <option value="">All statuses</option>
            <option value="unlabeled">Unlabeled</option>
            <option value="labeled">Labeled</option>
            <option value="in_review">In review</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm">
          <span className="font-medium">Class</span>
          <select
            className={inputClass}
            value={filters.classId ?? ""}
            onChange={(e) => setFilters({ classId: e.target.value || null })}
          >
            <option value="">All classes</option>
            {project.classes.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        {images.data && images.data.items.length > 0 && (
          <Link
            className="ml-auto rounded-md bg-accent px-3.5 py-2 text-sm font-medium text-white hover:bg-accent-strong"
            href={`/projects/${project.id}/label/${images.data.items[0]!.id}${labelQuery}`}
          >
            Start labeling these
          </Link>
        )}
      </div>

      {images.isPending ? (
        <LoadingState label="Loading images" />
      ) : images.isError ? (
        <ErrorState error={images.error} onRetry={() => images.refetch()} />
      ) : images.data.total === 0 ? (
        <EmptyState
          title={
            filters.status || filters.classId ? "No images match these filters" : "No images yet"
          }
        >
          {filters.status || filters.classId
            ? "Change or clear the filters above."
            : "Upload JPEG/PNG images or import a labeled ZIP to get started."}
        </EmptyState>
      ) : (
        <>
          <ul className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
            {images.data.items.map((image) => (
              <li key={image.id} className="overflow-hidden rounded-lg border border-line bg-white">
                <Link
                  href={`/projects/${project.id}/label/${image.id}${labelQuery}`}
                  className="block"
                  aria-label={`Label image ${shortId(image.id)} (${STATUS_BADGE[image.status].label})`}
                >
                  <Thumbnail image={image} onError={onImageError} />
                </Link>
                <div className="flex items-center justify-between gap-2 px-2 py-1.5">
                  <Badge tone={STATUS_BADGE[image.status].tone}>
                    {STATUS_BADGE[image.status].label}
                  </Badge>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setDetail(image.id)}
                    aria-label={`Details for image ${shortId(image.id)}`}
                  >
                    Details
                  </Button>
                </div>
              </li>
            ))}
          </ul>
          <Pagination
            page={filters.page}
            pageSize={PAGE_SIZE}
            total={images.data.total}
            label="images"
            onPage={(page) => setFilters({ page })}
          />
        </>
      )}
      <ImageDetailDialog
        projectId={project.id}
        imageId={detail}
        labelQuery={labelQuery}
        onClose={() => setDetail(null)}
      />
    </section>
  );
}

function Thumbnail({ image, onError }: { image: Image; onError: () => void }) {
  if (!image.thumbnail_url) {
    return (
      <div className="flex aspect-square items-center justify-center bg-slate-100 text-xs text-muted">
        Preview pending
      </div>
    );
  }
  return (
    // Signed, short-lived URLs from private storage; next/image optimization does not apply.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={image.thumbnail_url}
      alt=""
      loading="lazy"
      onError={onError}
      className="aspect-square w-full bg-slate-100 object-contain"
    />
  );
}

function ImageDetailDialog({
  projectId,
  imageId,
  labelQuery,
  onClose,
}: {
  projectId: string;
  imageId: string | null;
  labelQuery: string;
  onClose: () => void;
}) {
  const image = useQuery({
    queryKey: ["image", imageId],
    queryFn: ({ signal }) => api.images.get(imageId!, signal),
    enabled: !!imageId,
    staleTime: 0,
  });
  return (
    <Dialog
      open={!!imageId}
      onClose={onClose}
      title="Image details"
      footer={
        <>
          <Button onClick={onClose}>Close</Button>
          {imageId && (
            <Link
              href={`/projects/${projectId}/label/${imageId}${labelQuery}`}
              className="rounded-md bg-accent px-3.5 py-2 text-sm font-medium text-white"
            >
              Open in labeler
            </Link>
          )}
        </>
      }
    >
      {image.isPending ? (
        <LoadingState />
      ) : image.isError ? (
        <ErrorState error={image.error} onRetry={() => image.refetch()} />
      ) : (
        <div className="flex flex-col gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={image.data.image_url}
            alt="Original upload"
            onError={() => void image.refetch()}
            className="max-h-80 w-full rounded bg-slate-100 object-contain"
          />
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-muted">Size</dt>
            <dd>
              {image.data.width} × {image.data.height} px
            </dd>
            <dt className="text-muted">Status</dt>
            <dd>{STATUS_BADGE[image.data.status].label}</dd>
            <dt className="text-muted">Added</dt>
            <dd>{formatDateTime(image.data.created_at)}</dd>
            <dt className="text-muted">SHA-256</dt>
            <dd className="font-mono text-xs break-all">{image.data.sha256}</dd>
          </dl>
        </div>
      )}
    </Dialog>
  );
}
