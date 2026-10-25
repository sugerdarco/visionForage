import type { ImageStatus } from "@/lib/api/types";

export interface GalleryFilters {
  status: ImageStatus | null;
  classId: string | null;
  page: number;
}

const STATUSES: ImageStatus[] = ["unlabeled", "labeled", "in_review"];

export function readFilters(params: URLSearchParams): GalleryFilters {
  const status = params.get("status");
  return {
    status: STATUSES.includes(status as ImageStatus) ? (status as ImageStatus) : null,
    classId: params.get("class") || null,
    page: Math.max(1, Number(params.get("page")) || 1),
  };
}

export function filterQuery(filters: GalleryFilters): string {
  const params = new URLSearchParams();
  if (filters.status) params.set("status", filters.status);
  if (filters.classId) params.set("class", filters.classId);
  if (filters.page > 1) params.set("page", String(filters.page));
  const query = params.toString();
  return query ? `?${query}` : "";
}

/** Cache key for the labeler's previous/next snapshot; the gallery clears it. */
export function labelQueueKey(projectId: string) {
  return ["label-queue", projectId] as const;
}
