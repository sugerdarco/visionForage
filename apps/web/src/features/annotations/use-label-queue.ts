"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef } from "react";
import { labelQueueKey, type GalleryFilters } from "@/features/images/gallery-filters";
import { api } from "@/lib/api/endpoints";

export const LABEL_QUEUE_PAGE_SIZE = 24;

interface QueueSnapshot {
  ids: string[];
  total: number;
  lastPage: number;
}

/**
 * Previous/next navigation over the gallery selection the user came from. The list
 * is a snapshot kept in the query cache for the whole labeling session, so labeling
 * an image (which can remove it from an "unlabeled" filter) does not shift the queue.
 * The gallery clears it, so starting again from the gallery takes a fresh snapshot.
 */
export function useLabelQueue(projectId: string, filters: GalleryFilters, imageId: string) {
  const queryClient = useQueryClient();
  const key = [...labelQueueKey(projectId), filters.status, filters.classId, filters.page] as const;
  const loading = useRef(false);
  const list = (page: number) =>
    api.images.list(projectId, {
      page,
      pageSize: LABEL_QUEUE_PAGE_SIZE,
      status: filters.status,
      classId: filters.classId,
    });

  const snapshot = useQuery({
    queryKey: key,
    queryFn: async (): Promise<QueueSnapshot> => {
      const page = await list(filters.page);
      return { ids: page.items.map((i) => i.id), total: page.total, lastPage: filters.page };
    },
    staleTime: Infinity,
    gcTime: 30 * 60_000,
  });

  const ids = snapshot.data?.ids ?? [];
  const index = ids.indexOf(imageId);

  const next = async (): Promise<string | null> => {
    const current = queryClient.getQueryData<QueueSnapshot>(key);
    if (!current) return null;
    const at = current.ids.indexOf(imageId);
    if (at >= 0 && at < current.ids.length - 1) return current.ids[at + 1]!;
    if (loading.current) return null;
    loading.current = true;
    try {
      // With the "unlabeled" filter, labeled images leave the list, so re-read the
      // same page; otherwise continue with the following page.
      const page = filters.status === "unlabeled" ? current.lastPage : current.lastPage + 1;
      const result = await list(page);
      const fresh = result.items.map((i) => i.id).filter((id) => !current.ids.includes(id));
      if (fresh.length === 0) return null;
      queryClient.setQueryData<QueueSnapshot>(key, {
        ...current,
        ids: [...current.ids, ...fresh],
        lastPage: page,
      });
      return fresh[0]!;
    } catch {
      return null;
    } finally {
      loading.current = false;
    }
  };

  const previous = index > 0 ? ids[index - 1]! : null;
  return { index, total: snapshot.data?.total ?? null, previous, next };
}
