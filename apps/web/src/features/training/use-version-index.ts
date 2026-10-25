"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api/endpoints";

/** Version numbers by ID for labeling jobs (first 100 versions, newest first). */
export function useVersionIndex(projectId: string) {
  return useQuery({
    queryKey: ["versions", projectId, "index"],
    queryFn: ({ signal }) => api.versions.list(projectId, 1, 100, signal),
    select: (page) => ({
      items: page.items,
      total: page.total,
      byId: new Map(page.items.map((v) => [v.id, v])),
    }),
  });
}
