"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api/endpoints";
import type { Deployment } from "@/lib/api/types";
export function useDeployments(projectId: string) {
  return useQuery({
    queryKey: ["deployments", projectId],
    queryFn: async ({ signal }) => {
      const items: Deployment[] = [];
      for (let page = 1; ; page++) {
        const result = await api.deployments.list(projectId, page, signal);
        items.push(...result.items);
        if (!result.items.length || page * result.page_size >= result.total) return items;
      }
    },
    refetchInterval: 10000,
  });
}
