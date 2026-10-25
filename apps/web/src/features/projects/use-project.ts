"use client";

import { useQuery } from "@tanstack/react-query";
import { useParams } from "next/navigation";
import { toApiError } from "@/lib/api/errors";
import { api } from "@/lib/api/endpoints";

export function useProjectId(): string {
  const params = useParams<{ projectId: string }>();
  return params.projectId;
}

export function useProject(projectId: string) {
  return useQuery({
    queryKey: ["project", projectId],
    queryFn: ({ signal }) => api.projects.get(projectId, signal),
    refetchInterval: (query) =>
      query.state.data?.deleting && toApiError(query.state.error).kind !== "not_found"
        ? 2000
        : false,
  });
}
