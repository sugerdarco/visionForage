"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useProjectId } from "@/features/projects/use-project";

export default function ProjectIndex() {
  const projectId = useProjectId();
  const router = useRouter();
  useEffect(() => router.replace(`/projects/${projectId}/images`), [projectId, router]);
  return null;
}
