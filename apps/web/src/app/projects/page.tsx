"use client";

import { Suspense } from "react";
import { ProjectDashboard } from "@/features/projects/project-dashboard";

export default function ProjectsPage() {
  return (
    <Suspense>
      <ProjectDashboard />
    </Suspense>
  );
}
