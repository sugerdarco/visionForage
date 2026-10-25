"use client";

import { ProjectFrame } from "@/features/projects/project-frame";

export default function ProjectLayout({ children }: { children: React.ReactNode }) {
  return <ProjectFrame>{children}</ProjectFrame>;
}
