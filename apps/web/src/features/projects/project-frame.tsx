"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { toApiError } from "@/lib/api/errors";
import { TaskBadge } from "./task-badge";
import { useProject, useProjectId } from "./use-project";

const NAV = [
  { href: "images", label: "Images", hint: "Upload, import and browse" },
  { href: "versions", label: "Dataset versions", hint: "Freeze labels for training" },
  { href: "training", label: "Training", hint: "Runs and live progress" },
  { href: "compare", label: "Compare runs", hint: "Parameters and curves" },
  { href: "models", label: "Models", hint: "Registered results" },
  { href: "settings", label: "Settings", hint: "Name and classes" },
];

export function ProjectFrame({ children }: { children: React.ReactNode }) {
  const projectId = useProjectId();
  const pathname = usePathname();
  const project = useProject(projectId);
  const base = `/projects/${projectId}`;
  // The labeling workspace uses the full width.
  const fullScreen = pathname.startsWith(`${base}/label/`);

  if (project.isPending) {
    return (
      <div className="p-6">
        <LoadingState label="Loading project" />
      </div>
    );
  }
  if (project.isError && project.data?.deleting && toApiError(project.error).kind === "not_found") {
    return (
      <main id="main" className="mx-auto w-full max-w-3xl p-6">
        <h1 className="mb-4 text-xl font-semibold">Project deleted</h1>
        <Link href="/projects" className="text-accent underline">
          Back to projects
        </Link>
      </main>
    );
  }
  if (project.isError) {
    return (
      <div className="mx-auto max-w-3xl p-6">
        <ErrorState error={project.error} onRetry={() => project.refetch()} />
        <Link href="/projects" className="mt-4 inline-block text-sm text-accent underline">
          Back to projects
        </Link>
      </div>
    );
  }
  if (project.data.deleting) {
    return (
      <main id="main" className="mx-auto w-full max-w-3xl p-6">
        <h1 className="mb-4 text-xl font-semibold">Deleting project</h1>
        <Notice>
          “{project.data.name}” is being removed. Running jobs must stop before cleanup finishes.
        </Notice>
        <p className="mt-3 text-sm text-muted">
          You can leave this page. Cleanup continues in the background.
        </p>
        <Link href="/projects" className="mt-4 inline-block text-accent underline">
          Back to projects
        </Link>
      </main>
    );
  }
  if (fullScreen) return <>{children}</>;

  return (
    <div className="flex flex-1 flex-col md:flex-row">
      <aside className="border-b border-line bg-panel md:w-60 md:shrink-0 md:border-r md:border-b-0">
        <div className="p-4">
          <nav aria-label="Breadcrumb" className="text-xs text-muted">
            <Link href="/projects" className="hover:underline">
              Projects
            </Link>{" "}
            /
          </nav>
          <p className="mt-1 font-semibold break-words" title={project.data.name}>
            {project.data.name}
          </p>
          <div className="mt-2">
            <TaskBadge task={project.data.task_type} />
          </div>
        </div>
        <nav
          aria-label="Project"
          className="flex gap-1 overflow-x-auto px-2 pb-3 md:flex-col md:overflow-visible"
        >
          {NAV.map((item) => {
            const href = `${base}/${item.href}`;
            const active = pathname === href || pathname.startsWith(`${href}/`);
            return (
              <Link
                key={item.href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={`shrink-0 rounded-md px-3 py-2 text-sm ${
                  active
                    ? "bg-blue-50 font-medium text-accent-strong"
                    : "text-ink hover:bg-slate-100"
                }`}
              >
                {item.label}
                <span className="hidden text-xs font-normal text-muted md:block">{item.hint}</span>
              </Link>
            );
          })}
        </nav>
      </aside>
      <main id="main" className="min-w-0 flex-1 px-4 py-6 md:px-8">
        {children}
      </main>
    </div>
  );
}
