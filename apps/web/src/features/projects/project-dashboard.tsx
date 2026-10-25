"use client";

import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Field, inputClass } from "@/components/ui/field";
import { PageHeader } from "@/components/ui/page-header";
import { Pagination } from "@/components/ui/pagination";
import { EmptyState, ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import type { TaskType } from "@/lib/api/types";
import { formatDateTime } from "@/lib/format";
import {
  classDraftErrors,
  ClassListEditor,
  draftClass,
  type DraftClass,
} from "./class-list-editor";
import { TaskBadge } from "./task-badge";

const PAGE_SIZE = 20;

export function ProjectDashboard() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const page = Math.max(1, Number(params.get("page")) || 1);
  const [creating, setCreating] = useState(false);
  const projects = useQuery({
    queryKey: ["projects", page],
    queryFn: ({ signal }) => api.projects.list(page, PAGE_SIZE, signal),
    placeholderData: keepPreviousData,
    refetchInterval: (query) =>
      query.state.data?.items.some((project) => project.deleting) ? 2000 : false,
  });

  return (
    <main id="main" className="mx-auto w-full max-w-6xl px-4 py-8">
      <PageHeader
        title="Projects"
        description="Each project holds one inspection task: its images, labels, frozen datasets and training runs."
        actions={
          <Button variant="primary" onClick={() => setCreating(true)}>
            New project
          </Button>
        }
      />
      {projects.isPending ? (
        <LoadingState label="Loading projects" />
      ) : projects.isError ? (
        <ErrorState error={projects.error} onRetry={() => projects.refetch()} />
      ) : projects.data.total === 0 ? (
        <EmptyState
          title="No projects yet"
          action={
            <Button variant="primary" onClick={() => setCreating(true)}>
              Create your first project
            </Button>
          }
        >
          Start with a classification project (one label per image, e.g. OK vs defective) or a
          detection project (boxes around each defect).
        </EmptyState>
      ) : (
        <div className="flex flex-col gap-4">
          <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {projects.data.items.map((project) => (
              <li key={project.id}>
                <Link
                  href={`/projects/${project.id}/images`}
                  className="block h-full rounded-lg border border-line bg-panel p-4 transition hover:border-accent hover:shadow-sm"
                >
                  <div className="flex items-start justify-between gap-2">
                    <h2 className="font-semibold break-words">{project.name}</h2>
                    <TaskBadge task={project.task_type} />
                  </div>
                  <p className="mt-3 text-xs text-muted">
                    {project.deleting
                      ? "Deleting…"
                      : `Created ${formatDateTime(project.created_at)}`}
                  </p>
                </Link>
              </li>
            ))}
          </ul>
          <Pagination
            page={page}
            pageSize={PAGE_SIZE}
            total={projects.data.total}
            label="projects"
            onPage={(next) => router.push(`${pathname}?page=${next}`)}
          />
        </div>
      )}
      <CreateProjectDialog open={creating} onClose={() => setCreating(false)} />
    </main>
  );
}

function CreateProjectDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");
  const [task, setTask] = useState<TaskType>("classification");
  const [drafts, setDrafts] = useState<DraftClass[]>(() => [draftClass([])]);
  const [errors, setErrors] = useState<{ name?: string; classes?: Record<number, string> }>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const classErrors = classDraftErrors(drafts);
    const next = {
      name: name.trim()
        ? name.trim().length > 200
          ? "Use at most 200 characters."
          : undefined
        : "Enter a project name.",
      classes: classErrors,
    };
    setErrors(next);
    setFormError(null);
    if (next.name || Object.keys(classErrors).length) return;
    setBusy(true);
    try {
      const project = await api.projects.create({
        name: name.trim(),
        task_type: task,
        classes: drafts.map((d) => ({ name: d.name.trim(), color: d.color.toLowerCase() })),
      });
      await queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.setQueryData(["project", project.id], project);
      onClose();
      router.push(`/projects/${project.id}/images`);
    } catch (raw) {
      const error = toApiError(raw);
      if (error.kind === "validation") {
        const fields = error.fieldErrors();
        setErrors({ ...next, name: fields.name });
        setFormError(fields.name ? null : error.message);
      } else setFormError(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog open={open} onClose={onClose} title="New project">
      <form noValidate onSubmit={submit} className="flex flex-col gap-4">
        <Field label="Project name" error={errors.name}>
          {(props) => (
            <input
              {...props}
              className={inputClass}
              value={name}
              maxLength={200}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Casting line 3"
            />
          )}
        </Field>
        <fieldset className="flex flex-col gap-2">
          <legend className="mb-1 text-sm font-medium">Task</legend>
          <TaskOption
            value="classification"
            current={task}
            onSelect={setTask}
            title="Classification"
            description="One label per image, such as OK or defective."
          />
          <TaskOption
            value="detection"
            current={task}
            onSelect={setTask}
            title="Object detection"
            description="Draw a box around every defect in an image."
          />
          <p className="text-xs text-muted">The task cannot be changed later.</p>
        </fieldset>
        <fieldset>
          <legend className="mb-2 text-sm font-medium">Classes</legend>
          <ClassListEditor drafts={drafts} onChange={setDrafts} errors={errors.classes ?? {}} />
          <p className="mt-2 text-xs text-muted">
            You can append classes later, but classes cannot be renamed, reordered or removed.
          </p>
        </fieldset>
        {formError && <Notice tone="error">{formError}</Notice>}
        <div className="flex justify-end gap-2">
          <Button onClick={onClose}>Cancel</Button>
          <Button type="submit" variant="primary" busy={busy}>
            Create project
          </Button>
        </div>
      </form>
    </Dialog>
  );
}

function TaskOption({
  value,
  current,
  onSelect,
  title,
  description,
}: {
  value: TaskType;
  current: TaskType;
  onSelect: (value: TaskType) => void;
  title: string;
  description: string;
}) {
  return (
    <label
      className={`flex cursor-pointer gap-3 rounded-md border p-3 ${
        current === value ? "border-accent bg-blue-50" : "border-line"
      }`}
    >
      <input
        type="radio"
        name="task"
        value={value}
        checked={current === value}
        onChange={() => onSelect(value)}
        className="mt-1"
      />
      <span>
        <span className="block text-sm font-medium">{title}</span>
        <span className="block text-xs text-muted">{description}</span>
      </span>
    </label>
  );
}
