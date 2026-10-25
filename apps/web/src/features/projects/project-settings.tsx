"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { Field, inputClass } from "@/components/ui/field";
import { Card, PageHeader } from "@/components/ui/page-header";
import { Notice } from "@/components/ui/states";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import type { ProjectDetail } from "@/lib/api/types";
import { classDraftErrors, ClassListEditor, type DraftClass } from "./class-list-editor";
import { useProject, useProjectId } from "./use-project";

export function ProjectSettings() {
  const projectId = useProjectId();
  const project = useProject(projectId);
  if (!project.data) return null;
  return (
    <>
      <PageHeader title="Settings" />
      <div className="flex max-w-3xl flex-col gap-6">
        <RenameCard project={project.data} />
        <ClassesCard project={project.data} />
        <DeleteCard project={project.data} />
      </div>
    </>
  );
}

function RenameCard({ project }: { project: ProjectDetail }) {
  const queryClient = useQueryClient();
  const [name, setName] = useState(project.name);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setSaved(false);
    if (!name.trim()) return setError("Enter a project name.");
    if (name.trim().length > 200) return setError("Use at most 200 characters.");
    setError(null);
    setBusy(true);
    try {
      const updated = await api.projects.rename(project.id, name.trim());
      queryClient.setQueryData(["project", project.id], updated);
      await queryClient.invalidateQueries({ queryKey: ["projects"] });
      setName(updated.name);
      setSaved(true);
    } catch (raw) {
      setError(toApiError(raw).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Project name">
      <form noValidate onSubmit={submit} className="flex flex-col gap-3 sm:flex-row sm:items-start">
        <Field label="Name" error={error} className="flex-1">
          {(props) => (
            <input
              {...props}
              className={inputClass}
              value={name}
              maxLength={200}
              onChange={(e) => {
                setName(e.target.value);
                setSaved(false);
              }}
            />
          )}
        </Field>
        <Button type="submit" variant="primary" busy={busy} className="sm:mt-6">
          Rename
        </Button>
      </form>
      {saved && (
        <p role="status" className="mt-2 text-sm text-ok">
          Saved.
        </p>
      )}
    </Card>
  );
}

function ClassesCard({ project }: { project: ProjectDetail }) {
  const queryClient = useQueryClient();
  const [drafts, setDrafts] = useState<DraftClass[]>([]);
  const [errors, setErrors] = useState<Record<number, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const existingNames = project.classes.map((c) => c.name);
  const existingColors = project.classes.map((c) => c.color);

  async function submit() {
    const nextErrors = classDraftErrors(drafts, existingNames);
    setErrors(nextErrors);
    setFormError(null);
    if (Object.keys(nextErrors).length || drafts.length === 0) return;
    setBusy(true);
    try {
      await api.projects.addClasses(
        project.id,
        drafts.map((d) => ({ name: d.name.trim(), color: d.color.toLowerCase() })),
      );
      await queryClient.invalidateQueries({ queryKey: ["project", project.id] });
      setDrafts([]);
    } catch (raw) {
      const error = toApiError(raw);
      setFormError(
        error.kind === "conflict"
          ? "One of these class names already exists in the project. Class names are case-sensitive."
          : error.message,
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Classes">
      <p className="mb-3 text-sm text-muted">
        Each class has a fixed model index assigned by the server. Classes can be appended but not
        renamed, reordered or deleted. Models trained earlier keep their own frozen class list.
      </p>
      <table className="mb-4 w-full text-sm">
        <thead className="text-left text-xs text-muted">
          <tr>
            <th className="py-1 pr-3 font-medium">Index</th>
            <th className="py-1 pr-3 font-medium">Class</th>
            <th className="py-1 font-medium">Color</th>
          </tr>
        </thead>
        <tbody>
          {project.classes.map((c) => (
            <tr key={c.id} className="border-t border-line">
              <td className="py-2 pr-3 font-mono">{c.index}</td>
              <td className="py-2 pr-3 break-all">{c.name}</td>
              <td className="py-2">
                <span className="inline-flex items-center gap-2 font-mono text-xs">
                  <span
                    className="h-4 w-4 rounded border border-line"
                    style={{ background: c.color }}
                  />
                  {c.color}
                </span>
              </td>
            </tr>
          ))}
          {project.classes.length === 0 && (
            <tr>
              <td colSpan={3} className="py-3 text-muted">
                No classes yet. Add at least one before labeling.
              </td>
            </tr>
          )}
        </tbody>
      </table>
      <ClassListEditor
        drafts={drafts}
        onChange={setDrafts}
        errors={errors}
        existingColors={existingColors}
      />
      {formError && (
        <Notice tone="error" className="mt-3">
          {formError}
        </Notice>
      )}
      {drafts.length > 0 && (
        <Button className="mt-3" variant="primary" busy={busy} onClick={submit}>
          Append {drafts.length === 1 ? "class" : `${drafts.length} classes`}
        </Button>
      )}
    </Card>
  );
}

function DeleteCard({ project }: { project: ProjectDetail }) {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function remove() {
    setBusy(true);
    setError(null);
    try {
      await api.projects.remove(project.id);
      setOpen(false);
      queryClient.setQueryData(["project", project.id], { ...project, deleting: true });
      await queryClient.invalidateQueries({ queryKey: ["projects"] });
    } catch (raw) {
      setError(toApiError(raw).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Delete project">
      <p className="text-sm text-muted">
        Permanently delete this project, its images, labels, dataset versions, trained models and
        deployments. Running jobs are cancelled before cleanup finishes. This cannot be undone.
      </p>
      {error && (
        <Notice tone="error" className="mt-3">
          {error}
        </Notice>
      )}
      <Button className="mt-3" variant="danger" onClick={() => setOpen(true)}>
        Delete project
      </Button>
      <Dialog
        open={open}
        onClose={() => setOpen(false)}
        title="Delete this project?"
        footer={
          <>
            <Button onClick={() => setOpen(false)}>Cancel</Button>
            <Button variant="danger" busy={busy} onClick={remove}>
              Delete
            </Button>
          </>
        }
      >
        <p>
          “{project.name}” and all its data will be permanently removed. Active model endpoints will
          stop.
        </p>
      </Dialog>
    </Card>
  );
}
