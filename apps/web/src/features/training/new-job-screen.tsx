"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Field, inputClass } from "@/components/ui/field";
import { Card, PageHeader } from "@/components/ui/page-header";
import { ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { MIN_TEST_IMAGES_FOR_TRAINING } from "@/features/datasets/version-form";
import { useProject, useProjectId } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import { submitOnce } from "@/lib/api/outcome";
import { versionCounts, type TaskType } from "@/lib/api/types";
import { formatDateTime } from "@/lib/format";
import { ARCHITECTURE, buildJob, initialValues, PARAM_SPECS, type ParamValues } from "./params";
import { useVersionIndex } from "./use-version-index";

export function NewJobScreen() {
  const projectId = useProjectId();
  const project = useProject(projectId);
  const versions = useVersionIndex(projectId);
  const requested = useSearchParams().get("version");
  const requestedVersion = useQuery({
    queryKey: ["version", projectId, requested],
    queryFn: ({ signal }) => api.versions.get(projectId, requested!, signal),
    enabled: !!requested,
  });
  if (!project.data || versions.isPending || (requested && requestedVersion.isPending))
    return <LoadingState label="Loading" />;
  if (requested && requestedVersion.isError)
    return <ErrorState error={requestedVersion.error} onRetry={() => requestedVersion.refetch()} />;
  if (versions.isError)
    return <ErrorState error={versions.error} onRetry={() => versions.refetch()} />;
  const items =
    requestedVersion.data && !versions.data.byId.has(requestedVersion.data.id)
      ? [requestedVersion.data, ...versions.data.items]
      : versions.data.items;
  return (
    <NewJobForm
      key={`${projectId}:${requested ?? "latest"}`}
      initialVersionId={requested ?? items[0]?.id ?? ""}
      projectId={projectId}
      task={project.data.task_type as TaskType}
      versions={{ ...versions.data, items, byId: new Map(items.map((v) => [v.id, v])) }}
    />
  );
}

function NewJobForm({
  initialVersionId,
  projectId,
  task,
  versions,
}: {
  initialVersionId: string;
  projectId: string;
  task: TaskType;
  versions: NonNullable<ReturnType<typeof useVersionIndex>["data"]>;
}) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [versionId, setVersionId] = useState(initialVersionId);
  const [values, setValues] = useState<ParamValues>(() => initialValues(task));
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [rejected, setRejected] = useState<string | null>(null);
  const [uncertain, setUncertain] = useState(false);
  const [checked, setChecked] = useState(false);

  const selected = versions.byId.get(versionId);
  const testImages = selected ? versionCounts(selected).images?.test : undefined;
  const tooFewTest = testImages !== undefined && testImages < MIN_TEST_IMAGES_FOR_TRAINING;
  const arch = ARCHITECTURE[task];

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (busy || tooFewTest || (uncertain && !checked)) return;
    const { body, errors: nextErrors } = buildJob(task, versionId, values);
    setErrors(nextErrors);
    setRejected(null);
    if (!body) return;
    setBusy(true);
    const outcome = await submitOnce(() => api.jobs.create(projectId, body));
    setBusy(false);
    await queryClient.invalidateQueries({ queryKey: ["jobs", projectId] });
    if (outcome.kind === "created") {
      router.push(`/projects/${projectId}/training/${outcome.value.id}`);
    } else if (outcome.kind === "rejected") {
      const fieldErrors = outcome.error.fieldErrors();
      if (outcome.error.kind === "validation" && Object.keys(fieldErrors).length)
        setErrors(fieldErrors);
      setRejected(outcome.error.message);
    } else {
      setUncertain(true);
      setChecked(false);
    }
  }

  if (versions.items.length === 0) {
    return (
      <>
        <PageHeader title="New training run" />
        <Notice tone="info" title="No dataset versions yet">
          Training uses a frozen dataset version.{" "}
          <Link className="underline" href={`/projects/${projectId}/versions`}>
            Create a version
          </Link>{" "}
          first.
        </Notice>
      </>
    );
  }

  const specs = PARAM_SPECS[task];
  return (
    <>
      <PageHeader
        title="New training run"
        description="Training runs on a GPU worker. It may wait in the queue until a worker is free."
      />
      <form noValidate onSubmit={submit} className="flex max-w-4xl flex-col gap-6">
        <Card title="Data and model">
          <div className="grid gap-4 sm:grid-cols-2">
            <Field
              label="Dataset version"
              error={errors.dataset_version_id}
              hint={
                versions.total > versions.items.length
                  ? `Showing the newest ${versions.items.length} of ${versions.total} versions.`
                  : undefined
              }
            >
              {(props) => (
                <select
                  {...props}
                  className={inputClass}
                  value={versionId}
                  onChange={(e) => setVersionId(e.target.value)}
                >
                  {versions.items.map((v) => {
                    const test = versionCounts(v).images?.test;
                    return (
                      <option key={v.id} value={v.id}>
                        v{v.version} · {formatDateTime(v.created_at)} · {test ?? "?"} test images
                      </option>
                    );
                  })}
                </select>
              )}
            </Field>
            <Field
              label="Architecture"
              hint="The only architecture currently supported for this task."
            >
              {(props) => (
                <select {...props} className={inputClass} value={arch.id} disabled>
                  <option value={arch.id}>{arch.label}</option>
                </select>
              )}
            </Field>
          </div>
          {tooFewTest && (
            <Notice tone="warn" className="mt-4" title="This version cannot be trained">
              It has {testImages} test images; training needs at least{" "}
              {MIN_TEST_IMAGES_FOR_TRAINING}. Label more images and create a new version.
            </Notice>
          )}
        </Card>
        <Card title="Hyperparameters">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {specs
              .filter((s) => s.kind !== "bool")
              .map((spec) => (
                <Field key={spec.key} label={spec.label} error={errors[spec.key]} hint={spec.help}>
                  {(props) => (
                    <input
                      {...props}
                      inputMode={spec.kind === "int" ? "numeric" : "decimal"}
                      className={inputClass}
                      value={String(values[spec.key] ?? "")}
                      onChange={(e) => setValues((v) => ({ ...v, [spec.key]: e.target.value }))}
                    />
                  )}
                </Field>
              ))}
          </div>
          <fieldset className="mt-4 flex flex-col gap-2">
            <legend className="mb-1 text-sm font-medium">Augmentation</legend>
            {specs
              .filter((s) => s.kind === "bool")
              .map((spec) => (
                <label key={spec.key} className="flex items-start gap-2 text-sm">
                  <input
                    type="checkbox"
                    className="mt-1"
                    checked={values[spec.key] === true}
                    onChange={(e) => setValues((v) => ({ ...v, [spec.key]: e.target.checked }))}
                  />
                  <span>
                    {spec.label}
                    <span className="block text-xs text-muted">{spec.help}</span>
                  </span>
                </label>
              ))}
          </fieldset>
          {task === "classification" && (
            <p className="mt-3 text-xs text-muted">Input size is fixed by the classifier.</p>
          )}
        </Card>
        {rejected && (
          <Notice tone="error" title="Run not started">
            {rejected}
          </Notice>
        )}
        {uncertain && (
          <Notice tone="warn" title="Result unknown">
            <p>
              The server did not confirm the request, so a run may have been queued anyway. Check
              the{" "}
              <Link className="underline" href={`/projects/${projectId}/training`}>
                training list
              </Link>{" "}
              before submitting again.
            </p>
            <label className="mt-2 flex items-center gap-2">
              <input
                type="checkbox"
                checked={checked}
                onChange={(e) => setChecked(e.target.checked)}
              />
              I checked the list and want to start another run
            </label>
          </Notice>
        )}
        <div className="flex gap-2">
          <Button
            type="submit"
            variant="primary"
            busy={busy}
            disabled={tooFewTest || (uncertain && !checked)}
          >
            Start training
          </Button>
          <Link
            href={`/projects/${projectId}/training`}
            className="rounded-md border border-line bg-white px-3.5 py-2 text-sm font-medium hover:bg-slate-50"
          >
            Cancel
          </Link>
        </div>
      </form>
    </>
  );
}
