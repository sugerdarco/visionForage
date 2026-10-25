"use client";

import { keepPreviousData, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Field, inputClass } from "@/components/ui/field";
import { Card, PageHeader } from "@/components/ui/page-header";
import { Pagination } from "@/components/ui/pagination";
import { EmptyState, ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { useProject, useProjectId } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import { submitOnce } from "@/lib/api/outcome";
import { versionCounts, type DatasetVersion } from "@/lib/api/types";
import { formatDateTime } from "@/lib/format";
import {
  DEFAULT_VERSION_FORM,
  MIN_TEST_IMAGES_FOR_TRAINING,
  parseVersionForm,
  type VersionFormErrors,
  type VersionFormValues,
} from "./version-form";

const PAGE_SIZE = 20;

export function VersionsScreen() {
  const projectId = useProjectId();
  const project = useProject(projectId);
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const page = Math.max(1, Number(params.get("page")) || 1);
  const versions = useQuery({
    queryKey: ["versions", projectId, page],
    queryFn: ({ signal }) => api.versions.list(projectId, page, PAGE_SIZE, signal),
    placeholderData: keepPreviousData,
  });
  const detection = project.data?.task_type === "detection";

  return (
    <>
      <PageHeader
        title="Dataset versions"
        description="A version freezes the currently labeled images, their labels and the class list into fixed train / validation / test splits. Training always uses a frozen version, so results stay reproducible."
      />
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <section aria-label="Versions" className="flex min-w-0 flex-col gap-4">
          {versions.isPending ? (
            <LoadingState label="Loading versions" />
          ) : versions.isError ? (
            <ErrorState error={versions.error} onRetry={() => versions.refetch()} />
          ) : versions.data.total === 0 ? (
            <EmptyState title="No dataset versions yet">
              Label images first, then create a version with the form.
            </EmptyState>
          ) : (
            <>
              <div className="overflow-x-auto rounded-lg border border-line bg-panel">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 text-left text-xs text-muted">
                    <tr>
                      <th className="px-3 py-2 font-medium">Version</th>
                      <th className="px-3 py-2 font-medium">Created</th>
                      <th className="px-3 py-2 text-right font-medium">Train</th>
                      <th className="px-3 py-2 text-right font-medium">Validation</th>
                      <th className="px-3 py-2 text-right font-medium">Test</th>
                      <th className="px-3 py-2 font-medium">Training</th>
                    </tr>
                  </thead>
                  <tbody>
                    {versions.data.items.map((v) => (
                      <VersionRow key={v.id} version={v} projectId={projectId} />
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-muted">
                Counts are images per split{detection ? "; boxes are shown on the detail page" : ""}
                .
              </p>
              <Pagination
                page={page}
                pageSize={PAGE_SIZE}
                total={versions.data.total}
                label="versions"
                onPage={(next) => router.push(`${pathname}?page=${next}`)}
              />
            </>
          )}
        </section>
        <CreateVersion
          projectId={projectId}
          detection={detection}
          onCreated={(id) => router.push(`/projects/${projectId}/versions/${id}`)}
        />
      </div>
    </>
  );
}

function VersionRow({ version, projectId }: { version: DatasetVersion; projectId: string }) {
  const counts = versionCounts(version);
  const test = counts.images?.test;
  const warnings = counts.warnings?.length ?? 0;
  return (
    <tr className="border-t border-line">
      <td className="px-3 py-2">
        <Link
          className="font-medium text-accent underline"
          href={`/projects/${projectId}/versions/${version.id}`}
        >
          v{version.version}
        </Link>
        {warnings > 0 && (
          <span className="ml-2">
            <Badge tone="amber">
              {warnings} warning{warnings === 1 ? "" : "s"}
            </Badge>
          </span>
        )}
      </td>
      <td className="px-3 py-2 whitespace-nowrap">{formatDateTime(version.created_at)}</td>
      <td className="px-3 py-2 text-right tabular-nums">{counts.images?.train ?? "—"}</td>
      <td className="px-3 py-2 text-right tabular-nums">{counts.images?.val ?? "—"}</td>
      <td className="px-3 py-2 text-right tabular-nums">{test ?? "—"}</td>
      <td className="px-3 py-2">
        <TrainingEligibility testImages={test} />
      </td>
    </tr>
  );
}

export function TrainingEligibility({ testImages }: { testImages: number | undefined }) {
  if (testImages === undefined) return <Badge>Unknown</Badge>;
  return testImages >= MIN_TEST_IMAGES_FOR_TRAINING ? (
    <Badge tone="green">Eligible</Badge>
  ) : (
    <Badge tone="amber">Needs {MIN_TEST_IMAGES_FOR_TRAINING}+ test images</Badge>
  );
}

function CreateVersion({
  projectId,
  detection,
  onCreated,
}: {
  projectId: string;
  detection: boolean;
  onCreated: (id: string) => void;
}) {
  const queryClient = useQueryClient();
  const [values, setValues] = useState<VersionFormValues>(DEFAULT_VERSION_FORM);
  const [errors, setErrors] = useState<VersionFormErrors>({});
  const [busy, setBusy] = useState(false);
  const [rejected, setRejected] = useState<string | null>(null);
  const [uncertain, setUncertain] = useState(false);
  const [checked, setChecked] = useState(false);

  const set = (key: keyof VersionFormValues) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setValues((v) => ({ ...v, [key]: e.target.value }));

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (busy || (uncertain && !checked)) return;
    const parsed = parseVersionForm(values);
    setErrors(parsed.errors);
    setRejected(null);
    if (!parsed.body) return;
    setBusy(true);
    const outcome = await submitOnce(() => api.versions.create(projectId, parsed.body!));
    setBusy(false);
    await queryClient.invalidateQueries({ queryKey: ["versions", projectId] });
    if (outcome.kind === "created") {
      setUncertain(false);
      setChecked(false);
      onCreated(outcome.value.id);
    } else if (outcome.kind === "rejected") {
      setRejected(
        outcome.error.status === 503
          ? `${outcome.error.message} Another version may be building; wait a few seconds, then check the list before trying again.`
          : outcome.error.message,
      );
    } else {
      // Never retry automatically: the version may exist even though the response was lost.
      setUncertain(true);
      setChecked(false);
    }
  }

  return (
    <Card title="Create a version" className="h-fit">
      <form noValidate onSubmit={submit} className="flex flex-col gap-4">
        <ul className="list-disc space-y-1 pl-4 text-xs text-muted">
          <li>
            Only images marked Labeled are included. Unlabeled and in-review images are excluded and
            counted.
          </li>
          {detection && <li>Images reviewed as “no defects” are included as clean examples.</li>}
          <li>
            Identical images always land in the same split. Percentages target groups of identical
            images, so actual image counts can differ.
          </li>
          <li>Versions never change. Later label edits need a new version.</li>
          <li>Training needs at least {MIN_TEST_IMAGES_FOR_TRAINING} test images.</li>
        </ul>
        <fieldset>
          <legend className="mb-2 text-sm font-medium">Split (%)</legend>
          <div className="grid grid-cols-3 gap-2">
            {(["train", "val", "test"] as const).map((split) => (
              <Field
                key={split}
                label={split === "train" ? "Train" : split === "val" ? "Validation" : "Test"}
                error={errors[split]}
              >
                {(props) => (
                  <input
                    {...props}
                    inputMode="decimal"
                    className={inputClass}
                    value={values[split]}
                    onChange={set(split)}
                  />
                )}
              </Field>
            ))}
          </div>
          {errors.ratios && <p className="mt-1 text-sm text-danger">{errors.ratios}</p>}
        </fieldset>
        <Field
          label="Seed"
          error={errors.seed}
          hint="The same labels, seed and split always produce the same version."
        >
          {(props) => (
            <input
              {...props}
              inputMode="numeric"
              className={inputClass}
              value={values.seed}
              onChange={set("seed")}
            />
          )}
        </Field>
        {rejected && (
          <Notice tone="error" title="Version not created">
            {rejected}
          </Notice>
        )}
        {uncertain && (
          <Notice tone="warn" title="Result unknown">
            <p>
              The server did not confirm the request. A version may have been created anyway. The
              list has been refreshed: check whether a new version appeared before submitting again.
            </p>
            <label className="mt-2 flex items-center gap-2">
              <input
                type="checkbox"
                checked={checked}
                onChange={(e) => setChecked(e.target.checked)}
              />
              I checked the list and want to create another version
            </label>
          </Notice>
        )}
        <Button type="submit" variant="primary" busy={busy} disabled={uncertain && !checked}>
          {busy ? "Building version…" : "Create version"}
        </Button>
        {busy && (
          <p className="text-xs text-muted" role="status">
            The server is reading and hashing every labeled image. This can take a while.
          </p>
        )}
      </form>
    </Card>
  );
}
