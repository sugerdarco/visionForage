"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Badge } from "@/components/ui/badge";
import { Card, PageHeader } from "@/components/ui/page-header";
import { ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { useProject } from "@/features/projects/use-project";
import { api } from "@/lib/api/endpoints";
import { versionCounts, type Split } from "@/lib/api/types";
import { formatDateTime } from "@/lib/format";
import { MIN_TEST_IMAGES_FOR_TRAINING } from "./version-form";
import { TrainingEligibility } from "./versions-screen";

const SPLITS: { key: Split; label: string }[] = [
  { key: "train", label: "Train" },
  { key: "val", label: "Validation" },
  { key: "test", label: "Test" },
];

const EXCLUDED_LABELS: Record<string, string> = {
  unlabeled: "Unlabeled",
  in_review: "In review",
};

export function VersionDetail() {
  const { projectId, versionId } = useParams<{ projectId: string; versionId: string }>();
  const project = useProject(projectId);
  const version = useQuery({
    queryKey: ["version", projectId, versionId],
    queryFn: ({ signal }) => api.versions.get(projectId, versionId, signal),
    // Download URLs expire after 300 s; keep them fresh while the page is open.
    staleTime: 0,
    refetchInterval: 200_000,
  });

  if (version.isPending) return <LoadingState label="Loading version" />;
  if (version.isError)
    return <ErrorState error={version.error} onRetry={() => version.refetch()} />;

  const v = version.data;
  const counts = versionCounts(v);
  const detection = project.data?.task_type === "detection";
  const classMap = Object.entries(counts.class_map ?? {}).sort(([, a], [, b]) => a - b);
  const testImages = counts.images?.test;

  return (
    <>
      <nav aria-label="Breadcrumb" className="mb-2 text-sm text-muted">
        <Link href={`/projects/${projectId}/versions`} className="hover:underline">
          Dataset versions
        </Link>{" "}
        / v{v.version}
      </nav>
      <PageHeader
        title={`Dataset version ${v.version}`}
        description={`Frozen ${formatDateTime(v.created_at)}. This version never changes; label edits made later need a new version.`}
        actions={
          testImages !== undefined && testImages >= MIN_TEST_IMAGES_FOR_TRAINING ? (
            <Link
              href={`/projects/${projectId}/training/new?version=${v.id}`}
              className="rounded-md bg-accent px-3.5 py-2 text-sm font-medium text-white hover:bg-accent-strong"
            >
              Train on this version
            </Link>
          ) : undefined
        }
      />
      <div className="flex flex-col gap-6">
        {(counts.warnings?.length ?? 0) > 0 && (
          <Notice tone="warn" title="Coverage warnings">
            <ul className="list-disc pl-4">
              {counts.warnings!.map((w) => (
                <li key={w.split}>
                  {w.split === "val" ? "Validation" : w.split === "test" ? "Test" : w.split} split
                  has no examples of: {w.missing_classes.join(", ")}. Metrics for these classes
                  cannot be measured on that split.
                </li>
              ))}
            </ul>
          </Notice>
        )}
        {testImages !== undefined && testImages < MIN_TEST_IMAGES_FOR_TRAINING && (
          <Notice tone="warn" title="Not enough test images to train">
            Training needs at least {MIN_TEST_IMAGES_FOR_TRAINING} test images; this version has{" "}
            {testImages}. Label more images and create a new version.
          </Notice>
        )}
        <Card title="Splits">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Split</th>
                  <th className="py-2 pr-4 text-right font-medium">Requested</th>
                  <th className="py-2 pr-4 text-right font-medium">Images</th>
                  <th className="py-2 pr-4 text-right font-medium">
                    {detection ? "Boxes" : "Labels"}
                  </th>
                  <th className="py-2 text-right font-medium">Unique image groups</th>
                </tr>
              </thead>
              <tbody>
                {SPLITS.map(({ key, label }) => (
                  <tr key={key} className="border-t border-line">
                    <td className="py-2 pr-4 font-medium">{label}</td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {v.split_ratios[key] !== undefined
                        ? `${+(v.split_ratios[key]! * 100).toFixed(4)}%`
                        : "Not available"}
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {counts.images?.[key] ?? "Not available"}
                    </td>
                    <td className="py-2 pr-4 text-right tabular-nums">
                      {counts.labels?.[key] ?? "Not available"}
                    </td>
                    <td className="py-2 text-right tabular-nums">
                      {counts.pixel_groups?.[key] ?? "Not available"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted">
            Requested percentages apply to groups of identical images, so realized image counts can
            differ.
            {detection
              ? " Box counts can exceed image counts because one image can contain several boxes."
              : ""}
          </p>
          <div className="mt-3 flex items-center gap-2 text-sm">
            Training: <TrainingEligibility testImages={testImages} />
          </div>
        </Card>

        <Card title={detection ? "Boxes per class" : "Images per class"}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-muted">
                <tr>
                  <th className="py-2 pr-4 font-medium">Index</th>
                  <th className="py-2 pr-4 font-medium">Class</th>
                  {SPLITS.map((s) => (
                    <th key={s.key} className="py-2 pr-4 text-right font-medium">
                      {s.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {classMap.map(([name, index]) => (
                  <tr key={name} className="border-t border-line">
                    <td className="py-2 pr-4 font-mono">{index}</td>
                    <td className="py-2 pr-4 break-all">{name}</td>
                    {SPLITS.map((s) => {
                      const count = counts.classes?.[s.key]?.[name];
                      return (
                        <td key={s.key} className="py-2 pr-4 text-right tabular-nums">
                          {count === undefined ? (
                            "Not available"
                          ) : count === 0 ? (
                            <Badge tone="amber">0</Badge>
                          ) : (
                            count
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted">
            This is the frozen class map: the model output index for each class. Classes added to
            the project later are not part of this version.
          </p>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card title="Settings and exclusions">
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
              <dt className="text-muted">Seed</dt>
              <dd className="font-mono">{counts.seed ?? "Not available"}</dd>
              <dt className="text-muted">Split method</dt>
              <dd className="font-mono text-xs break-all">
                {counts.split_method ?? "Not available"}
              </dd>
              <dt className="text-muted">Groups of duplicate images</dt>
              <dd>{counts.duplicate_groups ?? "Not available"}</dd>
              {Object.entries(counts.excluded ?? {}).map(([status, count]) => (
                <div key={status} className="contents">
                  <dt className="text-muted">Excluded ({EXCLUDED_LABELS[status] ?? status})</dt>
                  <dd>{count}</dd>
                </div>
              ))}
            </dl>
          </Card>
          <Card title="Artifacts">
            <p className="mb-3 text-sm text-muted">
              The manifest is the exact training input. The audit records image hashes and split
              assignments. Links are private and short-lived; they refresh automatically while this
              page is open.
            </p>
            <div className="flex flex-wrap gap-2">
              <a
                className="rounded-md border border-line bg-white px-3.5 py-2 text-sm font-medium hover:bg-slate-50"
                href={v.manifest_url}
                target="_blank"
                rel="noopener noreferrer"
              >
                Download manifest
              </a>
              {v.audit_url && (
                <a
                  className="rounded-md border border-line bg-white px-3.5 py-2 text-sm font-medium hover:bg-slate-50"
                  href={v.audit_url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Download audit
                </a>
              )}
            </div>
            <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
              <dt className="text-muted">Manifest SHA-256</dt>
              <dd className="font-mono break-all">{counts.manifest_sha256 ?? "Not available"}</dd>
              <dt className="text-muted">Audit SHA-256</dt>
              <dd className="font-mono break-all">{counts.audit_sha256 ?? "Not available"}</dd>
            </dl>
          </Card>
        </div>
      </div>
    </>
  );
}
