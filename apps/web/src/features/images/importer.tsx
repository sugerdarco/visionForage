"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Field, inputClass } from "@/components/ui/field";
import { Notice } from "@/components/ui/states";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import { putToSignedUrl } from "@/lib/api/storage";
import type { ImportFormat, ImportResponse, TaskType } from "@/lib/api/types";
import { formatBytes } from "@/lib/format";
import { sha256Hex } from "@/lib/hash";

export const MAX_ARCHIVE_BYTES = 64 * 1024 * 1024;

type Phase =
  | { kind: "idle" }
  | { kind: "hashing" }
  | { kind: "uploading"; progress: number }
  | { kind: "importing"; token: string }
  | { kind: "lost"; token: string; message: string }
  | { kind: "done"; result: ImportResponse }
  | { kind: "error"; message: string };

const FORMAT_HELP: Record<ImportFormat, string> = {
  imagefolder:
    "One folder per class (e.g. ok/…, defect/…). train/, val/, test/ top-level folders are also accepted.",
  yolo: "data.yaml with class names, images/… and matching labels/….txt files with center-based boxes.",
  coco: "A COCO JSON file (default annotations.json) and its images, with pixel [x, y, width, height] boxes.",
};

export function Importer({ projectId, task }: { projectId: string; task: TaskType }) {
  const queryClient = useQueryClient();
  const formats: ImportFormat[] = task === "classification" ? ["imagefolder"] : ["yolo", "coco"];
  const [format, setFormat] = useState<ImportFormat>(formats[0]!);
  const [root, setRoot] = useState("");
  const [annotationsFile, setAnnotationsFile] = useState("annotations.json");
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });
  const inputRef = useRef<HTMLInputElement>(null);
  const busy = phase.kind === "hashing" || phase.kind === "uploading" || phase.kind === "importing";

  function choose(selected: File | undefined) {
    setPhase({ kind: "idle" });
    if (!selected) return setFile(null);
    if (!/\.zip$/i.test(selected.name) && selected.type !== "application/zip") {
      setFileError("Choose a .zip archive.");
      return setFile(null);
    }
    if (selected.size > MAX_ARCHIVE_BYTES) {
      setFileError(
        `Archives must be 64 MiB or smaller (this one is ${formatBytes(selected.size)}).`,
      );
      return setFile(null);
    }
    setFileError(null);
    setFile(selected);
  }

  async function complete(token: string) {
    setPhase({ kind: "importing", token });
    try {
      const result = await api.imports.complete(projectId, token);
      setPhase({ kind: "done", result });
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["images", projectId] }),
        queryClient.invalidateQueries({ queryKey: ["project", projectId] }),
      ]);
    } catch (raw) {
      const error = toApiError(raw);
      if (error.status === null || error.status >= 500) {
        // The import may have finished (or the server was busy); repeating completion
        // with the same grant is safe within its 15-minute lifetime.
        setPhase({ kind: "lost", token, message: error.message });
      } else {
        setPhase({ kind: "error", message: error.message });
      }
    }
  }

  async function start(event: React.FormEvent) {
    event.preventDefault();
    if (!file) return setFileError("Choose a ZIP archive to import.");
    try {
      setPhase({ kind: "hashing" });
      const sha256 = await sha256Hex(file);
      const grant = await api.imports.presign(projectId, {
        format,
        size: file.size,
        sha256,
        root: root.trim(),
        ...(format === "coco"
          ? { annotations_file: annotationsFile.trim() || "annotations.json" }
          : {}),
      });
      setPhase({ kind: "uploading", progress: 0 });
      await putToSignedUrl(grant.upload_url, grant.headers, file, {
        onProgress: (loaded, total) => setPhase({ kind: "uploading", progress: loaded / total }),
      });
      await complete(grant.upload_token);
    } catch (raw) {
      setPhase({ kind: "error", message: toApiError(raw).message });
    }
  }

  return (
    <form noValidate onSubmit={start} className="flex flex-col gap-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="Format">
          {(props) => (
            <select
              {...props}
              className={inputClass}
              value={format}
              disabled={busy}
              onChange={(e) => setFormat(e.target.value as ImportFormat)}
            >
              {formats.map((f) => (
                <option key={f} value={f}>
                  {f === "imagefolder" ? "ImageFolder" : f.toUpperCase()}
                </option>
              ))}
            </select>
          )}
        </Field>
        <Field
          label="Folder inside the ZIP (optional)"
          hint="Leave empty to use the archive root. No wrapper folder is guessed."
        >
          {(props) => (
            <input
              {...props}
              className={inputClass}
              value={root}
              disabled={busy}
              maxLength={200}
              placeholder="e.g. my-dataset"
              onChange={(e) => setRoot(e.target.value)}
            />
          )}
        </Field>
        {format === "coco" && (
          <Field label="Annotations file" hint="Path relative to the folder above.">
            {(props) => (
              <input
                {...props}
                className={inputClass}
                value={annotationsFile}
                disabled={busy}
                maxLength={200}
                onChange={(e) => setAnnotationsFile(e.target.value)}
              />
            )}
          </Field>
        )}
      </div>
      <p className="text-sm text-muted">{FORMAT_HELP[format]}</p>
      {format === "yolo" && (
        <Notice tone="warn" title="Missing label files become reviewed negatives">
          An image without a matching .txt label file (or with an empty one) is imported as a
          labeled image with no defects. Check for missing label files before importing.
        </Notice>
      )}
      <Notice tone="info">
        Train/val/test folders in the archive are pooled into the project. Splits are chosen later,
        when you create a dataset version. Limits: 64 MiB ZIP, 256 MiB uncompressed, 1,000 images.
      </Notice>
      <div className="flex flex-wrap items-center gap-3">
        <input
          ref={inputRef}
          type="file"
          accept=".zip,application/zip"
          className="sr-only"
          aria-label="Choose ZIP archive"
          onChange={(e) => {
            choose(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <Button disabled={busy} onClick={() => inputRef.current?.click()}>
          Choose ZIP
        </Button>
        <span className="text-sm">
          {file ? (
            <>
              {file.name} <span className="text-muted">({formatBytes(file.size)})</span>
            </>
          ) : (
            <span className="text-muted">No archive selected</span>
          )}
        </span>
      </div>
      {fileError && <p className="text-sm text-danger">{fileError}</p>}
      <div>
        <Button type="submit" variant="primary" busy={busy} disabled={!file}>
          Import
        </Button>
      </div>
      <ImportStatus phase={phase} onCheckAgain={(token) => void complete(token)} />
    </form>
  );
}

function ImportStatus({
  phase,
  onCheckAgain,
}: {
  phase: Phase;
  onCheckAgain: (token: string) => void;
}) {
  switch (phase.kind) {
    case "idle":
      return null;
    case "hashing":
      return <Notice>Reading the archive…</Notice>;
    case "uploading":
      return (
        <Notice>
          Uploading archive… {Math.round(phase.progress * 100)}%
          <progress
            className="mt-2 block h-1.5 w-full"
            max={1}
            value={phase.progress}
            aria-label="Archive upload progress"
          />
        </Notice>
      );
    case "importing":
      return (
        <Notice title="Validating and importing">
          The server is checking every image and label. This can take a while for large archives;
          keep this page open.
        </Notice>
      );
    case "lost":
      return (
        <Notice tone="warn" title="The import result did not arrive">
          <p>
            {phase.message} The import may still have completed. Checking again is safe: an
            unchanged import does not create duplicate images.
          </p>
          <Button size="sm" className="mt-2" onClick={() => onCheckAgain(phase.token)}>
            Check again
          </Button>
        </Notice>
      );
    case "error":
      return (
        <Notice tone="error" title="Import failed">
          {phase.message}
        </Notice>
      );
    case "done": {
      const r = phase.result;
      return (
        <Notice tone="success" title="Import complete">
          <ul className="mt-1 grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-3">
            <li>New images: {r.created_images}</li>
            <li>Existing images labeled: {r.labeled_images}</li>
            <li>Unchanged: {r.unchanged_images}</li>
            <li>Duplicate files in archive: {r.duplicate_files}</li>
            <li>New classes: {r.created_classes}</li>
          </ul>
        </Notice>
      );
    }
  }
}
