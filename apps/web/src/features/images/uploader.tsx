"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Notice } from "@/components/ui/states";
import { api } from "@/lib/api/endpoints";
import { formatBytes } from "@/lib/format";
import { preflightError, uploadFiles, type UploadItem, type UploadState } from "./upload-pipeline";

const STATE_LABEL: Record<
  UploadState,
  { label: string; tone: "neutral" | "blue" | "green" | "amber" | "red" }
> = {
  queued: { label: "Waiting", tone: "neutral" },
  hashing: { label: "Checking file", tone: "blue" },
  uploading: { label: "Uploading", tone: "blue" },
  verifying: { label: "Verifying", tone: "blue" },
  registered: { label: "Added", tone: "green" },
  duplicate: { label: "Already in project", tone: "amber" },
  failed: { label: "Failed", tone: "red" },
  cancelled: { label: "Not uploaded", tone: "neutral" },
};

let nextId = 0;

export function Uploader({ projectId }: { projectId: string }) {
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [items, setItems] = useState<UploadItem[]>([]);
  const [running, setRunning] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [cancelled, setCancelled] = useState(false);

  const update = (id: number, patch: Partial<UploadItem>) =>
    setItems((current) => current.map((item) => (item.id === id ? { ...item, ...patch } : item)));

  async function run(batch: UploadItem[]) {
    if (batch.length === 0) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setCancelled(false);
    try {
      await uploadFiles(api, projectId, batch, update, controller.signal);
    } finally {
      setRunning(false);
      abortRef.current = null;
      await queryClient.invalidateQueries({ queryKey: ["images", projectId] });
    }
  }

  function addFiles(files: FileList | File[]) {
    const added: UploadItem[] = Array.from(files).map((file) => {
      const error = preflightError(file);
      return {
        id: ++nextId,
        file,
        state: error ? "failed" : "queued",
        progress: 0,
        error,
        retryable: !error,
        imageId: null,
      };
    });
    setItems((current) => [...added, ...current.filter((i) => i.state !== "queued")]);
    if (!running) void run(added.filter((i) => i.state === "queued"));
  }

  const counts = items.reduce<Record<string, number>>((acc, item) => {
    acc[item.state] = (acc[item.state] ?? 0) + 1;
    return acc;
  }, {});
  const retryable = items.filter((i) => i.state === "failed" && i.retryable);

  return (
    <div className="flex flex-col gap-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!running && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
        }}
        className={`flex flex-col items-center gap-2 rounded-lg border-2 border-dashed p-6 text-center ${
          dragging ? "border-accent bg-blue-50" : "border-line bg-white"
        }`}
      >
        <p className="text-sm font-medium">Drop JPEG or PNG images here</p>
        <p className="text-xs text-muted">
          Single-frame RGB or grayscale, 10×10 to 20 megapixels, up to 20 MiB each. No transparency
          and no EXIF rotation.
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png"
          multiple
          className="sr-only"
          aria-label="Choose images to upload"
          onChange={(e) => {
            if (e.target.files?.length) addFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <Button variant="primary" disabled={running} onClick={() => inputRef.current?.click()}>
          Choose images
        </Button>
      </div>

      {items.length > 0 && (
        <div className="rounded-lg border border-line bg-white">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2 text-sm">
            <p aria-live="polite">
              {running ? "Uploading… " : ""}
              {counts.registered ?? 0} added · {counts.duplicate ?? 0} already present ·{" "}
              {counts.failed ?? 0} failed
              {counts.cancelled ? ` · ${counts.cancelled} not uploaded` : ""}
            </p>
            <div className="flex gap-2">
              {running && (
                <Button
                  size="sm"
                  onClick={() => {
                    abortRef.current?.abort();
                    setCancelled(true);
                  }}
                >
                  Stop
                </Button>
              )}
              {!running && retryable.length > 0 && (
                <Button size="sm" onClick={() => void run(retryable)}>
                  Retry {retryable.length} failed
                </Button>
              )}
              {!running && (
                <Button size="sm" variant="ghost" onClick={() => setItems([])}>
                  Clear list
                </Button>
              )}
            </div>
          </div>
          {cancelled && !running && (
            <Notice tone="info" className="m-3">
              Upload stopped in this browser. Images already marked “Added” remain in the project;
              stopping does not delete them.
            </Notice>
          )}
          <ul className="max-h-72 divide-y divide-line overflow-y-auto text-sm">
            {items.map((item) => (
              <li key={item.id} className="flex flex-col gap-1 px-4 py-2">
                <div className="flex items-center justify-between gap-3">
                  <span className="truncate" title={item.file.name}>
                    {item.file.name}{" "}
                    <span className="text-xs text-muted">{formatBytes(item.file.size)}</span>
                  </span>
                  <Badge tone={STATE_LABEL[item.state].tone}>{STATE_LABEL[item.state].label}</Badge>
                </div>
                {item.state === "uploading" && (
                  <progress
                    className="h-1.5 w-full"
                    max={1}
                    value={item.progress}
                    aria-label={`Upload progress for ${item.file.name}`}
                  />
                )}
                {item.error && <p className="text-xs text-danger">{item.error}</p>}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
