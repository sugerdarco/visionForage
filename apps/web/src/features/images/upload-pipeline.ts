import type { Api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import { putToSignedUrl } from "@/lib/api/storage";
import type { FileClaim } from "@/lib/api/types";
import { sha256Hex } from "@/lib/hash";
import { chunk, runPool } from "@/lib/pool";

export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;
export const PRESIGN_BATCH = 20;
const HASH_CONCURRENCY = 3;
const PUT_CONCURRENCY = 4;

export type UploadState =
  | "queued"
  | "hashing"
  | "uploading"
  | "verifying"
  | "registered"
  | "duplicate"
  | "failed"
  | "cancelled";

export interface UploadItem {
  id: number;
  file: File;
  state: UploadState;
  progress: number;
  error: string | null;
  /** False when the server rejected the file itself, so retrying cannot help. */
  retryable: boolean;
  imageId: string | null;
}

export type UploadUpdate = (id: number, patch: Partial<UploadItem>) => void;

export function contentTypeFor(file: File): FileClaim["content_type"] | null {
  const type = file.type.toLowerCase();
  if (type === "image/jpeg" || type === "image/png") return type;
  if (!type) {
    if (/\.(jpe?g)$/i.test(file.name)) return "image/jpeg";
    if (/\.png$/i.test(file.name)) return "image/png";
  }
  return null;
}

/** Checks the browser can make without decoding; the server verifies everything else. */
export function preflightError(file: File): string | null {
  if (!contentTypeFor(file)) return "Only JPEG and PNG images are supported.";
  if (file.size === 0) return "The file is empty.";
  if (file.size > MAX_IMAGE_BYTES) return "Images must be 20 MiB or smaller.";
  return null;
}

/** Turn API rejection text into guidance a non-specialist can act on. */
export function explainRejection(message: string): string {
  const lower = message.toLowerCase();
  if (lower.includes("orientation"))
    return `${message} Rotate the image in an editor and save it without EXIF rotation, then upload again.`;
  if (lower.includes("transparen") || lower.includes("alpha") || lower.includes("palette"))
    return `${message} Save the image as RGB or grayscale without transparency.`;
  if (lower.includes("animat") || lower.includes("frame"))
    return `${message} Upload a single still frame.`;
  if (lower.includes("pixel") || lower.includes("dimension"))
    return `${message} Images must be at least 10×10 and at most 20 megapixels.`;
  return message;
}

/**
 * Upload accepted files: hash exactly the bytes to be sent, presign in batches of
 * at most 20, PUT each original File to its signed URL, then complete each file
 * separately. Each batch is presigned just before its PUTs so grants (900 s) stay
 * valid. One file's failure never affects the others' results.
 */
export async function uploadFiles(
  api: Api,
  projectId: string,
  items: readonly UploadItem[],
  report: UploadUpdate,
  signal: AbortSignal,
): Promise<void> {
  const states = new Map<number, UploadState>();
  const update: UploadUpdate = (id, patch) => {
    if (patch.state) states.set(id, patch.state);
    report(id, patch);
  };
  for (const batch of chunk(items, PRESIGN_BATCH)) {
    if (signal.aborted) break;
    const hashed: { item: UploadItem; claim: FileClaim }[] = [];
    await runPool(batch, HASH_CONCURRENCY, async (item) => {
      if (signal.aborted) return;
      update(item.id, { state: "hashing", error: null, progress: 0 });
      try {
        const sha256 = await sha256Hex(item.file);
        hashed.push({
          item,
          claim: { content_type: contentTypeFor(item.file)!, size: item.file.size, sha256 },
        });
      } catch {
        update(item.id, { state: "failed", error: "The file could not be read.", retryable: true });
      }
    });
    if (signal.aborted || hashed.length === 0) break;
    hashed.sort((a, b) => a.item.id - b.item.id);
    let grants;
    try {
      grants = await api.uploads.presign(
        projectId,
        hashed.map((h) => h.claim),
      );
    } catch (raw) {
      const error = toApiError(raw);
      for (const { item } of hashed) {
        update(item.id, {
          state: "failed",
          error: error.message,
          retryable: error.kind !== "validation",
        });
      }
      continue;
    }
    await runPool(hashed, PUT_CONCURRENCY, async ({ item }, index) => {
      const grant = grants[index];
      if (!grant || signal.aborted) return;
      try {
        update(item.id, { state: "uploading" });
        await putToSignedUrl(grant.upload_url, grant.headers, item.file, {
          signal,
          onProgress: (loaded, total) => update(item.id, { progress: loaded / total }),
        });
        update(item.id, { state: "verifying", progress: 1 });
        const { data, status } = await api.uploads.complete(projectId, grant.upload_token);
        update(item.id, { state: status === 201 ? "registered" : "duplicate", imageId: data.id });
      } catch (raw) {
        const error = toApiError(raw);
        if (error.kind === "aborted") update(item.id, { state: "cancelled", error: null });
        else
          update(item.id, {
            state: "failed",
            error: explainRejection(error.message),
            retryable: error.kind !== "validation",
          });
      }
    });
  }
  if (signal.aborted) {
    // Files that never finished are marked; registered images stay registered on the server.
    for (const item of items) {
      const state = states.get(item.id);
      if (state !== "registered" && state !== "duplicate" && state !== "failed") {
        update(item.id, { state: "cancelled", error: null });
      }
    }
  }
}
