import { ApiError } from "./errors";

export interface SignedPutOptions {
  signal?: AbortSignal;
  onProgress?: (loaded: number, total: number) => void;
}

/**
 * PUT raw bytes to a signed storage URL with exactly the headers the API returned.
 * This deliberately bypasses the API client: no bearer token, no JSON encoding,
 * no multipart body, and no manual Content-Length (the browser derives it).
 */
export function putToSignedUrl(
  url: string,
  headers: Record<string, string>,
  body: Blob,
  options: SignedPutOptions = {},
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url);
    for (const [name, value] of Object.entries(headers)) {
      if (name.toLowerCase() === "content-length" || name.toLowerCase() === "authorization")
        continue;
      xhr.setRequestHeader(name, value);
    }
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) options.onProgress?.(event.loaded, event.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else
        reject(
          new ApiError({
            kind: xhr.status === 403 ? "bad_request" : "unavailable",
            status: xhr.status,
            message:
              xhr.status === 403
                ? "Storage rejected the upload. The signed URL may have expired; try again."
                : `Storage rejected the upload (HTTP ${xhr.status}).`,
          }),
        );
    };
    xhr.onerror = () =>
      reject(new ApiError({ kind: "network", message: "Could not reach object storage." }));
    xhr.onabort = () => reject(new ApiError({ kind: "aborted", message: "Upload cancelled." }));
    if (options.signal) {
      if (options.signal.aborted) {
        reject(new ApiError({ kind: "aborted", message: "Upload cancelled." }));
        return;
      }
      options.signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.send(body);
  });
}
