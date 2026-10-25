import { toApiError, type ApiError } from "./errors";

/**
 * Result of a non-idempotent POST. `uncertain` means the request may have been
 * applied even though no response arrived; callers must inspect the server list
 * before offering another submission and must never retry automatically.
 */
export type SubmissionOutcome<T> =
  | { kind: "created"; value: T }
  | { kind: "rejected"; error: ApiError }
  | { kind: "uncertain"; error: ApiError };

export function isDefiniteRejection(error: ApiError): boolean {
  if (error.status === null) return false;
  if (error.status >= 400 && error.status < 500) return true;
  // 503 with Retry-After is the "another build/import is running" guard, rejected up front.
  return error.status === 503 && error.retryAfterSeconds !== null;
}

export async function submitOnce<T>(send: () => Promise<T>): Promise<SubmissionOutcome<T>> {
  try {
    return { kind: "created", value: await send() };
  } catch (raw) {
    const error = toApiError(raw);
    return isDefiniteRejection(error) ? { kind: "rejected", error } : { kind: "uncertain", error };
  }
}
