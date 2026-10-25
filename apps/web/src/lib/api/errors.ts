import type { ValidationIssue } from "./types";

export type ErrorKind =
  | "network"
  | "unauthorized"
  | "not_found"
  | "conflict"
  | "validation"
  | "capacity"
  | "unavailable"
  | "bad_request"
  | "aborted"
  | "unknown";

/** A normalized API failure that screens can render without inspecting raw responses. */
export class ApiError extends Error {
  readonly kind: ErrorKind;
  readonly status: number | null;
  readonly issues: ValidationIssue[];
  readonly retryAfterSeconds: number | null;

  constructor(options: {
    kind: ErrorKind;
    message: string;
    status?: number | null;
    issues?: ValidationIssue[];
    retryAfterSeconds?: number | null;
  }) {
    super(options.message);
    this.name = "ApiError";
    this.kind = options.kind;
    this.status = options.status ?? null;
    this.issues = options.issues ?? [];
    this.retryAfterSeconds = options.retryAfterSeconds ?? null;
  }

  /** Validation messages keyed by the last body location, e.g. "password" or "split_ratios". */
  fieldErrors(): Record<string, string> {
    const fields: Record<string, string> = {};
    for (const issue of this.issues) {
      const loc = issue.loc.filter((part) => part !== "body");
      const key = loc.length ? String(loc[0]) : "_";
      fields[key] ??= cleanMessage(issue.msg);
    }
    return fields;
  }
}

function cleanMessage(message: string): string {
  return message.replace(/^Value error, /, "");
}

export function kindForStatus(status: number): ErrorKind {
  if (status === 400) return "bad_request";
  if (status === 401) return "unauthorized";
  if (status === 404) return "not_found";
  if (status === 409) return "conflict";
  if (status === 422) return "validation";
  if (status === 429) return "capacity";
  if (status === 503 || status === 502 || status === 504) return "unavailable";
  return "unknown";
}

const FALLBACK_MESSAGES: Record<ErrorKind, string> = {
  network: "The VisionForge API could not be reached. Check that it is running and try again.",
  unauthorized: "Your session has ended. Sign in again.",
  not_found: "This item was not found or is not available to your account.",
  conflict: "The request conflicts with the current state.",
  validation: "Some values are not valid.",
  capacity: "The service is busy. Wait a moment and try again.",
  unavailable: "The service is temporarily unavailable. Try again shortly.",
  bad_request: "The request was rejected.",
  aborted: "The request was cancelled.",
  unknown: "Something went wrong.",
};

export function fallbackMessage(kind: ErrorKind): string {
  return FALLBACK_MESSAGES[kind];
}

/** Parse FastAPI error bodies: `{detail: string}` or `{detail: ValidationIssue[]}`. */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  const kind = kindForStatus(response.status);
  let detail: unknown = null;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      detail = (body as { detail: unknown }).detail;
    }
  } catch {
    // Non-JSON body: keep the status-based message.
  }
  const retryHeader = response.headers.get("Retry-After");
  const retryAfterSeconds = retryHeader && /^\d+$/.test(retryHeader) ? Number(retryHeader) : null;
  if (Array.isArray(detail)) {
    const issues = detail.filter(isIssue);
    const first = issues[0];
    return new ApiError({
      kind,
      status: response.status,
      issues,
      retryAfterSeconds,
      message: first ? describeIssue(first) : fallbackMessage(kind),
    });
  }
  const message =
    typeof detail === "string" && detail.trim() && kind !== "not_found"
      ? detail
      : fallbackMessage(kind);
  return new ApiError({ kind, status: response.status, message, retryAfterSeconds });
}

function isIssue(value: unknown): value is ValidationIssue {
  return (
    !!value &&
    typeof value === "object" &&
    Array.isArray((value as ValidationIssue).loc) &&
    typeof (value as ValidationIssue).msg === "string"
  );
}

function describeIssue(issue: ValidationIssue): string {
  const field = issue.loc.filter((part) => part !== "body").join(".");
  const message = cleanMessage(issue.msg);
  return field ? `${field}: ${message}` : message;
}

export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  if (error instanceof DOMException && error.name === "AbortError") {
    return new ApiError({ kind: "aborted", message: fallbackMessage("aborted") });
  }
  if (error instanceof TypeError) {
    return new ApiError({ kind: "network", message: fallbackMessage("network") });
  }
  return new ApiError({
    kind: "unknown",
    message: error instanceof Error ? error.message : fallbackMessage("unknown"),
  });
}

export function errorMessage(error: unknown): string {
  return toApiError(error).message;
}
