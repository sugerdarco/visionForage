const DEFAULT_API_BASE = "http://127.0.0.1:8000";

function apiOrigin(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim() || DEFAULT_API_BASE;
  return configured.replace(/\/+$/, "");
}

/** The only place that appends the API version prefix. */
export const API_ORIGIN = apiOrigin();
export const API_BASE = `${API_ORIGIN}/api/v1`;
