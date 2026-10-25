import { errorMessage, toApiError } from "@/lib/api/errors";
import { Button, Spinner } from "./button";

export function LoadingState({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-line bg-panel p-6 text-muted">
      <Spinner />
      <span role="status">{label}…</span>
    </div>
  );
}

export function EmptyState({
  title,
  children,
  action,
}: {
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-line bg-panel p-8 text-center">
      <h3 className="text-base font-semibold">{title}</h3>
      {children && <div className="mx-auto mt-2 max-w-prose text-sm text-muted">{children}</div>}
      {action && <div className="mt-4 flex justify-center">{action}</div>}
    </div>
  );
}

export function ErrorState({
  error,
  onRetry,
  title,
}: {
  error: unknown;
  onRetry?: () => void;
  title?: string;
}) {
  const apiError = toApiError(error);
  const heading =
    title ??
    (apiError.kind === "not_found"
      ? "Not found"
      : apiError.kind === "network"
        ? "API unavailable"
        : "Could not load this data");
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-5">
      <h3 className="font-semibold text-danger">{heading}</h3>
      <p className="mt-1 text-sm text-red-900">{errorMessage(error)}</p>
      {onRetry && apiError.kind !== "not_found" && (
        <Button className="mt-3" size="sm" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

type Tone = "info" | "warn" | "error" | "success";

const TONES: Record<Tone, string> = {
  info: "border-blue-200 bg-blue-50 text-blue-950",
  warn: "border-amber-300 bg-amber-50 text-amber-950",
  error: "border-red-200 bg-red-50 text-red-950",
  success: "border-green-200 bg-green-50 text-green-950",
};

export function Notice({
  tone = "info",
  title,
  children,
  className = "",
}: {
  tone?: Tone;
  title?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      className={`rounded-md border px-4 py-3 text-sm ${TONES[tone]} ${className}`}
    >
      {title && <p className="font-semibold">{title}</p>}
      {children && <div className={title ? "mt-1" : ""}>{children}</div>}
    </div>
  );
}
