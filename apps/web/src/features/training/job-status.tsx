import { Badge } from "@/components/ui/badge";
import type { Job, JobStatus } from "@/lib/api/types";

const STATUS: Record<
  JobStatus,
  { label: string; tone: "neutral" | "blue" | "green" | "red" | "amber" }
> = {
  queued: { label: "Queued", tone: "neutral" },
  running: { label: "Running", tone: "blue" },
  succeeded: { label: "Succeeded", tone: "green" },
  failed: { label: "Failed", tone: "red" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};

export function JobStatusBadge({ job }: { job: Pick<Job, "status" | "cancel_requested_at"> }) {
  const s = STATUS[job.status];
  return (
    <span className="inline-flex flex-wrap gap-1">
      <Badge tone={s.tone}>{s.label}</Badge>
      {job.status === "running" && job.cancel_requested_at && (
        <Badge tone="amber">Cancellation requested</Badge>
      )}
    </span>
  );
}

export const ARCH_LABEL: Record<string, string> = { resnet18: "ResNet-18", yolov8s: "YOLOv8s" };
