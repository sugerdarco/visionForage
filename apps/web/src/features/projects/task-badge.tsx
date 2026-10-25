import { Badge } from "@/components/ui/badge";

export function TaskBadge({ task }: { task: string }) {
  return (
    <Badge tone={task === "detection" ? "violet" : "blue"}>
      {task === "detection"
        ? "Object detection"
        : task === "classification"
          ? "Classification"
          : task}
    </Badge>
  );
}
