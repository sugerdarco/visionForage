"use client";

import {
  metricGroup,
  metricKeys,
  metricLabel,
  type MetricGroup,
  type MetricHistory,
} from "@/lib/metrics";
import { MetricChart, type Series } from "./metric-chart";

const GROUPS: { group: MetricGroup; title: string; yLabel?: string }[] = [
  { group: "loss", title: "Loss", yLabel: "loss" },
  { group: "quality", title: "Validation quality (raw 0–1 scale)" },
  { group: "learning_rate", title: "Learning rate" },
  { group: "other", title: "Other metrics" },
];

export function seriesFor(
  history: MetricHistory,
  key: string,
  id = key,
  label = metricLabel(key),
): Series {
  const points = new Map<number, number>();
  for (const [epoch, values] of history) {
    const value = values[key];
    if (typeof value === "number" && Number.isFinite(value)) points.set(epoch, value);
  }
  return { id, label, points };
}

/** One chart per metric family, built from whatever raw keys the run recorded. */
export function MetricPanels({ history }: { history: MetricHistory }) {
  const keys = metricKeys(history);
  if (keys.length === 0) {
    return (
      <p className="rounded-lg border border-dashed border-line bg-white p-6 text-center text-sm text-muted">
        No epochs recorded yet.
      </p>
    );
  }
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {GROUPS.map(({ group, title, yLabel }) => {
        const groupKeys = keys.filter((k) => metricGroup(k) === group);
        if (groupKeys.length === 0) return null;
        return (
          <MetricChart
            key={group}
            title={title}
            yLabel={yLabel}
            series={groupKeys.map((k) => seriesFor(history, k))}
          />
        );
      })}
    </div>
  );
}
