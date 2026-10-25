import type { Metric } from "@/lib/api/types";

/** Epoch → raw metric key → value. Keys keep detector punctuation such as `val_mAP50(B)`. */
export type MetricHistory = Map<number, Record<string, number>>;

/** Same mapping as the API's SSE stream (services/api/vf_api/job_events.py). */
export function streamKey(split: string, name: string): string {
  return split === "train" && name === "learning_rate" ? "lr" : `${split}_${name}`;
}

export function historyFromRows(rows: readonly Metric[]): MetricHistory {
  const history: MetricHistory = new Map();
  for (const row of rows) {
    const epoch = history.get(row.epoch) ?? {};
    epoch[streamKey(row.split, row.name)] = row.value;
    history.set(row.epoch, epoch);
  }
  return history;
}

/** Upsert by epoch and key: replayed epochs replace their values instead of duplicating points. */
export function upsertEpoch(
  history: MetricHistory,
  epoch: number,
  metrics: Record<string, number>,
): MetricHistory {
  const next = new Map(history);
  next.set(epoch, { ...(history.get(epoch) ?? {}), ...metrics });
  return next;
}

export function metricKeys(history: MetricHistory): string[] {
  const keys = new Set<string>();
  for (const values of history.values()) for (const key of Object.keys(values)) keys.add(key);
  return [...keys].sort();
}

export type MetricGroup = "loss" | "quality" | "learning_rate" | "other";

export function metricGroup(key: string): MetricGroup {
  if (key === "lr" || key.startsWith("train_lr/") || key.startsWith("lr/")) return "learning_rate";
  if (/loss$/.test(key)) return "loss";
  if (/(accuracy|mAP|precision|recall|fitness)/.test(key)) return "quality";
  return "other";
}

const NAME_LABELS: Record<string, string> = {
  loss: "loss",
  accuracy: "accuracy",
  fitness: "fitness (mAP50-95 based)",
  box_loss: "box loss",
  cls_loss: "class loss",
  dfl_loss: "DFL loss",
  "precision(B)": "precision",
  "recall(B)": "recall",
  "mAP50(B)": "mAP@0.5",
  "mAP50-95(B)": "mAP@0.5:0.95",
};

/** Human label for a raw key; the raw key stays the data identity. */
export function metricLabel(key: string): string {
  if (key === "lr") return "Learning rate";
  const lr = /^(?:train_)?lr\/pg(\d+)$/.exec(key);
  if (lr) return `Learning rate (group ${lr[1]})`;
  const match = /^(train|val)_(.+)$/.exec(key);
  if (!match) return key;
  const split = match[1] === "train" ? "Train" : "Validation";
  const name = match[2] as string;
  return `${split} ${NAME_LABELS[name] ?? name}`;
}

export function formatMetric(value: number | undefined | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return "Not available";
  if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(2);
  return value.toFixed(4);
}

/** Rows for charts and tables; a missing metric stays undefined (a gap), never zero. */
export function historyRows(
  history: MetricHistory,
): ({ epoch: number } & Record<string, number>)[] {
  return [...history.entries()]
    .sort(([a], [b]) => a - b)
    .map(([epoch, values]) => ({ epoch, ...values }));
}
