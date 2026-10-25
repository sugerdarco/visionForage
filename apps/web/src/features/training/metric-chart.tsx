"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { formatMetric } from "@/lib/metrics";

export interface Series {
  id: string;
  label: string;
  /** Epoch → value. Missing epochs are gaps, never zero. */
  points: Map<number, number>;
}

const COLORS = [
  "#1d4ed8",
  "#dc2626",
  "#059669",
  "#d97706",
  "#7c3aed",
  "#0891b2",
  "#db2777",
  "#4b5563",
];
const DASHES = [undefined, "6 3", "2 3", "8 3 2 3", "1 4", "10 4"];

export function seriesStyle(index: number) {
  return { color: COLORS[index % COLORS.length]!, dash: DASHES[index % DASHES.length] };
}

/** Line chart with epoch on x and an accessible data table alternative. */
export function MetricChart({
  title,
  series,
  yLabel,
}: {
  title: string;
  series: Series[];
  yLabel?: string;
}) {
  const epochs = [...new Set(series.flatMap((s) => [...s.points.keys()]))].sort((a, b) => a - b);
  const rows = epochs.map((epoch) => {
    const row: Record<string, number | undefined> = { epoch };
    for (const s of series) row[s.id] = s.points.get(epoch);
    return row;
  });
  return (
    <figure className="rounded-lg border border-line bg-white p-4">
      <figcaption className="mb-2 text-sm font-semibold">{title}</figcaption>
      {rows.length === 0 ? (
        <p className="py-8 text-center text-sm text-muted">No recorded epochs yet.</p>
      ) : (
        <div className="h-64" role="img" aria-label={`${title} by epoch. A data table follows.`}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rows} margin={{ top: 5, right: 16, bottom: 20, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="epoch"
                type="number"
                domain={["dataMin", "dataMax"]}
                allowDecimals={false}
                label={{ value: "Epoch", position: "insideBottom", offset: -10 }}
              />
              <YAxis
                width={70}
                tickFormatter={(v: number) => formatMetric(v)}
                label={yLabel ? { value: yLabel, angle: -90, position: "insideLeft" } : undefined}
              />
              <Tooltip
                formatter={(value) =>
                  formatMetric(typeof value === "number" ? value : Number(value))
                }
                labelFormatter={(e) => `Epoch ${e}`}
              />
              <Legend verticalAlign="top" height={28} />
              {series.map((s, i) => {
                const style = seriesStyle(i);
                return (
                  <Line
                    key={s.id}
                    name={s.label}
                    dataKey={(row: Record<string, number | undefined>) => row[s.id]}
                    stroke={style.color}
                    strokeDasharray={style.dash}
                    strokeWidth={2}
                    dot={rows.length < 40 ? { r: 2 } : false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                );
              })}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
      {rows.length > 0 && (
        <details className="mt-2 text-sm">
          <summary className="cursor-pointer text-xs text-muted">Show data table</summary>
          <div className="mt-2 max-h-64 overflow-auto">
            <table className="w-full text-xs">
              <caption className="sr-only">{title}</caption>
              <thead className="sticky top-0 bg-white text-left text-muted">
                <tr>
                  <th className="py-1 pr-3 font-medium">Epoch</th>
                  {series.map((s) => (
                    <th key={s.id} className="py-1 pr-3 text-right font-medium">
                      {s.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.epoch} className="border-t border-line">
                    <td className="py-1 pr-3 tabular-nums">{row.epoch}</td>
                    {series.map((s) => (
                      <td key={s.id} className="py-1 pr-3 text-right font-mono tabular-nums">
                        {formatMetric(row[s.id])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </figure>
  );
}
