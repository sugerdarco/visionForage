"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/page-header";
import { ErrorState, LoadingState, Notice } from "@/components/ui/states";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import type { Artifact, Evaluation, ModelDetail } from "@/lib/api/types";

const score = (value: number | null | undefined) =>
  value == null ? "Not available" : value.toFixed(4);

export function ModelActions({ model }: { model: ModelDetail }) {
  const cache = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  async function promote() {
    setBusy(true);
    setError(null);
    try {
      await api.models.promote(model.id);
      await Promise.all([
        cache.invalidateQueries({ queryKey: ["model"] }),
        cache.invalidateQueries({ queryKey: ["models", model.project_id] }),
      ]);
    } catch (raw) {
      setError(toApiError(raw).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="mt-4 border-t border-line pt-4">
      <Button variant="primary" busy={busy} disabled={model.is_promoted} onClick={promote}>
        {model.is_promoted ? "Preferred model" : "Use as preferred model"}
      </Button>
      <p className="mt-2 text-xs text-muted">
        Selects the preferred version for this project. Existing deployments keep their model
        version.
      </p>
      {error && (
        <Notice tone="error" className="mt-2">
          {error}
        </Notice>
      )}
    </div>
  );
}

export function ModelResults({ modelId }: { modelId: string }) {
  const report = useQuery({
    queryKey: ["model-evaluation", modelId],
    queryFn: ({ signal }) => api.models.evaluation(modelId, signal),
  });
  const artifacts = useQuery({
    queryKey: ["model-artifacts", modelId],
    queryFn: ({ signal }) => api.models.artifacts(modelId, signal),
  });
  const [downloading, setDownloading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  async function download(artifact: Artifact) {
    setDownloading(artifact.name);
    setError(null);
    try {
      const result = await api.models.download(modelId, artifact.name);
      // A fresh private URL goes directly to storage; no bearer token is forwarded.
      const link = document.createElement("a");
      link.href = result.download_url;
      link.download = result.filename;
      link.rel = "noreferrer";
      link.referrerPolicy = "no-referrer";
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (raw) {
      setError(toApiError(raw).message);
    } finally {
      setDownloading(null);
    }
  }
  return (
    <div className="mt-6 flex min-w-0 flex-col gap-6">
      <Card title="Held-out test evaluation">
        {report.isPending ? (
          <LoadingState label="Loading evaluation" />
        ) : report.isError ? (
          <ErrorState error={report.error} onRetry={() => report.refetch()} />
        ) : (
          <EvaluationReport report={report.data} />
        )}
      </Card>
      <Card title="Model artifacts">
        {artifacts.isPending ? (
          <LoadingState label="Loading artifacts" />
        ) : artifacts.isError ? (
          <ErrorState error={artifacts.error} onRetry={() => artifacts.refetch()} />
        ) : (
          <ul className="flex flex-col gap-3">
            {artifacts.data.map((artifact) => (
              <li key={artifact.name} className="rounded border border-line p-3">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span>
                    {artifact.filename}{" "}
                    <span className="text-xs text-muted">
                      ({artifact.bytes.toLocaleString()} bytes)
                    </span>
                  </span>
                  <Button
                    busy={downloading === artifact.name}
                    disabled={downloading !== null}
                    onClick={() => download(artifact)}
                  >
                    Download {artifact.name}
                  </Button>
                </div>
                <p className="mt-2 break-all font-mono text-xs text-muted">
                  SHA-256: {artifact.sha256}
                </p>
              </li>
            ))}
          </ul>
        )}
        {error && (
          <Notice tone="error" className="mt-3">
            {error}
          </Notice>
        )}
        <p className="mt-3 text-xs text-muted">
          Downloads use private links valid for five minutes. Click again for a fresh link.
        </p>
      </Card>
    </div>
  );
}

function EvaluationReport({ report }: { report: Evaluation }) {
  const overall =
    report.task_type === "classification"
      ? {
          Accuracy: report.metrics.accuracy,
          "Macro F1": report.metrics.macro_f1,
          "ROC-AUC": report.metrics.roc_auc,
          Loss: report.metrics.loss,
        }
      : {
          "mAP@0.5": report.metrics.map50,
          "mAP@0.5:0.95": report.metrics.map50_95,
          Precision: report.metrics.fixed_threshold_micro.precision,
          Recall: report.metrics.fixed_threshold_micro.recall,
        };
  const rows =
    report.task_type === "classification"
      ? Object.entries(report.metrics.per_class).map(([name, values]) => ({
          name,
          ...values,
          extra: values.f1,
        }))
      : report.metrics.per_class.map((values) => ({
          name: values.class_name,
          ...values,
          extra: values.ap50,
        }));
  return (
    <>
      <p className="mb-3 text-sm text-muted">
        Saved checkpoint epoch {report.checkpoint_epoch}. Scores use a 0–1 scale; missing values are
        not zero.
      </p>
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {Object.entries(overall).map(([name, value]) => (
          <div key={name}>
            <dt className="text-xs text-muted">{name}</dt>
            <dd className="font-mono">{score(value)}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <caption className="mb-2 text-left font-semibold">Per-class results</caption>
          <thead>
            <tr>
              {[
                "Class",
                "Support",
                "Precision",
                "Recall",
                report.task_type === "classification" ? "F1" : "AP@0.5",
              ].map((label) => (
                <th key={label} className="p-2">
                  {label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name} className="border-t border-line">
                <th scope="row" className="p-2">
                  {row.name}
                </th>
                <td className="p-2">{row.support}</td>
                <td className="p-2">{score(row.precision)}</td>
                <td className="p-2">{score(row.recall)}</td>
                <td className="p-2">{score(row.extra)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {report.task_type === "classification" && (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <caption className="mb-2 text-left font-semibold">
              Confusion matrix — true rows, predicted columns
            </caption>
            <thead>
              <tr>
                <th className="p-2">True / predicted</th>
                {report.metrics.confusion_matrix.class_names.map((name) => (
                  <th key={name} className="p-2">
                    {name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {report.metrics.confusion_matrix.counts.map((row, index) => (
                <tr key={index} className="border-t border-line">
                  <th scope="row" className="p-2">
                    {report.metrics.confusion_matrix.class_names[index]}
                  </th>
                  {row.map((value, column) => (
                    <td key={column} className="p-2">
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {report.task_type === "detection" && (
        <p className="mt-3 text-xs text-muted">
          Precision/recall confidence: {report.metric_settings.precision_recall_confidence};
          matching IoU: {report.metric_settings.precision_recall_iou}; NMS IoU:{" "}
          {report.metric_settings.nms_iou}.
        </p>
      )}
      <div className="mt-4 border-t border-line pt-4 text-sm">
        <h3 className="font-semibold">ONNX verification</h3>
        <p>
          Passed on {report.parity.samples} test images, batch sizes{" "}
          {report.parity.batch_sizes.join(", ")}.
        </p>
        <p>
          Maximum absolute error: {report.parity.max_absolute_error}; tolerance:{" "}
          {report.parity.absolute_tolerance}.
        </p>
        {report.parity.box_absolute_tolerance != null && (
          <p>
            Box tolerance: {report.parity.box_absolute_tolerance} input pixels; score tolerance:{" "}
            {report.parity.score_absolute_tolerance}.
          </p>
        )}
        <h3 className="mt-3 font-semibold">Worker CPU model-only latency</h3>
        <p>
          Mean {report.latency.mean_ms.toFixed(2)} ms · p95 {report.latency.p95_ms.toFixed(2)} ms
        </p>
        <p className="text-xs text-muted">
          {report.latency.measured_runs} measured runs after {report.latency.warmup_runs} warmups,
          batch {report.latency.batch_size}. Excludes image decoding, preprocessing, HTTP and
          network time.
        </p>
      </div>
    </>
  );
}
