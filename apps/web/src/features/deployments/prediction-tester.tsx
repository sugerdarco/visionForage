"use client";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Field, inputClass } from "@/components/ui/field";
import { Card } from "@/components/ui/page-header";
import { Notice } from "@/components/ui/states";
import type { Deployment } from "@/lib/api/types";

export function PredictionTester({
  deployment,
  initialKey = "",
}: {
  deployment: Deployment;
  initialKey?: string;
}) {
  const [key, setKey] = useState(initialKey);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  async function predict(event: React.FormEvent) {
    event.preventDefault();
    if (!file || !key.trim() || !deployment.endpoint_url) return;
    if (!["image/jpeg", "image/png"].includes(file.type) || file.size > 20 * 1024 * 1024) {
      setError("Choose a JPEG or PNG no larger than 20 MiB.");
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    const controller = new AbortController();
    abort.current = controller;
    const timeout = setTimeout(() => controller.abort(), 120000);
    try {
      const body = new FormData();
      body.append("image", file);
      const response = await fetch(deployment.endpoint_url, {
        method: "POST",
        headers: { "X-API-Key": key.trim() },
        body,
        signal: controller.signal,
        redirect: "error",
        credentials: "omit",
        cache: "no-store",
      });
      if (!response.ok)
        throw new Error(
          response.status === 401
            ? "The endpoint API key was rejected."
            : `Prediction failed (${response.status}). Check the endpoint status and try again.`,
        );
      setResult(await response.json());
    } catch (raw) {
      setError(
        raw instanceof Error && raw.name === "AbortError"
          ? "Prediction cancelled or timed out. It may have been recorded; check monitoring before retrying."
          : raw instanceof Error
            ? raw.message
            : "Prediction unavailable.",
      );
    } finally {
      clearTimeout(timeout);
      setBusy(false);
    }
  }
  return (
    <Card title="Test prediction">
      <form onSubmit={predict} className="flex flex-col gap-3">
        <Field label="Endpoint API key">
          {(props) => (
            <input
              {...props}
              type="password"
              autoComplete="off"
              className={inputClass}
              value={key}
              onChange={(e) => setKey(e.target.value)}
              required
            />
          )}
        </Field>
        <Field label="Prediction image">
          {(props) => (
            <input
              {...props}
              type="file"
              accept="image/jpeg,image/png"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              required
            />
          )}
        </Field>
        <Button
          type="submit"
          variant="primary"
          busy={busy}
          disabled={deployment.status !== "running" || !file || !key.trim()}
        >
          Predict
        </Button>
      </form>
      {error && (
        <Notice tone="error" className="mt-3">
          {error}
        </Notice>
      )}
      {result !== null && (
        <div className="mt-3">
          <h3 className="font-semibold">Prediction result</h3>
          <pre
            data-testid="prediction-result"
            className="mt-2 max-h-80 overflow-auto rounded bg-slate-50 p-3 text-xs"
          >
            {JSON.stringify(result, null, 2)}
          </pre>
          <p className="mt-2 text-sm text-muted">
            Recorded predictions appear in monitoring after the background collector runs.
          </p>
        </div>
      )}
      <p className="mt-3 text-xs text-muted">
        The key stays in this page’s memory. This request uses the endpoint key, not your account
        token.
      </p>
    </Card>
  );
}
