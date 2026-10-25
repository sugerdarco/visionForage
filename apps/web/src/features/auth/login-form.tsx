"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Field, inputClass } from "@/components/ui/field";
import { Notice } from "@/components/ui/states";
import { toApiError } from "@/lib/api/errors";
import { AuthShell } from "./auth-shell";
import { useSession } from "./session";
import { emailError, safeNext } from "./validation";

export function LoginForm() {
  const { signIn } = useSession();
  const router = useRouter();
  const params = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const nextErrors: Record<string, string> = {};
    const emailProblem = emailError(email);
    if (emailProblem) nextErrors.email = emailProblem;
    if (!password) nextErrors.password = "Enter your password.";
    setErrors(nextErrors);
    setFormError(null);
    if (Object.keys(nextErrors).length) return;
    setBusy(true);
    try {
      await signIn(email.trim(), password);
      router.replace(safeNext(params.get("next")));
    } catch (raw) {
      const error = toApiError(raw);
      setFormError(
        error.kind === "unauthorized"
          ? "Email or password is incorrect."
          : error.kind === "validation"
            ? "Email or password is incorrect."
            : error.message,
      );
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthShell
      title="Sign in"
      footer={
        <>
          No account?{" "}
          <Link className="text-accent underline" href="/register">
            Create one
          </Link>
        </>
      }
    >
      {params.get("expired") && (
        <Notice tone="warn" className="mb-4">
          Your session ended. Sign in again to continue.
        </Notice>
      )}
      {params.get("registered") && (
        <Notice tone="success" className="mb-4">
          Account created. Sign in to continue.
        </Notice>
      )}
      <form noValidate onSubmit={submit} className="flex flex-col gap-4">
        <Field label="Email" error={errors.email}>
          {(props) => (
            <input
              {...props}
              type="email"
              autoComplete="email"
              className={inputClass}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          )}
        </Field>
        <Field label="Password" error={errors.password}>
          {(props) => (
            <input
              {...props}
              type="password"
              autoComplete="current-password"
              className={inputClass}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          )}
        </Field>
        {formError && <Notice tone="error">{formError}</Notice>}
        <Button type="submit" variant="primary" busy={busy}>
          Sign in
        </Button>
        <p className="text-xs text-muted">
          Sessions are kept in this tab only. Reloading the page signs you out.
        </p>
      </form>
    </AuthShell>
  );
}
