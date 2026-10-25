"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Field, inputClass } from "@/components/ui/field";
import { Notice } from "@/components/ui/states";
import { api } from "@/lib/api/endpoints";
import { toApiError } from "@/lib/api/errors";
import { AuthShell } from "./auth-shell";
import { emailError, passwordError } from "./validation";

export function RegisterForm() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [formError, setFormError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    const next: Record<string, string> = {};
    const emailProblem = emailError(email);
    if (emailProblem) next.email = emailProblem;
    const passwordProblem = passwordError(password);
    if (passwordProblem) next.password = passwordProblem;
    if (!next.password && confirm !== password) next.confirm = "Passwords do not match.";
    setErrors(next);
    setFormError(null);
    if (Object.keys(next).length) return;
    setBusy(true);
    try {
      await api.auth.register(email.trim(), password);
      // Registration does not sign in; continue on the login form.
      router.push("/login?registered=1");
    } catch (raw) {
      const error = toApiError(raw);
      if (error.kind === "conflict") setFormError("An account with this email already exists.");
      else if (error.kind === "validation") setErrors(error.fieldErrors());
      else setFormError(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthShell
      title="Create account"
      footer={
        <>
          Already registered?{" "}
          <Link className="text-accent underline" href="/login">
            Sign in
          </Link>
        </>
      }
    >
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
        <Field
          label="Password"
          error={errors.password}
          hint="At least 12 characters and no more than 72 bytes. Spaces count."
        >
          {(props) => (
            <input
              {...props}
              type="password"
              autoComplete="new-password"
              className={inputClass}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          )}
        </Field>
        <Field label="Confirm password" error={errors.confirm}>
          {(props) => (
            <input
              {...props}
              type="password"
              autoComplete="new-password"
              className={inputClass}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
            />
          )}
        </Field>
        {formError && <Notice tone="error">{formError}</Notice>}
        <Button type="submit" variant="primary" busy={busy}>
          Create account
        </Button>
      </form>
    </AuthShell>
  );
}
