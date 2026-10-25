"use client";

import { Suspense } from "react";
import { AppShell } from "@/components/app-shell";
import { LoadingState } from "@/components/ui/states";
import { useRequiredUser } from "@/features/auth/session";

export default function ProjectsLayout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense>
      <SignedIn>{children}</SignedIn>
    </Suspense>
  );
}

function SignedIn({ children }: { children: React.ReactNode }) {
  const user = useRequiredUser();
  if (!user) {
    return (
      <div className="p-6">
        <LoadingState label="Checking your session" />
      </div>
    );
  }
  // Keying by user guarantees no component state survives an account switch.
  return <AppShell key={user.id}>{children}</AppShell>;
}
