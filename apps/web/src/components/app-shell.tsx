"use client";

import Link from "next/link";
import { useSession } from "@/features/auth/session";
import { Button } from "./ui/button";

export function AppShell({ children }: { children: React.ReactNode }) {
  const { user, signOut } = useSession();
  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-30 border-b border-slate-800 bg-slate-900 text-white">
        <div className="flex h-12 items-center justify-between gap-4 px-4">
          <Link href="/projects" className="font-semibold tracking-tight">
            VisionForge
          </Link>
          {user && (
            <div className="flex items-center gap-3 text-sm">
              <span className="hidden text-slate-300 sm:inline" title="Signed in">
                {user.email}
              </span>
              <Button
                size="sm"
                variant="ghost"
                className="text-white hover:bg-slate-700"
                onClick={() => signOut()}
              >
                Sign out
              </Button>
            </div>
          )}
        </div>
      </header>
      {children}
    </div>
  );
}
