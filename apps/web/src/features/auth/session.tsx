"use client";

import { useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { apiClient } from "@/lib/api/client";
import { api } from "@/lib/api/endpoints";
import type { User } from "@/lib/api/types";

interface SessionValue {
  user: User | null;
  /** Set while an explicit or expiry sign-out is redirecting. */
  signingOut: boolean;
  signIn(email: string, password: string): Promise<void>;
  signOut(reason?: "expired"): void;
}

const SessionContext = createContext<SessionValue | null>(null);

/**
 * In-memory session only: tokens are never written to storage, cookies or URLs,
 * so a full page reload requires signing in again.
 */
export function SessionProvider({ children }: { children: React.ReactNode }) {
  const queryClient = useQueryClient();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [signingOut, setSigningOut] = useState(false);

  const clear = useCallback(() => {
    apiClient.setTokens(null);
    // Drop every cached response so another account never sees this one's data.
    queryClient.cancelQueries();
    queryClient.clear();
    setUser(null);
  }, [queryClient]);

  const signOut = useCallback(
    (reason?: "expired") => {
      setSigningOut(true);
      clear();
      const next = encodeURIComponent(window.location.pathname + window.location.search);
      router.replace(reason === "expired" ? `/login?expired=1&next=${next}` : "/login");
    },
    [clear, router],
  );

  useEffect(() => {
    apiClient.setSessionExpiredHandler(() => signOut("expired"));
    return () => apiClient.setSessionExpiredHandler(undefined);
  }, [signOut]);

  const signIn = useCallback(
    async (email: string, password: string) => {
      clear();
      setSigningOut(false);
      const tokens = await api.auth.login(email, password);
      apiClient.setTokens({ accessToken: tokens.access_token, refreshToken: tokens.refresh_token });
      try {
        setUser(await api.auth.me());
      } catch (error) {
        apiClient.setTokens(null);
        throw error;
      }
    },
    [clear],
  );

  const value = useMemo(
    () => ({ user, signingOut, signIn, signOut }),
    [user, signingOut, signIn, signOut],
  );
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionValue {
  const value = useContext(SessionContext);
  if (!value) throw new Error("useSession must be used inside SessionProvider");
  return value;
}

/** Signed-in user for protected screens; redirects to login (keeping the path) otherwise. */
export function useRequiredUser(): User | null {
  const { user, signingOut } = useSession();
  const router = useRouter();
  const pathname = usePathname();
  const search = useSearchParams();
  useEffect(() => {
    if (user || signingOut) return;
    const query = search.toString();
    router.replace(`/login?next=${encodeURIComponent(query ? `${pathname}?${query}` : pathname)}`);
  }, [user, signingOut, router, pathname, search]);
  return user;
}
