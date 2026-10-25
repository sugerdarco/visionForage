"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { toApiError } from "@/lib/api/errors";
import { SessionProvider } from "@/features/auth/session";

function makeQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Signed image URLs expire after 300 s; keep cached data comfortably fresher.
        staleTime: 30_000,
        refetchOnWindowFocus: false,
        retry: (count, error) => {
          const kind = toApiError(error).kind;
          return count < 2 && (kind === "network" || kind === "unavailable");
        },
      },
      // Mutations are never retried: several POSTs are not idempotent.
      mutations: { retry: false },
    },
  });
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(makeQueryClient);
  return (
    <QueryClientProvider client={queryClient}>
      <SessionProvider>{children}</SessionProvider>
    </QueryClientProvider>
  );
}
