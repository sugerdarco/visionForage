"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { useSession } from "@/features/auth/session";

export default function Home() {
  const { user } = useSession();
  const router = useRouter();
  useEffect(() => {
    router.replace(user ? "/projects" : "/login");
  }, [user, router]);
  return null;
}
