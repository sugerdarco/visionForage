"use client";

import { Suspense } from "react";
import { VersionsScreen } from "@/features/datasets/versions-screen";

export default function VersionsPage() {
  return (
    <Suspense>
      <VersionsScreen />
    </Suspense>
  );
}
