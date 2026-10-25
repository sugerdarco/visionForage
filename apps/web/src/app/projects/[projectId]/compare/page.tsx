"use client";

import { Suspense } from "react";
import { CompareScreen } from "@/features/training/compare-screen";

export default function ComparePage() {
  return (
    <Suspense>
      <CompareScreen />
    </Suspense>
  );
}
