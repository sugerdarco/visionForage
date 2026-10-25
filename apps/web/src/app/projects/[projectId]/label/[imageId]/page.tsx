"use client";

import { Suspense } from "react";
import { LabelScreen } from "@/features/annotations/label-screen";

export default function LabelPage() {
  return (
    <Suspense>
      <LabelScreen />
    </Suspense>
  );
}
