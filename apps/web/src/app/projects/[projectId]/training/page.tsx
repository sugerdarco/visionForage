"use client";

import { Suspense } from "react";
import { JobsScreen } from "@/features/training/jobs-screen";

export default function TrainingPage() {
  return (
    <Suspense>
      <JobsScreen />
    </Suspense>
  );
}
