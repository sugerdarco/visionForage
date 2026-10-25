"use client";

import { Suspense } from "react";
import { NewJobScreen } from "@/features/training/new-job-screen";

export default function NewTrainingPage() {
  return (
    <Suspense>
      <NewJobScreen />
    </Suspense>
  );
}
