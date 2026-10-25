"use client";

import { Suspense } from "react";
import { ModelsScreen } from "@/features/models/models-screen";

export default function ModelsPage() {
  return (
    <Suspense>
      <ModelsScreen />
    </Suspense>
  );
}
