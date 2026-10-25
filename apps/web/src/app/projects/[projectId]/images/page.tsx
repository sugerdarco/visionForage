"use client";

import { Suspense } from "react";
import { ImagesScreen } from "@/features/images/images-screen";

export default function ImagesPage() {
  return (
    <Suspense>
      <ImagesScreen />
    </Suspense>
  );
}
