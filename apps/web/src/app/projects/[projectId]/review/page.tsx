import { Suspense } from "react";
import { ReviewScreen } from "@/features/review/review-screen";
import { LoadingState } from "@/components/ui/states";
export default function Page() {
  return (
    <Suspense fallback={<LoadingState />}>
      <ReviewScreen />
    </Suspense>
  );
}
