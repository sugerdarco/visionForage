import { Suspense } from "react";
import { DeploymentsScreen } from "@/features/deployments/deployments-screen";
import { LoadingState } from "@/components/ui/states";
export default function Page() {
  return (
    <Suspense fallback={<LoadingState />}>
      <DeploymentsScreen />
    </Suspense>
  );
}
