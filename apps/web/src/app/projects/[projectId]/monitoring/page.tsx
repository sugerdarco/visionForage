import { Suspense } from "react";
import { MonitoringScreen } from "@/features/deployments/monitoring-screen";
import { LoadingState } from "@/components/ui/states";
export default function Page() {
  return (
    <Suspense fallback={<LoadingState />}>
      <MonitoringScreen />
    </Suspense>
  );
}
