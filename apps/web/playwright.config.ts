import { defineConfig, devices } from "@playwright/test";

/**
 * Browser tests run against the real local API (http://127.0.0.1:8000) and the
 * app on http://127.0.0.1:3000. They never start GPU training: submitted jobs are
 * cancelled while queued. See README "Verification".
 */
export default defineConfig({
  testDir: "tests/e2e",
  timeout: 120_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    viewport: { width: 1366, height: 860 },
    ...(process.env.E2E_CHANNEL === "bundled" ? {} : { channel: "chrome" }),
  },
  projects: [
    {
      name: "chrome",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1366, height: 860 } },
    },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://127.0.0.1:3000/login",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
