import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: "standalone",
  experimental: { cpus: 2 },
  poweredByHeader: false,
  // Do not generate AGENTS.md / CLAUDE.md files into the app directory.
  agentRules: false,
};

export default nextConfig;
