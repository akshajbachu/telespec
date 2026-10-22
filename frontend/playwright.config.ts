import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://127.0.0.1:8765", trace: "on-first-retry" },
  webServer: {
    command: "cd .. && uv run telespec view --no-open",
    url: "http://127.0.0.1:8765/api/health",
    reuseExistingServer: true,
    timeout: 120000
  }
});
