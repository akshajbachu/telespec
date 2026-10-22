import { expect, test } from "@playwright/test";
test("demo renders spectral evidence and exports", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("See the cycles.")).toBeVisible();
  await page.getByRole("button", { name: "Load scientific demo" }).click();
  await expect(page.getByText("Analysis complete")).toBeVisible({ timeout: 30000 });
  await expect(page.getByLabel("Generalized Lomb-Scargle periodogram")).toBeVisible();
  await expect(page.getByText("Interpretable findings")).toBeVisible();
  await expect(page.getByRole("link", { name: "JSON" })).toHaveAttribute("href", /api\/export/);
});
