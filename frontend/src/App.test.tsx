import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import App from "./App";

describe("App", () => {
  it("offers upload and demo entry points", () => {
    render(<App />);
    expect(screen.getByText(/See the cycles/i)).toBeTruthy();
    expect(screen.getByRole("button", { name: /Load scientific demo/i })).toBeTruthy();
    expect(screen.getByLabelText(/Telemetry file/i)).toBeTruthy();
  });
});
