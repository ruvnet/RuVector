/*
 * Real Obsidian E2E — gated on $OBSIDIAN_E2E=1 because it downloads a real
 * Obsidian AppImage on first run and drives a real windowed app. Uses
 * xvfb-run automatically when no $DISPLAY is set (requires xvfb installed).
 *
 * Typical usage (after `cargo build --release -p mcp-brain-server --features local`):
 *
 *   sudo apt install xvfb
 *   OBSIDIAN_E2E=1 npm test -- tests/e2e
 */

import { describe, expect, it } from "vitest";
import { resolve } from "node:path";
import { runObsidianE2E } from "./obsidian-runner";

const RUN = process.env.OBSIDIAN_E2E === "1";

describe.skipIf(!RUN)("real Obsidian + obsidian-brain plugin (E2E)", () => {
	it(
		"loads plugin, indexes a note, searches, bulk syncs, applies graph overlay",
		async () => {
			const result = await runObsidianE2E({
				pluginDir: resolve(__dirname, "..", ".."),
			});
			const failed = result.report.checks.filter((c) => !c.ok);
			if (failed.length > 0) {
				throw new Error(
					"harness reported failures:\n" +
						failed.map((c) => `  ✗ ${c.name}: ${c.detail}`).join("\n"),
				);
			}
			// Checks:
			//   1 brain plugin loaded
			//   2 base commands registered
			//   3 status bar populated
			//   4 index current note
			//   5 brain.search roundtrip
			//   6 bulk sync
			//   7 graph overlay
			//   8 pi commands registered
			//   9 pi status roundtrip (only when BRAIN_API_KEY plumbed)
			//  10 phase-4 commands registered (qa/ops/selection/publish/recall/queue)
			//  11 offline queue API accessible
			const minExpected = process.env.BRAIN_API_KEY ? 11 : 10;
			expect(result.report.passed).toBeGreaterThanOrEqual(minExpected);
		},
		300_000,
	);
});
