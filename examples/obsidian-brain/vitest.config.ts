import { defineConfig } from "vitest/config";

/*
 * No mocks. Tests fall into two buckets:
 *
 *   tests/protocol/    — spin up the real mcp-brain-server-local subprocess
 *                        and hit its REST endpoints with Node fetch. Validates
 *                        the exact response shapes the plugin's BrainClient
 *                        depends on.
 *
 *   tests/e2e/         — launch the real Obsidian desktop app (AppImage) with
 *                        the built plugin installed. A companion harness
 *                        plugin exercises obsidian-brain's commands inside
 *                        Obsidian and writes a JSON report the test asserts
 *                        against. Opt-in via OBSIDIAN_E2E=1 because it
 *                        downloads Obsidian on first run.
 */

export default defineConfig({
	test: {
		include: ["tests/**/*.test.ts"],
		exclude:
			process.env.OBSIDIAN_E2E || process.env.BRAIN_INTEGRATION
				? []
				: ["tests/e2e/**"],
		globals: false,
		testTimeout: 120_000,
		hookTimeout: 120_000,
	},
});
