/*
 * Brain E2E Harness — runs inside the real Obsidian app. Waits for
 * obsidian-brain to register its commands, exercises the user-visible
 * surface (commands, views, settings) against a real brain subprocess
 * wired up by the Node orchestrator, and writes a JSON report at
 * `$OBSIDIAN_BRAIN_E2E_REPORT` (a path passed through the test vault
 * config), then quits Obsidian.
 *
 * Kept intentionally dependency-free — only `obsidian` and `node:fs`.
 */

import { Plugin, TFile } from "obsidian";
import { writeFileSync } from "fs";

interface CheckResult {
	name: string;
	ok: boolean;
	detail?: string;
}

interface Report {
	version: string;
	startedAt: string;
	finishedAt: string;
	checks: CheckResult[];
	passed: number;
	failed: number;
}

function waitFor<T>(
	fn: () => T | null | undefined,
	timeoutMs: number,
	label: string,
): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		const deadline = Date.now() + timeoutMs;
		const tick = (): void => {
			const v = fn();
			if (v) return resolve(v);
			if (Date.now() > deadline) {
				return reject(new Error(`timeout waiting for ${label}`));
			}
			setTimeout(tick, 50);
		};
		tick();
	});
}

export default class HarnessPlugin extends Plugin {
	async onload(): Promise<void> {
		// Use the workspace "layout-ready" hook so Obsidian is fully booted.
		this.app.workspace.onLayoutReady(() => {
			void this.run().catch((e) => {
				console.error("[harness] fatal", e);
				this.writeReport({
					checks: [
						{ name: "harness", ok: false, detail: (e as Error).message },
					],
				});
				this.quit();
			});
		});
	}

	private async run(): Promise<void> {
		const startedAt = new Date().toISOString();
		const checks: CheckResult[] = [];
		const addCheck = async (
			name: string,
			body: () => Promise<string | undefined> | string | undefined,
		): Promise<void> => {
			try {
				const detail = await body();
				checks.push({ name, ok: true, detail });
			} catch (e) {
				checks.push({ name, ok: false, detail: (e as Error).message });
			}
		};

		await addCheck("brain plugin is loaded", () => {
			const plugins = (
				this.app as unknown as {
					plugins: { getPlugin(id: string): unknown | null };
				}
			).plugins;
			const p = plugins.getPlugin("obsidian-brain");
			if (!p) throw new Error("obsidian-brain plugin not found");
			return "obsidian-brain resolved";
		});

		await addCheck("commands registered", () => {
			const cmd = lookupCommand(this.app, "obsidian-brain:brain-search");
			if (!cmd) throw new Error("semantic-search command missing");
			for (const id of [
				"obsidian-brain:brain-related-panel",
				"obsidian-brain:brain-bulk-sync",
				"obsidian-brain:brain-index-current",
				"obsidian-brain:brain-dpo-mark-chosen",
				"obsidian-brain:brain-graph-overlay-apply",
			]) {
				if (!lookupCommand(this.app, id))
					throw new Error(`missing command ${id}`);
			}
			return "all expected commands present";
		});

		await addCheck("brain health reachable via plugin settings", async () => {
			const status = await probeBrainStatusBar(this.app);
			if (!/memories/i.test(status))
				throw new Error(`unexpected status: ${JSON.stringify(status)}`);
			return status;
		});

		await addCheck("index current note", async () => {
			const file = await ensureTestNote(
				this.app,
				"E2E-note.md",
				"---\nbrain-category: e2e\n---\nThis is an end-to-end test note.",
			);
			await (this.app.workspace.getLeaf(false)).openFile(file);
			await invokeCommand(this.app, "obsidian-brain:brain-index-current");
			// Allow the async indexer to complete.
			await new Promise((r) => setTimeout(r, 1500));
			return `indexed ${file.path}`;
		});

		await addCheck("semantic search returns our new memory", async () => {
			const api = (
				this.app as unknown as {
					plugins: {
						plugins: Record<
							string,
							{ brain?: { search?: (q: string, k: number) => Promise<unknown> } }
						>;
					};
				}
			).plugins.plugins["obsidian-brain"];
			const brain = api?.brain;
			if (!brain || typeof brain.search !== "function")
				throw new Error("BrainClient.search not accessible on plugin instance");
			const resp = (await brain.search("end-to-end test note", 5)) as {
				results: Array<{ category: string }>;
			};
			if (!resp.results.length)
				throw new Error("no search results");
			if (!resp.results.some((r) => r.category === "e2e"))
				throw new Error(
					`expected a result with category=e2e, got ${JSON.stringify(resp.results.map((r) => r.category))}`,
				);
			return `${resp.results.length} results, top category ${resp.results[0].category}`;
		});

		await addCheck("bulk sync completes without throwing", async () => {
			const api = (
				this.app as unknown as {
					plugins: {
						plugins: Record<
							string,
							{
								indexer?: {
									bulkSync?: (r: {
										update: () => void;
										cancelled: () => boolean;
									}) => Promise<{ indexed: number; failed: number }>;
								};
							}
						>;
					};
				}
			).plugins.plugins["obsidian-brain"];
			const indexer = api?.indexer;
			if (!indexer?.bulkSync)
				throw new Error("indexer.bulkSync not accessible on plugin instance");
			const r = await indexer.bulkSync({
				update: () => undefined,
				cancelled: () => false,
			});
			if (r.failed > 0) throw new Error(`${r.failed} failures`);
			return `indexed=${r.indexed} failed=${r.failed}`;
		});

		await addCheck("graph overlay writes color groups", async () => {
			await invokeCommand(this.app, "obsidian-brain:brain-graph-overlay-apply");
			await new Promise((r) => setTimeout(r, 500));
			const raw = await this.app.vault.adapter.read(".obsidian/graph.json");
			const obj = JSON.parse(raw) as { colorGroups: Array<{ query: string }> };
			const brainGroups = obj.colorGroups.filter((g) =>
				g.query.startsWith("tag:#brain/"),
			);
			if (!brainGroups.length)
				throw new Error("no #brain/* color groups written");
			return `${brainGroups.length} groups`;
		});

		this.writeReport({
			startedAt,
			checks,
		});
		this.quit();
	}

	private writeReport(partial: {
		startedAt?: string;
		checks: CheckResult[];
	}): void {
		const report: Report = {
			version: this.manifest.version,
			startedAt: partial.startedAt ?? new Date().toISOString(),
			finishedAt: new Date().toISOString(),
			checks: partial.checks,
			passed: partial.checks.filter((c) => c.ok).length,
			failed: partial.checks.filter((c) => !c.ok).length,
		};
		const path = process.env.OBSIDIAN_BRAIN_E2E_REPORT;
		if (!path) {
			console.warn("[harness] no OBSIDIAN_BRAIN_E2E_REPORT set");
			return;
		}
		try {
			writeFileSync(path, JSON.stringify(report, null, 2) + "\n", "utf8");
			console.log(`[harness] wrote report → ${path}`);
		} catch (e) {
			console.error("[harness] write failed", e);
		}
	}

	private quit(): void {
		const remote = (globalThis as unknown as {
			require?: (m: string) => { remote?: { app?: { quit?: () => void } } };
		}).require;
		try {
			const electron = remote ? remote("electron") : null;
			electron?.remote?.app?.quit?.();
		} catch {
			/* fall back */
		}
		// Last resort — browser-level close.
		setTimeout(() => {
			try {
				(window as unknown as { close: () => void }).close();
			} catch {
				/* ignore */
			}
		}, 250);
	}
}

function lookupCommand(app: unknown, id: string): unknown {
	const a = app as {
		commands: { commands: Record<string, unknown> };
	};
	return a.commands.commands[id] ?? null;
}

async function invokeCommand(app: unknown, id: string): Promise<void> {
	const a = app as {
		commands: {
			executeCommandById: (id: string) => unknown;
		};
	};
	a.commands.executeCommandById(id);
}

async function probeBrainStatusBar(app: unknown): Promise<string> {
	void app;
	// Wait for the plugin's status bar item to settle into a real readout —
	// the plugin seeds "Brain: …" on load, then replaces it with
	// "Brain: <backend> · N memories" after `refreshStatus()` fires.
	return waitFor<string>(
		() => {
			const bars = document.querySelectorAll<HTMLElement>(".brain-status-bar, .status-bar-item");
			const brainBar = Array.from(bars).find((el) => {
				const t = el.textContent ?? "";
				const lower = t.toLowerCase();
				if (!lower.includes("brain")) return false;
				// Skip the initial placeholder — keep waiting for a real value.
				if (lower.trim() === "brain: …" || lower.trim() === "brain:") return false;
				return true;
			});
			return brainBar?.textContent ?? null;
		},
		12000,
		"brain status bar text",
	);
}

async function ensureTestNote(
	app: { vault: { getAbstractFileByPath(p: string): unknown; create(p: string, c: string): Promise<TFile> } },
	name: string,
	content: string,
): Promise<TFile> {
	const existing = app.vault.getAbstractFileByPath(name);
	if (existing instanceof TFile) return existing;
	return app.vault.create(name, content);
}
