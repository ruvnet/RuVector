import { TFile, Notice, Plugin, parseYaml } from "obsidian";
import { BrainClient, BrainError } from "./brain";
import type { BrainSettings } from "./settings";
import type { OfflineQueue } from "./offline-queue";

/**
 * Map content_hash → memory id so the plugin can tell Obsidian files
 * apart from brain memories when it sees a search result.
 * Persisted under data.json via the plugin.
 */
export interface IndexState {
	/** note path → last indexed content hash */
	pathToHash: Record<string, string>;
	/** content hash → memory id */
	hashToId: Record<string, string>;
	/** memory id → note path (reverse lookup for search results) */
	idToPath: Record<string, string>;
	/** last-indexed epoch seconds */
	lastSync: number;
}

export const EMPTY_INDEX_STATE: IndexState = {
	pathToHash: {},
	hashToId: {},
	idToPath: {},
	lastSync: 0,
};

export interface ProgressReporter {
	update(done: number, total: number, note?: string): void;
	cancelled(): boolean;
}

async function hashContent(input: string): Promise<string> {
	// Obsidian runs in Chromium / Node — both expose SubtleCrypto.
	const enc = new TextEncoder().encode(input);
	const buf = await crypto.subtle.digest("SHA-256", enc);
	const b = new Uint8Array(buf);
	let out = "";
	for (let i = 0; i < b.length; i++) out += b[i].toString(16).padStart(2, "0");
	return out;
}

/**
 * Parse `brain-category` (or `category`) from the note's frontmatter if present.
 * We hand-parse rather than lean on MetadataCache to work even before the cache
 * has populated for newly-created files.
 */
export function extractCategory(content: string, fallback: string): string {
	if (!content.startsWith("---")) return fallback;
	const end = content.indexOf("\n---", 3);
	if (end < 0) return fallback;
	const yaml = content.slice(3, end);
	try {
		const fm = parseYaml(yaml) as Record<string, unknown> | null;
		if (!fm) return fallback;
		const cat = fm["brain-category"] ?? fm["category"];
		if (typeof cat === "string" && cat.trim()) return cat.trim();
	} catch {
		/* fall through */
	}
	return fallback;
}

/** Strip frontmatter from a note's raw text before indexing. */
export function stripFrontmatter(content: string): string {
	if (!content.startsWith("---")) return content;
	const end = content.indexOf("\n---", 3);
	if (end < 0) return content;
	return content.slice(end + 4).replace(/^\s+/, "");
}

export class Indexer {
	private debounceTimers = new Map<string, number>();
	state: IndexState = { ...EMPTY_INDEX_STATE };
	queue: OfflineQueue | null = null;

	constructor(
		private plugin: Plugin,
		private brain: BrainClient,
		private settings: BrainSettings,
	) {}

	setQueue(queue: OfflineQueue): void {
		this.queue = queue;
	}

	setState(s: IndexState): void {
		this.state = s;
	}

	/** Register/unregister debounced vault listeners according to settings. */
	configureAutoIndex(): void {
		// The plugin re-registers events via this.plugin.registerEvent in main.ts;
		// here we only track the debounce table.
		if (!this.settings.autoIndex) {
			this.debounceTimers.forEach((t) => window.clearTimeout(t));
			this.debounceTimers.clear();
		}
	}

	queueIndex(file: TFile): void {
		if (!this.settings.autoIndex) return;
		if (file.extension !== "md") return;
		const prev = this.debounceTimers.get(file.path);
		if (prev) window.clearTimeout(prev);
		const handle = window.setTimeout(() => {
			this.debounceTimers.delete(file.path);
			void this.indexFile(file).catch((e) => {
				console.warn("[obsidian-brain] auto-index failed", file.path, e);
			});
		}, Math.max(250, this.settings.autoIndexDebounceMs));
		this.debounceTimers.set(file.path, handle);
	}

	cancel(path: string): void {
		const t = this.debounceTimers.get(path);
		if (t) {
			window.clearTimeout(t);
			this.debounceTimers.delete(path);
		}
	}

	handleRename(oldPath: string, newPath: string): void {
		const h = this.state.pathToHash[oldPath];
		if (!h) return;
		delete this.state.pathToHash[oldPath];
		this.state.pathToHash[newPath] = h;
		const id = this.state.hashToId[h];
		if (id) this.state.idToPath[id] = newPath;
	}

	handleDelete(path: string): void {
		// We only forget the mapping; the brain keeps the memory until the user
		// purges it explicitly — vault deletions shouldn't silently lose memories.
		const h = this.state.pathToHash[path];
		if (h) {
			const id = this.state.hashToId[h];
			if (id && this.state.idToPath[id] === path) delete this.state.idToPath[id];
		}
		delete this.state.pathToHash[path];
	}

	async indexFile(
		file: TFile,
		opts: { force?: boolean; notify?: boolean } = {},
	): Promise<{ indexed: boolean; reason?: string; id?: string }> {
		if (file.extension !== "md") {
			return { indexed: false, reason: "not-markdown" };
		}
		const raw = await this.plugin.app.vault.read(file);
		const body = stripFrontmatter(raw).trim();
		if (body.replace(/\s+/g, "").length < this.settings.indexMinChars) {
			return { indexed: false, reason: "too-short" };
		}
		const hash = await hashContent(body);
		if (!opts.force && this.state.pathToHash[file.path] === hash) {
			return { indexed: false, reason: "unchanged" };
		}
		if (this.settings.enableAIDefence) {
			try {
				const scan = await this.brain.scan(body);
				if (!scan.safe) {
					if (opts.notify !== false) {
						new Notice(
							`Brain: blocked by AIDefence (${scan.threat_level}) — ${file.name}`,
							6000,
						);
					}
					return { indexed: false, reason: `aidefence-${scan.threat_level}` };
				}
			} catch (e) {
				// If scan endpoint is unreachable, be conservative and skip.
				if (e instanceof BrainError && e.status === 0) {
					if (opts.notify !== false) {
						new Notice(
							`Brain unreachable, skipping index of ${file.name}`,
							4000,
						);
					}
					return { indexed: false, reason: "network" };
				}
				// Other errors → let the create call surface them.
			}
		}
		const category = extractCategory(raw, this.settings.defaultCategory);
		try {
			const result = await this.brain.createMemory(category, body);
			this.state.pathToHash[file.path] = result.content_hash;
			this.state.hashToId[result.content_hash] = result.id;
			this.state.idToPath[result.id] = file.path;
			this.state.lastSync = Math.floor(Date.now() / 1000);
			if (opts.notify) {
				new Notice(`Indexed ${file.name} → ${category}`);
			}
			return { indexed: true, id: result.id };
		} catch (e) {
			// Network-level failure → enqueue for retry when the brain
			// comes back. Everything else bubbles up.
			if (e instanceof BrainError && e.status === 0 && this.queue) {
				this.queue.enqueue({
					path: file.path,
					category,
					content: body,
					queuedAt: Date.now(),
				});
				if (opts.notify !== false) {
					new Notice(
						`Brain offline — queued ${file.name} for later (${this.queue.size()} pending)`,
					);
				}
				return { indexed: false, reason: "queued-offline" };
			}
			throw e;
		}
	}

	async bulkSync(reporter: ProgressReporter): Promise<{
		indexed: number;
		skipped: number;
		failed: number;
	}> {
		const include = splitList(this.settings.bulkSyncIncludeFolders);
		const exclude = splitList(this.settings.bulkSyncExcludeFolders);
		const files = this.plugin.app.vault
			.getMarkdownFiles()
			.filter((f) => folderMatch(f.path, include, exclude));

		const total = files.length;
		const batch = Math.max(1, this.settings.bulkSyncBatchSize);
		let indexed = 0;
		let skipped = 0;
		let failed = 0;

		for (let i = 0; i < files.length; i += batch) {
			if (reporter.cancelled()) break;
			const slice = files.slice(i, i + batch);
			const results = await Promise.allSettled(
				slice.map((f) => this.indexFile(f, { notify: false })),
			);
			for (const r of results) {
				if (r.status === "fulfilled") {
					if (r.value.indexed) indexed++;
					else skipped++;
				} else {
					failed++;
				}
			}
			reporter.update(
				Math.min(i + slice.length, total),
				total,
				slice[slice.length - 1]?.name,
			);
		}
		return { indexed, skipped, failed };
	}
}

function splitList(raw: string): string[] {
	return raw
		.split(",")
		.map((s) => s.trim().replace(/^\/+|\/+$/g, ""))
		.filter((s) => s.length > 0);
}

function folderMatch(path: string, include: string[], exclude: string[]): boolean {
	for (const ex of exclude) {
		if (path === ex || path.startsWith(ex + "/")) return false;
	}
	if (include.length === 0) return true;
	for (const inc of include) {
		if (path === inc || path.startsWith(inc + "/")) return true;
	}
	return false;
}
