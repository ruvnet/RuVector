import { App, Notice, TFile } from "obsidian";
import type { BrainSettings } from "./settings";
import type { IndexState } from "./indexer";

interface ColorGroup {
	query: string;
	color: { a: number; rgb: number };
}

/**
 * Graph overlay — writes category color groups to `.obsidian/graph.json`
 * and ensures the configured notes carry a matching `#brain/<category>` tag.
 *
 * Obsidian doesn't expose a typed API for setting color groups at runtime,
 * so we edit the on-disk config. The user can revert at any time from
 * Obsidian → Graph view → color groups.
 */
export class GraphOverlay {
	constructor(
		private app: App,
		private indexState: IndexState,
		private settings: BrainSettings,
	) {}

	async apply(): Promise<void> {
		const byCategory = await this.collectCategories();
		if (byCategory.size === 0) {
			new Notice("Brain overlay: no indexed notes in this vault yet");
			return;
		}
		await this.tagNotes(byCategory);
		const groups = Array.from(byCategory.keys())
			.sort()
			.map<ColorGroup>((cat) => ({
				query: `tag:#brain/${safeTag(cat)}`,
				color: { a: 1, rgb: hashRgbInt(cat) },
			}));

		const graphPath = ".obsidian/graph.json";
		const adapter = this.app.vault.adapter;
		let existing: Record<string, unknown> = {};
		try {
			if (await adapter.exists(graphPath)) {
				const raw = await adapter.read(graphPath);
				existing = JSON.parse(raw) as Record<string, unknown>;
			}
		} catch (e) {
			console.warn("[obsidian-brain] graph.json unreadable", e);
		}
		const prior = Array.isArray(existing.colorGroups)
			? (existing.colorGroups as ColorGroup[]).filter(
					(g) => !(typeof g.query === "string" && g.query.startsWith("tag:#brain/")),
				)
			: [];
		existing.colorGroups = [...prior, ...groups];
		await adapter.write(graphPath, JSON.stringify(existing, null, 2));
		new Notice(
			`Brain overlay: ${groups.length} category color groups written to graph.json`,
		);
		void this.settings;
	}

	async clear(): Promise<void> {
		const adapter = this.app.vault.adapter;
		const graphPath = ".obsidian/graph.json";
		if (!(await adapter.exists(graphPath))) {
			new Notice("Brain overlay: no graph.json to clear");
			return;
		}
		const raw = await adapter.read(graphPath);
		const obj = JSON.parse(raw) as Record<string, unknown>;
		if (Array.isArray(obj.colorGroups)) {
			obj.colorGroups = (obj.colorGroups as ColorGroup[]).filter(
				(g) => !(typeof g.query === "string" && g.query.startsWith("tag:#brain/")),
			);
			await adapter.write(graphPath, JSON.stringify(obj, null, 2));
			new Notice("Brain overlay cleared");
		}
	}

	private async collectCategories(): Promise<Map<string, TFile[]>> {
		const out = new Map<string, TFile[]>();
		for (const [path] of Object.entries(this.indexState.pathToHash)) {
			const file = this.app.vault.getAbstractFileByPath(path);
			if (!(file instanceof TFile)) continue;
			const cache = this.app.metadataCache.getFileCache(file);
			let cat =
				(cache?.frontmatter?.["brain-category"] as string | undefined) ??
				(cache?.frontmatter?.["category"] as string | undefined) ??
				this.settings.defaultCategory;
			cat = (cat || "").trim() || this.settings.defaultCategory;
			const list = out.get(cat) ?? [];
			list.push(file);
			out.set(cat, list);
		}
		return out;
	}

	private async tagNotes(byCategory: Map<string, TFile[]>): Promise<void> {
		for (const [cat, files] of byCategory) {
			const tag = safeTag(cat);
			for (const file of files) {
				await this.app.fileManager
					.processFrontMatter(file, (fm) => {
						const tags = toTagArray(fm.tags);
						const marker = `brain/${tag}`;
						if (!tags.includes(marker)) tags.push(marker);
						fm.tags = tags;
					})
					.catch((e: unknown) => {
						console.warn("[obsidian-brain] tag failed", file.path, e);
					});
			}
		}
	}
}

function safeTag(s: string): string {
	return s.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "general";
}

function toTagArray(v: unknown): string[] {
	if (Array.isArray(v)) return v.filter((x) => typeof x === "string") as string[];
	if (typeof v === "string") {
		return v
			.split(/[\s,]+/)
			.map((s) => s.replace(/^#/, ""))
			.filter(Boolean);
	}
	return [];
}

function hashRgbInt(s: string): number {
	let h = 0;
	for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
	const hue = Math.abs(h) % 360;
	const [r, g, b] = hslToRgb(hue / 360, 0.55, 0.55);
	return (r << 16) | (g << 8) | b;
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
	const c = (1 - Math.abs(2 * l - 1)) * s;
	const x = c * (1 - Math.abs(((h * 6) % 2) - 1));
	const m = l - c / 2;
	let r = 0, g = 0, b = 0;
	if (h < 1 / 6) [r, g, b] = [c, x, 0];
	else if (h < 2 / 6) [r, g, b] = [x, c, 0];
	else if (h < 3 / 6) [r, g, b] = [0, c, x];
	else if (h < 4 / 6) [r, g, b] = [0, x, c];
	else if (h < 5 / 6) [r, g, b] = [x, 0, c];
	else [r, g, b] = [c, 0, x];
	return [
		Math.round((r + m) * 255),
		Math.round((g + m) * 255),
		Math.round((b + m) * 255),
	];
}
