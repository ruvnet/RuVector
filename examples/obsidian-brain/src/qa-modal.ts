import { App, Modal, Notice, TFile, MarkdownRenderer, Component } from "obsidian";
import { BrainClient, Memory } from "./brain";
import { PiClient, PiMemory } from "./pi-client";
import type { BrainSettings } from "./settings";
import type { IndexState } from "./indexer";

/**
 * Retrieval-first Q&A — no LLM required. Retrieves top-k from both the
 * local brain and (optionally) pi.ruv.io, shows the grounded context
 * inline, and lets the user "open" or "insert" any card. This keeps the
 * plugin honest: it surfaces *what the brain actually knows* without
 * fabricating an answer.
 *
 * The answer panel is intentionally simple: a synthesized excerpt list
 * with per-card provenance. A future phase can route through a local
 * LLM by setting `qaLlmUrl` in settings (not implemented here).
 */
interface QaRow {
	source: "local" | "pi";
	id: string;
	category: string;
	content: string;
	score?: number;
	title?: string;
	path?: string;
}

export class BrainQaModal extends Modal {
	private inputEl!: HTMLInputElement;
	private answerEl!: HTMLElement;
	private statusEl!: HTMLElement;
	private debounceHandle: number | null = null;
	private generation = 0;

	constructor(
		app: App,
		private brain: BrainClient,
		private pi: PiClient,
		private settings: BrainSettings,
		private indexState: IndexState,
		private component: Component,
	) {
		super(app);
	}

	onOpen(): void {
		const { contentEl, modalEl } = this;
		modalEl.addClass("obsidian-brain-search-modal");
		modalEl.addClass("obsidian-brain-qa-modal");
		contentEl.empty();
		contentEl.createEl("div", { cls: "brain-search-title", text: "Brain Q&A — retrieval-grounded" });
		this.inputEl = contentEl.createEl("input", {
			type: "text",
			cls: "brain-search-input",
			attr: {
				placeholder: "Ask the brain (local + pi if configured)…",
				spellcheck: "false",
			},
		});
		this.statusEl = contentEl.createEl("div", { cls: "brain-search-status" });
		this.answerEl = contentEl.createEl("div", { cls: "brain-qa-answer" });

		this.inputEl.addEventListener("input", () => this.schedule());
		this.inputEl.addEventListener("keydown", (e) => {
			if (e.key === "Escape") this.close();
			if (e.key === "Enter") void this.run();
		});
		setTimeout(() => this.inputEl.focus(), 0);
	}

	onClose(): void {
		if (this.debounceHandle) window.clearTimeout(this.debounceHandle);
		this.contentEl.empty();
	}

	private schedule(): void {
		if (this.debounceHandle) window.clearTimeout(this.debounceHandle);
		this.debounceHandle = window.setTimeout(() => void this.run(), 300);
	}

	private async run(): Promise<void> {
		const q = this.inputEl.value.trim();
		if (!q) {
			this.answerEl.empty();
			this.statusEl.setText("");
			return;
		}
		const gen = ++this.generation;
		this.statusEl.setText("Retrieving context…");
		this.answerEl.empty();

		const rows: QaRow[] = [];
		const k = this.settings.searchLimit;

		const [local, pi] = await Promise.allSettled([
			this.brain.search(q, k),
			this.pi.configured
				? this.pi.search(q, Math.max(3, Math.floor(k / 2)))
				: Promise.resolve<PiMemory[]>([]),
		]);
		if (gen !== this.generation) return;

		if (local.status === "fulfilled") {
			for (const m of local.value.results) {
				rows.push({
					source: "local",
					id: m.id,
					category: m.category,
					content: (m.content ?? "").trim(),
					score: m.score,
					path: this.indexState.idToPath[m.id],
				});
			}
		}
		if (pi.status === "fulfilled") {
			for (const m of pi.value) {
				rows.push({
					source: "pi",
					id: m.id,
					category: m.category ?? "pi",
					content: (m.content ?? "").trim(),
					score: m.score,
					title: m.title,
				});
			}
		}

		// Deduplicate by normalized content prefix so local-mirror-of-pi
		// doesn't show the same passage twice.
		const seen = new Set<string>();
		const deduped: QaRow[] = [];
		for (const r of rows) {
			const key = r.content.replace(/\s+/g, " ").slice(0, 140).toLowerCase();
			if (seen.has(key)) continue;
			seen.add(key);
			deduped.push(r);
		}
		// Rank: local first by score desc, then pi by score desc.
		deduped.sort((a, b) => {
			if (a.source !== b.source) return a.source === "local" ? -1 : 1;
			return (b.score ?? 0) - (a.score ?? 0);
		});

		if (deduped.length === 0) {
			this.statusEl.setText("No grounded context found — try a different phrasing.");
			return;
		}
		this.statusEl.setText(
			`${deduped.filter((r) => r.source === "local").length} local · ${deduped.filter((r) => r.source === "pi").length} pi`,
		);
		await this.renderRows(deduped);
	}

	private async renderRows(rows: QaRow[]): Promise<void> {
		for (const r of rows) {
			const card = this.answerEl.createEl("div", { cls: "brain-qa-card" });
			const header = card.createEl("div", { cls: "brain-qa-card-header" });
			const badge = header.createEl("span", {
				cls: `brain-search-category brain-qa-source-${r.source}`,
				text: `${r.source} · ${r.category}`,
			});
			badge.style.setProperty("--brain-category", hashColor(r.category));
			if (r.score !== undefined) {
				header.createEl("span", {
					cls: "brain-search-meta",
					text: `score ${r.score.toFixed(3)}`,
				});
			}
			const body = card.createEl("div", { cls: "brain-qa-card-body" });
			await MarkdownRenderer.render(
				this.app,
				truncate(r.content, 600),
				body,
				r.path ?? "",
				this.component,
			);
			const actions = card.createEl("div", { cls: "brain-qa-card-actions" });
			if (r.source === "local" && r.path) {
				const open = actions.createEl("button", { text: "Open note" });
				open.addEventListener("click", () => void this.openNote(r.path!));
			}
			const insert = actions.createEl("button", { text: "Insert into active note" });
			insert.addEventListener("click", () => void this.insertIntoActive(r));
			const copy = actions.createEl("button", { text: "Copy as quote" });
			copy.addEventListener("click", () => {
				navigator.clipboard.writeText(formatAsQuote(r));
				new Notice("Copied to clipboard");
			});
		}
	}

	private async openNote(path: string): Promise<void> {
		const f = this.app.vault.getAbstractFileByPath(path);
		if (f instanceof TFile) {
			await this.app.workspace.getLeaf(false).openFile(f);
			this.close();
		}
	}

	private async insertIntoActive(r: QaRow): Promise<void> {
		const file = this.app.workspace.getActiveFile();
		if (!file || file.extension !== "md") {
			new Notice("Open a markdown note to insert into.");
			return;
		}
		const existing = await this.app.vault.read(file);
		const block = `\n${formatAsQuote(r)}\n`;
		await this.app.vault.modify(file, existing + block);
		new Notice(`Inserted ${r.source}:${r.category} quote`);
	}
}

function truncate(s: string, n: number): string {
	if (s.length <= n) return s;
	return s.slice(0, n) + "…";
}

function formatAsQuote(r: QaRow): string {
	const lines = truncate(r.content, 600).split("\n").map((l) => `> ${l}`);
	lines.push(
		`> — *brain:${r.source} · ${r.category}${r.score !== undefined ? ` · score ${r.score.toFixed(3)}` : ""}*`,
	);
	return lines.join("\n");
}

function hashColor(s: string): string {
	let h = 0;
	for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
	return `hsl(${Math.abs(h) % 360}, 55%, 55%)`;
}
