import { App, Modal, Notice, TFile, prepareFuzzySearch } from "obsidian";
import { BrainClient, Memory } from "./brain";
import type { BrainSettings } from "./settings";
import type { IndexState } from "./indexer";

interface SearchState {
	query: string;
	results: Memory[];
	loading: boolean;
	error: string | null;
	selected: number;
}

/**
 * Cmd+Shift+B — semantic search against the brain.
 * Falls back to local fuzzy ranking if the brain is unavailable, so the
 * command still feels responsive without a backend.
 */
export class BrainSearchModal extends Modal {
	private inputEl!: HTMLInputElement;
	private resultsEl!: HTMLElement;
	private statusEl!: HTMLElement;
	private state: SearchState = {
		query: "",
		results: [],
		loading: false,
		error: null,
		selected: 0,
	};
	private debounceHandle: number | null = null;
	private generation = 0;

	constructor(
		app: App,
		private brain: BrainClient,
		private settings: BrainSettings,
		private indexState: IndexState,
	) {
		super(app);
	}

	onOpen(): void {
		const { contentEl, modalEl } = this;
		modalEl.addClass("obsidian-brain-search-modal");
		contentEl.empty();
		contentEl.createEl("div", { cls: "brain-search-title", text: "Brain semantic search" });

		this.inputEl = contentEl.createEl("input", {
			type: "text",
			cls: "brain-search-input",
			attr: { placeholder: "Search the brain (DiskANN)…", spellcheck: "false" },
		});
		this.statusEl = contentEl.createEl("div", { cls: "brain-search-status" });
		this.resultsEl = contentEl.createEl("div", { cls: "brain-search-results" });

		this.inputEl.addEventListener("input", () => {
			this.state.query = this.inputEl.value;
			this.state.selected = 0;
			this.scheduleSearch();
		});
		this.inputEl.addEventListener("keydown", (e) => this.onKey(e));
		setTimeout(() => this.inputEl.focus(), 0);

		this.render();
	}

	onClose(): void {
		if (this.debounceHandle) window.clearTimeout(this.debounceHandle);
		this.contentEl.empty();
	}

	private scheduleSearch(): void {
		if (this.debounceHandle) window.clearTimeout(this.debounceHandle);
		this.debounceHandle = window.setTimeout(() => this.runSearch(), 180);
	}

	private async runSearch(): Promise<void> {
		const q = this.state.query.trim();
		if (!q) {
			this.state.results = [];
			this.state.error = null;
			this.render();
			return;
		}
		const gen = ++this.generation;
		this.state.loading = true;
		this.state.error = null;
		this.render();
		try {
			const resp = await this.brain.search(q, this.settings.searchLimit);
			if (gen !== this.generation) return;
			this.state.results = resp.results;
		} catch (e) {
			if (gen !== this.generation) return;
			this.state.error = (e as Error).message;
			this.state.results = this.localFallback(q);
		} finally {
			if (gen === this.generation) {
				this.state.loading = false;
				this.render();
			}
		}
	}

	private localFallback(q: string): Memory[] {
		const match = prepareFuzzySearch(q);
		const files = this.app.vault.getMarkdownFiles();
		const scored: Array<{ score: number; file: TFile }> = [];
		for (const f of files) {
			const r = match(f.path);
			if (r) scored.push({ score: r.score, file: f });
		}
		scored.sort((a, b) => b.score - a.score);
		return scored.slice(0, this.settings.searchLimit).map<Memory>((s) => ({
			id: `local:${s.file.path}`,
			category: "local",
			content_hash: "",
			created_at: Math.floor(s.file.stat.mtime / 1000),
			score: s.score,
			content: s.file.basename,
		}));
	}

	private onKey(e: KeyboardEvent): void {
		if (e.key === "ArrowDown") {
			e.preventDefault();
			this.state.selected = Math.min(
				this.state.results.length - 1,
				this.state.selected + 1,
			);
			this.render();
		} else if (e.key === "ArrowUp") {
			e.preventDefault();
			this.state.selected = Math.max(0, this.state.selected - 1);
			this.render();
		} else if (e.key === "Enter") {
			e.preventDefault();
			const r = this.state.results[this.state.selected];
			if (r) void this.activate(r);
		} else if (e.key === "Escape") {
			this.close();
		}
	}

	private async activate(mem: Memory): Promise<void> {
		// Prefer opening a note if we know which file this memory came from.
		const path = mem.id.startsWith("local:")
			? mem.id.slice("local:".length)
			: this.indexState.idToPath[mem.id];
		if (path) {
			const file = this.app.vault.getAbstractFileByPath(path);
			if (file instanceof TFile) {
				await this.app.workspace.getLeaf(false).openFile(file);
				this.close();
				return;
			}
		}
		// Otherwise, show a preview pane inline.
		this.showPreview(mem);
	}

	private async showPreview(mem: Memory): Promise<void> {
		try {
			if (!mem.content) {
				const full = await this.brain.getMemory(mem.id);
				mem = { ...mem, content: full.content };
			}
		} catch {
			/* ignore */
		}
		const { contentEl } = this;
		contentEl.empty();
		contentEl.createEl("div", {
			cls: "brain-search-title",
			text: `${mem.category} — ${mem.id.slice(0, 12)}`,
		});
		const pre = contentEl.createEl("pre", { cls: "brain-search-preview" });
		pre.createEl("code", { text: mem.content ?? "(no content)" });
		const close = contentEl.createEl("button", {
			cls: "mod-cta",
			text: "Back to results",
		});
		close.onclick = () => {
			this.onOpen();
			this.inputEl.value = this.state.query;
		};
	}

	private render(): void {
		const { resultsEl, statusEl, state } = this;
		statusEl.empty();
		resultsEl.empty();
		if (state.loading) statusEl.setText("Searching…");
		else if (state.error) statusEl.setText(`Brain offline — fuzzy fallback (${state.error})`);
		else if (state.query && state.results.length === 0)
			statusEl.setText("No results");
		else if (state.query)
			statusEl.setText(`${state.results.length} result${state.results.length === 1 ? "" : "s"}`);

		state.results.forEach((r, i) => {
			const row = resultsEl.createEl("div", {
				cls: `brain-search-row${i === state.selected ? " selected" : ""}`,
			});
			const cat = row.createEl("span", { cls: "brain-search-category", text: r.category });
			const meta = row.createEl("span", { cls: "brain-search-meta" });
			const score = r.score !== undefined ? r.score.toFixed(3) : "—";
			meta.setText(`score ${score}`);
			const snippet =
				(r.content ?? "").replace(/\s+/g, " ").trim().slice(0, 140) ||
				r.id.slice(0, 16);
			row.createEl("div", { cls: "brain-search-snippet", text: snippet });
			cat.style.setProperty("--brain-category", hashColor(r.category));
			row.addEventListener("click", () => void this.activate(r));
			row.addEventListener("mousemove", () => {
				this.state.selected = i;
				this.render();
			});
		});
		if (!state.query) {
			statusEl.setText("Type to search. Enter opens, Esc closes.");
		}
	}
}

function hashColor(s: string): string {
	let h = 0;
	for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
	const hue = Math.abs(h) % 360;
	return `hsl(${hue}, 55%, 55%)`;
}

export async function quickSearchAndOpen(
	app: App,
	brain: BrainClient,
	settings: BrainSettings,
	indexState: IndexState,
): Promise<void> {
	const active = app.workspace.getActiveFile();
	if (!active) {
		new Notice("Open a note first to find related memories");
		return;
	}
	const raw = await app.vault.read(active);
	try {
		const resp = await brain.search(raw.slice(0, 2000), settings.relatedLimit);
		new Notice(`Brain: ${resp.results.length} related memories — see side panel`);
	} catch (e) {
		new Notice(`Brain search failed: ${(e as Error).message}`);
	}
	// Help the user — make sure the related view is visible.
	const leaves = app.workspace.getLeavesOfType("obsidian-brain-related");
	if (leaves[0]) app.workspace.revealLeaf(leaves[0]);
	void indexState; // retained for future ranking tweaks
}
