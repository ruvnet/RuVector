import { App, Modal, Notice, TFile } from "obsidian";
import { BrainClient } from "./brain";
import { PiClient, PiMemory } from "./pi-client";
import type { BrainSettings } from "./settings";
import type { IndexState, Indexer } from "./indexer";

/**
 * Pulls a slice of pi.ruv.io memories and mirrors them into the local
 * brain + vault. Each pulled memory becomes:
 *   - a POST /memories entry in the local brain (category `pi-<orig>`),
 *   - a stub markdown note under `Brain/Pi/<title>.md` with the
 *     `pi-id`, `pi-source`, and `brain-category` frontmatter keys,
 *   - an entry in the plugin's indexState so click-to-open works.
 *
 * Dedup: content hash on the brain side collapses duplicates. We also
 * skip rewriting the stub note if the `pi-id` frontmatter matches.
 */
export class PiSync {
	constructor(
		private app: App,
		private brain: BrainClient,
		private pi: PiClient,
		private indexer: Indexer,
		private settings: BrainSettings,
		private indexState: IndexState,
	) {}

	async pullOnce(
		onProgress?: (done: number, total: number, label?: string) => void,
	): Promise<{ pulled: number; blocked: number; total: number }> {
		if (!this.pi.configured) {
			new Notice("pi.ruv.io: set URL + bearer token in settings first");
			return { pulled: 0, blocked: 0, total: 0 };
		}
		let memories: PiMemory[];
		try {
			memories = this.settings.piPullQuery
				? await this.pi.search(this.settings.piPullQuery, this.settings.piPullLimit)
				: await this.pi.list(this.settings.piPullLimit);
		} catch (e) {
			new Notice(`pi.ruv.io pull failed: ${(e as Error).message}`, 8000);
			return { pulled: 0, blocked: 0, total: 0 };
		}

		const folder = "Brain/Pi";
		if (!this.app.vault.getAbstractFileByPath(folder)) {
			await this.app.vault.createFolder(folder).catch(() => undefined);
		}

		let pulled = 0;
		let blocked = 0;
		for (let i = 0; i < memories.length; i++) {
			const mem = memories[i];
			const title = (mem.title ?? mem.id ?? "untitled").trim() || mem.id;
			onProgress?.(i + 1, memories.length, title);
			const origCat = (mem.category ?? "pi").trim() || "pi";
			const category = `pi-${origCat}`;
			const content = (mem.content ?? "").trim();
			if (!content) continue;

			let created: Awaited<ReturnType<BrainClient["createMemory"]>> | null = null;
			try {
				created = await this.brain.createMemory(category, content);
			} catch (e) {
				const msg = (e as Error).message;
				if (/422/.test(msg) || /AIDefence/i.test(msg)) {
					blocked++;
					continue;
				}
				console.warn("[pi-sync] brain POST failed", e);
				continue;
			}

			const safe = title.replace(/[^A-Za-z0-9 _.-]+/g, "_").slice(0, 80) || mem.id;
			const path = `${folder}/${safe}.md`;
			const frontmatter =
				`---\n` +
				`brain-category: ${category}\n` +
				`tags: [brain/${category}]\n` +
				`pi-id: "${mem.id}"\n` +
				`pi-source: "${this.settings.piUrl}"\n` +
				`---\n\n# ${title}\n\n${content}\n`;
			const existing = this.app.vault.getAbstractFileByPath(path);
			if (existing instanceof TFile) {
				await this.app.vault.modify(existing, frontmatter);
			} else {
				await this.app.vault.create(path, frontmatter);
			}
			this.indexState.pathToHash[path] = created.content_hash;
			this.indexState.hashToId[created.content_hash] = created.id;
			this.indexState.idToPath[created.id] = path;
			this.indexer.state = this.indexState;
			pulled++;
		}
		this.indexState.lastSync = Math.floor(Date.now() / 1000);
		return { pulled, blocked, total: memories.length };
	}
}

export class PiSyncModal extends Modal {
	private progressBar!: HTMLElement;
	private label!: HTMLElement;
	private detail!: HTMLElement;
	private startBtn!: HTMLButtonElement;
	private running = false;

	constructor(
		app: App,
		private sync: PiSync,
	) {
		super(app);
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.createEl("h2", { text: "Pull memories from pi.ruv.io" });
		contentEl.createEl("p", {
			text:
				"Mirrors pi.ruv.io memories into the local brain and writes a stub note per memory under Brain/Pi/.",
		});
		const bar = contentEl.createEl("div", { cls: "brain-bulk-progress" });
		this.progressBar = bar.createEl("div", { cls: "brain-bulk-progress-fill" });
		this.label = contentEl.createEl("div", {
			cls: "brain-bulk-progress-label",
			text: "Idle",
		});
		this.detail = contentEl.createEl("div", { cls: "brain-bulk-progress-detail" });
		const row = contentEl.createEl("div", { cls: "brain-bulk-buttons" });
		this.startBtn = row.createEl("button", { cls: "mod-cta", text: "Pull" });
		const close = row.createEl("button", { text: "Close" });
		this.startBtn.addEventListener("click", () => void this.start());
		close.addEventListener("click", () => {
			if (!this.running) this.close();
		});
	}

	onClose(): void {
		this.contentEl.empty();
	}

	private async start(): Promise<void> {
		if (this.running) return;
		this.running = true;
		this.startBtn.disabled = true;
		try {
			const result = await this.sync.pullOnce((done, total, label) => {
				const pct = total === 0 ? 0 : Math.round((done / total) * 100);
				this.progressBar.style.width = `${pct}%`;
				this.label.setText(`${done} / ${total} (${pct}%)`);
				if (label) this.detail.setText(label);
			});
			const msg =
				`pi.ruv.io pull done: ${result.pulled} pulled, ` +
				`${result.blocked} blocked, ${result.total} total`;
			this.label.setText(msg);
			new Notice(msg, 6000);
		} catch (e) {
			new Notice(`pi.ruv.io pull failed: ${(e as Error).message}`, 8000);
		} finally {
			this.running = false;
			this.startBtn.disabled = false;
		}
	}
}

export class PiSearchModal extends Modal {
	private inputEl!: HTMLInputElement;
	private resultsEl!: HTMLElement;
	private statusEl!: HTMLElement;
	private debounceHandle: number | null = null;
	private generation = 0;

	constructor(
		app: App,
		private pi: PiClient,
		private limit: number,
	) {
		super(app);
	}

	onOpen(): void {
		const { contentEl, modalEl } = this;
		modalEl.addClass("obsidian-brain-search-modal");
		contentEl.empty();
		contentEl.createEl("div", {
			cls: "brain-search-title",
			text: "Search pi.ruv.io directly",
		});
		this.inputEl = contentEl.createEl("input", {
			type: "text",
			cls: "brain-search-input",
			attr: { placeholder: "Query pi.ruv.io…", spellcheck: "false" },
		});
		this.statusEl = contentEl.createEl("div", { cls: "brain-search-status" });
		this.resultsEl = contentEl.createEl("div", { cls: "brain-search-results" });
		this.inputEl.addEventListener("input", () => this.scheduleSearch());
		this.inputEl.addEventListener("keydown", (e) => {
			if (e.key === "Escape") this.close();
		});
		setTimeout(() => this.inputEl.focus(), 0);
	}

	onClose(): void {
		if (this.debounceHandle) window.clearTimeout(this.debounceHandle);
		this.contentEl.empty();
	}

	private scheduleSearch(): void {
		if (this.debounceHandle) window.clearTimeout(this.debounceHandle);
		this.debounceHandle = window.setTimeout(() => void this.runSearch(), 220);
	}

	private async runSearch(): Promise<void> {
		const q = this.inputEl.value.trim();
		if (!q) {
			this.resultsEl.empty();
			this.statusEl.setText("");
			return;
		}
		const gen = ++this.generation;
		this.statusEl.setText("Searching pi.ruv.io…");
		this.resultsEl.empty();
		try {
			const results = await this.pi.search(q, this.limit);
			if (gen !== this.generation) return;
			this.statusEl.setText(
				`${results.length} result${results.length === 1 ? "" : "s"}`,
			);
			results.forEach((r) => {
				const row = this.resultsEl.createEl("div", {
					cls: "brain-search-row",
				});
				row.createEl("span", {
					cls: "brain-search-category",
					text: r.category ?? "pi",
				});
				row.createEl("span", {
					cls: "brain-search-meta",
					text: r.score !== undefined ? `score ${r.score.toFixed(3)}` : "",
				});
				const snippet =
					(r.title ? `${r.title} — ` : "") +
					(r.content ?? "").replace(/\s+/g, " ").slice(0, 180);
				row.createEl("div", { cls: "brain-search-snippet", text: snippet });
			});
		} catch (e) {
			if (gen !== this.generation) return;
			this.statusEl.setText(`pi.ruv.io error: ${(e as Error).message}`);
		}
	}
}
