import { ItemView, TFile, WorkspaceLeaf } from "obsidian";
import { BrainClient, BrainError, Memory } from "./brain";
import type { BrainSettings } from "./settings";
import type { IndexState } from "./indexer";

export const RELATED_VIEW_TYPE = "obsidian-brain-related";

interface ViewState {
	loading: boolean;
	results: Memory[];
	error: string | null;
	sourcePath: string | null;
}

export class RelatedView extends ItemView {
	private state: ViewState = {
		loading: false,
		results: [],
		error: null,
		sourcePath: null,
	};
	private generation = 0;

	constructor(
		leaf: WorkspaceLeaf,
		private brain: BrainClient,
		private settings: BrainSettings,
		private indexState: IndexState,
	) {
		super(leaf);
	}

	getViewType(): string {
		return RELATED_VIEW_TYPE;
	}

	getDisplayText(): string {
		return "Brain: related";
	}

	getIcon(): string {
		return "brain-circuit";
	}

	async onOpen(): Promise<void> {
		this.render();
		this.registerEvent(
			this.app.workspace.on("active-leaf-change", () => void this.refreshForActive()),
		);
		await this.refreshForActive();
	}

	async onClose(): Promise<void> {
		this.generation++;
	}

	async refreshForActive(): Promise<void> {
		const file = this.app.workspace.getActiveFile();
		if (!file || file.extension !== "md") return;
		await this.loadForFile(file);
	}

	async loadForFile(file: TFile): Promise<void> {
		const gen = ++this.generation;
		this.state = { loading: true, results: [], error: null, sourcePath: file.path };
		this.render();
		try {
			const raw = await this.app.vault.read(file);
			const body = raw.slice(0, 2000);
			const resp = await this.brain.search(body, this.settings.relatedLimit);
			if (gen !== this.generation) return;
			// Drop self-matches (same path we just asked about).
			const filtered = resp.results.filter((r) => {
				const path = this.indexState.idToPath[r.id];
				return path !== file.path;
			});
			this.state.results = filtered;
		} catch (e) {
			if (gen !== this.generation) return;
			this.state.error =
				e instanceof BrainError
					? `Brain ${e.status || "?"}: ${e.message}`
					: (e as Error).message;
		} finally {
			if (gen === this.generation) {
				this.state.loading = false;
				this.render();
			}
		}
	}

	private render(): void {
		const root = this.contentEl;
		root.empty();
		root.addClass("obsidian-brain-related-view");

		const header = root.createEl("div", { cls: "brain-related-header" });
		header.createEl("span", { cls: "brain-related-title", text: "Related memories" });
		const refresh = header.createEl("button", { cls: "clickable-icon", text: "↻" });
		refresh.ariaLabel = "Refresh";
		refresh.addEventListener("click", () => void this.refreshForActive());

		if (this.state.sourcePath) {
			root.createEl("div", {
				cls: "brain-related-source",
				text: this.state.sourcePath,
			});
		}
		if (this.state.loading) {
			root.createEl("div", { cls: "brain-related-status", text: "Searching brain…" });
			return;
		}
		if (this.state.error) {
			root.createEl("div", {
				cls: "brain-related-status brain-related-error",
				text: this.state.error,
			});
			return;
		}
		if (this.state.results.length === 0) {
			root.createEl("div", {
				cls: "brain-related-status",
				text: "No related memories. Try bulk-syncing your vault.",
			});
			return;
		}

		const list = root.createEl("div", { cls: "brain-related-list" });
		this.state.results.forEach((mem) => {
			const row = list.createEl("div", { cls: "brain-related-row" });
			const cat = row.createEl("span", {
				cls: "brain-related-category",
				text: mem.category,
			});
			cat.style.setProperty("--brain-category", hashColor(mem.category));
			const score = row.createEl("span", {
				cls: "brain-related-score",
				text: mem.score !== undefined ? mem.score.toFixed(3) : "",
			});
			void score;
			const snippet =
				(mem.content ?? "").replace(/\s+/g, " ").trim().slice(0, 180) ||
				mem.id.slice(0, 16);
			row.createEl("div", { cls: "brain-related-snippet", text: snippet });
			row.addEventListener("click", () => void this.activate(mem));
		});
	}

	private async activate(mem: Memory): Promise<void> {
		const path = this.indexState.idToPath[mem.id];
		if (path) {
			const file = this.app.vault.getAbstractFileByPath(path);
			if (file instanceof TFile) {
				await this.app.workspace.getLeaf(false).openFile(file);
				return;
			}
		}
		// Otherwise, prompt a temporary preview as a Notice.
		const content = mem.content
			? mem.content
			: (await this.brain.getMemory(mem.id)).content ?? "(no content)";
		const sample = content.slice(0, 400);
		const el = this.contentEl.createEl("div", { cls: "brain-related-preview" });
		el.createEl("pre").setText(sample);
		setTimeout(() => el.remove(), 12000);
	}
}

function hashColor(s: string): string {
	let h = 0;
	for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
	const hue = Math.abs(h) % 360;
	return `hsl(${hue}, 55%, 55%)`;
}
