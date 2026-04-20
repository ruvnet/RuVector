import { App, Modal, Notice, TFile, SuggestModal } from "obsidian";
import { BrainClient } from "./brain";
import type { BrainSettings } from "./settings";
import type { IndexState, Indexer } from "./indexer";

/**
 * DPO / preference pair workflow:
 *   1. User opens a note and runs "Mark current note as preferred (chosen)".
 *      We make sure the note is indexed, store the id in `heldChosen`.
 *   2. User opens a second note and runs "Create DPO pair with current note (rejected)".
 *      We index it too, POST the pair, and clear `heldChosen`.
 */
export class DpoController {
	heldChosen: { id: string; path: string } | null = null;

	constructor(
		private app: App,
		private brain: BrainClient,
		private indexer: Indexer,
		private settings: BrainSettings,
		private indexState: IndexState,
	) {}

	status(): string {
		return this.heldChosen ? `holding ${this.heldChosen.path}` : "empty";
	}

	async markChosen(): Promise<void> {
		const file = this.app.workspace.getActiveFile();
		if (!file || file.extension !== "md") {
			new Notice("Open a markdown note first");
			return;
		}
		const id = await this.ensureIndexed(file);
		if (!id) return;
		this.heldChosen = { id, path: file.path };
		new Notice(`Brain DPO: held '${file.name}' as chosen. Now open another and create the pair.`);
	}

	async clearChosen(): Promise<void> {
		this.heldChosen = null;
		new Notice("Brain DPO: cleared held chosen");
	}

	async createPairWithRejected(): Promise<void> {
		if (!this.heldChosen) {
			new Notice("Brain DPO: mark a chosen note first");
			return;
		}
		const file = this.app.workspace.getActiveFile();
		if (!file || file.extension !== "md") {
			new Notice("Open a markdown note first");
			return;
		}
		if (file.path === this.heldChosen.path) {
			new Notice("Brain DPO: chosen and rejected must differ");
			return;
		}
		const rejId = await this.ensureIndexed(file);
		if (!rejId) return;
		new DirectionPromptModal(this.app, this.settings.dpoDirection, async (direction) => {
			if (!direction) return;
			try {
				const pair = await this.brain.createPreferencePair(
					this.heldChosen!.id,
					rejId,
					direction,
				);
				new Notice(`Brain DPO: pair created (${pair.id.slice(0, 12)}, ${direction})`);
				this.heldChosen = null;
			} catch (e) {
				new Notice(`DPO pair failed: ${(e as Error).message}`, 8000);
			}
		}).open();
	}

	async exportPairs(): Promise<void> {
		try {
			const { pairs, count } = await this.brain.listPreferencePairs(500);
			const lines = [
				"# RuVector Brain — preference pairs export",
				"",
				`*count*: ${count}`,
				"",
				"| id | chosen | rejected | direction | created_at |",
				"| --- | --- | --- | --- | --- |",
				...pairs.map((p) => {
					const obj = p as Record<string, string | number>;
					return `| ${short(obj.id)} | ${short(obj.chosen_id)} | ${short(obj.rejected_id)} | ${obj.direction ?? ""} | ${obj.created_at ?? ""} |`;
				}),
			];
			const target = "Brain/Exports/preference-pairs.md";
			const folder = "Brain/Exports";
			if (!this.app.vault.getAbstractFileByPath(folder)) {
				await this.app.vault.createFolder(folder).catch(() => undefined);
			}
			const existing = this.app.vault.getAbstractFileByPath(target);
			if (existing instanceof TFile) {
				await this.app.vault.modify(existing, lines.join("\n"));
			} else {
				await this.app.vault.create(target, lines.join("\n"));
			}
			new Notice(`Brain DPO: exported ${count} pairs → ${target}`);
			void this.indexState;
		} catch (e) {
			new Notice(`DPO export failed: ${(e as Error).message}`, 8000);
		}
	}

	private async ensureIndexed(file: TFile): Promise<string | null> {
		const res = await this.indexer.indexFile(file, { force: false, notify: false });
		if (res.id) return res.id;
		// Already indexed: look up in state.
		const h = this.indexer.state.pathToHash[file.path];
		if (!h) {
			new Notice(`Brain DPO: '${file.name}' could not be indexed — ${res.reason}`);
			return null;
		}
		const existing = this.indexer.state.hashToId[h];
		return existing ?? null;
	}
}

function short(s: unknown): string {
	if (typeof s !== "string") return String(s ?? "");
	return s.slice(0, 12);
}

class DirectionPromptModal extends SuggestModal<string> {
	constructor(
		app: App,
		private defaultDir: string,
		private done: (dir: string | null) => void,
	) {
		super(app);
		this.setPlaceholder(`Direction (default: ${defaultDir})`);
	}

	getSuggestions(query: string): string[] {
		const base = new Set(["quality", "relevance", "recency", "tone", "clarity"]);
		if (this.defaultDir) base.add(this.defaultDir);
		if (query.trim()) base.add(query.trim());
		return Array.from(base);
	}

	renderSuggestion(value: string, el: HTMLElement): void {
		el.setText(value);
	}

	onChooseSuggestion(item: string): void {
		this.done(item);
	}

	onClose(): void {
		// If closed without choosing, surface a cancel to the caller.
		setTimeout(() => this.done(null), 0);
	}
}

export class DpoStatusModal extends Modal {
	constructor(
		app: App,
		private controller: DpoController,
	) {
		super(app);
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.createEl("h2", { text: "DPO / preference pair state" });
		contentEl.createEl("p", { text: `Held chosen: ${this.controller.status()}` });
		const row = contentEl.createEl("div", { cls: "brain-dpo-buttons" });
		const clear = row.createEl("button", { text: "Clear held" });
		clear.onclick = () => {
			void this.controller.clearChosen();
			this.close();
		};
		const exp = row.createEl("button", { text: "Export pairs → vault", cls: "mod-cta" });
		exp.onclick = () => {
			void this.controller.exportPairs();
			this.close();
		};
	}

	onClose(): void {
		this.contentEl.empty();
	}
}
