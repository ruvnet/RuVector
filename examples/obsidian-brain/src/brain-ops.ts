import { App, Modal, Notice, requestUrl } from "obsidian";
import { BrainClient } from "./brain";
import type { BrainSettings } from "./settings";

/**
 * Surfaces the "second-tier" brain endpoints that were previously
 * unused by the plugin: /brain/workload, /brain/training-stats,
 * /learning/stats, /brain/checkpoint, /brain/export-pairs,
 * /brain/store_mode. One modal, read-only + two buttons.
 */
export class BrainOpsModal extends Modal {
	constructor(
		app: App,
		private brain: BrainClient,
		private settings: BrainSettings,
	) {
		super(app);
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.createEl("h2", { text: "Brain Ops" });
		contentEl.createEl("p", {
			cls: "brain-ops-hint",
			text: `Endpoint: ${this.settings.brainUrl}`,
		});
		const grid = contentEl.createEl("div", { cls: "brain-ops-grid" });
		const left = grid.createEl("div", { cls: "brain-ops-col" });
		const right = grid.createEl("div", { cls: "brain-ops-col" });

		const panels: Array<{
			title: string;
			path: string;
			el: HTMLElement;
			kind?: "left" | "right";
		}> = [
			{ title: "Store mode", path: "/brain/store_mode", el: left },
			{ title: "Index stats", path: "/brain/index_stats", el: left },
			{ title: "Learning stats", path: "/learning/stats", el: left },
			{ title: "Workload", path: "/brain/workload", el: right },
			{ title: "Training stats", path: "/brain/training-stats", el: right },
			{ title: "Info", path: "/brain/info", el: right },
		];
		for (const p of panels) {
			const card = p.el.createEl("div", { cls: "brain-ops-card" });
			card.createEl("h3", { text: p.title });
			const pre = card.createEl("pre", { cls: "brain-ops-pre", text: "loading…" });
			void this.loadInto(pre, p.path);
		}

		const actions = contentEl.createEl("div", { cls: "brain-ops-actions" });
		const checkpoint = actions.createEl("button", {
			cls: "mod-cta",
			text: "Checkpoint (WAL flush)",
		});
		checkpoint.addEventListener("click", () => void this.checkpoint());

		const exportPairs = actions.createEl("button", {
			text: "Export DPO pairs (JSONL)",
		});
		exportPairs.addEventListener("click", () => void this.exportPairs());
	}

	onClose(): void {
		this.contentEl.empty();
	}

	private async loadInto(target: HTMLElement, path: string): Promise<void> {
		try {
			const resp = await requestUrl({
				url: this.settings.brainUrl.replace(/\/$/, "") + path,
				method: "GET",
				throw: false,
			});
			if (resp.status >= 400) {
				target.setText(`HTTP ${resp.status}`);
				return;
			}
			target.setText(JSON.stringify(resp.json, null, 2));
		} catch (e) {
			target.setText(`error: ${(e as Error).message}`);
		}
	}

	private async checkpoint(): Promise<void> {
		try {
			const resp = await requestUrl({
				url: this.settings.brainUrl.replace(/\/$/, "") + "/brain/checkpoint",
				method: "POST",
				throw: false,
				contentType: "application/json",
				body: "{}",
			});
			if (resp.status >= 400) {
				new Notice(`Checkpoint failed: ${resp.status}`);
				return;
			}
			new Notice(`Checkpoint OK: ${JSON.stringify(resp.json)}`);
		} catch (e) {
			new Notice(`Checkpoint failed: ${(e as Error).message}`, 6000);
		}
	}

	private async exportPairs(): Promise<void> {
		try {
			const resp = await requestUrl({
				url:
					this.settings.brainUrl.replace(/\/$/, "") +
					"/brain/export-pairs?limit=500",
				method: "GET",
				throw: false,
			});
			if (resp.status >= 400) {
				new Notice(`Export failed: ${resp.status}`);
				return;
			}
			const folder = "Brain/Exports";
			if (!this.app.vault.getAbstractFileByPath(folder)) {
				await this.app.vault.createFolder(folder).catch(() => undefined);
			}
			const path = `${folder}/dpo-pairs-${Date.now()}.json`;
			await this.app.vault.create(path, JSON.stringify(resp.json, null, 2));
			new Notice(`Exported DPO pairs → ${path}`);
			void this.brain; // kept for parity with other modules
		} catch (e) {
			new Notice(`Export failed: ${(e as Error).message}`, 6000);
		}
	}
}
