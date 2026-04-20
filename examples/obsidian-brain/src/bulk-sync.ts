import { App, Modal, Notice } from "obsidian";
import type { Indexer, ProgressReporter } from "./indexer";

export class BulkSyncModal extends Modal {
	private cancelled = false;
	private startBtn!: HTMLButtonElement;
	private cancelBtn!: HTMLButtonElement;
	private progressBar!: HTMLElement;
	private progressLabel!: HTMLElement;
	private detail!: HTMLElement;
	private running = false;

	constructor(
		app: App,
		private indexer: Indexer,
	) {
		super(app);
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.createEl("h2", { text: "Bulk-sync vault → brain" });
		contentEl.createEl("p", {
			text:
				"Indexes every markdown note matching your include/exclude filters. " +
				"Unchanged notes are skipped automatically.",
		});

		const bar = contentEl.createEl("div", { cls: "brain-bulk-progress" });
		this.progressBar = bar.createEl("div", { cls: "brain-bulk-progress-fill" });
		this.progressLabel = contentEl.createEl("div", {
			cls: "brain-bulk-progress-label",
			text: "Idle",
		});
		this.detail = contentEl.createEl("div", { cls: "brain-bulk-progress-detail" });

		const buttons = contentEl.createEl("div", { cls: "brain-bulk-buttons" });
		this.startBtn = buttons.createEl("button", { text: "Start", cls: "mod-cta" });
		this.cancelBtn = buttons.createEl("button", { text: "Close" });

		this.startBtn.addEventListener("click", () => void this.start());
		this.cancelBtn.addEventListener("click", () => {
			if (this.running) {
				this.cancelled = true;
				this.cancelBtn.setText("Cancelling…");
			} else {
				this.close();
			}
		});
	}

	onClose(): void {
		this.cancelled = true;
		this.contentEl.empty();
	}

	private async start(): Promise<void> {
		if (this.running) return;
		this.running = true;
		this.cancelled = false;
		this.startBtn.disabled = true;
		this.cancelBtn.setText("Cancel");
		const reporter: ProgressReporter = {
			update: (done, total, note) => {
				const pct = total === 0 ? 0 : Math.round((done / total) * 100);
				this.progressBar.style.width = `${pct}%`;
				this.progressLabel.setText(`${done} / ${total} (${pct}%)`);
				if (note) this.detail.setText(note);
			},
			cancelled: () => this.cancelled,
		};
		const started = Date.now();
		try {
			const result = await this.indexer.bulkSync(reporter);
			const elapsed = ((Date.now() - started) / 1000).toFixed(1);
			const msg =
				`Bulk sync done in ${elapsed}s — ` +
				`${result.indexed} indexed, ${result.skipped} skipped, ${result.failed} failed`;
			this.progressLabel.setText(msg);
			new Notice(msg);
		} catch (e) {
			const msg = `Bulk sync failed: ${(e as Error).message}`;
			this.progressLabel.setText(msg);
			new Notice(msg, 8000);
		} finally {
			this.running = false;
			this.startBtn.disabled = false;
			this.cancelBtn.setText("Close");
		}
	}
}
