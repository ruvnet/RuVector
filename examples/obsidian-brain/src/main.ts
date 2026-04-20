import { Component, MarkdownView, Notice, Plugin, TFile, WorkspaceLeaf } from "obsidian";
import { BrainClient } from "./brain";
import {
	BrainSettings,
	BrainSettingTab,
	DEFAULT_SETTINGS,
} from "./settings";
import {
	EMPTY_INDEX_STATE,
	Indexer,
	IndexState,
} from "./indexer";
import { BrainSearchModal, quickSearchAndOpen } from "./search-modal";
import { RelatedView, RELATED_VIEW_TYPE } from "./related-view";
import { BulkSyncModal } from "./bulk-sync";
import { DpoController, DpoStatusModal } from "./dpo";
import { GraphOverlay } from "./graph-overlay";
import { PiClient } from "./pi-client";
import {
	PiSync,
	PiSyncModal,
	PiSearchModal,
	publishActiveNoteToPi,
} from "./pi-sync";
import { BrainQaModal } from "./qa-modal";
import { BrainOpsModal } from "./brain-ops";
import {
	EMPTY_QUEUE_STATE,
	OfflineQueue,
	OfflineQueueState,
} from "./offline-queue";

interface PluginData {
	settings: BrainSettings;
	indexState: IndexState;
	offlineQueue?: OfflineQueueState;
}

export default class ObsidianBrainPlugin extends Plugin {
	settings!: BrainSettings;
	brain!: BrainClient;
	indexer!: Indexer;
	pi!: PiClient;
	piSync!: PiSync;
	queue!: OfflineQueue;
	private dpo!: DpoController;
	private graph!: GraphOverlay;
	private statusBar!: HTMLElement;
	private statusTimer: number | null = null;
	private mdComponent = new Component();

	async onload(): Promise<void> {
		const data = (await this.loadData()) as PluginData | null;
		this.settings = { ...DEFAULT_SETTINGS, ...(data?.settings ?? {}) };
		const indexState: IndexState = data?.indexState ?? { ...EMPTY_INDEX_STATE };

		this.brain = new BrainClient(this.settings.brainUrl, this.settings.embedderUrl);
		this.indexer = new Indexer(this, this.brain, this.settings);
		this.indexer.setState(indexState);
		this.dpo = new DpoController(this.app, this.brain, this.indexer, this.settings, indexState);
		this.graph = new GraphOverlay(this.app, indexState, this.settings);
		this.pi = new PiClient(this.settings.piUrl, this.settings.piToken);
		this.piSync = new PiSync(
			this.app,
			this.brain,
			this.pi,
			this.indexer,
			this.settings,
			indexState,
		);

		const queueState: OfflineQueueState = data?.offlineQueue ?? {
			...EMPTY_QUEUE_STATE,
		};
		this.queue = new OfflineQueue(this, this.brain, queueState, () =>
			void this.persist(),
		);
		this.indexer.setQueue(this.queue);
		this.queue.start(30_000);
		this.mdComponent.load();

		this.registerView(
			RELATED_VIEW_TYPE,
			(leaf) => new RelatedView(leaf, this.brain, this.settings, indexState),
		);

		this.addRibbonIcon("brain-circuit", "Open Brain related panel", () =>
			void this.activateRelatedView(),
		);

		this.addCommand({
			id: "brain-search",
			name: "Semantic search",
			hotkeys: [{ modifiers: ["Mod", "Shift"], key: "b" }],
			callback: () => {
				new BrainSearchModal(this.app, this.brain, this.settings, indexState).open();
			},
		});

		this.addCommand({
			id: "brain-search-selection",
			name: "Semantic search on current selection",
			editorCallback: (editor) => {
				const sel = editor.getSelection().trim();
				if (!sel) {
					new Notice("Select some text first.");
					return;
				}
				new BrainSearchModal(
					this.app,
					this.brain,
					this.settings,
					indexState,
					sel.slice(0, 400),
				).open();
			},
		});

		this.addCommand({
			id: "brain-qa",
			name: "Ask the brain (Q&A modal)",
			hotkeys: [{ modifiers: ["Mod", "Shift"], key: "k" }],
			callback: () => {
				new BrainQaModal(
					this.app,
					this.brain,
					this.pi,
					this.settings,
					indexState,
					this.mdComponent,
				).open();
			},
		});

		this.addCommand({
			id: "brain-ops",
			name: "Brain ops (workload, training-stats, checkpoint, export)",
			callback: () => new BrainOpsModal(this.app, this.brain, this.settings).open(),
		});

		this.addCommand({
			id: "brain-related-panel",
			name: "Toggle related panel",
			callback: () => void this.activateRelatedView(),
		});

		this.addCommand({
			id: "brain-related-refresh",
			name: "Find related memories for current note",
			callback: () =>
				void quickSearchAndOpen(this.app, this.brain, this.settings, indexState),
		});

		this.addCommand({
			id: "brain-index-current",
			name: "Index current note",
			callback: async () => {
				const f = this.app.workspace.getActiveFile();
				if (!f) return new Notice("Open a markdown note first");
				const r = await this.indexer.indexFile(f, { force: true, notify: true });
				if (!r.indexed) new Notice(`Not indexed: ${r.reason}`);
				await this.persist();
			},
		});

		this.addCommand({
			id: "brain-bulk-sync",
			name: "Bulk-sync vault → brain",
			callback: () => new BulkSyncModal(this.app, this.indexer).open(),
		});

		this.addCommand({
			id: "brain-dpo-mark-chosen",
			name: "DPO: mark current note as chosen",
			callback: () => void this.dpo.markChosen(),
		});

		this.addCommand({
			id: "brain-dpo-pair-with-rejected",
			name: "DPO: create pair with current note (rejected)",
			callback: () => void this.dpo.createPairWithRejected(),
		});

		this.addCommand({
			id: "brain-dpo-status",
			name: "DPO: status / clear / export",
			callback: () => new DpoStatusModal(this.app, this.dpo).open(),
		});

		this.addCommand({
			id: "brain-graph-overlay-apply",
			name: "Graph overlay: apply category colors",
			callback: () => void this.graph.apply().then(() => this.persist()),
		});

		this.addCommand({
			id: "brain-graph-overlay-clear",
			name: "Graph overlay: clear category colors",
			callback: () => void this.graph.clear(),
		});

		this.addCommand({
			id: "brain-info",
			name: "Brain info / health",
			callback: () => void this.showInfo(),
		});

		this.addCommand({
			id: "brain-pi-pull",
			name: "pi.ruv.io: pull memories into local brain",
			callback: () => new PiSyncModal(this.app, this.piSync).open(),
		});

		this.addCommand({
			id: "brain-pi-search",
			name: "pi.ruv.io: search shared brain directly",
			callback: () =>
				new PiSearchModal(this.app, this.pi, this.settings.piPullLimit).open(),
		});

		this.addCommand({
			id: "brain-pi-publish",
			name: "pi.ruv.io: publish current note",
			callback: () =>
				void publishActiveNoteToPi(this.app, this.pi, this.settings),
		});

		this.addCommand({
			id: "brain-daily-recall",
			name: "Daily recall — memories from this day",
			callback: () => void this.dailyRecall(),
		});

		this.addCommand({
			id: "brain-offline-queue-flush",
			name: "Offline queue: retry pending now",
			callback: async () => {
				const before = this.queue.size();
				const r = await this.queue.drain();
				new Notice(
					`Queue: ${before} pending → ${this.queue.size()} remaining (sent ${r.sent}, dropped ${r.failed})`,
					6000,
				);
			},
		});

		this.addCommand({
			id: "brain-pi-status",
			name: "pi.ruv.io: status",
			callback: async () => {
				try {
					const s = await this.pi.status();
					new Notice(
						`pi.ruv.io — ${s.total_memories.toLocaleString()} memories, ` +
							`${s.graph_edges.toLocaleString()} edges, ` +
							`dim ${s.embedding_dim}, ${s.drift_status}`,
						8000,
					);
				} catch (e) {
					new Notice(`pi.ruv.io unreachable: ${(e as Error).message}`, 6000);
				}
			},
		});

		this.registerEvent(
			this.app.vault.on("modify", (f) => {
				if (f instanceof TFile) this.indexer.queueIndex(f);
			}),
		);
		this.registerEvent(
			this.app.vault.on("rename", (f, oldPath) => {
				this.indexer.handleRename(oldPath, f.path);
				void this.persist();
			}),
		);
		this.registerEvent(
			this.app.vault.on("delete", (f) => {
				this.indexer.handleDelete(f.path);
				void this.persist();
			}),
		);

		this.addSettingTab(new BrainSettingTab(this.app, this));

		this.statusBar = this.addStatusBarItem();
		this.statusBar.setText("Brain: …");
		this.statusBar.addClass("brain-status-bar");
		this.statusBar.addEventListener("click", () => void this.showInfo());
		this.scheduleStatusRefresh(0);

		// Persist on unload so that path→hash state survives restarts.
		this.register(() => void this.persist());
	}

	async onunload(): Promise<void> {
		if (this.statusTimer) window.clearTimeout(this.statusTimer);
		this.queue?.stop();
		this.mdComponent.unload();
		this.app.workspace.getLeavesOfType(RELATED_VIEW_TYPE).forEach((l) => l.detach());
		await this.persist();
	}

	async saveSettings(): Promise<void> {
		this.brain.brainUrl = this.settings.brainUrl;
		this.brain.embedderUrl = this.settings.embedderUrl;
		this.pi.url = this.settings.piUrl;
		this.pi.token = this.settings.piToken;
		await this.persist();
	}

	private async persist(): Promise<void> {
		const payload: PluginData = {
			settings: this.settings,
			indexState: this.indexer.state,
			offlineQueue: this.queue?.state,
		};
		await this.saveData(payload);
	}

	/**
	 * Daily recall — list memories whose created_at matches today's
	 * month/day across any year (~"on this day"). We use /memories with
	 * a high limit and filter client-side because the brain's list
	 * endpoint has no native date filter.
	 */
	private async dailyRecall(): Promise<void> {
		try {
			const today = new Date();
			const thisMonth = today.getMonth();
			const thisDay = today.getDate();
			const resp = await this.brain.listMemories(0, 500);
			const matches = resp.memories.filter((m) => {
				const d = new Date(m.created_at * 1000);
				return (
					d.getMonth() === thisMonth &&
					d.getDate() === thisDay &&
					d.getFullYear() !== today.getFullYear()
				);
			});
			if (matches.length === 0) {
				new Notice("No memories from prior years on this day.");
				return;
			}
			const folder = "Brain/Recall";
			if (!this.app.vault.getAbstractFileByPath(folder)) {
				await this.app.vault.createFolder(folder).catch(() => undefined);
			}
			const stamp = `${today.getFullYear()}-${String(thisMonth + 1).padStart(2, "0")}-${String(thisDay).padStart(2, "0")}`;
			const path = `${folder}/Recall-${stamp}.md`;
			const lines = [
				"---",
				`brain-category: recall`,
				"tags: [brain/recall]",
				"---",
				"",
				`# Brain recall for ${stamp}`,
				"",
				`${matches.length} memor${matches.length === 1 ? "y" : "ies"} from earlier years.`,
				"",
			];
			for (const m of matches) {
				const when = new Date(m.created_at * 1000).toISOString().slice(0, 10);
				lines.push(`## ${when} — ${m.category} · ${m.id.slice(0, 12)}`);
				lines.push("");
				lines.push((m.content ?? "").slice(0, 600));
				lines.push("");
			}
			const existing = this.app.vault.getAbstractFileByPath(path);
			if (existing instanceof TFile) {
				await this.app.vault.modify(existing, lines.join("\n"));
			} else {
				await this.app.vault.create(path, lines.join("\n"));
				const file = this.app.vault.getAbstractFileByPath(path);
				if (file instanceof TFile) {
					await this.app.workspace.getLeaf(false).openFile(file);
				}
			}
			new Notice(`Daily recall: ${matches.length} memories → ${path}`, 6000);
		} catch (e) {
			new Notice(`Daily recall failed: ${(e as Error).message}`, 6000);
		}
	}

	private async activateRelatedView(): Promise<void> {
		const { workspace } = this.app;
		let leaf: WorkspaceLeaf | null = workspace.getLeavesOfType(RELATED_VIEW_TYPE)[0] ?? null;
		if (!leaf) {
			leaf = workspace.getRightLeaf(false);
			if (leaf) await leaf.setViewState({ type: RELATED_VIEW_TYPE, active: true });
		}
		if (leaf) workspace.revealLeaf(leaf);
	}

	private scheduleStatusRefresh(delay: number): void {
		if (this.statusTimer) window.clearTimeout(this.statusTimer);
		this.statusTimer = window.setTimeout(() => void this.refreshStatus(), delay);
	}

	private async refreshStatus(): Promise<void> {
		try {
			const [h, info] = await Promise.all([
				this.brain.health(),
				this.brain.info().catch(() => null),
			]);
			const count = info?.memories_count ?? 0;
			const pending = this.queue?.size() ?? 0;
			const suffix = pending > 0 ? ` · ${pending} queued` : "";
			this.statusBar.setText(
				`Brain: ${h.backend} · ${count.toLocaleString()} memories${suffix}`,
			);
			this.statusBar.removeClass("brain-status-offline");
			// Opportunistic drain when the status call succeeded.
			if (pending > 0) void this.queue.drain();
		} catch (e) {
			const pending = this.queue?.size() ?? 0;
			const suffix = pending > 0 ? ` · ${pending} queued` : "";
			this.statusBar.setText(`Brain: offline${suffix}`);
			this.statusBar.addClass("brain-status-offline");
			void e;
		}
		this.scheduleStatusRefresh(30_000);
	}

	private async showInfo(): Promise<void> {
		try {
			const [info, stats] = await Promise.all([
				this.brain.info(),
				this.brain.indexStats(),
			]);
			new Notice(
				`Brain v${info.version} — ${info.memories_count} memories, ` +
					`index ${(stats as Record<string, unknown>).engine ?? "?"} ` +
					`(${(stats as Record<string, unknown>).mode ?? "?"})`,
				8000,
			);
		} catch (e) {
			new Notice(`Brain unreachable: ${(e as Error).message}`, 6000);
		}
	}
}
