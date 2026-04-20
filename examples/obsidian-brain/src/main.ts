import { Notice, Plugin, TFile, WorkspaceLeaf } from "obsidian";
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

interface PluginData {
	settings: BrainSettings;
	indexState: IndexState;
}

export default class ObsidianBrainPlugin extends Plugin {
	settings!: BrainSettings;
	brain!: BrainClient;
	indexer!: Indexer;
	private dpo!: DpoController;
	private graph!: GraphOverlay;
	private statusBar!: HTMLElement;
	private statusTimer: number | null = null;

	async onload(): Promise<void> {
		const data = (await this.loadData()) as PluginData | null;
		this.settings = { ...DEFAULT_SETTINGS, ...(data?.settings ?? {}) };
		const indexState: IndexState = data?.indexState ?? { ...EMPTY_INDEX_STATE };

		this.brain = new BrainClient(this.settings.brainUrl, this.settings.embedderUrl);
		this.indexer = new Indexer(this, this.brain, this.settings);
		this.indexer.setState(indexState);
		this.dpo = new DpoController(this.app, this.brain, this.indexer, this.settings, indexState);
		this.graph = new GraphOverlay(this.app, indexState, this.settings);

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
		this.app.workspace.getLeavesOfType(RELATED_VIEW_TYPE).forEach((l) => l.detach());
		await this.persist();
	}

	async saveSettings(): Promise<void> {
		this.brain.brainUrl = this.settings.brainUrl;
		this.brain.embedderUrl = this.settings.embedderUrl;
		await this.persist();
	}

	private async persist(): Promise<void> {
		const payload: PluginData = {
			settings: this.settings,
			indexState: this.indexer.state,
		};
		await this.saveData(payload);
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
			this.statusBar.setText(
				`Brain: ${h.backend} · ${count.toLocaleString()} memories`,
			);
			this.statusBar.removeClass("brain-status-offline");
		} catch (e) {
			this.statusBar.setText(`Brain: offline`);
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
