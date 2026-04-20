import { App, PluginSettingTab, Setting, Notice } from "obsidian";
import type ObsidianBrainPlugin from "./main";

export interface BrainSettings {
	brainUrl: string;
	embedderUrl: string;
	defaultCategory: string;
	autoIndex: boolean;
	autoIndexDebounceMs: number;
	indexMinChars: number;
	enableAIDefence: boolean;
	searchLimit: number;
	relatedLimit: number;
	bulkSyncBatchSize: number;
	bulkSyncIncludeFolders: string;
	bulkSyncExcludeFolders: string;
	storeMapping: Record<string, string>;
	dpoDirection: string;
}

export const DEFAULT_SETTINGS: BrainSettings = {
	brainUrl: "http://127.0.0.1:9876",
	embedderUrl: "http://127.0.0.1:9877",
	defaultCategory: "obsidian",
	autoIndex: false,
	autoIndexDebounceMs: 3000,
	indexMinChars: 40,
	enableAIDefence: true,
	searchLimit: 8,
	relatedLimit: 8,
	bulkSyncBatchSize: 20,
	bulkSyncIncludeFolders: "",
	bulkSyncExcludeFolders: ".obsidian,.trash",
	storeMapping: {},
	dpoDirection: "quality",
};

export class BrainSettingTab extends PluginSettingTab {
	constructor(
		app: App,
		private plugin: ObsidianBrainPlugin,
	) {
		super(app, plugin);
	}

	display(): void {
		const { containerEl } = this;
		containerEl.empty();
		containerEl.createEl("h2", { text: "RuVector Brain" });

		new Setting(containerEl)
			.setName("Brain URL")
			.setDesc("Local RuVector brain REST endpoint")
			.addText((t) =>
				t
					.setPlaceholder("http://127.0.0.1:9876")
					.setValue(this.plugin.settings.brainUrl)
					.onChange(async (v) => {
						this.plugin.settings.brainUrl = v.trim();
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Embedder URL")
			.setDesc("Local embedder REST endpoint (POST /embed)")
			.addText((t) =>
				t
					.setPlaceholder("http://127.0.0.1:9877")
					.setValue(this.plugin.settings.embedderUrl)
					.onChange(async (v) => {
						this.plugin.settings.embedderUrl = v.trim();
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Test connection")
			.setDesc("Probe /health on both endpoints")
			.addButton((b) =>
				b
					.setButtonText("Test")
					.setCta()
					.onClick(async () => {
						try {
							const h = await this.plugin.brain.health();
							new Notice(
								`Brain OK — ${h.version} (${h.backend}). Testing embedder…`,
							);
							const v = await this.plugin.brain.embed("hello");
							new Notice(`Embedder OK — dim ${v.length}.`);
						} catch (e) {
							new Notice(`Connection failed: ${(e as Error).message}`, 8000);
						}
					}),
			);

		containerEl.createEl("h3", { text: "Indexing" });

		new Setting(containerEl)
			.setName("Default category")
			.setDesc("Category used when indexing notes without an explicit one")
			.addText((t) =>
				t
					.setValue(this.plugin.settings.defaultCategory)
					.onChange(async (v) => {
						this.plugin.settings.defaultCategory = v.trim() || "obsidian";
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Auto-index on save")
			.setDesc("Embed + store notes in the brain a few seconds after each edit")
			.addToggle((t) =>
				t.setValue(this.plugin.settings.autoIndex).onChange(async (v) => {
					this.plugin.settings.autoIndex = v;
					await this.plugin.saveSettings();
					this.plugin.indexer.configureAutoIndex();
				}),
			);

		new Setting(containerEl)
			.setName("Auto-index debounce (ms)")
			.addText((t) =>
				t
					.setValue(String(this.plugin.settings.autoIndexDebounceMs))
					.onChange(async (v) => {
						const n = parseInt(v, 10);
						if (!Number.isFinite(n) || n < 100) return;
						this.plugin.settings.autoIndexDebounceMs = n;
						await this.plugin.saveSettings();
						this.plugin.indexer.configureAutoIndex();
					}),
			);

		new Setting(containerEl)
			.setName("Minimum characters to index")
			.setDesc("Skip notes with fewer than N non-whitespace characters")
			.addText((t) =>
				t
					.setValue(String(this.plugin.settings.indexMinChars))
					.onChange(async (v) => {
						const n = parseInt(v, 10);
						if (!Number.isFinite(n) || n < 1) return;
						this.plugin.settings.indexMinChars = n;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("AIDefence scan before indexing")
			.setDesc("Reject notes flagged as injection or containing PII")
			.addToggle((t) =>
				t
					.setValue(this.plugin.settings.enableAIDefence)
					.onChange(async (v) => {
						this.plugin.settings.enableAIDefence = v;
						await this.plugin.saveSettings();
					}),
			);

		containerEl.createEl("h3", { text: "Search & Related" });

		new Setting(containerEl)
			.setName("Search result limit (k)")
			.addText((t) =>
				t
					.setValue(String(this.plugin.settings.searchLimit))
					.onChange(async (v) => {
						const n = parseInt(v, 10);
						if (!Number.isFinite(n) || n < 1 || n > 100) return;
						this.plugin.settings.searchLimit = n;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Related panel limit")
			.addText((t) =>
				t
					.setValue(String(this.plugin.settings.relatedLimit))
					.onChange(async (v) => {
						const n = parseInt(v, 10);
						if (!Number.isFinite(n) || n < 1 || n > 50) return;
						this.plugin.settings.relatedLimit = n;
						await this.plugin.saveSettings();
					}),
			);

		containerEl.createEl("h3", { text: "Bulk sync" });

		new Setting(containerEl)
			.setName("Batch size")
			.setDesc("Notes indexed in parallel per batch")
			.addText((t) =>
				t
					.setValue(String(this.plugin.settings.bulkSyncBatchSize))
					.onChange(async (v) => {
						const n = parseInt(v, 10);
						if (!Number.isFinite(n) || n < 1 || n > 200) return;
						this.plugin.settings.bulkSyncBatchSize = n;
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Include folders")
			.setDesc("Comma-separated, empty = entire vault")
			.addText((t) =>
				t
					.setValue(this.plugin.settings.bulkSyncIncludeFolders)
					.onChange(async (v) => {
						this.plugin.settings.bulkSyncIncludeFolders = v.trim();
						await this.plugin.saveSettings();
					}),
			);

		new Setting(containerEl)
			.setName("Exclude folders")
			.setDesc("Comma-separated")
			.addText((t) =>
				t
					.setValue(this.plugin.settings.bulkSyncExcludeFolders)
					.onChange(async (v) => {
						this.plugin.settings.bulkSyncExcludeFolders = v.trim();
						await this.plugin.saveSettings();
					}),
			);

		containerEl.createEl("h3", { text: "DPO / preference pairs" });

		new Setting(containerEl)
			.setName("Default direction label")
			.setDesc("Used when marking chosen/rejected pairs")
			.addText((t) =>
				t
					.setValue(this.plugin.settings.dpoDirection)
					.onChange(async (v) => {
						this.plugin.settings.dpoDirection = v.trim() || "quality";
						await this.plugin.saveSettings();
					}),
			);
	}
}
