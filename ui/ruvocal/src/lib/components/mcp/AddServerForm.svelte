<script lang="ts">
	import type { KeyValuePair } from "$lib/types/Tool";
	import {
		validateMcpServerUrl,
		validateHeader,
		isSensitiveHeader,
	} from "$lib/utils/mcpValidation";
	import {
		RVAGENT_PRESETS,
		buildPresetUrl,
		buildPresetCliCommand,
		type RvAgentPreset,
	} from "$lib/constants/rvagentPresets";
	import IconEye from "~icons/carbon/view";
	import IconEyeOff from "~icons/carbon/view-off";
	import IconTrash from "~icons/carbon/trash-can";
	import IconAdd from "~icons/carbon/add";
	import IconWarning from "~icons/carbon/warning";
	import IconRocket from "~icons/carbon/rocket";
	import IconTerminal from "~icons/carbon/terminal";
	import IconChevronDown from "~icons/carbon/chevron-down";

	interface Props {
		onsubmit: (server: { name: string; url: string; headers?: KeyValuePair[] }) => void;
		oncancel: () => void;
		initialName?: string;
		initialUrl?: string;
		initialHeaders?: KeyValuePair[];
		submitLabel?: string;
	}

	let {
		onsubmit,
		oncancel,
		initialName = "",
		initialUrl = "",
		initialHeaders = [],
		submitLabel = "Add Server",
	}: Props = $props();

	let name = $state(initialName);
	let url = $state(initialUrl);
	let headers = $state<KeyValuePair[]>(initialHeaders.length > 0 ? [...initialHeaders] : []);
	let showHeaderValues = $state<Record<number, boolean>>({});
	let error = $state<string | null>(null);
	let showPresets = $state(true);
	let selectedPreset = $state<RvAgentPreset | null>(null);
	let customPort = $state<number | null>(null);
	let showCliCommand = $state(false);

	/**
	 * Apply a preset to the form
	 */
	function applyPreset(preset: RvAgentPreset) {
		selectedPreset = preset;
		const port = customPort ?? preset.defaultPort;
		name = `rvAgent - ${preset.name}`;
		url = buildPresetUrl(preset, "localhost", port);
		error = null;
	}

	/**
	 * Update URL when port changes
	 */
	function updatePortInUrl() {
		if (selectedPreset && customPort) {
			url = buildPresetUrl(selectedPreset, "localhost", customPort);
		}
	}

	function addHeader() {
		headers = [...headers, { key: "", value: "" }];
	}

	function removeHeader(index: number) {
		headers = headers.filter((_, i) => i !== index);
		delete showHeaderValues[index];
	}

	function toggleHeaderVisibility(index: number) {
		showHeaderValues = {
			...showHeaderValues,
			[index]: !showHeaderValues[index],
		};
	}

	function validate(): boolean {
		if (!name.trim()) {
			error = "Server name is required";
			return false;
		}

		if (!url.trim()) {
			error = "Server URL is required";
			return false;
		}

		const urlValidation = validateMcpServerUrl(url);
		if (!urlValidation) {
			error = "Invalid URL.";
			return false;
		}

		// Validate headers
		for (let i = 0; i < headers.length; i++) {
			const header = headers[i];
			if (header.key.trim() || header.value.trim()) {
				const headerError = validateHeader(header.key, header.value);
				if (headerError) {
					error = `Header ${i + 1}: ${headerError}`;
					return false;
				}
			}
		}

		error = null;
		return true;
	}

	function handleSubmit() {
		if (!validate()) return;

		// Filter out empty headers
		const filteredHeaders = headers.filter((h) => h.key.trim() && h.value.trim());

		onsubmit({
			name: name.trim(),
			url: url.trim(),
			headers: filteredHeaders.length > 0 ? filteredHeaders : undefined,
		});
	}
</script>

<div class="space-y-4">
	<!-- rvAgent Presets Section -->
	<div class="rounded-lg border border-gold-200 bg-gold-50/50 p-4 dark:border-gold-800/50 dark:bg-gold-900/10">
		<button
			type="button"
			onclick={() => (showPresets = !showPresets)}
			class="flex w-full items-center justify-between text-left"
		>
			<div class="flex items-center gap-2">
				<IconRocket class="size-5 text-gold-600 dark:text-gold-400" />
				<span class="text-sm font-semibold text-gray-900 dark:text-gray-100">
					Quick Add: rvAgent MCP Presets
				</span>
			</div>
			<IconChevronDown
				class="size-4 text-gray-500 transition-transform {showPresets ? 'rotate-180' : ''}"
			/>
		</button>

		{#if showPresets}
			<div class="mt-4 space-y-3">
				<p class="text-xs text-gray-600 dark:text-gray-400">
					Select a preset to quickly configure rvAgent MCP server with specific tool groups.
				</p>

				<!-- Preset Grid -->
				<div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
					{#each RVAGENT_PRESETS as preset}
						<button
							type="button"
							onclick={() => applyPreset(preset)}
							class="flex flex-col items-center gap-1 rounded-lg border p-2 text-center transition-all
								{selectedPreset?.id === preset.id
									? 'border-gold-500 bg-gold-100 dark:border-gold-400 dark:bg-gold-900/30'
									: 'border-gray-200 bg-white hover:border-gold-300 hover:bg-gold-50 dark:border-gray-700 dark:bg-gray-800 dark:hover:border-gold-600 dark:hover:bg-gold-900/20'}"
						>
							<span class="text-lg">{preset.icon}</span>
							<span class="text-xs font-medium text-gray-900 dark:text-gray-100">{preset.name}</span>
							<span class="text-[10px] text-gray-500 dark:text-gray-400">{preset.groups.join(", ")}</span>
						</button>
					{/each}
				</div>

				<!-- Selected Preset Details -->
				{#if selectedPreset}
					<div class="rounded-lg border border-gray-200 bg-white p-3 dark:border-gray-700 dark:bg-gray-800">
						<div class="flex items-start justify-between">
							<div>
								<p class="text-sm font-medium text-gray-900 dark:text-gray-100">
									{selectedPreset.icon} {selectedPreset.name}
								</p>
								<p class="text-xs text-gray-600 dark:text-gray-400">{selectedPreset.description}</p>
							</div>
							<span class="rounded bg-gold-100 px-2 py-0.5 text-xs font-medium text-gold-800 dark:bg-gold-900/30 dark:text-gold-300">
								Port {customPort ?? selectedPreset.defaultPort}
							</span>
						</div>

						<!-- Port Override -->
						<div class="mt-3 flex items-center gap-2">
							<label for="custom-port" class="text-xs text-gray-600 dark:text-gray-400">Custom port:</label>
							<input
								id="custom-port"
								type="number"
								placeholder={String(selectedPreset.defaultPort)}
								bind:value={customPort}
								onchange={updatePortInUrl}
								class="w-20 rounded border border-gray-300 bg-white px-2 py-1 text-xs dark:border-gray-600 dark:bg-gray-700 dark:text-white"
							/>
						</div>

						<!-- CLI Command -->
						<div class="mt-3">
							<button
								type="button"
								onclick={() => (showCliCommand = !showCliCommand)}
								class="flex items-center gap-1 text-xs text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
							>
								<IconTerminal class="size-3" />
								{showCliCommand ? "Hide" : "Show"} CLI command
							</button>
							{#if showCliCommand}
								<div class="mt-2 rounded bg-gray-900 p-2 dark:bg-gray-950">
									<code class="text-xs text-green-400">
										{buildPresetCliCommand(selectedPreset, customPort ?? undefined)}
									</code>
								</div>
								<p class="mt-1 text-[10px] text-gray-500 dark:text-gray-400">
									Run this command to start the MCP server before connecting.
								</p>
							{/if}
						</div>

						<!-- Use Cases -->
						<div class="mt-3 flex flex-wrap gap-1">
							{#each selectedPreset.useCases as useCase}
								<span class="rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-600 dark:bg-gray-700 dark:text-gray-400">
									{useCase}
								</span>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		{/if}
	</div>

	<!-- Divider -->
	<div class="relative">
		<div class="absolute inset-0 flex items-center">
			<div class="w-full border-t border-gray-200 dark:border-gray-700"></div>
		</div>
		<div class="relative flex justify-center">
			<span class="bg-white px-3 text-xs text-gray-500 dark:bg-gray-800 dark:text-gray-400">
				{selectedPreset ? "or customize below" : "or add manually"}
			</span>
		</div>
	</div>

	<!-- Server Name -->
	<div>
		<label
			for="server-name"
			class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300"
		>
			Server Name <span class="text-red-500">*</span>
		</label>
		<input
			id="server-name"
			type="text"
			bind:value={name}
			placeholder="My MCP Server"
			class="mt-1.5 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
		/>
	</div>

	<!-- Server URL -->
	<div>
		<label for="server-url" class="mb-1 block text-sm font-medium text-gray-700 dark:text-gray-300">
			Server URL <span class="text-red-500">*</span>
		</label>
		<input
			id="server-url"
			type="url"
			bind:value={url}
			placeholder="https://example.com/mcp"
			class="mt-1.5 w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
		/>
		<!-- <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
			Only HTTPS is supported (e.g., https://localhost:5101).
		</p> -->
	</div>

	<!-- HTTP Headers -->
	<details class="rounded-lg border border-gray-200 dark:border-gray-700">
		<summary class="cursor-pointer px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300">
			HTTP Headers (Optional)
		</summary>
		<div class="space-y-2 border-t border-gray-200 p-4 dark:border-gray-700">
			{#if headers.length === 0}
				<p class="text-sm text-gray-500 dark:text-gray-400">No headers configured</p>
			{:else}
				{#each headers as header, i}
					<div class="flex gap-2">
						<input
							bind:value={header.key}
							placeholder="Header name (e.g., Authorization)"
							class="flex-1 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
						/>
						<div class="relative flex-1">
							<input
								bind:value={header.value}
								type={showHeaderValues[i] ? "text" : "password"}
								placeholder="Value"
								class="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 pr-10 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-white"
							/>
							{#if isSensitiveHeader(header.key)}
								<button
									type="button"
									onclick={() => toggleHeaderVisibility(i)}
									class="absolute right-2 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
									title={showHeaderValues[i] ? "Hide value" : "Show value"}
								>
									{#if showHeaderValues[i]}
										<IconEyeOff class="size-4" />
									{:else}
										<IconEye class="size-4" />
									{/if}
								</button>
							{/if}
						</div>
						<button
							type="button"
							onclick={() => removeHeader(i)}
							class="rounded-lg bg-red-100 p-2 text-red-600 hover:bg-red-200 dark:bg-red-900/30 dark:text-red-400 dark:hover:bg-red-900/50"
							title="Remove header"
						>
							<IconTrash class="size-4" />
						</button>
					</div>
				{/each}
			{/if}

			<button
				type="button"
				onclick={addHeader}
				class="flex items-center gap-1.5 rounded-lg bg-gray-100 px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
			>
				<IconAdd class="size-4" />
				Add Header
			</button>

			<p class="text-xs text-gray-500 dark:text-gray-400">
				Common examples:<br />
				• Bearer token:
				<code class="rounded bg-gray-100 px-1 dark:bg-gray-700"
					>Authorization: Bearer YOUR_TOKEN</code
				><br />
				• API key:
				<code class="rounded bg-gray-100 px-1 dark:bg-gray-700">X-API-Key: YOUR_KEY</code>
			</p>
		</div>
	</details>

	<!-- Security warning about custom MCP servers -->
	<div
		class="rounded-lg border border-amber-200 bg-amber-50 p-3 text-amber-900 dark:border-yellow-900/40 dark:bg-yellow-900/20 dark:text-yellow-100"
	>
		<div class="flex items-start gap-3">
			<IconWarning class="mt-0.5 size-4 flex-none text-amber-600 dark:text-yellow-300" />
			<div class="text-sm leading-5">
				<p class="font-medium">Be careful with custom MCP servers.</p>
				<p class="mt-1 text-[13px] text-amber-800 dark:text-yellow-100/90">
					They receive your requests (including conversation context and any headers you add) and
					can run powerful tools on your behalf. Only add servers you trust and review their source.
					Never share confidental informations.
				</p>
			</div>
		</div>
	</div>

	<!-- Error message -->
	{#if error}
		<div
			class="rounded-lg border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-900/20"
		>
			<p class="text-sm text-red-800 dark:text-red-200">{error}</p>
		</div>
	{/if}

	<!-- Actions -->
	<div class="flex justify-end gap-2">
		<button
			type="button"
			onclick={oncancel}
			class="rounded-lg bg-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-300 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600"
		>
			Cancel
		</button>
		<button
			type="button"
			onclick={handleSubmit}
			class="rounded-lg bg-gold-500 px-4 py-2 text-sm font-medium text-gray-900 hover:bg-gold-400"
		>
			{submitLabel}
		</button>
	</div>
</div>
