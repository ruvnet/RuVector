<script lang="ts">
	import { usePublicConfig } from "$lib/utils/PublicConfig.svelte";
	import Modal from "$lib/components/Modal.svelte";
	import ServerCard from "./ServerCard.svelte";
	import AddServerForm from "./AddServerForm.svelte";
	import GalleryPanel from "$lib/components/wasm/GalleryPanel.svelte";
	import {
		allMcpServers,
		selectedServerIds,
		enabledServersCount,
		addCustomServer,
		refreshMcpServers,
		healthCheckServer,
	} from "$lib/stores/mcpServers";
	import { RVAGENT_PRESETS, buildPresetUrl } from "$lib/constants/rvagentPresets";
	import type { KeyValuePair } from "$lib/types/Tool";
	import IconAddLarge from "~icons/carbon/add-large";
	import IconRefresh from "~icons/carbon/renew";
	import LucideHammer from "~icons/lucide/hammer";
	import IconMCP from "$lib/components/icons/IconMCP.svelte";
	import IconRocket from "~icons/carbon/rocket";
	import IconGrid from "~icons/carbon/grid";

	const publicConfig = usePublicConfig();

	interface Props {
		onclose: () => void;
	}

	let { onclose }: Props = $props();

	type View = "list" | "add" | "gallery";
	let currentView = $state<View>("list");
	let isRefreshing = $state(false);

	const baseServers = $derived($allMcpServers.filter((s) => s.type === "base"));
	const customServers = $derived($allMcpServers.filter((s) => s.type === "custom"));
	const wasmServers = $derived($allMcpServers.filter((s) => s.type === "wasm"));
	const enabledCount = $derived($enabledServersCount);

	function handleAddServer(serverData: { name: string; url: string; headers?: KeyValuePair[] }) {
		addCustomServer(serverData);
		currentView = "list";
	}

	function handleCancel() {
		currentView = "list";
	}

	async function handleRefresh() {
		if (isRefreshing) return;
		isRefreshing = true;
		try {
			await refreshMcpServers();
			// After refreshing the list, re-run health checks for all known servers
			const servers = $allMcpServers;
			await Promise.allSettled(servers.map((s) => healthCheckServer(s)));
		} finally {
			isRefreshing = false;
		}
	}
</script>

<Modal width={currentView === "list" ? "w-[800px]" : currentView === "gallery" ? "w-[700px]" : "w-[600px]"} {onclose} closeButton>
	<div class="p-6">
		<!-- Header -->
		<div class="mb-6">
			<h2 class="mb-1 text-xl font-semibold text-gray-900 dark:text-gray-200">
				{#if currentView === "list"}
					MCP Servers
				{:else if currentView === "gallery"}
					RVF Agent Gallery
				{:else}
					Add MCP server
				{/if}
			</h2>
			<p class="text-sm text-gray-600 dark:text-gray-400">
				{#if currentView === "list"}
					Manage MCP servers to extend {publicConfig.PUBLIC_APP_NAME} with external tools.
				{:else if currentView === "gallery"}
					Browse and load pre-built agent templates for the WASM server.
				{:else}
					Add a custom MCP server to {publicConfig.PUBLIC_APP_NAME}.
				{/if}
			</p>
		</div>

		<!-- Content -->
		{#if currentView === "list"}
			<div
				class="mb-6 flex justify-between rounded-lg p-4 max-sm:flex-col max-sm:gap-4 sm:items-center {!enabledCount
					? 'bg-gray-100 dark:bg-white/5'
					: 'bg-gold-50 dark:bg-gold-900/10'}"
			>
				<div class="flex items-center gap-3">
					<div
						class="flex size-10 items-center justify-center rounded-xl bg-gold-500/10"
						class:grayscale={!enabledCount}
					>
						<IconMCP classNames="size-8 text-gold-500 dark:text-gold-400" />
					</div>
					<div>
						<p class="text-sm font-semibold text-gray-900 dark:text-gray-100">
							{$allMcpServers.length}
							{$allMcpServers.length === 1 ? "server" : "servers"} configured
						</p>
						<p class="text-xs text-gray-600 dark:text-gray-400">
							{enabledCount} enabled
						</p>
					</div>
				</div>

				<div class="flex gap-2">
					<button
						onclick={handleRefresh}
						disabled={isRefreshing}
						class="btn gap-1.5 rounded-lg border border-gray-200 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
					>
						<IconRefresh class="size-4 {isRefreshing ? 'animate-spin' : ''}" />
						{isRefreshing ? "Refreshing…" : "Refresh"}
					</button>
					<button
						onclick={() => (currentView = "add")}
						class="btn flex items-center gap-0.5 rounded-lg bg-gold-500 py-1.5 pl-2 pr-3 text-sm font-medium text-gray-900 hover:bg-gold-400"
					>
						<IconAddLarge class="size-4" />
						Add Server
					</button>
				</div>
			</div>
			<div class="space-y-5">
				<!-- WASM Local Servers -->
				{#if wasmServers.length > 0}
					<div>
						<div class="mb-3 flex items-center justify-between">
							<h3 class="text-sm font-medium text-gray-700 dark:text-gray-300">
								<span class="inline-flex items-center gap-1.5">
									<span class="inline-block size-2 rounded-full bg-purple-500"></span>
									Local WASM Servers ({wasmServers.length})
								</span>
							</h3>
							<button
								onclick={() => (currentView = "gallery")}
								class="flex items-center gap-1.5 rounded-lg border border-purple-200 bg-purple-50 px-2.5 py-1 text-xs font-medium text-purple-700 hover:bg-purple-100 dark:border-purple-800/50 dark:bg-purple-900/20 dark:text-purple-300 dark:hover:bg-purple-900/40"
							>
								<IconGrid class="size-3" />
								Agent Gallery
							</button>
						</div>
						<div class="grid grid-cols-1 gap-3 md:grid-cols-2">
							{#each wasmServers as server (server.id)}
								<ServerCard {server} isSelected={$selectedServerIds.has(server.id)} />
							{/each}
						</div>
					</div>
				{/if}

				<!-- Base Servers -->
				{#if baseServers.length > 0}
					<div>
						<h3 class="mb-3 text-sm font-medium text-gray-700 dark:text-gray-300">
							Base Servers ({baseServers.length})
						</h3>
						<div class="grid grid-cols-1 gap-3 md:grid-cols-2">
							{#each baseServers as server (server.id)}
								<ServerCard {server} isSelected={$selectedServerIds.has(server.id)} />
							{/each}
						</div>
					</div>
				{/if}

				<!-- Custom Servers -->
				<div>
					<h3 class="mb-3 text-sm font-medium text-gray-700 dark:text-gray-300">
						Custom Servers ({customServers.length})
					</h3>
					{#if customServers.length === 0}
						<div
							class="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-gray-300 p-8 dark:border-gray-700"
						>
							<LucideHammer class="mb-3 size-12 text-gray-400" />
							<p class="mb-1 text-sm font-medium text-gray-900 dark:text-gray-100">
								No custom servers yet
							</p>
							<p class="mb-4 text-xs text-gray-600 dark:text-gray-400">
								Add your own MCP servers with custom tools
							</p>
							<div class="flex flex-col gap-3">
								<button
									onclick={() => (currentView = "add")}
									class="flex items-center gap-1.5 rounded-lg bg-gold-500 px-4 py-2 text-sm font-medium text-gray-900 hover:bg-gold-400"
								>
									<IconAddLarge class="size-4" />
									Add Server
								</button>

								<!-- rvAgent Quick Add -->
								<div class="text-center">
									<p class="mb-2 text-xs text-gray-500 dark:text-gray-400">or quick add rvAgent:</p>
									<div class="flex flex-wrap justify-center gap-1">
										{#each RVAGENT_PRESETS.slice(0, 4) as preset}
											<button
												onclick={() => {
													addCustomServer({
														name: `rvAgent - ${preset.name}`,
														url: buildPresetUrl(preset),
													});
												}}
												class="flex items-center gap-1 rounded border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700 hover:border-gold-300 hover:bg-gold-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:border-gold-600 dark:hover:bg-gold-900/20"
												title={preset.description}
											>
												<span>{preset.icon}</span>
												<span>{preset.name}</span>
											</button>
										{/each}
									</div>
								</div>
							</div>
						</div>
					{:else}
						<div class="grid grid-cols-1 gap-3 md:grid-cols-2">
							{#each customServers as server (server.id)}
								<ServerCard {server} isSelected={$selectedServerIds.has(server.id)} />
							{/each}
						</div>
					{/if}
				</div>

				<!-- rvAgent Quick Reference -->
				<div class="rounded-lg border border-gold-200 bg-gold-50/50 p-4 dark:border-gold-800/30 dark:bg-gold-900/10">
					<h4 class="mb-2 flex items-center gap-2 text-sm font-medium text-gray-900 dark:text-gray-100">
						<IconRocket class="size-4 text-gold-600 dark:text-gold-400" />
						rvAgent MCP Server
					</h4>
					<p class="mb-2 text-xs text-gray-600 dark:text-gray-400">
						Start the rvAgent MCP server to access 46+ AI agent tools:
					</p>
					<div class="rounded bg-gray-900 p-2 dark:bg-gray-950">
						<code class="text-xs text-green-400">rvagent-mcp --transport sse --port 9000 --all</code>
					</div>
					<p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
						Use <code class="rounded bg-gray-200 px-1 dark:bg-gray-700">--groups file,shell,memory</code> to expose specific tool groups.
					</p>
				</div>

				<!-- Help Text -->
				<div class="rounded-lg bg-gray-50 p-4 dark:bg-gray-700">
					<h4 class="mb-2 text-sm font-medium text-gray-900 dark:text-gray-100">💡 Quick Tips</h4>
					<ul class="space-y-1 text-xs text-gray-600 dark:text-gray-400">
						<li>• Only connect to servers you trust</li>
						<li>• Enable servers to make their tools available in chat</li>
						<li>• Use the Health Check button to verify server connectivity</li>
						<li>• You can add HTTP headers for authentication when required</li>
					</ul>
				</div>
			</div>
		{:else if currentView === "add"}
			<AddServerForm onsubmit={handleAddServer} oncancel={handleCancel} />
		{:else if currentView === "gallery"}
			<div class="mb-4">
				<button
					onclick={() => (currentView = "list")}
					class="flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
				>
					← Back to servers
				</button>
			</div>
			<div class="h-[500px] overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
				<GalleryPanel />
			</div>
		{/if}
	</div>
</Modal>
