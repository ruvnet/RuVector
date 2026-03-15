<script lang="ts" module>
	export const titles: { [key: string]: string } = {
		today: "Today",
		week: "This week",
		month: "This month",
		older: "Older",
	} as const;
</script>

<script lang="ts">
	import { base } from "$app/paths";

	import Logo from "$lib/components/icons/Logo.svelte";
	import IconSun from "$lib/components/icons/IconSun.svelte";
	import IconMoon from "$lib/components/icons/IconMoon.svelte";
	import { switchTheme, subscribeToTheme } from "$lib/switchTheme";
	import { isAborted } from "$lib/stores/isAborted";
	import { onDestroy } from "svelte";

	import NavConversationItem from "./NavConversationItem.svelte";
	import type { LayoutData } from "../../routes/$types";
	import type { ConvSidebar } from "$lib/types/ConvSidebar";
	import type { Model } from "$lib/types/Model";
	import { page } from "$app/state";
	import InfiniteScroll from "./InfiniteScroll.svelte";
	import { CONV_NUM_PER_PAGE } from "$lib/constants/pagination";
	import { browser } from "$app/environment";
	import { usePublicConfig } from "$lib/utils/PublicConfig.svelte";
	import { useAPIClient, handleResponse } from "$lib/APIClient";
	import { requireAuthUser } from "$lib/utils/auth";
	import { enabledServersCount } from "$lib/stores/mcpServers";
	import { isPro } from "$lib/stores/isPro";
	import IconPro from "$lib/components/icons/IconPro.svelte";
	import MCPServerManager from "./mcp/MCPServerManager.svelte";

	const publicConfig = usePublicConfig();
	const client = useAPIClient();

	interface Props {
		conversations: ConvSidebar[];
		user: LayoutData["user"];
		p?: number;
		ondeleteConversation?: (id: string) => void;
		oneditConversationTitle?: (payload: { id: string; title: string }) => void;
	}

	let {
		conversations = $bindable(),
		user,
		p = $bindable(0),
		ondeleteConversation,
		oneditConversationTitle,
	}: Props = $props();

	let hasMore = $state(true);

	function handleNewChatClick(e: MouseEvent) {
		isAborted.set(true);

		if (requireAuthUser()) {
			e.preventDefault();
		}
	}

	function handleNavItemClick(e: MouseEvent) {
		if (requireAuthUser()) {
			e.preventDefault();
		}
	}

	const dateRanges = [
		new Date().setDate(new Date().getDate() - 1),
		new Date().setDate(new Date().getDate() - 7),
		new Date().setMonth(new Date().getMonth() - 1),
	];

	// Sort conversations by updatedAt descending (newest first) within each group
	const sortByNewest = (convs: ConvSidebar[]) =>
		[...convs].sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());

	let groupedConversations = $derived({
		today: sortByNewest(
			conversations.filter(({ updatedAt }) => updatedAt.getTime() > dateRanges[0])
		),
		week: sortByNewest(
			conversations.filter(
				({ updatedAt }) => updatedAt.getTime() > dateRanges[1] && updatedAt.getTime() < dateRanges[0]
			)
		),
		month: sortByNewest(
			conversations.filter(
				({ updatedAt }) => updatedAt.getTime() > dateRanges[2] && updatedAt.getTime() < dateRanges[1]
			)
		),
		older: sortByNewest(
			conversations.filter(({ updatedAt }) => updatedAt.getTime() < dateRanges[2])
		),
	});

	const nModels: number = page.data.models.filter((el: Model) => !el.unlisted).length;

	async function handleVisible() {
		p++;
		const newConvs = await client.conversations
			.get({
				query: {
					p,
				},
			})
			.then(handleResponse)
			.then((r) => r.conversations)
			.catch((): ConvSidebar[] => []);

		if (newConvs.length === 0) {
			hasMore = false;
		}

		// Merge and sort all conversations by updatedAt (newest first)
		const merged = [...conversations, ...newConvs];
		merged.sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
		conversations = merged;
	}

	$effect(() => {
		if (conversations.length <= CONV_NUM_PER_PAGE) {
			// reset p to 0 if there's only one page of content
			// that would be caused by a data loading invalidation
			p = 0;
		}
	});

	let isDark = $state(false);
	let unsubscribeTheme: (() => void) | undefined;
	let showMcpModal = $state(false);

	if (browser) {
		unsubscribeTheme = subscribeToTheme(({ isDark: nextIsDark }) => {
			isDark = nextIsDark;
		});
	}

	onDestroy(() => {
		unsubscribeTheme?.();
	});
</script>

<div
	class="sticky top-0 flex flex-none touch-none items-center justify-between px-1.5 py-3.5 max-sm:pt-0"
>
	<a
		class="relative flex select-none items-center rounded-xl text-lg font-semibold"
		href="{publicConfig.PUBLIC_ORIGIN}{base}/"
	>
		<Logo classNames="mr-[2px]" />
		{publicConfig.PUBLIC_APP_NAME}
		<!-- Quantum dots floating over "lo" -->
		<span class="pointer-events-none absolute right-[-2px] top-[-1px] flex gap-[3px]">
			<span class="quantum-dot qd1"></span>
			<span class="quantum-dot qd2"></span>
			<span class="quantum-dot qd3"></span>
		</span>
	</a>
	<a
		href={`${base}/`}
		onclick={handleNewChatClick}
		class="flex rounded-lg border bg-white px-2 py-0.5 text-center shadow-sm hover:shadow-none dark:border-gold-600/50 dark:bg-gold-600/20 dark:text-gold-300 dark:hover:bg-gold-600/30 sm:text-smd"
		title="Ctrl/Cmd + Shift + O"
	>
		New Chat
	</a>
</div>

<div
	class="scrollbar-custom flex touch-pan-y flex-col gap-1 overflow-y-auto rounded-r-xl border border-l-0 border-gray-100 from-gray-50 px-3 pb-3 pt-2 text-[.9rem] dark:border-transparent dark:from-gray-800/30 max-sm:bg-gradient-to-t md:bg-gradient-to-l"
>
	<div class="flex flex-col gap-0.5">
		{#each Object.entries(groupedConversations) as [group, convs]}
			{#if convs.length}
				<h4 class="mb-1.5 mt-4 pl-0.5 text-sm text-gray-400 first:mt-0 dark:text-gray-500">
					{titles[group]}
				</h4>
				{#each convs as conv}
					<NavConversationItem {conv} {oneditConversationTitle} {ondeleteConversation} />
				{/each}
			{/if}
		{/each}
	</div>
	{#if hasMore}
		<InfiniteScroll onvisible={handleVisible} />
	{/if}
</div>
<div
	class="flex touch-none flex-col gap-1 rounded-r-xl border border-l-0 border-gray-100 p-3 text-sm dark:border-transparent md:mt-3 md:bg-gradient-to-l md:from-gray-50 md:dark:from-gray-800/30"
>
	{#if user?.username || user?.email}
		<div
			class="group flex h-9 items-center gap-1.5 rounded-lg pl-2.5 pr-2 hover:bg-gray-100 first:hover:bg-transparent dark:hover:bg-gray-700 first:dark:hover:bg-transparent"
		>
			<img
				src="https://huggingface.co/api/users/{user.username}/avatar?redirect=true"
				class="size-3.5 rounded-full border bg-gray-500 dark:border-white/40"
				alt=""
			/>
			<span
				class="flex flex-none shrink items-center gap-1.5 truncate pr-2 text-gray-500 dark:text-gray-400"
				>{user?.username || user?.email}</span
			>

			{#if publicConfig.isHuggingChat && $isPro === false}
				<a
					href="https://huggingface.co/subscribe/pro?from=HuggingChat"
					target="_blank"
					rel="noopener noreferrer"
					class="ml-auto flex h-[20px] items-center gap-1 px-1.5 py-0.5 text-xs text-gray-500 dark:text-gray-400"
				>
					<IconPro />
					Get PRO
				</a>
			{:else if publicConfig.isHuggingChat && $isPro === true}
				<span
					class="ml-auto flex h-[20px] items-center gap-1 px-1.5 py-0.5 text-xs text-gray-500 dark:text-gray-400"
				>
					<IconPro />
					PRO
				</span>
			{/if}
		</div>
	{/if}
	<a
		href="{base}/models"
		class="flex h-9 flex-none items-center gap-1.5 rounded-lg pl-2.5 pr-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gold-500/10 dark:hover:text-gold-400"
		onclick={handleNavItemClick}
	>
		Models
		<span
			class="ml-auto rounded-md bg-gray-500/5 px-1.5 py-0.5 text-xs text-gray-400 dark:bg-gold-500/20 dark:text-gold-400"
			>{nModels}</span
		>
	</a>

	<button
		onclick={() => (showMcpModal = true)}
		class="flex h-9 flex-none items-center gap-1.5 rounded-lg pl-2.5 pr-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gold-500/10 dark:hover:text-gold-400"
	>
		MCP Servers
		{#if $enabledServersCount > 0}
			<span
				class="ml-auto rounded-md bg-gold-600/10 px-1.5 py-0.5 text-xs text-gold-600 dark:bg-gold-600/20 dark:text-gold-400"
			>
				{$enabledServersCount}
			</span>
		{/if}
	</button>

	<span class="flex gap-1">
		<a
			href="{base}/settings/application"
			class="flex h-9 flex-none flex-grow items-center gap-1.5 rounded-lg pl-2.5 pr-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gold-500/10 dark:hover:text-gold-400"
			onclick={handleNavItemClick}
		>
			Settings
		</a>
		<button
			onclick={() => {
				switchTheme();
			}}
			aria-label="Toggle theme"
			class="flex size-9 min-w-[1.5em] flex-none items-center justify-center rounded-lg p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gold-500/10 dark:hover:text-gold-400"
		>
			{#if browser}
				{#if isDark}
					<IconSun />
				{:else}
					<IconMoon />
				{/if}
			{/if}
		</button>
	</span>
</div>

{#if showMcpModal}
	<MCPServerManager onclose={() => (showMcpModal = false)} />
{/if}

<style>
	.quantum-dot {
		display: block;
		width: 4px;
		height: 4px;
		border-radius: 50%;
	}
	.qd1 {
		background: #e8a634;
		animation: qdot 2.2s ease-in-out infinite;
	}
	.qd2 {
		background: #f0d89a;
		animation: qdot 2.2s ease-in-out infinite 0.35s;
	}
	.qd3 {
		background: #d18a1a;
		animation: qdot 2.2s ease-in-out infinite 0.7s;
	}
	@keyframes qdot {
		0%, 100% { transform: translateY(0); opacity: 0.4; }
		50% { transform: translateY(-3px); opacity: 1; }
	}
</style>
