<script lang="ts">
	import Modal from "$lib/components/Modal.svelte";
	import RuFloUniverse from "$lib/components/RuFloUniverse.svelte";
	import { usePublicConfig } from "$lib/utils/PublicConfig.svelte";

	const publicConfig = usePublicConfig();

	interface Props {
		close: () => void;
	}

	let { close }: Props = $props();
</script>

<Modal closeOnBackdrop={false} onclose={close} width="!max-w-[420px] !m-4">
	<div
		class="flex w-full flex-col gap-8 bg-white bg-gradient-to-b to-transparent px-6 pb-7 dark:bg-[#0a0a0f] dark:from-white/5 dark:to-transparent"
	>
		<div class="relative -mx-6 h-48 select-none overflow-hidden">
			<RuFloUniverse width={420} height={192} />
			<div
				class="absolute bottom-3 right-3 rounded-lg border border-gold-500/30 bg-gold-500/20 px-2 py-0.5 text-sm font-semibold text-gold-400"
			>
				MCP Tools
			</div>
		</div>

		<div class="text-gray-700 dark:text-gray-200">
			<p class="text-[15px] leading-relaxed">
				Welcome to <strong>{publicConfig.PUBLIC_APP_NAME}</strong>, your intelligent workflow
				automation assistant.
			</p>
			<p class="mt-3 text-[15px] leading-relaxed">
				Powered by AI models with MCP tool integration for search, analysis, and workflow
				execution.
			</p>
		</div>

		<button
			class="w-full rounded-xl bg-gold-500 px-5 py-2.5 text-base font-medium text-gray-900 transition-all hover:bg-gold-400 hover:shadow-[0_0_50px_rgba(232,166,52,0.2)] hover:-translate-y-0.5 dark:bg-gold-500 dark:hover:bg-gold-400"
			onclick={close}
		>
			Start chatting
		</button>
	</div>
</Modal>
