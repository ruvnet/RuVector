<script lang="ts">
	import { usePublicConfig } from "$lib/utils/PublicConfig.svelte";

	const publicConfig = usePublicConfig();

	interface Props {
		classNames?: string;
	}

	let { classNames = "" }: Props = $props();
</script>

<svg
	width="32"
	height="32"
	viewBox="0 0 32 32"
	fill="none"
	xmlns="http://www.w3.org/2000/svg"
	class="logo {classNames}"
	aria-label="{publicConfig.PUBLIC_APP_NAME} logo"
	role="img"
>
	<defs>
		<linearGradient id="rv-gold-grad" x1="0" y1="0" x2="1" y2="1">
			<stop offset="0%" stop-color="#e8a634" />
			<stop offset="100%" stop-color="#f0d89a" />
		</linearGradient>
		<linearGradient id="rv-gold-dim" x1="0" y1="0" x2="1" y2="1">
			<stop offset="0%" stop-color="#e8a634" stop-opacity="0.6" />
			<stop offset="100%" stop-color="#f0d89a" stop-opacity="0.4" />
		</linearGradient>
	</defs>

	<!-- RuVector stylized "R" with vector arrow -->
	<path class="rv-main" d="M8 6 L8 26 M8 6 L18 6 Q24 6 24 12 Q24 16 20 17 L8 17 M18 17 L26 26"
		stroke="url(#rv-gold-grad)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none" />

	<!-- Accent vector lines -->
	<path class="rv-accent a1" d="M22 22 L26 26 L22 26"
		stroke="url(#rv-gold-dim)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none" />

</svg>

<style>
	.rv-main {
		filter: drop-shadow(0 0 4px rgba(232, 166, 52, 0.3));
	}
	.rv-accent {
		opacity: 0.8;
	}
	.a1 {
		animation: arrow-pulse 2s ease-in-out infinite;
	}

	@keyframes arrow-pulse {
		0%, 100% { opacity: 0.6; }
		50% { opacity: 1; }
	}

	:global(.dark) .rv-main {
		filter: drop-shadow(0 0 6px rgba(232, 166, 52, 0.4));
	}
</style>
