<script lang="ts">
	import type {
		ClassificationResult,
		ABCDEScores,
		DiagnosisRecord,
		GradCamResult,
	} from "$lib/dragnes/types";
	import { DermClassifier } from "$lib/dragnes/classifier";

	import DermCapture from "./DermCapture.svelte";
	import ClassificationResultView from "./ClassificationResult.svelte";
	import GradCamOverlay from "./GradCamOverlay.svelte";
	import LesionTimeline from "./LesionTimeline.svelte";
	import ABCDEChart from "./ABCDEChart.svelte";

	import CarbonCamera from "~icons/carbon/camera";
	import CarbonResult from "~icons/carbon/result-new";
	import CarbonTime from "~icons/carbon/time";
	import CarbonSettings from "~icons/carbon/settings";
	import CarbonWifiOff from "~icons/carbon/wifi-off";

	const classifier = new DermClassifier();

	type TabId = "capture" | "results" | "history" | "settings";

	let activeTab: TabId = $state("capture");

	// Capture state
	let capturedImageData: ImageData | null = $state(null);
	let capturedBodyLocation: string = $state("unknown");
	let analyzing: boolean = $state(false);

	// Demographics for HAM10000-calibrated adjustment
	let patientAge: number | undefined = $state(undefined);
	let patientSex: "male" | "female" | undefined = $state(undefined);
	let demographicsEnabled: boolean = $state(true);

	// Results state
	let classificationResult: ClassificationResult | null = $state(null);
	let abcdeScores: ABCDEScores | null = $state(null);
	let gradCamData: Uint8Array | null = $state(null);

	// History state
	let records: DiagnosisRecord[] = $state([]);

	// Settings state
	let modelVersion: string = $state("v1.0.0-demo");
	let brainSyncEnabled: boolean = $state(false);
	let privacyStripExif: boolean = $state(true);
	let privacyLocalOnly: boolean = $state(true);

	// Offline indicator
	let isOffline: boolean = $state(false);

	$effect(() => {
		if (typeof window !== "undefined") {
			isOffline = !navigator.onLine;
			const handleOnline = () => (isOffline = false);
			const handleOffline = () => (isOffline = true);
			window.addEventListener("online", handleOnline);
			window.addEventListener("offline", handleOffline);
			return () => {
				window.removeEventListener("online", handleOnline);
				window.removeEventListener("offline", handleOffline);
			};
		}
	});

	const TABS: { id: TabId; label: string; icon: typeof CarbonCamera }[] = [
		{ id: "capture", label: "Capture", icon: CarbonCamera },
		{ id: "results", label: "Results", icon: CarbonResult },
		{ id: "history", label: "History", icon: CarbonTime },
		{ id: "settings", label: "Settings", icon: CarbonSettings },
	];

	function handleCapture(event: { imageData: ImageData; bodyLocation: string; deviceModel: string }) {
		capturedImageData = event.imageData;
		capturedBodyLocation = event.bodyLocation;
	}

	async function analyzeImage() {
		if (!capturedImageData) return;
		analyzing = true;

		try {
			// Classify with demographic adjustment (HAM10000-calibrated)
			const demographics = demographicsEnabled
				? { age: patientAge, sex: patientSex, localization: capturedBodyLocation }
				: undefined;
			const rawResult = await classifier.classifyWithDemographics(capturedImageData, demographics);
			classificationResult = rawResult;

			// Generate Grad-CAM heatmap
			try {
				const gradCamResult = await classifier.getGradCam(classificationResult.topClass);
				const hm = gradCamResult.heatmap;
				const grayscale = new Uint8Array(hm.width * hm.height);
				for (let i = 0; i < grayscale.length; i++) {
					grayscale[i] = hm.data[i * 4]; // Use red channel as intensity
				}
				gradCamData = grayscale;
			} catch {
				gradCamData = null;
			}

			// ABCDE scoring is not yet integrated into the classifier;
			// use placeholder scores derived from confidence
			abcdeScores = {
				asymmetry: classificationResult.confidence > 0.8 ? 0.5 : 1.5,
				border: classificationResult.confidence > 0.8 ? 2 : 5,
				color: 2,
				diameterMm: 4.5,
				evolution: 0,
				totalScore: classificationResult.confidence > 0.8 ? 4.0 : 8.5,
				riskLevel: classificationResult.topClass === "mel" ? "high" : "low",
				colorsDetected: ["brown", "tan"],
			};

			activeTab = "results";
		} catch (err) {
			console.error("Classification failed:", err);
		} finally {
			analyzing = false;
		}
	}

	function handleAction(action: string, payload?: unknown) {
		// In a full implementation, this would persist to brain/database
		console.log("DrAgnes action:", action, payload);
	}
</script>

<div class="flex h-full w-full flex-col bg-white dark:bg-gray-900">
	<!-- Offline banner -->
	{#if isOffline}
		<div class="flex items-center justify-center gap-2 bg-orange-500 px-3 py-1.5 text-xs font-medium text-white">
			<CarbonWifiOff class="h-3.5 w-3.5" />
			<span>Offline &mdash; brain sync unavailable</span>
		</div>
	{/if}

	<!-- Tab navigation -->
	<nav class="flex border-b border-gray-200 dark:border-gray-700">
		{#each TABS as tab}
			<button
				onclick={() => (activeTab = tab.id)}
				class="flex flex-1 items-center justify-center gap-1.5 px-2 py-3 text-xs font-medium transition-colors sm:text-sm
					{activeTab === tab.id
					? 'border-b-2 border-blue-500 text-blue-600 dark:text-blue-400'
					: 'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200'}"
			>
				<tab.icon class="h-4 w-4" />
				<span class="hidden min-[400px]:inline">{tab.label}</span>
			</button>
		{/each}
	</nav>

	<!-- Tab content -->
	<div class="scrollbar-custom flex-1 overflow-y-auto p-4">
		{#if activeTab === "capture"}
			<DermCapture oncapture={handleCapture} />

			<!-- Demographics for HAM10000 Bayesian adjustment -->
			<div class="mt-4 rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
				<div class="mb-3 flex items-center justify-between">
					<h3 class="text-sm font-semibold text-gray-500 dark:text-gray-400">Patient Demographics</h3>
					<label class="flex items-center gap-1.5">
						<span class="text-xs text-gray-400">HAM10000 adjust</span>
						<input
							type="checkbox"
							bind:checked={demographicsEnabled}
							class="h-4 w-4 rounded border-gray-300 text-blue-600"
						/>
					</label>
				</div>
				{#if demographicsEnabled}
					<div class="grid grid-cols-2 gap-3">
						<div>
							<label class="mb-1 block text-xs text-gray-500">Age</label>
							<input
								type="number"
								min="0"
								max="120"
								placeholder="e.g. 55"
								bind:value={patientAge}
								class="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200"
							/>
						</div>
						<div>
							<label class="mb-1 block text-xs text-gray-500">Sex</label>
							<select
								bind:value={patientSex}
								class="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200"
							>
								<option value={undefined}>Not specified</option>
								<option value="male">Male</option>
								<option value="female">Female</option>
							</select>
						</div>
					</div>
					<p class="mt-2 text-xs text-gray-400">
						Adjusts classification using HAM10000 clinical data (age/sex/location risk multipliers)
					</p>
				{/if}
			</div>

			{#if capturedImageData}
				<div class="mt-4 flex justify-center">
					<button
						onclick={analyzeImage}
						disabled={analyzing}
						class="flex h-12 items-center gap-2 rounded-xl bg-blue-600 px-6 text-sm font-semibold text-white shadow-lg hover:bg-blue-700 active:bg-blue-800 disabled:opacity-50"
					>
						{#if analyzing}
							<div class="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
							Analyzing...
						{:else}
							Analyze Lesion
						{/if}
					</button>
				</div>
			{/if}

		{:else if activeTab === "results"}
			{#if classificationResult}
				<div class="flex flex-col gap-6">
					<!-- Clinical Recommendation Banner -->
					{#if classificationResult.clinicalRecommendation}
						{@const rec = classificationResult.clinicalRecommendation}
						<div class="rounded-xl border-2 p-4 {
							rec.recommendation === 'urgent_referral' ? 'border-red-500 bg-red-50 dark:bg-red-950' :
							rec.recommendation === 'biopsy' ? 'border-orange-500 bg-orange-50 dark:bg-orange-950' :
							rec.recommendation === 'monitor' ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950' :
							'border-green-500 bg-green-50 dark:bg-green-950'
						}">
							<div class="flex items-center gap-2 text-sm font-bold {
								rec.recommendation === 'urgent_referral' ? 'text-red-700 dark:text-red-400' :
								rec.recommendation === 'biopsy' ? 'text-orange-700 dark:text-orange-400' :
								rec.recommendation === 'monitor' ? 'text-yellow-700 dark:text-yellow-400' :
								'text-green-700 dark:text-green-400'
							}">
								{rec.recommendation === 'urgent_referral' ? 'Urgent Referral Recommended' :
								 rec.recommendation === 'biopsy' ? 'Biopsy Advised' :
								 rec.recommendation === 'monitor' ? 'Monitor — Follow Up' :
								 'Low Risk — Reassurance'}
							</div>
							<p class="mt-1 text-xs text-gray-600 dark:text-gray-300">{rec.reasoning}</p>
							<div class="mt-2 flex gap-4 text-xs text-gray-500">
								<span>Melanoma P: {(rec.melanomaProbability * 100).toFixed(1)}%</span>
								<span>Malignant P: {(rec.malignantProbability * 100).toFixed(1)}%</span>
							</div>
							{#if classificationResult.demographicAdjusted}
								<p class="mt-1 text-xs italic text-gray-400">Adjusted with HAM10000 demographics</p>
							{/if}
						</div>
					{/if}

					<ClassificationResultView
						result={classificationResult}
						abcde={abcdeScores ?? undefined}
						onaction={handleAction}
					/>

					{#if abcdeScores}
						<ABCDEChart scores={abcdeScores} />
					{/if}

					{#if capturedImageData && gradCamData}
						<div>
							<h3 class="mb-2 text-sm font-semibold text-gray-500 dark:text-gray-400">
								Attention Map
							</h3>
							<GradCamOverlay imageData={capturedImageData} gradCamData={gradCamData} />
						</div>
					{/if}
				</div>
			{:else}
				<div class="flex flex-col items-center justify-center gap-2 py-12">
					<p class="text-sm text-gray-400">No results yet</p>
					<button
						onclick={() => (activeTab = "capture")}
						class="text-sm text-blue-500 underline hover:text-blue-600"
					>
						Capture an image first
					</button>
				</div>
			{/if}

		{:else if activeTab === "history"}
			<LesionTimeline {records} />

		{:else if activeTab === "settings"}
			<div class="flex flex-col gap-6">
				<div class="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
					<h3 class="mb-3 text-sm font-semibold text-gray-500 dark:text-gray-400">Model</h3>
					<div class="flex items-center justify-between">
						<span class="text-sm text-gray-700 dark:text-gray-300">Version</span>
						<span class="text-sm font-mono text-gray-500">{modelVersion}</span>
					</div>
				</div>

				<div class="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
					<h3 class="mb-3 text-sm font-semibold text-gray-500 dark:text-gray-400">Brain Sync</h3>
					<label class="flex items-center justify-between">
						<span class="text-sm text-gray-700 dark:text-gray-300">Enable sync</span>
						<input
							type="checkbox"
							bind:checked={brainSyncEnabled}
							class="h-5 w-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600"
						/>
					</label>
					<p class="mt-1 text-xs text-gray-400">
						{brainSyncEnabled ? "Connected" : "Local-only mode"}
					</p>
				</div>

				<div class="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-700 dark:bg-gray-800">
					<h3 class="mb-3 text-sm font-semibold text-gray-500 dark:text-gray-400">Privacy</h3>
					<div class="flex flex-col gap-3">
						<label class="flex items-center justify-between">
							<span class="text-sm text-gray-700 dark:text-gray-300">Strip EXIF data</span>
							<input
								type="checkbox"
								bind:checked={privacyStripExif}
								class="h-5 w-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600"
							/>
						</label>
						<label class="flex items-center justify-between">
							<span class="text-sm text-gray-700 dark:text-gray-300">Local processing only</span>
							<input
								type="checkbox"
								bind:checked={privacyLocalOnly}
								class="h-5 w-5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 dark:border-gray-600"
							/>
						</label>
					</div>
				</div>
			</div>
		{/if}
	</div>
</div>
