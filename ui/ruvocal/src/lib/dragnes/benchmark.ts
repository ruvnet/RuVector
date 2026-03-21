/**
 * DrAgnes Classification Benchmark Module
 *
 * Generates synthetic dermoscopic test images and runs classification
 * benchmarks to measure inference latency and per-class accuracy.
 */

import { DermClassifier } from "./classifier";
import type { LesionClass } from "./types";

/** Fitzpatrick skin phototype (I-VI) */
export type FitzpatrickType = "I" | "II" | "III" | "IV" | "V" | "VI";

/** Per-class accuracy metrics */
export interface ClassMetrics {
	className: LesionClass;
	truePositives: number;
	falsePositives: number;
	falseNegatives: number;
	trueNegatives: number;
	sensitivity: number;
	specificity: number;
	precision: number;
	f1: number;
}

/** Inference latency statistics in milliseconds */
export interface LatencyStats {
	min: number;
	max: number;
	mean: number;
	median: number;
	p95: number;
	p99: number;
	samples: number;
}

/** Full benchmark result */
export interface BenchmarkResult {
	totalImages: number;
	overallAccuracy: number;
	latency: LatencyStats;
	perClass: ClassMetrics[];
	modelId: string;
	usedWasm: boolean;
	runDate: string;
	durationMs: number;
}

const ALL_CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

/** Base skin tones per Fitzpatrick type (RGB) */
const SKIN_TONES: Record<FitzpatrickType, [number, number, number]> = {
	I: [255, 224, 196],
	II: [240, 200, 166],
	III: [210, 170, 130],
	IV: [175, 130, 90],
	V: [130, 90, 60],
	VI: [80, 55, 35],
};

/**
 * Color profiles for each lesion class.
 * Each entry defines primary color, secondary accents, and shape parameters.
 */
interface LesionProfile {
	primary: [number, number, number];
	secondary?: [number, number, number];
	irregularity: number; // 0-1, how irregular the border is
	multiColor: boolean;
}

const LESION_PROFILES: Record<LesionClass, LesionProfile> = {
	mel: {
		primary: [40, 20, 15],
		secondary: [60, 30, 80], // blue-black patches
		irregularity: 0.7,
		multiColor: true,
	},
	nv: {
		primary: [140, 90, 50],
		irregularity: 0.1,
		multiColor: false,
	},
	bcc: {
		primary: [200, 180, 170], // pearly/translucent
		secondary: [180, 60, 60], // visible vessels
		irregularity: 0.3,
		multiColor: true,
	},
	akiec: {
		primary: [180, 80, 60], // rough reddish
		irregularity: 0.5,
		multiColor: false,
	},
	bkl: {
		primary: [170, 140, 90], // waxy tan-brown
		irregularity: 0.2,
		multiColor: false,
	},
	df: {
		primary: [150, 100, 70], // firm brownish
		irregularity: 0.15,
		multiColor: false,
	},
	vasc: {
		primary: [190, 40, 50], // red/purple vascular
		secondary: [120, 30, 100],
		irregularity: 0.25,
		multiColor: true,
	},
};

/**
 * Generate a synthetic 224x224 dermoscopic image simulating a specific lesion class.
 *
 * @param lesionClass - Target HAM10000 class
 * @param fitzpatrickType - Skin phototype for background
 * @returns ImageData with realistic color distribution
 */
export function generateSyntheticLesion(
	lesionClass: LesionClass,
	fitzpatrickType: FitzpatrickType = "III"
): ImageData {
	const size = 224;
	const data = new Uint8ClampedArray(size * size * 4);
	const skin = SKIN_TONES[fitzpatrickType];
	const profile = LESION_PROFILES[lesionClass];

	const cx = size / 2 + (seededRandom(lesionClass.length) - 0.5) * 20;
	const cy = size / 2 + (seededRandom(lesionClass.length + 1) - 0.5) * 20;
	const baseRadius = size / 5 + seededRandom(lesionClass.length + 2) * 15;

	for (let y = 0; y < size; y++) {
		for (let x = 0; x < size; x++) {
			const idx = (y * size + x) * 4;

			// Compute distance with border irregularity
			const angle = Math.atan2(y - cy, x - cx);
			const radiusVariation = 1 + profile.irregularity * 0.3 *
				(Math.sin(angle * 5) * 0.5 + Math.sin(angle * 3) * 0.3 + Math.sin(angle * 7) * 0.2);
			const effectiveRadius = baseRadius * radiusVariation;
			const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);

			if (dist < effectiveRadius) {
				// Inside lesion
				const t = dist / effectiveRadius; // 0 at center, 1 at border
				const [pr, pg, pb] = profile.primary;

				if (profile.multiColor && profile.secondary && t > 0.4) {
					// Blend secondary color in outer region
					const blend = (t - 0.4) / 0.6;
					const [sr, sg, sb] = profile.secondary;
					data[idx] = Math.round(pr * (1 - blend) + sr * blend);
					data[idx + 1] = Math.round(pg * (1 - blend) + sg * blend);
					data[idx + 2] = Math.round(pb * (1 - blend) + sb * blend);
				} else {
					// Slight gradient from center to edge
					data[idx] = Math.round(pr + (skin[0] - pr) * t * 0.3);
					data[idx + 1] = Math.round(pg + (skin[1] - pg) * t * 0.3);
					data[idx + 2] = Math.round(pb + (skin[2] - pb) * t * 0.3);
				}
			} else if (dist < effectiveRadius + 5) {
				// Border transition zone
				const blend = (dist - effectiveRadius) / 5;
				data[idx] = Math.round(profile.primary[0] * (1 - blend) + skin[0] * blend);
				data[idx + 1] = Math.round(profile.primary[1] * (1 - blend) + skin[1] * blend);
				data[idx + 2] = Math.round(profile.primary[2] * (1 - blend) + skin[2] * blend);
			} else {
				// Skin background with slight variation
				data[idx] = clampByte(skin[0] + (hashNoise(x, y) - 0.5) * 10);
				data[idx + 1] = clampByte(skin[1] + (hashNoise(x + 1000, y) - 0.5) * 10);
				data[idx + 2] = clampByte(skin[2] + (hashNoise(x, y + 1000) - 0.5) * 10);
			}
			data[idx + 3] = 255;
		}
	}

	return new ImageData(data, size, size);
}

/**
 * Run a full classification benchmark with synthetic images.
 *
 * Generates 100 test images (varied classes and Fitzpatrick types),
 * classifies each, and computes latency and accuracy metrics.
 *
 * @param classifier - Optional pre-initialized DermClassifier
 * @returns Complete benchmark results
 */
export async function runBenchmark(classifier?: DermClassifier): Promise<BenchmarkResult> {
	const cls = classifier ?? new DermClassifier();
	await cls.init();

	const fitzpatrickTypes: FitzpatrickType[] = ["I", "II", "III", "IV", "V", "VI"];
	const totalImages = 100;
	const imagesPerClass = Math.floor(totalImages / ALL_CLASSES.length);
	const remainder = totalImages - imagesPerClass * ALL_CLASSES.length;

	// Generate test set: ground truth labels + images
	const testSet: Array<{ image: ImageData; groundTruth: LesionClass }> = [];

	for (let ci = 0; ci < ALL_CLASSES.length; ci++) {
		const count = ci < remainder ? imagesPerClass + 1 : imagesPerClass;
		for (let i = 0; i < count; i++) {
			const fitz = fitzpatrickTypes[(ci * imagesPerClass + i) % fitzpatrickTypes.length];
			testSet.push({
				image: generateSyntheticLesion(ALL_CLASSES[ci], fitz),
				groundTruth: ALL_CLASSES[ci],
			});
		}
	}

	// Run inference and collect results
	const latencies: number[] = [];
	const predictions: Array<{ predicted: LesionClass; actual: LesionClass }> = [];
	let modelId = "";
	let usedWasm = false;

	const startTime = performance.now();

	for (const { image, groundTruth } of testSet) {
		const t0 = performance.now();
		const result = await cls.classify(image);
		const elapsed = performance.now() - t0;

		latencies.push(elapsed);
		predictions.push({ predicted: result.topClass, actual: groundTruth });
		modelId = result.modelId;
		usedWasm = result.usedWasm;
	}

	const durationMs = Math.round(performance.now() - startTime);

	// Compute latency stats
	const sortedLatencies = [...latencies].sort((a, b) => a - b);
	const latency: LatencyStats = {
		min: sortedLatencies[0],
		max: sortedLatencies[sortedLatencies.length - 1],
		mean: latencies.reduce((a, b) => a + b, 0) / latencies.length,
		median: sortedLatencies[Math.floor(sortedLatencies.length / 2)],
		p95: sortedLatencies[Math.floor(sortedLatencies.length * 0.95)],
		p99: sortedLatencies[Math.floor(sortedLatencies.length * 0.99)],
		samples: latencies.length,
	};

	// Compute per-class metrics
	const perClass: ClassMetrics[] = ALL_CLASSES.map((cls) => {
		const tp = predictions.filter((p) => p.predicted === cls && p.actual === cls).length;
		const fp = predictions.filter((p) => p.predicted === cls && p.actual !== cls).length;
		const fn = predictions.filter((p) => p.predicted !== cls && p.actual === cls).length;
		const tn = predictions.filter((p) => p.predicted !== cls && p.actual !== cls).length;

		const sensitivity = tp + fn > 0 ? tp / (tp + fn) : 0;
		const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
		const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
		const f1 = precision + sensitivity > 0
			? (2 * precision * sensitivity) / (precision + sensitivity)
			: 0;

		return { className: cls, truePositives: tp, falsePositives: fp, falseNegatives: fn, trueNegatives: tn, sensitivity, specificity, precision, f1 };
	});

	const correct = predictions.filter((p) => p.predicted === p.actual).length;

	return {
		totalImages,
		overallAccuracy: correct / totalImages,
		latency,
		perClass,
		modelId,
		usedWasm,
		runDate: new Date().toISOString(),
		durationMs,
	};
}

/** Deterministic pseudo-random from seed */
function seededRandom(seed: number): number {
	const x = Math.sin(seed * 9301 + 49297) * 233280;
	return x - Math.floor(x);
}

/** Deterministic noise for pixel variation */
function hashNoise(x: number, y: number): number {
	const n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
	return n - Math.floor(n);
}

/** Clamp to valid byte range */
function clampByte(v: number): number {
	return Math.max(0, Math.min(255, Math.round(v)));
}
