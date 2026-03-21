/**
 * DrAgnes Benchmark Module Tests
 *
 * Tests synthetic image generation, benchmark execution,
 * latency measurement, and per-class metric computation.
 */

import { describe, it, expect } from "vitest";
import {
	generateSyntheticLesion,
	runBenchmark,
	type BenchmarkResult,
	type ClassMetrics,
	type LatencyStats,
	type FitzpatrickType,
} from "./benchmark";
import { DermClassifier } from "./classifier";
import type { LesionClass } from "./types";

// ---- Polyfill ImageData for Node.js ----

if (typeof globalThis.ImageData === "undefined") {
	(globalThis as Record<string, unknown>).ImageData = class ImageData {
		readonly data: Uint8ClampedArray;
		readonly width: number;
		readonly height: number;
		readonly colorSpace: string = "srgb";

		constructor(dataOrWidth: Uint8ClampedArray | number, widthOrHeight: number, height?: number) {
			if (dataOrWidth instanceof Uint8ClampedArray) {
				this.data = dataOrWidth;
				this.width = widthOrHeight;
				this.height = height ?? dataOrWidth.length / 4 / widthOrHeight;
			} else {
				this.width = dataOrWidth;
				this.height = widthOrHeight;
				this.data = new Uint8ClampedArray(this.width * this.height * 4);
			}
		}
	};
}

// ---- Synthetic Image Generation Tests ----

describe("generateSyntheticLesion", () => {
	const ALL_CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
	const ALL_FITZPATRICK: FitzpatrickType[] = ["I", "II", "III", "IV", "V", "VI"];

	it("should generate 224x224 RGBA ImageData for each class", () => {
		for (const cls of ALL_CLASSES) {
			const img = generateSyntheticLesion(cls);
			expect(img.width).toBe(224);
			expect(img.height).toBe(224);
			expect(img.data.length).toBe(224 * 224 * 4);
		}
	});

	it("should produce valid pixel values (0-255)", () => {
		for (const cls of ALL_CLASSES) {
			const img = generateSyntheticLesion(cls);
			for (let i = 0; i < img.data.length; i++) {
				expect(img.data[i]).toBeGreaterThanOrEqual(0);
				expect(img.data[i]).toBeLessThanOrEqual(255);
			}
		}
	});

	it("should set alpha channel to 255 for all pixels", () => {
		const img = generateSyntheticLesion("mel");
		for (let i = 3; i < img.data.length; i += 4) {
			expect(img.data[i]).toBe(255);
		}
	});

	it("should produce different color profiles for different classes", () => {
		const avgColors: Record<string, [number, number, number]> = {};

		for (const cls of ALL_CLASSES) {
			const img = generateSyntheticLesion(cls);
			let totalR = 0, totalG = 0, totalB = 0;
			const pixelCount = img.width * img.height;

			for (let i = 0; i < img.data.length; i += 4) {
				totalR += img.data[i];
				totalG += img.data[i + 1];
				totalB += img.data[i + 2];
			}

			avgColors[cls] = [totalR / pixelCount, totalG / pixelCount, totalB / pixelCount];
		}

		// Melanoma should be darker than nevus on average
		const melBrightness = avgColors.mel[0] + avgColors.mel[1] + avgColors.mel[2];
		const nvBrightness = avgColors.nv[0] + avgColors.nv[1] + avgColors.nv[2];
		expect(melBrightness).toBeLessThan(nvBrightness);

		// Vascular lesions should have higher red component relative to blue
		expect(avgColors.vasc[0]).toBeGreaterThan(avgColors.vasc[2]);
	});

	it("should vary background skin tone with Fitzpatrick type", () => {
		const brightnesses: number[] = [];

		for (const fitz of ALL_FITZPATRICK) {
			const img = generateSyntheticLesion("nv", fitz);
			// Sample corner pixel (should be skin background)
			const idx = 0; // top-left pixel
			const brightness = img.data[idx] + img.data[idx + 1] + img.data[idx + 2];
			brightnesses.push(brightness);
		}

		// Fitzpatrick I should be brightest, VI darkest
		expect(brightnesses[0]).toBeGreaterThan(brightnesses[5]);
	});

	it("should generate distinct images for mel class with multicolor", () => {
		const img = generateSyntheticLesion("mel", "III");
		const cx = 112, cy = 112; // center
		const centerIdx = (cy * 224 + cx) * 4;
		const edgeIdx = (cy * 224 + cx + 40) * 4; // offset toward border

		// Center and edge should have different colors for multicolor lesions
		const centerColor = [img.data[centerIdx], img.data[centerIdx + 1], img.data[centerIdx + 2]];
		const edgeColor = [img.data[edgeIdx], img.data[edgeIdx + 1], img.data[edgeIdx + 2]];

		const colorDiff = Math.abs(centerColor[0] - edgeColor[0]) +
			Math.abs(centerColor[1] - edgeColor[1]) +
			Math.abs(centerColor[2] - edgeColor[2]);

		// Multicolor melanoma should show color variation between center and edge
		expect(colorDiff).toBeGreaterThan(0);
	});
});

// ---- Benchmark Execution Tests ----

describe("runBenchmark", () => {
	it("should return a complete BenchmarkResult", async () => {
		const classifier = new DermClassifier();
		await classifier.init();
		const result = await runBenchmark(classifier);

		expect(result.totalImages).toBe(100);
		expect(result.overallAccuracy).toBeGreaterThanOrEqual(0);
		expect(result.overallAccuracy).toBeLessThanOrEqual(1);
		expect(result.modelId).toBeDefined();
		expect(result.runDate).toBeDefined();
		expect(result.durationMs).toBeGreaterThan(0);
	}, 30000);

	it("should include latency stats with correct structure", async () => {
		const classifier = new DermClassifier();
		await classifier.init();
		const result = await runBenchmark(classifier);
		const latency = result.latency;

		expect(latency.samples).toBe(100);
		expect(latency.min).toBeGreaterThanOrEqual(0);
		expect(latency.max).toBeGreaterThanOrEqual(latency.min);
		expect(latency.mean).toBeGreaterThanOrEqual(latency.min);
		expect(latency.mean).toBeLessThanOrEqual(latency.max);
		expect(latency.median).toBeGreaterThanOrEqual(latency.min);
		expect(latency.median).toBeLessThanOrEqual(latency.max);
		expect(latency.p95).toBeGreaterThanOrEqual(latency.median);
		expect(latency.p99).toBeGreaterThanOrEqual(latency.p95);
	}, 30000);

	it("should compute per-class metrics for all 7 classes", async () => {
		const classifier = new DermClassifier();
		await classifier.init();
		const result = await runBenchmark(classifier);

		expect(result.perClass).toHaveLength(7);

		const classNames = result.perClass.map((m) => m.className);
		expect(classNames).toContain("akiec");
		expect(classNames).toContain("bcc");
		expect(classNames).toContain("bkl");
		expect(classNames).toContain("df");
		expect(classNames).toContain("mel");
		expect(classNames).toContain("nv");
		expect(classNames).toContain("vasc");
	}, 30000);

	it("should have valid per-class metric ranges", async () => {
		const classifier = new DermClassifier();
		await classifier.init();
		const result = await runBenchmark(classifier);

		for (const metrics of result.perClass) {
			expect(metrics.sensitivity).toBeGreaterThanOrEqual(0);
			expect(metrics.sensitivity).toBeLessThanOrEqual(1);
			expect(metrics.specificity).toBeGreaterThanOrEqual(0);
			expect(metrics.specificity).toBeLessThanOrEqual(1);
			expect(metrics.precision).toBeGreaterThanOrEqual(0);
			expect(metrics.precision).toBeLessThanOrEqual(1);
			expect(metrics.f1).toBeGreaterThanOrEqual(0);
			expect(metrics.f1).toBeLessThanOrEqual(1);
			expect(metrics.truePositives + metrics.falseNegatives).toBeGreaterThan(0);
		}
	}, 30000);

	it("should sum TP+FP+FN+TN to total for each class", async () => {
		const classifier = new DermClassifier();
		await classifier.init();
		const result = await runBenchmark(classifier);

		for (const metrics of result.perClass) {
			const total = metrics.truePositives + metrics.falsePositives +
				metrics.falseNegatives + metrics.trueNegatives;
			expect(total).toBe(100);
		}
	}, 30000);
});
