/**
 * DrAgnes CNN Classification Engine
 *
 * Loads MobileNetV3 Small WASM module from @ruvector/cnn for
 * browser-based skin lesion classification. Falls back to a
 * demo classifier using color/texture analysis when WASM is unavailable.
 *
 * Supports Grad-CAM heatmap generation for attention visualization.
 */

import type {
	ClassificationResult,
	ClassProbability,
	GradCamResult,
	ImageTensor,
	LesionClass,
} from "./types";
import { LESION_LABELS } from "./types";
import { preprocessImage, resizeBilinear, toNCHWTensor } from "./preprocessing";

/** All HAM10000 classes in canonical order */
const CLASSES: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];

/** Interface for the WASM CNN module */
interface WasmCnnModule {
	init(modelPath?: string): Promise<void>;
	predict(tensor: Float32Array, shape: number[]): Promise<Float32Array>;
	gradCam(tensor: Float32Array, classIdx: number): Promise<Float32Array>;
}

/**
 * Dermoscopy CNN classifier with WASM backend and demo fallback.
 */
export class DermClassifier {
	private wasmModule: WasmCnnModule | null = null;
	private initialized = false;
	private usesWasm = false;
	private lastTensor: ImageTensor | null = null;
	private lastImageData: ImageData | null = null;

	/**
	 * Initialize the classifier.
	 * Attempts to load the @ruvector/cnn WASM module.
	 * Falls back to demo mode if unavailable.
	 */
	async init(): Promise<void> {
		if (this.initialized) return;

		try {
			// Dynamic import of the WASM CNN package
			// Use variable to prevent Vite from pre-bundling this optional dependency
			const moduleName = "@ruvector/cnn";
			const cnnModule = await import(/* @vite-ignore */ moduleName);
			if (cnnModule && typeof cnnModule.init === "function") {
				await cnnModule.init();
				this.wasmModule = cnnModule;
				this.usesWasm = true;
			}
		} catch {
			// WASM module not available, use demo fallback
			this.wasmModule = null;
			this.usesWasm = false;
		}

		this.initialized = true;
	}

	/**
	 * Classify a dermoscopic image.
	 *
	 * @param imageData - RGBA ImageData from canvas
	 * @returns Classification result with probabilities for all 7 classes
	 */
	async classify(imageData: ImageData): Promise<ClassificationResult> {
		if (!this.initialized) {
			await this.init();
		}

		const startTime = performance.now();

		// Preprocess: normalize, resize, convert to NCHW tensor
		const tensor = await preprocessImage(imageData);
		this.lastTensor = tensor;
		this.lastImageData = imageData;

		let rawProbabilities: number[];

		if (this.usesWasm && this.wasmModule) {
			rawProbabilities = await this.classifyWasm(tensor);
		} else {
			rawProbabilities = this.classifyDemo(imageData);
		}

		const inferenceTimeMs = Math.round(performance.now() - startTime);

		// Build sorted probabilities
		const probabilities: ClassProbability[] = CLASSES.map((cls, i) => ({
			className: cls,
			probability: rawProbabilities[i],
			label: LESION_LABELS[cls],
		})).sort((a, b) => b.probability - a.probability);

		const topClass = probabilities[0].className;
		const confidence = probabilities[0].probability;

		return {
			topClass,
			confidence,
			probabilities,
			modelId: this.usesWasm ? "mobilenetv3-small-wasm" : "demo-color-texture",
			inferenceTimeMs,
			usedWasm: this.usesWasm,
		};
	}

	/**
	 * Generate Grad-CAM heatmap for the last classified image.
	 *
	 * @param targetClass - Optional class to explain (defaults to top predicted)
	 * @returns Grad-CAM heatmap and overlay
	 */
	async getGradCam(targetClass?: LesionClass): Promise<GradCamResult> {
		if (!this.lastTensor || !this.lastImageData) {
			throw new Error("No image classified yet. Call classify() first.");
		}

		const classIdx = targetClass ? CLASSES.indexOf(targetClass) : 0;
		const target = targetClass || CLASSES[0];

		if (this.usesWasm && this.wasmModule) {
			return this.gradCamWasm(classIdx, target);
		}

		return this.gradCamDemo(target);
	}

	/**
	 * Check if the WASM module is loaded.
	 */
	isWasmLoaded(): boolean {
		return this.usesWasm;
	}

	/**
	 * Check if the classifier is initialized.
	 */
	isInitialized(): boolean {
		return this.initialized;
	}

	// ---- WASM backend ----

	private async classifyWasm(tensor: ImageTensor): Promise<number[]> {
		const raw = await this.wasmModule!.predict(tensor.data, [...tensor.shape]);
		return softmax(Array.from(raw));
	}

	private async gradCamWasm(classIdx: number, target: LesionClass): Promise<GradCamResult> {
		const rawHeatmap = await this.wasmModule!.gradCam(this.lastTensor!.data, classIdx);
		const heatmap = heatmapToImageData(rawHeatmap, 224, 224);
		const overlay = overlayHeatmap(this.lastImageData!, heatmap);

		return { heatmap, overlay, targetClass: target };
	}

	// ---- Demo fallback ----

	/**
	 * Demo classifier using color and texture analysis.
	 * Provides plausible probabilities based on image characteristics.
	 */
	private classifyDemo(imageData: ImageData): number[] {
		const { data, width, height } = imageData;
		const pixelCount = width * height;

		// Analyze color distribution
		let totalR = 0,
			totalG = 0,
			totalB = 0;
		let darkPixels = 0,
			redPixels = 0,
			brownPixels = 0,
			bluePixels = 0;

		for (let i = 0; i < data.length; i += 4) {
			const r = data[i],
				g = data[i + 1],
				b = data[i + 2];
			totalR += r;
			totalG += g;
			totalB += b;

			const brightness = (r + g + b) / 3;
			if (brightness < 60) darkPixels++;
			if (r > 150 && g < 100 && b < 100) redPixels++;
			if (r > 100 && r < 180 && g > 50 && g < 120 && b > 30 && b < 80) brownPixels++;
			if (b > 120 && r < 100 && g < 120) bluePixels++;
		}

		const avgR = totalR / pixelCount;
		const avgG = totalG / pixelCount;
		const avgB = totalB / pixelCount;
		const darkRatio = darkPixels / pixelCount;
		const redRatio = redPixels / pixelCount;
		const brownRatio = brownPixels / pixelCount;
		const blueRatio = bluePixels / pixelCount;

		// Generate class scores based on color features
		const scores = [
			0.1 + brownRatio * 2 + redRatio * 0.5, // akiec
			0.1 + redRatio * 1.5 + avgR / 500, // bcc
			0.15 + brownRatio * 1.5 + avgG / 600, // bkl
			0.05 + brownRatio * 0.5 + redRatio * 0.3, // df
			0.1 + darkRatio * 3 + blueRatio * 2, // mel
			0.3 + brownRatio * 1.0 + (1 - darkRatio) * 0.2, // nv (most common)
			0.05 + redRatio * 3 + blueRatio * 0.5, // vasc
		];

		return softmax(scores);
	}

	private gradCamDemo(target: LesionClass): GradCamResult {
		const size = 224;
		const heatmapData = new Float32Array(size * size);

		// Generate a Gaussian-centered heatmap (simulated attention)
		const cx = size / 2,
			cy = size / 2;
		const sigma = size / 4;

		for (let y = 0; y < size; y++) {
			for (let x = 0; x < size; x++) {
				const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
				heatmapData[y * size + x] = Math.exp(-(dist ** 2) / (2 * sigma ** 2));
			}
		}

		// Add some noise for realism
		for (let i = 0; i < heatmapData.length; i++) {
			heatmapData[i] = Math.max(0, Math.min(1, heatmapData[i] + (Math.random() - 0.5) * 0.1));
		}

		const heatmap = heatmapToImageData(heatmapData, size, size);
		const resizedOriginal = resizeBilinear(this.lastImageData!, size, size);
		const overlay = overlayHeatmap(resizedOriginal, heatmap);

		return { heatmap, overlay, targetClass: target };
	}
}

/**
 * Softmax activation function.
 */
function softmax(logits: number[]): number[] {
	const maxLogit = Math.max(...logits);
	const exps = logits.map((l) => Math.exp(l - maxLogit));
	const sum = exps.reduce((a, b) => a + b, 0);
	return exps.map((e) => e / sum);
}

/**
 * Convert a Float32 heatmap [0,1] to RGBA ImageData using a jet colormap.
 */
function heatmapToImageData(heatmap: Float32Array, width: number, height: number): ImageData {
	const rgba = new Uint8ClampedArray(width * height * 4);

	for (let i = 0; i < heatmap.length; i++) {
		const v = Math.max(0, Math.min(1, heatmap[i]));
		const px = i * 4;

		// Jet colormap approximation
		if (v < 0.25) {
			rgba[px] = 0;
			rgba[px + 1] = Math.round(v * 4 * 255);
			rgba[px + 2] = 255;
		} else if (v < 0.5) {
			rgba[px] = 0;
			rgba[px + 1] = 255;
			rgba[px + 2] = Math.round((1 - (v - 0.25) * 4) * 255);
		} else if (v < 0.75) {
			rgba[px] = Math.round((v - 0.5) * 4 * 255);
			rgba[px + 1] = 255;
			rgba[px + 2] = 0;
		} else {
			rgba[px] = 255;
			rgba[px + 1] = Math.round((1 - (v - 0.75) * 4) * 255);
			rgba[px + 2] = 0;
		}
		rgba[px + 3] = Math.round(v * 180); // Alpha based on intensity
	}

	return new ImageData(rgba, width, height);
}

/**
 * Overlay a heatmap on the original image with alpha blending.
 */
function overlayHeatmap(original: ImageData, heatmap: ImageData): ImageData {
	const width = heatmap.width;
	const height = heatmap.height;
	const resized = original.width === width && original.height === height
		? original
		: resizeBilinear(original, width, height);

	const result = new Uint8ClampedArray(width * height * 4);

	for (let i = 0; i < width * height; i++) {
		const px = i * 4;
		const alpha = heatmap.data[px + 3] / 255;

		result[px] = Math.round(resized.data[px] * (1 - alpha) + heatmap.data[px] * alpha);
		result[px + 1] = Math.round(resized.data[px + 1] * (1 - alpha) + heatmap.data[px + 1] * alpha);
		result[px + 2] = Math.round(resized.data[px + 2] * (1 - alpha) + heatmap.data[px + 2] * alpha);
		result[px + 3] = 255;
	}

	return new ImageData(result, width, height);
}
