/**
 * ONNX WASM Embedder - Semantic embeddings for hooks
 *
 * Provides real transformer-based embeddings using all-MiniLM-L6-v2
 * running in pure WASM (no native dependencies).
 *
 * Uses bundled ONNX WASM files from src/core/onnx/
 *
 * Features:
 * - 384-dimensional semantic embeddings
 * - Real semantic understanding (not hash-based)
 * - Cached model loading (downloads from HuggingFace on first use)
 * - Batch embedding support
 * - Optional parallel workers for 3.8x batch speedup
 */

import * as path from 'path';
import * as fs from 'fs';
import { pathToFileURL } from 'url';
import { createRequire } from 'module';

// Extend globalThis type for ESM require compatibility
declare global {
  // eslint-disable-next-line no-var
  var __ruvector_require: NodeRequire | undefined;
}

// Set up ESM-compatible require for WASM module (fixes Windows/ESM compatibility)
// The WASM bindings use module.require for Node.js crypto, this provides a fallback
if (typeof globalThis !== 'undefined' && !globalThis.__ruvector_require) {
  try {
    // In ESM context, use createRequire with __filename
    globalThis.__ruvector_require = createRequire(__filename);
  } catch {
    // Fallback: require should be available in CommonJS
    try {
      globalThis.__ruvector_require = require;
    } catch {
      // Neither available - WASM will fall back to crypto.getRandomValues
    }
  }
}

// Force native dynamic import (avoids TypeScript transpiling to require)
// eslint-disable-next-line @typescript-eslint/no-implied-eval
const dynamicImport = new Function('specifier', 'return import(specifier)') as (specifier: string) => Promise<any>;

// Types
export interface OnnxEmbedderConfig {
  modelId?: string;
  maxLength?: number;
  normalize?: boolean;
  cacheDir?: string;
  /**
   * Enable parallel workers for batch operations
   * - 'auto' (default): Enable for long-running processes, skip for CLI
   * - true: Always enable workers
   * - false: Never use workers
   */
  enableParallel?: boolean | 'auto';
  /** Number of worker threads (default: CPU cores - 1) */
  numWorkers?: number;
  /** Minimum batch size to use parallel processing (default: 4) */
  parallelThreshold?: number;
}

// Capability detection
let simdAvailable = false;
let parallelAvailable = false;

export interface EmbeddingResult {
  embedding: number[];
  dimension: number;
  timeMs: number;
}

export interface SimilarityResult {
  similarity: number;
  timeMs: number;
}

// Lazy-loaded module state
let wasmModule: any = null;
let embedder: any = null;
let parallelEmbedder: any = null;
let loadError: Error | null = null;
let loadPromise: Promise<void> | null = null;
let isInitialized = false;
let parallelEnabled = false;
let parallelThreshold = 4;

// Captured at init so the bundled worker pool can reuse the loaded model bytes
// (shared to workers via SharedArrayBuffer) instead of re-downloading per worker.
let loadedModelBytes: Uint8Array | null = null;
let loadedTokenizerJson: string | null = null;
let loadedMaxLength = 256;
let bundledPool: any = null;

// Default model
const DEFAULT_MODEL = 'all-MiniLM-L6-v2';

/**
 * Check if the ONNX embedder is *available* — i.e. the bundled WASM files are
 * present and the embedder can be initialized.
 *
 * NOTE: This is a capability check, NOT a readiness check. It returns `true`
 * before `initOnnxEmbedder()` has run (so callers can decide whether to init).
 * To check whether the model has actually been loaded, use `isOnnxInitialized()`
 * or `isReady()`. See https://github.com/ruvnet/RuVector/issues/523.
 */
export function isOnnxAvailable(): boolean {
  try {
    const pkgPath = path.join(__dirname, 'onnx', 'pkg', 'ruvector_onnx_embeddings_wasm.js');
    return fs.existsSync(pkgPath);
  } catch {
    return false;
  }
}

/**
 * Check whether the bundled parallel worker pool can be loaded — i.e. the
 * `onnx/bundled-parallel.mjs` file ships in the package. This reflects the
 * *bundled* pool (the only parallel implementation), NOT the unpublished
 * external `ruvector-onnx-embeddings-wasm/parallel` package, which was rejected
 * in ADR-194. See https://github.com/ruvnet/RuVector/issues/531.
 */
function detectParallelAvailable(): boolean {
  try {
    const poolPath = path.join(__dirname, 'onnx', 'bundled-parallel.mjs');
    parallelAvailable = fs.existsSync(poolPath);
    return parallelAvailable;
  } catch {
    parallelAvailable = false;
    return false;
  }
}

/**
 * Check if SIMD is available (from WASM module)
 */
function detectSimd(): boolean {
  try {
    if (wasmModule && typeof wasmModule.simd_available === 'function') {
      simdAvailable = wasmModule.simd_available();
      return simdAvailable;
    }
  } catch {}
  return false;
}

/**
 * Initialize the bundled, zero-dependency worker pool for batch throughput.
 *
 * Opt-in only (`enableParallel === true`) so the default/'auto' path does not
 * silently spawn worker threads for existing callers. Output vectors are
 * bit-identical to the single-thread path (issue #523).
 *
 * The previously-referenced external package
 * `ruvector-onnx-embeddings-wasm/parallel` was never published and was rejected
 * in ADR-194; the bundled pool (`onnx/bundled-parallel.mjs`) is the only
 * parallel implementation. See https://github.com/ruvnet/RuVector/issues/531.
 */
async function tryInitParallel(config: OnnxEmbedderConfig): Promise<boolean> {
  // Skip unless parallelism is explicitly requested (covers false and 'auto').
  if (config.enableParallel !== true) {
    parallelAvailable = false;
    return false;
  }
  if (!detectParallelAvailable()) {
    console.error('Parallel embedder not available: bundled worker pool (onnx/bundled-parallel.mjs) missing');
    return false;
  }
  try {
    if (!loadedModelBytes || !loadedTokenizerJson) {
      throw new Error('model bytes unavailable for bundled pool');
    }
    const poolUrl = pathToFileURL(path.join(__dirname, 'onnx', 'bundled-parallel.mjs')).href;
    const { ParallelEmbedder } = await dynamicImport(poolUrl);
    const pool = new ParallelEmbedder({
      modelBytes: loadedModelBytes,
      tokenizerJson: loadedTokenizerJson,
      maxLength: loadedMaxLength,
      dimension: embedder ? embedder.dimension() : 384,
      numWorkers: config.numWorkers,
    });
    await pool.init();
    parallelEmbedder = pool;
    parallelThreshold = config.parallelThreshold || 4;
    parallelEnabled = true;
    parallelAvailable = true;
    console.error(`Parallel embedder ready (bundled): ${pool.numWorkers} workers, SIMD: ${simdAvailable}`);
    return true;
  } catch (e: any) {
    parallelAvailable = false;
    console.error(`Parallel embedder not available: ${e.message}`);
    return false;
  }
}

/**
 * Initialize the ONNX embedder (downloads model if needed)
 */
export async function initOnnxEmbedder(config: OnnxEmbedderConfig = {}): Promise<boolean> {
  if (isInitialized) return true;
  if (loadError) throw loadError;
  if (loadPromise) {
    await loadPromise;
    return isInitialized;
  }

  loadPromise = (async () => {
    try {
      // Paths to bundled ONNX files
      const bgJsPath = path.join(__dirname, 'onnx', 'pkg', 'ruvector_onnx_embeddings_wasm_bg.js');
      const wasmPath = path.join(__dirname, 'onnx', 'pkg', 'ruvector_onnx_embeddings_wasm_bg.wasm');
      const loaderPath = path.join(__dirname, 'onnx', 'loader.js');

      if (!fs.existsSync(bgJsPath) || !fs.existsSync(wasmPath)) {
        throw new Error('ONNX WASM files not bundled. The onnx/ directory is missing.');
      }

      // Load the bg.js module directly (avoids the ESM `import * as wasm from "*.wasm"`
      // in the main .js shim which requires --experimental-wasm-modules on Node 18-24).
      const bgUrl = pathToFileURL(bgJsPath).href;
      const loaderUrl = pathToFileURL(loaderPath).href;
      wasmModule = await dynamicImport(bgUrl);

      // Instantiate the .wasm bytes via WebAssembly API (no --experimental-wasm-modules needed).
      const wasmBytes = fs.readFileSync(wasmPath);
      const wasmResult = await WebAssembly.instantiate(wasmBytes, { './ruvector_onnx_embeddings_wasm_bg.js': wasmModule });
      const wasmExports = wasmResult.instance.exports;
      if (typeof wasmModule.__wbg_set_wasm === 'function') {
        wasmModule.__wbg_set_wasm(wasmExports);
      }
      if (typeof (wasmExports as any).__wbindgen_start === 'function') {
        (wasmExports as any).__wbindgen_start();
      }

      const loaderModule = await dynamicImport(loaderUrl);
      const { ModelLoader } = loaderModule;

      // Create model loader with caching
      const modelLoader = new ModelLoader({
        cache: true,
        cacheDir: config.cacheDir || path.join(process.env.HOME || '/tmp', '.ruvector', 'models'),
      });

      // Load model (downloads from HuggingFace on first use)
      const modelId = config.modelId || DEFAULT_MODEL;
      console.error(`Loading ONNX model: ${modelId}...`);

      const { modelBytes, tokenizerJson, config: modelConfig } = await modelLoader.loadModel(modelId);

      // Retain for the bundled parallel worker pool (see initParallelEmbedder).
      loadedModelBytes = modelBytes;
      loadedTokenizerJson = tokenizerJson;
      loadedMaxLength = config.maxLength || modelConfig.maxLength || 256;

      // Create embedder with config
      const embedderConfig = new wasmModule.WasmEmbedderConfig()
        .setMaxLength(config.maxLength || modelConfig.maxLength || 256)
        .setNormalize(config.normalize !== false)
        .setPooling(0); // Mean pooling

      embedder = wasmModule.WasmEmbedder.withConfig(modelBytes, tokenizerJson, embedderConfig);

      // Detect SIMD capability
      detectSimd();
      console.error(`ONNX embedder ready: ${embedder.dimension()}d, SIMD: ${simdAvailable}`);

      isInitialized = true;

      // Determine if we should use parallel workers
      // - true: always enable
      // - false: never enable
      // - 'auto'/undefined: enable for long-running processes (MCP, servers), skip for CLI
      let shouldTryParallel = false;
      if (config.enableParallel === true) {
        shouldTryParallel = true;
      } else if (config.enableParallel === false) {
        shouldTryParallel = false;
      } else {
        // Auto-detect: check if running as CLI hook or long-running process
        const isCLI = process.argv[1]?.includes('cli.js') ||
                      process.argv[1]?.includes('bin/ruvector') ||
                      process.env.RUVECTOR_CLI === '1';
        const isMCP = process.env.MCP_SERVER === '1' ||
                      process.argv.some(a => a.includes('mcp'));
        const forceParallel = process.env.RUVECTOR_PARALLEL === '1';

        // Enable parallel for MCP/servers or if explicitly requested, skip for CLI
        shouldTryParallel = forceParallel || (isMCP && !isCLI);
      }

      if (shouldTryParallel) {
        await tryInitParallel(config);
      }
    } catch (e: any) {
      loadError = new Error(`Failed to initialize ONNX embedder: ${e.message}`);
      throw loadError;
    }
  })();

  await loadPromise;
  return isInitialized;
}

/**
 * Generate embedding for text
 */
export async function embed(text: string): Promise<EmbeddingResult> {
  if (!isInitialized) {
    await initOnnxEmbedder();
  }
  if (!embedder) {
    throw new Error('ONNX embedder not initialized');
  }

  const start = performance.now();
  const embedding = embedder.embedOne(text);
  const timeMs = performance.now() - start;

  return {
    embedding: Array.from(embedding),
    dimension: embedding.length,
    timeMs,
  };
}

/**
 * Generate embeddings for multiple texts
 * Uses parallel workers automatically for batches >= parallelThreshold
 */
export async function embedBatch(texts: string[]): Promise<EmbeddingResult[]> {
  if (!isInitialized) {
    await initOnnxEmbedder();
  }
  if (!embedder) {
    throw new Error('ONNX embedder not initialized');
  }

  const start = performance.now();

  // Use parallel workers for large batches
  if (parallelEnabled && parallelEmbedder && texts.length >= parallelThreshold) {
    const batchResults = await parallelEmbedder.embedBatch(texts);
    const totalTime = performance.now() - start;
    const dimension = parallelEmbedder.dimension || 384;

    return batchResults.map((emb: number[]) => ({
      embedding: Array.from(emb),
      dimension,
      timeMs: totalTime / texts.length,
    }));
  }

  // Sequential fallback
  const batchEmbeddings = embedder.embedBatch(texts);
  const totalTime = performance.now() - start;

  const dimension = embedder.dimension();
  const results: EmbeddingResult[] = [];

  for (let i = 0; i < texts.length; i++) {
    const embedding = batchEmbeddings.slice(i * dimension, (i + 1) * dimension);
    results.push({
      embedding: Array.from(embedding),
      dimension,
      timeMs: totalTime / texts.length,
    });
  }

  return results;
}

/**
 * Calculate cosine similarity between two texts
 */
export async function similarity(text1: string, text2: string): Promise<SimilarityResult> {
  if (!isInitialized) {
    await initOnnxEmbedder();
  }
  if (!embedder) {
    throw new Error('ONNX embedder not initialized');
  }

  const start = performance.now();
  const sim = embedder.similarity(text1, text2);
  const timeMs = performance.now() - start;

  return { similarity: sim, timeMs };
}

/**
 * Calculate cosine similarity between two embeddings
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Embeddings must have same dimension');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
  return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Get embedding dimension
 */
export function getDimension(): number {
  return embedder ? embedder.dimension() : 384;
}

/**
 * Check if the embedder has been initialized (model loaded) and is ready to
 * embed. Returns `false` until `initOnnxEmbedder()` (or the first `embed()`,
 * which auto-initializes) has completed successfully.
 */
export function isReady(): boolean {
  return isInitialized;
}

/**
 * Whether the ONNX embedder has been initialized (model loaded).
 *
 * Post-init counterpart to `isOnnxAvailable()` (which only checks that the
 * bundled files exist). Named distinctly from the WASM-core `isInitialized()`
 * export to avoid a barrel name collision. Equivalent to `isReady()`; provided
 * as a self-documenting gate so callers can distinguish "bundled" (available)
 * from "loaded" (initialized). See
 * https://github.com/ruvnet/RuVector/issues/523.
 */
export function isOnnxInitialized(): boolean {
  return isInitialized;
}

/**
 * Get embedder stats including SIMD and parallel capabilities
 */
export function getStats(): {
  ready: boolean;
  dimension: number;
  model: string;
  simd: boolean;
  parallel: boolean;
  parallelWorkers: number;
  parallelThreshold: number;
} {
  return {
    ready: isInitialized,
    dimension: embedder ? embedder.dimension() : 384,
    model: DEFAULT_MODEL,
    simd: simdAvailable,
    parallel: parallelEnabled,
    parallelWorkers: parallelEmbedder?.numWorkers || 0,
    parallelThreshold,
  };
}

/**
 * Shutdown parallel workers (call on exit)
 */
export async function shutdown(): Promise<void> {
  if (parallelEmbedder) {
    await parallelEmbedder.shutdown();
    parallelEmbedder = null;
    parallelEnabled = false;
  }
  await shutdownParallelEmbedder();
}

/**
 * Initialize the bundled-WASM worker pool for high-throughput batch embedding
 * (issue #523 SOTA). Self-contained — uses Node worker_threads + the bundled
 * WASM over SharedArrayBuffer model bytes, no external dependency. Vectors are
 * identical to the single-thread path (cosine-equivalent).
 *
 * @param numWorkers number of worker threads (default: min(cpus-2, 16))
 */
export async function initParallelEmbedder(numWorkers?: number): Promise<boolean> {
  if (bundledPool) return true;
  if (!isInitialized) await initOnnxEmbedder();
  if (!loadedModelBytes || !loadedTokenizerJson) {
    throw new Error('Model bytes unavailable; cannot start parallel embedder.');
  }
  const poolUrl = pathToFileURL(path.join(__dirname, 'onnx', 'bundled-parallel.mjs')).href;
  const { ParallelEmbedder } = await dynamicImport(poolUrl);
  const pool = new ParallelEmbedder({
    modelBytes: loadedModelBytes,
    tokenizerJson: loadedTokenizerJson,
    maxLength: loadedMaxLength,
    dimension: getDimension(),
    numWorkers,
  });
  await pool.init();
  bundledPool = pool;
  return true;
}

/**
 * Batch-embed via the bundled worker pool, sharded across CPU cores. Lazily
 * starts the pool on first use. Returns embeddings in input order.
 */
export async function embedBatchParallel(texts: string[]): Promise<number[][]> {
  if (!bundledPool) await initParallelEmbedder();
  return bundledPool.embedBatch(texts);
}

/** Number of active pool workers (0 if the pool isn't started). */
export function getParallelWorkerCount(): number {
  return bundledPool ? bundledPool.numWorkers : 0;
}

/** Shut down the bundled worker pool and release its threads. */
export async function shutdownParallelEmbedder(): Promise<void> {
  if (bundledPool) {
    await bundledPool.shutdown();
    bundledPool = null;
  }
}

// Export class wrapper for compatibility
export class OnnxEmbedder {
  private config: OnnxEmbedderConfig;

  constructor(config: OnnxEmbedderConfig = {}) {
    this.config = config;
  }

  async init(): Promise<boolean> {
    return initOnnxEmbedder(this.config);
  }

  async embed(text: string): Promise<number[]> {
    const result = await embed(text);
    return result.embedding;
  }

  async embedBatch(texts: string[]): Promise<number[][]> {
    const results = await embedBatch(texts);
    return results.map(r => r.embedding);
  }

  async similarity(text1: string, text2: string): Promise<number> {
    const result = await similarity(text1, text2);
    return result.similarity;
  }

  get dimension(): number {
    return getDimension();
  }

  get ready(): boolean {
    return isReady();
  }
}

export default OnnxEmbedder;
