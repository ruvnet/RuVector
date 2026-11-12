/**
 * Native bindings loader for RuvLLM
 *
 * Automatically loads the correct native binary for the current platform.
 */

import { join, resolve } from 'path';

// Try to load the native module
let nativeModule: NativeRuvLLM | null = null;

interface NativeRuvLLM {
  // Native exports RuvLlmEngine (camelCase), we normalize to RuvLLMEngine
  RuvLLMEngine: new (config?: NativeConfig) => NativeEngine;
  SimdOperations: new () => NativeSimdOps;
  version: () => string;
  hasSimdSupport: () => boolean;
}

// Raw native module interface (actual export names)
interface RawNativeModule {
  RuvLlmEngine?: new (config?: NativeConfig) => NativeEngine;
  RuvLLMEngine?: new (config?: NativeConfig) => NativeEngine;
  SimdOperations: new () => NativeSimdOps;
  version: () => string;
  hasSimdSupport: () => boolean;
}

interface NativeConfig {
  embedding_dim?: number;
  router_hidden_dim?: number;
  hnsw_m?: number;
  hnsw_ef_construction?: number;
  hnsw_ef_search?: number;
  learning_enabled?: boolean;
  quality_threshold?: number;
  ewc_lambda?: number;
}

interface NativeEngine {
  query(text: string, config?: NativeGenConfig): NativeQueryResponse;
  /**
   * Builds with `loadModel` return a generation result and throw while no
   * model is loaded; older builds (the 2.0.x platform packages) return a
   * string that is not model output, so the wrapper never calls them.
   */
  generate(prompt: string, config?: NativeGenConfig): NativeGenerationResult | string;
  /**
   * Present on builds since model loading was added. Those built without the
   * `candle` feature throw `RUVLLM_NO_INFERENCE_BACKEND`; see
   * `supportsModelLoading`.
   */
  loadModel?(path: string, options?: NativeLoadModelOptions): NativeModelInfo;
  /** Whether the binary was built with the `candle` feature (absent on older builds). */
  supportsModelLoading?(): boolean;
  unloadModel?(): void;
  isModelLoaded?(): boolean;
  modelInfo?(): NativeModelInfo | null;
  chat?(messages: NativeChatMessage[], config?: NativeGenConfig): NativeGenerationResult;
  route(text: string): NativeRoutingDecision;
  searchMemory(text: string, k?: number): NativeMemoryResult[];
  /** Current native builds return a UUID string; older ones returned a number. */
  addMemory(content: string, metadata?: string): string | number;
  feedback(requestId: string, rating: number, correction?: string): boolean;
  stats(): NativeStats;
  forceLearn(): string;
  embed(text: string): number[];
  similarity(text1: string, text2: string): number;
  hasSimd(): boolean;
  simdCapabilities(): string[];
}

// napi-rs reads `#[napi(object)]` fields by their camelCase names; snake_case
// keys are silently ignored (which is how `maxTokens` used to be dropped).
interface NativeGenConfig {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
  seed?: number;
}

interface NativeLoadModelOptions {
  maxContext?: number;
}

interface NativeChatMessage {
  role: string;
  content: string;
}

interface NativeGenerationResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
  finishReason: string;
}

interface NativeModelInfo {
  name: string;
  architecture: string;
  numParameters: number;
  vocabSize: number;
  hiddenSize: number;
  numLayers: number;
  maxContextLength: number;
  quantization?: string | null;
  memoryUsageBytes: number;
  chatTemplate?: string | null;
}

// napi-rs camelCases Rust struct fields when it builds the JS object, so the
// real native shape is camelCase. These interfaces previously declared only
// snake_case, which is why the wrapper's snake_case reads typechecked while
// silently yielding undefined at runtime. Both spellings are declared —
// camelCase required, snake_case optional — so older native builds still fit.
interface NativeQueryResponse {
  text: string;
  confidence: number;
  model: string;
  contextSize: number;
  latencyMs: number;
  requestId: string;
  /** Builds with `loadModel` also report token usage. */
  promptTokens?: number;
  completionTokens?: number;
  finishReason?: string;
  context_size?: number;
  latency_ms?: number;
  request_id?: string;
}

interface NativeRoutingDecision {
  model: string;
  contextSize: number;
  temperature: number;
  top_p?: number;
  topP: number;
  confidence: number;
  context_size?: number;
}

interface NativeMemoryResult {
  id: string | number;
  /** Distance (smaller is closer) as reported by the current native build. */
  distance?: number;
  /** Similarity, emitted by older native builds. */
  score?: number;
  content: string;
  metadata: string;
}

interface NativeStats {
  total_queries: number;
  memory_nodes: number;
  patterns_learned: number;
  avg_latency_ms: number;
  cache_hit_rate: number;
  router_accuracy: number;
}

interface NativeSimdOps {
  dotProduct(a: number[], b: number[]): number;
  cosineSimilarity(a: number[], b: number[]): number;
  l2Distance(a: number[], b: number[]): number;
  matvec(matrix: number[][], vector: number[]): number[];
  softmax(input: number[]): number[];
}

// Platform-specific package names
const PLATFORM_PACKAGES: Record<string, string> = {
  'darwin-x64': '@ruvector/ruvllm-darwin-x64',
  'darwin-arm64': '@ruvector/ruvllm-darwin-arm64',
  'linux-x64': '@ruvector/ruvllm-linux-x64-gnu',
  'linux-arm64': '@ruvector/ruvllm-linux-arm64-gnu',
  'win32-x64': '@ruvector/ruvllm-win32-x64-msvc',
};

function getPlatformKey(): string {
  const platform = process.platform;
  const arch = process.arch;
  return `${platform}-${arch}`;
}

function loadNativeModule(): NativeRuvLLM | null {
  if (nativeModule) {
    return nativeModule;
  }

  // An explicit binary (e.g. one built locally with `npm run build:native`)
  // wins over the installed platform package. Failing to load it is an
  // error, not a silent fallback to a different binary.
  const override = process.env.RUVLLM_NATIVE_PATH;
  if (override) {
    try {
      // resolve(): a relative path means relative to the working directory,
      // not to this file.
      nativeModule = normalize(require(resolve(override)) as RawNativeModule);
      return nativeModule;
    } catch (e) {
      throw new Error(
        `RUVLLM_NATIVE_PATH=${override} could not be loaded: ${(e as Error).message}`,
      );
    }
  }

  const platformKey = getPlatformKey();
  const packageName = PLATFORM_PACKAGES[platformKey];

  if (!packageName) {
    // Silently fail - JS fallback will be used
    return null;
  }
  // `napi build --platform` names its output ruvllm.<platform>.node
  const platformFile = `ruvllm.${packageName.replace('@ruvector/ruvllm-', '')}.node`;

  // Try loading from optional dependencies
  const attempts = [
    // Try the platform-specific package
    () => require(packageName),
    // Try loading from local .node file (CJS build)
    () => require(join(__dirname, '..', '..', 'ruvllm.node')),
    // Try loading from local .node file (root)
    () => require(join(__dirname, '..', 'ruvllm.node')),
    // Output of `npm run build:native` in the package root
    () => require(join(__dirname, '..', '..', platformFile)),
  ];

  for (const attempt of attempts) {
    try {
      nativeModule = normalize(attempt() as RawNativeModule);
      return nativeModule;
    } catch {
      // Continue to next attempt
    }
  }

  // Silently fall back to JS implementation
  return null;
}

// Normalize: native exports RuvLlmEngine, we expose as RuvLLMEngine
function normalize(raw: RawNativeModule): NativeRuvLLM {
  return {
    RuvLLMEngine: raw.RuvLLMEngine ?? raw.RuvLlmEngine!,
    SimdOperations: raw.SimdOperations,
    version: raw.version,
    hasSimdSupport: raw.hasSimdSupport,
  };
}

// Export functions to get native bindings
export function getNativeModule(): NativeRuvLLM | null {
  return loadNativeModule();
}

export function version(): string {
  const mod = loadNativeModule();
  return mod?.version() ?? '0.1.0-js';
}

export function hasSimdSupport(): boolean {
  const mod = loadNativeModule();
  return mod?.hasSimdSupport() ?? false;
}

// Export types for internal use
export type {
  NativeRuvLLM,
  NativeConfig,
  NativeEngine,
  NativeGenConfig,
  NativeLoadModelOptions,
  NativeChatMessage,
  NativeGenerationResult,
  NativeModelInfo,
  NativeQueryResponse,
  NativeRoutingDecision,
  NativeMemoryResult,
  NativeStats,
  NativeSimdOps,
};
