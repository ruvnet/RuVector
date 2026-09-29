/**
 * @ruvector/ruvllm-wasm - browser LLM runtime (in progress)
 *
 * Status: capability checks (WebGPU, SIMD, SharedArrayBuffer) and helpers work.
 * Model loading and text generation are NOT implemented yet: `loadModel`,
 * `generate` and `chat` throw `RuvLLMWasmNotImplementedError` (see
 * `INFERENCE_AVAILABLE`) rather than returning placeholder text. The example
 * below shows the intended API.
 *
 * @example
 * ```typescript
 * import { RuvLLMWasm } from '@ruvector/ruvllm-wasm';
 *
 * // Initialize with WebGPU (if available)
 * const llm = await RuvLLMWasm.create({ useWebGPU: true });
 *
 * // Load a model
 * await llm.loadModel('https://example.com/model.gguf', {
 *   onProgress: (loaded, total) => console.log(`${loaded}/${total}`)
 * });
 *
 * // Generate text
 * const result = await llm.generate('Hello, world!', {
 *   maxTokens: 100,
 *   temperature: 0.7,
 * });
 *
 * console.log(result.text);
 * ```
 *
 * @packageDocumentation
 */

export {
  WebGPUStatus,
  LoadingStatus,
  ModelArchitecture,
  ModelMetadata,
  WASMConfig,
  GenerationConfig,
  TokenCallback,
  ProgressCallback,
  InferenceStats,
  ChatMessage,
  CompletionResult,
  DownloadProgress,
} from './types.js';

/** Package version */
export const VERSION = '0.1.0';

/**
 * Check WebGPU availability
 */
export async function checkWebGPU(): Promise<import('./types.js').WebGPUStatus> {
  if (typeof navigator === 'undefined') {
    return 'not_supported' as import('./types.js').WebGPUStatus;
  }

  if (!('gpu' in navigator)) {
    return 'not_supported' as import('./types.js').WebGPUStatus;
  }

  try {
    const adapter = await (navigator as any).gpu.requestAdapter();
    if (adapter) {
      return 'available' as import('./types.js').WebGPUStatus;
    }
    return 'unavailable' as import('./types.js').WebGPUStatus;
  } catch {
    return 'unavailable' as import('./types.js').WebGPUStatus;
  }
}

/**
 * Check SharedArrayBuffer support (required for threading)
 */
export function checkSharedArrayBuffer(): boolean {
  return typeof SharedArrayBuffer !== 'undefined';
}

/**
 * Check SIMD support
 */
export async function checkSIMD(): Promise<boolean> {
  try {
    // Check for WASM SIMD support
    const simdTest = new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
      0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
      0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
      0x41, 0x00, 0xfd, 0x0f, 0x00, 0x0b,
    ]);
    await WebAssembly.compile(simdTest);
    return true;
  } catch {
    return false;
  }
}

/**
 * Get browser capabilities for LLM inference
 */
export async function getCapabilities(): Promise<{
  webgpu: import('./types.js').WebGPUStatus;
  sharedArrayBuffer: boolean;
  simd: boolean;
  crossOriginIsolated: boolean;
}> {
  const [webgpu, simd] = await Promise.all([
    checkWebGPU(),
    checkSIMD(),
  ]);

  return {
    webgpu,
    sharedArrayBuffer: checkSharedArrayBuffer(),
    simd,
    crossOriginIsolated: typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
  };
}

/**
 * Format file size for display
 */
export function formatFileSize(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;

  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }

  return `${size.toFixed(1)} ${units[unitIndex]}`;
}

/**
 * Estimate memory requirements for a model
 */
export function estimateMemory(fileSizeBytes: number): {
  minimum: number;
  recommended: number;
} {
  // Rough estimates based on model size
  const fileSizeMB = fileSizeBytes / (1024 * 1024);

  return {
    minimum: Math.ceil(fileSizeMB * 1.2), // 20% overhead
    recommended: Math.ceil(fileSizeMB * 1.5), // 50% overhead for KV cache
  };
}

/**
 * Whether this package can load models and generate text. `false`: model
 * loading and text generation are not implemented in the WASM build yet.
 * `loadModel`, `generate` and `chat` throw {@link RuvLLMWasmNotImplementedError}
 * instead of returning placeholder output that could be mistaken for a model's.
 */
export const INFERENCE_AVAILABLE = false;

/** Thrown by `loadModel`, `generate` and `chat` while inference is unavailable. */
export class RuvLLMWasmNotImplementedError extends Error {
  readonly code = 'RUVLLM_WASM_NOT_IMPLEMENTED';

  constructor(method: string) {
    super(
      `\`${method}\` is not implemented: @ruvector/ruvllm-wasm cannot load models or ` +
        'generate text yet. For GGUF inference use the Rust ruvllm crate (candle feature); the WASM crate ' +
        '(crates/ruvllm-wasm) currently ships kernels, KV cache, chat templates, ' +
        'MicroLoRA, SONA and the HNSW router.'
    );
    this.name = 'RuvLLMWasmNotImplementedError';
    Object.setPrototypeOf(this, RuvLLMWasmNotImplementedError.prototype);
  }
}

/**
 * Browser LLM runtime facade. Capability checks and configuration work;
 * model loading and generation fail closed until the WASM runtime lands
 * (see {@link INFERENCE_AVAILABLE}).
 */
export class RuvLLMWasm {
  private config: import('./types.js').WASMConfig;
  private status: import('./types.js').LoadingStatus = 'idle' as import('./types.js').LoadingStatus;

  private constructor(config: import('./types.js').WASMConfig) {
    this.config = config;
  }

  /**
   * Create a new RuvLLMWasm instance
   */
  static async create(options?: {
    useWebGPU?: boolean;
    threads?: number;
    memoryLimit?: number;
  }): Promise<RuvLLMWasm> {
    const config: import('./types.js').WASMConfig = {
      threads: options?.threads,
      memoryLimit: options?.memoryLimit,
      simd: await checkSIMD(),
      cacheModels: true,
    };

    if (options?.useWebGPU) {
      const webgpuStatus = await checkWebGPU();
      if (webgpuStatus === 'available') {
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter) {
          config.device = await adapter.requestDevice();
        }
      }
    }

    return new RuvLLMWasm(config);
  }

  /**
   * Get current loading status
   */
  getStatus(): import('./types.js').LoadingStatus {
    return this.status;
  }

  /**
   * Load a model from URL or ArrayBuffer
   */
  async loadModel(
    source: string | ArrayBuffer,
    options?: {
      onProgress?: import('./types.js').ProgressCallback;
    }
  ): Promise<import('./types.js').ModelMetadata> {
    void source;
    void options;
    this.status = 'error' as import('./types.js').LoadingStatus;
    throw new RuvLLMWasmNotImplementedError('loadModel');
  }

  /**
   * Generate text completion
   */
  async generate(
    prompt: string,
    config?: import('./types.js').GenerationConfig,
    onToken?: import('./types.js').TokenCallback
  ): Promise<import('./types.js').CompletionResult> {
    void prompt;
    void config;
    void onToken;
    throw new RuvLLMWasmNotImplementedError('generate');
  }

  /**
   * Chat completion with message history
   */
  async chat(
    messages: import('./types.js').ChatMessage[],
    config?: import('./types.js').GenerationConfig,
    onToken?: import('./types.js').TokenCallback
  ): Promise<import('./types.js').CompletionResult> {
    void messages;
    void config;
    void onToken;
    throw new RuvLLMWasmNotImplementedError('chat');
  }

  /**
   * Unload model and free memory
   */
  unload(): void {
    this.status = 'idle' as import('./types.js').LoadingStatus;
  }
}
