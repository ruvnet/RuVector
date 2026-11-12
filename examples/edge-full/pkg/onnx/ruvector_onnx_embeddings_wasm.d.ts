/* tslint:disable */
/* eslint-disable */

/**
 * Strategy for pooling token embeddings into a single sentence embedding
 */
export enum PoolingStrategy {
    /**
     * Average all token embeddings (most common)
     */
    Mean = 0,
    /**
     * Use only the [CLS] token embedding
     */
    Cls = 1,
    /**
     * Take the maximum value across all tokens for each dimension
     */
    Max = 2,
    /**
     * Mean pooling normalized by sqrt of sequence length
     */
    MeanSqrtLen = 3,
    /**
     * Use the last token embedding (for decoder models)
     */
    LastToken = 4,
}

/**
 * WASM-compatible embedder using Tract for inference
 */
export class WasmEmbedder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Get the embedding dimension
     */
    dimension(): number;
    /**
     * Generate embeddings for multiple texts
     */
    embedBatch(texts: string[]): Float32Array;
    /**
     * Generate embedding for a single text
     */
    embedOne(text: string): Float32Array;
    /**
     * Get maximum sequence length
     */
    maxLength(): number;
    /**
     * Create a new embedder from model and tokenizer bytes
     *
     * # Arguments
     * * `model_bytes` - ONNX model file bytes
     * * `tokenizer_json` - Tokenizer JSON configuration
     */
    constructor(model_bytes: Uint8Array, tokenizer_json: string);
    /**
     * Compute similarity between two texts
     */
    similarity(text1: string, text2: string): number;
    /**
     * Create embedder with custom configuration
     */
    static withConfig(model_bytes: Uint8Array, tokenizer_json: string, config: WasmEmbedderConfig): WasmEmbedder;
}

/**
 * Configuration for the WASM embedder
 */
export class WasmEmbedderConfig {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Create a new configuration
     */
    constructor();
    /**
     * Set maximum sequence length
     */
    setMaxLength(max_length: number): WasmEmbedderConfig;
    /**
     * Set whether to normalize embeddings
     */
    setNormalize(normalize: boolean): WasmEmbedderConfig;
    /**
     * Set pooling strategy (0=Mean, 1=Cls, 2=Max, 3=MeanSqrtLen, 4=LastToken)
     */
    setPooling(pooling: number): WasmEmbedderConfig;
}

/**
 * Compute cosine similarity between two embedding vectors (JS-friendly)
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number;

/**
 * Initialize panic hook for better error messages in WASM
 */
export function init(): void;

/**
 * L2 normalize an embedding vector (JS-friendly)
 */
export function normalizeL2(embedding: Float32Array): Float32Array;

/**
 * Check if SIMD is available (for performance info)
 * Returns true if compiled with WASM SIMD128 support
 */
export function simd_available(): boolean;

/**
 * Get the library version
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly init: () => void;
    readonly simd_available: () => number;
    readonly version: () => [number, number];
    readonly __wbg_wasmembedder_free: (a: number, b: number) => void;
    readonly __wbg_wasmembedderconfig_free: (a: number, b: number) => void;
    readonly cosineSimilarity: (a: number, b: number, c: number, d: number) => number;
    readonly normalizeL2: (a: number, b: number) => [number, number];
    readonly wasmembedder_dimension: (a: number) => number;
    readonly wasmembedder_embedBatch: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmembedder_embedOne: (a: number, b: number, c: number) => [number, number, number, number];
    readonly wasmembedder_maxLength: (a: number) => number;
    readonly wasmembedder_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly wasmembedder_similarity: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmembedder_withConfig: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmembedderconfig_new: () => number;
    readonly wasmembedderconfig_setMaxLength: (a: number, b: number) => number;
    readonly wasmembedderconfig_setNormalize: (a: number, b: number) => number;
    readonly wasmembedderconfig_setPooling: (a: number, b: number) => number;
    readonly __wbindgen_malloc_command_export: (a: number, b: number) => number;
    readonly __wbindgen_realloc_command_export: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store_command_export: (a: number) => void;
    readonly __externref_table_alloc_command_export: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free_command_export: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc_command_export: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
