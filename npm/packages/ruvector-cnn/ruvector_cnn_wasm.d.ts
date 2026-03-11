/* tslint:disable */
/* eslint-disable */

/**
 * Configuration for CNN embedder
 */
export class EmbedderConfig {
    free(): void;
    [Symbol.dispose](): void;
    constructor();
    /**
     * Output embedding dimension
     */
    embedding_dim: number;
    /**
     * Input image size (square)
     */
    input_size: number;
    /**
     * Whether to L2 normalize embeddings
     */
    normalize: boolean;
}

/**
 * Layer operations for building custom networks
 */
export class LayerOps {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Apply batch normalization (returns new array)
     */
    static batch_norm(input: Float32Array, gamma: Float32Array, beta: Float32Array, mean: Float32Array, _var: Float32Array, epsilon: number): Float32Array;
    /**
     * Apply global average pooling
     * Returns one value per channel
     */
    static global_avg_pool(input: Float32Array, height: number, width: number, channels: number): Float32Array;
}

/**
 * SIMD-optimized operations
 */
export class SimdOps {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Dot product of two vectors
     */
    static dot_product(a: Float32Array, b: Float32Array): number;
    /**
     * L2 normalize a vector (returns new array)
     */
    static l2_normalize(data: Float32Array): Float32Array;
    /**
     * ReLU activation (returns new array)
     */
    static relu(data: Float32Array): Float32Array;
    /**
     * ReLU6 activation (returns new array)
     */
    static relu6(data: Float32Array): Float32Array;
}

/**
 * WASM CNN Embedder for image feature extraction
 */
export class WasmCnnEmbedder {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Compute cosine similarity between two embeddings
     */
    cosine_similarity(a: Float32Array, b: Float32Array): number;
    /**
     * Extract embedding from image data (RGB format, row-major)
     */
    extract(image_data: Uint8Array, width: number, height: number): Float32Array;
    /**
     * Create a new CNN embedder
     */
    constructor(config?: EmbedderConfig | null);
    /**
     * Get the embedding dimension
     */
    readonly embedding_dim: number;
}

/**
 * InfoNCE loss for contrastive learning (SimCLR style)
 */
export class WasmInfoNCELoss {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Compute loss for a batch of embedding pairs
     * embeddings: [2N, D] flattened where (i, i+N) are positive pairs
     */
    forward(embeddings: Float32Array, batch_size: number, dim: number): number;
    /**
     * Create new InfoNCE loss with temperature parameter
     */
    constructor(temperature: number);
    /**
     * Get the temperature parameter
     */
    readonly temperature: number;
}

/**
 * Triplet loss for metric learning
 */
export class WasmTripletLoss {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Compute loss for a batch of triplets
     */
    forward(anchors: Float32Array, positives: Float32Array, negatives: Float32Array, dim: number): number;
    /**
     * Compute loss for a single triplet
     */
    forward_single(anchor: Float32Array, positive: Float32Array, negative: Float32Array): number;
    /**
     * Create new triplet loss with margin
     */
    constructor(margin: number);
    /**
     * Get the margin parameter
     */
    readonly margin: number;
}

/**
 * Initialize panic hook for better error messages
 */
export function init(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_embedderconfig_free: (a: number, b: number) => void;
    readonly __wbg_get_embedderconfig_embedding_dim: (a: number) => number;
    readonly __wbg_get_embedderconfig_input_size: (a: number) => number;
    readonly __wbg_get_embedderconfig_normalize: (a: number) => number;
    readonly __wbg_layerops_free: (a: number, b: number) => void;
    readonly __wbg_set_embedderconfig_embedding_dim: (a: number, b: number) => void;
    readonly __wbg_set_embedderconfig_input_size: (a: number, b: number) => void;
    readonly __wbg_set_embedderconfig_normalize: (a: number, b: number) => void;
    readonly __wbg_wasmcnnembedder_free: (a: number, b: number) => void;
    readonly __wbg_wasminfonceloss_free: (a: number, b: number) => void;
    readonly __wbg_wasmtripletloss_free: (a: number, b: number) => void;
    readonly embedderconfig_new: () => number;
    readonly layerops_batch_norm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => void;
    readonly layerops_global_avg_pool: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly simdops_dot_product: (a: number, b: number, c: number, d: number) => number;
    readonly simdops_l2_normalize: (a: number, b: number, c: number) => void;
    readonly simdops_relu: (a: number, b: number, c: number) => void;
    readonly simdops_relu6: (a: number, b: number, c: number) => void;
    readonly wasmcnnembedder_cosine_similarity: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly wasmcnnembedder_embedding_dim: (a: number) => number;
    readonly wasmcnnembedder_extract: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly wasmcnnembedder_new: (a: number, b: number) => void;
    readonly wasminfonceloss_forward: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly wasminfonceloss_new: (a: number) => number;
    readonly wasminfonceloss_temperature: (a: number) => number;
    readonly wasmtripletloss_forward: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly wasmtripletloss_forward_single: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly wasmtripletloss_margin: (a: number) => number;
    readonly wasmtripletloss_new: (a: number) => number;
    readonly init: () => void;
    readonly __wbg_simdops_free: (a: number, b: number) => void;
    readonly __wbindgen_export: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export2: (a: number, b: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
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
