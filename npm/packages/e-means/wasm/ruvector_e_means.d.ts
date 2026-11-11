/* tslint:disable */
/* eslint-disable */

/**
 * A temporary owner for one fit result. JavaScript takes independent output
 * arrays and then frees this handle, releasing any vectors not taken.
 */
export class FittedModel {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Cluster indices fit u32 because wasm32 arrays and sample counts do.
     */
    takeAssignments(): Uint32Array;
    /**
     * wasm-bindgen copies the returned vector into a caller-owned Float32Array
     * before freeing its Rust allocation. No view into WASM memory escapes.
     */
    takeCentroids(): Float32Array;
    readonly batchSize: number;
    readonly dimensions: number;
    readonly mse: number;
    readonly peakStep: number;
    readonly revivedClusters: number;
    readonly updates: number;
}

/**
 * Compatibility transport and an independent parity oracle for typed output.
 */
export function fitJson(data: Float32Array, dimensions: number, config_json: string): string;

/**
 * Only the small configuration crosses JSON; result vectors stay numeric.
 */
export function fitTyped(data: Float32Array, dimensions: number, config_json: string): FittedModel;

export function predictFlat(data: Float32Array, dimensions: number, centroids: Float32Array): Uint32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_fittedmodel_free: (a: number, b: number) => void;
    readonly fitJson: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly fitTyped: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly fittedmodel_batchSize: (a: number) => number;
    readonly fittedmodel_dimensions: (a: number) => number;
    readonly fittedmodel_mse: (a: number) => number;
    readonly fittedmodel_peakStep: (a: number) => number;
    readonly fittedmodel_revivedClusters: (a: number) => number;
    readonly fittedmodel_takeAssignments: (a: number, b: number) => void;
    readonly fittedmodel_takeCentroids: (a: number, b: number) => void;
    readonly fittedmodel_updates: (a: number) => number;
    readonly predictFlat: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
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
