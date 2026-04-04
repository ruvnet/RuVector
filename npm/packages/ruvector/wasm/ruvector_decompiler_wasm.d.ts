/* tslint:disable */
/* eslint-disable */

/**
 * Decompile a minified JavaScript bundle using the full Louvain pipeline.
 *
 * # Arguments
 *
 * * `source` - The minified JavaScript source code.
 * * `config_json` - JSON string of `DecompileConfig` fields. Pass `"{}"` for defaults.
 *
 * # Returns
 *
 * A JSON string containing the `DecompileResult` (modules, witness, inferred names, etc.)
 * or a JSON object with an `"error"` field on failure.
 */
export function decompile(source: string, config_json: string): string;

/**
 * Initialize the WASM module (sets up panic hook for better error messages).
 */
export function init(): void;

/**
 * Return the version of the decompiler WASM module.
 */
export function version(): string;
