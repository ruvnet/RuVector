/**
 * @ruvector/rvf-wasm ESM entry point.
 *
 * Re-exports the init function. The package is ESM-only ("type": "module"),
 * so both rvf_wasm.js and rvf_wasm.mjs are ES modules.
 */
import init from './rvf_wasm.js';
export default init;
export { init };
