/**
 * @ruvector/rvf-wasm — JS glue for the RVF WASM microkernel.
 *
 * Loads the .wasm binary and re-exports all C-ABI functions plus the
 * WASM linear memory object.
 *
 * This package is ESM-only ("type": "module"). Use it via:
 *   import init from '@ruvector/rvf-wasm';
 * or from CommonJS via dynamic import:
 *   const { default: init } = await import('@ruvector/rvf-wasm');
 */

let wasmInstance = null;

const _isNode = typeof process !== 'undefined' &&
  typeof process.versions !== 'undefined' &&
  typeof process.versions.node === 'string';

/**
 * Initialize the WASM module.
 * Returns the exports object with all rvf_* functions and `memory`.
 *
 * @param {ArrayBuffer|BufferSource|WebAssembly.Module|string} [input]
 *   Optional pre-loaded bytes, Module, or file path override.
 */
async function init(input) {
  if (wasmInstance) return wasmInstance;

  let wasmBytes;

  if (input instanceof ArrayBuffer || ArrayBuffer.isView(input)) {
    wasmBytes = input;
  } else if (typeof WebAssembly !== 'undefined' && input instanceof WebAssembly.Module) {
    const inst = await WebAssembly.instantiate(input, {});
    wasmInstance = inst.exports;
    return wasmInstance;
  } else if (_isNode) {
    // Node.js: always use readFile (fetch on file:// is unreliable)
    const fs = await import('node:fs/promises');
    const url = await import('node:url');
    const path = await import('node:path');
    let wasmPath;
    if (typeof input === 'string') {
      wasmPath = input;
    } else {
      // ESM context — import.meta.url is available
      const thisDir = path.default.dirname(url.default.fileURLToPath(import.meta.url));
      wasmPath = path.default.join(thisDir, 'rvf_wasm_bg.wasm');
    }
    wasmBytes = await fs.default.readFile(wasmPath);
  } else {
    // Browser: use fetch + instantiateStreaming
    const wasmUrl = new URL('rvf_wasm_bg.wasm', import.meta.url);
    if (typeof WebAssembly.instantiateStreaming === 'function') {
      const resp = await fetch(wasmUrl);
      const result = await WebAssembly.instantiateStreaming(resp, {});
      wasmInstance = result.instance.exports;
      return wasmInstance;
    }
    const resp2 = await fetch(wasmUrl);
    wasmBytes = await resp2.arrayBuffer();
  }

  const compiled = await WebAssembly.instantiate(wasmBytes, {});
  wasmInstance = compiled.instance.exports;
  return wasmInstance;
}

// Back-compat: `mod.default` resolves to the init function whether the caller
// reads the ESM default export or destructures it off the namespace object.
init.default = init;

export default init;
export { init };
