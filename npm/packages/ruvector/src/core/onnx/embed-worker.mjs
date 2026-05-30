/**
 * Worker-thread entry for the bundled-WASM parallel embedder (issue #523 SOTA).
 *
 * Each worker loads its own instance of the bundled ONNX WASM embedder from the
 * SAME model bytes (shared via SharedArrayBuffer — no per-worker download) and
 * the SAME config, so the vectors it produces are identical to the single-thread
 * path (cosine-equivalent by construction).
 *
 * Protocol:
 *   workerData: { modelSab: SharedArrayBuffer, tokenizerJson: string, maxLength: number }
 *   → posts { type: 'ready' } once the WASM embedder is constructed
 *   message { type: 'embed', id, texts: string[] }
 *   → posts { type: 'result', id, dim, count, buffer } (Float32Array buffer, transferred)
 *   errors → { type: 'error', id, error }
 */
import { parentPort, workerData } from 'node:worker_threads';
import { pathToFileURL, fileURLToPath } from 'node:url';
import * as path from 'node:path';
import * as fs from 'node:fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let embedder = null;

async function init() {
  const bgJsPath = path.join(__dirname, 'pkg', 'ruvector_onnx_embeddings_wasm_bg.js');
  const wasmPath = path.join(__dirname, 'pkg', 'ruvector_onnx_embeddings_wasm_bg.wasm');

  const wasmModule = await import(pathToFileURL(bgJsPath).href);
  const wasmBytes = fs.readFileSync(wasmPath);
  const wasmResult = await WebAssembly.instantiate(wasmBytes, {
    './ruvector_onnx_embeddings_wasm_bg.js': wasmModule,
  });
  const wasmExports = wasmResult.instance.exports;
  if (typeof wasmModule.__wbg_set_wasm === 'function') wasmModule.__wbg_set_wasm(wasmExports);
  if (typeof wasmExports.__wbindgen_start === 'function') wasmExports.__wbindgen_start();

  // Reconstruct model bytes from the shared buffer (zero-copy view, then handed
  // to wasm-bindgen which copies into WASM linear memory).
  const modelBytes = new Uint8Array(workerData.modelSab);

  const cfg = new wasmModule.WasmEmbedderConfig()
    .setMaxLength(workerData.maxLength || 256)
    .setNormalize(true)
    .setPooling(0); // Mean pooling — matches the single-thread path.

  embedder = wasmModule.WasmEmbedder.withConfig(modelBytes, workerData.tokenizerJson, cfg);
}

parentPort.on('message', (msg) => {
  if (msg.type !== 'embed') return;
  try {
    const dim = embedder.dimension();
    const flat = embedder.embedBatch(msg.texts); // length = texts.length * dim
    const arr = Float32Array.from(flat);
    parentPort.postMessage(
      { type: 'result', id: msg.id, dim, count: msg.texts.length, buffer: arr.buffer },
      [arr.buffer],
    );
  } catch (e) {
    parentPort.postMessage({ type: 'error', id: msg.id, error: e?.message || String(e) });
  }
});

init()
  .then(() => parentPort.postMessage({ type: 'ready' }))
  .catch((e) => parentPort.postMessage({ type: 'init-error', error: e?.message || String(e) }));
