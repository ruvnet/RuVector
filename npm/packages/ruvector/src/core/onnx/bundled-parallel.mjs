/**
 * Bundled-WASM parallel embedder (issue #523 SOTA).
 *
 * A self-contained worker_threads pool — NO external dependency — that shards
 * batches of text across CPU cores, each worker running the bundled ONNX WASM
 * embedder over the SAME model bytes (shared via SharedArrayBuffer) and config.
 * Output vectors are identical to the single-thread path (cosine-equivalent),
 * so this is a pure throughput optimization with no quality change.
 *
 * Drop-in shape compatible with the optional `ruvector-onnx-embeddings-wasm/parallel`
 * package: { numWorkers, dimension, init(), embedBatch(texts) -> number[][], shutdown() }.
 */
import { Worker } from 'node:worker_threads';
import * as os from 'node:os';
import { fileURLToPath } from 'node:url';
import * as path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export class ParallelEmbedder {
  /**
   * @param {object} opts
   * @param {Uint8Array} opts.modelBytes  raw ONNX model bytes (loaded once by caller)
   * @param {string}     opts.tokenizerJson
   * @param {number}     [opts.maxLength=256]
   * @param {number}     [opts.dimension=384]
   * @param {number}     [opts.numWorkers]  defaults to min(cpus-2, 16), >=2
   */
  constructor(opts = {}) {
    this.numWorkers = opts.numWorkers || Math.max(2, Math.min((os.cpus().length || 4) - 2, 16));
    this.dimension = opts.dimension || 384;
    this._modelBytes = opts.modelBytes;
    this._tokenizerJson = opts.tokenizerJson;
    this._maxLength = opts.maxLength || 256;
    this._requestTimeoutMs = opts.requestTimeoutMs ?? 30000;
    this._workers = [];
    this._pending = new Map();   // id -> { resolve, reject, worker, timer }
    this._seq = 0;
    this._shuttingDown = false;
  }

  async init() {
    if (!this._modelBytes || !this._tokenizerJson) {
      throw new Error('ParallelEmbedder requires modelBytes and tokenizerJson');
    }
    // Share model bytes across all workers via a single SharedArrayBuffer.
    const sab = new SharedArrayBuffer(this._modelBytes.length);
    new Uint8Array(sab).set(this._modelBytes);

    const workerUrl = new URL('./embed-worker.mjs', import.meta.url);
    const readies = [];

    for (let i = 0; i < this.numWorkers; i++) {
      const w = new Worker(workerUrl, {
        workerData: { modelSab: sab, tokenizerJson: this._tokenizerJson, maxLength: this._maxLength },
      });
      w.on('message', (m) => this._onMessage(m));
      // If a worker dies (uncaught error or unexpected exit), fail every request
      // currently routed to it instead of letting those promises hang forever.
      w.on('error', (e) => this._failWorker(w, e instanceof Error ? e : new Error(String(e))));
      w.on('exit', (code) => {
        if (!this._shuttingDown && code !== 0) {
          this._failWorker(w, new Error(`embed worker exited unexpectedly (code ${code})`));
        }
      });
      this._workers.push(w);
      readies.push(new Promise((resolve, reject) => {
        const onReady = (m) => {
          if (m.type === 'ready') { cleanup(); resolve(); }
          else if (m.type === 'init-error') { cleanup(); reject(new Error('worker init failed: ' + m.error)); }
        };
        const onErr = (e) => { cleanup(); reject(e); };
        const cleanup = () => { w.off('message', onReady); w.off('error', onErr); };
        w.on('message', onReady);
        w.once('error', onErr);
      }));
    }

    await Promise.all(readies);
    // Drop the main-thread reference; the SAB keeps the shared copy alive.
    this._modelBytes = null;
  }

  _settle(id, fn) {
    const p = this._pending.get(id);
    if (!p) return;
    this._pending.delete(id);
    if (p.timer) clearTimeout(p.timer);
    fn(p);
  }

  _onMessage(m) {
    if (m.type !== 'result' && m.type !== 'error') return;
    this._settle(m.id, (p) => {
      if (m.type === 'error') p.reject(new Error(m.error));
      else p.resolve({ dim: m.dim, count: m.count, flat: new Float32Array(m.buffer) });
    });
  }

  /** Reject every in-flight request routed to a dead worker. */
  _failWorker(worker, err) {
    for (const [id, p] of this._pending) {
      if (p.worker === worker) this._settle(id, () => p.reject(err));
    }
  }

  _send(worker, texts) {
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      const entry = { resolve, reject, worker, timer: null };
      if (this._requestTimeoutMs > 0) {
        entry.timer = setTimeout(() => {
          this._settle(id, () =>
            reject(new Error(`embed request timed out after ${this._requestTimeoutMs}ms`)));
        }, this._requestTimeoutMs);
        // Don't keep the event loop alive solely for this timer.
        if (typeof entry.timer.unref === 'function') entry.timer.unref();
      }
      this._pending.set(id, entry);
      worker.postMessage({ type: 'embed', id, texts });
    });
  }

  /**
   * Embed many texts across workers. Returns number[][] in input order.
   *
   * Texts are dispatched in bounded chunks (default 8) that workers pull as
   * they finish (work-stealing), rather than one giant shard per worker:
   * a large bulk batch (ADR-210 D3 ingest) would otherwise exceed the
   * per-request timeout (~400ms/text in WASM x hundreds of texts), and a
   * single slow worker would gate the whole batch.
   */
  async embedBatch(texts, opts = {}) {
    if (!texts || texts.length === 0) return [];
    const chunkSize = Math.max(1, opts.chunkSize ?? 8);
    const chunks = [];
    for (let start = 0; start < texts.length; start += chunkSize) {
      chunks.push({ start, texts: texts.slice(start, start + chunkSize) });
    }
    const out = new Array(texts.length);
    let next = 0;
    const drain = async (worker) => {
      for (;;) {
        const idx = next++;
        if (idx >= chunks.length) return;
        const { start, texts: chunkTexts } = chunks[idx];
        const { dim, count, flat } = await this._send(worker, chunkTexts);
        for (let j = 0; j < count; j++) {
          out[start + j] = Array.from(flat.subarray(j * dim, (j + 1) * dim));
        }
      }
    };
    await Promise.all(this._workers.map(drain));
    return out;
  }

  async shutdown() {
    this._shuttingDown = true;
    // Reject anything still in flight so callers don't hang on shutdown.
    for (const [id, p] of this._pending) {
      this._settle(id, () => p.reject(new Error('ParallelEmbedder shut down')));
    }
    const ws = this._workers;
    this._workers = [];
    await Promise.all(ws.map((w) => w.terminate()));
  }
}

export default ParallelEmbedder;
