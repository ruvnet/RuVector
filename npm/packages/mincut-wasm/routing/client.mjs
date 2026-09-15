const boundedInteger = (value, max) => Number.isInteger(value) && value >= 0 && value <= max;
const privateArray = (value, Type) => value instanceof Type && value.buffer instanceof ArrayBuffer;

/** Dedicated WASM worker. Aborting any in-flight call terminates this instance
 * and rejects ALL pending work, including mutations; construct a new instance.
 * Arrays are cloned, not transferred. Never pass untrusted URLs to a worker.
 */
export class RoutingWorker {
  #worker;
  #pending = new Map();
  #next = 0;
  #closed = false;
  constructor(worker) {
    this.#worker = worker;
    const receive = ({ id, result, error }) => {
      const pending = this.#pending.get(id);
      if (!pending) return;
      this.#pending.delete(id);
      pending.cleanup();
      if (error) pending.reject(new Error(error)); else pending.resolve(result);
    };
    if (typeof worker.on === 'function') {
      worker.on('message', receive);
      worker.on('error', error => this.close(error));
      worker.on('exit', () => this.close(new Error('Routing worker exited')));
    } else {
      worker.addEventListener('message', event => receive(event.data));
      worker.addEventListener('error', () => this.close(new Error('Routing worker failed')));
      worker.addEventListener('messageerror', () => this.close(new Error('Invalid worker message')));
    }
  }
  static async create(nodes, endpoints, costs, turns = new Uint32Array(), options = {}) {
    if (!Number.isInteger(nodes) || nodes < 1 || nodes > 1_000_000 ||
        !privateArray(endpoints, Uint32Array) || !privateArray(costs, Uint32Array) ||
        !privateArray(turns, Uint32Array) || costs.length > 4_000_000 ||
        endpoints.length !== costs.length * 2 || turns.length % 2 || turns.length > 8_000_000) {
      throw new TypeError('Invalid graph arrays or size');
    }
    let worker;
    if (typeof process !== 'undefined' && process.versions?.node) {
      const { Worker } = await import('node:worker_threads');
      worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
    } else {
      worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
    }
    const client = new RoutingWorker(worker);
    try { await client.#call('create', [nodes, endpoints, costs, turns], options); }
    catch (error) { client.close(error); throw error; }
    return client;
  }
  #call(operation, args, { signal, timeoutMs = 30_000 } = {}) {
    if (this.#closed) return Promise.reject(new Error('Routing worker closed'));
    if (this.#pending.size >= 32) return Promise.reject(new Error('Routing queue full'));
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0 || timeoutMs > 300_000) return Promise.reject(new RangeError('Invalid timeout'));
    if (signal && (typeof signal.addEventListener !== 'function' || typeof signal.removeEventListener !== 'function' || typeof signal.aborted !== 'boolean')) return Promise.reject(new TypeError('Invalid abort signal'));
    if (signal?.aborted) return Promise.reject(new Error('Routing request aborted'));
    // Bound cloned message bytes across the queue, independent of worker throughput.
    const bytes = args.reduce((sum, arg) => sum + (ArrayBuffer.isView(arg) ? arg.byteLength : 8), 0);
    let queued = bytes;
    for (const item of this.#pending.values()) queued += item.bytes;
    if (queued > 128 * 1024 * 1024) return Promise.reject(new Error('Routing queue byte limit'));
    return new Promise((resolve, reject) => {
      const id = ++this.#next;
      const abort = () => this.close(new Error('Routing request aborted'));
      const timer = setTimeout(() => this.close(new Error('Routing request timed out')), timeoutMs);
      const cleanup = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort); };
      this.#pending.set(id, { resolve, reject, cleanup, bytes });
      signal?.addEventListener('abort', abort, { once: true });
      try { this.#worker.postMessage({ id, operation, args }); }
      catch (error) { this.#pending.delete(id); cleanup(); reject(error); }
    });
  }
  route(source, target, options = {}) {
    if (!boundedInteger(source, 0xffffffff) || !boundedInteger(target, 0xffffffff) || !boundedInteger(options.budget ?? 5_000_000, 200_000_000) || typeof (options.landmarks ?? true) !== 'boolean') return Promise.reject(new TypeError('Invalid route arguments'));
    return this.#call('route', [source, target, options.landmarks ?? true, options.budget ?? 5_000_000], options);
  }
  prepare(landmarks, options = {}) {
    if (!privateArray(landmarks, Uint32Array) || landmarks.length > 16 || !boundedInteger(options.budget ?? 100_000_000, 200_000_000)) return Promise.reject(new TypeError('Invalid landmarks'));
    return this.#call('prepare', [landmarks, options.budget ?? 100_000_000], options);
  }
  update(ids, costs, options = {}) {
    if (!privateArray(ids, Uint32Array) || !privateArray(costs, Uint32Array) || ids.length !== costs.length || ids.length > 4_000_000) return Promise.reject(new TypeError('Invalid updates'));
    return this.#call('update', [ids, costs], options);
  }
  setCoordinates(latLon, options = {}) {
    if (!privateArray(latLon, Float64Array) || latLon.length > 2_000_000) return Promise.reject(new TypeError('Invalid coordinates'));
    return this.#call('setCoordinates', [latLon], options);
  }
  nearest(lat, lon, radiusM, options = {}) {
    if (!Number.isFinite(lat) || Math.abs(lat)>90 || !Number.isFinite(lon) || Math.abs(lon)>180 || !Number.isFinite(radiusM) || radiusM<0) return Promise.reject(new TypeError('Invalid map query'));
    return this.#call('nearest', [lat, lon, radiusM], options);
  }
  close(reason = new Error('Routing worker closed')) {
    if (this.#closed) return;
    this.#closed = true;
    this.#worker.terminate();
    for (const pending of this.#pending.values()) { pending.cleanup(); pending.reject(reason); }
    this.#pending.clear();
  }
}
