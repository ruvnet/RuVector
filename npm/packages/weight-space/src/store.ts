/**
 * Vector index behind the weight registry.
 *
 * - `FlatVectorStore`: exact cosine search over one contiguous Float32Array (no dependencies).
 * - `RuvectorVectorStore`: `@ruvector/core` VectorDb (native Rust, HNSW or flat), an optional peer.
 *
 * Both report cosine *similarity* (higher is closer); @ruvector/core's Cosine score is a
 * distance and is converted at this boundary.
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { WeightSpaceError } from './model.js';

const dynamicImport = new Function('specifier', 'return import(specifier)') as (s: string) => Promise<any>;

export interface VectorHit {
  id: string;
  /** Cosine similarity in [-1, 1]. */
  similarity: number;
}

export interface VectorStore {
  readonly kind: 'flat' | 'ruvector-core';
  readonly dimensions: number;
  size(): number;
  /** Insert or replace. */
  upsert(id: string, vector: ArrayLike<number>): Promise<void>;
  search(vector: ArrayLike<number>, k: number): Promise<VectorHit[]>;
  close(): Promise<void>;
  /**
   * Bring the index in line with `entries` (the registry's replayed log). Persistent stores may
   * skip work when they already hold exactly these entries.
   */
  sync?(entries: { id: string; vector: ArrayLike<number> }[]): Promise<{ reindexed: number }>;
}

export function checkVector(vector: ArrayLike<number>, dimensions: number, name = 'vector'): void {
  if (!vector || typeof (vector as ArrayLike<number>).length !== 'number') throw new WeightSpaceError(`${name} must be an array`, 'EINVALID');
  if (vector.length !== dimensions) throw new WeightSpaceError(`${name} must have ${dimensions} dimensions, got ${vector.length}`, 'ESHAPE');
  let norm = 0;
  for (let i = 0; i < vector.length; i++) {
    const v = vector[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new WeightSpaceError(`${name}[${i}] must be finite`, 'EINVALID');
    norm += v * v;
  }
  if (norm === 0) throw new WeightSpaceError(`${name} must not be all zeros`, 'EINVALID');
}

/** Exact cosine top-k over normalised rows in one growable Float32Array. */
export class FlatVectorStore implements VectorStore {
  readonly kind = 'flat' as const;
  private data: Float32Array;
  private ids: string[] = [];
  private slot = new Map<string, number>();
  private scores = new Float32Array(0);
  private query: Float32Array;

  constructor(readonly dimensions: number, capacity = 64) {
    if (!Number.isSafeInteger(dimensions) || dimensions < 1) throw new WeightSpaceError('dimensions must be a positive integer', 'ERANGE');
    this.data = new Float32Array(dimensions * Math.max(1, capacity));
    this.query = new Float32Array(dimensions);
  }

  size(): number {
    return this.ids.length;
  }

  async upsert(id: string, vector: ArrayLike<number>): Promise<void> {
    checkVector(vector, this.dimensions);
    let row = this.slot.get(id);
    if (row === undefined) {
      row = this.ids.length;
      if ((row + 1) * this.dimensions > this.data.length) {
        const grown = new Float32Array(this.data.length * 2);
        grown.set(this.data);
        this.data = grown;
      }
      this.ids.push(id);
      this.slot.set(id, row);
    }
    normaliseInto(vector, this.data, row * this.dimensions, this.dimensions);
  }

  async search(vector: ArrayLike<number>, k: number): Promise<VectorHit[]> {
    checkVector(vector, this.dimensions, 'query');
    const n = this.ids.length;
    const kk = Math.min(Math.max(0, Math.floor(k)), n);
    if (kk === 0) return [];
    const d = this.dimensions;
    const q = this.query;
    normaliseInto(vector, q, 0, d);
    if (this.scores.length < n) this.scores = new Float32Array(Math.max(n, this.scores.length * 2));
    const scores = this.scores;
    const data = this.data;
    for (let r = 0, off = 0; r < n; r++, off += d) {
      let s = 0;
      for (let i = 0; i < d; i++) s += q[i] * data[off + i];
      scores[r] = s;
    }
    // Partial selection: keep the k best in a small sorted array.
    const top: number[] = [];
    for (let r = 0; r < n; r++) {
      const s = scores[r];
      if (top.length < kk) {
        let p = top.length;
        top.push(r);
        while (p > 0 && scores[top[p - 1]] < s) { top[p] = top[p - 1]; p--; }
        top[p] = r;
      } else if (s > scores[top[kk - 1]]) {
        let p = kk - 1;
        while (p > 0 && scores[top[p - 1]] < s) { top[p] = top[p - 1]; p--; }
        top[p] = r;
      }
    }
    return top.map((r) => ({ id: this.ids[r], similarity: scores[r] }));
  }

  async close(): Promise<void> {
    /* nothing to release */
  }
}

function normaliseInto(src: ArrayLike<number>, dst: Float32Array, offset: number, d: number): void {
  let norm = 0;
  for (let i = 0; i < d; i++) norm += src[i] * src[i];
  const inv = norm > 0 ? 1 / Math.sqrt(norm) : 0;
  for (let i = 0; i < d; i++) dst[offset + i] = src[i] * inv;
}

export interface RuvectorStoreOptions {
  /** On-disk index file. Without one, a temp file is created and removed on close(). */
  storagePath?: string;
  /** HNSW parameters; omit for @ruvector/core's exact flat index. */
  hnsw?: { m?: number; efConstruction?: number; efSearch?: number; maxElements?: number };
}

/** `@ruvector/core` VectorDb adapter. Ids are kept in memory so size() is synchronous. */
export class RuvectorVectorStore implements VectorStore {
  readonly kind = 'ruvector-core' as const;
  private known = new Set<string>();

  private constructor(readonly dimensions: number, private db: any, private tempDir: string | null) {}

  static async open(dimensions: number, options: RuvectorStoreOptions = {}): Promise<RuvectorVectorStore> {
    let core: any;
    try {
      const mod = await dynamicImport('@ruvector/core');
      core = mod.VectorDb ? mod : mod.default;
    } catch (err) {
      throw new WeightSpaceError(`@ruvector/core is not installed (${(err as Error).message}); run: npm i @ruvector/core`, 'ENOCORE');
    }
    // Not a declared peer (an exact pin conflicts with every other core version npm resolves),
    // so check the API shape this adapter relies on (tested with @ruvector/core 0.1.38).
    const proto = core && typeof core.VectorDb === 'function' ? core.VectorDb.prototype : null;
    if (!proto || !['insert', 'insertBatch', 'search', 'get', 'len'].every((m) => typeof proto[m] === 'function')) {
      throw new WeightSpaceError('installed @ruvector/core does not expose the VectorDb API this adapter needs', 'ENOCORE');
    }
    const tempDir = options.storagePath ? null : mkdtempSync(join(tmpdir(), 'weight-space-'));
    const storagePath = options.storagePath ?? join(tempDir as string, 'index.db');
    const db = new core.VectorDb({
      dimensions,
      storagePath,
      distanceMetric: 'Cosine',
      ...(options.hnsw ? { hnswConfig: options.hnsw } : {}),
    });
    return new RuvectorVectorStore(dimensions, db, tempDir);
  }

  size(): number {
    return this.known.size;
  }

  async upsert(id: string, vector: ArrayLike<number>): Promise<void> {
    checkVector(vector, this.dimensions);
    await this.db.insert({ id, vector: Float32Array.from(vector) });
    this.known.add(id);
  }

  /**
   * Reuse the persisted index when it already holds exactly these ids (same count and the newest
   * id present); otherwise insert everything in batches. Ids are content hashes, so a matching id
   * implies a matching vector.
   */
  async sync(entries: { id: string; vector: ArrayLike<number> }[]): Promise<{ reindexed: number }> {
    if (entries.length === 0) return { reindexed: 0 };
    const count = await this.db.len();
    if (count === entries.length && (await this.db.get(entries[entries.length - 1].id))) {
      for (const e of entries) this.known.add(e.id);
      return { reindexed: 0 };
    }
    for (let i = 0; i < entries.length; i += 1000) {
      const batch = entries.slice(i, i + 1000);
      for (const e of batch) checkVector(e.vector, this.dimensions);
      await this.db.insertBatch(batch.map((e) => ({ id: e.id, vector: Float32Array.from(e.vector) })));
      for (const e of batch) this.known.add(e.id);
    }
    return { reindexed: entries.length };
  }

  async search(vector: ArrayLike<number>, k: number): Promise<VectorHit[]> {
    checkVector(vector, this.dimensions, 'query');
    const kk = Math.min(Math.max(0, Math.floor(k)), this.known.size);
    if (kk === 0) return [];
    const hits: { id: string; score: number }[] = await this.db.search({ vector: Float32Array.from(vector), k: kk });
    return hits.map((h) => ({ id: h.id, similarity: 1 - h.score }));
  }

  async close(): Promise<void> {
    this.db = null;
    if (this.tempDir) rmSync(this.tempDir, { recursive: true, force: true });
  }
}

export type StoreChoice = 'auto' | 'flat' | 'ruvector-core';

/** `auto` uses @ruvector/core when it loads and falls back to the flat store otherwise. */
export async function openStore(dimensions: number, choice: StoreChoice = 'auto', options: RuvectorStoreOptions = {}): Promise<VectorStore> {
  if (choice === 'flat') return new FlatVectorStore(dimensions);
  try {
    return await RuvectorVectorStore.open(dimensions, options);
  } catch (err) {
    if (choice === 'ruvector-core') throw err;
    return new FlatVectorStore(dimensions);
  }
}
