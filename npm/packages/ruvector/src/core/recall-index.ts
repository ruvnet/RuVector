/**
 * Quantized recall index for IntelligenceEngine memories.
 *
 * Above a size threshold, memory recall stops scanning every embedding and
 * instead asks a RaBitQ 1-bit index (optional @ruvector/rabitq-wasm) for a
 * candidate pool, then re-scores that pool with exact cosine similarity. The
 * returned scores are therefore exact; only candidate selection is
 * approximate, and the pool is oversampled (10x k) so top-k recall stays
 * effectively 1.0 on 20k x 384 embeddings in our benchmark.
 *
 * Measured on 5k x 384 unit vectors: build ~420 ms, one brute-force scan
 * ~8.6 ms, one index search + rerank ~0.5-2.7 ms. Build costs ~50 scans, so it
 * only pays off in a long-lived process (the MCP server). Hook CLI invocations
 * are one-shot, so the index is built only once a process has served more than
 * one recall at or above the threshold -- a single-query process never pays.
 *
 * The RaBitQ index is immutable. Mutations are tracked as a delta instead of
 * forcing a rebuild per insert: ids added since the build are brute-forced
 * (a small set) and merged in, removed ids are skipped, and the index is
 * rebuilt once the delta exceeds 10% of the indexed set.
 *
 * Everything degrades to "return null" -- the caller then does its normal
 * brute-force scan -- when the module is missing, the set is small, or
 * anything throws.
 */

import { isRabitqAvailable, RabitqIndex } from './rabitq-wrapper';

export interface RecallCandidate {
  id: string;
  embedding: number[];
}

export interface RecallHit {
  id: string;
  score: number;
}

export const DEFAULT_RABITQ_THRESHOLD = 5000;

/** Size at which recall switches to the index. Env: RUVECTOR_RABITQ_THRESHOLD (0 disables). */
export function rabitqThreshold(): number {
  const raw = process.env.RUVECTOR_RABITQ_THRESHOLD;
  if (raw === undefined || raw === '') return DEFAULT_RABITQ_THRESHOLD;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : DEFAULT_RABITQ_THRESHOLD;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return 0;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const d = Math.sqrt(na) * Math.sqrt(nb);
  return d > 0 ? dot / d : 0;
}

function unit(v: ArrayLike<number>): Float32Array {
  const out = new Float32Array(v.length);
  let n = 0;
  for (let i = 0; i < v.length; i++) n += v[i] * v[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < v.length; i++) out[i] = v[i] / n;
  return out;
}

export class QuantizedRecallIndex {
  private index: RabitqIndex | null = null;
  private indexDim = 0;
  private indexedIds = new Set<string>();
  private added = new Set<string>();
  private removed = new Set<string>();
  private stale = true;
  private eligibleQueries = 0;
  private disabled = false;
  /** Engine that served the last search(): 'rabitq' or 'brute-force'. */
  lastEngine: 'rabitq' | 'brute-force' = 'brute-force';

  /** A memory was inserted or overwritten. */
  noteAdded(id: string): void {
    if (this.index) {
      this.added.add(id);
      this.removed.delete(id);
    }
  }

  /** A memory was removed. */
  noteRemoved(id: string): void {
    if (this.index) {
      this.added.delete(id);
      if (this.indexedIds.has(id)) this.removed.add(id);
    }
  }

  /** The whole set changed (replace/import/clear): rebuild on next use. */
  invalidate(): void {
    this.stale = true;
  }

  dispose(): void {
    if (this.index) {
      try { this.index.free(); } catch { /* already freed */ }
    }
    this.index = null;
    this.indexedIds.clear();
    this.added.clear();
    this.removed.clear();
    this.stale = true;
  }

  /**
   * Top-k by cosine, or null when the caller should brute-force instead.
   * `entries` must be the engine's live memory map.
   */
  search(
    query: number[],
    k: number,
    entries: Map<string, RecallCandidate>
  ): RecallHit[] | null {
    this.lastEngine = 'brute-force';
    const threshold = rabitqThreshold();
    if (this.disabled || threshold === 0 || entries.size < threshold || k <= 0) return null;
    // First eligible query in a process is served by brute force: a one-shot
    // hook process must never pay the build.
    if (++this.eligibleQueries < 2) return null;
    if (!isRabitqAvailable()) {
      this.disabled = true;
      return null;
    }
    try {
      const dim = query.length;
      const deltaTooBig = this.added.size + this.removed.size > Math.max(64, this.indexedIds.size * 0.1);
      if (this.stale || !this.index || this.indexDim !== dim || deltaTooBig) {
        if (!this.rebuild(dim, entries)) return null;
      }
      if (this.indexedIds.size - this.removed.size < k) return null;

      const q = unit(query);
      const pool = Math.min(this.indexedIds.size, Math.max(10 * k, 50));
      const scored = new Map<string, number>();
      for (const hit of this.index!.search(q, pool)) {
        if (this.removed.has(hit.id)) continue;
        const e = entries.get(hit.id);
        if (e && e.embedding?.length === dim) scored.set(hit.id, cosine(query, e.embedding));
      }
      for (const id of this.added) {
        const e = entries.get(id);
        if (e && e.embedding?.length === dim) scored.set(id, cosine(query, e.embedding));
      }
      if (scored.size < k) return null;
      this.lastEngine = 'rabitq';
      return Array.from(scored, ([id, score]) => ({ id, score }))
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
    } catch {
      // Any native failure: stop trying for this process, caller brute-forces.
      this.disabled = true;
      this.dispose();
      return null;
    }
  }

  private rebuild(dim: number, entries: Map<string, RecallCandidate>): boolean {
    this.dispose();
    const ids: string[] = [];
    const vectors: Float32Array[] = [];
    for (const [id, e] of entries) {
      if (e.embedding?.length === dim) {
        ids.push(id);
        vectors.push(unit(e.embedding));
      }
    }
    if (ids.length === 0) return false;
    // Unit vectors: L2^2 = 2 - 2cos, so RaBitQ's L2 order is cosine order.
    this.index = RabitqIndex.build(ids, vectors, { dim });
    this.indexDim = dim;
    this.indexedIds = new Set(ids);
    this.stale = false;
    return true;
  }
}
