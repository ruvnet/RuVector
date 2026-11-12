/**
 * WeightRegistry: immutable, versioned ruv-fann model snapshots indexed by task embedding.
 *
 * State changes are events appended to a JSON-lines log (`snapshot`, `outcome`); opening a
 * registry replays the log and rebuilds the vector index. Snapshots are frozen and never edited
 * in place: every edit or fine-tune produces a child snapshot linked to its parent.
 */
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { createHash } from 'node:crypto';
import {
  DenseModel, Topology, WeightSpaceError, freezeModel, modelHash, sameTopology, topologyOf, validateModel,
  validateModelWithJson,
} from './model.js';
import { LayerPatch, ModelDiff, PerturbOptions, diff as diffModels, perturb, setLayer } from './edit.js';
import { RuvectorStoreOptions, StoreChoice, VectorStore, checkVector, openStore } from './store.js';
import { LearnOptions, LearnResult, Trajectory, learnFromOutcomes } from './finetune.js';

export interface OutcomeEntry {
  score: number;
  at: string;
  note?: string;
}

export type SnapshotOp = 'save' | 'setLayer' | 'perturb' | 'finetune' | 'commit';

export interface SnapshotRecord {
  readonly id: string;
  /** 1 for a root snapshot, parent.version + 1 otherwise. */
  readonly version: number;
  readonly parent: string | null;
  readonly task: string | null;
  /** sha256 of the canonical model JSON. */
  readonly hash: string;
  readonly topology: Topology;
  readonly createdAt: string;
  readonly op: SnapshotOp;
  readonly tags: readonly string[];
  readonly note: string | null;
  /** Free-form provenance (e.g. fine-tune holdout losses). */
  readonly detail: Readonly<Record<string, unknown>> | null;
  readonly embedding: readonly number[];
  readonly model: DenseModel;
  /** Append-only outcome log for this snapshot. */
  readonly outcomes: readonly OutcomeEntry[];
}

export interface SaveMeta {
  task?: string;
  parent?: string | null;
  tags?: string[];
  note?: string;
  /** Initial outcome scores to record with the snapshot. */
  outcomes?: number[];
  detail?: Record<string, unknown>;
}

export interface NearestHit {
  id: string;
  similarity: number;
  record: SnapshotRecord;
}

export interface RegistryOptions {
  /** Embedding dimensions. Optional when `path` already holds snapshots, or set by the first save. */
  dimensions?: number;
  /** JSON-lines event log. Omit for an in-memory registry. */
  path?: string;
  /** Vector index: 'auto' (default; @ruvector/core when installed, else flat), 'flat', 'ruvector-core', or an instance. */
  store?: StoreChoice | VectorStore;
  /** Options for the @ruvector/core index (defaults to `<path>.index.db` when `path` is set). */
  ruvector?: RuvectorStoreOptions;
}

type LogEvent =
  | { v: 1; type: 'snapshot'; record: SnapshotRecord }
  | { v: 1; type: 'outcome'; id: string; entry: OutcomeEntry };

function snapshotId(hash: string, parent: string | null, task: string | null, embedding: readonly number[]): string {
  return 'ws-' + createHash('sha256').update(JSON.stringify([hash, parent, task, embedding])).digest('hex').slice(0, 16);
}

function checkScore(score: unknown): number {
  if (typeof score !== 'number' || !Number.isFinite(score)) throw new WeightSpaceError('outcome score must be a finite number', 'EINVALID');
  return score;
}

export class WeightRegistry {
  private records = new Map<string, SnapshotRecord>();
  private outcomes = new Map<string, OutcomeEntry[]>();
  private children = new Map<string, string[]>();
  private store: VectorStore | null = null;
  private dims: number | null;
  private readonly path: string | null;

  private constructor(private readonly options: RegistryOptions) {
    this.path = options.path ? resolve(options.path) : null;
    this.dims = options.dimensions ?? null;
  }

  /** Open (or create) a registry, replaying its event log if `path` exists. */
  static async open(options: RegistryOptions = {}): Promise<WeightRegistry> {
    if (options.dimensions !== undefined && (!Number.isSafeInteger(options.dimensions) || options.dimensions < 1)) {
      throw new WeightSpaceError('dimensions must be a positive integer', 'ERANGE');
    }
    const reg = new WeightRegistry(options);
    const events = reg.path && existsSync(reg.path) ? reg.readLog(reg.path) : [];
    for (const ev of events) {
      if (ev.type === 'snapshot') {
        const rec = reg.verify(ev.record);
        if (reg.dims === null) reg.dims = rec.embedding.length;
        reg.index(rec);
      } else if (ev.type === 'outcome' && reg.records.has(ev.id)) {
        reg.outcomes.get(ev.id)!.push({ ...ev.entry, score: checkScore(ev.entry.score) });
      }
    }
    if (reg.dims !== null) {
      await reg.ensureStore(reg.dims);
      const entries = [...reg.records.values()].map((r) => ({ id: r.id, vector: r.embedding }));
      if (reg.store!.sync) await reg.store!.sync(entries);
      else for (const e of entries) await reg.store!.upsert(e.id, e.vector);
    }
    return reg;
  }

  get dimensions(): number | null {
    return this.dims;
  }

  /** 'flat' or 'ruvector-core', once the index exists. */
  get storeKind(): VectorStore['kind'] | null {
    return this.store ? this.store.kind : null;
  }

  get size(): number {
    return this.records.size;
  }

  /** Save a model snapshot keyed by a task embedding. Saving identical content twice returns the same record. */
  async save(model: DenseModel, taskEmbedding: ArrayLike<number>, meta: SaveMeta = {}, op: SnapshotOp = 'save'): Promise<SnapshotRecord> {
    const { model: clean, json } = validateModelWithJson(model);
    const dims = this.dims ?? taskEmbedding?.length;
    checkVector(taskEmbedding, dims, 'taskEmbedding');
    const parentId = meta.parent ?? null;
    const parent = parentId ? this.require(parentId) : null;
    const embedding = Array.from(taskEmbedding);
    const hash = modelHash(clean, json);
    const task = meta.task ?? parent?.task ?? null;
    const id = snapshotId(hash, parentId, task, embedding);
    const existing = this.records.get(id);
    if (existing) return this.view(existing);
    if (meta.tags !== undefined && (!Array.isArray(meta.tags) || meta.tags.some((t) => typeof t !== 'string'))) {
      throw new WeightSpaceError('tags must be an array of strings', 'EINVALID');
    }
    const rec: SnapshotRecord = {
      id, version: parent ? parent.version + 1 : 1, parent: parentId, task, hash,
      topology: topologyOf(clean), createdAt: new Date().toISOString(), op,
      tags: meta.tags ?? [], note: meta.note ?? null, detail: meta.detail ?? null,
      embedding, model: freezeModel(clean), outcomes: [],
    };
    await this.ensureStore(dims);
    this.append({ v: 1, type: 'snapshot', record: rec });
    this.index(rec);
    await this.store!.upsert(id, embedding);
    for (const score of meta.outcomes ?? []) await this.recordOutcome(id, score);
    return this.view(this.records.get(id)!);
  }

  /** Snapshot by id, or undefined. */
  get(id: string): SnapshotRecord | undefined {
    const rec = this.records.get(id);
    return rec ? this.view(rec) : undefined;
  }

  list(): SnapshotRecord[] {
    return [...this.records.values()].map((r) => this.view(r));
  }

  /**
   * The k snapshots whose task embeddings are most similar (cosine) to `taskEmbedding`.
   * `compatibleWith` keeps only snapshots with the same topology as the given model.
   */
  async nearest(taskEmbedding: ArrayLike<number>, k = 1, filter: { compatibleWith?: DenseModel } = {}): Promise<NearestHit[]> {
    if (!Number.isSafeInteger(k) || k < 1) throw new WeightSpaceError('k must be a positive integer', 'ERANGE');
    if (!this.store || this.records.size === 0) return [];
    const want = filter.compatibleWith ? validateModel(filter.compatibleWith) : null;
    // Over-fetch so that ties (children share their parent's embedding) and topology filtering
    // can be resolved; ties go to the newer version.
    let fetch = Math.min(this.records.size, k * 4 + 8);
    for (;;) {
      const hits = await this.store.search(taskEmbedding, fetch);
      const out: NearestHit[] = [];
      for (const h of hits) {
        const rec = this.records.get(h.id);
        if (!rec || (want && !sameTopology(rec.model, want))) continue;
        out.push({ id: h.id, similarity: h.similarity, record: rec });
      }
      out.sort((a, b) => (Math.abs(a.similarity - b.similarity) > 1e-6 ? b.similarity - a.similarity : b.record.version - a.record.version));
      const lowestFetched = hits.length ? hits[hits.length - 1].similarity : -Infinity;
      const settled = out.length >= k && out[k - 1].similarity - lowestFetched > 1e-6;
      if (settled || fetch >= this.records.size) {
        return out.slice(0, k).map((h) => ({ ...h, record: this.view(h.record) }));
      }
      fetch = Math.min(this.records.size, fetch * 4);
    }
  }

  /** Ancestry from the root snapshot down to `id` (inclusive). */
  lineage(id: string): SnapshotRecord[] {
    const chain: SnapshotRecord[] = [];
    let cur: SnapshotRecord | undefined = this.require(id);
    while (cur) {
      chain.push(this.view(cur));
      cur = cur.parent ? this.records.get(cur.parent) : undefined;
    }
    return chain.reverse();
  }

  /** Direct children of a snapshot. */
  childrenOf(id: string): SnapshotRecord[] {
    this.require(id);
    return (this.children.get(id) ?? []).map((c) => this.view(this.records.get(c)!));
  }

  /** Append an outcome score (e.g. task success in [0, 1]) to a snapshot. */
  async recordOutcome(id: string, score: number, note?: string): Promise<OutcomeEntry> {
    this.require(id);
    const entry: OutcomeEntry = { score: checkScore(score), at: new Date().toISOString(), ...(note ? { note } : {}) };
    this.append({ v: 1, type: 'outcome', id, entry });
    this.outcomes.get(id)!.push(entry);
    return entry;
  }

  /** Save `model` as a child of `parentId` (same task embedding unless overridden). */
  async commit(parentId: string, model: DenseModel, meta: Omit<SaveMeta, 'parent'> & { embedding?: ArrayLike<number> } = {}, op: SnapshotOp = 'commit'): Promise<SnapshotRecord> {
    const parent = this.require(parentId);
    const { embedding, ...rest } = meta;
    return this.save(model, embedding ?? parent.embedding, { ...rest, parent: parentId, tags: rest.tags ?? [...parent.tags] }, op);
  }

  /** Child snapshot with one layer patched. */
  async setLayer(id: string, index: number, patch: LayerPatch, meta: Omit<SaveMeta, 'parent'> = {}): Promise<SnapshotRecord> {
    const next = setLayer(this.require(id).model, index, patch);
    return this.commit(id, next, { ...meta, detail: { ...(meta.detail ?? {}), layer: index, fields: Object.keys(patch) } }, 'setLayer');
  }

  /** Child snapshot with seeded Gaussian noise added. */
  async perturb(id: string, options: PerturbOptions = {}, meta: Omit<SaveMeta, 'parent'> = {}): Promise<SnapshotRecord> {
    const next = perturb(this.require(id).model, options);
    return this.commit(id, next, { ...meta, detail: { ...(meta.detail ?? {}), perturb: { scale: options.scale ?? 0.01, seed: options.seed ?? 1 } } }, 'perturb');
  }

  /** Fine-tune `id` on outcome trajectories; saves a child only if a held-out check improves. */
  learnFromOutcomes(id: string, trajectories: Trajectory[], options: LearnOptions = {}): Promise<LearnResult> {
    return learnFromOutcomes(this, id, trajectories, options);
  }

  diff(idA: string, idB: string): ModelDiff {
    return diffModels(this.require(idA).model, this.require(idB).model);
  }

  async close(): Promise<void> {
    if (this.store) await this.store.close();
    this.store = null;
  }

  // ── internals ──────────────────────────────────────────────────────────────

  private require(id: string): SnapshotRecord {
    if (typeof id !== 'string') throw new WeightSpaceError('id must be a string', 'EINVALID');
    const rec = this.records.get(id);
    if (!rec) throw new WeightSpaceError(`unknown snapshot '${id}'`, 'ENOTFOUND');
    return rec;
  }

  private view(rec: SnapshotRecord): SnapshotRecord {
    return { ...rec, outcomes: (this.outcomes.get(rec.id) ?? []).slice() };
  }

  private index(rec: SnapshotRecord): void {
    this.records.set(rec.id, rec);
    this.outcomes.set(rec.id, []);
    if (rec.parent) {
      const list = this.children.get(rec.parent) ?? [];
      list.push(rec.id);
      this.children.set(rec.parent, list);
    }
  }

  private async ensureStore(dims: number): Promise<void> {
    if (this.store) return;
    this.dims = dims;
    const choice = this.options.store ?? 'auto';
    if (typeof choice === 'object') {
      if (choice.dimensions !== dims) throw new WeightSpaceError(`store has ${choice.dimensions} dimensions, registry needs ${dims}`, 'ESHAPE');
      this.store = choice;
      return;
    }
    const ruvector: RuvectorStoreOptions = { ...(this.options.ruvector ?? {}) };
    if (!ruvector.storagePath && this.path) ruvector.storagePath = this.path + '.index.db';
    this.store = await openStore(dims, choice, ruvector);
  }

  private verify(raw: SnapshotRecord): SnapshotRecord {
    const { model, json } = validateModelWithJson(raw.model);
    const hash = modelHash(model, json);
    if (hash !== raw.hash) throw new WeightSpaceError(`snapshot ${raw.id} hash mismatch (log tampered or corrupt)`, 'EINTEGRITY');
    if (this.dims !== null) checkVector(raw.embedding, this.dims, `snapshot ${raw.id} embedding`);
    if (raw.parent && !this.records.has(raw.parent)) throw new WeightSpaceError(`snapshot ${raw.id} references unknown parent ${raw.parent}`, 'EINTEGRITY');
    if (snapshotId(hash, raw.parent ?? null, raw.task ?? null, raw.embedding) !== raw.id) {
      throw new WeightSpaceError(`snapshot ${raw.id} id does not match its content`, 'EINTEGRITY');
    }
    return { ...raw, model: freezeModel(model), outcomes: [] };
  }

  private readLog(path: string): LogEvent[] {
    const out: LogEvent[] = [];
    const lines = readFileSync(path, 'utf8').split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      if (!line) continue;
      let ev: LogEvent;
      try {
        ev = JSON.parse(line);
      } catch {
        // A torn final line (crash mid-append) is skipped; corruption elsewhere is an error.
        if (i >= lines.length - 2) continue;
        throw new WeightSpaceError(`${path}:${i + 1} is not valid JSON`, 'EINTEGRITY');
      }
      if (ev && ev.v === 1 && (ev.type === 'snapshot' || ev.type === 'outcome')) out.push(ev);
    }
    return out;
  }

  private append(ev: LogEvent): void {
    if (!this.path) return;
    mkdirSync(dirname(this.path), { recursive: true });
    const { record } = ev as { record?: SnapshotRecord };
    const line = record ? { ...ev, record: { ...record, outcomes: [] } } : ev;
    appendFileSync(this.path, JSON.stringify(line) + '\n');
  }
}
