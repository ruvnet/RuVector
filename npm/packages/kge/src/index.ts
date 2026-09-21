/**
 * @ruvector/kge — holographic knowledge-graph embeddings (HolE / RotatE) with
 * link prediction, relation similarity and 2-hop composition, native
 * (napi-rs) with a WASM fallback. No network, no per-query cost.
 *
 * ```ts
 * import { createKge, defineSchema } from '@ruvector/kge';
 *
 * const schema = defineSchema({ relations: ['bornIn', 'locatedIn'] as const });
 * const kge = createKge({ scorer: 'hole', dims: 256, schema });
 * kge.addTriples([{ s: 'Ada', r: 'bornIn', o: 'London' }]);
 * const r = kge.predict({ s: 'Ada', r: 'bornIn', k: 10 });
 * r.candidates[0].entity;   // best-ranked tail
 * ```
 */

export { createKge, loadKge } from './client';
export type {
  ComposeQuery,
  HeadQuery,
  Kge,
  KgeOptions,
  PredictQuery,
  TailQuery,
} from './client';

export { defineSchema } from './schema';
export type { EntityOf, RelationOf, Schema } from './schema';

export { KgeError, isErrorShape } from './errors';

export { serve } from './serve';
export type { ServeHandle, ServeOptions } from './serve';

export { toOptionsJson } from './binding';
export type {
  Binding,
  EngineOptions,
  ModelCtor,
  ModelInstance,
} from './binding';

export type {
  AddReport,
  Candidate,
  KgeErrorKind,
  KgeErrorShape,
  PredictResult,
  RelationScore,
  ScorerKind,
  SimilarResult,
  SplitCounts,
  SplitTag,
  Stats,
  Triple,
} from './types';
