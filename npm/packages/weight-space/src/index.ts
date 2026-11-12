/**
 * @ruvector/weight-space: a versioned weight registry for ruv-fann networks, indexed by task
 * embedding in a RuVector index, with safe weight-editing primitives and outcome-driven fine-tuning.
 */
export {
  ACTIVATIONS, LIMITS, WeightSpaceError, canonicalJson, createMlp, modelHash, parameterCount,
  sameTopology, topologyOf, validateLayer, validateModel,
} from './model.js';
export type { DenseActivation, DenseLayer, DenseModel, Limits, Topology } from './model.js';

export { loadFann, meanSquaredError, trainWithinBudget, trainingWork } from './fann.js';
export type { FannRuntime, PreparedModel, TrainLoopOptions, TrainLoopResult } from './fann.js';

export { FlatVectorStore, RuvectorVectorStore, openStore } from './store.js';
export type { RuvectorStoreOptions, StoreChoice, VectorHit, VectorStore } from './store.js';

export { diff, getLayer, perturb, setLayer } from './edit.js';
export type { LayerDiff, LayerPatch, LayerView, ModelDiff, PerturbOptions } from './edit.js';

export { WeightRegistry } from './registry.js';
export type { NearestHit, OutcomeEntry, RegistryOptions, SaveMeta, SnapshotOp, SnapshotRecord } from './registry.js';

export { learnFromOutcomes } from './finetune.js';
export type { LearnOptions, LearnResult, Trajectory } from './finetune.js';

export const VERSION = '0.1.0';
