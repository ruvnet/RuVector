import type { EMeansOptions } from '@ruvector/e-means';

export interface ClusteringParameter {
  min: number;
  max: number;
  type: 'int' | 'float';
  scale: 'linear' | 'log';
  /** If supplied, must equal the baseline's value for this parameter. */
  default?: number;
}

export interface ClusteringBudget {
  /** Includes the baseline and every admitted dispatch. Integer 1..32. */
  maxEvaluations: number;
  /** Conservative scalar-work allowance, including short-batch centroid updates and revival; at most 1e9. */
  maxDistanceCoordinates: number;
  /** Cooperative deadline in milliseconds, 1..120000; default 30000. A synchronous fit cannot be preempted. */
  timeoutMs?: number;
}

export interface ClusteringEvaluationOptions {
  training: Float32Array;
  /** Nonempty, nonoverlapping validation rows. Selection uses this split; supply another test split for generalization claims. */
  holdout: Float32Array;
  dimensions: number;
  baseline: EMeansOptions;
  /** Searched parameters need explicit baseline values; peakStep uses float, all other fields use int. */
  domain: Partial<Record<'clusters' | 'epochs' | 'batchSize' | 'peakStep', ClusteringParameter>>;
  budget: ClusteringBudget;
  /** Relative holdout-MSE degradation allowance in [0,1], default 0. */
  qualityTolerance?: number;
}

export interface ClusteringScoreCard {
  variantId: string;
  /** Higher is better: 1/(1+holdoutMse). */
  primary: number;
  regressed: boolean;
  noopRate: 0;
  costPerWin: 0;
  raw: {
    fitConfig: EMeansOptions;
    trainMse: number;
    holdoutMse: number;
    baselineHoldoutMse: number;
    fitMs: number;
    holdoutMs: number;
    estimatedWork: number;
    updates: number;
    revivedClusters: number;
    trainingRows: number;
    holdoutRows: number;
    dimensions: number;
  };
}

export interface ClusteringTrial {
  variantId: string;
  genome: Record<string, number>;
  score: ClusteringScoreCard;
}

export interface ClusteringEvaluationReceipt {
  schema: 'e-means.clustering-evaluation/1';
  baseline?: ClusteringTrial;
  trials: ClusteringTrial[];
  baselineGenome: Record<string, number>;
  baselineConfig: EMeansOptions;
  domain: Record<string, ClusteringParameter>;
  dimensions: number;
  trainingRows: number;
  holdoutRows: number;
  qualityTolerance: number;
  budget: Required<ClusteringBudget>;
  dispatchCount: number;
  consumedWork: number;
  worstCaseWorkPerTrial: number;
  numericSnapshotBytes: number;
  sourceApplied: false;
  scope: string;
  timingScope: string;
  limits: { maxInputBytes: number; maxDistanceCoordinates: number; cooperativeDeadline: true; workUnit: string };
}

/** Structurally compatible with @metaharness/darwin's NumericEvaluator. */
export interface ClusteringEvaluator {
  readonly baselineGenome: Readonly<Record<string, number>>;
  readonly domain: Readonly<Record<string, Readonly<ClusteringParameter>>>;
  evaluate(genome: Readonly<Record<string, number>>, variantId: string): Promise<ClusteringScoreCard>;
  /** Returns an independent snapshot. Changes never affect evaluator state. */
  receipt(): ClusteringEvaluationReceipt;
}

export interface ClusteringTuningOptions extends ClusteringEvaluationOptions {
  generations: number;
  childrenPerGeneration: number;
  mutationSeed?: number;
}

export interface ClusteringTuningReceipt extends Omit<ClusteringEvaluationReceipt, 'schema' | 'baseline'> {
  schema: 'e-means.clustering-tuning/1';
  engine: { package: '@metaharness/darwin'; version: '0.13.0'; algorithm: 'evolveNumeric' };
  baseline: ClusteringTrial;
  winner: ClusteringTrial;
  recommendation: EMeansOptions;
  winnerRejected: boolean;
  improvedOnObservedHoldout: boolean;
  search: { generations: number; childrenPerGeneration: number; mutationSeed: number; worstCaseDispatches: number; concurrency: 1 };
  temporaryArchiveRemoved: true;
}

/** Copies inputs/configuration synchronously. First evaluate() must use baselineGenome. No SDK import occurs here. */
export function createClusteringEvaluator(options: ClusteringEvaluationOptions): ClusteringEvaluator;
/** Node-only bounded search using the actual optional @metaharness/darwin0.13.0 peer. Never applies a model. */
export function tuneClustering(options: ClusteringTuningOptions): Promise<ClusteringTuningReceipt>;
