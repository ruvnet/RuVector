// Type-level compatibility check, compiled by test/learning-import-types.js (tsc --noEmit).
// LearningEngine.import() must keep accepting state shaped like a pre-0.3.22 export()
// (no `neural` field). 0.3.22 made `neural` required and broke this.
import LearningEngine, { AlgorithmStats, LearningConfig, LearningTrajectory } from '../../dist/core/learning-engine';

/** Exactly the return type of LearningEngine.export() in ruvector 0.3.21. */
interface ExportStateV0321 {
  qTables: Record<string, Record<string, number>>;
  qTables2: Record<string, Record<string, number>>;
  eligibilityTraces: Record<string, Record<string, number>>;
  actorWeights: Record<string, number[]>;
  criticValues: Record<string, number>;
  trajectories: LearningTrajectory[];
  stats: Record<string, AlgorithmStats>;
  configs: Record<string, LearningConfig>;
  rewardHistory: number[];
}

const legacy: ExportStateV0321 = {
  qTables: { 'fix:.ts': { coder: 0.5 } },
  qTables2: {},
  eligibilityTraces: {},
  actorWeights: {},
  criticValues: {},
  trajectories: [],
  stats: {},
  configs: {},
  rewardHistory: [],
};

const engine = new LearningEngine();
engine.import(legacy);
// The current export() round-trips too.
engine.import(engine.export());
