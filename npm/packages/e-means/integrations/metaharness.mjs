import { init, fit, predict } from '../e-means.mjs';
import { checkOptions, checkGenome, clone, integer, object, workEstimate, MAX_INPUT_BYTES, MAX_WORK } from './clustering-validation.mjs';

function holdoutMse(data, dimensions, centroids) {
  const assignments = predict(data, dimensions, centroids);
  let loss = 0;
  for (let row = 0; row < assignments.length; row++) {
    const centroid = assignments[row] * dimensions;
    for (let d = 0; d < dimensions; d++) {
      const delta = data[row * dimensions + d] - centroids[centroid + d];
      loss += delta * delta;
    }
  }
  return loss / assignments.length;
}

/** NumericEvaluator compatible with published Darwin0.13.0. Snapshots happen
 * synchronously before any await. The first dispatch must be the baseline.
 */
export function createClusteringEvaluator(options) {
  const config = checkOptions(options);
  const baselineGenome = Object.freeze(Object.fromEntries(Object.keys(config.domain).map(key => [key, config.baseline[key]])));
  const trials = [], ids = new Set();
  let active = false, calls = 0, consumedWork = 0, baseline;
  const started = performance.now();
  const check = () => {
    if (performance.now() - started > config.budget.timeoutMs) throw new Error('Clustering cooperative deadline exceeded');
  };
  const receipt = () => clone({ schema: 'e-means.clustering-evaluation/1', baseline, trials,
    baselineGenome, baselineConfig: config.baseline, domain: config.domain,
    dimensions: config.dimensions, trainingRows: config.n, holdoutRows: config.h,
    qualityTolerance: config.qualityTolerance, budget: config.budget,
    dispatchCount: calls, consumedWork, worstCaseWorkPerTrial: config.worstCaseWorkPerTrial,
    numericSnapshotBytes: config.numericSnapshotBytes, sourceApplied: false,
    scope: 'Caller-supplied training and holdout snapshots. Holdout guides selection and is not an independent generalization test.',
    timingScope: 'One synchronous Rust/WASM fit and one separate holdout prediction/MSE calculation per dispatch; initialization excluded.',
    limits: { maxInputBytes: MAX_INPUT_BYTES, maxDistanceCoordinates: MAX_WORK, cooperativeDeadline: true,
      workUnit: 'Conservative coordinate-scale work estimate including centroid updates and donor selection; not measured instructions.' } });

  const evaluator = {
    baselineGenome,
    domain: config.domain,
    async evaluate(genome, variantId) {
      check();
      if (active) throw new Error('Concurrent clustering evaluations are not supported');
      if (calls >= config.budget.maxEvaluations) throw new RangeError('Clustering evaluation budget exhausted');
      calls++;
      if (typeof variantId !== 'string' || !variantId || variantId.length > 128 || ids.has(variantId)) throw new TypeError('variantId must be a unique nonempty label of at most 128 characters');
      const selected = checkGenome(genome, config.domain);
      if (!baseline && JSON.stringify(selected) !== JSON.stringify(baselineGenome)) throw new Error('First evaluation must use baselineGenome');
      const fitConfig = { ...config.baseline, ...selected };
      const work = workEstimate(fitConfig, config.n, config.h, config.dimensions);
      if (work > config.budget.maxDistanceCoordinates - consumedWork) throw new RangeError('Clustering coordinate work budget exhausted');
      ids.add(variantId); consumedWork += work; active = true;
      try {
        await init();
        check();
        const fitStart = performance.now();
        const model = fit(config.training, config.dimensions, fitConfig);
        const fitMs = performance.now() - fitStart;
        check();
        const scoreStart = performance.now();
        const mse = holdoutMse(config.holdout, config.dimensions, model.centroids);
        const holdoutMs = performance.now() - scoreStart;
        check();
        if (!Number.isFinite(mse) || mse < 0) throw new Error('Nonfinite clustering evaluation');
        const baselineMse = baseline?.score.raw.holdoutMse ?? mse;
        const score = { variantId, primary: 1 / (1 + mse), regressed: mse > baselineMse * (1 + config.qualityTolerance),
          noopRate: 0, costPerWin: 0, raw: { fitConfig, trainMse: model.mse, holdoutMse: mse, baselineHoldoutMse: baselineMse,
            fitMs, holdoutMs, estimatedWork: work, updates: model.updates, revivedClusters: model.revivedClusters,
            trainingRows: config.n, holdoutRows: config.h, dimensions: config.dimensions } };
        const trial = clone({ variantId, genome: selected, score });
        trials.push(trial);
        if (!baseline) baseline = trial;
        return clone(score);
      } finally { active = false; }
    },
    receipt,
  };
  return Object.freeze(evaluator);
}

async function loadDarwin() {
  const { readFile } = await import('node:fs/promises');
  let entry;
  try { entry = import.meta.resolve('@metaharness/darwin'); }
  catch (cause) { throw new Error('tuneClustering requires optional peer @metaharness/darwin@0.13.0', { cause }); }
  const manifest = JSON.parse(await readFile(new URL('../package.json', entry), 'utf8'));
  if (manifest.name !== '@metaharness/darwin' || manifest.version !== '0.13.0') throw new Error('Expected @metaharness/darwin 0.13.0');
  const { evolveNumeric } = await import('@metaharness/darwin');
  if (typeof evolveNumeric !== 'function') throw new Error('Darwin numeric API is unavailable');
  return evolveNumeric;
}

/** Run bounded, local numeric evolution. Never mutates source or applies a model. */
export async function tuneClustering(options) {
  const started = performance.now();
  object(options, ['training', 'holdout', 'dimensions', 'baseline', 'domain', 'budget', 'qualityTolerance', 'generations', 'childrenPerGeneration', 'mutationSeed'], 'tuning options');
  const { generations, childrenPerGeneration, mutationSeed = 0, ...evaluation } = options;
  integer(generations, 1, 8, 'generations');
  integer(childrenPerGeneration, 1, 8, 'childrenPerGeneration');
  integer(mutationSeed, 0, 0xffff_ffff, 'mutationSeed');
  const evaluator = createClusteringEvaluator(evaluation);
  const initial = evaluator.receipt();
  const check = () => {
    if (performance.now() - started > initial.budget.timeoutMs) throw new Error('Clustering cooperative deadline exceeded');
  };
  const worstCaseDispatches = 1 + childrenPerGeneration + 2 * childrenPerGeneration * (generations - 1);
  if (worstCaseDispatches > initial.budget.maxEvaluations) throw new RangeError('Darwin worst-case dispatches exceed maxEvaluations');
  if (worstCaseDispatches * initial.worstCaseWorkPerTrial > initial.budget.maxDistanceCoordinates) throw new RangeError('Darwin worst-case work exceeds maxDistanceCoordinates');
  const evolveNumeric = await loadDarwin();
  check();
  const { mkdtemp, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  let archive, failure, answer;
  try {
    archive = await mkdtemp(join(tmpdir(), 'e-means-darwin-'));
    check();
    const result = await evolveNumeric({ genomeSpec: evaluator.domain, baselineGenome: evaluator.baselineGenome,
      evaluator: { evaluate: async (...args) => {
        try { return await evaluator.evaluate(...args); }
        catch (error) { failure = error; throw error; }
      } }, generations, childrenPerGeneration, seed: mutationSeed, concurrency: 1,
      mutationSigma: 0.2, crossover: false, workRoot: archive });
    if (failure) throw failure;
    check();
    const measured = evaluator.receipt();
    if (!Array.isArray(result?.records) || result.records.length !== measured.trials.length || measured.dispatchCount !== measured.trials.length) throw new Error('Darwin archive differs from observed evaluations');
    const observed = new Map(measured.trials.map(trial => [trial.variantId, trial]));
    const verified = record => {
      const trial = observed.get(record?.variant?.id);
      if (!trial || JSON.stringify(trial.genome) !== JSON.stringify(record.variant.genome) || JSON.stringify(trial.score) !== JSON.stringify(record.score)) throw new Error('Darwin returned an unobserved result');
      return trial;
    };
    for (const record of result.records) verified(record);
    const baseline = verified(result.baseline);
    if (JSON.stringify(baseline.genome) !== JSON.stringify(evaluator.baselineGenome)) throw new Error('Darwin changed the requested baseline');
    let winner = result.winner ? verified(result.winner) : baseline;
    const winnerRejected = winner.score.regressed || winner.score.primary < baseline.score.primary;
    if (winnerRejected) winner = baseline;
    answer = { engine: { package: '@metaharness/darwin', version: '0.13.0', algorithm: 'evolveNumeric' },
      ...measured, schema: 'e-means.clustering-tuning/1', baseline, winner,
      recommendation: clone(winner.score.raw.fitConfig), winnerRejected,
      improvedOnObservedHoldout: winner.score.primary > baseline.score.primary,
      search: { generations, childrenPerGeneration, mutationSeed, worstCaseDispatches, concurrency: 1 },
      sourceApplied: false, temporaryArchiveRemoved: true };
  } finally { if (archive) await rm(archive, { recursive: true, force: true }); }
  check();
  return answer;
}
