/**
 * SHAPER-pattern frozen-weight evolution loop — one-generation skeleton
 * (PIR WP9, ADR-313, issue #841; SHAPER = arXiv:2608.11350).
 *
 * Orchestrates ONE generation:
 *
 *   propose (Darwin/GEPA surfaces) ──► evaluate (dreamMachine.ts adapter)
 *        ──► verdict ──► promotion RECOMMENDATION (vetoes/flywheel path)
 *
 * FROZEN WEIGHTS ARE STRUCTURAL HERE, not policy:
 *   - The model-weights path is simply absent: the loop takes a
 *     `FrozenModelRef` (id + loop-start sha256, witness-recorded via the
 *     dream-machine report stamp) and no weight-file path exists anywhere in
 *     the option or result types. `MutationSurface` (genome.ts) has no
 *     "weights" member, so a weight-mutating candidate is unrepresentable.
 *   - The SAME FrozenModelRef instance is handed to both the proposer
 *     (planner role) and the evaluator (optimizer role) — SHAPER's structural
 *     claim that one frozen model serves as planner and optimizer. The result
 *     records both roles' model identity for the configuration test.
 *   - This file lives inside a mutation-surface directory enforced by
 *     scripts/frozen-weights-check.mjs: importing a training / weight-writing
 *     API here fails CI.
 *
 * CONSTITUTIONAL BOUNDARY — identical to WP2's (dreamMachine.ts, ADR-306):
 * the loop emits recommendations only. It exports no promote/merge function,
 * the result is frozen data with no callable members, and
 * test/shaperLoop.test.ts asserts both. A capability-expanding candidate is
 * additionally routed through the ADR-315 constitutional gate (stubbed in
 * genome.ts, WP11 integration point) and is BLOCKED by default.
 *
 * ACCEPTED RESIDUALS (PR #869 security audit, on record — not closed here):
 *   - The capability gate fires on the proposer's SELF-DECLARED
 *     `capabilityDelta`. A mutation that adds a tool via its skill body while
 *     declaring no delta would skip the gate and reach evaluation — but not
 *     promotion: the human + veto/flywheel path still own that. Independent
 *     extraction of the capability delta from genome content is exactly the
 *     WP11 wiring named in genome.ts.
 *   - The model-hash binding is RECORD-ONLY in this slice: sha256 is
 *     validated as 64-hex and witness-stamped into the report (tamper-evident
 *     about which hash was claimed), but nothing here recomputes it from real
 *     weights. The day-0/day-30 re-hash that ENFORCES the freeze is WP12's
 *     30-day acceptance harness.
 */
import { runRuvectorGepa, policyFromGenome, type RuvectorGepaOptions } from "./darwin.js";
import {
  evaluateCandidate,
  vetoesFromVerdict,
  type DreamMachineVerdict,
} from "./dreamMachine.js";
import { pairedBootstrapDecision } from "./statistics.js";
import {
  assertEvolvableGenome,
  assertFrozenModelRef,
  constitutionalGateStub,
  expandsCapabilities,
  type CandidateMutation,
  type ConstitutionalDecision,
  type EvolvableGenome,
  type FrozenModelRef,
} from "./genome.js";

/** Propose one candidate mutation. Receives the frozen model in its PLANNER role. */
export type CandidateProposer = (
  parent: EvolvableGenome,
  plannerModel: FrozenModelRef,
) => CandidateMutation | Promise<CandidateMutation>;

/**
 * Score a genome over the pinned evaluation items, returning per-item primary
 * samples. Receives the frozen model in its OPTIMIZER role — the same
 * instance the proposer saw.
 */
export type GenomeEvaluator = (
  genome: EvolvableGenome,
  optimizerModel: FrozenModelRef,
) => readonly number[] | Promise<readonly number[]>;

export interface ShaperGenerationOptions {
  /** The frozen foundation model — one instance, both roles (SHAPER). */
  frozenModel: FrozenModelRef;
  /** The current champion genome and its per-item baseline samples. */
  parent: { genome: EvolvableGenome; samples: readonly number[] };
  propose: CandidateProposer;
  evaluate: GenomeEvaluator;
  /** Git commit sha the evaluation runs against (witness-stamped). */
  commit: string;
  /** Evaluation date, YYYY-MM-DD (defaults to today, UTC). */
  date?: string;
}

/**
 * One generation's outcome. Deliberately pure, frozen data — a
 * RECOMMENDATION, never a promotion: there is no promote(), no merge(), and
 * no side-effect handle here. Promotion belongs to the existing
 * vetoes/flywheel path plus a human (WP2's contract, adopted unchanged).
 */
export interface ShaperGenerationResult {
  /** Loop-start hash of the frozen model, embedded in the witness-stamped report. */
  readonly modelSha256: string;
  /** SHAPER structural claim: both roles served by the same frozen instance. */
  readonly plannerModelId: string;
  readonly optimizerModelId: string;
  readonly mutation: CandidateMutation;
  /** The WP2 evaluation-stage verdict (engine-owned vocabulary). */
  readonly verdict: DreamMachineVerdict;
  /** Conjunctive veto reasons for the existing promotion rule (flywheel.ts). */
  readonly vetoReasons: readonly string[];
  /** "recommend" only ever means "no veto objected" — a human owns promotion. */
  readonly recommendation: "recommend" | "blocked";
  /** Present iff the candidate crossed the ADR-315 capability boundary. */
  readonly capabilityGate?: ConstitutionalDecision;
}

/**
 * Run one SHAPER generation: mutate → evaluate → verdict → recommendation.
 *
 * A capability-expanding candidate never reaches evaluation: it is routed to
 * the constitutional gate first (ADR-315 — independently blocking, distinct
 * from behavioral promotion) and, with the WP11 wiring absent, the stub
 * blocks it. Its verdict is recorded as evaluated="blocked", which the WP2
 * adapter maps to INCONCLUSIVE — never rounded up.
 */
export async function runShaperGeneration(
  options: ShaperGenerationOptions,
): Promise<ShaperGenerationResult> {
  const { frozenModel, parent, commit } = options;
  assertFrozenModelRef(frozenModel);
  assertEvolvableGenome(parent.genome);

  // Planner role: the frozen model proposes; skills/context/harness only.
  const mutation = await options.propose(parent.genome, frozenModel);
  assertEvolvableGenome(mutation.candidate);

  const capabilityExpanding = expandsCapabilities(mutation.capabilityDelta);
  const capabilityGate = capabilityExpanding ? constitutionalGateStub(mutation) : undefined;
  const blockedByGate = capabilityExpanding && !capabilityGate!.allowed;

  // Optimizer role: the SAME frozen instance scores the candidate — unless
  // the constitutional gate already blocked it, in which case the evaluator
  // never runs (evaluated="blocked").
  const candidateSamples = blockedByGate
    ? []
    : [...await options.evaluate(mutation.candidate, frozenModel)];
  const decision = pairedBootstrapDecision([...parent.samples], candidateSamples);

  const report = [
    "# SHAPER generation evaluation",
    `mutation: ${mutation.describe}`,
    `surface: ${mutation.candidate.surface} v${mutation.candidate.version}`,
    `frozen_model: ${frozenModel.id}`,
    `frozen_model_sha256: ${frozenModel.sha256}`,
    `planner_is_optimizer: true`,
  ].join("\n");

  const verdict = await evaluateCandidate({
    finding: mutation.describe,
    report,
    commit,
    decision,
    evaluated: blockedByGate ? "blocked" : "yes",
    ...(options.date ? { date: options.date } : {}),
  });

  const vetoReasons = [...new Set([
    ...vetoesFromVerdict(verdict),
    ...(blockedByGate ? ["capability_expansion_unauthorized"] : []),
  ])].sort();

  return Object.freeze({
    modelSha256: frozenModel.sha256,
    plannerModelId: frozenModel.id,
    optimizerModelId: frozenModel.id,
    mutation,
    verdict,
    vetoReasons: Object.freeze(vetoReasons),
    recommendation: vetoReasons.length === 0 ? "recommend" as const : "blocked" as const,
    ...(capabilityGate ? { capabilityGate } : {}),
  });
}

/**
 * A CandidateProposer over the HARNESS surface that REUSES Darwin's existing
 * GEPA search (darwin.ts runRuvectorGepa — @metaharness/darwin 0.9.x) instead
 * of inventing a new mutator. The best GEPA genome becomes the candidate
 * harness genome; GEPA mutates ANN/runner parameters only — no capability
 * delta, no model files.
 */
export function gepaHarnessProposer(gepa: RuvectorGepaOptions): CandidateProposer {
  return async (parent) => {
    if (parent.surface !== "harness") {
      throw new Error(`gepaHarnessProposer mutates the harness surface, got "${parent.surface}"`);
    }
    const result = await runRuvectorGepa(gepa);
    const best = result.pool.find((candidate) => candidate.id === result.best) ?? result.pool[0];
    if (!best) throw new Error("Darwin GEPA returned an empty candidate pool");
    return {
      parent,
      candidate: {
        surface: "harness",
        version: parent.version + 1,
        parameters: policyFromGenome(best.genome),
      },
      describe: `darwin-gepa harness mutation (candidate ${best.id}, bestMean=${result.bestMean})`,
    };
  };
}
