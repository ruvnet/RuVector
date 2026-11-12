/**
 * Zetta-pattern three-timescale evolution harness — synthetic slice
 * (PIR WP24, ADR-327, issue #883; Zetta = arXiv:2608.16590).
 *
 * ────────────────────────────────────────────────────────────────────────
 * HONEST SCOPE — read before extending. THIS IS A SYNTHETIC-ENVIRONMENT
 * PATTERN SLICE, NOT PHYSICAL ROBOTICS.
 *
 *   - No repo in the ruvnet org operates a physical robot or a robot-rollout
 *     harness today (verified: RVM is an agentic runtime, not robotics;
 *     RuView is RF sensing, never actuation). This module therefore adopts
 *     ONLY the timescale-separated evolution-harness PATTERN over a SYNTHETIC
 *     environment. It drives no hardware, no simulator, no robot.
 *   - The physical / RuView actuation target is EXPLICITLY DEFERRED to a
 *     future ADR + hardware (ADR-327 Decision §5). Nothing here operates, or
 *     is about to operate, a physical robot. Any future real-world path is
 *     sensing-side coordination with RuView only — never actuation.
 *   - This is the timescale-architecture pattern, NOT a reproduction of
 *     Zetta's reported LIBERO-Pro / RoboCasa numbers. Those figures
 *     (90.8% / 93.6% / 11.1x) are the source paper's own measurements on
 *     robot benchmarks this program cannot run; the preprint-reproduction
 *     rule forbids claiming them here (ADR-327 "Preprint-reproduction rule").
 * ────────────────────────────────────────────────────────────────────────
 *
 * Zetta evolves runtime critics and recovery skills online while keeping the
 * base policy FROZEN, across three timescale-separated loops (arXiv:2608.16590
 * abstract, verbatim: "action-frequency governance, rollout-level
 * critic-recovery proposal, and validation-gated skill updates"). This module
 * is a timescale-structured VARIANT of WP9's SHAPER frozen-weight loop
 * (shaperLoop.ts / genome.ts, ADR-313): same frozen `FrozenModelRef`, same
 * structural frozen-weights enforcement (scripts/frozen-weights-check.mjs
 * scans this directory), same ADR-306 evaluation → ADR-315 constitutional
 * boundary → recommendation-only contract. It adds the three-tier separation.
 *
 * THREE TIERS, structurally distinct (ADR-327 Decision §3):
 *
 *   1. ms — LOCAL CRITIC (action-frequency governance). A fast, cheap,
 *      model-free check gating one action: synthetic signal-quality /
 *      staleness / confidence / anomaly critics. `runLocalCriticTier` is a
 *      SYNCHRONOUS function taking only a `SyntheticObservation` — no
 *      `FrozenModelRef` parameter exists in the critic path, and the ONLY
 *      route to the frozen base policy (the async dream-machine evaluation
 *      the minutes tier uses) cannot be reached from a synchronous function.
 *      So the critic provably cannot invoke the frozen model. Test-asserted.
 *
 *   2. seconds — RECOVERY (rollout-level critic-recovery proposal). On a
 *      critic-flagged failure, a local recovery behavior (retry / fallback /
 *      degrade) runs. Also synchronous, also model-free: recovery selects a
 *      behavior over the synthetic env and NEVER evolves or mutates the base.
 *
 *   3. minutes–hours — SKILL EVOLUTION (validation-gated skill updates). A
 *      validated skill/critic mutation is promoted ONLY through WP9's
 *      existing dream-machine → vetoes → flywheel path (`runShaperGeneration`,
 *      reused unchanged), and ONLY after held-out-seed validation: the
 *      candidate is scored on seeds it never trained against (maps to ruv's
 *      Wave-3 acceptance-test held-out requirement and Zetta's own
 *      "Held-out seeds → Reject/Promote" gate).
 *
 * FROZEN-BASE INVARIANT (ADR-327 Decision §2, reuses ADR-313's mechanism):
 * NO tier may mutate the base policy — only critics, recovery, and skills
 * evolve. Enforced STRUCTURALLY, not by policy:
 *   - The ms and seconds tiers never receive a `FrozenModelRef` at all.
 *   - The minutes tier receives the frozen model only as `genome.ts`'s
 *     immutable `FrozenModelRef` (id + loop-start sha256) and routes through
 *     `runShaperGeneration`, whose `MutationSurface` has no "weights" member —
 *     a weight-mutating candidate is unrepresentable.
 *   - This file lives in the frozen-weights-check mutation surface: importing
 *     a training / weight-writing API here fails CI.
 *
 * CONSTITUTIONAL BOUNDARY (ADR-315, adopted verbatim from WP9): a promoted
 * skill/critic that expands capability routes to the ADR-315 constitutional
 * gate stub (blocked by default) via `runShaperGeneration`. This module
 * exports NO promote/merge/approve function; every result is frozen data with
 * no callable members. Promotion belongs to the vetoes/flywheel path plus a
 * human. Test-asserted, identically to shaperLoop.test.ts.
 */
import {
  runShaperGeneration,
  type CandidateProposer,
  type GenomeEvaluator,
  type ShaperGenerationResult,
} from "./shaperLoop.js";
import {
  assertFrozenModelRef,
  type CandidateMutation,
  type EvolvableGenome,
  type FrozenModelRef,
} from "./genome.js";

// ───────────────────────────── ms tier: LOCAL CRITIC ─────────────────────
//
// action-frequency governance. Fast, cheap, model-free per-action gating.

/**
 * One synthetic-environment observation the ms-tier critic inspects. Purely
 * synthetic scalars — this is NOT sensor data from any real device.
 */
export interface SyntheticObservation {
  /** Rollout seed this observation belongs to (synthetic). */
  readonly seed: number;
  /** Step index within the rollout. */
  readonly step: number;
  /** Synthetic signal-quality score in [0, 1]. */
  readonly signalQuality: number;
  /** Synthetic state staleness, milliseconds since last refresh. */
  readonly stateAgeMs: number;
  /** Synthetic policy confidence in [0, 1]. */
  readonly confidence: number;
  /** Synthetic scalar whose magnitude, out of bounds, reads as an anomaly. */
  readonly value: number;
}

/** The four fast critics Zetta names for the immediate-governance tier. */
export type CriticFlag = "signal_quality" | "stale_state" | "low_confidence" | "anomaly";

/** Thresholds parameterising the local critic. */
export interface CriticThresholds {
  readonly minSignalQuality: number;
  readonly maxStateAgeMs: number;
  readonly minConfidence: number;
  /** |value| strictly greater than this reads as an anomaly. */
  readonly anomalyAbs: number;
}

/**
 * The ms-tier verdict. Frozen data. `tier` documents which timescale produced
 * it — a fast local critic, never the frozen base policy.
 */
export interface CriticSignal {
  readonly ok: boolean;
  readonly flags: readonly CriticFlag[];
  readonly tier: "ms-local-critic";
}

/**
 * A local critic: synchronous, and — by TYPE — takes ONLY an observation.
 * There is no `FrozenModelRef` parameter, so the critic cannot be handed the
 * frozen base policy. This is the ms tier's structural frozen-base proof.
 */
export type LocalCritic = (observation: SyntheticObservation) => CriticSignal;

/**
 * Build the default model-free local critic from thresholds. Pure arithmetic —
 * no I/O, no await, no model. Its evolvable form is a `skills`/`context`
 * genome promoted through the minutes tier, never a weight change.
 */
export function defaultLocalCritic(thresholds: CriticThresholds): LocalCritic {
  return (observation) => {
    const flags: CriticFlag[] = [];
    if (observation.signalQuality < thresholds.minSignalQuality) flags.push("signal_quality");
    if (observation.stateAgeMs > thresholds.maxStateAgeMs) flags.push("stale_state");
    if (observation.confidence < thresholds.minConfidence) flags.push("low_confidence");
    if (Math.abs(observation.value) > thresholds.anomalyAbs) flags.push("anomaly");
    return Object.freeze({
      ok: flags.length === 0,
      flags: Object.freeze(flags),
      tier: "ms-local-critic" as const,
    });
  };
}

/**
 * Runtime guard: a local critic must be model-free. A critic that declared a
 * second parameter (e.g. to smuggle in a `FrozenModelRef`) is rejected. The
 * type system already forbids it; this backstops untyped JS callers.
 */
export function assertModelFreeCritic(critic: LocalCritic): void {
  if (typeof critic !== "function") throw new Error("local critic must be a function");
  if (critic.length > 1) {
    throw new Error(
      `local critic must take ONLY an observation (arity ${critic.length}) — ` +
      "the ms tier is model-free; the frozen base policy is unreachable here",
    );
  }
}

/** The ms tier's decision for one action. */
export interface GatedAction {
  readonly observation: SyntheticObservation;
  readonly signal: CriticSignal;
  /** "proceed" when the critic is clean, "gated" when any critic flagged. */
  readonly action: "proceed" | "gated";
}

/**
 * ms tier — gate ONE action with the local critic. DELIBERATELY SYNCHRONOUS:
 * the only path to the frozen base policy is the async dream-machine
 * evaluation the minutes tier uses (`runShaperGeneration` → `evaluateCandidate`
 * → dream-machine `run`, all async). A synchronous function cannot await that
 * path, so this tier provably cannot reach the frozen model — the structural
 * frozen-base proof for the fast critic (test-asserted: the return value is a
 * plain object, never a thenable).
 */
export function runLocalCriticTier(critic: LocalCritic, observation: SyntheticObservation): GatedAction {
  assertModelFreeCritic(critic);
  const signal = critic(observation);
  return Object.freeze({
    observation,
    signal,
    action: signal.ok ? "proceed" : "gated",
  });
}

// ───────────────────────────── seconds tier: RECOVERY ────────────────────
//
// rollout-level critic-recovery. Local behavior selection; base never evolves.

/** The recovery behaviors this synthetic tier can select (no base change). */
export type RecoveryStrategy = "retry" | "fallback" | "degrade";

/** The seconds-tier outcome. Frozen data; no model, no base mutation. */
export interface RecoveryOutcome {
  readonly strategy: RecoveryStrategy;
  /** Which critic flags this recovery behavior addresses. */
  readonly resolvedFlags: readonly CriticFlag[];
  readonly recovered: boolean;
  readonly tier: "seconds-recovery";
}

/**
 * Map a critic flag to a recovery behavior. Pure. Staleness → retry (refresh
 * state); signal/confidence → fallback (a safer known behavior); anomaly →
 * degrade (reduce authority). None of these change the base policy.
 */
export function recoveryStrategyForFlags(flags: readonly CriticFlag[]): RecoveryStrategy {
  if (flags.includes("anomaly")) return "degrade";
  if (flags.includes("stale_state")) return "retry";
  return "fallback";
}

/**
 * seconds tier — run a recovery behavior for a critic-flagged action. Also
 * SYNCHRONOUS and model-free: it selects and records a local behavior, and
 * has no `FrozenModelRef` parameter and no genome-mutation path. It NEVER
 * evolves the base — recovery is a runtime behavior, not a base change.
 *
 * A clean (un-flagged) action needs no recovery and throws if passed here:
 * the seconds tier fires only on a critic-flagged failure.
 */
export function runRecoveryTier(gated: GatedAction): RecoveryOutcome {
  if (gated.action !== "gated") {
    throw new Error("runRecoveryTier is only invoked on a critic-flagged (gated) action");
  }
  const flags = gated.signal.flags;
  const strategy = recoveryStrategyForFlags(flags);
  return Object.freeze({
    strategy,
    resolvedFlags: Object.freeze([...flags]),
    // Synthetic recovery model: a single local behavior clears the flagged
    // failure. This is a pattern stub over a synthetic env, not a physical
    // recovery guarantee.
    recovered: true,
    tier: "seconds-recovery",
  });
}

// ──────────────────── minutes–hours tier: SKILL EVOLUTION ─────────────────
//
// validation-gated skill updates via WP9's flywheel path + held-out seeds.

/**
 * A train / held-out seed split. Promotion is validated on seeds the
 * evolution loop NEVER trained against (ADR-327 §3 "Held-out seeds", ruv's
 * Wave-3 held-out acceptance requirement).
 */
export interface HeldOutSplit {
  /** Seeds the proposer learns the mutation from (the failure cluster). */
  readonly trainSeeds: readonly number[];
  /** Disjoint seeds the candidate is validated on but never trained against. */
  readonly heldOutSeeds: readonly number[];
}

/**
 * A seed split must be non-empty on both sides AND disjoint — a candidate
 * validated on a seed it trained on is not held-out validation. Structural.
 */
export function assertDisjointSplit(split: HeldOutSplit): void {
  if (split.trainSeeds.length === 0) throw new Error("held-out split requires at least one train seed");
  if (split.heldOutSeeds.length === 0) throw new Error("held-out split requires at least one held-out seed");
  const trainSet = new Set(split.trainSeeds);
  const leaked = split.heldOutSeeds.filter((seed) => trainSet.has(seed));
  if (leaked.length > 0) {
    throw new Error(
      `held-out seeds leaked into training: [${leaked.join(", ")}] — ` +
      "the candidate must be validated only on seeds it never trained against",
    );
  }
}

/** Proposes a skill/critic mutation from the failure cluster's TRAIN seeds. */
export type SkillProposer = (
  parent: EvolvableGenome,
  plannerModel: FrozenModelRef,
  trainSeeds: readonly number[],
) => CandidateMutation | Promise<CandidateMutation>;

/** Scores a candidate over the HELD-OUT seeds — the seeds it never trained on. */
export type HeldOutEvaluator = (
  genome: EvolvableGenome,
  optimizerModel: FrozenModelRef,
  heldOutSeeds: readonly number[],
) => readonly number[] | Promise<readonly number[]>;

export interface SkillEvolutionOptions {
  /** The frozen base policy — immutable, id + loop-start sha256 (genome.ts). */
  frozenModel: FrozenModelRef;
  /** Champion genome and its per-held-out-seed baseline samples. */
  parent: { genome: EvolvableGenome; samples: readonly number[] };
  /** Disjoint train / held-out seed split. */
  split: HeldOutSplit;
  /** Proposes the mutation; sees ONLY the train seeds. */
  propose: SkillProposer;
  /** Validates the mutation; scores ONLY on held-out seeds. */
  evaluateHeldOut: HeldOutEvaluator;
  /** Git commit sha the evaluation runs against (witness-stamped). */
  commit: string;
  /** Evaluation date, YYYY-MM-DD (defaults to today, UTC). */
  date?: string;
}

/**
 * The minutes-tier outcome. Frozen data — a RECOMMENDATION, never a
 * promotion. Wraps WP9's `ShaperGenerationResult` unchanged and adds the
 * held-out validation record.
 */
export interface SkillEvolutionResult {
  readonly tier: "minutes-skill-evolution";
  /** WP9's generation result: verdict, veto reasons, capability gate. */
  readonly generation: ShaperGenerationResult;
  readonly heldOut: {
    readonly trainSeeds: readonly number[];
    readonly heldOutSeeds: readonly number[];
    /** The candidate cleared the flywheel gate on held-out seeds. */
    readonly validatedOnHeldOut: boolean;
  };
  /** "recommend" only when the flywheel path recommends on held-out seeds. */
  readonly recommendation: "recommend" | "blocked";
}

/**
 * minutes–hours tier — validate a skill/critic mutation and produce a
 * promotion RECOMMENDATION, reusing WP9's `runShaperGeneration` verbatim.
 *
 * The proposer sees ONLY the train seeds; the evaluator scores ONLY on the
 * disjoint held-out seeds. `parent.samples` are the baseline over those same
 * held-out seeds, so the paired-bootstrap compares candidate-on-held-out
 * against baseline-on-held-out — a mutation validated on data it did not
 * train on. A capability-expanding mutation is routed to the ADR-315
 * constitutional gate by `runShaperGeneration` and blocked by default; the
 * base policy is never mutated (genome.ts has no weights surface).
 */
export async function runSkillEvolutionTier(options: SkillEvolutionOptions): Promise<SkillEvolutionResult> {
  assertFrozenModelRef(options.frozenModel);
  assertDisjointSplit(options.split);
  if (options.parent.samples.length !== options.split.heldOutSeeds.length) {
    throw new Error(
      `parent baseline samples (${options.parent.samples.length}) must be measured over the ` +
      `held-out seeds (${options.split.heldOutSeeds.length}) — held-out validation compares like for like`,
    );
  }

  const seenTrainSeeds: number[][] = [];
  const seenEvalSeeds: number[][] = [];

  // Bind WP9's proposer contract to the TRAIN seeds only.
  const propose: CandidateProposer = async (parent, plannerModel) => {
    seenTrainSeeds.push([...options.split.trainSeeds]);
    return options.propose(parent, plannerModel, options.split.trainSeeds);
  };

  // Bind WP9's evaluator contract to the HELD-OUT seeds only.
  const evaluate: GenomeEvaluator = async (genome, optimizerModel) => {
    seenEvalSeeds.push([...options.split.heldOutSeeds]);
    return options.evaluateHeldOut(genome, optimizerModel, options.split.heldOutSeeds);
  };

  const generation = await runShaperGeneration({
    frozenModel: options.frozenModel,
    parent: options.parent,
    propose,
    evaluate,
    commit: options.commit,
    ...(options.date ? { date: options.date } : {}),
  });

  // Belt-and-braces: the evaluator must never have seen a train seed.
  for (const evalSeeds of seenEvalSeeds) {
    const trainSet = new Set(options.split.trainSeeds);
    if (evalSeeds.some((seed) => trainSet.has(seed))) {
      throw new Error("held-out validation invariant broken: evaluator saw a train seed");
    }
  }

  const validatedOnHeldOut = generation.recommendation === "recommend";
  return Object.freeze({
    tier: "minutes-skill-evolution",
    generation,
    heldOut: Object.freeze({
      trainSeeds: Object.freeze([...options.split.trainSeeds]),
      heldOutSeeds: Object.freeze([...options.split.heldOutSeeds]),
      validatedOnHeldOut,
    }),
    recommendation: generation.recommendation,
  });
}

// ─────────────────────────── three-tier orchestration ────────────────────

/** One synthetic rollout: an ordered stream of observations for a seed. */
export interface SyntheticRollout {
  readonly seed: number;
  readonly observations: readonly SyntheticObservation[];
}

export interface TimescaleGenerationOptions {
  /** The frozen base policy (only the minutes tier ever receives it). */
  frozenModel: FrozenModelRef;
  /** The ms-tier local critic (model-free). */
  critic: LocalCritic;
  /** Synthetic rollouts to run the fast critic + recovery over. */
  rollouts: readonly SyntheticRollout[];
  /** Champion genome + baseline over held-out seeds (minutes tier). */
  parent: { genome: EvolvableGenome; samples: readonly number[] };
  /** Train / held-out seed split for held-out validation. */
  split: HeldOutSplit;
  propose: SkillProposer;
  evaluateHeldOut: HeldOutEvaluator;
  commit: string;
  date?: string;
}

/** The tier order, recorded so the ordering is asserted, not assumed. */
export const TIER_ORDER = ["ms-local-critic", "seconds-recovery", "minutes-skill-evolution"] as const;
export type TierOrder = typeof TIER_ORDER;

/**
 * The whole generation's outcome. Frozen data, RECOMMENDATION only — there is
 * no promote(), no merge(), no side-effect handle. Promotion belongs to the
 * vetoes/flywheel path plus a human (WP2/WP9 contract, adopted unchanged).
 */
export interface TimescaleGenerationResult {
  readonly ms: {
    readonly gated: readonly GatedAction[];
    readonly flaggedCount: number;
  };
  readonly seconds: {
    readonly recoveries: readonly RecoveryOutcome[];
  };
  readonly minutes: SkillEvolutionResult;
  /** Proof the three tiers ran, in Zetta's timescale order. */
  readonly tiersRunInOrder: TierOrder;
  /** The minutes-tier recommendation is the generation's recommendation. */
  readonly recommendation: "recommend" | "blocked";
}

/**
 * Run ONE three-timescale generation over a synthetic environment, in order:
 *
 *   ms local critic (gate every action) → seconds recovery (per flagged
 *   failure) → minutes–hours skill evolution (validate on held-out seeds via
 *   the flywheel gate).
 *
 * The three tiers are independently governed (ADR-327 "independent per-tier
 * gating"): the ms critic gates actions, the seconds tier recovers flagged
 * failures, and NEITHER promotes anything — only the minutes tier yields a
 * promotion recommendation, and only through WP9's flywheel path with
 * held-out validation. The frozen base policy is threaded to the minutes tier
 * ALONE; the fast tiers never see it.
 */
export async function runTimescaleGeneration(
  options: TimescaleGenerationOptions,
): Promise<TimescaleGenerationResult> {
  assertFrozenModelRef(options.frozenModel);
  assertModelFreeCritic(options.critic);

  // Tier 1 (ms): gate every action with the fast, model-free local critic.
  const gated: GatedAction[] = [];
  for (const rollout of options.rollouts) {
    for (const observation of rollout.observations) {
      gated.push(runLocalCriticTier(options.critic, observation));
    }
  }
  const flagged = gated.filter((g) => g.action === "gated");

  // Tier 2 (seconds): run recovery for each critic-flagged failure. Model-free;
  // the base is never evolved here.
  const recoveries = flagged.map((g) => runRecoveryTier(g));

  // Tier 3 (minutes–hours): the failure cluster drives a validated skill
  // mutation, promoted ONLY through the flywheel gate on held-out seeds. This
  // is the ONLY tier that receives the frozen base policy.
  const minutes = await runSkillEvolutionTier({
    frozenModel: options.frozenModel,
    parent: options.parent,
    split: options.split,
    propose: options.propose,
    evaluateHeldOut: options.evaluateHeldOut,
    commit: options.commit,
    ...(options.date ? { date: options.date } : {}),
  });

  return Object.freeze({
    ms: Object.freeze({ gated: Object.freeze(gated), flaggedCount: flagged.length }),
    seconds: Object.freeze({ recoveries: Object.freeze(recoveries) }),
    minutes,
    tiersRunInOrder: TIER_ORDER,
    recommendation: minutes.recommendation,
  });
}
