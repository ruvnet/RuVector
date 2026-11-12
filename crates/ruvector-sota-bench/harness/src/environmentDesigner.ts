/**
 * SPADE-pattern self-play Environment Designer — first shippable slice
 * (PIR WP21, ADR-324, issue #880).
 *
 * Source paper, cited in full on EVERY reference (naming discipline is binding,
 * ADR-324 §5): SPADE (arXiv:2608.19197, Self-Play in Adaptive Synthetic
 * Executable Environments). NEVER the bare word — it collides, in the same
 * multi-agent domain, with the long-established Smart Python multi-Agent
 * Development Environment framework.
 *
 * A SECOND, structurally different self-supplied Darwin mutation-candidate
 * SOURCE feeding ADR-313's SHAPER loop (shaperLoop.ts / genome.ts, WP9),
 * sibling to ADR-321's synthetic-issue source (skillForge.ts, WP19). Where
 * SkillForge synthesizes point regressions from existing test coverage, the
 * Environment Designer generates whole sandboxed EXECUTABLE environments —
 * training curricula with a Gym-style reset()/step()/reward/verification
 * contract, per SPADE (arXiv:2608.19197)'s confirmed mechanism — whose
 * difficulty a regret signal keeps at the frontier of the learner's capability:
 *
 *   GENERATE env (FROM external evidence) ──► MEASURE regret (normal vs
 *        privileged attempt) ──► ASSESS frontier (solvable ∧ gap-exposing)
 *        ──► feed the EXISTING dream-machine eval → vetoes → flywheel path
 *
 * THE HARD INVARIANT (ADR-324 §4, the load-bearing gate). Every generated
 * environment MUST trace to external evidence — a repo test, a real observed
 * failure, a RuVector memory of a real observation, or an independently
 * published benchmark. A closed, self-invented environment (the designer's only
 * input is its own prior generations) is a REJECT, not a warning. This is
 * enforced as a HARD VETO in the promotion path — `externalGroundingVetoProvider`
 * — composed via vetoes.ts's `composeVetoProviders` alongside the existing
 * dream-machine and HarnessRisk providers, and fired BEFORE dream-machine's
 * evaluation stage runs (`groundingGatedVetoProvider`), never after: an
 * ungrounded environment never reaches evaluation at all. This is this
 * program's own closed-epistemic-loop safeguard AND a direct generalization of
 * SPADE (arXiv:2608.19197)'s own ablation finding — external corpus grounding
 * "critical to success" (ADR-324 §Context; the "closed-epistemic-loop" framing
 * is this program's, NOT the paper's).
 *
 * PREPRINT-REPRODUCTION RULE (ADR-324 §Context, evidence grade A): the paper's
 * numbers (+5.3 / +5.7 / +13.9, at 30B parameters, on its own suite) are
 * HYPOTHESES, not this program's bar. No performance magnitude is quoted here;
 * the only citable delta is this program's own research-gate-measured one.
 *
 * CONSTITUTIONAL BOUNDARY — identical to skillForge.ts / shaperLoop.ts /
 * dreamMachine.ts: this module PRODUCES candidate environments; it does NOT
 * promote them. A generated environment is a proposal
 * (`EnvironmentProposal.kind === "environment-proposal"`). There is no
 * promote/merge/apply function, every result is frozen data with no callable
 * members, and promotion stays with the WP9 loop → vetoes → flywheel path plus
 * a human. An environment that would certify a capability EXPANSION is routed
 * to ADR-315's constitutional gate stub (genome.ts) and is BLOCKED by default.
 *
 * FROZEN WEIGHTS ARE STRUCTURAL (ADR-324 §Security): this file lives inside the
 * mutation-surface directory scripts/frozen-weights-check.mjs scans. It imports
 * no training / weight-writing API and names no model weight file. Environment
 * generation and any agent play against a generated environment are a
 * harness/curriculum mutation, never a weight update.
 *
 * STUBBED vs REAL in this slice (honest scope, ADR-324 §Consequences): the live
 * 30B self-play, the real reasoning-agent rollout, real external-evidence
 * resolution (filesystem / RuVector / benchmark lookup — modelled by the
 * injected `EvidenceResolver`), and the empirical acceptance run are ALL
 * deferred. This slice is the designer + grounding-gate + regret-signal.
 */
import { createHash } from "node:crypto";
import {
  constitutionalGateStub,
  expandsCapabilities,
  type CandidateMutation,
  type CapabilityDelta,
  type ConstitutionalDecision,
  type ContextGenome,
} from "./genome.js";
import type { PairedDecision } from "./statistics.js";
import { pairedBootstrapDecision } from "./statistics.js";
import type { DreamCandidate } from "./dreamMachine.js";
import type { PromotionVetoProvider, VetoContext } from "./vetoes.js";

/** Exact ADR-324 citation. No performance magnitude (preprint-reproduction rule). */
export const SPADE_CITATION =
  "SPADE (arXiv:2608.19197, Self-Play in Adaptive Synthetic Executable Environments)" as const;

const HEX64 = /^[0-9a-f]{64}$/;

// ---------------------------------------------------------------------------
// EXTERNAL EVIDENCE — the ONLY admissible grounding roots (ADR-324 §4).
// ---------------------------------------------------------------------------

/**
 * The four external-evidence kinds ADR-324 §4 admits. Deliberately closed:
 * nothing self-generated by the designer is on this list — that is the whole
 * point of the invariant.
 */
export type ExternalEvidenceKind =
  | "repo-test"
  | "observed-failure"
  | "ruvector-memory"
  | "published-benchmark";

const EVIDENCE_KINDS: readonly ExternalEvidenceKind[] = [
  "repo-test",
  "observed-failure",
  "ruvector-memory",
  "published-benchmark",
];

/**
 * A concrete, resolvable trace to an external source. `locator` is the
 * resolvable handle (a repo-relative test path, an incident id, a RuVector
 * memory record id, a benchmark citation). `contentSha256` pins the source's
 * content at generation time so drift is detectable (integrity anchor, mirrors
 * skillForge's entity-attachment hash, ADR-324 §Security). A well-formed shape
 * is necessary but NOT sufficient — resolution against a real external store
 * (the injected `EvidenceResolver`) is what proves it is not a citation of
 * convenience.
 */
export interface ExternalEvidence {
  readonly kind: ExternalEvidenceKind;
  readonly locator: string;
  readonly contentSha256: string;
  readonly description: string;
}

/** Validate an evidence record's SHAPE. Resolution is a separate, harder step. */
export function assertExternalEvidence(evidence: ExternalEvidence): void {
  if (!EVIDENCE_KINDS.includes(evidence.kind)) {
    throw new Error(`external evidence kind must be one of ${EVIDENCE_KINDS.join(", ")}`);
  }
  if (!evidence.locator) {
    throw new Error("external evidence requires a non-empty locator (the resolvable external handle)");
  }
  if (!HEX64.test(evidence.contentSha256)) {
    throw new Error("external evidence contentSha256 must be 64 lowercase hex chars (integrity anchor)");
  }
}

/**
 * Where a generated environment came from. `evidence` is the external-grounding
 * trace: EMPTY means a self-invented closed loop — representable (so the veto
 * can be proven to reject it), never promotable. `designerNote` is the
 * designer's own one-line rationale (self-reported, NOT trusted as grounding).
 */
export interface EnvironmentProvenance {
  readonly evidence: readonly ExternalEvidence[];
  readonly designerNote: string;
  readonly citation: typeof SPADE_CITATION;
}

// ---------------------------------------------------------------------------
// THE EXECUTABLE ENVIRONMENT — Gym-style reset()/step()/reward/verification.
// ---------------------------------------------------------------------------

/** An observation handed to the reasoning agent. `privileged` exposes the key. */
export interface EnvObservation {
  readonly prompt: string;
  readonly privileged: boolean;
}

/** Immutable environment state (event-sourced: step() returns a fresh state). */
export interface EnvState {
  readonly observation: EnvObservation;
  readonly solved: boolean;
  readonly reward: number;
}

/** The result of one step(): the next state, its reward, and terminality. */
export interface EnvTransition {
  readonly state: EnvState;
  readonly reward: number;
  readonly done: boolean;
}

/**
 * The typed EvaluationEnvironment interface (SPADE (arXiv:2608.19197)'s
 * Gym-style contract). Pure and deterministic in this slice — no real sandbox
 * process is spawned; a generated environment is data + closures. `verify` is
 * the success predicate; `reward` the scalar signal.
 */
export interface EvaluationEnvironment {
  readonly id: string;
  readonly provenance: EnvironmentProvenance;
  /** `privileged` mode reveals the verification key in the observation. */
  reset(options?: { readonly privileged?: boolean }): EnvState;
  step(state: EnvState, action: string): EnvTransition;
  reward(state: EnvState): number;
  verify(state: EnvState): boolean;
}

/**
 * The verification key an environment checks an action against — the SECOND,
 * defense-in-depth grounding layer (the resolver-backed veto is the primary
 * gate). The key is bound to the RESOLVER's CURRENT resolved content hash
 * (`currentSha256`), NOT to the self-declared `contentSha256`, so it is
 * genuinely UNCOMPUTABLE without access to the real external source:
 *   - the resolver must run (you must be able to resolve the evidence), and
 *   - the key mixes in the content hash the resolver returns from the real
 *     source, which a designer holding only fabricated strings does not have.
 *
 * Returns `null` — an UNUSABLE key, never a fixed public constant — when the
 * provenance has NO evidence (self-invented loop) or when ANY evidence fails to
 * resolve (forged / invented cite). An environment with a null key is
 * intrinsically UNSOLVABLE: no action can satisfy it. So a designer with no
 * real external anchor cannot construct a solvable challenge at all — grounding
 * is load-bearing at the environment mechanic itself, not only at the veto.
 * (Requiring EVERY evidence to resolve mirrors the veto: a real cite cannot
 * launder a forged one alongside it.)
 */
export async function verificationKeyFor(
  provenance: EnvironmentProvenance,
  resolve: EvidenceResolver,
): Promise<string | null> {
  if (provenance.evidence.length === 0) return null;
  const hash = createHash("sha256").update("ruvector.spade.env-key.v1\0");
  for (const item of provenance.evidence) {
    const outcome = await resolve(item);
    if (!outcome.resolved || outcome.currentSha256 === undefined) return null;
    // Bind to the resolver's CURRENT content hash from the real source, never
    // to the designer's self-declared item.contentSha256.
    hash.update(`${item.kind}\0${item.locator}\0${outcome.currentSha256}\0`);
  }
  return hash.digest("hex").slice(0, 16);
}

/** The declarative spec the designer emits; `generateEnvironment` executes it. */
export interface EnvironmentSpec {
  /** Stable id (e.g. "env-<hash>"); must be non-empty. */
  readonly id: string;
  /** One-line description of the capability the environment exercises. */
  readonly task: string;
  readonly provenance: EnvironmentProvenance;
  /** Non-empty ⇒ certifying with this environment would EXPAND capability. */
  readonly capabilityDelta?: CapabilityDelta;
}

/**
 * A generated environment: the executable interface plus its originating spec.
 * `verificationKey` is `null` when the provenance did not resolve to real
 * external content (self-invented or forged) — such an environment is
 * intrinsically unsolvable (the second grounding layer, see `verificationKeyFor`).
 */
export interface GeneratedEnvironment {
  readonly spec: EnvironmentSpec;
  readonly environment: EvaluationEnvironment;
  readonly verificationKey: string | null;
}

function buildObservation(spec: EnvironmentSpec, key: string | null, privileged: boolean): EnvObservation {
  const base =
    `[${SPADE_CITATION}] Task: ${spec.task}. ` +
    `Grounded in ${spec.provenance.evidence.length} external-evidence source(s). ` +
    `Emit the verification key to solve.`;
  // Only a resolver-backed key can be revealed; an unresolved (null) key leaves
  // even the privileged observation with no solvable answer.
  return Object.freeze({
    prompt: privileged && key !== null ? `${base} PRIVILEGED HINT — key=${key}` : base,
    privileged,
  });
}

/**
 * Build an executable EvaluationEnvironment from a spec, threading the same
 * `resolve` the grounding veto uses. Does NOT throw on empty / unresolved
 * evidence: a self-invented environment is representable so the grounding veto
 * can be PROVEN to reject it (the invariant is a promotion-path veto, ADR-324
 * §4, not a constructor guard) — but its `verificationKey` is `null`, making it
 * intrinsically UNSOLVABLE (defense-in-depth). It validates the shape of any
 * evidence present, and the env id/task.
 */
export async function generateEnvironment(
  spec: EnvironmentSpec,
  resolve: EvidenceResolver,
): Promise<GeneratedEnvironment> {
  if (!spec.id) throw new Error("environment spec requires a non-empty id");
  if (!spec.task) throw new Error("environment spec requires a non-empty task");
  if (spec.provenance.citation !== SPADE_CITATION) {
    throw new Error("environment provenance must carry the exact SPADE (arXiv:2608.19197) citation");
  }
  for (const item of spec.provenance.evidence) assertExternalEvidence(item);

  // Resolver-bound: null when the provenance does not resolve to real external
  // content — an unusable key, never a fixed public constant.
  const key = await verificationKeyFor(spec.provenance, resolve);

  const environment: EvaluationEnvironment = Object.freeze({
    id: spec.id,
    provenance: spec.provenance,
    reset(options?: { readonly privileged?: boolean }): EnvState {
      const privileged = options?.privileged ?? false;
      return Object.freeze({
        observation: buildObservation(spec, key, privileged),
        solved: false,
        reward: 0,
      });
    },
    step(state: EnvState, action: string): EnvTransition {
      // A null (unresolved) key is unsatisfiable: no action solves it.
      const solved = key !== null && action === key;
      const reward = solved ? 1 : 0;
      return Object.freeze({
        state: Object.freeze({ observation: state.observation, solved, reward }),
        reward,
        done: true,
      });
    },
    reward(state: EnvState): number {
      return state.reward;
    },
    verify(state: EnvState): boolean {
      return state.solved;
    },
  });

  return Object.freeze({ spec, environment, verificationKey: key });
}

// ---------------------------------------------------------------------------
// REGRET — difficulty follows the learner (SPADE (arXiv:2608.19197) §self-play).
// ---------------------------------------------------------------------------

/**
 * A reasoning agent: given an environment and whether it plays with privileged
 * evidence, it runs its own rollout and returns the total reward. Injected in
 * this slice — the real frozen-model reasoning path is deferred.
 */
export type ReasoningAgent = (
  environment: EvaluationEnvironment,
  options: { readonly privileged: boolean },
) => number | Promise<number>;

/**
 * A first-party reference agent: reads the verification key straight from the
 * observation when it is present (privileged play) and otherwise cannot solve.
 * On a fresh grounded environment this yields regret ≈ 1 — the maximal signal.
 * Deterministic and dependency-free; a stand-in for the deferred real agent.
 */
export const evidenceReadingAgent: ReasoningAgent = (environment, options) => {
  const state = environment.reset({ privileged: options.privileged });
  const match = /key=([0-9a-f]{16})/.exec(state.observation.prompt);
  const action = match?.[1] ?? "";
  return environment.step(state, action).reward;
};

/** The regret measurement: paired normal-vs-privileged rewards + their gap. */
export interface RegretMeasurement {
  readonly normalReward: number;
  readonly privilegedReward: number;
  /** privilegedReward − normalReward — the difficulty/regret signal (ADR-324 §2). */
  readonly regret: number;
  /** Per-episode samples, so the EXISTING paired-bootstrap statistics apply. */
  readonly normalSamples: readonly number[];
  readonly privilegedSamples: readonly number[];
}

/**
 * Measure regret by having the agent attempt the environment twice: once
 * normally, once with privileged evidence. The reward difference IS the
 * difficulty/regret signal that keeps difficulty tracking the learner (ADR-324
 * §2, ruv's concrete implementation note on #880). `episodes` produces the
 * paired sample arrays statistics.ts consumes; deterministic in this slice, so
 * samples are constant — stochastic rollouts are deferred.
 */
export async function measureRegret(
  environment: EvaluationEnvironment,
  agent: ReasoningAgent = evidenceReadingAgent,
  episodes = 5,
): Promise<RegretMeasurement> {
  if (!Number.isInteger(episodes) || episodes < 2) {
    throw new Error("measureRegret requires at least 2 episodes for paired statistics");
  }
  const normalSamples: number[] = [];
  const privilegedSamples: number[] = [];
  for (let index = 0; index < episodes; index += 1) {
    normalSamples.push(await agent(environment, { privileged: false }));
    privilegedSamples.push(await agent(environment, { privileged: true }));
  }
  const mean = (values: readonly number[]) => values.reduce((s, v) => s + v, 0) / values.length;
  const normalReward = mean(normalSamples);
  const privilegedReward = mean(privilegedSamples);
  return Object.freeze({
    normalReward,
    privilegedReward,
    regret: privilegedReward - normalReward,
    normalSamples: Object.freeze(normalSamples),
    privilegedSamples: Object.freeze(privilegedSamples),
  });
}

// ---------------------------------------------------------------------------
// EXTERNAL-GROUNDING VETO — the crux (ADR-324 §4). HARD, fires before eval.
// ---------------------------------------------------------------------------

/**
 * Resolve one external-evidence record against a REAL external source,
 * returning whether it resolved and the source's CURRENT content hash (for
 * drift detection). There is deliberately NO self-certifying default: a
 * resolver that always returns `resolved: true` would BE the closed epistemic
 * loop ADR-324 forbids. Real filesystem/RuVector/benchmark resolution is
 * deferred; `staticEvidenceResolver` is the first-party stand-in.
 */
export type EvidenceResolver = (
  evidence: ExternalEvidence,
) => { readonly resolved: boolean; readonly currentSha256?: string }
  | Promise<{ readonly resolved: boolean; readonly currentSha256?: string }>;

/**
 * A resolver that confirms evidence against a KNOWN external record set —
 * matching (kind, locator) and reporting that record's current hash. Forged or
 * invented evidence (a locator not in the set) does not resolve. Models "the
 * evidence must exist in an external store", never "the designer says so".
 */
export function staticEvidenceResolver(known: readonly ExternalEvidence[]): EvidenceResolver {
  const index = new Map(known.map((item) => [`${item.kind}\0${item.locator}`, item.contentSha256]));
  return (evidence) => {
    const currentSha256 = index.get(`${evidence.kind}\0${evidence.locator}`);
    return currentSha256 === undefined
      ? { resolved: false }
      : { resolved: true, currentSha256 };
  };
}

/**
 * Grounding veto reasons for one environment's provenance. Empty ⇒ grounded.
 * Fail-closed and STRICT: the environment is grounded only if it has at least
 * one evidence record AND every record resolves cleanly with no drift — a real
 * citation cannot launder a forged one alongside it (ADR-324 §Negative:
 * "superficial citation of convenience").
 */
async function groundingReasons(
  provenance: EnvironmentProvenance,
  resolve: EvidenceResolver,
): Promise<string[]> {
  const reasons: string[] = [];
  if (provenance.evidence.length === 0) {
    reasons.push("external_grounding_absent");
    return reasons;
  }
  for (const item of provenance.evidence) {
    if (!HEX64.test(item.contentSha256) || !item.locator || !EVIDENCE_KINDS.includes(item.kind)) {
      reasons.push("external_grounding_malformed");
      continue;
    }
    const outcome = await resolve(item);
    if (!outcome.resolved) {
      reasons.push("external_grounding_unresolved");
    } else if (outcome.currentSha256 !== undefined && outcome.currentSha256 !== item.contentSha256) {
      reasons.push("external_grounding_drift");
    }
  }
  return [...new Set(reasons)].sort();
}

/**
 * A PromotionVetoProvider (vetoes.ts) that REJECTS any environment whose
 * provenance does not resolve to a concrete external-evidence source. Compose
 * it into vetoes.ts's `composeVetoProviders` set alongside the dream-machine
 * and HarnessRisk providers — conjunctive, it can only ever say "no" or "no
 * objection". `environmentFor` maps the veto context to the environment under
 * evaluation, mirroring dreamMachineVetoProvider's `candidateFor` seam.
 */
export function externalGroundingVetoProvider(
  environmentFor: (context: VetoContext) => GeneratedEnvironment | Promise<GeneratedEnvironment>,
  resolve: EvidenceResolver,
): PromotionVetoProvider {
  return async (context) => {
    const generated = await environmentFor(context);
    return groundingReasons(generated.spec.provenance, resolve);
  };
}

/**
 * The structural realization of "the veto fires BEFORE Dream Machine's
 * evaluation stage runs, not after" (ADR-324 §4, and its rejected alternative).
 * Runs the grounding veto FIRST; if it objects, returns those reasons and the
 * `downstream` provider (in production, dreamMachineVetoProvider — the
 * evaluation stage) is NEVER invoked. An ungrounded environment thus never
 * reaches evaluation at all, rather than being evaluated and then discarded.
 */
export function groundingGatedVetoProvider(
  environmentFor: (context: VetoContext) => GeneratedEnvironment | Promise<GeneratedEnvironment>,
  resolve: EvidenceResolver,
  downstream: PromotionVetoProvider,
): PromotionVetoProvider {
  const grounding = externalGroundingVetoProvider(environmentFor, resolve);
  return async (context) => {
    const groundingVetoes = await grounding(context);
    if (groundingVetoes.length > 0) return groundingVetoes; // short-circuit: eval never runs.
    return [...new Set(await downstream(context))].sort();
  };
}

// ---------------------------------------------------------------------------
// FRONTIER — Darwin promotes environments that stay solvable while exposing a
// capability gap (ADR-324 §3).
// ---------------------------------------------------------------------------

/** Thresholds for the frontier assessment (learner-relative, ADR-324 §2/§3). */
export interface FrontierThresholds {
  /** Privileged reward at/above which the environment is "solvable in principle". */
  readonly solvable: number;
  /** Regret at/above which the environment "exposes a capability gap". */
  readonly gap: number;
}

export const DEFAULT_FRONTIER_THRESHOLDS: FrontierThresholds = { solvable: 0.5, gap: 0.1 };

/**
 * Whether a generated environment lives at the learner's frontier: it must be
 * (a) solvable — the agent can solve it WITH privileged evidence — AND (b)
 * gap-exposing — meaningfully harder (regret) WITHOUT it. An environment the
 * learner already masters (regret ≈ 0) or cannot solve even with evidence
 * (regret saturated low) is NOT at the frontier and is a retirement candidate,
 * not a promotion candidate (ADR-324 §2/§3). Grounding is adjudicated
 * separately by the veto in the promotion path — a non-frontier verdict here is
 * about informativeness, not provenance.
 */
export interface FrontierAssessment {
  readonly atFrontier: boolean;
  readonly solvable: boolean;
  readonly exposesGap: boolean;
  readonly regret: number;
  readonly reasons: readonly string[];
}

export function atFrontier(
  measurement: RegretMeasurement,
  thresholds: FrontierThresholds = DEFAULT_FRONTIER_THRESHOLDS,
): FrontierAssessment {
  const solvable = measurement.privilegedReward >= thresholds.solvable;
  const exposesGap = measurement.regret >= thresholds.gap;
  const reasons: string[] = [];
  if (!solvable) reasons.push("not_solvable_with_privileged_evidence");
  if (!exposesGap) reasons.push(measurement.regret <= 0 ? "mastered_no_regret_signal" : "regret_below_frontier_threshold");
  return Object.freeze({
    atFrontier: solvable && exposesGap,
    solvable,
    exposesGap,
    regret: measurement.regret,
    reasons: Object.freeze(reasons),
  });
}

// ---------------------------------------------------------------------------
// PROPOSAL — a candidate, never a promotion (constitutional boundary).
// ---------------------------------------------------------------------------

/**
 * Build the ADR-313 `CandidateMutation` that would add a generated environment
 * to a parent CONTEXT genome (a generated environment is curriculum context —
 * ADR-313 §1's "context" surface — so the closed `MutationSurface` union is
 * unchanged; ADR-324 adds no surface member). Carries any `capabilityDelta`
 * unchanged so shaperLoop's existing gate re-adjudicates it. No promotion here.
 */
export function toContextCandidateMutation(
  parent: ContextGenome,
  generated: GeneratedEnvironment,
): CandidateMutation {
  const candidate: ContextGenome = {
    surface: "context",
    version: parent.version + 1,
    entries: { ...parent.entries, [`env:${generated.spec.id}`]: generated.spec.task },
  };
  return {
    parent,
    candidate,
    describe: `spade environment candidate: ${generated.spec.id} — ${generated.spec.task}`,
    ...(generated.spec.capabilityDelta ? { capabilityDelta: generated.spec.capabilityDelta } : {}),
  };
}

/**
 * A generated environment offered to the WP9 loop. `kind` marks it a PROPOSAL,
 * not a promotion. `capabilityGate` is present iff the environment would
 * certify a capability EXPANSION (ADR-324 §4 tie-in to ADR-315 / WP9) — routed
 * to the constitutional gate stub and BLOCKED by default. `frontier` records
 * the solvable ∧ gap-exposing assessment. Grounding stays with the veto in the
 * promotion path — a proposal is a candidate, and the veto, dream-machine eval,
 * flywheel, and a human still own whether it is ever promoted.
 */
export interface EnvironmentProposal {
  readonly kind: "environment-proposal";
  readonly environment: GeneratedEnvironment;
  readonly frontier: FrontierAssessment;
  readonly candidateMutation: CandidateMutation;
  readonly capabilityGate?: ConstitutionalDecision;
  /** True iff a capability-expansion routed to the gate and was denied. */
  readonly blockedByGate: boolean;
}

/**
 * Turn a generated environment + its frontier assessment into a proposal: build
 * its context candidate mutation and, if it would expand capability, route it to
 * ADR-315's constitutional gate stub (blocked by default). Never promotes.
 */
export function buildEnvironmentProposal(
  parent: ContextGenome,
  generated: GeneratedEnvironment,
  frontier: FrontierAssessment,
): EnvironmentProposal {
  const candidateMutation = toContextCandidateMutation(parent, generated);
  const capabilityExpanding = expandsCapabilities(generated.spec.capabilityDelta);
  const capabilityGate = capabilityExpanding ? constitutionalGateStub(candidateMutation) : undefined;
  const blockedByGate = capabilityExpanding && !capabilityGate!.allowed;
  return Object.freeze({
    kind: "environment-proposal" as const,
    environment: generated,
    frontier,
    candidateMutation,
    ...(capabilityGate ? { capabilityGate } : {}),
    blockedByGate,
  });
}

// ---------------------------------------------------------------------------
// DREAM-MACHINE HAND-OFF — feed the EXISTING evaluation path (ADR-324 §3).
// ---------------------------------------------------------------------------

/**
 * Build the `DreamCandidate` (dreamMachine.ts) that carries a generated
 * environment into the EXISTING dream-machine evaluation stage. The
 * paired-bootstrap decision compares normal vs privileged reward samples — so a
 * dream-machine ACCEPT means the environment produces a STATISTICALLY
 * SIGNIFICANT regret signal (it is informative), scored the same way
 * research-gate scores everything else (statistics.ts). Pure data — building a
 * candidate is not evaluating or promoting one.
 */
export function toDreamCandidate(
  generated: GeneratedEnvironment,
  measurement: RegretMeasurement,
  commit: string,
): DreamCandidate {
  const decision: PairedDecision = pairedBootstrapDecision(
    measurement.normalSamples,
    measurement.privilegedSamples,
  );
  const report = [
    `# Self-play environment evaluation`,
    `citation: ${SPADE_CITATION}`,
    `environment: ${generated.spec.id}`,
    `task: ${generated.spec.task}`,
    `external_evidence_sources: ${generated.spec.provenance.evidence.length}`,
    `normal_reward: ${measurement.normalReward}`,
    `privileged_reward: ${measurement.privilegedReward}`,
    `regret: ${measurement.regret}`,
  ].join("\n");
  return {
    finding: `spade environment ${generated.spec.id} (regret=${measurement.regret})`,
    report,
    commit,
    decision,
  };
}
