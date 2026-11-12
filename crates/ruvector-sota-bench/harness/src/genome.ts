/**
 * Skill / context / harness genome types (PIR WP9, ADR-313, issue #841).
 *
 * ADR-313 §1 enumerates exactly what MAY evolve in the SHAPER-pattern loop:
 * skills, context, and the execution harness. Weights are NOT a mutation
 * surface — `MutationSurface` has no "weights" member, so a weight-mutating
 * candidate is unrepresentable in this type system. The structural backstop
 * for JS callers (and for this module's own source) is
 * scripts/frozen-weights-check.mjs, which scans this directory as an enforced
 * mutation surface.
 *
 * CAPABILITY-EXPANSION BOUNDARY (ADR-315): a mutation that adds tools, action
 * classes, or communication peers requires the constitutional gate — a
 * separately-authorized approval DISTINCT from ordinary behavioral promotion.
 * That gate is stubbed here against autogenous's `promoteAuthorized` concept
 * (ADR-315 Decision §2: `Promote = Better ∧ Safe ∧ Authorized ∧ Reversible`)
 * and BLOCKS by default: until WP11 wires PIR's capability/tool/action tables
 * into the `authorized` conjunct, no approval record can exist, so no
 * capability expansion can be recommended. WP11 integration point.
 */

/**
 * What may evolve (ADR-313 §1). Deliberately closed: adding a member to this
 * union is an ADR-level decision, not a code change.
 */
export type MutationSurface = "skills" | "context" | "harness";

/** A reusable skill the frozen model can invoke — SHAPER's evolving skill set. */
export interface SkillDefinition {
  readonly name: string;
  /** The skill body (prompt/program text the frozen model executes against). */
  readonly body: string;
}

/** The evolving skill library. */
export interface SkillGenome {
  readonly surface: "skills";
  readonly version: number;
  readonly skills: readonly SkillDefinition[];
}

/** The evolving context/configuration handed to the frozen model. */
export interface ContextGenome {
  readonly surface: "context";
  readonly version: number;
  readonly entries: Readonly<Record<string, string>>;
}

/**
 * The evolving execution-harness parameters. Maps onto Darwin's existing
 * `Policy` surface (flywheel.ts PARAMETER_CANDIDATES / darwin.ts GEPA).
 */
export interface HarnessGenome {
  readonly surface: "harness";
  readonly version: number;
  readonly parameters: Readonly<Record<string, string>>;
}

export type EvolvableGenome = SkillGenome | ContextGenome | HarnessGenome;

const SURFACES: readonly MutationSurface[] = ["skills", "context", "harness"];

/**
 * Runtime guard for untyped callers: the only admissible surfaces are
 * ADR-313's three. Anything else — "weights" above all — throws.
 */
export function assertEvolvableGenome(genome: EvolvableGenome): void {
  if (!SURFACES.includes(genome.surface)) {
    throw new Error(
      `inadmissible mutation surface "${String(genome.surface)}" — ` +
      `ADR-313 permits only ${SURFACES.join(", ")}; weights never evolve`,
    );
  }
  if (!Number.isInteger(genome.version) || genome.version < 0) {
    throw new Error("genome.version must be a non-negative integer (surfaces are separately versioned)");
  }
}

/**
 * A reference to the frozen foundation model. The sha256 is the loop-start
 * hash recorded for the witness chain (ADR-313: day-30 re-hash must be
 * bit-identical to day 0). There is deliberately no weight-file path here.
 */
export interface FrozenModelRef {
  readonly id: string;
  readonly sha256: string;
}

export function assertFrozenModelRef(model: FrozenModelRef): void {
  if (!/^[0-9a-f]{64}$/.test(model.sha256)) {
    throw new Error("frozen model sha256 must be 64 lowercase hex chars (loop-start hash, witness-recorded)");
  }
  if (!model.id) throw new Error("frozen model ref requires an id");
}

/**
 * What a candidate mutation would ADD to the agent's capability set.
 * Any non-empty field crosses the ADR-315 constitutional boundary.
 */
export interface CapabilityDelta {
  readonly addsTools: readonly string[];
  readonly addsActionClasses: readonly string[];
  readonly addsCommunicationPeers: readonly string[];
}

/** A proposed mutation: parent genome, candidate genome, and its capability delta. */
export interface CandidateMutation {
  readonly parent: EvolvableGenome;
  readonly candidate: EvolvableGenome;
  /** One-line human-auditable description of the mutation. */
  readonly describe: string;
  /** Omitted or all-empty means: no capability expansion. */
  readonly capabilityDelta?: CapabilityDelta;
}

export function expandsCapabilities(delta?: CapabilityDelta): boolean {
  if (!delta) return false;
  return delta.addsTools.length > 0
    || delta.addsActionClasses.length > 0
    || delta.addsCommunicationPeers.length > 0;
}

/**
 * The constitutional gate's decision shape, mirroring the conjuncts of
 * autogenous `promoteAuthorized(candidate, champion, { authorized, reversible })`
 * (ADR-315 Decision §2 — implemented and DONE upstream).
 */
export interface ConstitutionalDecision {
  readonly allowed: boolean;
  readonly reasons: readonly string[];
}

/**
 * ADR-315 constitutional gate — STUB (WP11 integration point).
 *
 * Models the `authorized` conjunct of autogenous's `promoteAuthorized`
 * predicate. This program's residual scope (ADR-315 Decision §2a/2b) is to
 * wire PIR's capability/tool/action tables into that conjunct and to enforce
 * the proposer≠approver distinctness rule. None of that wiring exists yet, so
 * NO approval record can exist, so the gate blocks — capability expansion is
 * denied by default, never rounded up. WP11 replaces this stub with the real
 * autogenous integration; the test suite pins the blocked-by-default contract.
 */
export function constitutionalGateStub(mutation: CandidateMutation): ConstitutionalDecision {
  if (!expandsCapabilities(mutation.capabilityDelta)) {
    return Object.freeze({
      allowed: false,
      reasons: Object.freeze(["not_a_capability_expansion — this gate only adjudicates capability expansion (ADR-315 §1)"]),
    });
  }
  return Object.freeze({
    allowed: false,
    reasons: Object.freeze([
      "constitutional_gate_stub_unwired — WP11 has not wired PIR capability tables into promoteAuthorized's authorized conjunct",
      "no_approval_record — ADR-315 requires an explicit, witness-anchored approval distinct from behavioral promotion",
    ]),
  });
}
