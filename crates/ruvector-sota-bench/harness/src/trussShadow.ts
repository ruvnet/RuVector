/**
 * Shadow-execution admission gate for generated capabilities, following the
 * TRUSS pattern (Task-Reliable and User-Safe Skill generation,
 * arXiv:2608.17588) — PIR WP17, ADR-319, issue #864.
 *
 * FIRST-PARTY REPRODUCTION: the paper ships no code (evidence review §2);
 * this is an implementation of its described mechanism — a static gate,
 * then a shadow agent whose every tool request goes through a BROKER that
 * only exposes granted tools and records every request as a
 * provenance-preserving trace — not a port. Per the program-wide
 * preprint-reproduction rule, TRUSS's published figures (effectiveness
 * 17.11%→52.94%, security 50.80%→100%) describe the paper's own benchmark,
 * never this implementation; promotion of anything through this gate is
 * conditioned on this program's own research-gate-recomputed deltas.
 *
 * HONEST SCOPING (ADR-319 §2): the broker here is a MODELED, in-process
 * mediator — the shadow executor runs in this process against simulated
 * tool backends supplied by the caller. The fuller form (RVM-runtime
 * process/sandbox isolation of the shadow agent, per ADR-319's
 * "RVM-backed shadow runtime") is explicitly out of this slice.
 *
 * SEPARATION OF POWERS (ADR-305, from ruflo ADR-322B; ADR-319 §5):
 * this stage evaluates and can REJECT or suggest REPAIR — it NEVER
 * promotes. Structurally: no export of this module matches
 * /promote|merge|approve|apply/; the verdict is frozen data with no
 * callable members; the broker is grant-closed at construction (no method
 * adds a grant afterwards) and can only reach the simulated backends it
 * was constructed with — never the real tool set or active policy.
 * test/trussShadow.test.ts pins all of this.
 *
 * PIPELINE POSITION: sits between Darwin's proposal step and Dream
 * Machine's evaluation step. Its verdict feeds the EXISTING
 * evaluation→vetoes→flywheel path as one more conjunctive
 * PromotionVetoProvider (composable with dreamMachineVetoProvider and
 * WP15's harness-risk gate): a clean shadow run adds no credit and rescues
 * nothing; anything else blocks. A capability that the shadow pass shows
 * would EXPAND capability (new tool/action/peer — declared, or observed as
 * an ungranted invocation attempt) is routed to the ADR-315 constitutional
 * gate (genome.ts constitutionalGateStub, WP11 integration point) and is
 * blocked by default; that gate is called, never duplicated, here.
 */
import { createHash } from "node:crypto";
import { assertNoLiveCredential } from "@metaharness/redblue";
import {
  assertEvolvableGenome,
  constitutionalGateStub,
  expandsCapabilities,
  type CandidateMutation,
  type CapabilityDelta,
  type ConstitutionalDecision,
  type EvolvableGenome,
} from "./genome.js";
import type { PromotionVetoProvider, VetoContext } from "./vetoes.js";

// ---------------------------------------------------------------------------
// Content-hash-bound artifact (ADR-318 discipline) + canonical encoding
// ---------------------------------------------------------------------------

/**
 * Canonical JSON: sorted keys, no non-finite numbers, no negative zero —
 * the encoding discipline ADR-312 adopts from ruflo ADR-322C, applied at
 * this module's receipt boundary. Deterministic across runs by construction.
 */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value === "boolean" || typeof value === "string") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("canonicalJson: non-finite number rejected (ADR-312)");
    if (Object.is(value, -0)) throw new Error("canonicalJson: negative zero rejected (ADR-312)");
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((entry) => canonicalJson(entry)).join(",")}]`;
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`);
    return `{${entries.join(",")}}`;
  }
  throw new Error(`canonicalJson: unsupported value of type ${typeof value}`);
}

export function sha256Hex(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

/**
 * A generated capability packaged as a content-hash-bound artifact
 * (ADR-319 §1, per ADR-318): every downstream trace and receipt references
 * the exact candidate version under test via `artifactId`.
 */
export interface CapabilityArtifact {
  /** sha256 of the canonical encoding of the candidate genome. */
  readonly artifactId: string;
  readonly surface: EvolvableGenome["surface"];
  readonly version: number;
  readonly describe: string;
}

export function capabilityArtifact(mutation: CandidateMutation): CapabilityArtifact {
  assertEvolvableGenome(mutation.candidate);
  return Object.freeze({
    artifactId: sha256Hex(canonicalJson(mutation.candidate)),
    surface: mutation.candidate.surface,
    version: mutation.candidate.version,
    describe: mutation.describe,
  });
}

// ---------------------------------------------------------------------------
// Brokered tools + provenance trace (ADR-319 §2–3)
// ---------------------------------------------------------------------------

/** A simulated tool backend the broker may expose to the shadow agent. */
export interface ShadowToolGrant {
  readonly tool: string;
  /** Simulated backend — NEVER a real side-effecting tool in this slice. */
  readonly handler: (input: string) => string | Promise<string>;
}

/** One brokered invocation, as the broker (not the candidate) recorded it. */
export interface BrokeredInvocation {
  readonly seq: number;
  readonly tool: string;
  readonly granted: boolean;
  readonly outcome: "allowed" | "denied";
  readonly inputSha256: string;
  /** Present only for allowed invocations. */
  readonly resultSha256?: string;
}

/** The provenance-preserving execution trace — TRUSS's "receipts". */
export interface ProvenanceTrace {
  readonly artifactId: string;
  readonly invocations: readonly BrokeredInvocation[];
}

/**
 * The capability-controlled mediator handed to the shadow executor: the
 * ONLY surface through which a candidate can reach any tool. Grant-closed —
 * there is deliberately no method that adds a grant or reaches outside the
 * simulated backends.
 */
export interface BrokeredTools {
  readonly grantedTools: readonly string[];
  invoke(tool: string, input: string): Promise<string>;
}

interface BrokerInternals {
  readonly tools: BrokeredTools;
  seal(): readonly BrokeredInvocation[];
  deniedTools(): readonly string[];
}

function createBroker(grants: readonly ShadowToolGrant[]): BrokerInternals {
  // Grants are copied at construction; later mutation of the caller's array
  // cannot widen the broker, and nothing here writes back to it.
  const handlers = new Map(grants.map((grant) => [grant.tool, grant.handler]));
  const invocations: BrokeredInvocation[] = [];
  const denied = new Set<string>();
  let sealed = false;

  const tools: BrokeredTools = Object.freeze({
    grantedTools: Object.freeze([...handlers.keys()]),
    async invoke(tool: string, input: string): Promise<string> {
      if (sealed) throw new Error("shadow trace sealed — the brokered sandbox is closed");
      const handler = handlers.get(tool);
      if (!handler) {
        denied.add(tool);
        invocations.push(Object.freeze({
          seq: invocations.length,
          tool,
          granted: false,
          outcome: "denied" as const,
          inputSha256: sha256Hex(input),
        }));
        throw new Error(`brokered tool "${tool}" is not granted to this candidate`);
      }
      const result = await handler(input);
      invocations.push(Object.freeze({
        seq: invocations.length,
        tool,
        granted: true,
        outcome: "allowed" as const,
        inputSha256: sha256Hex(input),
        resultSha256: sha256Hex(result),
      }));
      return result;
    },
  });

  return {
    tools,
    seal() {
      sealed = true;
      return Object.freeze([...invocations]);
    },
    deniedTools: () => Object.freeze([...denied].sort()),
  };
}

// ---------------------------------------------------------------------------
// Static evaluation pass (TRUSS's static gate)
// ---------------------------------------------------------------------------

export interface StaticFinding {
  readonly severity: "critical" | "repairable";
  readonly kind: string;
  readonly detail: string;
  /** For repairable findings: what a TRUSS-pattern repair pass would do. */
  readonly repairHint?: string;
}

/** The `tool:<name>(...)` convention generated skill bodies use. */
const TOOL_REFERENCE = /\btool:([A-Za-z0-9_-]+)/g;

function genomeText(genome: EvolvableGenome): string[] {
  switch (genome.surface) {
    case "skills":
      return genome.skills.flatMap((skill) => [skill.name, skill.body]);
    case "context":
      return Object.values(genome.entries);
    case "harness":
      return Object.values(genome.parameters);
  }
}

/**
 * Static gate: evaluate the candidate WITHOUT executing it. Critical
 * findings deny admission to shadow execution outright; repairable findings
 * admit it but downgrade a clean run to "repair-suggested".
 */
export function staticEvaluateCapability(
  mutation: CandidateMutation,
  grantedTools: readonly string[],
): readonly StaticFinding[] {
  assertEvolvableGenome(mutation.candidate);
  const findings: StaticFinding[] = [];
  const granted = new Set(grantedTools);

  for (const text of genomeText(mutation.candidate)) {
    try {
      assertNoLiveCredential(text);
    } catch (error) {
      findings.push(Object.freeze({
        severity: "critical" as const,
        kind: "live_credential_shaped_token",
        detail: error instanceof Error ? error.message : String(error),
      }));
    }
    for (const match of text.matchAll(TOOL_REFERENCE)) {
      const tool = match[1]!;
      if (!granted.has(tool)) {
        findings.push(Object.freeze({
          severity: "repairable" as const,
          kind: "ungranted_tool_reference",
          detail: `candidate references tool "${tool}" which is not granted`,
          repairHint: `remove the "${tool}" reference, or seek an ADR-315 capability-expansion approval for it`,
        }));
      }
    }
  }
  return Object.freeze(findings);
}

// ---------------------------------------------------------------------------
// Shadow execution + verdict (ADR-319 §2, §4–6)
// ---------------------------------------------------------------------------

/**
 * The shadow agent: exercises the candidate against the brokered sandbox
 * ONLY. It receives no other handle — in particular no filesystem, policy,
 * or real-tool handle — from this module.
 */
export type ShadowExecutor = (
  capability: EvolvableGenome,
  tools: BrokeredTools,
) => void | Promise<void>;

/**
 * Default shadow executor for skills-surface candidates: interprets each
 * `tool:<name>(<arg>)` occurrence in each skill body, in order, as one
 * brokered invocation. Denied invocations are recorded by the broker and
 * interpretation continues, so the trace shows the candidate's FULL
 * intended tool usage, not just its prefix up to the first denial.
 */
export const interpretSkillCapability: ShadowExecutor = async (capability, tools) => {
  if (capability.surface !== "skills") {
    throw new Error(`interpretSkillCapability executes the skills surface, got "${capability.surface}"`);
  }
  const call = /\btool:([A-Za-z0-9_-]+)\(([^)]*)\)/g;
  for (const skill of capability.skills) {
    for (const match of skill.body.matchAll(call)) {
      await tools.invoke(match[1]!, match[2]!).catch(() => undefined);
    }
  }
};

export type TrussShadowOutcome = "clean" | "repair-suggested" | "reject";

/**
 * The pass's verdict: pure, frozen data. An INPUT to the existing
 * vetoes/flywheel promotion path (and, downstream, to Dream Machine's
 * evaluation per ADR-319 §4) — never an approval, never a bypass.
 */
export interface TrussShadowVerdict {
  readonly outcome: TrussShadowOutcome;
  readonly reasons: readonly string[];
  readonly artifact: CapabilityArtifact;
  readonly staticFindings: readonly StaticFinding[];
  readonly trace: ProvenanceTrace;
  /** Whether the shadow executor actually ran (statically denied → "no"). */
  readonly shadowExecuted: "yes" | "no";
  /**
   * Present iff the candidate crossed the ADR-315 capability boundary
   * (declared delta, or observed ungranted invocation attempts). Produced
   * by genome.ts's constitutionalGateStub — called, not duplicated.
   */
  readonly capabilityGate?: ConstitutionalDecision;
}

export interface TrussShadowOptions {
  mutation: CandidateMutation;
  /** The candidate's granted tools — the whole reachable tool universe. */
  grants: readonly ShadowToolGrant[];
  /** Defaults to interpretSkillCapability for skills-surface candidates. */
  execute?: ShadowExecutor;
}

function mergedCapabilityDelta(
  declared: CapabilityDelta | undefined,
  observedUngrantedTools: readonly string[],
): CapabilityDelta {
  return {
    addsTools: [...new Set([...(declared?.addsTools ?? []), ...observedUngrantedTools])].sort(),
    addsActionClasses: declared?.addsActionClasses ?? [],
    addsCommunicationPeers: declared?.addsCommunicationPeers ?? [],
  };
}

/**
 * Run one generated capability through the two TRUSS-pattern passes:
 * (a) static evaluation, then — if statically admitted — (b) shadow
 * execution with brokered tools and a provenance-preserving trace.
 *
 * Ordering of the ADR-315 boundary (mirrors shaperLoop.ts): a candidate
 * whose DECLARED capabilityDelta expands capability is routed to the
 * constitutional gate first and, blocked by default, never shadow-executes.
 * An UNDECLARED expansion the shadow pass observes (ungranted invocation
 * attempts) is routed to the same gate after the run.
 */
export async function runTrussShadow(options: TrussShadowOptions): Promise<TrussShadowVerdict> {
  const { mutation, grants } = options;
  assertEvolvableGenome(mutation.parent);
  assertEvolvableGenome(mutation.candidate);
  const artifact = capabilityArtifact(mutation);
  const grantedNames = grants.map((grant) => grant.tool);
  const reasons = new Set<string>();

  // ADR-315 first: a declared capability expansion is independently blocking
  // and never reaches execution (same contract as shaperLoop.ts).
  const declaredExpansion = expandsCapabilities(mutation.capabilityDelta);
  let capabilityGate = declaredExpansion ? constitutionalGateStub(mutation) : undefined;
  if (declaredExpansion && !capabilityGate!.allowed) {
    reasons.add("capability_expansion_unauthorized");
  }

  // (a) Static gate.
  const staticFindings = staticEvaluateCapability(mutation, grantedNames);
  const staticallyDenied = staticFindings.some((finding) => finding.severity === "critical");
  if (staticallyDenied) reasons.add("static_gate_denied");

  // (b) Shadow execution — only for statically admitted, non-expanding
  // candidates (TRUSS: "Candidates admitted by this static gate are loaded
  // by a shadow agent").
  const admitted = !staticallyDenied && reasons.size === 0;
  const broker = createBroker(grants);
  if (admitted) {
    const execute = options.execute ?? interpretSkillCapability;
    try {
      await execute(mutation.candidate, broker.tools);
    } catch {
      reasons.add("shadow_execution_error");
    }
  }
  const invocations = broker.seal();
  const observedUngranted = broker.deniedTools();
  if (invocations.some((invocation) => invocation.outcome === "denied")) {
    reasons.add("ungranted_tool_invocation");
  }

  // Observed (undeclared) capability expansion also routes to ADR-315.
  if (!capabilityGate && observedUngranted.length > 0) {
    const delta = mergedCapabilityDelta(mutation.capabilityDelta, observedUngranted);
    capabilityGate = constitutionalGateStub({ ...mutation, capabilityDelta: delta });
    if (!capabilityGate.allowed) reasons.add("capability_expansion_unauthorized");
  }

  const repairable = staticFindings.some((finding) => finding.severity === "repairable");
  const outcome: TrussShadowOutcome =
    reasons.size > 0 ? "reject" : repairable ? "repair-suggested" : "clean";
  if (outcome === "repair-suggested") reasons.add("static_repairable_findings");

  return Object.freeze({
    outcome,
    reasons: Object.freeze([...reasons].sort()),
    artifact,
    staticFindings,
    trace: Object.freeze({ artifactId: artifact.artifactId, invocations }),
    shadowExecuted: admitted ? "yes" as const : "no" as const,
    ...(capabilityGate ? { capabilityGate } : {}),
  });
}

// ---------------------------------------------------------------------------
// Veto-provider wiring (ADR-319 §4–5: feeds the EXISTING promotion path)
// ---------------------------------------------------------------------------

/**
 * Map a verdict into vetoes.ts veto reasons. Pure and conjunctive with all
 * other vetoes (dreamMachine, WP15 harness-risk): "clean" adds no credit
 * and rescues nothing; anything else blocks the recommendation.
 */
export function vetoesFromTrussVerdict(verdict: TrussShadowVerdict): string[] {
  if (verdict.outcome === "clean") return [];
  if (verdict.outcome === "repair-suggested") return ["truss_shadow_repair_suggested"];
  return ["truss_shadow_reject", ...verdict.reasons.map((reason) => `truss_shadow_${reason}`)];
}

/**
 * A PromotionVetoProvider for flywheel.ts — the ONLY wiring between this
 * gate and promotion, and it can only ever say "no" or "no objection".
 * WP15 (#862) consumes the same TrussShadowVerdict as the shadow-execution
 * leg of the Wave-2 acceptance test.
 */
export function trussShadowVetoProvider(
  optionsFor: (context: VetoContext) => TrussShadowOptions | Promise<TrussShadowOptions>,
): PromotionVetoProvider {
  return async (context) => {
    const verdict = await runTrussShadow(await optionsFor(context));
    return vetoesFromTrussVerdict(verdict);
  };
}

// ---------------------------------------------------------------------------
// Receipt seam (ADR-319 §3 — ADR-312 / ruflo ADR-322C anchoring contract)
// ---------------------------------------------------------------------------

/**
 * The shadow-execution receipt: the artifact, its verdict, and the full
 * provenance trace — "the receipts" the TRUSS pattern
 * (arXiv:2608.17588) describes as provenance-preserving execution traces.
 */
export interface ShadowExecutionReceipt {
  readonly artifact: CapabilityArtifact;
  readonly outcome: TrussShadowOutcome;
  readonly reasons: readonly string[];
  readonly trace: ProvenanceTrace;
}

export interface EmittedReceipt {
  /** ruflo ADR-322C canonical encoding of the receipt. */
  readonly canonical: string;
  /** sha256 of the canonical encoding. */
  readonly digest: string;
  /** True only when a witness-anchoring backend signed the receipt. */
  readonly anchored: boolean;
  /** Domain-separated Ed25519 signature, when anchored. */
  readonly signature?: string;
}

/**
 * The 322C receipt-emission contract this slice defines but does not fully
 * wire: an emitter takes a receipt and returns its canonical/anchored form.
 * Full ADR-312 anchoring (domain-separated Ed25519 over the canonical
 * encoding, witness-chain append) is the follow-on wiring.
 */
export interface ReceiptEmitter {
  emit(receipt: ShadowExecutionReceipt): Promise<EmittedReceipt>;
}

export function receiptFromVerdict(verdict: TrussShadowVerdict): ShadowExecutionReceipt {
  return Object.freeze({
    artifact: verdict.artifact,
    outcome: verdict.outcome,
    reasons: verdict.reasons,
    trace: verdict.trace,
  });
}

/**
 * Stub emitter (in-slice): canonical-encodes and digests the receipt with
 * the ADR-312 encoding discipline, but does NOT sign or anchor it —
 * `anchored: false`, never rounded up.
 */
export const unanchoredReceiptEmitter: ReceiptEmitter = Object.freeze({
  async emit(receipt: ShadowExecutionReceipt): Promise<EmittedReceipt> {
    const canonical = canonicalJson(receipt);
    return Object.freeze({
      canonical,
      digest: sha256Hex(canonical),
      anchored: false,
    });
  },
});
