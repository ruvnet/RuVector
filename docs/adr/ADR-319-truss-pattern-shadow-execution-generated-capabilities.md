# ADR-319: TRUSS-Pattern Shadow Execution for Generated Capabilities

- **Status**: Proposed
- **Date**: 2026-08-20
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-311 (PIR, extends — anomaly quarantine for latent channels, WP7); ADR-315 (PIR, extends — governance constitution / capability-expansion gate, WP11); ADR-306 (PIR, extends — Dream Machine evaluation gate, WP2); ADR-305 (PIR, separation-of-powers invariant, binding here); ADR-317, ADR-318 (PIR, combined-effect acceptance test); ADR-312 (PIR, receipt anchoring); see `docs/research/perpetual-intelligence-runtime/06-wave2-evidence-review.md` §2
- **Tags**: pir, wave-2, truss, shadow-execution, brokered-tools, security

## Context

Wave-2 evidence review grades this paper **A** —
[arXiv:2608.17588](https://arxiv.org/abs/2608.17588), "TRUSS: Towards
Task-Reliable and User-Safe Automated Agent Skill Generation" (Zhang,
Ouyang, Shi, Wang; cs.AI, cs.SE), submitted 2026-08-18. Every claim the
Wave-2 brief cited checks out verbatim against the abstract:

- **Static evaluation + shadow-agent execution with brokered tools and
  provenance traces**: "Candidates admitted by this static gate are loaded
  by a shadow agent inside a Controllable Execution Environment, where
  brokered tools expose requested actions to policy enforcement and record
  their results as provenance preserving execution traces."
- **Effectiveness 17.11% → 52.94%**: "TRUSS raises task effectiveness from
  17.11% without Skills to 52.94%."
- **Security rate 50.80% → 100%**: "increasing the benchmark Security rate
  from 50.80% to 100.00%."
- Additional figures, also verbatim in the abstract, not part of the
  original brief but relevant context: 100.00% precision/recall in
  vulnerability detection; repair reduces attack success from 38.71%→19.35%
  (GPT-5.5) and 46.45%→29.68% (GPT-5.4), "with zero attack regression";
  evaluated on 168 SkillInject artifacts, 155 SkillSafetyBench cases, and
  all 187 SkillGenBench tasks — three **pre-existing, independently
  published** benchmarks (SkillInject; SkillSafetyBench, arXiv:2605.12015;
  SkillGenBench, arXiv:2605.18693) this paper evaluates *against*, not
  artifacts it itself ships.

**Artifact availability — checked, not assumed**: no "Comments:" field on
the arXiv abstract page, and a direct fetch of the paper's full HTML text
found no code, GitHub, dataset, "Availability," or "Reproducibility" section
anywhere. A targeted search for a standalone repository returned nothing.
**No code is available today.** This ADR is a first-party implementation of
the paper's described mechanism, not a port — the same posture ADR-318
applies to StagedWorkspace.

**Name-collision discipline (binding on this ADR and every derived work
package or code artifact)**: "TRUSS"/"Truss" is moderately overloaded in
exactly this program's domain — an active commercial security platform at
`truss-security.com`, a distinct AI coding-agent product at
`truss-agent.com` ("bring your own model, review what the agent can do
before it acts, choose permission policies") that is uncomfortably close in
subject matter to this paper's brokered-tool/policy-enforcement mechanism,
and a software consultancy at `truss.works`. None of these are the arXiv
paper. This ADR always spells out **"TRUSS (Task-Reliable and User-Safe
Skill generation, arXiv:2608.17588)"** on first use in any document, code
comment, or issue derived from it, and never uses bare "TRUSS" as a product,
package, or crate name — the same discipline this program applied to TARL in
Wave 1 (ADR-307).

**Preprint-reproduction rule** (applies uniformly across this program, per
`07-wave2-program-plan.md`): TRUSS's own reported figures (task
effectiveness 17.11%→52.94%, security rate 50.80%→100%, repair reducing
attack success with zero attack regression) describe the paper's own
benchmark and its own authors' unreleased implementation, not this
program's shadow-execution stage. This ADR treats the shadow-execution
mechanism it defines as a **candidate mutation**, not adopted prior art —
promotion of any candidate through this gate is conditioned on this
program's own `research-gate`-recomputed benchmark delta over the pre-gate
baseline, never on citing TRUSS's published numbers as if they already
describe this program's implementation.

`ruvector` already has two of the three pieces this pattern needs: ADR-311's
anomaly quarantine (HMAC integrity + statistical anomaly scoring on
latent-channel payloads) and ADR-315's constitutional capability-expansion
gate (separate, explicit approval before any mutation expands an agent's
tool/action/peer set). What is missing is TRUSS's middle piece: a
**brokered-tool-call shadow-execution stage** that inspects a candidate's
tool calls *before* they reach the execution backend, rather than only
scoring final behavior after the fact.

## Decision

Build a brokered-tool-call shadow-execution admission gate between Darwin's
mutation-proposal step (ADR-313) and Dream Machine's evaluation step
(ADR-306), informed by TRUSS's Static-Gate + Controllable-Execution-
Environment design (arXiv:2608.17588, first-party implementation — no
upstream code to port):

1. **Package the generated capability as an RVF artifact.** A Darwin-proposed
   skill or harness candidate is packaged and content-hash-bound per
   ADR-318 before it enters shadow execution, so every downstream trace can
   reference the exact candidate version under test.
2. **Load it into an RVM-backed shadow runtime with brokered tools.**
   Modeled on TRUSS's Controllable Execution Environment: the candidate
   executes inside a sandbox where every requested tool call is intercepted
   by a broker, checked against policy, and only then allowed to reach the
   real (or a simulated) backend. This is a new interception point on the
   Darwin mutation-proposal path — the highest-security-sensitivity change
   in this wave.
3. **Every tool call and its result is recorded as a provenance-preserving
   receipt**, anchored via ADR-312's shared witness contract (canonical
   encoding, domain-separated Ed25519 signatures) — mirroring TRUSS's
   "provenance preserving execution traces."
4. **Dream Machine's evaluation core (ADR-306) consumes those receipts** to
   reject, repair, or promote the candidate: a candidate whose brokered-tool
   trace shows a security violation is rejected outright; one whose trace is
   clean but whose task-effectiveness is below bar may be repaired (a
   TRUSS-pattern repair pass, targeting reduced attack success with zero
   attack regression, per the paper's confirmed figures) and resubmitted; a
   candidate clearing both bars proceeds to ADR-306's normal
   statistical-significance promotion gate.
5. **This shadow-execution stage never gains promotion authority.** Per
   ADR-305's separation-of-powers invariant (from ruflo ADR-322B), the
   broker inspects and can block a tool call or reject a candidate; it
   cannot itself promote one. A clean shadow run is one necessary input to
   ADR-306's pipeline, not a self-sufficient approval.
6. **A shadow-executed capability that would expand the agent's tool,
   action, or peer set still requires ADR-315's separate constitutional
   approval.** A capability passing this ADR's shadow-execution gate cleanly
   does not, by itself, satisfy ADR-315's capability-expansion gate — the
   two are independently blocking, exactly as ADR-315 already requires
   relative to ADR-306's ordinary promotion gate.
7. This gate interacts with, but does not replace, ADR-311's anomaly
   quarantine: a brokered tool call that passes policy enforcement but scores
   anomalously on ADR-311's statistical detector is still quarantined:
   quarantine and shadow-execution admission are complementary controls on
   the same candidate, not substitutes for each other.

## Consequences

### Positive

- Closes the one property this program's own novelty search (Wave-2
  evidence review, "Novelty claim" section) named as TRUSS's closest match
  among external candidates — capability-controlled shadow execution with
  brokered tools — and integrates it structurally rather than leaving it as
  an unimplemented gap this program's own novelty claim depends on.
- Gives a candidate's tool-call behavior visibility *before* it reaches a
  real backend, not just an after-the-fact outcome score — directly
  supporting ADR-317's "successful recovery" condition, since a
  shadow-execution receipt can show exactly which tool call a compromise
  originated from.
- TRUSS's own confirmed figures (task effectiveness 17.11%→52.94%, security
  rate 50.80%→100%, zero attack regression on repair) give this ADR a
  concrete, citable target shape for what "the gate is working" looks like,
  even though this program's own numbers must be independently measured.

### Negative

- No upstream reference implementation exists — this is the second
  from-scratch build in this wave (alongside ADR-318), and it is also the
  highest-security-sensitivity item: a bug in the broker's policy
  enforcement is a new attack surface on the mutation-proposal path itself.
- Two failure regimes to tune and test separately: policy-enforcement
  rejection (deterministic, rule-based) and ADR-311's statistical anomaly
  scoring (probabilistic) now both sit on the same candidate's evaluation
  path, compounding the tuning burden ADR-311 already flagged for its own
  two-mechanism design.
- Depends on ADR-311 and ADR-315 both existing first (per
  `07-wave2-program-plan.md` WP17's dependency list) — there is no
  quarantine mechanism to interact with and no capability-expansion gate to
  defer to until those land.

## Security / Validation Gates

- **Brokered-tool interception**: every tool call a shadow-executed
  candidate makes is intercepted, policy-checked, and logged before it can
  reach a real or simulated backend.
- **Provenance receipts**: every intercepted tool call and its result is
  recorded as a receipt anchored via ADR-312's shared witness contract.
- **Separation-of-powers invariant** (ADR-305, from ruflo ADR-322B): the
  shadow-execution broker inspects and can block; it never itself promotes.
- **Capability-expansion distinctness** (ADR-315): a clean shadow-execution
  run does not satisfy ADR-315's constitutional capability-expansion gate;
  the two remain independently blocking.
- **Quarantine interaction** (ADR-311): a policy-clean but anomaly-scored
  tool call is still quarantined; shadow-execution admission does not
  override quarantine.
- **Wave-2 combined acceptance test**: this ADR supplies the "TRUSS style
  shadow execution" component of ruv's verbatim Wave-2 acceptance criterion
  (see ADR-317's Security/Validation Gates for the full text) — its own
  promotion does not by itself satisfy that combined test.
- **Name-collision citation discipline**: spell out "TRUSS (Task-Reliable
  and User-Safe Skill generation, arXiv:2608.17588)" on first use; never use
  bare "TRUSS" as a package, crate, or module name, given the
  `truss-agent.com` collision.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  this module lands, per `07-wave2-program-plan.md`'s explicit call-out that
  WP17 touches mutation-proposal/capability surfaces.

## Affected Repos

- `ruvnet/ruvector` only — new module alongside `crates/rvm/crates/rvm-cap`
  (capability/tool-broker enforcement) and Darwin's mutation-proposal
  surface (`harness/src/darwin.ts`). Single-repo scope per
  `07-wave2-program-plan.md`.

## Dependencies

Depends on ADR-311 (anomaly quarantine — complementary control on the same
channel), ADR-315 (capability-expansion gate — this ADR's shadow-execution
admission is distinct from, and does not satisfy, that gate), and ADR-306
(Dream Machine — the pipeline this gate's receipts feed into). Depends
transitively on ADR-305's separation-of-powers invariant and ADR-312's
witness/anchoring contract (via ADR-318's content-hash binding on the
packaged candidate). ADR-317 and ADR-318 do not block this ADR's own
per-candidate gate, but the combined Wave-2 acceptance test names all three.

## Alternatives Considered

- **Score a candidate's final behavior only, without brokered-tool-call
  interception**: rejected — this is exactly the gap TRUSS's own design is
  adopted to close; final-behavior scoring cannot distinguish a candidate
  that behaved safely by chance from one whose every tool call was
  policy-clean, and it gives ADR-317's recovery condition nothing to trace
  a compromise back to.
- **Describe this mechanism as "implementing TRUSS" without the full
  disambiguated citation**: rejected — the evidence review's name-collision
  finding (§2) is explicit that `truss-agent.com`'s subject-matter overlap
  makes an unqualified "TRUSS" reference genuinely ambiguous to a reader;
  every reference spells out the arXiv ID, following the same discipline
  ADR-311 established for LATTE and ADR-307 established for TARL.
- **Grant the shadow-execution broker promotion authority for
  policy-clean candidates, skipping ADR-306's pipeline for the common
  case**: rejected — directly contradicts ADR-305's separation-of-powers
  invariant; a fast path that lets one gate self-promote is exactly the
  single-point-of-failure risk ADR-313's two-independent-layer design was
  built to avoid.
