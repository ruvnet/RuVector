# ADR-333: RVM Semantic-Authority Layer Above OpenShell-Class Secure Runtimes

- **Status**: Proposed (cross-repo posture — ADR-only in this repo; RVM-side work lands in `ruvnet/rvm` under maintainer review, USER ACTION for merge)
- **Date**: 2026-08-22
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-312 (PIR — shared witness/anchoring contract); ADR-315 (PIR — capability-expansion gate with witness-anchored approval log); ruvector ADR-134 (witness record schema); `ruvnet/rvm` ("The Virtual Machine Built for the Agentic Age"); see `docs/research/perpetual-intelligence-runtime/10-wave4-evidence-review.md` and `11-wave4-program-plan.md`
- **Tags**: pir, wave-4, rvm, openshell, secure-runtime, interop, cross-repo, security

## Context

Two NVIDIA items, both verified real (see `10-wave4-evidence-review.md`
§NVIDIA):

1. **"Where Security Fits in an AI Agent Stack"**
   (developer.nvidia.com, published 2026-08-21; Greco, Thadaka, Golshan,
   Watson) describes a five-layer agent stack — distribution/product,
   orchestration/meta-harness, agent harness, secure runtime, inference
   infrastructure — and argues authoritative controls must live **below
   the agent boundary**. Two load-bearing quotes: "The harness guides
   what an agent tries. The infrastructure controls what an agent can
   do," and "A layer designed to be modified cannot reliably enforce
   controls against its own modification."
2. **NVIDIA OpenShell** — [github.com/NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell),
   Apache-2.0, created 2026-02, 8.3k stars, very active — a kernel-level
   sandboxed runtime for autonomous agents: Landlock filesystem
   restriction + seccomp process/syscall restriction, declarative YAML
   policy across four layers (filesystem, network, process, inference
   API routing; network+inference hot-reloadable), runs Claude
   Code/Codex-class agent CLIs **unmodified**, with "a full audit trail
   of every allow and deny decision."

**Why this matters to this program**: the placement thesis is this
program's own frozen-model / evolving-harness invariant, articulated by
a third party with an 8k-star enforcement surface behind it. NVIDIA is
effectively defining the enterprise vocabulary for the category. The
mutable-harness layers this program deliberately evolves (MetaHarness,
Darwin candidates, skills) are exactly the layers the essay says cannot
self-enforce — which this program already accepts: its enforcement
points (research-gate vetoes, proof-gate, ADR-315's capability gate)
live outside the evolving loop.

**The competitive read from ruv's briefing, adopted**: the correct
response is **differentiation, not imitation**. OpenShell already owns
kernel isolation, declarative policy, controlled networking, and audit
records; duplicating any of that in RVM would be building a worse
version of a free Apache-2.0 component. What OpenShell does *not*
provide — and what nothing at its layer can, because it is
per-host and syscall-semantic — is portable **semantic authority** for
agents: who an agent is across hosts, what its memory provenance is,
which mutations it is entitled to make, which capabilities it can carry
or transfer, and what it can prove afterward about what it did.

## Decision

Position RVM one layer above OpenShell-class secure runtimes, in
semantic authority, and interoperate rather than compete:

1. **RVM's differentiated surface** (the semantic-authority layer):
   signed agent identity; signed memory; mutation rights; transferable
   capabilities; provenance chains; revocation; rollback; and portable
   execution receipts. These compose with — and are not derivable from —
   kernel-level allow/deny enforcement.
2. **OpenShell interoperability adapter** (the briefing's difficulty-3
   item): (a) **policy projection** — an RVM capability grant projects
   down to OpenShell YAML (filesystem/network/process/inference rules),
   so semantic grants become kernel-enforced facts, using the
   hot-reloadable network/inference layers for revocation; (b) **audit
   ingestion** — OpenShell's allow/deny audit trail ingests into RVM's
   witness chain (ADR-134 schema, ADR-312 anchoring), so a portable
   execution receipt can cite kernel-level enforcement events as
   evidence.
3. **PIR runtime posture**: OpenShell becomes a supported (not
   required) secure-runtime substrate for PIR agent execution —
   evaluated against RVM scope rather than duplicated, per the evidence
   review's recommendation.
4. **Cross-repo boundary, stated plainly**: this ADR records posture and
   the adapter's contract from the ruvector/PIR side. Implementation
   lands in `ruvnet/rvm`, which requires maintainer review — merging
   there is **USER ACTION**, never agent-approved, per the program's
   standing merge-policy rule.

## Consequences

### Positive

- RVM's roadmap sharpens around what only it can provide, while gaining
  a credible, widely-deployed enforcement substrate below it for free.
- NVIDIA's essay becomes external validation to cite for the program's
  existing enforcement-below-the-mutable-layer architecture rather than
  a competitive threat.
- Projected policy + ingested audit gives portable receipts real
  kernel-level evidence — strictly stronger claims than either layer
  alone.

### Negative

- Adapter scope depends on OpenShell's policy/audit formats remaining
  stable; version pinning and re-validation on upstream changes are
  required (it is a fast-moving project).
- Linux-specific enforcement (Landlock/seccomp) — the adapter's
  guarantees do not transfer to non-Linux hosts; receipts must record
  which enforcement substrate was actually present.
- Cross-repo sequencing risk: nothing in this repo blocks on it, but the
  differentiation story is only real once RVM-side work merges — which
  is outside this program's merge authority.

## Security / Validation Gates

- **No permission-laundering**: RVM-side merges require maintainer
  review; surfaced as USER ACTION, never routed around.
- **Projection is downgrade-only**: policy projection may only narrow
  what OpenShell would otherwise allow — an RVM grant can never widen a
  host's baseline policy; widening requires ADR-315's capability-
  expansion gate with its witness-anchored approval.
- **Receipt honesty**: an execution receipt must state the enforcement
  substrate (OpenShell version/policy hash, or "none") — absence of
  kernel enforcement is recorded, never implied away.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan`
  on any ruvector-side adapter-contract code when it lands.

## Affected Repos

- `ruvnet/ruvector`: docs only this wave. `ruvnet/rvm`: adapter +
  semantic-authority surface (maintainer review — USER ACTION).

## Dependencies

Soft: ADR-312 (witness anchoring for audit ingestion), ADR-315
(capability-expansion gate for any projection widening). Nothing in
Wave 4 blocks on this ADR.

## Alternatives Considered

- **Build RVM's own kernel isolation**: rejected — duplicates a free,
  Apache-2.0, 8k-star, actively-maintained component; RVM's value is the
  layer OpenShell structurally cannot occupy.
- **Ignore OpenShell**: rejected — NVIDIA is defining the category's
  enterprise vocabulary; non-interoperability would strand RVM's
  semantic layer without an enforcement substrate customers already run.
- **Deep OpenShell dependency (require it)**: rejected — support, don't
  require; RVM must remain substrate-portable and receipts must record
  what was actually enforced.
