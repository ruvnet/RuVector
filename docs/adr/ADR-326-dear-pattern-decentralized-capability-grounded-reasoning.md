# ADR-326: DeAR-Pattern Decentralized Capability-Grounded Reasoning

- **Status**: Proposed
- **Date**: 2026-08-21
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-309 (PIR, extends — LatentMesh greenfield crates); ADR-310 (PIR, extends — causal-attribution gate for latent communication); ADR-311 (PIR, extends — anomaly quarantine for latent channels); see `docs/research/perpetual-intelligence-runtime/08-wave3-evidence-review.md` and `09-wave3-program-plan.md`
- **Tags**: pir, wave-3, dear, decentralized-reasoning, capability-grounding, latentmesh, thought-graph

## Context

Wave-3 evidence review grades this paper **B+, not A** —
[arXiv:2608.17282](https://arxiv.org/abs/2608.17282), "DeAR: Decentralized
Agentic Reasoning via Capability Grounding and Collaborative Thought
Navigation," submitted 2026-08-18.

Confirmed mechanism, verbatim against the abstract, three parts: "(1)
decentralized capability grounding for query-dependent agent specialization,
(2) thought map navigation for targeted peer interactions, and (3) topology
update for adaptive error correction." Evaluation scope, verbatim, exact
match to the program brief: "Evaluations across 9 diverse multimodal
reasoning and text-based QA benchmarks." Headline claim, verbatim: "DeAR
consistently outperforms recent baseline methods" — **qualitative only, no
percentage figure given in the abstract**. This ADR does not invent one, the
same posture ADR-321 already applies to SkillForge's own qualitative-only
claim.

**Why the grade is B+, not A — a specific, load-bearing gap.** The program
brief's original framing described DeAR's dead-end handling as: on hitting a
reasoning dead end, the topology change lets the system **"continue, not
restart."** This detail is **not verifiable from the abstract** —
the abstract's own text on this point is four words, "topology update for
adaptive error correction," with no continue-vs-restart mechanism specified.
**This ADR cites only the verbatim four-word phrase as confirmed.** The
continue-vs-restart behavior is this program's own working interpretation of
what "adaptive error correction" plausibly means, pending either a full-text
read of the paper or a released implementation — **it is not a reproduced or
confirmed paper claim, and no implementation built from this ADR may harden
around it as though it were one.** Any design decision that depends
specifically on "continue, not restart" (rather than some other form of
topology adaptation) must be flagged as this program's own design choice at
the point it is made, not attributed to DeAR.

**Artifact availability — checked, not assumed: explicitly unavailable.**
The abstract's own final sentence, verbatim: "The source code will be
available at https://open_upon_acceptance" — a literal placeholder URL, not
a working link, and there is no Comments field with a real repository
either. This ADR is therefore a first-party build from the paper's
description, the same posture ADR-325 (D²ACCI) and ADR-318/ADR-319 already
take for their own no-code sources — and, combined with the unverified
dead-end-handling detail, the weakest evidentiary footing of this wave's
four items.

**Name collision — low risk.** Bare "DeAR"/"Dear" collides with Dear ImGui
and its language bindings, `dear-github`, and DeArrow — all unrelated
software, none in the agentic-reasoning domain, none using "DeAR"'s
capitalization convention. Lower risk than this wave's SPADE collision or
Wave 2's MemFuse collision, but this ADR still always cites the source in
full as "DeAR (arXiv:2608.17282, Decentralized Agentic Reasoning via
Capability Grounding and Collaborative Thought Navigation)" on first use per
document.

ADR-309 already scopes the `ruvector`-side LatentMesh integration as
greenfield engineering — new crates for transport, RVF packaging, and RVM
admission enforcement, coordinated on wire format with the small external
`ruvnet/LatentMesh` prototype. ADR-310 gates every latent-channel deployment
behind a controlled-replacement causal audit; ADR-311 quarantines anomalous
or tampered latent-channel payloads. Today, none of these three ADRs specify
*how* agents are selected as communication peers or *what* structure the
evolving multi-agent reasoning process itself takes — they govern the
channel, not the topology using it. **The first-mover angle this ADR
captures**: test DeAR's decentralized, capability-grounded peer-selection
topology using LatentMesh's **compressed latent state** (not token streams)
as the communication substrate, with ADR-310's causal verification gating
message acceptance — a combination not attempted in the paper itself (which
reasons over natural-language thought maps, not a compressed KV-cache-style
channel) and not attempted by any existing `ruvnet` mesh mechanism.

**Genuinely new versus existing `ruvnet` mesh/routing mechanisms, checked
directly, not assumed.** `autogenous`'s `radio-moe` package
(`mesh.ts`/`mesh-evolve.ts`/`mixture.ts`/`reputation.ts`/`relevance.ts`,
confirmed via file listing) is MoE-style reputation-weighted expert
**routing with failover** — a fixed mixture-of-experts dispatch mechanism,
not peer-to-peer collaborative reasoning over a shared, dynamically
navigated thought map. A grep of `radio-moe`'s source for "topology update"
and "thought map" returned no hits. This is also distinct from `ruflo`'s own
swarm topologies (hierarchical, mesh, ring, star): those are **static
configuration choices** selected once at swarm-init time, not a topology
that restructures itself dynamically mid-run in response to a dead end, as
DeAR's confirmed mechanism (however its specific adaptation behavior works)
describes.

**Preprint-reproduction rule** (applies uniformly across this program, per
`09-wave3-program-plan.md`): DeAR's "consistently outperforms recent
baseline methods" claim is cited qualitatively only, exactly as stated in
Decision below; no magnitude is invented. This program's own
`research-gate`-measured delta, not a citation of DeAR's own claim, is what
determines whether any capability-grounded peer-selection or thought-graph
mechanism built from this ADR is promoted.

## Decision

Replace named Ruflo agent roles with **continuously-measured capability
vectors**, and build a decentralized, LatentMesh-carried collaborative
reasoning topology informed by DeAR's confirmed mechanism
(arXiv:2608.17282):

1. **Capability vector**: for every (agent `i`, peer `j`, query `q`) triple,
   define `C(i,j,q) = competence × trust × locality × freshness ÷ cost`.
   Query-dependent agent specialization (DeAR's confirmed part 1) is
   operationalized as this vector being recomputed per-query, not a static
   role assignment — an agent's effective specialization for a given query
   is a function of its currently measured capability profile against that
   query, not a fixed label like "coder" or "reviewer."
2. **Local peer selection**: an agent selects its next collaboration peer
   for a given reasoning step by locally comparing `C(i,j,q)` across
   candidate peers — decentralized, per DeAR's framing, rather than routed
   through a central dispatcher (the point of contrast with `radio-moe`'s
   centralized reputation-weighted routing).
3. **RuVector stores the evolving thought graph.** The collaborative
   reasoning process (DeAR's confirmed part 2, thought map navigation) is
   persisted as an evolving graph structure in RuVector, with each node a
   reasoning step and each edge a targeted peer interaction — giving the
   thought graph the same durable, queryable substrate ADR-307's memory
   tiers and ADR-320's causal episodic graph already provide for
   observations.
4. **LatentMesh carries messages, as compressed latent state — the
   first-mover test.** Peer-to-peer messages in this topology are carried
   over the transport crates ADR-309 builds inside `ruvector`, as compressed
   latent state (KV-cache-style), not natural-language token streams —
   testing DeAR's topology under a communication substrate the paper itself
   does not use. Every message must pass ADR-310's controlled-replacement
   causal-attribution audit before it is accepted by the receiving agent;
   messages that fail causal attribution or that ADR-311's quarantine flags
   as anomalous are never delivered into the thought graph.
5. **Edge reinforcement**: successful reasoning paths (paths whose
   downstream outcome is confirmed good, per this program's existing
   promotion/verification machinery) reinforce their corresponding
   thought-graph edges; failed paths weaken theirs — an evolving,
   experience-weighted topology rather than a static one.
6. **Topology adaptation on dead ends is this program's own design choice,
   not a verified paper claim.** Where a reasoning path hits a dead end
   (no further productive peer interaction is found), this program adopts a
   specific behavior — e.g. re-routing to a different peer via updated
   capability vectors rather than restarting the entire reasoning
   process — but this behavior, and any framing of it as "continue, not
   restart," is explicitly marked in code comments, design docs, and every
   derived document as **this program's own design decision**, informed by
   but not confirmed against DeAR's four-word "topology update for adaptive
   error correction" abstract text. It must never be presented as
   reproduced from the paper.

## Consequences

### Positive

- Gives the Latent Communication Fabric (ADR-309/310/311) a concrete,
  capability-driven peer-selection and topology mechanism where today it has
  a transport, a causal-audit gate, and a quarantine layer but no defined
  notion of *which* peer an agent should talk to or *how* the collaboration
  structure evolves.
- Testing DeAR's topology over compressed latent state, gated by ADR-310's
  causal-attribution audit, is a genuine first-mover combination — neither
  the paper (natural-language thought maps) nor any existing `ruvnet`
  mechanism (`radio-moe`'s centralized routing, `ruflo`'s static swarm
  topologies) has attempted this pairing.
- Storing the thought graph in RuVector reuses this program's existing
  durable-graph substrate (ADR-307, ADR-320) rather than building a fourth
  parallel persistence mechanism for reasoning-process state.
- Explicitly flagging the continue-vs-restart behavior as a design choice,
  not a verified claim, prevents this program's own implementation from
  silently drifting into treating an unconfirmed detail as established fact
  — the same evidence-honesty discipline ADR-311 applies to correcting the
  "LATTE" citation.

### Negative

- This is this wave's weakest-evidence item: B+, not A, with an explicitly
  unverified load-bearing detail (dead-end continue-vs-restart) and no
  released code to check the mechanism against. The capability-vector
  formula (`competence × trust × locality × freshness ÷ cost`) and the
  thought-graph structure are this program's own operationalization of a
  four-part abstract description, not a port of a validated implementation.
- Running peer-to-peer reasoning over compressed latent state rather than
  token streams is a genuinely novel combination with no prior art to lean
  on for either side (DeAR's topology, LatentMesh's transport) — the
  interaction between the two is itself an open research question this
  program's own implementation must validate empirically.
- Depends on ADR-309's greenfield transport crates, ADR-310's causal-audit
  gate, and ADR-311's quarantine module all existing first — there is no
  channel to carry capability-grounded messages over, and no gate to
  validate them against, until all three land.
- A capability vector recomputed per-query for every candidate peer adds
  real computational overhead to every reasoning step compared to a fixed
  role assignment or a centrally-routed dispatch — this ADR does not itself
  bound that overhead, leaving it to the owning work package's
  implementation and benchmarking.

## Security / Validation Gates

- **Causal-attribution gate (ADR-310, unchanged)**: every message carried
  over LatentMesh in this topology must pass the controlled-replacement
  causal audit before the receiving agent accepts it into its thought graph
  — this ADR adds no exception.
- **Anomaly quarantine (ADR-311, unchanged)**: messages flagged anomalous or
  failing HMAC integrity verification are quarantined, never delivered into
  the thought graph, regardless of the sending peer's measured capability
  vector.
- **Capability-vector integrity**: `competence`, `trust`, `locality`,
  `freshness`, and `cost` inputs to `C(i,j,q)` must themselves be
  tamper-evident (witness-logged updates, per ADR-312's shared contract) —
  a compromised agent inflating its own `trust` or `competence` score to
  attract more peer traffic is a concrete attack this gate must resist, not
  an assumed-honest input.
- **Witness-chain requirement**: every thought-graph edge creation,
  reinforcement, and weakening event emits an RVM witness record (ruvector
  ADR-134 schema), anchored via ADR-312, so the evolving topology's history
  is auditable end-to-end.
- **Design-choice labeling discipline (binding)**: any code, comment, or
  document describing the dead-end continue-vs-restart behavior must
  explicitly label it as this program's own design choice, not a DeAR paper
  claim; a missing label is a review-blocking defect in any PR implementing
  this ADR.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to the capability-vector computation, peer-selection logic, or
  thought-graph write path.

## Affected Repos

- `ruvnet/ruvector` (the thought-graph storage layer, `crates/rvAgent`)
- `crates/rvAgent/latentmesh-*` — ADR-309's WP5 greenfield crates this
  ADR's compressed-latent-state messages are carried over; the
  capability-grounded thought-map navigation likely lands as a new
  `latentmesh-thoughtmap` crate, or as an extension of `latentmesh-align`,
  the exact split being an implementation decision for the owning work
  package, not fixed by this ADR. Coordinates with `latentmesh-gate`'s
  `causal.rs` per ADR-310's existing CI gate.
- `ruvnet/LatentMesh` (wire-format coordination for compressed-state
  message carriage, per ADR-309's existing coordination channel — no new
  coordination channel opened by this ADR)

## Dependencies

Depends on ADR-309 (the greenfield transport crates this topology's messages
are carried over), ADR-310 (the causal-attribution gate every message must
pass), and ADR-311 (the anomaly-quarantine module every message is checked
against), all directly — there is no channel, no audit, and no quarantine to
build a capability-grounded topology on top of until all three exist.

## Alternatives Considered

- **Keep named Ruflo agent roles (coder, reviewer, tester, …) as the
  peer-selection basis instead of continuously-measured capability
  vectors**: rejected — a fixed role label does not capture query-dependent
  specialization, which is DeAR's confirmed part 1 and the specific gap this
  ADR is adopted to close; a capability vector recomputed per query is
  strictly more expressive.
- **Route messages centrally through `radio-moe`'s existing
  reputation-weighted mixture-of-experts mechanism instead of local,
  decentralized peer selection**: rejected — this would replicate
  `radio-moe`'s existing centralized-routing pattern rather than testing
  DeAR's genuinely different decentralized mechanism, and would not exercise
  the first-mover latent-state-plus-causal-verification combination this ADR
  is written to test.
- **Present the dead-end continue-vs-restart behavior as a confirmed DeAR
  mechanism**: rejected outright — the abstract's four-word description does
  not support that level of specificity; doing so would misrepresent a
  B+-grade, code-unavailable source as if it carried the same evidentiary
  weight as this wave's A-grade items.
- **Carry messages as natural-language token streams, matching the paper's
  own thought-map format, instead of compressed latent state**: rejected —
  this would forgo the first-mover test this ADR is specifically designed to
  run (DeAR's topology over LatentMesh's compressed-state channel with
  ADR-310's causal gate), reducing this ADR to a plain reproduction of the
  paper's own setup rather than a novel combination worth this program's
  investment.
