# ADR-320: MemFuse-Pattern AtomicObservation and Causal Episodic Graph

- **Status**: Proposed
- **Date**: 2026-08-20
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-307 (PIR, extends — three-level persistent memory, continuous-latent-state tier); ADR-310 (PIR, extends — causal-attribution gate for latent communication, WP6); see `docs/research/perpetual-intelligence-runtime/06-wave2-evidence-review.md` §3
- **Tags**: pir, wave-2, memfuse, atomic-observation, causal-graph, memory, provenance

## Context

Wave-2 evidence review grades this paper **A** —
[arXiv:2608.18704](https://arxiv.org/abs/2608.18704), "MemFuse: Multi-Source
Memory Fusion from Fragmented Observations," submitted 2026-08-19. The
brief's qualitative claim matches the abstract closely, verbatim:
"MemFuse, a structured memory system that preserves source-level evidence
in event-layer atomic memory and organizes related atomic events into
cluster-layer fused memory within a causal fusion graph." Retrieval-time
traceability is separately confirmed: "During retrieval, MemFuse retrieves
and organizes related evidence fragments while maintaining traceability to
original source events."

**Artifact availability — checked, not assumed**: code is referenced at
`github.com/Darwin-Agent/Mi-Memory/tree/master/MemFuse`. Verified directly
via the GitHub API: `Darwin-Agent/Mi-Memory` is public, 14.5 MB, last pushed
2026-08-19T03:23:36Z, and the `MemFuse` subdirectory contains real content
(`MemFuse.pdf`, `MemFuseBench/`, `README.md`, `figure/`) — not a
placeholder. This ADR adapts the confirmed-live implementation rather than
rebuilding from the paper alone, the same posture ADR-321 applies to
SkillForge.

**Name collision — SEVERE, and binding on this ADR's naming decisions.**
`github.com/memfuse/memfuse` is an established, actively used, unrelated
open-source project — "the lightning-fast open-source memory layer that
gives LLMs persistent, queryable memory across conversations and sessions,"
with its own layered L0 (episodic)/L1 (semantic)/L2 (knowledge-graph)
architecture. This predates the arXiv paper, is maintained by a different
team, and is a direct conceptual competitor: both are literally "memory
fusion for LLM agents" under the same name. A third, likely-fork repository
(`xuyongfu/memfuse-0630`) mirrors the pre-existing project's description,
reinforcing real usage/forking activity. **This ADR, and every deliverable
derived from it, never uses bare "MemFuse" as an npm package, crate, or
module name.** Every reference disambiguates as **"MemFuse (arXiv:2608.18704,
`Darwin-Agent/Mi-Memory`)"**, explicitly distinct from the pre-existing
`memfuse/memfuse` open-source memory layer.

ADR-307 already formalizes `ruvector`'s continuous-latent-state memory tier
around LiveMem's fixed-capacity, context-independent recurrent-state design
— but that design is single-agent. ADR-310's causal-attribution gate exists
to make cross-agent latent communication attributable, but it operates on
communication events, not on a structured, multi-source, provenance-preserving
observation record. MemFuse's atomic-event/cluster-layer architecture is the
missing structural link between the two: a way to fuse observations arriving
from multiple agents or sensors into one causal graph, with provenance
preserved back to each originating event, that ADR-310's audit can then
consume.

## Decision

Adopt **AtomicObservation** as the atomic memory unit written into ADR-307's
continuous-latent-state tier, and a **causal episodic graph** as the
fusion structure over it, informed by MemFuse's event-layer/cluster-layer
architecture (arXiv:2608.18704) and adapting the confirmed-live
`Darwin-Agent/Mi-Memory/MemFuse` implementation:

1. **`AtomicObservation` schema**: every observation written by any agent or
   sensor into the continuous-latent-state tier carries, at minimum:
   `source` (the originating agent/sensor identity), `time` (observation
   timestamp), `confidence` (the source's own confidence in the observation),
   `tenant` (multi-tenant isolation boundary), `signature` (RVM-witness-style
   authenticity binding, per ADR-307's existing witness-chain requirement),
   and `causal_parents` (references to the `AtomicObservation`(s), if any,
   that causally preceded and informed this one).
2. **Multi-source fusion** builds a causal graph linking related
   `AtomicObservation`s into cluster-layer fused memory — mirroring
   MemFuse's event-layer→cluster-layer structure — while preserving
   traceability from every fused cluster back to its original source
   observations, per the paper's confirmed retrieval-time traceability
   guarantee.
3. **The fused causal graph feeds ADR-310's causal-attribution gate** as
   structured evidence: where ADR-310 today audits latent-communication
   events for causal value, it can now consume this ADR's fused,
   provenance-preserved observation graph as its evidentiary substrate for
   multi-agent attribution, rather than reasoning over unstructured
   communication events alone.
4. Extends ADR-307's continuous-latent-state tier — a single-agent design as
   originally formalized — to the Latent Communication Fabric's multi-agent
   setting, without altering that tier's existing fixed-capacity,
   context-independent recurrent-state properties (LiveMem-derived, per
   ADR-307) for any individual agent's own observation stream.
5. Benchmark against the released `MemFuseBench` (part of the confirmed-live
   artifact), plus this program's own `research-gate`-recomputed delta per
   the candidate-mutation framing — MemFuse's own reported qualitative
   claims are the starting design target, not an adopted acceptance bar
   without independent verification.

## Consequences

### Positive

- Gives multi-agent, multi-source observations a structured record with
  built-in provenance (`causal_parents`) instead of requiring ADR-310's
  causal-attribution gate to reconstruct causal relationships from
  unstructured event logs after the fact.
- The confirmed-live `Darwin-Agent/Mi-Memory/MemFuse` implementation and
  `MemFuseBench` give WP18 a genuine port-and-adapt scope, materially
  smaller than ADR-318's or ADR-319's from-scratch builds.
- `tenant` and `signature` fields on every `AtomicObservation` extend this
  program's existing multi-tenant isolation and witness-chain disciplines
  (ADR-307) down to the level of individual observations, not just tier-level
  writes.

### Negative

- The severe MemFuse/memfuse name collision creates real ongoing naming
  discipline overhead — every package, module, and doc reference must be
  checked against the disambiguation rule, and a single missed instance (a
  commit message, an issue title) reintroduces the exact ambiguity this ADR
  is written to prevent.
- Fusing observations from multiple sources into one causal graph is a
  genuine new attack surface: a compromised or malfunctioning source could
  inject `AtomicObservation`s with fabricated `causal_parents` links,
  polluting the fused graph's structure even if each individual observation
  passes its own signature check — this ADR's causal-parents chain requires
  its own acyclicity/verifiability enforcement (see Security Gates), which
  is additional validation surface beyond a single-source memory write.
- Depends on both ADR-307 (memory tiers) and ADR-310 (causal-audit gate)
  existing first — WP18 cannot start meaningfully until both land, per
  `07-wave2-program-plan.md`'s WP18 dependency list.

## Security / Validation Gates

- **Naming discipline (binding, no exceptions)**: never ship an npm package,
  crate, or module literally named `memfuse`; every reference disambiguates
  as "MemFuse (arXiv:2608.18704, `Darwin-Agent/Mi-Memory`)."
- **Per-observation signature**: every `AtomicObservation` is signed
  (RVM-witness-style, per ADR-307's existing witness-chain requirement)
  before it is considered committed to the continuous-latent-state tier.
- **Causal-parents integrity**: the `causal_parents` chain across
  `AtomicObservation`s must be acyclic and independently verifiable — a
  cycle or an unresolvable parent reference is a hard rejection at fusion
  time, not a warning.
- **Tenant isolation**: the `tenant` field is enforced as a hard boundary at
  fusion time — an `AtomicObservation` from one tenant never fuses into
  another tenant's causal graph.
- **Complementary to, not a substitute for, ADR-310's audit**: this ADR's
  fused causal graph is evidentiary input to ADR-310's causal-attribution
  gate; a well-formed fused graph does not itself constitute a passed
  causal-attribution audit.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to the fusion write path, since multi-source ingestion is a
  direct target for observation-injection attacks.

## Affected Repos

- `ruvnet/ruvector` only — `crates/rvf/rvf-adapters/agentdb`, `crates/rvf` (continuous-latent-state
  tier), coordinates with `latentmesh-align`-consuming code per the existing
  ADR-310 CI gate. Single-repo scope per `07-wave2-program-plan.md`.

## Dependencies

Depends on ADR-307 (the continuous-latent-state memory tier this ADR's
`AtomicObservation`s are written into) and ADR-310 (the causal-attribution
gate this ADR's fused causal graph feeds as evidence). No dependency on
ADR-317, ADR-318, ADR-319, ADR-321, or ADR-323 — this ADR is not part of the
Wave-2 combined acceptance test.

## Alternatives Considered

- **Extend ADR-307's memory tier for multi-agent fusion without a
  causal-parents field**: rejected — this would leave ADR-310's
  causal-attribution gate reconstructing causal relationships from
  unstructured logs, exactly the gap MemFuse's event-layer/cluster-layer
  design is adopted to close.
- **Rebuild the fusion mechanism from the paper's description alone,
  ignoring the confirmed-live `Darwin-Agent/Mi-Memory/MemFuse`
  implementation**: rejected — the evidence review confirms this code is
  real, non-empty, and includes `MemFuseBench`; ignoring it would repeat the
  from-scratch-build cost ADR-318 and ADR-319 must pay only because no code
  exists for those two papers.
- **Use "MemFuse" or a close variant as this program's own package/module
  name, relying on context to disambiguate from `memfuse/memfuse`**:
  rejected outright — the evidence review grades this the most severe
  collision risk of the wave; relying on context is exactly the failure mode
  that produces genuine mislabeling in fast-moving swarm work, per
  `07-wave2-program-plan.md`'s Top Risks §3.
