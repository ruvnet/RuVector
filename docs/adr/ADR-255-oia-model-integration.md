---
adr: 255
title: "ruvector ↔ OIA Model integration — alignment profile for the Open Intelligence Architecture (v0.1)"
status: proposed
date: 2026-06-16
authors: [ruvnet, claude-flow]
related: [ADR-122, ADR-134, ADR-074, ADR-155, ADR-193, ADR-251]
tags: [oia, reference-architecture, integration, provenance, witness-chain, rvf, mcp, sona, interoperability, standards]
---

# ADR-255 — ruvector ↔ OIA Model integration (Open Intelligence Architecture v0.1)

> **Maturity caveat (read first).** The OIA Model
> ([agenticsorg/OIA-Model](https://github.com/agenticsorg/OIA-Model)) is **v0.1**
> and self-describes as "an interactive review and implementation workspace,"
> not a finalized standard. It has **no machine-readable conformance schema, no
> registry, no versioning SLA** — its conformance instrument is a human
> COHERENT/GAP/NOT-YET/N-A rating embedded in a React UI. Any "OIA conformance"
> claim today is therefore **narrative, not verifiable**. This ADR decides a
> *non-binding alignment profile*, deliberately avoiding committing ruvector's
> public APIs to OIA layer semantics before OIA reaches ≥ v1.0.

## Context

The **Open Intelligence Architecture (OIA)** is the Agentics Foundation's
reference architecture for enterprise intelligent systems (MIT-licensed, ©2026).
It defines (per `src/content/layers.ts` + `docs/OIA-Model-v0.1-Digest.md`):

- **Ten layers, L0–L9** — physical compute at the bottom, human/browser interface
  at the top, with **two state-holding layers (L3 Agent Data Substrate, L8
  Continuity Fabric)** bracketing the operational core (L4–L7). The L3/L8
  asymmetry (substrate vs active fabric) is a deliberate design statement
  (Decision 03/05).
- **Six cross-layer spans** (concern classes, not APIs): SPAN-SEC, SPAN-SOV,
  SPAN-AUD, SPAN-IDN, SPAN-ENG, SPAN-PRV.
- L7 explicitly names **Model Context Protocol (MCP)** as the reference
  interoperability protocol; L8 explicitly names **witness chains** and
  "cognitive-state sovereignty."

ruvector already implements much of the operational/state core. The question is
*where ruvector fits in an OIA-conformant stack* and *what to publish* so others
can integrate it, without over-claiming.

### Layer → ruvector mapping (from the codebase; evidence H/M/L)

| OIA Layer | ruvector component(s) | Coverage |
|---|---|---|
| L0 Physical Compute | ruvix (bare-metal ARM), agentic-robotics-embedded | M (no energy/facility/supply-chain plane) |
| L1 Silicon Abstraction | SIMD/AVX-512 feature flags, WASM targets | H portability / no ISA attestation |
| L2 Sovereign Infrastructure | mcp-brain-server on Cloud Run, GCS/Firestore, DP engine (ε=1.0) | H but **GCP-specific** |
| **L3 Agent Data Substrate** | ruvector-core (HNSW + redb + memmap), RVF manifest (TLV level0/level1), RaBitQ/DiskANN/IVF, BrainMemory provenance fields | **H — strong** |
| L4 Model Training & Adaptation | SONA (two-tier LoRA, EWC++, ReasoningBank), federated LoRA (Gate A/B aggregation) | H **adaptation only** — no pre-training/eval gates |
| **L5 Inference & Retrieval** | HNSW ANN, RaBitQ, router crates, ranking engine + GWT attention rerank | **H — strong** |
| **L6 Context & Knowledge** | mcp-brain-server KnowledgeGraph (PPR + MinCut + sparsifier), Brainpedia lifecycle, Common-Crawl/RSS ingest, drift monitoring | **H — strong** |
| **L7 Orchestration & Workflow** | MCP server surface (OIA-named protocol), rvf-adapters (claude-flow/agentdb/agentic-flow), AgiContainer tool registry | **H — MCP-native** |
| **L8 Continuity Fabric** | RVF cognitive container (AgiContainerBuilder), witness chain (rvf-crypto, SHAKE-256 linked), SONA EWC++, emergent-time | **H — strong** (research-tier in parts) |
| L9 Human & Browser Interface | ui/ruvocal (SvelteKit), SSE streaming, voice | M thin (no consent/handoff/accessibility) |

ruvector is strongest exactly where OIA's operational + state core lives:
**L3, L5, L6, L7, L8** — and weak at the edges OIA leaves to infrastructure/UX
(L0/L1/L9) and at full L4 (pre-training).

## Decision

Adopt a **non-binding OIA alignment profile** for ruvector. Concretely:

1. **Position ruvector as an L3 + L5–L8 provider** in OIA-framed materials, with
   explicit "out of scope" notes for L0/L1 (infrastructure), L2 portability
   (GCP-specific today), L4 pre-training, and L9 UX.
2. **Designate the RVF cognitive container as ruvector's reference L8 artifact.**
   `AgiContainerBuilder` already bundles model pinning, governance/coherence/
   authority config, MCP tool registry, witness chain, skill library, and replay
   script in one signed binary — the closest existing match to OIA L8. Publish an
   `oia-l8-profile` mapping (TLV tag range in the RVF Level-1 namespace) — **doc
   + tag reservation, no engine change**.
3. **Designate the witness chain as the SPAN-AUD / SPAN-PRV primitive.** Define a
   `witness_type` registry mapping OIA action categories (inference, grounding
   retrieval, orchestration step, continuity checkpoint) to the existing
   `witness_type` byte. Reuses ADR-134 witness work; doc-level.
4. **Expose mcp-brain-server as the L6 grounding service** via an OIA-framed
   response envelope (`/v1/oia/context` wrapping `brain_search` + GWT rerank +
   citation pass-through) — ~150-line additive route, optional.
5. **Surface the MCP tool registry as the L7 connector** (`oia-l7.json` schema
   referencing the existing MCP surface) and the federated-LoRA endpoints
   (`/v1/lora/submit`, `/v1/lora/latest`) as L4-adaptation interfaces — doc-level,
   no code change.
6. **Do NOT** claim machine-verifiable OIA conformance, **do NOT** rename or
   re-shape existing public APIs to OIA terms, and **do NOT** take an OIA
   dependency in any crate. The alignment lives in documentation + reserved
   tags/schemas until OIA ships a real conformance spec.

The deliverable is an **alignment document + reserved RVF tag ranges/schemas**,
not a refactor. Integration points 2–5 are independently shippable, lowest-risk
first (the L8 container profile and witness-type registry are pure
documentation/tag reservation).

## Consequences

### Positive
- Clear, honest positioning of ruvector within a recognized layer taxonomy —
  useful for vendor/RFP/architecture-review conversations OIA targets.
- The highest-value integrations are **doc-only or additive** (L8 container
  profile, witness-type registry, L6 envelope) — near-zero blast radius.
- ruvector's existing strengths (RVF provenance, witness chain, MCP-native
  orchestration, SONA adaptation) map cleanly onto OIA's L3/L7/L8 — little new
  to build, mostly to *describe*.
- Reuses prior decisions: ADR-122 (RVF container), ADR-134 (witness chain),
  ADR-074 (SONA), ADR-155 (rulake dispatcher).

### Negative
- **OIA is pre-1.0 and volatile.** A v0.2+ conformance model could invalidate
  this profile; mitigated by keeping it non-binding and out of code.
- **No machine-checkable conformance** — alignment is narrative; we cannot
  certify, only map.
- **Genuine gaps** ruvector does not cover: L0/L1 (no energy-sovereignty/ISA
  attestation), L2 portability (GCP lock-in vs OIA's portability intent), L4
  pre-training + eval gates, L9 consent/handoff/accessibility, SPAN-IDN (no
  formal agent-identity framework). These must be stated as "relies on external
  infrastructure," not papered over.
- Some L8 pieces (emergent-time integration) are research-tier (`dead_code`/
  lint-relaxed); the L8 claim is "strong but maturing," not production-frozen.

### Neutral
- No new runtime dependency; OIA stays an external reference, ruvector stays
  self-contained. The profile can be withdrawn if OIA stalls.

## Links
- OIA Model: [agenticsorg/OIA-Model](https://github.com/agenticsorg/OIA-Model) ·
  [v0.1 Digest](https://github.com/agenticsorg/OIA-Model/blob/main/docs/OIA-Model-v0.1-Digest.md) ·
  live: oia.agentics.org
- ruvector: ADR-122 (RVF cognitive container), ADR-134 (witness chain),
  ADR-074 (SONA), ADR-155 (rulake dispatcher), ADR-193 (rairs IVF),
  ADR-251 (emergent-time)
- Source of mapping: deep-research brief over OIA `src/content/layers.ts` +
  digests and the ruvector codebase (rvf-runtime/agi_container.rs,
  rvf-crypto/witness.rs, mcp-brain-server/{graph,types}.rs, sona).
