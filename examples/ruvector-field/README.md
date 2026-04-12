# ruvector-field

**Status:** research sketch (runnable) · **Edition:** 2021 · **Dependencies:** none (std only)

A runnable reference implementation of the **RuVector field subsystem** —
an optional semantic and relational layer that sits above the RuVix kernel and
the existing coherence engine.

Full specification: [`docs/research/ruvector-field/SPEC.md`](../../docs/research/ruvector-field/SPEC.md)

---

## Table of contents

1. [What it is](#1-what-it-is)
2. [Why it exists](#2-why-it-exists)
3. [Design principles](#3-design-principles)
4. [Core concepts](#4-core-concepts)
5. [Architecture](#5-architecture)
6. [API surface](#6-api-surface)
7. [Scoring model](#7-scoring-model)
8. [Run the demo](#8-run-the-demo)
9. [Walkthrough of the demo output](#9-walkthrough-of-the-demo-output)
10. [File layout](#10-file-layout)
11. [Integration with the rest of RuVector](#11-integration-with-the-rest-of-ruvector)
12. [What this example is **not**](#12-what-this-example-is-not)
13. [Applications unlocked](#13-applications-unlocked)
14. [Roadmap to production](#14-roadmap-to-production)
15. [Acceptance gate](#15-acceptance-gate)
16. [License](#16-license)

---

## 1. What it is

`ruvector-field` is a small, self-contained Rust crate that builds an
in-memory **field engine**. The engine turns RuVector's field concept into
a concrete compute primitive for five jobs:

1. Contradiction-aware retrieval
2. Shell-based memory organization
3. Drift detection across sessions
4. Routing by missing field function
5. Observability of coherence, fracture, and recovery

The engine implements every primitive from the spec — shells, antipodes,
resonance, drift, routing hints — without pulling in ANN libraries, async
runtimes, or external storage. Everything fits in ~500 lines of Rust so the
shape of the spec stays visible end to end.

## 2. Why it exists

The current RuVix EPIC already defines coherence domains, mincut, cut pressure,
partitions, capabilities, and witnesses. That gives you a **structure plane**
and an **authority plane**. What it does not yet define is a **semantic plane**
— a place where meaning, contradiction, and abstraction depth are first class.

Adding that inside the kernel would break the 50 µs coherence epoch budget and
the sub-10 µs partition switch target. So the field layer lives **above** the
kernel as a RuVector crate, and only exports hints inward after benchmarks
prove it earns its keep.

This example is the smallest thing that can be called a field engine.

## 3. Design principles

1. **Hints, not mutations.** The engine emits advisory signals. Any hint that
   touches privileged state still passes through the existing proof and witness
   gates.
2. **Shells are logical, tiers are physical.** Shell depth answers *"what level
   of abstraction is this?"* Memory tier answers *"where does this live?"*
   They are orthogonal.
3. **Geometric vs. semantic antipodes are not the same thing.** A vector flip
   is cheap but meaningless. An explicit contradiction link is expensive but
   real. Keep them separate.
4. **Multiplicative resonance.** Averages hide collapse. A product makes one
   failing axis collapse the whole score — exactly the behavior you want for
   contradiction detection.
5. **Start with four shells.** Not 33. Four matches the existing four-tier
   memory discipline and is enough to demonstrate promotion and compression.
6. **Std only.** No dependencies. This is a spec in executable form, not a
   production library.

## 4. Core concepts

### Shells

Four logical shells describe abstraction depth:

| Shell       | Depth | Contents                                                   |
|-------------|-------|------------------------------------------------------------|
| `Event`     | 0     | raw exchanges, logs, observations, tool calls, sensor frames |
| `Pattern`   | 1     | recurring motifs, local summaries, contradiction clusters  |
| `Concept`   | 2     | durable summaries, templates, domain concepts, working theories |
| `Principle` | 3     | policies, invariants, contracts, proofs, operating rules   |

Shell depth is **not** the same as memory tier (hot/warm/dormant/cold). A
`Principle` can live in cold storage; an `Event` can be in hot memory.

### Antipodes

Two separate layers, never conflated:

- **Geometric antipode** — normalized negative of the embedding. Used for
  search geometry and novelty detection only. Cheap, general, meaningless on
  its own.
- **Semantic antipode** — explicit link saying "this node contradicts that
  node," sourced from humans, policies, model-detected opposition with
  explanation, or historical reversals. Powers contradiction reasoning.

### Field axes

Four default axes (pluggable per domain):

- **Limit** — what must not be crossed
- **Care** — what must be preserved
- **Bridge** — what must be connected
- **Clarity** — what must be understood

Each axis is normalized to `[0, 1]`. Resonance is their product times
coherence and continuity — so a single zero collapses the score to zero.

### Phi-scaled compression

Compression budget per shell follows the golden ratio:

```
Event     budget = B
Pattern   budget = B / φ
Concept   budget = B / φ²
Principle budget = B / φ³
```

Phi is used as a compression rule, not as a geometric primitive. This gives
graceful compaction as abstraction deepens without forcing exotic geometry
into the runtime.

## 5. Architecture

```
                    ┌──────────────────────────────────┐
                    │        ruvector-field            │
                    │                                  │
  ingest ─▶ embed ─▶│  bind contrast → assign shell    │
                    │  update graph  → compute         │
                    │  coherence     → detect drift    │
                    │  rerank        → issue hints     │
                    └────────────┬─────────────────────┘
                                 │  hints only
                                 ▼
                    ┌──────────────────────────────────┐
                    │     RuVix coherence engine       │
                    │  (graph state, mincut, pressure) │
                    └────────────┬─────────────────────┘
                                 │  proof + witness gated
                                 ▼
                    ┌──────────────────────────────────┐
                    │         RuVix kernel             │
                    │  (authority, partitions, boot)   │
                    └──────────────────────────────────┘
```

The field engine never bypasses the kernel. It never runs inside the 50 µs
scheduler epoch until a benchmark proves it's safe.

## 6. API surface

### Types (`src/types.rs`)

```rust
enum Shell { Event, Pattern, Concept, Principle }
enum NodeKind { Interaction, Summary, Policy, Agent, Partition, Region, Witness }
enum EdgeKind { Supports, Contrasts, Refines, RoutesTo, DerivedFrom, SharesRegion, BindsWitness }

struct AxisScores { limit, care, bridge, clarity: f32 }
struct Embedding  { values: Vec<f32> }     // L2 normalized on construction

struct FieldNode {
    id, kind, shell, axes,
    semantic_embedding, geometric_antipode,
    semantic_antipode: Option<u64>,
    coherence, continuity, resonance: f32,
    policy_mask: u64,
    witness_ref: Option<u64>,
    ts_ns: u64,
    text: String,
}

struct FieldEdge     { src, dst, kind, weight, ts_ns }
struct DriftSignal   { semantic, structural, policy, identity, total }
struct RoutingHint   { target_partition, target_agent, gain, cost, ttl, reason }
struct RetrievalResult { selected, rejected, contradiction_frontier, explanation }
```

### Engine (`src/engine.rs`)

```rust
impl FieldEngine {
    fn new() -> Self;

    /// Ingest a node into the Event shell with geometric antipode bound.
    fn ingest(
        &mut self,
        kind: NodeKind,
        text: impl Into<String>,
        embedding: Embedding,
        axes: AxisScores,
        policy_mask: u64,
    ) -> u64;

    /// Create an explicit bidirectional semantic antipode link.
    fn bind_semantic_antipode(&mut self, a: u64, b: u64, weight: f32);

    /// Generic edge insertion.
    fn add_edge(&mut self, src: u64, dst: u64, kind: EdgeKind, weight: f32);

    /// Recompute coherence per node using an effective-resistance proxy.
    fn recompute_coherence(&mut self);

    /// Apply shell promotion rules and return the list of changes.
    fn promote_candidates(&mut self) -> Vec<(u64, Shell, Shell)>;

    /// Shell-aware retrieval with contradiction frontier and explanation trace.
    fn retrieve(
        &self,
        query: &Embedding,
        allowed_shells: &[Shell],
        top_k: usize,
    ) -> RetrievalResult;

    /// Four-channel drift against a reference centroid.
    fn drift(&self, reference_centroid: &Embedding) -> DriftSignal;

    /// Pick the best-matching role and emit a routing hint.
    fn route(
        &self,
        query: &Embedding,
        roles: &[(u64, &str, Embedding)],
    ) -> Option<RoutingHint>;
}
```

## 7. Scoring model

### Resonance

```
resonance = limit · care · bridge · clarity · coherence · continuity
```

All factors are normalized to `[0, 1]`. Multiplication is intentional.

### Coherence

```
coherence = 1 / (1 + avg_effective_resistance)
```

The example approximates effective resistance via `1 − avg_cosine_similarity`
within the same shell. A production implementation would use the solver.

### Retrieval

```
candidate_score = semantic_similarity
                · shell_fit
                · coherence_fit
                · continuity_fit
                · resonance_fit

risk        = contradiction_risk + drift_risk + policy_risk
safety      = 1 / (1 + risk)
final_score = candidate_score · safety
```

### Routing

```
route_score = capability_fit
            · role_fit
            · locality_fit
            · shell_fit
            · expected_gain / expected_cost
```

## 8. Run the demo

From the repository root:

```bash
cargo run --manifest-path examples/ruvector-field/Cargo.toml
```

Or from inside the example directory:

```bash
cd examples/ruvector-field
cargo run
cargo run --release    # same thing, faster build output
cargo build            # build without executing
cargo check            # type-check only
```

No external dependencies, no network, no state files. The demo is
deterministic apart from timestamps.

## 9. Walkthrough of the demo output

The demo ingests seven nodes about an authentication bug, wires relationships,
binds an explicit semantic antipode, recomputes coherence, runs promotion,
retrieves, checks drift, and issues a routing hint. Abridged output:

```
=== RuVector Field Subsystem Demo ===

Shell promotions:
  node   6: Event → Pattern        ← the principle, promoted by support edges
  node   5: Event → Pattern        ← the concept

Current nodes:
  id=  1 shell=Event   coherence=0.982 resonance=0.083 text="User reports ..."
  id=  2 shell=Event   coherence=0.991 resonance=0.092 text="User reports ..."
  id=  3 shell=Event   coherence=0.990 resonance=0.081 text="Session refresh ..."
  id=  4 shell=Event   coherence=0.991 resonance=0.142 text="Pattern: idle ..."
  id=  5 shell=Pattern coherence=0.994 resonance=0.185 text="Concept: refresh ..."
  id=  6 shell=Pattern coherence=0.992 resonance=0.322 text="Principle: sessions ..."
  id=  7 shell=Event   coherence=0.973 resonance=0.006 text="Claim: idle ..."   ← opposing
```

The opposing claim stays in `Event` — its resonance collapses to `0.006`
because its axis scores are weak. It is not selected, but it **is** surfaced
on the contradiction frontier:

```
Retrieval:
  selected nodes: [5, 6]
  contradiction frontier: [7]
  explanation trace:
    - node 6 has semantic antipode 7 — flagged on contradiction frontier
    - selected node 5 with final_score=0.279
    - selected node 6 with final_score=0.270
```

Drift and routing:

```
Drift: semantic=0.160 structural=0.100 policy=0.000 identity=0.000 total=0.260
  (no alert — threshold not crossed or not enough agreeing channels)

Routing hint: agent=Some(1001) gain=0.243 cost=0.200 ttl=4
              reason="best role match: constraint"
  note: hint is advisory — privileged mutations must still pass proof + witness gates

Shell budgets (base = 1024):
  Event     → 1024.0
  Pattern   → 632.9
  Concept   → 391.1
  Principle → 241.7
```

**What to notice:**

1. The opposing claim is *not* filtered out — it's returned separately as a
   contradiction frontier so the caller (LLM, agent, human) can reason about
   it explicitly.
2. Promotion is driven by graph structure (support and contrast counts), not
   by heuristics on text.
3. Drift stays below threshold because only one channel shows movement. The
   spec requires ≥ 2 agreeing channels for an alert.
4. The routing hint carries a TTL and a cost — it is not a command.

## 10. File layout

```
examples/ruvector-field/
├── Cargo.toml          # binary crate, no dependencies
├── Cargo.lock          # committed for binary reproducibility
├── README.md           # this file
└── src/
    ├── main.rs         # demo entry point (~170 lines)
    ├── types.rs        # data model (~180 lines)
    └── engine.rs       # field engine (~230 lines)
```

Total: under 600 lines of Rust.

## 11. Integration with the rest of RuVector

| RuVector crate         | Role in the field subsystem                                              |
|------------------------|--------------------------------------------------------------------------|
| `ruvector-sparsifier`  | compressed field graph for coherence sampling and drift at scale         |
| `ruvector-solver`      | local coherence, effective resistance, anomaly ranking, route-gain estimation |
| `ruvector-mincut`      | split / migration / fracture hints (outside the 50 µs epoch initially)   |
| RuVix coherence engine | consumes field hints as advisory inputs to cut pressure and migration    |
| RuVix kernel           | receives `PriorityHint`, `SplitHint`, `MergeHint`, `TierHint`, `RouteHint` — only after benchmarks show gain |

This example does not yet wire to those crates. That is the deliberate first
step — see the roadmap below.

## 12. What this example is **not**

- **Not production.** No ANN, no HNSW, no persistence, no concurrency, no
  crash safety. Retrieval is O(n) linear scan.
- **Not a replacement for the solver or mincut.** Coherence here is a naive
  cosine-based proxy. Production coherence uses effective resistance from the
  solver.
- **Not a model.** Embeddings are hand-written for the demo. Bring your own
  embedding model (or embedding store) in real use.
- **Not a witness implementation.** Witness refs are fields on nodes but no
  events are emitted. Witnessing belongs in the RuVix integration crate.
- **Not benchmarked.** The acceptance gate is intentionally strict and has
  not been run.

## 13. Applications unlocked

The field engine makes a handful of things practical that previously had to
be hand-rolled per project:

1. **Contradiction-surfacing RAG** — retrieval that returns opposing evidence
   explicitly instead of silently picking a side. Useful for legal research,
   medical literature review, compliance, due diligence.
2. **Long-horizon agents with early drift warning** — four-channel drift
   detection catches slow world-model slide weeks before catastrophic
   failure.
3. **Explainable retrieval with audit trails** — every result returns a
   rationale and (when integrated) witness refs. Enables regulated-industry
   use (FDA, HIPAA, SOC2).
4. **Shell-aware knowledge compaction** — query principles only, or drill
   into raw events, with phi-scaled budgets keeping storage bounded.
5. **Diagnostic routing** — route by which field axis is collapsed, not just
   by task description. A conversation missing `clarity` routes to a
   constraint-checking agent.
6. **Contradiction-driven active learning** — the contradiction frontier is
   a structured uncertainty signal, richer than scalar confidence.
7. **Semantic fracture detection in distributed systems** — combines
   structural fracture (mincut) with semantic fracture (contradiction
   density) for federated learning and multi-region deployments.

See `docs/research/ruvector-field/SPEC.md` sections 11–13 for the precise
retrieval, drift, and routing semantics that enable these.

## 14. Roadmap to production

Promote this example into real crates in this order:

1. **`ruvector-field-types`** — extract the data model, add `serde`, make it
   `no_std` compatible.
2. **`ruvector-field-core`** — replace the cosine-sum coherence proxy with
   `ruvector-solver` calls, add promotion hysteresis and minimum residence
   windows.
3. **`ruvector-field-index`** — replace linear scan with HNSW, add temporal
   buckets and shell-segmented candidate lists.
4. **`ruvector-field-router`** — add role libraries, capability fit learning,
   and expected-gain estimation from the solver.
5. **`ruvix-field-bridge`** — adapter crate that converts field hints into
   RuVix `PriorityHint` / `SplitHint` / `MergeHint` / `TierHint` /
   `RouteHint` and emits the witness events listed in SPEC section 14.

## 15. Acceptance gate

The field engine is **not** allowed to export hints into the RuVix kernel
until **all four** of the following hold on a contradiction-heavy benchmark:

1. Contradiction rate improves by ≥ **20 %**
2. Retrieval token cost improves by ≥ **20 %**
3. Long-session coherence improves by ≥ **15 %**
4. Enabling hints does **not** violate the 50 µs coherence epoch budget or
   the sub-10 µs partition switch target

Until then, `ruvector-field` lives entirely in user space. No exceptions.

## 16. License

MIT OR Apache-2.0, matching the rest of the RuVector workspace.
