# RuVector Field Subsystem — Specification

**Status:** Draft (research)
**Date:** 2026-04-12
**Context:** Extends RuVix EPIC (2026-04-04). Optional layer above the kernel and
coherence engine. Does not alter v1 boot, witness, or proof acceptance criteria.

---

## 1. Design intent

This subsystem turns the field concept into a compute primitive for five jobs:

1. Contradiction aware retrieval
2. Shell based memory organization
3. Drift detection over time
4. Routing by missing field function
5. Observability of coherence, fracture, and recovery

Translation of field language into runtime primitives:

1. Antipodal relation becomes **contrast pairing**
2. Nested tori become **shell indices across abstraction depth**
3. Projection into observed space becomes **action selection under proof and policy**
4. Resonance becomes a **bounded score** used for search, routing, and compaction

This subsystem lives **above** the kernel, not inside it. RuVix already separates
kernel, coherence, and agents as optional layers, so the field logic first lives
as a RuVector layer and only exports hints inward after it proves value.

## 2. Scope

A new optional RuVector subsystem called `ruvector-field`:

1. Adds a logical field layer over vectors and graphs.
2. Keeps logical shells separate from physical memory tiers. RuVix already has
   tiered memory as hot, warm, dormant, and cold. Shell depth answers "what level
   of abstraction is this state". Memory tier answers "where does this state
   live physically".
3. Keeps expensive logic out of the kernel hot path until proven. `ruvector-mincut`
   `no_std` work is already the highest risk integration item.

## 3. Non goals

- Do **not** implement literal RP4, Hopf, or 33 torus structures.
- Do **not** treat simple vector negation as semantic opposition.
- Do **not** push shell placement, antipode search, or contradiction synthesis
  into the scheduler epoch until they show an offline and user space benchmark win.
- Do **not** weaken v1 goals for boot, switch time, witness completeness, or
  recovery.

## 4. System placement

1. RuVix kernel — authority plane
2. Existing coherence engine — structure plane
3. New field engine — semantic and relational plane
4. Agents and services consume field hints through the execution layer

Conceptual pipeline:

```
ingest state
  → embed state
  → bind contrast
  → assign shell
  → update graph
  → compute coherence
  → detect drift
  → rerank retrieval
  → issue routing hints
  → proof gate any mutation
  → witness the committed result
```

## 5. Core abstractions

### 5.1 Shells

Start with **four** logical shells (matches existing four tier memory discipline):

1. **Event** — raw exchanges, logs, observations, tool calls, sensor frames
2. **Pattern** — recurring motifs, local summaries, contradiction clusters, frequent transitions
3. **Concept** — durable summaries, templates, domain concepts, working theories
4. **Principle** — policies, invariants, contracts, proofs, operating rules

33 shells is rejected until benchmarks justify it.

### 5.2 Antipodes

Two layer model:

1. **Geometric antipode** — contrastive transform of the embedding used for
   normalization and search geometry.
2. **Semantic antipode** — explicit link to a contradictory, opposing, or policy
   incompatible node.

Plain `v` and `-v` are not semantic opposites. Geometric antipodes are cheap.
Semantic antipodes power contradiction reasoning.

### 5.3 Field axes

Application semantics, not kernel semantics. Default axis set:

1. **Limit**
2. **Care**
3. **Bridge**
4. **Clarity**

Axes are pluggable for legal, industrial, security, or robotics contexts.

### 5.4 Field signals

1. coherence
2. continuity
3. resonance
4. drift
5. contradiction risk
6. policy fit
7. shell fit
8. routing gain

Extends existing `CoherenceScore` and `CutPressure`; does not replace them.

## 6. Data model

Four crates:

1. `ruvector-field-types` — shared structs and enums
2. `ruvector-field-core` — shell placement, antipode binding, resonance, drift
3. `ruvector-field-index` — shell aware retrieval, contradiction filters, snapshots
4. `ruvector-field-router` — agent and partition hint generation

Minimal type model:

```rust
pub enum Shell { Event, Pattern, Concept, Principle }

pub enum NodeKind {
    Interaction, Summary, Policy, Agent,
    Partition, Region, Witness,
}

pub enum EdgeKind {
    Supports, Contrasts, Refines, RoutesTo,
    DerivedFrom, SharesRegion, BindsWitness,
}

pub struct AxisScores {
    pub limit: f32,
    pub care: f32,
    pub bridge: f32,
    pub clarity: f32,
}

pub struct FieldNode {
    pub id: u64,
    pub kind: NodeKind,
    pub semantic_embedding: u64,
    pub geometric_antipode_embedding: u64,
    pub semantic_antipode: Option<u64>,
    pub shell: Shell,
    pub axes: AxisScores,
    pub coherence: f32,
    pub continuity: f32,
    pub resonance: f32,
    pub policy_mask: u64,
    pub witness_ref: Option<u64>,
    pub ts_ns: u64,
}

pub struct FieldEdge {
    pub src: u64,
    pub dst: u64,
    pub kind: EdgeKind,
    pub weight: f32,
    pub ts_ns: u64,
}

pub struct DriftSignal {
    pub semantic: f32,
    pub structural: f32,
    pub policy: f32,
    pub identity: f32,
    pub total: f32,
}

pub struct RoutingHint {
    pub target_partition: Option<u64>,
    pub target_agent: Option<u64>,
    pub gain_estimate: f32,
    pub cost_estimate: f32,
    pub ttl_epochs: u16,
}
```

## 7. Storage model

Two orthogonal indices.

**Semantic index:**

1. embedding id
2. shell
3. temporal bucket
4. node kind

**Relational index:**

1. node id
2. outgoing edges
3. incoming edges
4. witness binding
5. partition binding

Snapshots are periodic, append only, cheap to diff. A snapshot contains:

1. shell centroids
2. contradiction frontier
3. per partition coherence
4. drift totals
5. active routing hints
6. witness cursor

Matches RuVix witness native and reconstructable direction.

## 8. Scoring model

### 8.1 Resonance

```
resonance = limit * care * bridge * clarity * coherence * continuity
```

All factors normalized to `[0, 1]`. Multiplicative — a single collapse
collapses the whole field score.

### 8.2 Coherence

```
coherence = 1 / (1 + avg_effective_resistance)
```

Bounded and monotonic.

### 8.3 Retrieval score

```
candidate_score = semantic_similarity
                * shell_fit
                * coherence_fit
                * continuity_fit
                * resonance_fit

risk   = contradiction_risk + drift_risk + policy_risk
safety = 1 / (1 + risk)

final_score = candidate_score * safety
```

### 8.4 Routing score

```
route_score = capability_fit
            * role_fit
            * locality_fit
            * shell_fit
            * expected_gain / expected_cost
```

## 9. Shell assignment rules

### 9.1 Promotion

**Event → Pattern** when:

1. recurrence crosses threshold
2. reuse crosses threshold
3. contradiction density crosses threshold
4. local stability holds across a window

**Pattern → Concept** when:

1. compression quality is high
2. support spans multiple contexts
3. contradiction risk stays low
4. witness linked summaries agree

**Concept → Principle** when:

1. policy owner approves
2. proof or contract reference exists
3. contradiction risk is near zero
4. reuse value justifies invariance

### 9.2 Demotion

Demote when support decays, contradiction risk grows, drift persists, or shell
oscillation repeats.

### 9.3 Phi scaling

Phi is a compression budget rule, not a geometry primitive.

```
Event     budget = B
Pattern   budget = B / φ
Concept   budget = B / φ²
Principle budget = B / φ³
```

## 10. Antipode logic

### 10.1 Geometric antipode

Contrastive companion built for every embedding. Used for search normalization
and novelty detection. Answers "what is nearby", "what is maximally unlike this",
and "what direction is the field drifting toward".

### 10.2 Semantic antipode

Explicit semantic opposites from one of four sources:

1. human labeled contradictions
2. policy contradictions
3. model detected opposition with explanation
4. witness linked historical reversals

Must carry: source, confidence, scope, policy overlap.

### 10.3 Contradiction frontier

Per query or active session, maintain:

1. top opposing nodes
2. top policy conflicts
3. unresolved semantic forks
4. confidence spread

This is the core anti hallucination layer for RAG and agent plans.

## 11. Retrieval pipeline

### 11.1 Candidate generation

Input: query embedding, shell policy, time window, context partition.

1. choose target shells
2. run ANN inside those shells
3. add temporal and witness neighbors
4. add top semantic antipodes of top candidates

### 11.2 Reranking

Rerank by semantic similarity, shell fit, coherence fit, continuity fit, policy
fit, contradiction risk, drift risk.

### 11.3 Output contract

Every retrieval result returns:

1. selected nodes
2. rejected nodes
3. contradiction frontier
4. explanation trace
5. witness refs where applicable

## 12. Drift detection

Four channels:

1. **Semantic drift** — centroid movement across recent windows
2. **Structural drift** — edge changes, cluster splits, new cut pressure zones
3. **Policy drift** — movement toward nodes with lower policy fit
4. **Identity drift** — changes in claimed role, capability use, or agent signature

Alert fires only when total drift crosses threshold **and at least two channels
agree**.

## 13. Routing model

### 13.1 Roles

1. constraint role
2. structuring role
3. synthesis role
4. verification role

Maps to Limit, Clarity, Bridge, and external validation.

### 13.2 Router inputs

1. active field deficits
2. coherence and cut pressure
3. contradiction frontier
4. agent capability fit
5. locality and partition cost
6. shell depth mismatch

### 13.3 Router outputs

1. target agent
2. target partition
3. expected gain
4. proof requirement
5. witness requirement
6. expiry

### 13.4 Router policy

Routing is a hint until it touches privileged state. The moment routing implies
partition migration, shared memory remap, policy mutation, device lease, or
external actuation, it must pass the existing proof system. RuVix fixes P1
below 1 µs, P2 below 100 µs, P3 deferred — routing must separate cheap
eligibility from expensive commitment.

## 14. Witness model

Extends existing RuVix Witness model. New witness events:

1. `FieldNodeCreated`
2. `FieldEdgeUpserted`
3. `AntipodeBound`
4. `ShellPromoted`
5. `ShellDemoted`
6. `ContradictionFlagged`
7. `RoutingHintIssued`
8. `RoutingHintCommitted`
9. `FieldSnapshotCommitted`

Only committed mutations need mandatory witnessing. Pure read queries remain
outside the privileged witness path unless in regulated mode.

## 15. Integration with current RuVector direction

- **`ruvector-sparsifier`** — compressed field graph for coherence sampling,
  contradiction frontier discovery, drift estimation at scale.
- **`ruvector-solver`** — local coherence, effective resistance, anomaly ranking,
  route gain estimation.
- **`ruvector-mincut`** — split hints, migration hints, fracture zone detection.
  Shell logic and semantic antipode search **do not** run inside the 50 µs
  mincut epoch budget initially.
- **RuVix** — exposes only hints at first: `PriorityHint`, `SplitHint`,
  `MergeHint`, `TierHint`, `RouteHint`.

## 16. Recommended crate boundaries

1. **Step 1** — `ruvector-field-types`: shared model and serialization
2. **Step 2** — `ruvector-field-core`: shell assignment, antipode binding, resonance, drift
3. **Step 3** — `ruvector-field-index`: query planner, candidate generation, contradiction frontier, reranker
4. **Step 4** — `ruvector-field-router`: role selection, gain estimation, hint issuance
5. **Step 5** — `ruvix-field-bridge`: adapter converting field hints into RuVix scheduler and partition hints

## 17. Failure modes

| Mode | Fix |
|---|---|
| Literal negation treated as semantic opposition | keep geometric and semantic antipodes separate |
| Shell oscillation | promotion hysteresis and minimum residence windows |
| Witness log explosion | witness only committed mutations and snapshot deltas |
| Kernel budget breach | keep field engine outside scheduler epoch until proven safe |
| Role overfitting | pluggable axes, benchmark per domain |
| Story debt from cosmology language | document only in terms of equivalence, shells, projection, coherence, policy |

## 18. Benchmark and acceptance test

On a contradiction heavy enterprise corpus with a long horizon agent benchmark,
the field engine graduates from user space to RuVix hints only if **all four**
pass:

1. contradiction rate improves by at least **20%**
2. retrieval token cost improves by at least **20%**
3. long session coherence improves by at least **15%**
4. enabling hints does **not** violate the 50 µs coherence epoch budget or the
   sub 10 µs partition switch target

## 19. Straight recommendation

1. Build this as a RuVector field layer first.
2. Use four shells, not 33.
3. Use semantic antipodes, not just vector negation.
4. Export hints into RuVix only after retrieval and routing benchmarks show a
   real gain.

This compounds the current RuVector trajectory instead of competing with it.
