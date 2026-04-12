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

### 8.5 Policy fit

`policy_fit` and `policy_risk` are referenced by §8.3 and §11 but need a
concrete definition. A `Policy` is an application-level object registered
with the field engine at startup; the engine evaluates a node against every
policy whose capability bits overlap the node's `policy_mask`.

**Data model.**

```rust
pub struct Policy {
    pub id: u64,
    pub name: String,
    /// Minimum axis scores the node must exceed to be fully aligned.
    /// Each field is in [0, 1].
    pub required_axes: AxisScores,
    /// Edge kinds that, if incident to the node, are considered a
    /// hard policy violation (fit collapses to 0).
    pub forbidden_edges: Vec<EdgeKind>,
    /// Bits that select which nodes this policy applies to. A policy
    /// applies when `node.policy_mask & policy.capability_mask != 0`.
    pub capability_mask: u64,
}
```

Policies live in a `PolicyRegistry` owned by the field engine and are
snapshot-included (§7) so that `policy_fit` values are reproducible from
a snapshot plus the registry version.

**Algorithm.** `policy_fit(node, registry) -> f32`:

```
function policy_fit(node, registry):
    let eps = 1e-6
    let mut worst = 1.0                                 # most restrictive wins
    let applicable = registry.policies.filter(|p|
        (node.policy_mask & p.capability_mask) != 0
    )
    if applicable.is_empty():
        return 1.0                                      # no applicable policy, fully aligned
    for p in applicable:
        # 1. Hard gate: any forbidden edge kind incident to the node zeroes fit.
        if node_has_incident_edge_kind(node, p.forbidden_edges):
            return 0.0
        # 2. Soft axis alignment: min over four axes.
        let r = p.required_axes
        let fit = min(
            clamp01((node.axes.limit   - r.limit)   / (1.0 - r.limit   + eps)),
            clamp01((node.axes.care    - r.care)    / (1.0 - r.care    + eps)),
            clamp01((node.axes.bridge  - r.bridge)  / (1.0 - r.bridge  + eps)),
            clamp01((node.axes.clarity - r.clarity) / (1.0 - r.clarity + eps)),
        )
        # 3. Compose with running worst (min across policies).
        worst = min(worst, fit)
    return worst

function policy_risk(node, registry):
    return 1.0 - policy_fit(node, registry)
```

The per-axis formula is the linear headroom between the node's axis score
and the policy's required minimum, normalized against the remaining range
`(1 - required + eps)`. A node exactly at the required minimum gets `0.0`
on that axis; a node at `1.0` gets `1.0`; values below the minimum clamp to
`0.0`. Taking the minimum across axes means a single collapsed axis
collapses the policy fit, mirroring the multiplicative intent of resonance.
Taking the minimum across policies means the most restrictive applicable
policy wins.

**Worked example.** Node `N` has axes `(limit=0.80, care=0.60, bridge=0.90,
clarity=0.50)`, `policy_mask = 0b0011`, no incident forbidden edges.
Registry has two policies:

- `P_A`: `required_axes = (0.50, 0.50, 0.50, 0.50)`, `capability_mask = 0b0001`, `forbidden_edges = []`
- `P_B`: `required_axes = (0.70, 0.70, 0.70, 0.40)`, `capability_mask = 0b0010`, `forbidden_edges = [Contrasts]`

Both policies apply (`N.policy_mask & 0b0001 != 0` and `N.policy_mask &
0b0010 != 0`).

`P_A` with `eps` elided for readability:

- limit:   `clamp01((0.80 - 0.50) / (1.0 - 0.50)) = clamp01(0.30 / 0.50) = 0.60`
- care:    `clamp01((0.60 - 0.50) / (1.0 - 0.50)) = clamp01(0.10 / 0.50) = 0.20`
- bridge:  `clamp01((0.90 - 0.50) / (1.0 - 0.50)) = clamp01(0.40 / 0.50) = 0.80`
- clarity: `clamp01((0.50 - 0.50) / (1.0 - 0.50)) = clamp01(0.00 / 0.50) = 0.00`
- `fit_A = min(0.60, 0.20, 0.80, 0.00) = 0.00`

`P_B`:

- limit:   `clamp01((0.80 - 0.70) / (1.0 - 0.70)) = clamp01(0.10 / 0.30) ≈ 0.333`
- care:    `clamp01((0.60 - 0.70) / (1.0 - 0.70)) = clamp01(-0.10 / 0.30) = 0.000`
- bridge:  `clamp01((0.90 - 0.70) / (1.0 - 0.70)) = clamp01(0.20 / 0.30) ≈ 0.667`
- clarity: `clamp01((0.50 - 0.40) / (1.0 - 0.40)) = clamp01(0.10 / 0.60) ≈ 0.167`
- `fit_B = min(0.333, 0.000, 0.667, 0.167) = 0.000`

`policy_fit(N) = min(fit_A, fit_B) = 0.000`, `policy_risk(N) = 1.000`.
Both policies happen to be blocked by different axes — `P_A` by `clarity`
exactly at the minimum and `P_B` by `care` below the minimum. The linear
headroom model makes the "just barely at the bar" case a zero deliberately:
policies should require strict headroom before a node is considered
aligned.

## 9. Shell assignment rules

### 9.1 Promotion

Every promotion edge is gated on four pinned conditions. A node advances only
when **all four** rows for its transition hold simultaneously at the same
promote-cycle tick. Values below are the v1 defaults; domain profiles may
override them but must stay monotonic across shells (tighter as depth
increases).

| From      | To        | Threshold             | Value                                                            | Window                    | Rationale                                                                    |
|-----------|-----------|-----------------------|------------------------------------------------------------------|---------------------------|------------------------------------------------------------------------------|
| Event     | Pattern   | recurrence            | ≥ 3 support edges (`Supports` ∪ `DerivedFrom`)                   | rolling 24 h              | prove the node has been referenced, not just stored                          |
| Event     | Pattern   | reuse                 | retrieved in ≥ 2 distinct contexts (distinct partition or agent) | rolling 24 h              | reuse across contexts, not a single hot loop                                 |
| Event     | Pattern   | contradiction density | `contrast_edges / total_edges ≤ 0.10`                            | all incident edges        | local contradiction must be low before lifting out of raw Event              |
| Event     | Pattern   | local stability       | `coherence ≥ 0.6` sustained for ≥ 2 consecutive promote-cycles   | 2 promote-cycles          | avoid promoting a transient spike                                            |
| Pattern   | Concept   | compression quality   | `resonance ≥ 0.35`                                               | current tick              | pattern must carry real multiplicative score, not just support count         |
| Pattern   | Concept   | support breadth       | support spans ≥ 3 distinct partitions                            | rolling 7 d               | concepts are cross-partition; a single-partition pattern is not a concept    |
| Pattern   | Concept   | contradiction risk    | `contrast_edges / total_edges ≤ 0.05`                            | all incident edges        | concept level must not carry live contradictions                             |
| Pattern   | Concept   | witness agreement     | ≥ 2 witness-linked summaries agree (cosine ≥ 0.85)               | current tick              | consolidation requires two independent witness-bound summaries to line up    |
| Concept   | Principle | policy owner approval | explicit capability present in `policy_mask` (owner-granted)     | point-in-time             | principles are policy; they do not self-promote                              |
| Concept   | Principle | proof reference       | `proof_ref` is non-null and validates against the proof store    | point-in-time             | principles must be backed by a concrete proof or contract                    |
| Concept   | Principle | contradiction risk    | `contrast_edges / total_edges ≤ 0.01`                            | all incident edges        | principles tolerate essentially no live contradictions                       |
| Concept   | Principle | reuse value           | ≥ 5 retrievals                                                   | rolling 7 d               | only durable, frequently reused concepts earn invariance                     |

All edge counts are taken from the relational index at tick time and use the
edge kinds declared in §6. The `total_edges` denominator for a node is the
count of all edges incident to it (in + out), not just `Supports` and
`Contrasts`. Ties at the threshold are resolved conservatively: equality
counts as passing only for the lower-bound conditions (≥) and only for the
upper-bound conditions (≤).

### 9.1.1 Promotion hysteresis

Promotion and demotion are throttled by two mechanisms that together make
shell assignment stable and auditable.

1. **Minimum residence time.** A node must reside in its current shell for
   at least `MIN_RESIDENCE_NS` before any shell change is permitted. The
   constant is per-shell:

   | Shell     | `MIN_RESIDENCE_NS` |
   |-----------|--------------------|
   | Event     | 5 min              |
   | Pattern   | 1 h                |
   | Concept   | 24 h               |
   | Principle | 7 d                |

   Residence is measured from the timestamp of the last shell transition
   (or ingest time for the initial shell). Any attempted promotion or
   demotion before the residence window is skipped silently — no witness
   event, no error.

2. **Hysteresis window.** `HYSTERESIS_WINDOW` is fixed at **4 consecutive
   promote-cycles**. A node is only eligible for a shell change after it
   has satisfied the target transition's conditions (or, for demotion, its
   current shell's demotion conditions) on every tick inside the window.
   The window is a sliding counter per node; any tick that fails the
   condition resets the counter to zero.

3. **Oscillation ban.** If a node has moved shells ≥ 2 times within a
   `HYSTERESIS_WINDOW`, further shell changes are **blocked** until the
   window resets — i.e. until at least `HYSTERESIS_WINDOW` consecutive
   ticks pass with zero shell transitions. Blocked transitions emit a
   `ShellOscillationSuppressed` diagnostic (not a witness event; see §14)
   so operators can tune thresholds per domain.

Hysteresis applies symmetrically to promotion and demotion. The combination
of minimum residence, a 4-tick sliding window, and the oscillation ban makes
it impossible for a node to cross more than two shell boundaries inside a
single hysteresis window.

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

### 14.1 Witness cursor semantics

`WitnessCursor` is the single source of truth for "what has this engine
committed, and in what order". It appears in snapshot metadata (§7) and in
the diff protocol below; its contract is:

- **Type.** `WitnessCursor` is a monotonically increasing `u64` assigned at
  the point of event emission inside the engine. Cursor values never
  decrease and never repeat.
- **Gap-free allocation.** Every committed mutation is assigned exactly one
  cursor value with no skipped integers. A mutation that is rolled back
  before commit never consumes a cursor value. A mutation that commits then
  fails to append to the witness log is a panic condition — the engine is
  poisoned and must be recovered from the previous snapshot.
- **Scope.** The cursor is global per `FieldEngine` instance: one counter,
  shared across all shells, partitions, and node kinds. Sharding the
  cursor is explicitly out of scope for v1.
- **Reads do not advance.** `retrieve`, `drift`, `route`, and any pure
  observation path must not allocate cursor values. Only committed
  mutations emit witness events and only witness events advance the
  cursor.
- **Snapshot semantics.** A snapshot captures `high_cursor`, the largest
  cursor value observed at the moment of snapshot commit. Any event with
  cursor `≤ high_cursor` is guaranteed to be reflected in the snapshot
  state; any event with cursor `> high_cursor` is not.
- **Diffs.** Given two snapshots `S_low` and `S_high`, the exact event
  range replayed to reconstruct the state delta is the half-open interval
  `(S_low.high_cursor, S_high.high_cursor]`. This is closed on the right
  so the diff ends at a well-defined snapshot boundary.
- **Ordering and concurrency.** Cursor order must match happens-before.
  The v1 implementation enforces this by **serializing all mutating
  operations through a single async actor** (`FieldEngineActor`) that
  owns the cursor counter and the mutation log. Reads may proceed in
  parallel against an immutable view. The actor model is chosen over a
  mutex because it lets the mutation log and the cursor advance be a
  single atomic step inside the actor's message handler; a mutex
  implementation would need a separate acquire/release around the log
  append, which would leak the gap-free invariant under panic.
- **Recovery.** On restart, the engine reads the highest cursor value from
  the last witness log segment and sets its internal counter to
  `high_cursor + 1`. Any partially committed event (log entry present but
  mutation not reflected in the snapshot) is re-applied before accepting
  new mutations.

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

### 15.5 Relationship to existing coherence signals

§5.4 says the field signals "extend, not replace" the existing coherence
signals. Spelled out: RuVector already exposes two coherence-shaped
quantities and this spec introduces a third. They are not interchangeable
and must not be collapsed in client code.

- **`CoherenceScore`** (existing, from the RuVix coherence engine) —
  structural coherence of a **partition**, computed from the graph's
  effective-resistance profile. Produced inside the 50 µs coherence epoch.
- **`CutPressure`** (existing) — readiness of a partition to split.
  Produced by the same coherence epoch, consumed by the mincut pass.
- **`FieldCoherence`** (new, this spec, §8.2) — semantic coherence of a
  **node**, computed from same-shell neighborhood similarity. Produced
  outside the scheduler epoch in the field engine.

The three signals live on different objects (partition / partition / node)
and carry different meaning (structural / readiness / semantic). The
composition rules below are the only sanctioned way to combine them.

**Composition rules.**

- For a **node** `n` that also belongs to a partition `P`:

  ```
  effective_coherence(n) = min(
      CoherenceScore(P),     # structural floor from the kernel
      FieldCoherence(n),     # semantic floor from the field engine
  )
  ```

  `min` is deliberate: a node is only as coherent as its weakest channel.
  A node with a high semantic score sitting inside a structurally
  fractured partition is not coherent, and vice versa.

- For a **partition** `P`:

  ```
  semantic_fracture(P) = fraction of nodes in P with
                         semantic_antipode != None
  effective_pressure(P) = CutPressure(P)
                        + clamp01(semantic_fracture(P)) * 0.3
  ```

  The field-side bonus is bounded at `0.3` so the field engine cannot by
  itself push `effective_pressure` past the kernel's split threshold —
  the kernel always gets the final call. `0.3` is set so that a partition
  that is 100 % semantically fractured but has zero structural pressure
  still needs some structural pressure to cross the split threshold.

**Precedence.** Kernel decisions — partition assignment, cut
authorization, tier migration — use `CoherenceScore` and `CutPressure`
**directly**. Field signals are advisory: they must pass the proof gate
(§13.4) before they can influence kernel state at all. The field engine
may store `effective_coherence` and `effective_pressure` in snapshots for
observability, but the kernel ignores those values on its hot path.

**Conflict resolution.** When field and kernel signals disagree — for
example, `FieldCoherence(n) = 0.9` but `CoherenceScore(P) = 0.2`, or
`semantic_fracture(P) = 0.8` but `CutPressure(P) = 0.0` — the kernel wins.
The field engine emits a `ContradictionFlagged` witness event (§14) whose
payload records both signals, the partition id, and the node id (when
applicable). Operators can then decide whether to retune field thresholds,
kernel thresholds, or both. The field engine does not auto-tune.

## 16. Recommended crate boundaries

1. **Step 1** — `ruvector-field-types`: shared model and serialization
2. **Step 2** — `ruvector-field-core`: shell assignment, antipode binding, resonance, drift
3. **Step 3** — `ruvector-field-index`: query planner, candidate generation, contradiction frontier, reranker
4. **Step 4** — `ruvector-field-router`: role selection, gain estimation, hint issuance
5. **Step 5** — `ruvix-field-bridge`: adapter converting field hints into RuVix scheduler and partition hints

### 16.1 Bridge contract: field hints → RuVix hints

The `ruvix-field-bridge` crate is the single adapter point where field
engine state crosses into RuVix. It owns the mapping table below, the
proof-gate invocation, and the bounded channel to RuVix. No other crate
is allowed to synthesize RuVix hints from field state.

| Field signal                                         | RuVix hint     | Condition                                                                                  | Payload                                          | Mode   | TTL          | Witness event emitted on conversion |
|------------------------------------------------------|----------------|---------------------------------------------------------------------------------------------|--------------------------------------------------|--------|--------------|-------------------------------------|
| High resonance on a node                             | `PriorityHint` | `node.resonance > 0.7`                                                                      | `(node_id: u64, priority_delta: f32)`            | stream | 1 epoch      | `RoutingHintIssued`                 |
| `semantic_antipode` bound + high partition fracture  | `SplitHint`    | partition `semantic_fracture > 0.25` (see §15.5)                                            | `(partition_id: u64, suggested_cut_edges: Vec<(u64,u64)>)` | batch  | 4 epochs     | `RoutingHintIssued`                 |
| Frequent `Refines` edges between partitions          | `MergeHint`    | `cross_partition_refines / total_refines > 0.3` for the partition pair over a 1 h window    | `(partition_a: u64, partition_b: u64)`           | batch  | 8 epochs     | `RoutingHintIssued`                 |
| Shell demotion with cold access pattern              | `TierHint`     | `shell < Pattern` **and** `now - node.last_access > 1 h`                                    | `(node_id: u64, target_tier: Tier)`              | batch  | 16 epochs    | `RoutingHintIssued`                 |
| Routing hint with `gain_estimate > 0.5`              | `RouteHint`    | `route_score > 0.5` **and** passes `ProofGate::authorize`                                   | `(agent_id: u64, partition_id: u64, ttl: u16)`   | stream | `hint.ttl_epochs` | `RoutingHintCommitted`          |

Notes on the table:

- **Precondition strictness.** Every field-side precondition is
  re-evaluated at bridge `tick()` time, not at the time the underlying
  field state was produced. A hint that was eligible at epoch `n` but
  fails re-evaluation at epoch `n+1` is dropped silently.
- **Payload types.** All payload types are the existing RuVix hint
  structs. The bridge never defines new RuVix-facing types; it only
  produces values for types already exported by RuVix.
- **Mode.** `stream` hints are emitted one at a time as soon as their
  condition flips true. `batch` hints accumulate across a bridge tick
  and are drained at the end of the tick into a single `Vec<Hint>`
  message on the channel. `PriorityHint` and `RouteHint` are streamed
  because they are time-sensitive; the rest are batched because they
  are derived from partition-level aggregates that are cheaper to
  compute once per tick.
- **TTL.** TTLs are in RuVix coherence epochs, not wall clock. The TTL
  is carried on the hint so RuVix can discard stale hints without
  consulting the bridge.
- **Witness.** Every conversion emits exactly one witness event on the
  field-engine side before the hint is enqueued. `RouteHint` is the only
  case that emits `RoutingHintCommitted` on enqueue, because that is the
  moment the field engine considers the hint "handed off"; RuVix may
  still reject it downstream and will emit its own witness in that case.
- **Proof gate.** Only `RouteHint` is proof-gated on conversion because
  it can cause partition migration or actuation (§13.4). The other four
  hints are advisory-only on the RuVix side and do not touch privileged
  state until RuVix itself decides to act on them, at which point RuVix
  runs its own proof gate.

**Bridge control loop.** The bridge runs a periodic `tick()` driven by an
external scheduler — one tick per RuVix coherence epoch is the default,
but the bridge does not itself run inside the 50 µs epoch budget. Shape:

```
function tick(engine: &FieldEngine, registry: &PolicyRegistry,
              proof_gate: &ProofGate, out: &BoundedChannel<Vec<Hint>>):
    let mut batch: Vec<Hint> = Vec::new()

    # 1. Read-only snapshot of engine state. No mutations here.
    let state = engine.observe()

    # 2. Walk the mapping table top to bottom.
    for row in MAPPING_TABLE:
        for source in state.sources_for(row):
            if not row.condition(source, state, registry):
                continue
            let hint = row.build_payload(source, state)
            if row.proof_gated:
                if not proof_gate.authorize(&hint):
                    continue
            engine.emit_witness(row.witness_event(&hint))
            if row.mode == Stream:
                out.try_send(vec![hint])   # bounded, non-blocking
            else:
                batch.push(hint)

    # 3. Drain the batch into one message.
    if !batch.is_empty():
        out.try_send(batch)                # bounded, non-blocking
```

The channel is bounded and `try_send` is non-blocking: if RuVix is
backpressured, the bridge drops the oldest batch and logs a
`BridgeBackpressure` diagnostic. Dropping hints is always safe because
every hint is advisory and carries a TTL. The bridge never retries
dropped hints; the next `tick()` will re-derive current state and emit a
fresh hint if the precondition still holds.

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

## 20. Appendix: benchmark corpus RuField-Bench-v1

The acceptance gate in §18 names four metric thresholds but does not name a
corpus. This appendix defines the v1 corpus against which those thresholds
are measured. The corpus is deterministic, synthetic, and self-contained so
benchmark runs are reproducible across machines and dates.

### 20.1 Composition

`RuField-Bench-v1` is a fixed mix of 1390 items drawn from a single
synthetic domain: **enterprise authentication and session management**.
The domain is narrow on purpose — contradictions must be semantically
meaningful for the benchmark to measure anything.

| Category              | Count | Shell target    | Role in the benchmark                                 |
|-----------------------|-------|-----------------|-------------------------------------------------------|
| Event interactions    | 1000  | Event           | raw observations, logs, user reports, tool calls      |
| Pattern summaries     | 200   | Pattern         | recurring motifs and local clusters                   |
| Concept summaries     | 100   | Concept         | durable working theories about auth behavior          |
| Principle policies    | 30    | Principle       | canonical policies with proof references              |
| Explicit contradictions | 50  | Event / Pattern | contradiction pairs with a `bind_semantic_antipode`   |
| Policy conflicts      | 10    | Principle       | mutually exclusive policy pairs                       |

The contradiction and policy-conflict counts are in addition to the other
rows — i.e. the 50 explicit contradictions are attached to nodes already
counted in the Event and Pattern rows.

### 20.2 Distribution

Within the 1000 Event interactions the distribution is:

- **60 % canonical** — axis-aligned with a single internally-consistent
  "correct" theme about session refresh, idle timeout, and OAuth flow
- **25 % drifting** — gradually moving away from the canonical centroid
  across a simulated 30 day span to exercise the drift detector
- **10 % contradicting** — directly opposed to a canonical node, half of
  which have an explicit `semantic_antipode` bound, half of which do not
  (so both detection modes are tested)
- **5 % policy-violating** — axes positioned below the requirements of at
  least one registered policy, to exercise `policy_fit = 0`

### 20.3 Metric definitions

All four acceptance metrics are measured precisely as follows. Each
definition assumes a retrieval with `top_k = k` and a query set of `Q`
queries drawn from the canonical theme.

- **Contradiction rate.** For each query, count the pairs of selected
  results that contradict each other. A pair is contradictory iff either
  result appears on the other's contradiction frontier under §10.3. The
  per-query contradiction rate is `contradictory_pairs / k`; the corpus
  metric is the mean over `Q` queries. A value of `0.0` means no
  selected pair contradicts any other; a value of `1.0` means every
  selected pair contradicts. Improvement is reported as percentage
  reduction against baseline.

- **Retrieval token cost.** For each query, `cost = sum(node.text.len()
  for node in selected) + 50 * |selected|`. The 50-byte per-node overhead
  models the framing cost of serializing a result (id, scores,
  explanation pointer) into a prompt. The corpus metric is the mean
  `cost` over `Q` queries. Improvement is reported as percentage
  reduction against baseline.

- **Long-session coherence.** Simulate a 100-query rolling session
  drawn from the canonical theme. For each query, compute
  `per_query_resonance = mean(node.resonance for node in selected)`.
  The corpus metric is the mean `per_query_resonance` across the 100
  queries. Improvement is reported as percentage increase against
  baseline.

- **Latency.** Time `retrieve` end-to-end for each query in
  microseconds. Report `p50` and `p99` across `Q` queries. The acceptance
  gate does not set a latency threshold directly; it requires that
  enabling field hints does not violate the kernel's 50 µs epoch budget
  or the sub-10 µs partition switch target when field engine is running
  alongside.

### 20.4 Baseline

The baseline against which all four metrics are measured is **naive
top-k cosine** on the same embeddings:

- Linear scan across all nodes regardless of shell
- Rank by raw cosine similarity to the query
- No shell filtering, no antipodes, no reranking, no contradiction
  frontier
- No policy filtering, no drift adjustment

The baseline is trivial on purpose: it is the simplest thing that
retrieves, and any improvement the field engine claims must hold against
it.

### 20.5 Thresholds (restated from §18 with definitions plugged in)

The field engine graduates from user space to RuVix hints only if **all
four** hold on `RuField-Bench-v1` against the naive baseline:

1. `contradiction_rate` improves by ≥ **20 %** (reduction)
2. `retrieval_token_cost` improves by ≥ **20 %** (reduction)
3. `long_session_coherence` improves by ≥ **15 %** (increase)
4. Enabling hints does **not** violate the 50 µs coherence epoch budget
   or the sub 10 µs partition switch target

### 20.6 Reproduction

The corpus is generated from a single fixed seed so results are
bit-reproducible. Pseudocode:

```
const SEED: u64 = 0xRUFIELD_BENCH_V1      # fixed, see corpus generator
const DIM:  usize = 128                    # embedding dimension for the benchmark

function generate_corpus(seed: u64) -> Corpus:
    let rng = SeedableRng::from_seed(seed)
    let canonical_axis = random_unit_vector(rng, DIM)     # the "correct" theme
    let mut corpus = Corpus::new()

    # 1. Canonical events (600)
    for i in 0..600:
        let v = jitter(canonical_axis, sigma=0.10, rng)
        corpus.push_event(text=template_event(i), embedding=v,
                          axes=(0.8, 0.8, 0.8, 0.8),
                          policy_mask=0b0001)

    # 2. Drifting events (250) — centroid slides over simulated time
    for i in 0..250:
        let t = i as f32 / 250.0                          # 0 → 1 across the span
        let drift = lerp(canonical_axis, random_unit_vector(rng, DIM), t * 0.4)
        let v = jitter(drift, sigma=0.10, rng)
        corpus.push_event(text=template_drift(i), embedding=v,
                          axes=(0.7 - 0.2*t, 0.7, 0.6, 0.6),
                          policy_mask=0b0001,
                          ts_offset_days=i*30/250)

    # 3. Contradicting events (100)
    for i in 0..100:
        let v = jitter(-canonical_axis, sigma=0.10, rng)
        let node = corpus.push_event(text=template_contradict(i), embedding=v,
                                     axes=(0.3, 0.3, 0.3, 0.3),
                                     policy_mask=0b0001)
        if i < 50:
            corpus.bind_semantic_antipode(node, corpus.canonical_peer(i))

    # 4. Policy-violating events (50)
    for i in 0..50:
        let v = jitter(canonical_axis, sigma=0.15, rng)
        corpus.push_event(text=template_policy_violation(i), embedding=v,
                          axes=(0.1, 0.1, 0.1, 0.1),             # below required
                          policy_mask=0b0010)

    # 5. Pattern / Concept / Principle summaries
    for i in 0..200: corpus.push_pattern_summary(i, rng)
    for i in 0..100: corpus.push_concept_summary(i, rng)
    for i in 0..30:  corpus.push_principle_policy(i, rng)

    # 6. Policy conflicts (10 pairs)
    for i in 0..10:
        corpus.push_conflicting_policy_pair(i, rng)

    return corpus
```

`jitter(v, sigma, rng)` adds Gaussian noise with standard deviation
`sigma` per component and re-normalizes. `template_*` functions are
deterministic string builders keyed on `i`. The reference
implementation lives under `benches/rufield_bench_v1.rs` when this
spec is implemented; until then, the pseudocode above is the
specification of record.

---

## Revision history

- **v1.1 (2026-04-12):** pinned promotion thresholds, added policy fit
  algorithm, witness cursor semantics, benchmark corpus, coherence
  composition rules, bridge contract.
- **v1.0 (2026-04-12):** initial draft.
