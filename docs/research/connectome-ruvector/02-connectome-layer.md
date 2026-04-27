# 02 - Connectome Layer: FlyWire Ingest and Graph Schema

> Framing reminder: this is a graph-native embodied connectome runtime. No upload, no consciousness claims. See `./00-master-plan.md` §1 and `./07-positioning.md`.

## 1. Purpose

Specify the node and edge schema for a Drosophila whole-brain connectome persisted in `ruvector-graph`, the ingest pipeline from FlyWire's public release, and the cost/throughput envelope. Consumers: `./03-neural-dynamics.md` reads this schema to wire the LIF kernel; `./05-analysis-layer.md` reads it to mount mincut/sparsifier/coherence.

## 2. Source dataset: FlyWire

FlyWire is the community-proofread adult female Drosophila melanogaster brain connectome derived from serial-section electron microscopy. The v783 release (Dorkenwald et al., 2024, Nature; Matsliah et al., 2024, Nature) provides approximately 139,255 neurons and 54.5 million chemical synapses with predicted neurotransmitter identity for ~130M synaptic predictions (consolidated to per-edge aggregates). Key tables published:

- `neurons.csv` — per-neuron metadata (id, super-class, class, sub-class, cell-type, hemilineage, side, nerve, soma position).
- `connections.csv` — pre→post pairs with synapse count, neuropil, predicted neurotransmitter.
- `classification.csv` — cell-type assignments with community-voted labels.
- `meshes/` — per-neuron triangle meshes (optional for morphology hashing).
- `nt_predictions.csv` — per-synapse NT predictions (ACh, Glu, GABA, DA, 5-HT, OA, histamine).

The Janelia hemibrain (`v1.2.1`, Scheffer et al., 2020) covers roughly half the brain (~25K neurons) with higher manual proof-reading density. FlyWire is the primary source; hemibrain is kept as a cross-validation target (see `./06-prior-art.md` §Hemibrain).

The 2024 Nature whole-fly-brain LIF paper is the ground-truth proof that behavior — feeding, grooming, and sensorimotor transformations — can emerge from a FlyWire-scale LIF model with no trained parameters. Our schema must preserve every feature that paper depended on: cell-type, neurotransmitter, synapse count per edge, and neuropil labels.

## 3. Graph schema

We use `ruvector-graph` labeled property graph with typed nodes and edges. Schema is versioned (`schema_version = "connectome/2026.04"`) and stored in the graph root properties for replay.

### 3.1 Node: Neuron

```rust
pub struct Neuron {
    pub id: NeuronId,              // u64, stable FlyWire root_id
    pub dataset: DatasetId,        // FlyWire | Hemibrain | Custom
    pub dataset_version: String,   // e.g., "flywire-v783"
    pub super_class: SuperClass,   // Central | OpticLobe | Ascending | Descending | Motor | SensoryPeriph
    pub class: Option<ClassId>,    // e.g., "Kenyon cell"
    pub sub_class: Option<String>,
    pub cell_type: Option<CellTypeId>,
    pub hemilineage: Option<String>,
    pub side: Side,                // Left | Right | Center | Bilateral
    pub region: RegionId,          // interned neuropil label (MB, EB, FB, LAL, ...)
    pub soma_xyz: Option<[f32; 3]>,
    pub neurotransmitter: NT,      // ACh|Glu|GABA|DA|5-HT|OA|Hist|Unknown
    pub nt_confidence: f32,        // [0,1]
    pub morphology_hash: Option<u64>, // LSH over skeleton or mesh
    pub flags: NeuronFlags,        // bitflags: Motor, Sensory, ProofEdited, Flagged, ...
}
```

`NeuronId` is 64-bit, globally unique across datasets via `(dataset, flywire_root_id)` pair mixed into a SipHash. The `flags` bitfield is the hinge for layers 2-4: `Motor`, `Sensory`, `VisualPR` (photoreceptor), `Chemosensory`, `Mechanosensory`, `Chordotonal`, etc. These flags are what `BodySim` and `DynamicsEngine` key off when routing sensory injection and motor readout.

Interning: `RegionId`, `CellTypeId`, `ClassId` are `u32` indices into intern tables stored as properties on the graph root. Keeps each `Neuron` under 120 bytes.

### 3.2 Edge: Synapse

```rust
pub struct Synapse {
    pub pre: NeuronId,
    pub post: NeuronId,
    pub neuropil: RegionId,
    pub nt: NT,
    pub sign: i8,                  // +1 excitatory, -1 inhibitory, 0 unknown/graded
    pub weight: f32,               // initial effective weight (count * gain)
    pub count: u32,                // raw synapse count from FlyWire
    pub delay_ms: f32,             // estimated axonal + synaptic delay
    pub confidence: f32,           // [0,1]
    pub weight_source: WeightSource, // Explicit | NtDefault | MorphologyEst
    pub edge_flags: EdgeFlags,     // Gap, Electrical, Recurrent, LongRange, ...
}
```

`sign` is derived from `nt`: ACh/Glu default to +1 in central brain circuits, GABA to -1, Glu in the optic lobe frequently +1 with known local exceptions. Where the sign is not safely inferable we set `sign = 0` and `weight_source = NtDefault`; the LIF kernel treats these as excitatory for the first pass and exposes the set for sensitivity analysis.

`delay_ms` is a hard problem. FlyWire does not publish conduction delays. We estimate as `delay_ms = base + k * soma_distance_microns` with `base ≈ 1.0 ms` and `k ≈ 0.003 ms/µm` (fly axonal conduction ~300 µm/ms), clamped to `[0.5, 20.0]`. Where neuron meshes are absent we fall back to `delay_ms = 2.0`. The field is explicit so it can be recalibrated per-region without schema change.

`EdgeFlags::Gap` marks electrical synapses (from gap-junction datasets where available; sparse in FlyWire but non-zero). `EdgeFlags::Recurrent` is set after a topological pass so layer 2 can optimize event handling for strongly connected components.

### 3.3 Hyperedges: motifs

`ruvector-graph::Hyperedge` captures discovered motifs (winner-take-all triplets, feedforward inhibition triads, reciprocal pairs). Populated by layer 4. Schema:

```rust
pub struct Motif {
    pub kind: MotifKind,           // WTA | FFI | Reciprocal | Custom(u32)
    pub members: Vec<NeuronId>,
    pub confidence: f32,
    pub discovered_at: Time,
    pub supporting_edges: Vec<EdgeId>,
}
```

Motifs are side-channels, not part of the runtime dynamics. They exist so analyses survive restarts and so `./05-analysis-layer.md` can index them in AgentDB.

### 3.4 Indexes

Required secondary indexes on the graph:

- `by_region: RegionId → Vec<NeuronId>` (scan by neuropil).
- `by_class: ClassId → Vec<NeuronId>`.
- `by_nt: NT → Vec<NeuronId>`.
- `motor_neurons: HashSet<NeuronId>` (flags bit test cached).
- `sensory_by_modality: Modality → Vec<NeuronId>`.
- `outgoing_csr: CSR<NeuronId, EdgeId>` (hot path for event dispatch in layer 2).
- `incoming_csr: CSR<NeuronId, EdgeId>` (for backward push / analysis).

`ruvector-graph::index` already supports property indexes; the CSR pair is a derived view materialized at ingest and refreshed on mutation.

## 4. Ingest pipeline

```
 FlyWire release ──┐
  (csv + meshes)   │
                   ▼
         ┌──────────────────────┐
         │ flywire-loader       │  (Rust, streaming CSV, no Python)
         │  · validate schema   │
         │  · intern region/type│
         │  · predict sign/delay│
         │  · hash morphology   │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ graph_writer         │  (ruvector-graph transactions)
         │  · batched Node insert
         │  · batched Edge insert (CSR-friendly order)
         │  · build indexes
         │  · materialize CSR
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ agentdb_embedder     │  (per-neuron vector)
         │  · ONNX MiniLM L6 v2 │
         │  · DiskANN index     │
         └──────────┬───────────┘
                    ▼
         rvf on-disk snapshot (dataset_hash captured)
```

The loader is a new Rust binary under `crates/ruvector-connectome/src/bin/flywire-loader.rs`. It streams FlyWire CSVs through `csv` + `serde`, builds `Neuron` / `Synapse` records, looks up interned IDs, and emits batched transactions into `GraphDB`. Batch size is 10K edges per transaction to keep WAL writes amortized.

Neurotransmitter → sign mapping table:

| NT | Default sign | Notes |
|---|---|---|
| ACh | +1 | Typical fast excitation in Drosophila central brain |
| Glu | +1 in most central circuits; context-dependent in optic lobe | Flagged for per-region override |
| GABA | -1 | Fast inhibition |
| DA | 0 (neuromodulatory) | Weight propagates via slow pool, not fast LIF |
| 5-HT | 0 (neuromodulatory) | Same |
| OA | 0 (neuromodulatory) | Same |
| Histamine | -1 | Photoreceptor output |
| Unknown | 0 | `weight_source = NtDefault`, excitatory fallback for v1 |

Neuromodulators are *not* routed through the event-driven LIF dispatcher in v1; they are aggregated into slower per-region concentration fields (see `./03-neural-dynamics.md` §Neuromodulation).

### 4.1 Morphology hashing

`morphology_hash` is an optional 64-bit LSH fingerprint of the per-neuron mesh or skeleton, built with an adapted version of `ruvector-cnn`'s locality-sensitive hashing pipeline. The hash lets AgentDB answer "neurons morphologically similar to X" without re-running mesh comparison. For v1 we can skip meshes and derive the hash from the tuple `(cell_type, region, side, hemilineage)` — crude, but useful until proper mesh embeddings are available.

## 5. Scale and cost analysis

### 5.1 Raw record sizes

| Kind | Fields | Bytes/record (packed) |
|---|---|---|
| Neuron | id, flags, enums, soma, NT, morph hash | ~112 |
| Synapse | pre, post, neuropil, NT, sign, weight, count, delay, conf, flags | ~56 |
| Motif | kind, 4-8 members, confidence | ~128 |

### 5.2 Totals (v1 FlyWire v783)

- Neurons: 139,255 × 112 B ≈ **15.6 MB** raw.
- Synapses (consolidated to per-edge): ~50 M × 56 B ≈ **2.8 GB** raw.
- CSR indexes (outgoing + incoming): ~2 × (139K × 8 B + 50M × 12 B) ≈ **1.2 GB**.
- Embeddings: 139K × 384 × f32 ≈ **214 MB**; with INT8 DiskANN ≈ **53 MB**.
- Motif store: bounded; target <100 MB.

Total on-disk budget: **~5 GB** for a full replay bundle. That is trivially SSD-resident and fits on the pi-brain node class (ADR-150). RAM working set for a run: ~3-4 GB with CSR warm plus LIF state (see `./03-neural-dynamics.md` §Memory).

### 5.3 Ingest throughput

A single-threaded loader on a modern laptop CPU should hit ~150K edges/s in Rust streaming CSV mode (bounded by CSV parsing, not graph writes). At 50M edges: **~5-6 minutes** for the full connectome. GraphDB transaction batching in `ruvector-graph` can absorb this without WAL blowup; we set `batch_size = 10_000` and use `IsolationLevel::ReadCommitted` during bulk ingest to avoid holding a global lock.

Consistency check after ingest:

- `node_count == published_node_count` (exact).
- `edge_count` within ±1% of published (synapse consolidation varies).
- Every `cell_type` referenced in edges resolves to a `Neuron` (no dangling FKs).
- Every NT prediction has `confidence >= 0.0 && <= 1.0`.

### 5.4 Query envelope

| Query | Expected latency (warm cache) |
|---|---|
| `neuron(id)` | <5 µs |
| `outgoing(id)` via CSR | O(deg) unbounded; p99 ~20 µs for avg degree |
| `by_region(region)` | 0.5-2 ms for largest neuropils |
| `by_nt(nt)` | <1 ms |
| Motif lookup (indexed) | <1 ms |
| Vector neighbor "neurons morphologically similar" via DiskANN | <10 ms @ k=50 |
| Full adjacency snapshot for sparsifier rebuild | ~200 ms single-thread, ~50 ms with rayon |

These numbers are inside the budget the architecture doc (`./01-architecture.md`) sets for a 25 Hz control-rate closed-loop run.

## 6. Incremental updates

Proof-reading and new FlyWire releases will change edges. The loader supports delta mode:

- `--since v780 --until v783` ingests only new/changed edges.
- Each neuron and edge carries `source_version`; queries can filter by version.
- `ruvector-mincut` and `ruvector-sparsifier` both support dynamic insert/delete; a FlyWire delta triggers incremental updates rather than full rebuild.

## 7. Cross-dataset support

Hemibrain is the obvious cross-validation target. The schema already supports multi-dataset because `NeuronId` is `(dataset, flywire_root_id)`-hashed. Loader: `hemibrain-loader` mirrors `flywire-loader` over `neuPrint`-exported CSVs. OpenWorm's `C. elegans` connectome (302 neurons, ~7K synapses) trivially fits and is useful as a sanity test bed (one run completes in seconds).

## 8. Data governance

- FlyWire citation attached to every replay bundle manifest.
- No proprietary data. No re-distribution of FlyWire meshes beyond project-internal storage.
- `nt_confidence < 0.5` edges are flagged in `EdgeFlags::LowConfidenceNT` so analyses can exclude them.
- Loader emits a manifest: `{dataset_version, loader_version, ingest_utc, node_count, edge_count, schema_hash}` so every downstream run is traceable.

## 9. Open questions for Phase 1

1. Should dendritic compartments (reduced multi-compartment per neuron) be modeled here or pushed into layer 2 state? The schema supports it via synthetic child nodes but doubles node count. Recommendation: defer to v2; use `ruvector-nervous-system::dendrite` in layer 2 for coincidence detection without schema changes.
2. Should gap junctions be distinct hyperedges or regular edges with `EdgeFlags::Gap`? We pick flags for simpler ingest; revisit if electrical coupling is required for a target behavior.
3. Neuromodulatory edges — keep as synapses or route to a separate region-level diffusion field? We keep them as synapses with `sign = 0` and let layer 2 route them to the slow pool.
4. Morphology hash provider — pure `(cell_type, region, side)` crude, or real mesh embedding from `ruvector-cnn`? Start crude, upgrade in M2.

See `./03-neural-dynamics.md` for how the LIF kernel consumes this schema, and `./05-analysis-layer.md` for the analyses that depend on the CSR indexes specified here.
