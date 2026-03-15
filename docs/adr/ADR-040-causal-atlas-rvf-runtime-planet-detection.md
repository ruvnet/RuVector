# ADR-040: Causal Atlas RVF Runtime — Planet Detection & Life Candidate Scoring

**Status:** Proposed
**Date:** 2026-02-18
**Author:** System Architect (AgentDB v3)
**Supersedes:** None
**Related:** ADR-003 (RVF Format), ADR-006 (Unified Self-Learning RVF), ADR-007 (Full Capability Integration), ADR-008 (Chat UI RVF)
**Package:** `@agentdb/causal-atlas`

## Context

ADR-008 demonstrated that a single RVF artifact can embed a minimal Linux
userspace, an LLM inference engine, and a self-learning pipeline into one
portable file. This ADR extends that pattern to scientific computing: a
portable RVF runtime that ingests public astronomy and physics datasets,
builds a multi-scale interaction graph, maintains a dynamic coherence field,
and emits replayable witness logs for every derived claim.

The design draws engineering inspiration from causal sets, loop-gravity-style
discretization, and holographic boundary encoding, but it is implemented as a
practical data system, not a physics simulator. The holographic principle
manifests as a concrete design choice: primarily store and index boundaries,
and treat interior state as reconstructable from boundary witnesses and
retained archetypes.

### Existing Capabilities (ADR-003 through ADR-008)

RVF segments, HNSW indexing, SHAKE-256 witness chains, graph transactions,
768-dim SIMD embeddings, SONA learning, federated coordination, contrastive
training, adaptive index tuning, kernel embedding (ADR-008), and lazy model
download (ADR-008). See ADR-003 through ADR-008 for full API details.

### What This ADR Adds

1. Domain adapters for astronomy data (light curves, spectra, galaxy catalogs)
2. Compressed causal atlas with partial-order event graph
3. Coherence field index with cut pressure and partition entropy
4. Multi-scale interaction memory with budget-controlled tiered retention
5. Boundary evolution tracker with holographic-style boundary-first storage
6. Planet detection pipeline (Kepler/TESS transit search)
7. Life candidate scoring pipeline (spectral disequilibrium signatures)
8. Progressive data download from public sources on first activation

## Goal State

A single RVF artifact that boots a minimal Linux userspace, progressively
downloads and ingests public astronomy and physics datasets on first
activation (lazy, like ADR-008's GGUF model download), builds a multi-scale
interaction graph, maintains a dynamic coherence field, and emits replayable
witness logs for every derived claim.

### Primary Outputs

| # | Output | Description |
|---|--------|-------------|
| 1 | **Atlas snapshots** | Queryable causal partial order plus embeddings |
| 2 | **Coherence field** | Partition tree plus cut pressure signals over time |
| 3 | **Multi-scale memory** | Delta-encoded interaction history from seconds to micro-windows |
| 4 | **Boundary tracker** | Boundary changes, drift, and anomaly alerts |
| 5 | **Planet candidates** | Ranked list with traceable evidence |
| 6 | **Life candidates** | Ranked list of spectral disequilibrium signatures with traceable evidence |

### Non-Goals

1. Proving quantum gravity
2. Replacing astrophysical pipelines end-to-end
3. Claiming life detection without conventional follow-up observation

## Public Data Sources

All data is progressively downloaded from public archives on first activation.
The RVF artifact ships with download manifests and integrity hashes, not the
raw data itself.

### Planet Finding

| Source | Access | Reference |
|--------|--------|-----------|
| Kepler light curves and pixel files | MAST bulk and portal | [archive.stsci.edu/kepler](https://archive.stsci.edu/missions-and-data/kepler) |
| TESS light curves and full-frame images | MAST portal | [archive.stsci.edu/tess](https://archive.stsci.edu/missions-and-data/tess) |

### Life-Relevant Spectra

| Source | Access | Reference |
|--------|--------|-----------|
| JWST exoplanet spectra | exo.MAST and MAST holdings | [archive.stsci.edu](https://archive.stsci.edu/home) |
| NASA Exoplanet Archive parameters | Cross-linking to spectra and mission products | [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu/) |

### Large-Scale Structure

| Source | Access | Reference |
|--------|--------|-----------|
| SDSS public catalogs (spectra, redshifts) | DR17 | [sdss4.org/dr17](https://www.sdss4.org/dr17/) |

### Progressive Download Strategy

Following the lazy-download pattern established in ADR-008 for GGUF models:

1. **Manifest-first**: RVF ships with `MANIFEST_SEG` containing download URLs,
   SHA-256 hashes, expected sizes, and priority tiers
2. **Tier 0 (boot)**: Minimal curated dataset (~50 MB) for offline demo —
   100 Kepler targets with known confirmed planets, embedded in VEC_SEG
3. **Tier 1 (first run)**: Download 1,000 Kepler targets on first pipeline
   activation. Background download, progress reported via CLI/HTTP
4. **Tier 2 (expansion)**: Full Kepler/TESS catalog download on explicit
   `rvf ingest --expand` command
5. **Tier 3 (spectra)**: JWST and archive spectra downloaded when life
   candidate pipeline is first activated
6. **Seal-on-complete**: After download, data is ingested into VEC_SEG and
   INDEX_SEG, a new witness root is committed, and the RVF is sealed into
   a reproducible snapshot

**State machine**: `[boot]` -> `[tier-0-only]` (offline demo) -> `[tier-1-ready]` (first inference) -> `[tier-2-ready]` (`rvf ingest --expand`) -> `[tier-3-ready]` (life pipeline) -> `[sealed-snapshot]`.

Each tier download:
- Resumes from last byte on interruption (HTTP Range headers)
- Validates SHA-256 after download
- Commits a witness record for the download event
- Can be skipped with `--offline` flag (uses whatever is already present)

## RVF Artifact Layout

Extends the ADR-003 segment model with domain-specific segments.

| # | Segment | Contents |
|---|---------|----------|
| 1 | `MANIFEST_SEG` | Segment table, hashes, policy, budgets, version gates, **download manifests** |
| 2 | `KERNEL_SEG` | Minimal Linux kernel image for portable boot (reuse ADR-008) |
| 3 | `INITRD_SEG` | Minimal userspace: busybox, RuVector binaries, data ingest tools, query server |
| 4 | `EBPF_SEG` | Socket allow-list and syscall reduction. Default: local loopback + explicit download ports only |
| 5 | `VEC_SEG` | Embedding vectors: light-curve windows, spectrum windows, graph node descriptors, partition boundary descriptors |
| 6 | `INDEX_SEG` | HNSW unified attention index for vectors and boundary descriptors |
| 7 | `GRAPH_SEG` | Dynamic interaction graph: nodes, edges, timestamps, authority, provenance |
| 8 | `DELTA_SEG` | Append-only change log of graph updates and field updates |
| 9 | `WITNESS_SEG` | Deterministic witness chain: canonical serialization, signed root hash progression |
| 10 | `POLICY_SEG` | Data provenance requirements, candidate publishing thresholds, deny rules, confidence floors |
| 11 | `DASHBOARD_SEG` | Vite-bundled Three.js visualization app — static assets served by runtime HTTP server |

## Data Model

### Core Entities

| Entity | Key Fields | Description |
|--------|-----------|-------------|
| **Event** | `id`, `t_start`, `t_end`, `domain`, `payload_hash`, `provenance` | Time-windowed observation or derived result. Domain: kepler, tess, jwst, sdss, derived |
| **Observation** | `id`, `instrument`, `target_id`, `data_pointer`, `calibration_version` | Raw instrument measurement with VEC_SEG offset |
| **InteractionEdge** | `src_event_id`, `dst_event_id`, `type`, `weight`, `lag`, `confidence` | Typed relationship: causal, periodicity, shape_similarity, co_occurrence, spatial |
| **Boundary** | `boundary_id`, `partition_*_hash`, `cut_weight`, `cut_witness`, `stability_score` | Graph partition boundary with witness chain reference |
| **Candidate** | `candidate_id`, `category`, `evidence_pointers[]`, `score`, `uncertainty`, `publishable`, `witness_trace` | Planet or life candidate with POLICY_SEG-gated publishability |
| **Provenance** | `source`, `download_witness`, `transform_chain[]`, `timestamp` | Full lineage from download through every transform |

### Domain Adapters

| Adapter | Input | Output |
|---------|-------|--------|
| **Planet Transit** | Flux time series + cadence metadata (Kepler/TESS FITS) | Event nodes, periodicity/shape InteractionEdges, dip Candidates |
| **Spectrum** | Wavelength, flux, error arrays (JWST NIRSpec, etc.) | Band Event nodes, molecule co-occurrence edges, disequilibrium scores |
| **Cosmic Web** (Phase 2+) | Galaxy positions and redshifts (SDSS) | Spatial adjacency graph with filament membership |

## The Four System Constructs

### 1. Compressed Causal Atlas

Partial order of events plus minimal sufficient descriptors to reproduce derived edges.

**Construction**: (1) Window light curves at scales 2h/12h/3d/27d. (2) Extract features: flux derivatives, autocorrelation peaks, wavelet energy, matched filter response. (3) Embed via RuVector SIMD into VEC_SEG. (4) Add causal edges where window A precedes B and improves predictability (prediction gain, POLICY_SEG constrained). (5) Compress: top-k parents per node, retain boundary witnesses, delta-encode into DELTA_SEG.

**API**: `atlas.query(event_id)` returns parents/children with provenance. `atlas.trace(candidate_id)` returns minimal causal chain.

### 2. Coherence Field Index

Field over the atlas graph assigning coherence pressure and cut stability over time. Signals: cut pressure (min-cut values), partition entropy (cluster size distribution), disagreement (cross-detector rate), drift (embedding distribution shift).

**Algorithm**: Maintain partition tree with dynamic min-cut on incremental changes. Each epoch: compute cut witnesses for top boundaries, emit to GRAPH_SEG, append to WITNESS_SEG. Index boundaries by descriptor: cut value, partition sizes, curvature proxy, churn.

**API**: `coherence.get(target_id, epoch)` returns field values. `boundary.nearest(descriptor)` returns similar historical states via INDEX_SEG.

### 3. Multi-Scale Interaction Memory

Tiered retention with strict budget control: **S** (seconds-minutes, high-fidelity deltas), **M** (hours-days, aggregated), **L** (weeks-months, boundary summaries and archetypes). Retention preserves boundary-critical events and candidate evidence; compresses the rest via archetype clustering. DELTA_SEG is append-only; periodic compaction produces a new RVF root with witness proof.

### 4. Boundary Evolution Tracker

Treats boundaries as primary objects evolving over time -- the holographic design principle. Boundaries are stored and indexed as first-class objects; interior state is reconstructable from boundary witnesses and retained archetypes.

**API**: `boundary.timeline(target_id)` returns boundary evolution. `boundary.alerts` fires on cut pressure spikes, boundary identity flips, disagreement threshold breaches, or persistent drift.

## Planet Detection Pipeline

### Stage P0: Ingest

**Input**: Kepler or TESS light curves from MAST (progressively downloaded)

1. Normalize flux
2. Remove obvious systematics (detrending)
3. Segment into windows and store as Event nodes

### Stage P1: Candidate Generation

1. Matched filter bank for transit-like dips
2. Period search on candidate dip times (BLS or similar)
3. Create Candidate node per period hypothesis

### Stage P2: Coherence Gating

Candidate must pass all gates:

| Gate | Requirement |
|------|-------------|
| Multi-scale stability | Stable across multiple window scales |
| Boundary consistency | Consistent boundary signature around transit times |
| Low drift | Drift below threshold across adjacent windows |

**Score components**:

| Component | Description |
|-----------|-------------|
| SNR-like strength | Signal-to-noise of transit dip |
| Shape consistency | Cross-transit shape agreement |
| Period stability | Variance of period estimates |
| Coherence stability | Coherence field stability around candidate |

**Emit**: Candidate with evidence pointers + witness trace listing exact
windows, transforms, and thresholds used.

## Life Candidate Pipeline

Life detection here means pre-screening for non-equilibrium atmospheric
chemistry signatures, not proof.

### Stage L0: Ingest

**Input**: Published or mission spectra tied to targets via MAST and NASA
Exoplanet Archive (progressively downloaded on first pipeline activation)

1. Normalize and denoise within instrument error model
2. Window spectra by wavelength bands
3. Create band Event nodes

### Stage L1: Feature Extraction

1. Identify absorption features and confidence bands
2. Encode presence vectors for key molecule families (H2O, CO2, CH4, O3, NH3, etc.)
3. Build InteractionEdges between features that co-occur in physically
   meaningful patterns

### Stage L2: Disequilibrium Scoring

**Core concept**: Life-like systems maintain chemical ratios that resist
thermodynamic relaxation.

**Implementation as graph scoring**:

1. Build a reaction plausibility graph (prior rule set in POLICY_SEG)
2. Compute inconsistency score between observed co-occurrences and expected
   equilibrium patterns
3. Track stability of that score across epochs and observation sets

**Score components**:

| Component | Description |
|-----------|-------------|
| Persistent multi-molecule imbalance | Proxy for non-equilibrium chemistry |
| Feature repeatability | Agreement across instruments or visits |
| Contamination risk penalty | Instrument artifact and stellar contamination |
| Stellar activity confound penalty | Host star variability coupling |

**Output**: Life candidate list with explicit uncertainty + required follow-up
observations list generated by POLICY_SEG rules.

### Microlensing & Cross-Domain Extensions

> See [ADR-040b: Microlensing & Graph-Cut Extensions](ADR-040b-microlensing-graphcut-extensions.md) for M0-M3 pipeline, cross-domain applications, measured results, and Rust crate structure.

## Runtime and Portability

### Boot Sequence

1. RVF boots minimal Linux from KERNEL_SEG and INITRD_SEG (reuse ADR-008 `KernelBuilder`)
2. Starts `rvf-runtime` daemon exposing local HTTP and CLI
3. On first inference/query, progressively downloads required data tier

### Local Interfaces

**CLI**:
```bash
rvf run artifact.rvf                    # boot the runtime
rvf query planet list                   # ranked planet candidates
rvf query life list                     # ranked life candidates
rvf trace <candidate_id>               # full witness trace for any candidate
rvf ingest --expand                     # download tier-2 full catalog
rvf status                              # download progress, segment sizes, witness count
```

**HTTP**:
```
GET /                                   # Three.js dashboard (served from DASHBOARD_SEG)
GET /assets/*                           # Dashboard static assets

GET /api/atlas/query?event_id=...       # causal parents/children
GET /api/atlas/trace?candidate_id=...   # minimal causal chain
GET /api/coherence?target_id=...&epoch= # field values
GET /api/boundary/timeline?target_id=...
GET /api/boundary/alerts
GET /api/candidates/planet              # ranked planet list
GET /api/candidates/life                # ranked life list
GET /api/candidates/:id/trace           # witness trace
GET /api/status                         # system health + download progress
GET /api/memory/tiers                   # tier S/M/L utilization

WS  /ws/live                            # real-time boundary alerts, pipeline progress, candidate updates
```

### Determinism

1. Fixed seeds for all stochastic operations
2. Canonical serialization of every intermediate artifact
3. Witness chain commits after each epoch
4. Two-machine reproducibility: identical RVF root hash for identical input

### Security Defaults

1. Network off by default
2. If enabled, eBPF allow-list: MAST/archive download ports + local loopback only
3. No remote writes without explicit policy toggle in POLICY_SEG
4. Downloaded data verified against MANIFEST_SEG hashes before ingestion

### Dashboard Architecture

> See [ADR-040a: Planet Detection Dashboard](ADR-040a-planet-detection-dashboard.md) for Views V1-V5, WebSocket streaming, Vite build configuration, and design decision D5.

## Package Structure

```
packages/agentdb-causal-atlas/
  src/
    index.ts                    # createCausalAtlasServer() factory
    CausalAtlasServer.ts        # HTTP + CLI runtime + dashboard serving + WS
    CausalAtlasEngine.ts        # Core atlas, coherence, memory, boundary
    adapters/
      PlanetTransitAdapter.ts   # Kepler/TESS light curve ingestion
      SpectrumAdapter.ts        # JWST/archive spectral ingestion
      CosmicWebAdapter.ts       # SDSS spatial graph (Phase 2)
    pipelines/
      PlanetDetection.ts        # P0-P2 planet detection pipeline
      LifeCandidate.ts          # L0-L2 life candidate pipeline
    constructs/
      CausalAtlas.ts            # Compressed causal partial order
      CoherenceField.ts         # Partition tree + cut pressure
      MultiScaleMemory.ts       # Tiered S/M/L retention
      BoundaryTracker.ts        # Boundary evolution + alerts
    download/
      ProgressiveDownloader.ts  # Tiered lazy download with resume
      DataManifest.ts           # URL + hash + size manifests
    KernelBuilder.ts            # Reuse/extend from ADR-008
  dashboard/                    # See ADR-040a for full dashboard tree
  tests/
    causal-atlas.test.ts
    planet-detection.test.ts
    life-candidate.test.ts
    progressive-download.test.ts
    coherence-field.test.ts
    boundary-tracker.test.ts
```

### Rust Implementation

> See [ADR-040b](ADR-040b-microlensing-graphcut-extensions.md#rust-implementation) for the full Rust crate table. Examples location: `examples/rvf/examples/`

## Implementation Phases

### Phase 1: Core Atlas + Planet Detection + Dashboard Shell (v0.1)

**Scope**: Kepler and TESS only. No spectra. No life scoring.

1. `ProgressiveDownloader` with tier-0 curated dataset (100 Kepler targets)
2. `PlanetTransitAdapter` for FITS light curve ingestion
3. `CausalAtlas` with windowing, feature extraction, SIMD embedding
4. `PlanetDetection` pipeline (P0-P2)
5. `WITNESS_SEG` with SHAKE-256 chain
6. CLI: `rvf run`, `rvf query planet list`, `rvf trace`
7. HTTP: `/api/candidates/planet`, `/api/atlas/trace`
8. Dashboard shell: V1, V3, V5 views (see ADR-040a), WebSocket `/ws/live`

**Acceptance**: 1,000 Kepler targets, top-100 ranked list includes >= 80
confirmed planets, every item replays to same score and witness root on two
machines.

### Phase 2: Coherence Field + Boundary Tracker (v0.2)

1. `CoherenceField` with dynamic min-cut, partition entropy
2. `BoundaryTracker` with timeline and alerts
3. `MultiScaleMemory` with S/M/L tiers and budget control
4. Coherence gating added to planet pipeline
5. HTTP: `/api/coherence`, `/api/boundary/*`, `/api/memory/tiers`
6. Dashboard V2 Coherence Heatmap (see ADR-040a)

### Phase 3: Life Candidate Pipeline (v0.3)

1. `SpectrumAdapter` for JWST/archive spectral data
2. `LifeCandidate` pipeline (L0-L2) with disequilibrium scoring
3. Tier-3 progressive download for spectral data
4. CLI: `rvf query life list`; HTTP: `/api/candidates/life`
5. Dashboard V4 Life Dashboard (see ADR-040a)

**Acceptance**: AUC > 0.8 on published spectra with known atmospheric
detections vs nulls, every score includes confound penalties and provenance.

### Phase 4: Cosmic Web + Full Integration (v0.4)

1. `CosmicWebAdapter` for SDSS spatial graph
2. Cross-domain coherence (planet candidates enriched by large-scale context)
3. Full offline demo with sealed RVF snapshot
4. `rvf ingest --expand` for tier-2 bulk download

## Evaluation Plan

### Planet Detection Acceptance Test

| Metric | Requirement |
|--------|-------------|
| Recall@100 | >= 80 confirmed planets in top 100 |
| False positives@100 | Documented with witness traces |
| Median time per star | Measured and reported |
| Reproducibility | Identical root hash on two machines |

### Life Candidate Acceptance Test

| Metric | Requirement |
|--------|-------------|
| AUC (detected vs null) | > 0.8 |
| Confound penalties | Present on every score |
| Provenance trace | Complete for every score |

### System Acceptance Test

| Test | Requirement |
|------|-------------|
| Boot reproducibility | Identical root hash across two machines |
| Query determinism | Identical results for same dataset snapshot |
| Witness verification | `verifyWitness` passes for all chains |
| Progressive download | Resumes correctly after interruption |

## Failure Modes and Fix Path

| Failure | Fix |
|---------|-----|
| Noise dominates coherence field | Strengthen policy priors, add confound penalties, enforce multi-epoch stability |
| Over-compression kills rare signals | Boundary-critical retention rules + candidate evidence pinning |
| Spurious life signals from stellar activity | Model stellar variability as its own interaction graph, penalize coupling |
| Compute blow-up | Strict budgets in POLICY_SEG, tiered memory, boundary-first indexing |
| Download interruption | HTTP Range resume, partial-ingest checkpoint, witness for partial state |

## Design Decisions

### D1: Kepler/TESS only in v1, spectra in v3

Phase 1 delivers a concrete, testable planet-detection system. Life scoring
requires additional instrument-specific adapters and more nuanced policy
rules. Separating them de-risks the schedule.

### D2: Progressive download with embedded demo subset

The RVF artifact ships with a curated ~50 MB tier-0 dataset for fully offline
demonstration. Full catalog data is downloaded lazily, following the pattern
proven in ADR-008 for GGUF model files. This keeps the initial artifact small
(< 100 MB without kernel) while supporting the full 1,000+ target benchmark.

### D3: Boundary-first storage (holographic principle)

Boundaries are stored as first-class indexed objects. Interior state is
reconstructed on-demand from boundary witnesses and retained archetypes.
This reduces storage by 10-50x for large graphs while preserving
queryability and reproducibility.

### D4: Witness chain for every derived claim

Every candidate, every coherence measurement, and every boundary change is
committed to the SHAKE-256 witness chain. This enables two-machine
reproducibility verification and provides a complete audit trail from raw
data to final score.

## Recent Enhancements (2026-03)

| Enhancement | Detail |
|-------------|--------|
| QAOA graph-cut solver | Quantum alternative to Edmonds-Karp via ruQu QAOA (`qaoa_graphcut.rs`) |
| Kepler's third law | Dashboard semi-major axis uses `a = P^(2/3)` instead of linear approx |
| Deterministic orbit params | Eccentricity/inclination derived from candidate name hash, not `Math.random()` |
| Logarithmic BLS period grid | 400 log-spaced trial periods for uniform sensitivity across period range |
| Multi-duration transit search | 5 trial durations (0.01-0.035) instead of single 0.02 duty cycle |
| Iterative cut refinement | 3-iteration mincut with lambda boost/decay (exomoon F1: 0.261 to 0.308) |
| Real OGLE/MOA manifest | 13 real microlensing events with published parameters |

## References

1. [MAST — Kepler](https://archive.stsci.edu/missions-and-data/kepler) | 2. [MAST — TESS](https://archive.stsci.edu/missions-and-data/tess) | 3. [MAST Home](https://archive.stsci.edu/home)
4. [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) | 5. [SDSS DR17](https://www.sdss4.org/dr17/)
6. ADR-003 | 7. ADR-006 | 8. ADR-007 | 9. ADR-008
