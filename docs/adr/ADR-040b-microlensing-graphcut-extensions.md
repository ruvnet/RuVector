# ADR-040b: Microlensing Detection & Cross-Domain Graph-Cut Extensions

**Status:** Proposed
**Date:** 2026-03-15
**Parent:** ADR-040
**Related:** ADR-003 (RVF Format), ADR-006 (Unified Self-Learning RVF)

## Context

This sub-ADR documents the microlensing detection pipeline (M0-M3) and
cross-domain graph-cut applications that extend ADR-040's core architecture.
These extensions demonstrate how the MRF/mincut + RuVector pattern generalizes
beyond transit-based planet detection to gravitational microlensing and to
non-astronomy domains including medical imaging, genomics, financial fraud
detection, supply chain monitoring, cybersecurity, and climate analysis.

Extracted from ADR-040 to keep individual files under 500 lines per project
guidelines.

## Microlensing Detection Pipeline (M0-M3)

Extends the transit pipeline to gravitational microlensing events for rogue
planet and exomoon candidate detection.

### Additional Data Sources

| Source | Access | Reference |
|--------|--------|-----------|
| OGLE-IV microlensing events | Public FTP | [ogle.astrouw.edu.pl](https://ogle.astrouw.edu.pl/ogle4/ews/ews.html) |
| MOA-II microlensing alerts | Public archive | [www.massey.ac.nz/~iabond/moa](https://www.massey.ac.nz/~iabond/moa/) |
| Gaia astrometric catalog | ESA public | [gea.esac.esa.int](https://gea.esac.esa.int/archive/) |
| Roman Space Telescope | Upcoming | High-cadence microlensing survey |

### Stage M0: Ingest

Microlensing light curve ingestion from OGLE/MOA archives.
- Normalize photometry across surveys (MOA-II: ~15 min cadence, 0.008sigma floor; OGLE-IV: ~40 min cadence, 0.005sigma floor)
- Segment into event windows around magnification peaks
- Store as RVF Event nodes with survey-specific metadata

### Stage M1: Single-Lens Detection

Standard Paczynski curve fitting for point-source point-lens (PSPL) events.
- Two-phase fit: coarse grid search + fine refinement
- Linear regression for source flux (F_s) and blending flux (F_b)
- Parameter estimation: Einstein crossing time (t_E), impact parameter (u_0), peak time (t_0)
- Create Candidate nodes for events exceeding SNR threshold

### Stage M2: Anomaly Detection via MRF/Mincut

Residual analysis after best-fit PSPL subtraction using Markov Random Field
optimization with graph cut inference.

**Three-statistic lambda computation:**
1. Excess chi2: window fit quality relative to global reduced chi2
2. Runs test: temporal coherence of residual sign changes
3. Gaussian bump fit: localized perturbation chi2 improvement

**Differential normalization:** Compare each window's signal to tau-space
neighbors, producing z-scores that are ~0 for uniform fit quality and positive
only for localized anomalies.

**Graph construction:**
- Temporal chain: consecutive time windows (alpha weight)
- RuVector kNN edges: embedding similarity from residual features (beta weight)
- Edmonds-Karp BFS max-flow solver for s-t mincut

### Stage M3: Coherence Gating

Dynamic mincut separates competing explanations (noise vs planet vs moon vs
binary lens) using support region analysis.

| Component | Role |
|-----------|------|
| RuVector embeddings | Cross-survey memory; retrieve similar anomalies |
| HNSW index | Fast similarity search across microlensing events |
| Dynamic mincut | Separate competing explanations |
| Witness chain | Full provenance from raw photometry to candidate |

**Important constraint:** Dynamic mincut provides coherent support extraction
but cannot replace lens modeling. The physics solver (PSPL/binary) provides
local evidence; mincut provides spatial coherence.

## Cross-Domain Graph-Cut Applications

The MRF/mincut + RuVector architecture generalizes beyond astronomy to any
domain requiring coherent anomaly detection in structured data.

### Implemented Verticals

| Vertical | Example | Graph Topology | Anomaly Types |
|----------|---------|----------------|---------------|
| **Medical Imaging** | `medical_graphcut.rs` | 4-connected spatial grid, gradient-weighted edges | Lesion segmentation (T1-MRI, T2-MRI, CT) |
| **Genomics** | `genomic_graphcut.rs` | Linear chain + kNN similarity | CNV gains/losses, LOH, cancer drivers (TP53, BRCA1, EGFR, MYC) |
| **Financial Fraud** | `financial_fraud_graphcut.rs` | Temporal chain + merchant edges + kNN | Card-not-present, account takeover, card clone, synthetic, refund |
| **Supply Chain** | `supply_chain_graphcut.rs` | Tier chain + geographic + kNN | Quality defect, shortage, price anomaly, delay, counterfeit, demand shock |
| **Cybersecurity** | `cyber_threat_graphcut.rs` | Source/destination chain + kNN | Port scan, brute force, exfiltration, C2 beacon, DDoS, lateral movement |
| **Climate** | `climate_graphcut.rs` | Spatial adjacency + gradient + kNN | Heat wave, pollution spike, drought, ocean warming, cold snap, sensor fault |

### Common Architecture

All verticals share:
1. **Domain-specific data generator** with realistic parameters from published datasets
2. **Feature extraction** producing 32-dim embeddings for RuVector storage
3. **Graph construction** combining domain topology (spatial/temporal/chain) with kNN similarity edges
4. **Edmonds-Karp BFS** s-t mincut solver (identical implementation)
5. **RVF integration**: witness chains, filtered metadata queries, lineage derivation
6. **Evaluation**: comparison against threshold baseline showing graph-cut improvement

### Additional Real Data Sources

| Domain | Source | Access |
|--------|--------|--------|
| Genomics | TCGA (The Cancer Genome Atlas) | [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov/) |
| Genomics | ClinVar variants | [ncbi.nlm.nih.gov/clinvar](https://www.ncbi.nlm.nih.gov/clinvar/) |
| Genomics | COSMIC (somatic mutations) | [cancer.sanger.ac.uk/cosmic](https://cancer.sanger.ac.uk/cosmic) |
| Medical | LIDC-IDRI (lung CT) | [cancerimagingarchive.net](https://www.cancerimagingarchive.net/) |
| Medical | BraTS (brain tumors) | [synapse.org](https://www.synapse.org/) |
| Cybersecurity | CICIDS2017 | [unb.ca/cic](https://www.unb.ca/cic/datasets/ids-2017.html) |
| Climate | NOAA GHCN | [ncei.noaa.gov](https://www.ncei.noaa.gov/) |
| Climate | EPA AQI | [epa.gov/aqs](https://www.epa.gov/aqs) |

## Measured Results

Results from running examples with `cargo run --example <name> --release`.

### Astronomy

| Pipeline | Metric | Value | Notes |
|----------|--------|-------|-------|
| `planet_detection` | Recall@10 | 8/10 confirmed | Synthetic Kepler-like targets |
| `exomoon_graphcut` | Precision | 0.25 | Chang-Refsdal perturbative limit |
| `exomoon_graphcut` | Recall | 0.25 | Limited by ~2-3sigma per-window SNR |
| `exomoon_graphcut` | F1 | 0.25 | 30 events, 12 with moons |
| `real_microlensing` | Planets rediscovered | 2/4 | OGLE-2005-BLG-390, OGLE-2016-BLG-1195 |
| `microlensing_detection` | Events processed | 20 | Synthetic PSPL + anomalies |

### Medical Imaging

| Modality | Graph Cut Dice | Threshold Dice | Improvement |
|----------|---------------|----------------|-------------|
| T1-MRI | 0.44-0.59 | 0.32-0.46 | +27-39% |
| T2-MRI | 0.50-0.62 | 0.38-0.48 | +29-32% |
| CT | 0.48-0.58 | 0.35-0.44 | +32-37% |

### Genomics

| Platform | Sensitivity | Specificity | Drivers Found |
|----------|-------------|-------------|---------------|
| WGS (30x) | 0.91-0.95 | 0.66-0.97 | 4/4 (TP53, BRCA1, EGFR, MYC) |
| WES (100x) | 0.88-0.93 | 0.70-0.95 | 4/4 |
| Panel (500x) | 0.93-0.97 | 0.72-0.98 | 4/4 |

## Rust Implementation

The pipeline examples are implemented in Rust using the RVF runtime crates:

| Crate | Role |
|-------|------|
| `rvf-types` | Core types, segment definitions, derivation types |
| `rvf-runtime` | RvfStore, HNSW indexing, metadata filters, queries |
| `rvf-crypto` | SHAKE-256 witness chains, verification |
| `rvf-wire` | Wire format serialization |
| `rvf-manifest` | Segment manifests and policies |
| `rvf-index` | Progressive HNSW index construction |
| `rvf-quant` | Vector quantization (PQ, SQ) |
| `rvf-kernel` | Kernel/initrd segment embedding |
| `rvf-launch` | Boot sequence and runtime launch |
| `rvf-ebpf` | eBPF socket/syscall policies |
| `rvf-server` | HTTP/WS server for dashboard and API |

Examples location: `examples/rvf/examples/`

## References

1. ADR-040: Causal Atlas RVF Runtime — Planet Detection & Life Candidate Scoring
2. [OGLE Microlensing Events](https://ogle.astrouw.edu.pl/ogle4/ews/ews.html)
3. [MOA Microlensing Alerts](https://www.massey.ac.nz/~iabond/moa/)
4. [Gaia Archive](https://gea.esac.esa.int/archive/)
5. [TCGA Data Portal](https://portal.gdc.cancer.gov/)
6. [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
