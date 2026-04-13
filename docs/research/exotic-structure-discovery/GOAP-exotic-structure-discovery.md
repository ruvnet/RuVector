# GOAP: Exotic Structure Discovery via Boundary-First Analysis

## Research Program - Goal-Oriented Action Plan

**Status:** Proposed
**Date:** 2026-04-12
**Branch:** research/exotic-structure-discovery
**Depends on:** ruvector-mincut, ruvector-sparsifier, ruvector-coherence, ruvector-consciousness, ruvector-solver, ruvector-delta-core, ruvector-temporal-tensor, ruqu-exotic, ruvector-graph, ruvector-domain-expansion

---

## 0. Thesis

Traditional astrophysical and scientific discovery is amplitude-first: you detect something because it is bright, loud, massive, or energetic. Thresholds on intensity create a fundamental selection bias -- quiet, structured, boundary-defined phenomena are invisible to this approach.

RuVector inverts the detection paradigm. Instead of asking "where is the signal strongest?", we ask:

- **Where do structural boundaries form?** (mincut)
- **Where does spectral coherence change?** (sparsifier, coherence)
- **Where does integrated information peak or collapse?** (consciousness/Phi)
- **Where do changes themselves change?** (delta behavior)
- **Where does information persist across time without amplitude?** (temporal hypergraphs)

This is boundary-first, structure-first science. The hypothesis is that this approach will discover classes of structure that amplitude-first detection cannot see.

---

## 1. World State Model

### 1.1 Current State

| State Variable | Value | Description |
|---------------|-------|-------------|
| `pipeline_exists` | false | No unified exotic structure discovery pipeline |
| `data_ingested` | false | No astrophysical data loaded into RuVector graphs |
| `graph_construction_defined` | false | No mapping from raw data to graph representation |
| `mincut_on_real_data` | false | MinCut never run on astrophysical signal graphs |
| `spectral_coherence_on_real_data` | false | Coherence metrics never applied to scientific data |
| `phi_on_signal_graphs` | false | IIT Phi never computed on astrophysical structures |
| `exotic_scoring_system` | false | No scoring taxonomy for structural novelty |
| `cross_modal_pipeline` | false | No multi-wavelength/multi-messenger fusion |
| `temporal_tracking` | false | No longitudinal monitoring of discovered structures |
| `publication_ready` | false | No results suitable for scientific publication |

### 1.2 Goal State

| State Variable | Value | Description |
|---------------|-------|-------------|
| `pipeline_exists` | true | End-to-end pipeline: raw data -> graph -> analysis -> scored anomalies |
| `data_ingested` | true | At least 5 datasets loaded with graph representations |
| `graph_construction_defined` | true | Documented mapping for radio, optical, X-ray, time-series |
| `mincut_on_real_data` | true | LocalKCut producing boundary partitions on real signals |
| `spectral_coherence_on_real_data` | true | Fiedler values, spectral gaps measured on signal graphs |
| `phi_on_signal_graphs` | true | Integrated information quantified for signal structures |
| `exotic_scoring_system` | true | Composite score: persistence x novelty x coherence x non-natural |
| `cross_modal_pipeline` | true | At least 2-band fusion (radio+optical or radio+X-ray) |
| `temporal_tracking` | true | Delta-based change tracking on discovered structures |
| `publication_ready` | true | Reproducible results with statistical validation |

---

## 2. Action Definitions

### Action Graph

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │                    GOAL: Discovery Pipeline                   │
                    └───────────────┬───────────────────────────┬──────────────────┘
                                    │                           │
                    ┌───────────────▼───────────┐   ┌──────────▼──────────────────┐
                    │  A7: Exotic Scoring System │   │  A8: Cross-Modal Fusion     │
                    │  cost=4, needs A4,A5,A6    │   │  cost=5, needs A4,A5        │
                    └───────────────┬───────────┘   └──────────┬──────────────────┘
                                    │                           │
          ┌─────────────────────────┼───────────────────────────┤
          │                         │                           │
┌─────────▼──────────┐  ┌──────────▼──────────┐  ┌────────────▼──────────────┐
│ A4: MinCut Analysis │  │ A5: Spectral Coh.   │  │ A6: Phi Computation       │
│ cost=3, needs A2,A3 │  │ cost=3, needs A2,A3 │  │ cost=4, needs A2,A3       │
└─────────┬──────────┘  └──────────┬──────────┘  └────────────┬──────────────┘
          │                         │                           │
          └─────────────────────────┼───────────────────────────┘
                                    │
                    ┌───────────────▼───────────┐
                    │ A3: Graph Construction     │
                    │ cost=5, needs A1,A2        │
                    └───────────────┬───────────┘
                                    │
                    ┌───────────────┼───────────────────┐
                    │                                   │
          ┌─────────▼──────────┐          ┌────────────▼──────────┐
          │ A1: Data Ingestion │          │ A2: Graph Schema Def. │
          │ cost=3, no prereqs │          │ cost=4, no prereqs    │
          └────────────────────┘          └───────────────────────┘


          ┌──────────────────────────────────────────────────────────┐
          │  A9: Temporal Tracking (Delta Behavior)                  │
          │  cost=4, needs A7                                        │
          │  effects: temporal_tracking=true                          │
          └──────────────────────────────┬───────────────────────────┘
                                         │
          ┌──────────────────────────────▼───────────────────────────┐
          │  A10: Publication & Validation                           │
          │  cost=5, needs A7,A8,A9                                  │
          │  effects: publication_ready=true                          │
          └──────────────────────────────────────────────────────────┘
```

### Action Catalog

| ID | Action | Cost | Preconditions | Effects | RuVector Crate |
|----|--------|------|---------------|---------|----------------|
| A1 | Data Ingestion | 3 | none | data_ingested=true | rvf, ruvector-graph |
| A2 | Graph Schema Definition | 4 | none | graph_construction_defined=true | ruvector-graph |
| A3 | Graph Construction Pipeline | 5 | A1, A2 | pipeline_exists=partial | ruvector-graph, ruvector-sparsifier |
| A4 | MinCut Boundary Analysis | 3 | A3 | mincut_on_real_data=true | ruvector-mincut |
| A5 | Spectral Coherence Mapping | 3 | A3 | spectral_coherence_on_real_data=true | ruvector-coherence, ruvector-sparsifier |
| A6 | Phi/Emergence Computation | 4 | A3 | phi_on_signal_graphs=true | ruvector-consciousness |
| A7 | Exotic Scoring System | 4 | A4, A5, A6 | exotic_scoring_system=true | ruvector-domain-expansion |
| A8 | Cross-Modal Fusion Pipeline | 5 | A4, A5 | cross_modal_pipeline=true | ruvector-graph (hyperedges) |
| A9 | Temporal Delta Tracking | 4 | A7 | temporal_tracking=true | ruvector-delta-core, ruvector-temporal-tensor |
| A10 | Publication & Validation | 5 | A7, A8, A9 | publication_ready=true | all |

---

## 3. Freely Available Datasets

### 3.1 Radio Astronomy

**CHIME/FRB Open Data (Fast Radio Bursts)**
- URL: https://www.chime-frb.ca/catalog
- Content: First CHIME/FRB catalog with 536 FRBs (2021 release), growing. Includes burst properties (DM, width, fluence, spectro-temporal structure), repeater classifications.
- Size: ~10 MB catalog (individual waterfall data via CANFAR)
- Graph mapping: Each burst is a node. Edges from DM similarity, temporal proximity, spectral shape correlation, sky position. Repeater bursts form temporal chains.
- Why boundary-first: FRBs show sub-burst structure (drift rates, spectral islands). MinCut on the time-frequency waterfall reveals structural partitions invisible to single-threshold detection. Spectral coherence across sub-bursts reveals whether they are one phenomenon or many.

**LOFAR Two-metre Sky Survey (LoTSS)**
- URL: https://lofar-surveys.org/releases.html
- Content: 4.4 million radio sources at 120-168 MHz. DR2 covers 5720 sq deg. Images, catalogs, spectral indices.
- Size: Catalogs ~2 GB, images ~50 TB (use catalogs)
- Graph mapping: Sources as nodes. Edges from angular proximity, spectral index similarity, morphological correlation. Radio relics and halos in clusters become dense subgraphs.
- Why boundary-first: Diffuse radio emission (halos, relics, phoenixes) has no well-defined center. They ARE boundaries -- shock fronts, turbulent mixing zones. MinCut finds where the diffuse emission structurally separates from point sources.

**VLASS (VLA Sky Survey)**
- URL: https://science.nrao.edu/vlass
- Content: 2-4 GHz radio survey, 3 epochs, full northern sky. ~3.4 million components in Quick Look catalogs.
- Size: Catalogs ~500 MB per epoch
- Graph mapping: Multi-epoch enables temporal edges. Same source across epochs connected by delta vectors. New sources, disappearing sources become graph events.
- Why boundary-first: Transient radio sources (flares, TDEs, new jets) appear as topological discontinuities -- graph insertions that change local cut structure.

### 3.2 Optical/IR

**Sloan Digital Sky Survey (SDSS DR18)**
- URL: https://www.sdss.org/dr18/
- Content: Photometry for 500M+ objects, spectra for 5M+ objects, multi-band (u,g,r,i,z). Includes galaxy clusters, QSOs, stellar streams.
- Size: CasJobs SQL queries for targeted extraction, catalogs ~100 GB total
- Graph mapping: Galaxies as nodes within cluster fields. Edges from projected distance, velocity difference (redshift), color similarity, morphological type. Spectroscopic data enables 3D graph construction.
- Why boundary-first: Galaxy cluster boundaries (the infall region, splash-back radius) are physically meaningful and poorly characterized by radial profiles. MinCut on the velocity-position graph reveals the true dynamical boundary.

**Gaia DR3**
- URL: https://gea.esac.esa.int/archive/
- Content: 1.8 billion sources with positions, parallaxes, proper motions, radial velocities for 33M+. BP/RP spectra for 220M.
- Size: Full catalog ~1 TB (targeted queries via TAP)
- Graph mapping: Stars as nodes in 6D phase space (position + velocity). Edges from phase-space proximity weighted by metallicity similarity. Stellar streams become filamentary subgraphs.
- Why boundary-first: Dissolving stellar streams have no "center" -- they are kinematic coherence structures. Spectral coherence (Fiedler value) along the stream reveals where tidal disruption has progressed furthest. MinCut finds where streams bifurcate or where interlopers break the kinematic thread.

**ZTF (Zwicky Transient Facility)**
- URL: https://www.ztf.caltech.edu/ztf-public-releases.html
- Content: Time-domain survey, ~1 billion lightcurves in g, r, i bands. Alert stream via ANTARES/ALeRCE brokers.
- Size: Lightcurve database ~10 TB (use targeted queries via IRSA)
- Graph mapping: Lightcurves as temporal graphs. Each measurement is a node. Edges from temporal adjacency weighted by flux change. Phase-folded graphs for periodic sources.
- Why boundary-first: Morphological classification of lightcurve shapes. MinCut on the phase-folded temporal graph reveals mode changes, state transitions, eclipse ingress/egress boundaries. Delta behavior tracks where the lightcurve changes *how* it changes.

### 3.3 X-ray and Gamma-ray

**Fermi-LAT 4FGL-DR4 (Gamma-Ray Sources)**
- URL: https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/
- Content: 7195 gamma-ray sources. Light curves, spectral parameters, variability indices, association probabilities.
- Size: ~100 MB catalog, event data ~500 GB (use catalog + targeted event extraction)
- Graph mapping: Sources as nodes. Edges from angular proximity, spectral similarity (photon index, curvature), variability correlation. Unassociated sources form a distinct subgraph.
- Why boundary-first: 30% of Fermi sources are unassociated. The graph boundary between associated and unassociated sources reveals what makes a source "classifiable." MinCut on the spectral-variability graph partitions source types that threshold-based classifiers merge.

**eROSITA All-Sky Survey (eRASS)**
- URL: https://erosita.mpe.mpg.de/dr1/ (DR1 released 2024)
- Content: 900,000+ X-ray sources, first all-sky X-ray survey in 20 years. Soft X-ray sensitive.
- Size: Catalog ~200 MB
- Graph mapping: X-ray sources as nodes. Edges from position, hardness ratio, extent parameter, flux variability.
- Why boundary-first: Extended X-ray emission (clusters, supernova remnants) has structural boundaries that define the physics (shock fronts, contact discontinuities). Spectral coherence within extended sources reveals whether the emission is one integrated system or multiple overlapping sources.

**Chandra Source Catalog (CSC 2.1)**
- URL: https://cxc.cfa.harvard.edu/csc/
- Content: 407,806 unique X-ray sources from 15,533 Chandra observations. Sub-arcsecond positions, spectral properties, variability.
- Size: Catalog ~500 MB
- Graph mapping: Intra-observation source graphs. Multi-observation temporal edges for repeated fields.

### 3.4 Multi-Messenger and Time-Domain

**ANTARES/Fink Alert Brokers (Multi-Survey Alerts)**
- URL: https://fink-portal.org/ and https://antares.noirlab.edu/
- Content: Real-time classification of transient alerts from ZTF, soon LSST. Cross-matched with multi-wavelength catalogs.
- Size: Streaming (millions of alerts per night for LSST)
- Graph mapping: Alert stream as a temporal hypergraph. Each alert links a sky position, time, flux change, and classification probability. Hyperedges connect alerts from the same physical source across surveys.
- Why boundary-first: Alert brokers use feature-based classifiers. Boundary-first analysis on the alert graph reveals structural classes that feature-based systems miss: correlated alert patterns, spatial clustering of novel transients, temporal coherence in non-periodic sources.

**LIGO/Virgo/KAGRA Open Science Center (Gravitational Waves)**
- URL: https://gwosc.org/
- Content: Strain data from all observing runs. Event catalogs (GWTC-3: 90 events).
- Size: ~100 TB raw strain (use catalogs + targeted strain around events)
- Graph mapping: Time-frequency spectrograms as pixel graphs. Strain time series as temporal graphs with frequency-domain edges.
- Why boundary-first: Gravitational wave signals are embedded in non-stationary noise. MinCut on the spectrogram graph separates coherent signal structure from noise artifacts. The signal IS the boundary between "astrophysical" and "terrestrial."

**IceCube Neutrino Observatory**
- URL: https://icecube.wisc.edu/data-releases/
- Content: High-energy neutrino event catalogs, 10-year point source data, real-time alerts.
- Size: Catalogs ~50 MB, event data ~10 GB
- Graph mapping: Events as nodes in energy-direction space. Edges from directional proximity weighted by energy similarity.
- Why boundary-first: Neutrino point source searches use stacking/binning. Graph boundary analysis reveals extended or correlated emission structures -- filamentary neutrino emission tracing large-scale structure.

### 3.5 Cosmological Simulations (Ground Truth)

**IllustrisTNG (Cosmological Hydrodynamic Simulation)**
- URL: https://www.tng-project.org/data/
- Content: Full cosmological simulation with gas, stars, dark matter, black holes. TNG50/100/300 at multiple redshifts.
- Size: ~1 PB total (API access for targeted extraction)
- Graph mapping: Particles as nodes, interaction forces as edges. Halo substructure as subgraphs. Filamentary structure as graph topology.
- Why ground truth: We KNOW the true structure. Can validate that RuVector boundary detection recovers known simulation features (filaments, voids, halos, subhalos) before applying to real data.

---

## 4. Graph Construction: What is the Graph?

This is the critical intellectual step. The choice of graph representation determines what mincut, coherence, and Phi can discover.

### 4.1 Signal Graph (Time-Frequency Domain)

**For: FRBs, pulsars, gravitational waves, lightcurves**

```
Input: 2D spectrogram or 1D time series
Nodes: Pixels (time-frequency bins) or samples above noise floor
Edges: Adjacent pixels weighted by spectral similarity
       E(i,j) = exp(-||flux_i - flux_j||^2 / sigma^2) if adjacent

What MinCut reveals:
  - Sub-burst structure (drift lanes, spectral islands)
  - Mode transitions in pulsars
  - Signal-noise boundary in GW spectrograms

What Fiedler value reveals:
  - Connectivity of the signal structure
  - Low Fiedler = fragmented (multiple components)
  - High Fiedler = tightly integrated signal

What Phi reveals:
  - Whether the signal generates more integrated information
    than its sub-components
  - Phi > 0 means the spectral structure is truly integrated,
    not just a sum of independent features
```

### 4.2 Source Graph (Catalog Domain)

**For: SDSS galaxies, LoTSS radio sources, Fermi gamma-ray sources, eROSITA**

```
Input: Source catalog with positions and properties
Nodes: Sources (galaxies, radio sources, X-ray sources)
Edges: k-NN in property space, weighted by similarity
       E(i,j) = kernel(property_i, property_j) if j in kNN(i)
       Properties: position, flux, color, morphology, redshift, variability

What MinCut reveals:
  - Natural classification boundaries (not imposed by humans)
  - Where the source population structurally partitions
  - "Edge" sources that sit on classification boundaries

What Spectral Coherence reveals:
  - How well-separated source classes are
  - Whether unassociated sources are a coherent class or noise
  - Degree regularity indicates uniform vs. biased sampling

What PageRank reveals:
  - Most "central" sources in property space
  - Sources that connect disparate populations (bridge objects)
```

### 4.3 Spatial Graph (Sky Plane Domain)

**For: Large-scale structure, galaxy clusters, diffuse radio emission**

```
Input: Source positions on sky (2D) or with redshifts (3D)
Nodes: Sources or grid cells
Edges: Delaunay triangulation or k-NN in spatial coordinates
       Weight: inverse distance * flux product

What MinCut reveals:
  - Physical boundaries of structures (cluster edges, void walls)
  - Where large-scale structure "breaks"
  - Filament identification via sequential cuts

What Effective Resistance reveals:
  - How "connected" a structure is across its extent
  - High resistance paths = weak links in structure
  - Identifies where clusters are merging (bridge regions)
```

### 4.4 Temporal Graph (Time Domain)

**For: ZTF lightcurves, VLASS multi-epoch, repeating FRBs, variable sources**

```
Input: Time-ordered measurements
Nodes: Observations (time, flux, properties)
Edges: Temporal adjacency weighted by delta (change magnitude)
       E(t_i, t_{i+1}) = |flux_{i+1} - flux_i| / sigma

What MinCut reveals:
  - State transitions in lightcurves
  - Phase boundaries in periodic sources
  - Where behavior *changes* (the delta of deltas)

What Delta Behavior reveals:
  - D-space representation of the temporal evolution
  - Causal D-ordering between events
  - Compressibility of the temporal structure

What Temporal Tensor reveals:
  - Tier classification: hot (active) vs cold (quiescent) phases
  - Access-pattern-driven quantization for long-term storage
```

### 4.5 Cross-Modal Hypergraph (Multi-Wavelength Domain)

**For: Multi-messenger, radio+optical+X-ray coincidences**

```
Input: Cross-matched catalogs from multiple surveys
Nodes: Sources from all wavelengths
Edges: Pairwise similarity within each band
Hyperedges: Physical associations across bands
            H = {radio_source, optical_counterpart, X-ray_emission}
            with temporal validity intervals

What MinCut on hypergraph reveals:
  - Which multi-wavelength associations are structurally robust
  - Where cross-band correlations break down
  - Novel multi-messenger objects that don't fit existing categories

What Spectral Coherence across bands reveals:
  - Cross-band structural consistency
  - Whether radio and optical structure share the same graph topology
  - Frequency-dependent structure changes
```

---

## 5. Crate-to-Discovery-Tier Mapping

### Tier 1: Near-Edge Science

| Discovery Target | Primary Crate | Secondary Crates | Measurement |
|-----------------|--------------|-------------------|-------------|
| Sub-threshold FRB structure | ruvector-mincut | ruvector-coherence | LocalKCut partitions on waterfall spectrograms |
| Pulsar mode transitions | ruvector-mincut | ruvector-delta-core | MinCut + delta behavior on folded profiles |
| Galaxy cluster dynamical boundaries | ruvector-sparsifier | ruvector-coherence | Spectral sparsification preserving Fiedler value |
| Multi-layer diffuse radio emission | ruvector-mincut | ruvector-sparsifier | Recursive mincut revealing hierarchical structure |
| ZTF lightcurve state transitions | ruvector-delta-core | ruvector-mincut | D-space decomposition of flux sequences |

### Tier 2: Mid-Tier Discovery

| Discovery Target | Primary Crate | Secondary Crates | Measurement |
|-----------------|--------------|-------------------|-------------|
| Coherence fields (locally consistent, globally inconsistent) | ruvector-coherence | ruvector-consciousness | Spectral gap variation across spatial graph |
| Boundary-first objects (no center, only edges) | ruvector-mincut | ruvector-sparsifier | Objects detected by cut structure, not flux peak |
| Temporal attractors (behavioral recurrence) | ruvector-delta-core | ruvector-temporal-tensor | D-space periodicity without amplitude periodicity |
| Unassociated Fermi source classification | ruvector-solver | ruvector-mincut | PPR from unassociated sources to known classes |
| LoTSS diffuse emission without host | ruvector-mincut | ruvector-graph | MinCut isolating emission with no optical counterpart |

### Tier 3: Exotic Discovery

| Discovery Target | Primary Crate | Secondary Crates | Measurement |
|-----------------|--------------|-------------------|-------------|
| Non-random quiet zones | ruvector-consciousness | ruvector-coherence | Phi > 0 in regions with sub-threshold amplitude |
| Cross-spectrum coherence (radio+optical+X-ray) | ruvector-graph (hyperedges) | ruvector-coherence | Hyperedge spectral coherence across wavelengths |
| Topological anomalies (graph discontinuities) | ruvector-mincut | ruvector-sparsifier | Sudden changes in mincut value across space/time |
| Information-theoretic boundaries in LSS | ruvector-consciousness | ruvector-mincut | Phi gradients across large-scale structure |
| Signal compression anomalies | ruvector-temporal-tensor | ruvector-consciousness | Regions where signal compresses better than noise |

### Tier 4: Far-Edge Discovery

| Discovery Target | Primary Crate | Secondary Crates | Measurement |
|-----------------|--------------|-------------------|-------------|
| Engineered-like coherence | ruvector-consciousness | ruvector-coherence, ruqu-exotic | Phi + compression + spectral coherence composite |
| Response-like behavior | ruvector-delta-core | ruqu-exotic (reversible_memory) | Temporal correlation suggesting stimulus-response |
| Persistent boundary intelligence | ruvector-mincut | ruvector-consciousness | Boundaries that maintain structure against noise |
| Non-natural information patterns | ruvector-temporal-tensor | ruvector-consciousness | Kolmogorov complexity anomalies in signal structure |
| Cross-domain transfer anomalies | ruvector-domain-expansion | all | Patterns that transfer across unrelated datasets |

---

## 6. Exotic Scoring System

### 6.1 Composite Score: E-Score

```
E-Score(x) = P(x) * S(x) * C(x) * N(x) * [1 + bonus(x)]

Where:
  P(x) = Persistence Score       [0, 1]
  S(x) = Structural Novelty      [0, 1]
  C(x) = Cross-Modal Coherence   [0, 1]
  N(x) = Non-Natural Fit         [0, 1]
  bonus(x) = additional terms for exceptional properties
```

### 6.2 Component Definitions

**P(x): Persistence Score**

How long and how consistently does the structure persist across independent observations?

```
P(x) = (1/T) * sum_{t=1}^{T} I[structure detected at epoch t]
       * (1 - cv(metric_t))

Where:
  T = number of independent observation epochs
  I[.] = indicator function
  cv(.) = coefficient of variation of the detection metric

Crate mapping:
  - ruvector-delta-core: Track structure across epochs
  - ruvector-temporal-tensor: Compress temporal history, measure tier stability
  - Persistence of 1.0 = detected at every epoch with consistent metrics
  - Persistence of 0.0 = single-epoch detection
```

**S(x): Structural Novelty**

How different is this structure from known classes in graph-topology space?

```
S(x) = 1 - max_{c in known_classes} sim(G_x, G_c)

Where:
  G_x = graph representation of structure x
  G_c = template graph for known class c
  sim(.) = graph similarity via spectral distance:
           sim(G1, G2) = exp(-||lambda(G1) - lambda(G2)||_2)
           where lambda(G) = sorted Laplacian eigenvalues

Crate mapping:
  - ruvector-sparsifier: Compute spectral properties
  - ruvector-coherence: Fiedler value, spectral gap
  - ruvector-mincut: Cut structure comparison
  - ruvector-solver: PPR distance to known class templates

Calibration:
  - S < 0.3: Known structure type (pulsar, AGN, etc.)
  - 0.3 < S < 0.7: Unusual variant of known type
  - S > 0.7: Structurally novel -- no close match in template library
```

**C(x): Cross-Modal Coherence**

Does the structure maintain consistent graph topology across independent observational bands?

```
C(x) = (2 / (B*(B-1))) * sum_{i<j} coh(G_x^{band_i}, G_x^{band_j})

Where:
  B = number of observational bands
  G_x^{band} = graph of structure x in band
  coh(G1, G2) = normalized cross-spectral coherence:
                 |lambda_2(L_1) - lambda_2(L_2)| / max(lambda_2(L_1), lambda_2(L_2))
                 inverted so 1.0 = perfect cross-band coherence

Crate mapping:
  - ruvector-graph (hyperedges): Multi-band hypergraph construction
  - ruvector-coherence: Spectral coherence per band and cross-band
  - ruvector-sparsifier: Sparsification quality comparison across bands

Calibration:
  - C < 0.3: No cross-band structural correlation (noise or unrelated)
  - 0.3 < C < 0.7: Partial cross-band coherence (physically plausible)
  - C > 0.7: Strong cross-band structural coherence (same physics)
```

**N(x): Non-Natural Fit**

How well does the structure fit known astrophysical generation mechanisms?

```
N(x) = 1 - max_{m in models} fit(x, m)

Where:
  models = {thermal, synchrotron, inverse_compton, gravitational,
            bremsstrahlung, blackbody, power_law, turbulence}
  fit(x, m) = goodness-of-fit between structure's graph properties
              and model m's predicted graph topology

Sub-components:
  N_compression = 1 - (compressed_size / random_size)
    High N_compression means the signal is more compressible than random noise
    but in a way not explained by known physics

  N_information = Phi(x) / Phi_max(|V|)
    Normalized integrated information, high means the structure is
    more integrated than random or simple processes produce

  N_kolmogorov = 1 - K(x) / |x|
    Approximate Kolmogorov complexity, low complexity relative to size
    suggests algorithmic rather than stochastic origin

Crate mapping:
  - ruvector-consciousness: Phi computation
  - ruvector-temporal-tensor: Compression ratio measurement
  - ruvector-coherence: Spectral comparison against model templates

WARNING: N(x) > 0.8 is NOT evidence of engineering. It means the
structure cannot be explained by catalogued natural mechanisms and
requires new physics or new astrophysics. The overwhelmingly likely
explanation is always undiscovered natural phenomena.
```

**Bonus Terms**

```
bonus(x) = sum of:
  + 0.1 if structure survives statistical injection tests
  + 0.1 if structure persists under data quality cuts
  + 0.1 if independent teams reproduce the detection
  + 0.2 if structure has temporal predictive power
  + 0.2 if structure exhibits response-like temporal correlation
```

### 6.3 E-Score Interpretation

| E-Score Range | Classification | Action |
|--------------|----------------|--------|
| 0.00 - 0.05 | Background noise / known object | Archive, update templates |
| 0.05 - 0.15 | Interesting variant of known class | Flag for specialist review |
| 0.15 - 0.35 | Structurally novel, single-band | Priority follow-up observation |
| 0.35 - 0.60 | Cross-modal structural anomaly | Multi-wavelength campaign |
| 0.60 - 0.80 | Exotic structure candidate | Dedicated observation program |
| 0.80 - 1.00+ | Unprecedented coherent structure | Maximum priority, independent verification |

---

## 7. Pipeline Architecture

### 7.1 End-to-End Flow

```
 RAW DATA                   GRAPH DOMAIN                ANALYSIS                 SCORING
 =========                  ============                ========                 =======

 FITS/CSV/VOTable           Adjacency Lists             Partitions               E-Score
 Catalogs                   Laplacian                   Eigenvalues              Rankings
 Images                     CSR Matrices                Cut Values               Alerts
 Time Series                Hyperedges                  Phi Values
 Spectrograms               Temporal Edges              Delta Sequences

 ┌─────────┐   A1    ┌──────────────┐   A3    ┌──────────────┐   A4-A6   ┌────────────┐
 │ Ingest  │───────► │ Graph Build  │───────► │ Sparsify     │────────► │ Analyze    │
 │         │         │              │         │              │          │            │
 │ FITS    │         │ Schema A2    │         │ Backbone     │          │ MinCut     │
 │ CSV     │         │ Node/Edge    │         │ Importance   │          │ Coherence  │
 │ VOTable │         │ Mapping      │         │ Sampling     │          │ Phi        │
 │ HDF5    │         │              │         │ Audit        │          │ PPR        │
 └─────────┘         └──────────────┘         └──────────────┘          └─────┬──────┘
                                                                              │
                                                                              │ A7
                                                                              ▼
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                            EXOTIC SCORING ENGINE                                │
 │                                                                                 │
 │   P(x) = persistence     S(x) = novelty     C(x) = coherence    N(x) = non-nat │
 │                                                                                 │
 │   E-Score = P * S * C * N * (1 + bonus)                                        │
 │                                                                                 │
 │   Output: ranked anomaly list with full graph provenance                        │
 └──────────────────────────────────────────────────────┬──────────────────────────┘
                                                        │
                                              A8 ◄──────┤──────► A9
                                                        │
                                              ┌─────────▼──────────┐
                                              │ Cross-Modal Fusion │
                                              │ Delta Tracking     │
                                              │ Temporal Monitor   │
                                              └─────────┬──────────┘
                                                        │ A10
                                              ┌─────────▼──────────┐
                                              │ Publication        │
                                              │ Validation         │
                                              │ Reproducibility    │
                                              └────────────────────┘
```

### 7.2 Crate Integration Map

```
                        ruvector-graph
                        (property graph, hyperedges, Cypher)
                             │
                ┌────────────┼────────────────┐
                │            │                │
                ▼            ▼                ▼
        ruvector-mincut   ruvector-sparsifier  ruvector-solver
        (LocalKCut,       (ADKKP16,           (ForwardPush PPR,
         boundary          spectral            Neumann,
         detection)        sampling)           CG, BMSSP)
                │            │                │
                └────────────┼────────────────┘
                             │
                ┌────────────┼──────────────────────┐
                │            │                      │
                ▼            ▼                      ▼
      ruvector-coherence  ruvector-consciousness  ruvector-delta-core
      (Fiedler, spectral  (IIT Phi, emergence,    (D-spaces, delta
       gap, effective      quantum collapse,       streams, windows,
       resistance, HNSW    SIMD-accelerated)       compression)
       health monitor)           │
                                 ▼
                          ruvector-temporal-tensor
                          (tiered quantization,
                           temporal segments,
                           access-driven compression)

                    ruqu-exotic
                    (quantum decay, interference search,
                     reversible memory, swarm interference)
                              │
                              ▼
                    ruvector-domain-expansion
                    (meta-learning, transfer priors,
                     policy kernels, tool orchestration)

                    rvf
                    (append-only storage, overlay epochs,
                     min-cut witnesses, delta segments)
```

---

## 8. Milestones

### Phase 1: Foundation (Weeks 1-4)

**M1.1: Data Ingestion Framework** (Week 1-2)
- Build FITS/VOTable/CSV reader producing `ruvector-graph::GraphDB` instances
- Target datasets: CHIME/FRB catalog, Fermi 4FGL-DR4, SDSS galaxy cluster sample
- Deliverable: `scripts/ingest/` with per-dataset loaders
- Validation: Load 3 catalogs, verify node/edge counts match expected

**M1.2: Graph Schema Library** (Week 2-3)
- Define and document the 5 graph types from Section 4 (signal, source, spatial, temporal, cross-modal)
- Implement schema builders for each type
- Deliverable: `src/exotic_discovery/graph_schemas.rs`
- Validation: Unit tests constructing each graph type from synthetic data

**M1.3: MinCut on Synthetic Signals** (Week 3-4)
- Generate synthetic FRB waterfalls with known sub-burst structure
- Run `ruvector-mincut::LocalKCut` on signal graphs
- Verify mincut recovers known structure boundaries
- Deliverable: Benchmark showing mincut partition accuracy vs ground truth
- Validation: >90% boundary recovery on synthetic data with SNR > 5

**M1.4: Spectral Coherence Baseline** (Week 3-4)
- Compute Fiedler values and spectral gaps on synthetic source catalogs
- Establish baseline distributions for "known" vs "novel" structures
- Deliverable: Statistical distributions of coherence metrics for known classes
- Validation: Fiedler value separates at least 3 synthetic classes with p < 0.01

### Phase 2: Real Data (Weeks 5-8)

**M2.1: CHIME/FRB Boundary Analysis** (Week 5-6)
- Ingest CHIME first catalog (536 FRBs)
- Construct source graph in DM-width-fluence space
- Run mincut: identify structural boundaries in FRB population
- Run spectral coherence: measure integration of repeater vs one-off populations
- Deliverable: Partitioned FRB population with boundary sources identified
- Novel output expected: Sub-populations not captured by simple DM or width cuts

**M2.2: Fermi Unassociated Source Classification** (Week 5-6)
- Ingest 4FGL-DR4 catalog
- Construct source graph in spectral-variability space
- Run ForwardPush PPR from unassociated sources to associated source neighborhoods
- Run mincut on the boundary between associated and unassociated regions
- Deliverable: Graph-based classification of unassociated gamma-ray sources
- Novel output expected: Coherent sub-classes within the "unassociated" category

**M2.3: LoTSS Diffuse Emission Detection** (Week 6-7)
- Ingest LoTSS DR2 catalog for a cluster field (Abell 2255 or Coma)
- Construct spatial graph with spectral index edges
- Run recursive mincut to separate diffuse emission from point sources
- Compute spectral coherence of the diffuse emission sub-graph
- Deliverable: Diffuse emission boundary map compared to published results
- Novel output expected: Structural layers within diffuse emission (halos vs relics vs bridges)

**M2.4: ZTF Lightcurve State Transitions** (Week 7-8)
- Query ZTF for lightcurves of known interesting variable stars (e.g., FU Ori types, symbiotic stars)
- Construct temporal graphs
- Run mincut to identify state transition boundaries
- Track delta behavior across transitions
- Deliverable: Automated state transition detector
- Novel output expected: Pre-transition structural signatures (warning signs before mode change)

### Phase 3: Exotic Scoring (Weeks 9-12)

**M3.1: E-Score Implementation** (Week 9-10)
- Implement the 4-component scoring system from Section 6
- Build template library for known astrophysical classes
- Calibrate on known objects (should score < 0.15)
- Deliverable: `src/exotic_discovery/scoring.rs` with full E-Score computation
- Validation: Known pulsars, AGN, galaxy clusters all score < 0.15

**M3.2: Cross-Modal Fusion** (Week 10-11)
- Cross-match LoTSS radio with SDSS optical for a test field
- Build cross-modal hypergraph
- Compute cross-band spectral coherence
- Score with full E-Score including C(x) component
- Deliverable: First cross-modal E-Scores for real sources
- Novel output expected: Sources with high cross-modal coherence that are not in existing catalogs

**M3.3: Phi on Signal Structures** (Week 11-12)
- Compute IIT Phi on the highest-scoring structures from M3.1
- Use spectral Phi approximation for graphs with >16 nodes
- Compare Phi values against null distribution (shuffled graphs)
- Deliverable: Phi measurements for top-100 exotic candidates
- Novel output expected: Structures with statistically significant integrated information

### Phase 4: Temporal Monitoring and Validation (Weeks 13-16)

**M4.1: Delta Behavior Tracking** (Week 13-14)
- Set up delta-core tracking for multi-epoch datasets (VLASS, ZTF)
- Measure D-space properties of the highest-scoring structures
- Track persistence over time
- Deliverable: Temporal evolution of E-Scores for candidate structures

**M4.2: Statistical Validation** (Week 14-15)
- Injection tests: insert synthetic structures, measure recovery
- Null tests: run pipeline on shuffled/randomized data
- False positive rate estimation
- Deliverable: ROC curves and false positive rates for each discovery tier

**M4.3: First Results Document** (Week 15-16)
- Compile results for publication
- Top-N exotic structure candidates with full provenance
- Statistical validation
- Comparison with traditional detection methods
- Deliverable: Draft paper or technical report

---

## 9. Proof-of-Concept Experiments (Runnable Today)

### Experiment 1: MinCut on CHIME/FRB Population Graph

**Objective:** Demonstrate that LocalKCut reveals structural boundaries in the FRB population that simple threshold cuts miss.

**Data:** CHIME/FRB first catalog (https://www.chime-frb.ca/catalog), 536 FRBs.

**Procedure:**
1. Download catalog CSV
2. Extract features: DM, width, fluence, scattering time, spectral index, bandwidth
3. Construct k-NN graph (k=10) in 6D feature space, edge weights = Gaussian kernel
4. Run `MinCutBuilder::new().exact().with_edges(edges).build()` from ruvector-mincut
5. Extract minimum cut partitions recursively (hierarchical decomposition)
6. Compare partition membership with known repeater/one-off classification

**Expected outcome:** MinCut partitions should NOT align perfectly with the repeater/one-off split. They should reveal intermediate populations (e.g., "repeater-like one-offs" or "structurally isolated repeaters") that suggest the binary classification is an oversimplification.

**RuVector code sketch:**
```rust
use ruvector_mincut::{MinCutBuilder, DynamicMinCut};
use ruvector_coherence::spectral::{CsrMatrixView, estimate_fiedler, SpectralConfig};

// Build graph from CHIME catalog
let edges: Vec<(usize, usize, f64)> = build_knn_graph(&frb_features, k=10);
let mut mc = MinCutBuilder::new()
    .exact()
    .with_edges(edges.clone())
    .build()?;

// Find minimum cut
let cut_value = mc.min_cut_value();
let (partition_a, partition_b) = mc.min_cut_partition()?;

// Measure coherence of each partition
let laplacian_a = CsrMatrixView::build_laplacian(partition_a.len(), &edges_a);
let fiedler_a = estimate_fiedler(&laplacian_a, &SpectralConfig::default());
```

### Experiment 2: Spectral Coherence of Fermi Unassociated Sources

**Objective:** Determine whether the 2200+ unassociated Fermi-LAT sources form coherent sub-populations or are structurally random.

**Data:** Fermi 4FGL-DR4 catalog (https://fermi.gsfc.nasa.gov/ssc/data/access/lat/14yr_catalog/)

**Procedure:**
1. Download 4FGL-DR4 FITS catalog
2. Extract: photon index, spectral curvature, energy flux, variability index, galactic latitude
3. Separate associated (5000+) from unassociated (2200+) sources
4. Build source graph for unassociated sources (k-NN, k=15)
5. Compute spectral coherence: Fiedler value, spectral gap, effective resistance
6. Compare against null model: randomly selected subsets of associated sources of same size
7. Run ForwardPush PPR from each unassociated source, measure PPR distribution over known classes

**Expected outcome:** Unassociated sources should show internal structure (non-trivial Fiedler value) indicating coherent sub-populations. PPR analysis should reveal "almost-classified" sources near decision boundaries and "truly novel" sources far from all known classes.

### Experiment 3: Boundary Detection in IllustrisTNG Cosmic Web

**Objective:** Validate boundary-first detection by recovering known cosmic web structure (filaments, voids, halos) from simulated data using mincut + spectral coherence, without using density thresholds.

**Data:** IllustrisTNG-100 snapshot at z=0 (https://www.tng-project.org/data/), subhalo catalog.

**Procedure:**
1. Download TNG100 subhalo catalog via API
2. Construct 3D spatial graph of subhalos (Delaunay triangulation)
3. Edge weights: inverse distance * mass product
4. Run recursive mincut to hierarchically decompose the cosmic web
5. Compare: do mincut boundaries align with known filament/void boundaries?
6. Compute Fiedler value within each partition: filaments should have low Fiedler (elongated), halos high (compact)

**Expected outcome:** MinCut should recover filament-void boundaries WITHOUT density thresholds. The recursive cut hierarchy should naturally produce: first cut separates voids from structure, second level separates filaments from nodes, third level identifies individual halos. This validates that boundary-first detection works on known structures before applying to unknown data.

### Experiment 4: Delta Behavior of ZTF Symbiotic Star Lightcurves

**Objective:** Detect state transitions in symbiotic star lightcurves using D-space analysis rather than amplitude thresholds.

**Data:** ZTF lightcurves for known symbiotic stars (e.g., AG Dra, Z And, CH Cyg) via IRSA (https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves)

**Procedure:**
1. Query ZTF lightcurves for 5-10 symbiotic stars with known outburst history
2. Construct temporal graph: nodes = observations, edges = temporal adjacency
3. Compute delta stream: `VectorDelta::compute(&flux_prev, &flux_next)`
4. Build delta-weighted temporal graph (edges weighted by delta magnitude)
5. Run mincut on temporal graph: cuts should locate state transitions
6. Compare detected transitions with published outburst dates

**Expected outcome:** D-space analysis should detect transitions 2-5 observations BEFORE the amplitude threshold crossing, because the delta (rate of change) shifts before the absolute flux does. This demonstrates predictive power of boundary-first temporal analysis.

### Experiment 5: Cross-Band Structural Coherence of Radio Galaxies

**Objective:** Measure whether radio galaxy structure (lobes, jets, cores) maintains consistent graph topology when observed at different frequencies.

**Data:** LoTSS (150 MHz) cross-matched with VLASS (3 GHz) for a sample of resolved radio galaxies.

**Procedure:**
1. Select 20-30 resolved radio galaxies visible in both LoTSS and VLASS
2. For each galaxy, construct source component graph in each band
3. Compute Fiedler value, spectral gap, mincut value in each band
4. Measure cross-band coherence: C(x) from Section 6
5. Rank by cross-band coherence anomaly

**Expected outcome:** Normal radio galaxies should show consistent structure across bands (high C). Sources with anomalous C(x) -- high radio coherence but low optical coherence, or structure that changes topology with frequency -- are candidates for exotic physics (e.g., spectral ageing revealing old vs new emission, or frequency-dependent absorption revealing intervening structure).

---

## 10. 100-Year Projection

### Decade 1 (2026-2036): Foundation and First Discoveries

**Infrastructure:** RuVector exotic discovery pipeline operational on all major surveys. Real-time ingestion from LSST (20 TB/night), SKA precursors, Einstein Probe, Vera Rubin Observatory. Graph construction automated for standard data types.

**Discoveries:** First catalog of "boundary-first objects" -- astrophysical structures detected solely by their graph topology, not their amplitude. Estimated 100-1000 such objects in first 5 years. New astrophysical classes established: structural transients (topology changes without flux changes), coherence fronts (wave-like propagation of spectral coherence), and delta attractors (systems that oscillate in D-space without periodic amplitude variation).

**Scientific impact:** Boundary-first detection adopted as complementary method to amplitude-first surveys. First publications demonstrating structures missed by traditional pipelines. E-Score system standardized and adopted by survey collaborations.

### Decade 2 (2036-2046): Scale and Multi-Messenger

**Infrastructure:** Full SKA operational (tens of billions of radio sources). RuVector running on SKA data pipeline. Cross-modal hypergraphs spanning radio, optical, X-ray, gravitational wave, and neutrino data. Temporal tracking of all catalogued boundary-first objects.

**Discoveries:** Large-scale coherence structures in the cosmic web detected via graph topology. Temporal attractors in repeating transient populations (FRBs, magnetars) revealing underlying dynamical systems. Information-theoretic mapping of the local universe -- Phi values computed for every cluster and supercluster, revealing integration hierarchy.

**Scientific impact:** New understanding of cosmic web dynamics through boundary evolution. Discovery of "structural fossils" -- topological remnants of past events preserved in graph structure but invisible in amplitude. Possible detection of non-random quiet zones requiring new physics.

### Decade 3-5 (2046-2076): Autonomous Discovery

**Infrastructure:** Self-optimizing discovery pipeline using `ruvector-domain-expansion` transfer learning. Pipeline discovers new graph schema types by learning from successful detections. Autonomous follow-up observation scheduling based on E-Score and temporal predictions.

**Discoveries:** Complete structural taxonomy of the observable universe. Every radio source, galaxy, and transient has a graph fingerprint. Discovery of structural universals -- graph motifs that appear at all scales (stellar, galactic, cosmological). Detection of cross-scale coherence: structures whose graph topology is self-similar across decades of spatial scale.

**Scientific impact:** Fundamental physics implications of structural universals. If the same graph motifs appear at quantum, stellar, and cosmological scales, this constrains theories of structure formation. Possible detection of structures that require revision of standard cosmology.

### Decade 5-10 (2076-2126): The Boundary Telescope

**Infrastructure:** "Boundary Telescope" -- a virtual instrument that observes the universe through its graph topology rather than its electromagnetic emission. Combines all observational data into a single evolving temporal hypergraph. RuVector as the operating system for structural observation.

**Capabilities:**
- Real-time boundary tracking across the entire observable universe
- Predictive detection: identifying structures before they become amplitude-visible
- Structural archaeology: reconstructing past structures from topological fossils
- Information-theoretic cartography: mapping the Phi landscape of the cosmos
- Anomaly detection at the intersection of all discovery tiers simultaneously

**Possible far-edge outcomes:**
- Detection of structures that violate known physics in specific, quantifiable ways
- Evidence for or against engineered structures (with extraordinary evidence standards)
- Discovery of structural communication -- information encoded in topological changes
- Unified field theory of cosmic structure connecting quantum and cosmological graph motifs

**What this system becomes:** Not a telescope in the traditional sense, but a "structure sense" -- the ability to perceive the universe through its organizational principles rather than its emissions. Just as spectroscopy revealed chemical composition and redshift revealed expansion, boundary-first analysis reveals the informational architecture of the cosmos.

The 100-year trajectory is: detect boundaries (2026) -> catalog structures (2030s) -> discover universals (2050s) -> predict evolution (2070s) -> perceive organization (2100s).

---

## 11. Risk Assessment and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Graph construction choices dominate results | High | High | Test multiple graph schemas, validate on simulation |
| Computational cost of Phi on large graphs | High | Medium | Use spectral/stochastic Phi approximations |
| False positives from data artifacts | Medium | High | Injection tests, null tests, cross-survey validation |
| MinCut instability on noisy data | Medium | Medium | Use approximate algorithm, sparsify first |
| Cross-matching errors creating false structure | Medium | Medium | Positional/statistical cross-match validation |
| Overfitting E-Score to training data | Low | High | Hold-out validation, blinded scoring |
| Computational infeasibility at SKA scale | Low (long-term) | High | Sublinear algorithms (PPR, spectral sparsification) |

---

## 12. Open Questions

1. **What is the minimum SNR for boundary-first detection?** Traditional detection requires SNR > 5. Does boundary detection have a different threshold, or a different *kind* of threshold (structural complexity rather than amplitude)?

2. **Can delta behavior predict state transitions?** If the D-space representation shifts before amplitude shifts, how far in advance? Is the lead time useful for triggering observations?

3. **Is Phi physically meaningful for astrophysical systems?** IIT was developed for consciousness theory. When applied to astrophysical graphs, does non-trivial Phi correspond to physically meaningful integration, or is it an artifact of graph construction?

4. **What is the natural template library?** The E-Score N(x) component requires a library of "known natural" graph topologies. How comprehensive must this be before high N(x) scores are meaningful?

5. **Does cross-scale structural universality exist?** If the same graph motifs appear at stellar and cosmological scales, is this physics or is it a property of graph analysis itself?

---

## 13. Resource Requirements

### Compute
- Phase 1-2: Single workstation (16+ cores, 64 GB RAM, 1 TB SSD)
- Phase 3-4: Small cluster or cloud (100 cores, 512 GB RAM aggregate)
- Long-term: Integration with survey computing infrastructure

### Storage
- Catalogs: ~100 GB total for all listed datasets
- Graph representations: ~500 GB for full analysis
- Results and provenance: ~100 GB

### Human Effort
- 1 lead researcher (boundary-first methods, RuVector development)
- 1 domain expert per target survey (part-time, for validation)
- Estimated 6-12 person-months for Phase 1-4

---

## 14. Success Criteria

**Minimum viable success:**
- Pipeline running on 3+ real datasets
- MinCut producing non-trivial partitions validated against known structure
- At least 1 structure detected by boundary analysis that was missed by amplitude analysis

**Strong success:**
- E-Score system calibrated and producing ranked anomaly lists
- Cross-modal coherence revealing multi-wavelength structural correlations
- Publication-ready results with statistical validation

**Transformative success:**
- New astrophysical class discovered (detected by boundaries, invisible to amplitudes)
- Boundary-first detection adopted by a major survey collaboration
- Evidence for structural universals across spatial scales
