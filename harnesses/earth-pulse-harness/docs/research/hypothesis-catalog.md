# Hypothesis Catalog: Mechanisms for the 26-Second Pulse

> Status. The scores below are **subjective priors**, set before any harness run.
> They encode current belief from the literature (see
> `26-second-pulse-literature.md`), **not** results. They must be replaced by
> posterior discovery scores (see `benchmark-design.md` §7) as evidence
> accumulates. Do not cite a prior as a finding.

> Leading bet. **Ocean shelf resonance** most plausibly explains the **stable
> carrier frequency** (why ~26 s, and why it's narrowband and fixed). But the
> **gliding tremors** (Bruland & Hadziioannou 2023) may require a **second**
> mechanism — most likely **volcanic/hydrothermal** near São Tomé. Therefore the
> **coupled ocean+geology** hypothesis is the one to watch: it is the only family
> that can naturally produce both a stable ocean-modulated carrier *and*
> drifting glides from the same fixed source.

## Ranked catalog

| Rank | Hypothesis | Prior | Core claim |
|------|------------|------:|-----------|
| 1 | Ocean shelf resonance | **0.72** | Swell forcing excites a geometric resonance of the shelf/coast near the Bight of Bonny that selects the 26 s line |
| 2 | Coupled ocean + geology (**watch**) | **0.68** | Ocean forcing sets the carrier; a volcanic/hydrothermal source near São Tomé produces the glides; the two interact |
| 3 | Water-column / bathymetric mode | **0.55** | A trapped water-column or bathymetric normal mode resonates at 26 s, excited by ocean energy |
| 4 | Volcanic / hydrothermal tremor | **0.46** | A subsurface fluid/magmatic oscillator near São Tomé generates a narrowband tremor at 26 s |
| 5 | Instrument / processing artifact | **0.12** | The line is an artifact of instrumentation or analysis, not a true ground motion |

> Priors do not sum to 1 — these are independent plausibility weights, not a
> normalized probability distribution. Several mechanisms can be partly true
> (hence the coupled hypothesis).

---

## H1 — Ocean shelf resonance  (prior ≈ 0.72)

- **Expected evidence / prediction:** Pulse amplitude tracks **source-region
  swell energy and direction relative to the shelf normal**, with a clear
  **seasonal** signature following the regional wave climate. The carrier
  frequency is set by **shelf geometry** (width/slope), so it should be stable
  as long as geometry is stable, but its *amplitude* should be highly
  predictable from ocean state. Strong skill at **benchmark Level 3** from
  swell+geometry features (baselines B1/B2).
- **Killer contradiction (falsifier):** Amplitude shows **no dependence** on
  source-region swell height/direction across many held-out months *and* a
  non-ocean proxy (e.g. volcanic) explains it better. Or: the carrier frequency
  **shifts** in a way inconsistent with fixed shelf geometry while swell is
  steady.
- **ruVector test:** Feature ablation at Level 4 — nearest-neighbor predictor
  retains skill with **swell + shelf-normal** features and **loses** skill when
  they are removed; analog retrieval should pull months with similar swell
  geometry. Distinguishes H1 by the *swell-geometry feature group* carrying the
  skill.

## H2 — Coupled ocean + geology  (prior ≈ 0.68, **the one to watch**)

- **Expected evidence / prediction:** **Carrier** amplitude is ocean-predictable
  (as H1), **but** the **gliding tremors** are *not* explained by ocean state
  alone and instead correlate with a **volcanic/hydrothermal proxy**. The two
  observables (steady carrier, glide onset/rate) require **different** feature
  groups to predict — the signature of two coupled processes sharing the source
  region.
- **Killer contradiction (falsifier):** A **single** feature group (purely ocean
  *or* purely volcanic) predicts **both** the carrier and the glides equally
  well, removing the need for coupling. Or the carrier and glides are shown to
  be statistically **independent** (no shared timing/phase relationship),
  undermining "coupled."
- **ruVector test:** Run Level 3 (carrier) and Level 5 (glides) ablations
  **separately**; H2 is supported iff the carrier skill comes from ocean
  features while glide skill comes from volcanic features, *and* a joint model
  beats either alone. Distinguishes H2 by **divergent feature attribution**
  across the two observables.

## H3 — Water-column / bathymetric mode  (prior ≈ 0.55)

- **Expected evidence / prediction:** The 26 s frequency matches a **normal mode**
  computable from the local **water depth and bathymetry** (a resonance period
  set by depth, not shelf slope). Frequency should be **insensitive** to swell
  direction but amplitude still excited by ocean energy. Predicted frequency
  from `data/bathymetry/` should match the observed 26 s within tolerance.
- **Killer contradiction (falsifier):** The **computed** water-column/bathymetric
  resonance period for the Bight of Bonny is **far from 26 s** (outside
  uncertainty), or the observed frequency depends on swell direction (which a
  depth mode should not).
- **ruVector test:** A semi-analytic check (bathymetry → predicted mode period)
  plus an ablation showing **direction-independence**: NN skill should be
  insensitive to swell *direction* features while sensitive to depth/energy.
  Distinguishes H3 from H1 by the **absence of a shelf-normal/direction
  dependence**.

## H4 — Volcanic / hydrothermal tremor  (prior ≈ 0.46)

- **Expected evidence / prediction:** Narrowband stability and fixed location
  arise from a **subsurface oscillator** near São Tomé. Amplitude / glide
  behavior should correlate with **volcanic-activity proxies** (thermal,
  degassing, local micro-seismicity) and be **weakly** dependent on ocean state.
  This hypothesis most naturally explains the **glides** (frequency drift as a
  fluid/conduit property changes).
- **Killer contradiction (falsifier):** Amplitude is **strongly and primarily**
  predicted by **ocean** state with **no** residual skill from any volcanic
  proxy; or there is **no** detectable volcanic activity correlated with the
  signal over the record while the line persists unchanged.
- **ruVector test:** Baseline B4 (volcanic-proxy → glides) and a glide-onset
  ablation — H4 is supported iff volcanic features carry the glide skill and
  ocean features add little. Distinguishes H4 by **volcanic-feature dominance**.

## H5 — Instrument / processing artifact  (prior ≈ 0.12)

- **Expected evidence / prediction:** If true, the line should appear with
  characteristics tied to **instrumentation or processing**, not geophysics:
  e.g. present on one station/instrument generation but not on independent ones,
  or an artifact of a specific filter/sampling. Already **disfavored** because
  the signal is seen across decades, instruments, and independent groups
  (literature §1) — hence the low prior.
- **Killer contradiction (falsifier):** Trivially falsified by the existing
  multi-station, multi-decade, multi-group detections; *confirmed* only if those
  detections collapse under a common processing/instrument flaw.
- **ruVector test:** Cross-station **coherence** and instrument-independence
  check at Level 1 — the artifact hypothesis predicts the line should *not*
  cohere across truly independent instruments. Kept in the catalog as the
  mandatory null hypothesis: a responsible harness must be able to detect that
  its target is an artifact.

---

## How priors become posteriors

1. Each hypothesis's `falsifiability` term in the discovery score (see
   `benchmark-design.md` §7) is set by whether its **killer-contradiction test
   actually ran** and the hypothesis **survived**.
2. The `skill` term comes from the relevant benchmark level (Level 3 for
   carrier-amplitude hypotheses, Level 5 for glide hypotheses).
3. Mechanism selection (Level 6) compares **discovery scores**, not priors.
4. Update this catalog's "Prior" column to a "Posterior" column only with
   **held-out** evidence; never tune scores on the test set.

> Expected trajectory (a hypothesis, not a result): H1 carries the carrier,
> H4 carries the glides, and **H2 (coupled)** ends up best because it is the only
> family that jointly explains both — *if and only if* the divergent
> feature-attribution prediction in the H2 ruVector test is observed.
