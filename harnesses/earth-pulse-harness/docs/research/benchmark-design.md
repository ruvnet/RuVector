# Benchmark Design: Earth Pulse Observatory

> Design principle. Treat the 26-second pulse as a **causal-discovery
> benchmark**, not a mystery story. The unit of progress is not a narrative
> ("Earth has a heartbeat") but a measurable claim: **"mechanism M predicts
> observable O with skill S, beating baseline B by margin Δ on held-out data."**
> Freeze the physics; evolve the harness.

> Sequencing rule. **Build the honest scoring loop BEFORE making anything
> autonomous.** A self-improving harness that optimizes a leaky or sloppy score
> will confidently converge on nonsense. Scoring correctness is gate zero.

## 1. The first discovery target

> **Can the system predict the amplitude of the 26-second pulse from
> ocean-state variables?**

This is deliberately the *easiest defensible* target. It does not require
solving the mechanism. It asks only: is the pulse amplitude **forecastable**
from ocean state, and if so, *which* ocean variables carry the predictive
signal? The answer immediately constrains the hypothesis space:

- If amplitude is well-predicted by swell height/direction + shelf geometry,
  that supports an **ocean-forcing** carrier.
- If amplitude is **not** predictable from ocean state but a volcanic proxy
  helps, that supports a **volcanic/hydrothermal** contribution.
- If both matter, that supports the **coupled** hypothesis (the one to watch).

The target is a regression: predict monthly (and later, finer-grained) pulse
amplitude `A(t)` at the source-region back-azimuth.

## 2. Baselines

Baselines are ordered from trivial to physically motivated. The candidate must
beat the strongest applicable one — not just the trivial one.

| ID | Baseline | What it encodes | Why it matters |
|----|----------|-----------------|----------------|
| B0 | **Seasonal monthly average** | `A(t) ≈ mean amplitude for that calendar month` | The "do nothing clever" bar. Captures pure seasonality. **Primary acceptance baseline.** |
| B1 | **Swell-height-only** | linear/local fit of `A` on significant wave height in source region | Tests whether raw wave energy alone explains amplitude |
| B2 | **Swell-direction + shelf-normal** | `A` as function of wave direction relative to local shelf normal | Tests primary-microseism geometry (waves hitting the shelf) |
| B3 | **Tide-phase-only** | `A` as function of tidal phase | Tests for a tidal modulation signature |
| B4 | **Volcanic-proxy → glides** | predict glide onset/amplitude from a volcanic/hydrothermal proxy | Tests the volcanic contribution, especially for the gliding tremors |

Each baseline is a falsifiable mini-hypothesis. We record skill for **all** of
them on every split, not just the winner — the *pattern* of which baselines work
is itself evidence about mechanism.

## 3. Candidate model

**ruVector nearest-neighbor predictor.** For a target time `t`, build a feature
vector and retrieve the *k* most similar historical states; predict `A(t)` from
their (distance-weighted) outcomes. Features:

- Swell **significant height** in the source region
- Swell **direction** (and direction relative to shelf normal)
- **Tide phase**
- **Source-region weather** (surface pressure, wind)
- **Prior amplitude** `A(t−1), A(t−2), …` (autoregressive memory)
- **Spectral coherence** of the 26 s line across stations (signal-quality / SNR)

Rationale for nearest-neighbor first: it is **interpretable** (you can inspect
the retrieved analog months), **non-parametric** (no strong functional-form
assumption to bias the discovery), and it makes the **feature ablation** the
unit of evidence — dropping a feature group and watching skill change tells us
which physics matters.

## 4. Acceptance test (first target)

> **The evolved harness must beat the seasonal baseline (B0) by ≥ 10% on
> held-out months.**

Specifics:
- **Metric:** skill = reduction in held-out error vs. B0. Report both RMSE and a
  correlation (e.g. anomaly correlation) so a model can't win by variance
  collapse. Margin is computed on the **same** held-out months for candidate
  and B0.
- **Split:** **time-blocked** held-out months (and later, held-out years). No
  random shuffling — random splits leak autocorrelated neighbors and inflate
  skill. Hold out contiguous blocks.
- **Masking:** exclude months/windows contaminated by large **earthquakes**
  (from the catalog) and known **instrument outages**. The mask is defined
  *before* scoring, not tuned to help the candidate.
- **Honesty guards:** features must be **causally available** at prediction time
  (no future leakage); the ablation must show the predictor degrades sensibly
  when informative features are removed (a model that ignores its inputs and
  still "wins" is a bug in the score).

## 5. The data spine

All real observations live under `data/`. **The harness never fabricates
observations** — empty is acceptable; invented is not.

| Spine | Directory | Content | Use |
|-------|-----------|---------|-----|
| Seismic | `data/seismic/` | broadband/long-period waveforms or precomputed 26 s amplitude & coherence time series, per station | the prediction **target** and signal quality |
| Ocean waves | `data/ocean/` | wave reanalysis (sig. wave height, period, direction) at source region | primary ocean-forcing features |
| Tides | `data/tides/` | tidal model / gauge phase & amplitude near source | tide-phase baseline & feature |
| Bathymetry | `data/bathymetry/` | seafloor depth grid for shelf geometry near Bight of Bonny | shelf-normal, resonance geometry |
| Weather / pressure | `data/weather/` (added when populated) | surface pressure, wind reanalysis in source region | weather features |
| Volcanic proxies | `data/volcanic/` (added when populated) | thermal/seismic/degassing proxies near São Tomé | volcanic baseline & glide features |
| Earthquake catalog | `data/earthquakes/` (added when populated) | event times/magnitudes | **masking** contaminated windows |
| Literature corpus | `data/papers/` | one file per cited source, embedded | provenance for every promoted claim |

Provenance rule: every dataset README states **source, units, time span, and
license**. Every promoted scientific claim maps to (a) a literature file in
`data/papers/` and (b) the observation files that support it.

## 6. The discovery ladder (Levels 1–6)

Each level has an explicit acceptance test. A level is **not** climbed until its
test passes on held-out data. Higher levels presuppose lower ones.

### Level 1 — Reproduce the observable
- **Goal:** independently recover the ~26 s line and its source back-azimuth
  from raw seismic data.
- **Acceptance:** spectral peak at 26 s detected on ≥ 2 independent stations;
  back-azimuth consistent with Gulf of Guinea within stated tolerance.

### Level 2 — Characterize variability
- **Goal:** produce a clean, masked amplitude (and coherence) time series.
- **Acceptance:** time series with documented masking; seasonal structure
  quantified; B0 (seasonal average) computed and stored as the reference bar.

### Level 3 — Predict amplitude from ocean state (FIRST TARGET)
- **Goal:** candidate ruVector NN beats B0.
- **Acceptance:** **≥ 10% skill improvement over B0** on time-blocked held-out
  months, with the honesty guards of §4 satisfied.

### Level 4 — Attribute the skill (which physics?)
- **Goal:** feature-group ablation identifies *which* inputs carry the signal.
- **Acceptance:** ablation is stable across folds; the winning feature set has a
  mechanistic interpretation mapped to a hypothesis in `hypothesis-catalog.md`.

### Level 5 — Predict the gliding tremors
- **Goal:** forecast glide onset / glide rate (the Bruland & Hadziioannou 2023
  observable), not just steady amplitude.
- **Acceptance:** beat a glide-specific baseline (e.g. B4 volcanic-proxy) by a
  pre-registered margin on held-out events; carrier and glide predictions are
  jointly consistent.

### Level 6 — Mechanism selection
- **Goal:** the *combination* of skill patterns (Levels 3–5) selects one
  hypothesis (or the coupled hypothesis) over its rivals by the discovery score.
- **Acceptance:** discovery score (§7) for the leading hypothesis exceeds the
  promotion gate AND the killer contradiction (see `hypothesis-catalog.md`) for
  rivals is observed in the data.

## 7. Discovery score and promotion gate

A single scalar that fuses predictive skill with scientific honesty, so the
autonomous loop optimizes the right thing.

```
discovery_score(H) =
      w_skill   * skill(H)            # held-out skill margin over best baseline (clipped to [0,1])
    + w_robust  * robustness(H)       # skill stability across folds / stations (1 - normalized variance)
    + w_falsif  * falsifiability(H)   # did H survive its own killer contradiction test? (0 or 1)
    + w_provN   * provenance(H)       # fraction of H's claims mapped to data/papers/ + observations
    - w_leak    * leakage_penalty(H)  # penalty for any detected future/target leakage
    - w_complex * complexity(H)       # penalty for free parameters (Occam)
```

Suggested initial weights (sum of positives = 1; penalties subtract):
`w_skill=0.40, w_robust=0.20, w_falsif=0.20, w_provN=0.20,
w_leak=0.50 (hard), w_complex=0.10`.

- `skill(H)` is normalized so that exactly meeting the **+10% over B0** bar maps
  to a defined positive value; below the bar contributes 0.
- `leakage_penalty` is intentionally large: **any** leakage zeroes promotion.
- `provenance(H)` enforces that you cannot promote a claim with no paper/data
  behind it.

**Promotion gate.** A hypothesis is promoted (its claim becomes part of the
harness's standing knowledge) only if, on held-out data:

1. `skill(H)` clears the **≥ 10% over B0** acceptance bar, AND
2. `falsifiability(H) = 1` (it ran and survived its killer-contradiction test),
   AND
3. `leakage_penalty(H) = 0`, AND
4. `provenance(H) ≥ 0.9`, AND
5. `discovery_score(H) ≥ 0.70` (initial threshold; tune only with held-out
   validation, never on the test set), AND
6. It exceeds the next-best rival hypothesis by a pre-registered margin.

## 8. Build order (honesty before autonomy)

1. **Scoring loop first.** Implement B0–B4, the masking, the time-blocked split,
   and the leakage detector. Verify by feeding the candidate a *deliberately
   leaky* feature and confirming the leakage penalty fires.
2. **Manual candidate.** Run the ruVector NN by hand; confirm it can clear or
   miss the bar honestly.
3. **Provenance plumbing.** Wire `data/papers/` so claims map to files.
4. **Only then automate.** Let the loop propose features/hypotheses, but every
   promotion still passes the gate of §7.

Anti-goals: tuning on the test set, random splits, unmasked earthquakes,
counting a narrative as a result, or promoting any claim without provenance.
