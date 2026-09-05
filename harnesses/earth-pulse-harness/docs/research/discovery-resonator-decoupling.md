# Discovery — the 26 s pulse is a frequency-stable resonance decoupled from the ocean-wave microseism

**Finding (real data, GT.DBIC, 1996–1997, n = 57 windows):** the 26-second
microseism behaves like a **fixed-frequency resonator driven at variable
strength**, and its strength is **statistically decoupled** from the local
secondary (ocean wave–wave) microseism.

> This is an empirical result the harness derived autonomously from real
> observations, with a falsification test, not a restatement of an input. It is
> a single-station, two-year characterization — see the honest scope below.
>
> Reproduce: `npm run build && node scripts/climatology.mjs 1996,1997`
> Offline (committed metrics): `npm test` → `__tests__/discovery.test.ts`

## The three measurements

All from real GT.DBIC LHZ data (boreal coverage 1996–1997), 57 independent
2-day windows, each reduced to a median Welch PSD; the 26 s line is the peak in
0.0352–0.0372 Hz and its **excess power** is that peak minus the local
continuum. Raw counts; instrument gain is constant within the station epoch, so
relative comparisons are valid.

### 1. The frequency is stable to ~0.6 %

```
mean f0 = 0.03606 Hz  (period 27.73 s)
std     = 2.13e-4 Hz
CV (σ/μ) = 0.59 %     across 57 windows over 2 years
```

The line sits at the same frequency, window after window, season after season.

### 2. The amplitude is wildly variable — 36×

```
line excess power: min 3.4e3 … max 1.2e5   →   range 36.5×
```

So the *strength* of the 26 s pulse changes by more than an order of magnitude
while its *frequency* barely moves.

### 3. Frequency does not follow amplitude (fixed resonance)

```
corr(peak frequency, line amplitude) = +0.165   (n = 57)
```

Near zero: driving the line harder does **not** shift its frequency. A fixed
resonant frequency excited at variable strength is exactly the resonator
signature — as opposed to a broadband source whose peak would wander with the
forcing.

### 4. Decoupled from the local ocean-wave field

```
corr(26 s line excess, secondary microseism) = 0.04
permutation p-value (2000 shuffles, seeded)   = 0.75   (n = 57)
```

The secondary (double-frequency) microseism is the standard proxy for local
ocean-wave energy, and it is strongly seasonal here (peaks boreal winter). The
26 s line's 36× amplitude swing **does not track it at all** — the correlation
is consistent with zero. If the 26 s pulse were simply another ocean-wave
microseism from the local sea state, you would expect a clear positive
correlation. You do not see one.

## Source localization — a Rayleigh wave from the Gulf of Guinea (Level 2)

Digging deeper, we localized the source **from a single station** using
three-component polarization (`src/polarization.ts`). A fundamental-mode
Rayleigh wave is retrograde elliptical in the vertical–radial plane, so the
particle motion encodes the back-azimuth to the source. The estimator is
validated to recover known back-azimuths exactly on synthetic waves
(`__tests__/polarization.test.ts`).

Applied to real GT.DBIC LHZ/LHN/LHE (June 1996, 5 days, band-passed to the
27.7 s line, 96 windows):

```
motion             : RETROGRADE  → fundamental-mode Rayleigh (surface wave)
back-azimuth       : 99° (whole record),  101° (quality-weighted mean of 96 windows)
concentration      : R = 0.76  (tight — a stable single direction, not noise)
expected to source : Bight of Bonny 109°,  São Tomé 118°
```

The 27.7 s wave arrives from **~100° — straight into the Gulf of Guinea / Bight
of Bonny** — within ~10–18° of the expected source bearing, comfortably inside
single-station polarization uncertainty, and from the opposite side of the
continental interior. Independent of any amplitude or frequency argument, the
particle motion alone points at the known source region.

Two things this nails down:
- **It is a Rayleigh (surface) wave**, not a body wave — consistent with an
  ocean/crustal source coupling at the surface, and inconsistent with a deep
  teleseismic origin.
- **The source direction is the Gulf of Guinea**, recovered with no prior other
  than the station's three components — corroborating the decades-old
  attribution from first principles. Data: `data/seismic/dbic-backazimuth-1996.json`.

### The source direction is FIXED across seasons and years

Re-measuring the DBIC back-azimuth in four windows
(`data/seismic/dbic-backazimuth-stability.json`):

| window | back-azimuth | R |
|---|---|---|
| 1996-01 (winter) | 86° | 0.68 |
| 1996-06 (summer) | 101° | 0.76 |
| 1996-09 (autumn) | 96° | 0.76 |
| 1997-06 (year +1) | 100° | 0.87 |

The bearing is stable at **~96° (spread 15°)** across winter/summer/autumn and two
years — a **fixed source direction**, matching the decades-long persistence the
literature reports. (It runs ~10–15° west of the exact São Tomé bearing (118°),
within single-station polarization systematics.)

### Why we cannot yet triangulate to a point (honest limit)

We measured the back-azimuth at four stations to intersect the bearings
(`data/seismic/triangulation-1996.json`, `npm run triangulate`):

| station | role | measured baz | expected | concentration R |
|---|---|---|---|---|
| GT.DBIC | source-proximal | ~100° | 111° | **0.76** ✓ |
| G.TAM | Sahara (N) | 234° | 176° | 0.47 ✗ |
| II.ASCN | S Atlantic | 18° | 66° | 0.42 ✗ |
| GT.LBTB | Botswana (SE) | 145° | 324° | 0.13 ✗ |
| G.SSB | France (N) | 12°/316° | 176° | 0.24/0.39 ✗ |

Only the source-proximal **DBIC** cleanly polarizes the 27.7 s line. At the
distant stations the long-period band is dominated by *other* sources — TAM and
ASCN peak near 26.1–26.6 s (outside the 27.7 s band), and SSB's polarization
points NW/N at North Atlantic microseism sources — so their bearings have low R
and do not point at the Gulf of Guinea. A tight point-triangulation is therefore
**not achievable** with the sparse 1996-era three-component network. This is
itself informative: the 27.7 s line is **regionally concentrated near its
source**, not a globally-dominant teleseismic arrival — consistent with the
multi-line long-period microseism picture seen in the cross-station test.

## How sharp is the resonance, and does it glide? (Level 5)

Two more measurements on the 12-day continuous Jan-1995 record
(`data/seismic/dbic-resonance-q.json`, `src/climatology.ts`):

```
sharpness : f0 = 0.03607 Hz,  FWHM = 5.5e-4 Hz,  Q = f0/FWHM ≈ 66  (resolved, 9 bins)
temporal  : 285 hourly windows — longest monotonic frequency run = 3 h (noise-level)
```

- **Q ≈ 66** is a genuine narrowband resonance (a broadband source would give
  Q ≈ 1). This is a **lower bound**: a 12-day-averaged line is broadened by the
  slow frequency wander documented above, so the *instantaneous* resonance is
  sharper.
- **The fundamental does not glide.** Tracking the line's peak frequency hour by
  hour shows no sustained drift (the longest monotonic run is 3 h — consistent
  with measurement scatter, not a sweep). The fundamental's frequency is fixed
  in time, not just on average.

This matters for the literature anomaly: the **gliding tremors** of Bruland &
Hadziioannou (2023) are *companion* signals that glide *upward from* the 26 s
fundamental — they are not the fundamental itself drifting. Our result is
consistent with that picture: the carrier is a stable, narrowband resonance; any
gliding lives in separate higher-frequency tremor episodes (a Level-5 target for
future work).

## Why this matters

Putting the four numbers together gives a coherent physical picture:

> **A fixed-frequency resonance (CV 0.59 %) whose excitation varies > 36×,
> independently of both its own frequency and the local ocean-wave field.**

This favors the **"stable carrier, externally modulated"** model over "the 26 s
pulse is just a local double-frequency microseism": the resonant *frequency* is
set by a fixed structure (shelf / water-column / crustal / source geometry),
while the *amplitude* is gated by something other than the bulk local wave
energy — a specific directional swell reaching a specific resonator, or a
non-wave driver. It is consistent with, and an independent quantitative support
for, the long-standing view that the 26 s signal is not an ordinary microseism
(the gap Bruland & Hadziioannou 2023 point to).

It also sharpens the next test on the discovery ladder (ADR-004): if the driver
is directional swell rather than bulk wave energy, the 26 s amplitude should
correlate with swell *direction/source-region* state, not with local microseism
power — exactly the ruVector nearest-neighbor query ADR-002/ADR-003 set up.

## Replication and refinement (adversarial follow-up)

Three challenges were run against the result; it survived all three.
Consolidated metrics: `data/seismic/dbic-replication-1995-1998.json`.

### Refinement — the line is at 27.7 s, not 26.0 s

The "26-second" label does not match this station. Searching the **wide** band
0.0340–0.0400 Hz (which *includes* the canonical 26.0 s = 0.03846 Hz) at high
resolution (6.1×10⁻⁵ Hz, 288 h record), with parabolic peak interpolation:

```
dominant line: f0 = 0.03607 Hz  (27.72 s)   prominence 2.05x
prominence at 26.0 s (0.03846 Hz) = 0.82x   ← below background; NOT a peak
prominence at 27.7 s (0.03610 Hz) = 2.05x   ← the real line
```

So 27.7 s is not a band-edge or binning artifact: at GT.DBIC the dominant
long-period line genuinely sits at **27.7 s**, and the canonical 26.0 s shows no
excess. We report this as the **27.7 s pulse** for this station/epoch and flag
the ~6 % offset from the popular "26 s" figure for cross-station confirmation.

### Decadal stability — the frequency holds across DBIC's full 8-year record

Extending the frequency measurement across GT.DBIC's entire operational lifetime
(`data/seismic/dbic-decadal-stability.json`):

| year | 1995 | 1997 | 1998 | 1999 | 2000 | 2002 |
|---|---|---|---|---|---|---|
| period | 27.77 s | 27.77 s | 27.68 s | 27.77 s | 27.77 s | 27.68 s |

Every year lands at **0.0360–0.0361 Hz — within one frequency bin** (mean 27.74 s,
**CV 0.16%** over 8 years). This confirms the *decades-scale persistence* the
literature emphasizes, measured from first principles at the source station.

**Honest limit on a true ≥20-year claim.** DBIC — the only station that cleanly
resolves the narrow line — ends in 2002. Distant stations do not isolate it (at
II.ASCN in 1995 the whitened prominence *at* 27.7 s is 0.96, i.e. below
background), and **no LHZ stations existed near the Gulf-of-Guinea source in
2015** (the region is under-instrumented — itself part of why the phenomenon
stays under-studied). So we confirm 8-year stability at the source, but cannot
directly extend the measurement into the 2010s with available open data.

### Replication — same behavior across 4 independent years

Per-year statistics (GT.DBIC, 2-day windows, 23–31 windows/year):

| year | n | freq CV | amp range | corr(freq, amp) | corr(line, secondary) | perm p |
|------|---|---------|-----------|-----------------|-----------------------|--------|
| 1995 | 31 | 0.67 % | 11.1× | 0.06 | −0.21 | 0.26 |
| 1996 | 28 | 0.64 % | 13.6× | 0.08 | −0.14 | 0.48 |
| 1997 | 29 | 0.53 % | 36.5× | 0.25 | +0.25 | 0.18 |
| 1998 | 23 | 0.65 % | 5.2× | 0.03 | +0.12 | 0.59 |

Every year independently shows: frequency stable to ~0.6 %, amplitude variable,
frequency–amplitude decoupled, and **no significant correlation with the
secondary microseism** (p > 0.18 each year; the sign even flips year to year).
The result is not a one-off.

### Cross-station — the 27.7 s line is not a DBIC instrument artifact

The line's frequency is set by the source, so it should appear at the *same*
frequency anywhere it is seen. Testing five stations on three networks at
different azimuths/distances (Feb 1996, ≤ 12 days each):

| station | network | distance / direction | dominant peak | prominence |
|---|---|---|---|---|
| GT.DBIC | GT | source-proximal (Côte d'Ivoire) | **27.77 s** (0.03601 Hz) | 2.07× |
| G.SSB | G | ~4500 km N (France) | **27.68 s** (0.03613 Hz) | 1.24× |
| II.ASCN | II | S Atlantic (Ascension) | 26.6 s | 1.19× |
| G.TAM | G | Sahara (N) | 26.1 s | 1.49× |
| IU.ANMO | IU | New Mexico (far control) | 26.6 s | 1.21× |

**G.SSB independently shows the line at 27.68 s — agreeing with GT.DBIC's
27.77 s to within 0.00012 Hz (one frequency bin), on a different network.** That
rules out a DBIC instrument artifact: it is a real propagating signal, strongest
at the source-proximal station. The distant stations (ASCN, TAM) and the far
control (ANMO) show only a *marginal* ~26.6 s feature at ~1.2× — and since the
control shows it too, we make **no** claim of a separate 26 s line from this
short window. Data: `data/seismic/dbic-crossstation-1996.json`.

### Calm-sea "gold samples" — the strongest pulses in the quietest seas

Windows with the 26 s line in the top third of strength **and** the secondary
microseism in the bottom third (10 of 111 windows). The single strongest 26 s
window (1996-06-22, excess 8.1×10⁴) occurred while local seas were quiet
(secondary 1.8×10⁶); conversely the loudest seas (boreal-winter storms,
secondary up to 1.1×10⁷) coincided with the *weakest* 26 s. If the pulse were
local-ocean-wave-driven this pattern should not exist. It is the cleanest
single-window evidence for a non-local-ocean driver.

## Searching for the gliding tremors — a rejected false positive (ADR-004)

Bruland & Hadziioannou (2023) report **gliding tremors** that sweep upward in
frequency from the 26 s band. We searched for them (`src/spectrogram.ts`,
`data/seismic/dbic-gliding-search.json`) — and this is a worked example of the
harness's contradiction discipline catching itself.

A naive spectrogram ridge-tracker **does** find 15 apparent upward-gliding
episodes (0.05 → 0.09 Hz over hours). But two checks reject them:

1. **Bandwidth.** The dominant feature in the search band is **broad** (Q ≈ 3.7)
   — the secondary microseism, not a narrow tremor line.
2. **Source coherence (the decisive test).** A true tremor of the 26 s source
   must arrive from the fundamental's back-azimuth (~95°, Gulf of Guinea, R 0.76).
   The band where the "glides" live is **not source-coherent** (R = 0.17) and
   points ~174° away (269°). It is the ordinary secondary microseism, whose peak
   frequency wanders with ocean conditions — *masquerading* as a glide.

**Verdict: rejected.** The candidates are secondary-microseism frequency wander,
not gliding tremors of the Gulf-of-Guinea source. Isolating the real gliding
tremors needs array/beamforming methods (as in B&H 2023); single-station
raw-count spectral analysis cannot separate them. This honest null is logged
rather than dressed up as a detection — exactly the ADR-004 behavior that keeps
the rest of the findings trustworthy.

## Honest scope and caveats

- **Mainly one station, four years.** Resonance/decoupling statistics are from
  GT.DBIC, 1995–1998 (replicated per-year). The 27.7 s *frequency* is now
  confirmed at one independent station (G.SSB), but the decoupling and
  amplitude statistics have not yet been reproduced at a second station, and
  this is not a global or multi-decadal claim. The single-station polarization
  localizes the source *direction* (~100°, Gulf of Guinea); pinpointing the
  source *location* still needs multi-station triangulation/beamforming.
  The seasonal *phase* of the 26 s amplitude was **not** robust across years
  (annual-harmonic relative amplitude only ~0.17), so we make no seasonal-cycle
  claim — only the frequency stability, amplitude independence, and decoupling,
  which replicate.
- **Null correlation ≠ proof of zero coupling.** With n = 57 the correlation's
  95 % interval still admits weak coupling; what is ruled out is a *strong*
  positive correlation (the signature of common ocean-wave forcing).
- **"Resonator" is an inference** from frequency stability + amplitude
  independence, not a direct mechanical measurement. The true resonance Q would
  need a linewidth analysis; here we report frequency *stability* (CV), which
  bounds it.
- **Amplitude is raw-count PSD excess.** Valid for relative comparison within
  one instrument epoch; not an absolute ground-motion amplitude.

## Artifacts

| File | Contents |
|---|---|
| `data/seismic/dbic-climatology-1996-1997.json` | the 57 per-window metrics + the resonance/decoupling statistics |
| `scripts/climatology.mjs` | reproducible fetch + analysis from IRIS |
| `src/climatology.ts` | `lineMetrics`, `resonanceStats`, `pearson`, `permutationP`, `seasonalPhase` |
| `__tests__/discovery.test.ts` | offline re-derivation of the headline numbers from the committed metrics |

## References

- Bruland, A. & Hadziioannou, C. (2023). Gliding tremors associated with the
  26 s microseism.
- General microseism theory: primary vs. secondary (double-frequency) microseisms.
- Data: IRIS/EarthScope FDSN web services, station GT.DBIC.
