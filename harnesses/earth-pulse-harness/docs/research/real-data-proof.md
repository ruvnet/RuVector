# Real-Data Proof — the 26-second pulse at GT.DBIC

**Claim proven:** a persistent, narrowband seismic spectral line exists in the
long-period microseism band at the station closest to the Gulf of Guinea source,
and the harness pipeline (detector → ruVector COW memory) recovers it from
**real observations** — not synthetic fixtures, not fabricated numbers.

> Reproduce: `npm run build && npm run fetch && npm run prove`
> Offline (committed day only): `npm test` → `__tests__/real-data.test.ts`

## Data (real, citable, no fabrication)

| | |
|---|---|
| Source | IRIS/EarthScope FDSN `timeseries` web service (public, no auth) |
| Station | **GT.DBIC** — Dimbokro, Côte d'Ivoire (lat 6.67016, lon −4.85656) |
| Why this station | Borehole broadband ~on the Gulf of Guinea coast — the closest permanent station to the 26 s microseism source |
| Channel | LHZ — vertical, 1 sample/s |
| Window | Boreal winter **1995-01-02 → 1995-01-13** (12 days, the strong-signal season), a continuous GT.DBIC segment |
| Volume | 1,036,800 samples (288 h) of raw counts |
| Provenance | exact request URLs in `data/seismic/raw/PROVENANCE.json`; one full day committed at `data/seismic/GT.DBIC.LHZ.1995-01-05.window.json` |

Station metadata (verifiable):
`https://service.iris.edu/fdsnws/station/1/query?net=GT&sta=DBIC&cha=LHZ&level=channel&format=text`

## Method

1. **Median Welch PSD** (`src/spectrum.ts`): 8192-sample Hann segments, 50 %
   overlap, **252 segments**, 0.000122 Hz resolution. Each segment is linearly
   detrended (kills tides/drift). Segments are combined by **median**, which
   rejects earthquake transients that would otherwise contaminate a mean.
2. **Spectral whitening**: divide each bin by a running-median background so a
   narrow persistent line stands out as prominence > 1, independent of the steep
   microseism continuum.
3. **Line detection** (`findPersistentLine`): the most prominent local maximum
   in the long-period search band 0.033–0.045 Hz (22–30 s).

## Result

```
PERSISTENT LINE: period = 27.68 s   frequency = 0.03613 Hz   whitened prominence = 2.16x
                 (252 segments, 288 h of real GT.DBIC LHZ data)
```

This line is **robust**:

- **Reproduces across record lengths** — a single committed day (1995-01-05)
  gives 27.68 s / 0.03613 Hz at 2.29× prominence; the full 12-day record gives
  the same frequency at 2.16×. (Asserted in the test suite.)
- **Survives instrument-response removal** — recomputing on IRIS
  instrument-corrected displacement (`correct=true&units=dis`) leaves the line
  at the same 0.0361 Hz, so it is not a response artifact.
- **Median ≠ mean is informative** — the mean PSD is pulled to a spurious
  24.98 s peak by transient-contaminated segments, while the median isolates the
  true persistent 27.68 s line. This is the earthquake-rejection argument made
  concrete.

The harness per-event detector (`src/detect-26s.ts`), run on a band-passed
(0.033–0.045 Hz) real window, independently recovers it:

```
detectPulse (band-passed real data): period = 27.51 s   coherence = 0.906   confidence = 0.937
```

And the **agenticow (ruVector COW) planetary memory** (`src/memory.ts`) ingests
the real events, retrieves each event's own embedding as its nearest analog
(distance ≈ 0), and branches a counterfactual "calm-week" scenario in a
COW-isolated child — the ADR-002 / ADR-004 storage pattern, exercised on real
data.

Finally, **ruVector dynamic MinCut** (`@ruvector/mincut-wasm`, `src/partition.ts`,
ADR-006) partitions the event-similarity graph: the real GT.DBIC events resolve
to **one signal class** — the honest result that, at a single station and epoch,
the 26 s microseism is a single coherent population, not a mix of mechanisms.
The partitioner separates ≥ 2 classes on controlled inputs and flags a new class
entering a stream (regime-change detection), verified in
`__tests__/partition.test.ts`.

## Honest interpretation (what this does and does NOT show)

- **The raw spectrum is dominated by the secondary microseism** (~6–16 s); the
  long-period 26–28 s line is a *subtle but unambiguous* feature that only
  emerges with median averaging + whitening. We do not claim it is the loudest
  thing in the record — it is not.
- **The measured period here is ~27.7 s**, not exactly 26.0 s. The "26-second"
  name is a round-number label; reported values in the literature span roughly
  26–28 s and vary with station, season, and processing. 27.68 s at GT.DBIC for
  this epoch sits squarely in that long-period Gulf-of-Guinea microseism family.
  We report what the data shows rather than forcing the canonical number.
- **This is one station, one epoch.** It proves the signal is real, persistent,
  and recoverable by the harness. It does **not** by itself localize the source,
  identify the mechanism, or rank the hypotheses — those are Levels 2–6 of the
  discovery ladder (`benchmark-design.md`), which need multi-station beamforming
  and ocean-state coupling, framed as targets, not yet measured.

## Artifacts

| File | Contents |
|---|---|
| `data/seismic/GT.DBIC.LHZ.1995-01-05.window.json` | one real day (86,400 counts) — the offline test fixture |
| `data/seismic/dbic-1995-median-psd.json` | the 12-day median + whitened PSD band and the detected line |
| `data/seismic/proof-summary.json` | machine-readable summary of every number above |
| `data/seismic/raw/PROVENANCE.json` | exact IRIS request URLs (regenerated by `npm run fetch`) |

## References

- Bruland, L. & Hadziioannou, C. (2023) — gliding tremors associated with the
  26 s microseism, Gulf of Guinea source (the anomaly motivating Levels 5–6).
- IRIS/EarthScope FDSN web services — `service.iris.edu` (data provider).
- General microseism theory: primary microseisms (ocean swell over sloping
  bathymetry) vs. secondary microseisms (nonlinear wave–wave interaction); see
  `26-second-pulse-literature.md`.
