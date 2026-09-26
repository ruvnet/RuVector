# Earth Pulse Observatory — `earth-pulse-harness`

A **MetaHarness Darwin-Mode** research pod for Earth's stable **~26-second microseism**
(the "26-second pulse") originating from the Gulf of Guinea, and the associated **gliding
tremors** documented by Bruland & Hadziioannou (2023).

> **The core idea: freeze the physics, evolve the harness.** Darwin Mode does **not**
> evolve the scientific truth. It evolves the *investigation workflow* around it — feature
> extraction, source-localization checks, ruVector embedding schemas, hypothesis scoring,
> anomaly tests, report generation, and failure detection — and keeps only the changes
> that *measurably* beat a baseline through a strict promotion gate.

This is a self-contained metaharness bundle, built in the same style as the sibling
`harnesses/timesfm-harness`.

## Why this is tractable

We do **not** try to "solve Earth's heartbeat." We turn it into a **bounded causal-discovery
benchmark**: move from *"Earth has a heartbeat"* to *"this mechanism predicts the pulse better
than all alternatives."* The first target is deliberately narrow:

> **Can the system predict the amplitude of the 26-second pulse from ocean-state variables,
> beating a seasonal baseline by ≥ 10 % on held-out months?**

A positive result is real but modest: *the pulse is predictably coupled to measurable planetary
state variables.* See `docs/research/benchmark-design.md`.

## Quick start

```bash
npm install
npm run doctor          # kernel + host adapter health check
npm test                # 17 tests: offline science pipeline + install smoke test
npm run build           # compile src/ -> dist/
npm run pipeline        # run detect->extract->embed->score over the bundled fixtures
```

## Proven on real data

This is not a toy. Run against **real seismic observations** from IRIS/EarthScope
(station **GT.DBIC**, Côte d'Ivoire — on the Gulf of Guinea coast), the harness
finds the long-period microseism as a persistent narrowband spectral line:

```
PERSISTENT LINE: period = 27.68 s   frequency = 0.0361 Hz   whitened prominence = 2.16x
                 (median Welch PSD over 252 segments / 288 h of real GT.DBIC LHZ data)
detectPulse (band-passed real data): period = 27.51 s   coherence = 0.906   confidence = 0.937
```

The line reproduces across record lengths and survives instrument-response
removal — it is a real geophysical signal, not an artifact. Full method, honest
caveats, and references: **`docs/research/real-data-proof.md`**.

### Discovery: a frequency-stable resonance, decoupled from the ocean-wave field

Pushing further (GT.DBIC, **1995–1998**, 111 windows), the harness finds the
pulse behaves like a **fixed-frequency resonator driven at variable strength**:

```
frequency stable to CV ~0.6 %  while  amplitude varies up to 36x
corr(frequency, amplitude)                 weak    → frequency doesn't shift with drive
corr(pulse amplitude, secondary microseism) ~0, p>0.18 every year → NOT ocean-wave-driven
```

Replicated independently in all 4 years; the strongest pulses occur in the
*quietest* local seas ("gold samples"). A precise refit puts the dominant line
at **27.72 s** (0.03607 Hz) — the canonical 26.0 s shows no excess here.

Digging deeper, three-component **polarization** localizes the source from a
single station: the 27.7 s wave is a **retrograde Rayleigh wave with
back-azimuth ~100°** (R = 0.76 over 96 windows) — pointing **straight into the
Gulf of Guinea / Bight of Bonny** (expected 109–118°). That bearing is **fixed
to ~96° across winter/summer/autumn and two years** (a stable source). Tight
triangulation to a point isn't yet possible — only the source-proximal DBIC
cleanly polarizes the 27.7 s line (distant stations see other ~26 s sources), an
honest limit that itself shows the line is regionally concentrated. A fixed resonant
frequency, excited independently of local ocean-wave energy, radiating from the
known source region: an independent, quantitative argument that the pulse is
*not* an ordinary microseism. Method, statistics, replication, localization, and
honest scope: **`docs/research/discovery-resonator-decoupling.md`**
(`npm run climatology`, `npm run localize`).

```bash
npm run build && npm run fetch && npm run prove   # fetch real IRIS data + prove
npm test                                          # offline proof from one committed real day
```

Memory is the **agenticow** library (ruvnet's Copy-On-Write vector branching over
ruVector/`rvf`) — see `src/memory.ts` and ADR-002.

## The pipeline (`src/`)

| File | Role |
|---|---|
| `spectrum.ts` | FFT, median Welch PSD, spectral whitening, band-pass — the engine that isolates the 26 s line in real data. |
| `polarization.ts` | 3-component Rayleigh-wave back-azimuth — single-station source localization (the 27.7 s wave points at the Gulf of Guinea). |
| `climatology.ts` | line metrics, resonance/decoupling statistics, permutation tests. |
| `memory.ts` | ruVector planetary memory backed by **agenticow** (COW vector branching): ingest, nearest-analog, branch scenarios. |
| `partition.ts` | Signal-class partition of the event graph via **ruVector dynamic MinCut** (`@ruvector/mincut-wasm`); detects when a new mechanism enters the record. |
| `detect-26s.ts` | DFT scan of the 24–28 s band → `PulseEvent` (period, amplitude, coherence, glide, confidence). |
| `extract-features.ts` | Spectral sub-band shape, amplitude envelope, glide slope, station geometry, environment context. |
| `embed-events.ts` | **Separate** L2-normalized waveform / environment / source embeddings (+combined), cosine NN search. |
| `score-hypotheses.ts` | Weighted discovery score + ranking of the candidate mechanisms, with killer contradictions. |
| `validate.ts` | The promotion gate: F1, false-positive, held-out error, citation grounding, leakage. |
| `pipeline.ts` | Wires the five stages together. |

All pipeline code is **deterministic and offline** — no network, no fabricated observations.

## Darwin Mode (`/evolve`)

```bash
npm run evolve          # real sandbox, deterministic mutator (no API key, no network)
npm run evolve:dry      # mock sandbox, fully offline dry run
# or pass flags straight through:
npx metaharness-darwin evolve . --generations 20 --children 8 --concurrency 4 --seed 26
```

**Evolvable surfaces** (`.metaharness/safety-policy.json`): detector band, feature schema,
embedding schema, retrieval strategy, scoring weights, validator/holdout strategy.

**Forbidden mutations:** fabricating observations, inventing citations, leaking test windows
into training, promoting a hypothesis without beating a baseline, or adding any new
import / network / filesystem / shell / env access.

**Promotion gate** (`.metaharness/objective.json`):

```
pulse_detection_f1 improves by >= 3%
AND false_positive_rate does not increase
AND held_out_prediction_error improves by >= 5%
AND every cited claim maps to a source document
AND no leakage from test windows into training windows
```

## Discovery score

```
score = 0.25 * source_stability
      + 0.20 * environmental_correlation
      + 0.20 * out_of_sample_prediction
      + 0.15 * contradiction_survival
      + 0.10 * mechanistic_plausibility
      + 0.10 * citation_grounding
```

## Candidate mechanisms (priors, not results)

| Hypothesis | Prior | Killer contradiction |
|---|---|---|
| Ocean shelf resonance | 0.72 | Strong pulses during calm-ocean windows |
| Coupled ocean + geology | 0.68 | Either factor alone explains everything |
| Water-column / bathymetric mode | 0.55 | Same geometry elsewhere lacks the signal |
| Volcanic / hydrothermal tremor | 0.46 | No thermal/gas/seismic volcanic proxy |
| Instrument artifact | 0.12 | Appears across independent global stations |

The bet to watch is **coupled ocean + geology**: ocean shelf resonance likely explains the
*carrier* frequency, while the gliding tremors may require a second mechanism.
See `docs/research/hypothesis-catalog.md`.

## Data spine (`data/`)

`seismic/`, `ocean/`, `tides/`, `bathymetry/`, `papers/` — real observations and the literature
corpus live here (each has a README describing the expected format). The harness **never
fabricates** observations, and `data/` is write-denied to agents by default.

## Documentation

- **ADRs** — `docs/adr/ADR-001…005` (see `docs/adr/README.md`).
- **Research** — `docs/research/` (literature review, benchmark design, hypothesis catalog).
- **Provenance** — `PROVENANCE.md`.

## License

MIT.
