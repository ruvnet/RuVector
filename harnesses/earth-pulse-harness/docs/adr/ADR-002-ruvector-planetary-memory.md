# ADR-002: ruVector as Planetary Memory with Separated Embedding Schemas

## Status

Accepted

## Context

Every detected 26-second pulse event is a small, dense bundle of
heterogeneous facts: a waveform shape, an oceanographic/environmental
context, a source geometry, and a thread of connection to the published
literature. We want to remember *all* of these events across years of
record so the harness can ask retrieval questions like:

- "Find me pulses with waveform shape similar to this one."
- "Find me events from the Gulf of Guinea region during low-swell
  windows."
- "Which events most resemble those Bruland & Hadziioannou (2023)
  describe as gliding tremors?"
- "Give me contrastive pairs: events that look spectrally alike but
  occurred under opposite ocean conditions."

ruVector provides the planetary memory/embedding layer for this. The
central design risk is **collapsing distinct kinds of similarity into a
single vector**. If we concatenate-then-embed waveform, ocean context,
source geometry, and literature into one undifferentiated vector, the
nearest-neighbor metric averages over them. Two events that are
spectrally identical but environmentally opposite would land *close*,
because the environment dimensions get diluted. That produces **false
similarity** and corrupts every downstream causality search and scoring
decision (ADR-003). Causal questions require us to hold one facet
constant while varying another — impossible if the facets are
pre-mixed.

## Decision

Store every pulse event in ruVector as a context-embedded event object,
using **separate, independently normalized embedding schemas** for the
distinct facets. Embedding logic lives in `src/embed-events.ts`; the
event/embedding contracts are defined in `src/types.ts`
(`PulseEvent`, `EventEmbedding`, `EnvironmentContext`, `FeatureVector`).

### Event object schema

Each stored event carries (drawn from `PulseEvent` +
`EnvironmentContext` + `EventEmbedding`):

```jsonc
{
  "eventId": "evt-2026-03-14T09:21:00Z-GUL",
  "timestamp": "2026-03-14T09:21:00Z",
  "dominantPeriodS": 26.1,            // target band ~25-28s, hypothesis-framed
  "amplitude": 0.0,                   // measured envelope amplitude (units per pipeline)
  "phaseCoherence": 0.0,              // [0,1], cross-station coherence
  "sourceRegion": "gulf-of-guinea",
  "swell": { "heightM": 0.0, "periodS": 0.0, "directionDeg": 0.0 },
  "tide": 0.0,                        // tidePhase in [0,1)
  "barometric": 0.0,                  // barometricGradient
  "volcanicProxy": 0.0,               // volcanicProxyScore, [0,1]
  "glideDetected": false,             // Bruland & Hadziioannou gliding-tremor flag
  "embedding": {
    "waveform":    [/* spectral shape + amplitude envelope */],
    "environment": [/* swell, tide, barometric, season */],
    "source":      [/* beam azimuth, coherence, array geometry */],
    "literature":  [/* citation-grounded text embedding */],
    "combined":    [/* concatenation of the above, L2-normalized */]
  }
}
```

### Separate embedding schemas

Four sub-embeddings, **each independently L2-normalized**, are computed
and stored alongside an optional `combined` concatenation:

1. **Waveform shape** — spectral features + amplitude-envelope shape +
   glide slope. Answers "does this *look* like that pulse?"
2. **Environment / ocean context** — swell height/period/direction,
   tide phase, barometric gradient, seasonal encoding. Answers "did
   this happen under similar ocean conditions?"
3. **Source geometry** — beamforming azimuth, cross-station phase
   coherence, array geometry descriptors. Answers "did this come from
   the same place by the same path?"
4. **Literature** — citation-grounded text embedding tying the event to
   relevant published claims (e.g., the Bruland & Hadziioannou gliding
   tremor description). Answers "what does the literature say about
   events like this?"

The `combined` vector exists only for coarse first-pass recall; **fine
queries and all causal reasoning use the individual facets.** We never
let `combined` be the sole retrieval key.

### Contrastive positive/negative pairing for causality search

To support causal questions, we mine **contrastive pairs** from memory:

- **Positive pairs**: events close in *waveform* space (high spectral
  similarity). These probe "same signal" hypotheses.
- **Negative pairs**: events close in waveform space but *far* in
  *environment* space (or vice versa). A pair that is spectrally
  identical yet environmentally opposite is exactly what isolates
  whether ocean conditions drive the pulse, because the only varying
  facet is the environment.

These pairs feed `environmentalCorrelation` and
`outOfSamplePrediction` evidence in `src/score-hypotheses.ts`
(ADR-003).

### Nearest-neighbor retrieval by region + spectral similarity

The primary retrieval path is a two-key query: filter/boost by
`sourceRegion` (e.g., `gulf-of-guinea`) and rank by **waveform**
(spectral) similarity. Region is a near-categorical prior; spectral
similarity is the fine ranker. Environment and literature facets are
applied as secondary re-rankers depending on the question. ruVector's
HNSW indexing keeps these queries fast enough to run interactively over
years of events.

### Implementation: agenticow (ruVector COW)

The planetary memory is implemented in `src/memory.ts` on top of
[`agenticow`](https://www.npmjs.com/package/agenticow) — ruvnet's
"Git for Agent Memory": Copy-On-Write vector branching over the
ruVector/`rvf` engine (HNSW, cosine). We chose it for one decisive
reason beyond fast nearest-neighbor search: **branch isolation**.

ADR-004 requires testing a hypothesis against a counterfactual ocean
state (a "storm week" vs a "calm week") *without mutating the base
record of real events*. `agenticow.branch(label)` forks the event
memory in ~0.5 ms / 162 bytes regardless of base size, gives mutation
isolation, and lets us `diff()` / `promote()` a scenario back only if it
survives the gate — the freeze-the-physics / evolve-the-investigation
pattern, expressed in storage. The `rvf` engine keys vectors by integer
ids, so `PlanetaryMemory` maps each string `eventId` to a sequential id
(shared across branches) and carries the `eventId` in the text payload.

This path is exercised on **real GT.DBIC data** in
`__tests__/real-data.test.ts` and `docs/research/real-data-proof.md`.

## Consequences

### Positive

- No false similarity: holding one facet constant while varying another
  becomes a first-class query, enabling honest causal search.
- Flexible retrieval: each scientific question selects the facet(s) it
  needs rather than fighting a pre-mixed average.
- Contrastive mining is natural, directly powering ADR-003 evidence.
- The embedding schema is itself an evolvable surface (ADR-001 #3): the
  *layout* can change, but the separation invariant cannot.

### Negative

- Higher storage and compute: four sub-embeddings plus a combined vector
  per event, and multiple HNSW indices.
- Query construction is more complex than single-vector lookup; callers
  must choose the right facet.
- Cross-facet normalization/weighting choices introduce their own tuning
  surface (delegated to ADR-003 scoring and Darwin Mode).

## Alternatives considered

1. **Single concatenated-and-embedded vector.** Rejected: produces false
   similarity and makes causal isolation impossible — the core failure
   this ADR exists to prevent.
2. **Store raw features only, embed at query time.** Rejected: too slow
   for interactive planetary-scale NN search and loses HNSW index
   reuse.
3. **Separate databases per facet with no shared event id.** Rejected:
   loses the ability to join facets for contrastive pairing; we keep one
   event object with separated sub-embeddings instead.

## References

- `src/embed-events.ts`, `src/types.ts`
  (`PulseEvent`, `EventEmbedding`, `EnvironmentContext`)
- Bruland, A. & Hadziioannou, C. (2023). Gliding tremors associated with
  the 26 s microseism.
- ruVector planetary memory / HNSW retrieval layer
- ADR-001 (evolvable surface #3), ADR-003 (scoring consumes facets)
