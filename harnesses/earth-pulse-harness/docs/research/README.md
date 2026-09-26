# Earth Pulse Observatory — Research Docs

Research documentation for a self-contained harness investigating Earth's
stable ~26-second microseism pulse from the Gulf of Guinea (Bight of Bonny, near
São Tomé) and its associated gliding tremors (Bruland & Hadziioannou 2023).

**Core principle:** treat the pulse as a **causal-discovery benchmark**, not a
mystery story. Move from "Earth has a heartbeat" to "this mechanism predicts the
pulse better than all alternatives." **Freeze the physics; evolve the harness.**

## Index

| Doc | Purpose |
|-----|---------|
| [`26-second-pulse-literature.md`](./26-second-pulse-literature.md) | Literature review: discovery history (Oliver, 1960s), Gulf of Guinea localization, competing mechanisms (ocean-wave vs. volcanic), microseism theory (primary vs. secondary), and the Bruland & Hadziioannou (2023) gliding-tremor anchor. Separates established facts from open questions. |
| [`benchmark-design.md`](./benchmark-design.md) | The Kaggle-style bounded benchmark: first discovery target (predict pulse amplitude from ocean state), baselines, ruVector candidate, acceptance test (beat seasonal baseline by ≥10%), data spine, discovery ladder (Levels 1–6), and the discovery-score / promotion gate. Honest scoring before autonomy. |
| [`hypothesis-catalog.md`](./hypothesis-catalog.md) | Ranked catalog of candidate mechanisms with expected evidence, killer contradictions, distinguishing ruVector tests, and subjective prior scores. Leading bet: ocean shelf resonance for the carrier; coupled ocean+geology is the one to watch for the glides. |

## Related

- Data spine and provenance: `../../data/` (each subdirectory has a README;
  `data/papers/` holds the literature corpus that every promoted claim must map
  to).

## Honesty contract

- Established facts are kept separate from hypotheses.
- No invented measured results and no fabricated citations.
- Every promoted claim maps to a source in `data/papers/` and to the
  observations that support it.
- The harness never fabricates observations; empty data directories are
  acceptable, invented data is not.
