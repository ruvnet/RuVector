# Papers Corpus — Literature Embeddings

This directory holds the **literature corpus** for the Earth Pulse Observatory.
It is **initially empty** by design.

## Rules

- **One document = one cited source.** Each file corresponds to exactly one
  paper, report, dataset description, or other primary source.
- **Every promoted claim must map to a file here.** The discovery score's
  `provenance` term (see `../../docs/research/benchmark-design.md` §7) is the
  fraction of a hypothesis's claims that map to a source file in this directory.
  A claim with no backing file cannot pass the promotion gate.
- **Embeddings live alongside sources.** This corpus is intended to be embedded
  (ruVector / vector index) so the harness can retrieve the source(s) relevant
  to any claim or feature.

## Expected file layout (per source)

For each source, store:
- the source text or extracted text (e.g. `<slug>.txt` or `<slug>.pdf`), and
- a metadata sidecar (`<slug>.json`) with: `title`, `authors`, `year`, `venue`,
  `doi_or_url`, `verified` (bool), `claims` (list of claim IDs this source
  supports).

## Provenance discipline

- Do **not** invent citations. If a source's bibliographic details are
  uncertain, mark `verified: false` and do not let claims depending on it be
  promoted past Level 2 of the discovery ladder.
- The anchor source to add first is **Bruland & Hadziioannou (2023)** (gliding
  tremors of the 26 s microseism). See `../../docs/research/26-second-pulse-literature.md`.
