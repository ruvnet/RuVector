# SepRAG — Future Directions & Research Backlog

After the CCH-contraction NO-GO on embedding graphs ([ADR-199 Empirical
Outcome](../../adr/ADR-199-public-corpus-benchmark-harness.md#empirical-outcome-2026-06-04)),
two ideas survived every test and several promising directions remain. This file keeps
them alive so they're explored deliberately, each under the same discipline.

## The "prove not hype" protocol (mandatory for every bet)

A result only counts if it satisfies **all five**:

1. **One claim, one number.** e.g. "N× cheaper at equal recall@10," not "faster."
2. **Beat the strongest in-repo incumbent, tuned** — HNSW / `ruvector-diskann` (Vamana) /
   `ruvector-acorn` (filtered ANN) — never a strawman.
3. **Public data + ground truth** (ogbn-arxiv in hand; BEIR / filtered-ANN sets available).
4. **Pre-register the win AND kill condition** before running. A loss is an acceptable,
   reportable outcome.
5. **Adversarial check.** Explicitly ask "would the baseline win if tuned harder?" and
   include that variant.

## Backlog (ranked by upside × provability)

### BET 1 — Customizable re-weight vs rebuild  ✅ WIN (diag + rot + non-linear), see [ADR-200]
Salvages ADR-198 (the customizable metric), decoupled from CCH. Result: a **fixed ANN
topology + recomputed distances** matches full Vamana **rebuild** on **both recall (±0.2%)
and per-query cost (±1%)** up to **36% relevant-set churn**, across diagonal, dense-
Mahalanobis (rotational), AND non-linear (tanh-warp) drift — at **zero** rebuild cost.
Stale-index control loses up to 29 points (benchmark has teeth). Full evidence + boundaries
in [ADR-200](../../adr/ADR-200-customizable-reweighting-fixed-topology-ann.md).
Harness: `crates/ruvector-seprag/examples/reweight_vs_rebuild.rs`.
- **Scale ✅** (`examples/scale_drift.rs`): recall parity within 2% from 5k→100k at
  **~1,000–4,000× lower update cost** (rebuild 142s vs reuse 0.035s at 100k). Honest caveat:
  gap widens with N (−0.2%→−1.7%) → *defer/batch rebuilds*, not *never*.
- **Region-local drift ✅** (`examples/region_drift.rs`): warp only a 15% cluster, grade
  in-region vs out-region separately. Reuse held *inside* the drifted region (A_in within
  0.7% of B_in, gate PASS) even at 53% in-region churn. Surfaced a transient rebuild dip
  (B_in 81% at t=0.25) = lite-Vamana build variance → motivates the diskann port.
- **Production-index port ✅** (`examples/diskann_drift.rs`): confirmed on the shipping
  `ruvector-diskann` Vamana (n=20k, recall 96–99%, reuse within 2% global + in-region). The
  t=0.25 reuse-beats-rebuild dip reproduced → it's a real property, not lite-Vamana noise;
  baseline-variance caveat resolved.
- **Hybrid policy ✅** (`examples/hybrid_policy.rs`): under aggressive compounding random-walk
  drift, `never` decays to 94.4% mean / 89.7% floor; **periodic-4 recovers 98.8% (≈ always
  99.1%) at 25% of the rebuild cost** (periodic-8: 98.4% at 12.5%). The drift-*triggered*
  monitor (Frobenius) underperformed simple periodic → periodic-K is the recommended knob.
- **Open (ranked):** (1) smarter rebuild trigger (sampled-recall probe vs the Frobenius
  monitor); (2) wire re-weight + periodic-rebuild into the `ruvector-diskann`/`ruvector-gnn`
  loop behind a flag (production payoff); (3) diskann at n≥10⁵ + ≥500 queries; (4)
  incremental-rebuild baseline.

### BET 2 — Filtered ANN vs `ruvector-acorn`
Region/predicate pruning for constrained ("nearest among items matching X") retrieval — a
real flat-ANN weakness. Higher effort; ACORN is a strong specialized incumbent in-repo, so
it's a harder, longer fight. Needs a filtered-ANN benchmark with selectivity sweeps.

### BET 3 — Multi-hop Graph-RAG on a sparse curated KG  ❌ NO-GO (treewidth gate), see [ADR-203]
ADR-196's true scope: structural + semantic retrieval on a Wikidata-style KG, on the bet
that bounded relation degree might be more road-like than the dense citation graph that
failed. **Probed first (treewidth go/no-go gate) before any QA harness — and it failed.**
All three curated KGs are high-treewidth: WN18RR (WordNet) blowup 59.9×, FB15k-237
(Freebase) elim_h≈0.46·n, CoDEx-L (Wikidata) exponent p=1.83 (tree-like periphery, but the
hub-dense core collapses treewidth). Minor-min-width **lower** bounds (28–44) are 7–11× the
road control's (4) → structurally certain, not a heuristic artifact. Cause: KGs are
small-world **with hubs** (max degree 520/1999/4999 vs road's 9); hubs kill separators
regardless of average degree. Combined with ADR-199 this **closes the CCH-contraction
line** — no retrieval backbone of interest has road-like treewidth. The QA benchmark was
(correctly) never built. Query kernel stays exact (recall 30/30) + treewidth-independent;
its only viable home remains a treewidth-immune IVF hierarchy (BET 4). Probe:
`crates/ruvector-seprag/examples/kg_treewidth_probe.rs`; ADR-203.

### BET 4 — Region pruning on an IVF/clustering hierarchy
Structural pivot: move the validated separator-tree **pruning query** off separators (which
need small treewidth) and onto a **clustering/IVF hierarchy** (`rairs-ivf`, ADR-193), which
is treewidth-immune. Most novel; define the baseline (plain IVF probe) before building.

## Salvaged, validated assets (reusable regardless of bet)

- `ruvector-seprag` — correct, tested CCH nested-dissection + separator-tree k-NN reference.
- The separator-tree **branch-and-bound pruning query** — exact recall, ~100% search-space
  reduction, *treewidth-independent*. The reusable kernel.
- Road-control + manifold + citation harnesses — a treewidth probe for any new backbone.

## Dead (do not revisit without new evidence)

- CCH **full contraction on embedding / dense-similarity graphs** — high treewidth,
  confirmed across citation + Euclidean feature backbones. HNSW already owns embedding kNN.
