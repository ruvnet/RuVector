# BET 2 ⊗ BET 4 — Region-Pruned Filtered ANN vs tuned ACORN

**Status:** Pre-registered (gate frozen before any run) · **Date:** 2026-06-04 ·
**Research line:** SepRAG (ruvnet/RuVector issue #534) · **Self-contained:** depends only on
crates already on `main` (`ruvector-acorn`, `ruvector-rairs`) — **independent of PR #535.** ·
**Builds on (by reference, not by compile):** ADR-200 (BET 1 WIN), ADR-193 (`ruvector-rairs`
IVF), ADR-199 (CCH NO-GO → why IVF, not separators) ·
**Outcome ADR:** ADR-201 (written from the result — WIN *or* NO-GO).

> This document is the **pre-registration**, committed before the harness runs. A loss is an
> acceptable, reportable outcome (cf. ADR-199). Editing the gate after seeing results voids
> the bet. Plumbing (M0) may be built before freeze; contender runs (M1+) may not.

## Prove-not-hype protocol (mandatory — all five)

1. **One claim, one number.** 2. **Beat the strongest in-repo incumbent, tuned.**
3. **Public data + ground truth.** 4. **Pre-register WIN *and* KILL.** 5. **Adversarial check.**

## Thesis (one claim, one number)

> For predicates whose membership **correlates with embedding-cluster structure** (ρ ≥ 0.7) at
> **selectivity ≤ 1%**, IVF **region-pruned** filtered search reaches **filtered-recall@10
> within 2%** of tuned ACORN at **≥ 5× fewer distance-evaluations per query** — and the cost
> advantage **grows monotonically as selectivity falls** (the mechanism signature).

Primary cost = **distance-evals/query** (hardware-independent, as ADR-200). Wall-clock is
reported and acts as an honesty guard (below).

## Why this scope is the honest one (central insight)

ACORN (SIGMOD 2024, arXiv:2403.04871; `ruvector-acorn::AcornIndexGamma`) is
**predicate-agnostic by design**: a denser γ·M graph + expand-all-neighbors traversal stay
navigable *through* predicate-failing nodes, computing a distance for every expanded node,
pass or fail. So ACORN's per-query distance count is **flat-to-rising as selectivity drops** —
and ACORN **owns** the uncorrelated case. Attacking it there is a guaranteed loss.

Region-pruning wins the opposite case: when the predicate correlates with cluster membership,
whole clusters with zero matches are skipped, and a cheap O(1) predicate test gates the
expensive 128-d distance — so A pays distance-evals only for `routing (≈√n centroids) +
actual matches in probed clusters`, which **shrinks as selectivity drops**. That asymmetry is
the entire bet, and it is the production-RAG metadata-filter case (`tenant_id`, `doc_type`,
`language`, `year≥Y`, `category=X`).

On embeddings the pruning kernel **cannot** live on graph separators (ADR-199: embedding
graphs are high-treewidth → CCH contraction blew up). Its only viable, treewidth-immune
substrate is the **IVF hierarchy** (`ruvector-rairs`) — i.e. BET 4. **BET 2 (benchmark +
incumbent) and BET 4 (mechanism) are one experiment.**

## Data & predicates (real, public — ogbn-arxiv)

n ≈ 169,343, 128-d features (`target/m1-data/arxiv/raw/`, in hand). Oracle =
`ruvector-acorn::exact_filtered_knn`.

| Predicate | Correlation ρ | Source |
|---|---|---|
| Subject-area label = c (one of 40) | **high** | `node-label.csv.gz` |
| Year ≥ Y / year ∈ [a,b] | **medium** | `node_year.csv.gz` |
| Random Bernoulli(p), equal selectivity | **ρ = 0 (kill control)** | synthetic |

**Correlation knob ρ:** interpolate a real label predicate toward a random one of equal
selectivity by shuffling a fraction `1−ρ` of membership. Sweep ρ ∈ {0, 0.3, 0.5, 0.7, 1.0}.
**Selectivity sweep:** {0.1, 0.5, 1, 5, 10, 30}% (sub-10% is where post-filter collapses).

## Contenders

| ID | Index | Role |
|---|---|---|
| **A** | IVF region-pruned filtered search (`rairs::IvfFlat` + per-cluster match-count pruning, predicate-gated distance) | **the bet** |
| **B** | `AcornIndexGamma`, tuned (γ∈{2,3}, ef∈{64,128,200}; best cost at equal recall) | strong incumbent |
| **D** | ACORN + predicate-aware entry points | adversarial "tune harder" (rule #5) |
| **C** | flat / post-filter | floor — proves benchmark teeth (recall collapse at low sel) |

All scored against `exact_filtered_knn` ground truth.

## Pre-registered gate

- **WIN** — at **ρ ≥ 0.7**: A within **2%** filtered-recall@10 of best{B, D} **and** the
  distance-eval ratio is **≥ 5× at sel ≤ 1%** and **≥ 2× at sel = 5%**, **monotonically
  increasing as selectivity falls** (the mechanism must be visible, not a single lucky cell).
- **Graceful-degradation guard** — in ACORN's regime (sel ≥ 10% **or** ρ ≤ 0.3) A may lose,
  but by **≤ 1.5×** in distance-evals (no catastrophic blowup). Cost-axis analogue of the
  recall-collapse control.
- **Wall-clock honesty guard** — wall-clock reported alongside; a distance-eval win that
  **reverses on wall-clock → "inconclusive," not WIN** (IVF cluster scans vs ACORN's graph
  walk have different cache behavior; the win must survive both).
- **KILL (reportable NO-GO)** — *either* A's recall **collapses** at the ρ=0 control (must
  degrade *safely* to ≈ the floor, not catastrophically), *or* no (selectivity, ρ) cell meets
  the WIN bar.
- **Reported regardless:** the crossover correlation **ρ\*** (and crossover selectivity) where
  ACORN overtakes A on cost — the iso-cost frontier is itself a publishable result.

**Named live risk (not a formality):** ACORN on correlated subgraphs may already be cheap
enough that ≥5× is unreachable → that is a clean, reportable KILL, written up like ADR-199.

## Where it lives (self-contained off main)

New crate **`crates/ruvector-filtered-bench`**, depending only on `ruvector-acorn` +
`ruvector-rairs` (+ `rand`). Contender A and the predicate / ρ-knob / selectivity generators
live in `src/`; the harness is `examples/filtered_ann_pruning.rs`. No dependency on
`ruvector-seprag` (PR #535) — this PR stands alone.

## Milestones

- **M0 — substrate + oracle wiring.** Load arxiv feat+label+year; build `IvfFlat`; confirm
  `exact_filtered_knn` + `recall_at_k` on a slice (use a selectivity floor so #matches ≥ k=10).
  Predicate + ρ-knob + selectivity generators. *Gate: oracle matches brute-force exactly.*
- **M1 — contenders B/C/D.** Tuned ACORN sweep + post-filter floor; reproduce the documented
  low-selectivity post-filter recall collapse (proves teeth).
- **M2 — contender A.** Cluster probe order (match-count, then centroid distance); zero-match
  cluster skip; predicate-gated distance; per-query distance-eval + wall-clock counters.
- **M3 — full sweep + gate eval.** selectivity × ρ grid; emit WIN/KILL table; find ρ\*;
  apply the wall-clock honesty guard.
- **M4 — ADR-201.** Write the outcome (WIN or NO-GO) with ADR-199/200 honesty.

## Out of scope (named, not silently assumed)

- The uncorrelated/agnostic regime as a *target* (kill control only — ACORN owns it).
- Multi-predicate conjunctions, streaming updates, the live-GNN metric (BET 1's frontier).
- Disk-resident / billion-scale (in-memory ogbn-arxiv is the stage).
