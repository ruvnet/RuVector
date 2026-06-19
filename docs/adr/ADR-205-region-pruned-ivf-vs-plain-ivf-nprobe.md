---
adr: 205
title: "Triangle-Inequality Cluster Pruning vs Tuned Plain IVF nprobe — Structural NO-GO"
status: proposed
date: 2026-06-05
authors: [ofershaal, claude-flow]
related: [ADR-193, ADR-199, ADR-201]
tags: [ruvector, retrieval, ann, ivf, rairs, pruning, branch-and-bound, no-go]
---

# ADR-205 — Triangle-Inequality Cluster Pruning vs Tuned Plain IVF `nprobe` (Structural NO-GO)

## Status

**Proposed — NO-GO (robust, structural), 2026-06-05.** Closes the BET 4 caveat left open by
ADR-201: the region-pruning IVF kernel (`RegionPruneIvf`) was built and validated *exact* there but
only ever run as BET 2's mechanism **against ACORN** — never head-to-head against its natural
incumbent, **plain IVF `nprobe`**, on unfiltered ANN. This is that head-to-head. The gate was
**pre-registered and frozen before any run** (`docs/plans/bet4-ivf-pruning/PRE-REGISTRATION.md`).

**Lower-bound branch-and-bound IVF probing provides essentially zero benefit over a tuned plain
`nprobe` — a flat 1.00× member-eval ratio in every cell, at both n=20k and n=50k, in both 128-d and
a PCA-8 low-dim control.** The cause is **structural, not dimensional**: the triangle-inequality
cluster bound can only prune *far* clusters, which a tuned `nprobe` already never visits — so the
bound is **redundant** with `nprobe`'s centroid-distance cutoff. High dimensionality only makes the
faithful BET-2 kernel (which probes in *LB order*) strictly **worse** (0.18–0.25×).

## Context

`ruvector-rairs::IvfFlat` (ADR-193) is plain IVF: k-means centroids + inverted lists;
`search(q, k, nprobe)` scans all members of the `nprobe` nearest-centroid lists. BET 4 asked whether
adding a triangle-inequality lower bound — `LB(q,c) = max(0, ‖q−μ_c‖ − r_c)`, `r_c` the cluster
radius — and probing with branch-and-bound (skip/stop on clusters that provably cannot hold a
top-k point) beats tuned `nprobe` at matched recall@10, on real 128-d arxiv embeddings.

The kernel was rebuilt self-contained (`crates/ruvector-bet4-ivf-bench`), off clean `main`, over the
same `ruvector-rairs` k-means substrate as the incumbent (BET 2's kernel lives only on the #536
branch). Two correctness gates passed before any claim: full-budget B&B is **exact** (recall ≥ 0.999
vs brute force), and the instrumented incumbent **matches `IvfFlat`** within 0.01 recall at matched
params (so its measured cost is the real incumbent's).

Three contenders share one index per `nclusters` (only the probe loop differs):
- **plain `nprobe`** — the incumbent.
- **B&B LB-order** — the faithful BET-2 `RegionPruneIvf`: probe in ascending `LB`, global `break`
  when `LB ≥ τ` (exact at full budget).
- **B&B steelman** — centroid-distance order (the effective `nprobe` ordering, so τ tightens fast)
  + per-cluster **LB-skip** (correctness-safe in any order). The *strongest* cluster-level B&B: if
  it cannot beat `nprobe`, the bound does not pay.

## Decision / Finding

**NO-GO.** Cost at matched recall@10 = 0.95, 200 queries; member distance-evals per query
(steelman is the strongest contender, so it sets the verdict):

**n = 50,000, 128-d (real arxiv features):**

| nclusters | exact-prune | plain `nprobe` | B&B LB-order | **B&B steelman** | steelman ratio |
|---|---|---|---|---|---|
| 64   | 0.0%  | 11,102 ev | 49,182 (recall 0.99) | **11,102** | **1.00×** |
| 256  | 4.7%  | 7,890 ev  | 49,979 (recall 1.00) | **7,890**  | **1.00×** |
| 1024 | 13.1% | 5,682 ev  | 45,373 (recall 1.00) | **5,682**  | **1.00×** |

**n = 50,000, PCA-8 (low-dim control — bound is tight here):**

| nclusters | exact-prune | plain `nprobe` | **B&B steelman** | steelman ratio |
|---|---|---|---|---|
| 64   | 8.0%  | 4,393 ev | **4,393** | **1.00×** |
| 256  | 45.1% | 1,835 ev | **1,835** | **1.00×** |
| 1024 | 82.5% | 731 ev   | **731**   | **1.00×** |

n=20k reproduces identically (steelman 1.00× in all six cells). Wall-clock tracks the eval ratio
(0.94–1.02×) — no reversal, but no win either.

**Mechanism (structural, the key result).** The true top-k neighbours live in the *nearest*
clusters; any method must scan those members to find them. The LB bound only lets B&B *skip far
clusters* — but a tuned `nprobe` already does not visit them. So at matched recall the steelman
scans **exactly** the members `nprobe` scans (the near clusters all have `LB < τ`, so nothing is
skipped inside the operating budget) → 1.00×, **in every dimension**. The win is not "hard"; it is
**structurally impossible** against a tuned incumbent, because the bound and `nprobe`'s
centroid-distance cutoff exploit the *same* locality.

**Why the LB-order kernel is strictly worse (0.18–0.25×).** Ordering clusters by `LB = max(0, d −
r_c)` pushes any *large-radius* cluster toward `LB ≈ 0` regardless of how far its centroid is, so
B&B probes far, low-yield clusters early and needs ~all clusters to reach 0.95. LB-order is correct
for *exact* early termination but a poor *priority* for approximate probing — centroid distance is
better. High-dimensional concentration (large radii) makes this pathology severe.

## The pre-registered low-dim control — an honest deviation

The frozen pre-registration expected the **PCA-8 control to show B&B *winning*** ("tight bound ⇒
B&B beats tuned `nprobe`; if it does not win even at 8-d, the implementation is suspect"). **It did
not** — the steelman is 1.00× at PCA-8 too. That expectation was built on a **false premise**: a
tight bound implies beating *full exact scan*, **not** beating *tuned `nprobe`*. The control still
did its real job two ways, so the 128-d NO-GO is **interpretable, not voided**:

1. **The kernel is sound.** The exact-regime pruning fraction scales correctly and strongly with
   dimension — 0–13% at 128-d vs 8–82.5% at PCA-8 (n=50k). The bound *does* prune hard when it can;
   the harness measures it correctly. The implementation is not suspect.
2. **It replaced the predicted mechanism with a better one.** The control is what revealed the kill
   is *structural redundancy* (dimension-independent), not *dimensional looseness*. The bound prunes
   87% of clusters vs full-scan at PCA-8 yet still ties `nprobe`, because `nprobe`'s tuning already
   captures that same pruning.

Recording the deviation — the control disproved my predicted sign and taught the real finding — is
the point, per the prove-not-hype protocol (cf. ADR-203's three documented deviations).

## Consequences

**Positive (a clean, general kill).**
- **Companion to ADR-199.** Classical exact-pruning structures do not pay on embedding retrieval:
  graph separators/contraction there (high treewidth), triangle-inequality cluster bounds here
  (redundant with `nprobe`). The kills keep sharpening *where* these ideas work — and IVF `nprobe`
  is simply already near-optimal at exploiting cluster locality.
- **No code to ship, and that is the right outcome.** `ruvector-rairs::IvfFlat` needs no B&B add-on;
  the result protects it from a complexity-adding non-improvement.

**Boundaries / honest caveats.**
- **Scope: cluster-level bounds vs tuned `nprobe`, recall@10 ≈ 0.95.** This does **not** speak to
  finer techniques — IVFADC / product-quantized asymmetric distance, per-member bounds, or learned
  routing — which prune *within* lists by a different mechanism and are outside the frozen claim.
- **The structural argument predicts the same sign at other recall targets** (neighbours still live
  in the near clusters at R=0.99), but only R=0.95 was measured.
- **`nprobe` is the right incumbent precisely because it is already tuned.** Against an *untuned*
  full-exact-scan baseline the bound wins (that is the exact-prune fraction) — but that baseline is
  not what anyone ships.

## Scoreboard

**2 WINS** (ADR-200/202 reuse+periodic; ADR-204 incremental high-recall tier) /
**4 KILLS** (ADR-199 CCH-on-embeddings; ADR-201 filtered-ANN vs ACORN; ADR-203 KG-treewidth;
ADR-205 IVF cluster-pruning vs `nprobe`).

## Next steps

1. If IVF acceleration is ever revisited, the open lever is **within-list** pruning
   (PQ/IVFADC asymmetric distance), a different mechanism than the cluster-level bound killed here.
2. None for this kernel — the structural redundancy is dimension-independent and reproduced at two
   scales; further `n`/recall sweeps would only reconfirm.

## Alternatives considered

- **B&B in LB order** (the faithful BET-2 kernel) — measured; strictly worse than `nprobe`
  (0.18–0.25×) because LB is a poor approximate priority.
- **B&B steelman** (centroid order + LB-skip) — the strongest cluster-level variant; ties `nprobe`
  (1.00×). Retained as the verdict-setting contender.
- **Within-list / PQ pruning** — not built; a different mechanism, noted as the only open lever.
