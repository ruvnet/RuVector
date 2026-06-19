---
adr: 206
title: "PQ/IVFADC Within-List Pruning vs Tuned Plain IVF nprobe — Scale-Gated WIN"
status: proposed
date: 2026-06-05
authors: [ofershaal, claude-flow]
related: [ADR-193, ADR-199, ADR-201, ADR-205]
tags: [ruvector, retrieval, ann, ivf, rairs, pq, ivfadc, product-quantization, win]
---

# ADR-206 — PQ/IVFADC Within-List Pruning vs Tuned Plain IVF `nprobe` (Scale-Gated WIN)

## Status

**Proposed — WIN (scale-gated), 2026-06-05.** Opens the one lever ADR-205 left explicitly open:
ADR-205 killed *cluster-level* triangle-inequality pruning vs tuned `nprobe` (the bound was
**redundant** with `nprobe`'s centroid cutoff — same axis, 1.00× in every cell). Its "Next steps #1"
named a **different** mechanism — within-list pruning via **product-quantized / IVFADC asymmetric
distance** — as the only open lever. This is that head-to-head, on **unfiltered** 128-d arxiv ANN.
The gate was **pre-registered and frozen before any run** (`docs/plans/bet5-ivf-pq/PRE-REGISTRATION.md`).

**Product-quantized within-list pruning (an IVFADC cheap-ADC scan + a small exact-L2 re-rank) beats
a *tuned* plain `nprobe` — and the early-abandon exact-L2 steelman — by ≥ 2× full-L2-equivalent
member-evals at matched recall@10 = 0.95, AND on wall-clock, across all three `nclusters ∈
{64,256,1024}` at N = 100k.** The win **grows with N** and the crossover `n*` **increases with
`nclusters`** — a clean amortization signature, not a flat pass. Unlike ADR-205, the mechanism is
**orthogonal** to `nprobe` (it cheapens the *per-member* distance, not the *list selection*), so the
win is real rather than structurally impossible.

## Context

`ruvector-rairs::IvfFlat` (ADR-193) is plain IVF: k-means centroids + inverted lists; `search(q, k,
nprobe)` scans **all** members of the `nprobe` nearest lists with exact `D`-dim L2. PQ/IVFADC adds a
product quantizer: split each 128-d vector into `m` subvectors, train 256 sub-centroids per subspace
(8-bit codes), encode every vector to `m` bytes. Per query, build an **ADC lookup table** (query
subvector → its 256 sub-centroid distances, `m × 256` entries) and approximate any member's distance
by `m` table lookups — then recover exactness with an exact-L2 **re-rank** of the top-`R` ADC
candidates.

The kernel (`crates/ruvector-bet4-ivf-bench/src/pq.rs::PqIvf`) is built standalone over the same
`ruvector-rairs` k-means substrate as the incumbent (a shared `IvfParts` is clustered **once** per
cell and reused for every contender — identical centroids/lists by construction, certified in
`tests/pq_gate.rs`). Two correctness gates passed before any claim: PQ with a full re-rank pool is
**exact** (recall ≥ 0.999 — the lossy ADC only *orders*, exact L2 *decides*), and the early-abandon
steelman is **exact** vs full L2.

Three contenders share one index per `nclusters` (only the within-list scan differs):
- **plain `nprobe`** — full `D`-dim L2 on every member (ADR-205's incumbent; validated == `IvfFlat`).
- **early-abandon steelman** — exact L2 abandoned dim-by-dim at `τ²` (PQ-free within-list pruning;
  the user-confirmed verdict-setting incumbent — rule #5).
- **PQ/IVFADC** — cheap ADC scan of the same `nprobe` lists + exact re-rank of the top-`R` (the bet).

## Cost accounting (one honest unit — no free lunch)

**One unit = one full `D`-dim L2 = "1 member-eval-equivalent."** Everything converts to it:

| Operation | full-L2-equivalents |
|---|---|
| Plain full-L2 member | 1 |
| Early-abandoned L2 member | (dims touched) / D |
| **Centroid routing (charged to *all* contenders)** | **`nclusters` × 1** |
| PQ ADC table build (per query) | 256 (= `m`·256·(D/m)/D) |
| PQ ADC member scan | `m`/D |
| PQ exact re-rank member | 1 |

PQ total = `nclusters` (routing) + `256` (LUT) + `members · m/D` (ADC) + `R` (re-rank). Incumbent =
`nclusters` (routing) + `members · 1` (or less, early-abandoned). **Routing is charged equally to
both** — the pre-registered "no free routing" check. It is decisive at high `nclusters`, where it
nearly equals the working set (see deviation note below).

## Decision / Finding

**WIN, scale-gated.** Cost at matched recall@10 = 0.95, 200 queries; **total full-L2-equivalent
member-evals** (routing charged to both; **best `m` per cell**, PQ tuned like `nprobe`). Steelman
(early-abandon) is the cheaper incumbent in every cell, so it sets every ratio.

**Total-cost ratio (the frozen gate metric), PQ vs best PQ-free incumbent:**

| N | nclusters=64 | nclusters=256 | nclusters=1024 |
|---|---|---|---|
| 20,000  | **2.51×** WIN | 1.95× qual    | 1.33× miss    |
| 50,000  | **3.20×** WIN | **2.50×** WIN | 1.65× qual    |
| 100,000 | **3.38×** WIN | **2.80×** WIN | **2.03×** WIN |

**Wall-clock per query wins in every cell** (e.g. n=100k/nc=64: 346 µs vs 1664 µs plain / 1788 µs
abandon; the knife-edge n=100k/nc=1024: 216 µs vs 631 / 742) — **no reversal anywhere**, so the
eval win is corroborated by reality, not contradicted by it.

**Gate WIN condition — "≥ 2× AND wall-clock AND all three `nclusters` at ≥ one N ≥ 50k" — is MET at
N = 100k** (2.03× / 2.80× / 3.14–3.38×, wall-win throughout). At N = 50k it holds at `nclusters ∈
{64,256}` (qualified at 1024); at N = 20k only at `nclusters = 64`.

**Mechanism (the orthogonal axis — the key result).** `nprobe` decides *which* members to consider;
PQ cheapens the cost of *considering* one (`m/D ≈ 1/8` of a full L2 at `m=16`) and defers exact L2 to
a small re-rank. There is **no redundancy** with `nprobe`'s centroid cutoff (the ADR-205 failure
mode), so the saving is genuine. Its size is governed by **amortization**: PQ's fixed overhead
(`256` LUT + `R` re-rank + `nclusters` routing) is repaid only once the within-list working set
`members ≈ n·nprobe/nclusters` is large. Hence the two monotonic trends, both visible in the table:
- **grows with N** (working set ∝ n): nc=1024 goes 1.33× → 1.65× → 2.03× across 20k/50k/100k;
- **crossover `n*` rises with `nclusters`** (routing ∝ nclusters, working set ∝ 1/nclusters):
  nc=64 crosses 2× by n≈20k, nc=256 by n≈50k, nc=1024 only by n≈100k.

In the **sensible IVF range `nclusters ≈ √n`** (≈ 140–320 for these scales), PQ wins ≥ 2× from
n ≈ 20–50k upward. Over-clustering (nc=1024 for n ≤ 50k) is the only regime PQ loses — and there
routing dominates *every* method, so the within-list choice barely matters (at n=5k/nc=1024 the
total ratio is 0.95×, pulled toward 1.0 by 1024 routing evals shared by both).

## Honest caveats (the prove-not-hype core — none buried)

1. **The win rides on the exact re-rank, not the PQ distance itself.** Pure-ADC recall@10 is only
   **~0.48–0.52 (m=16)** / **~0.29–0.36 (m=8)** — PQ alone recovers barely half the true top-10 (the
   128-d concentration risk, real and named in the prior). The exact re-rank `R` carries recall from
   there to 0.95: `R* = 150→200→300` (m=16) and `500→1000→1500` (m=8) as N grows. **This is IVFADC +
   refine — FAISS's standard `IVFPQ,Refine` design — validated to pay on RuVector's data/scales, not
   a novel algorithm.** The honest claim is "ruvector-rairs should add an IVFPQ+rerank path," not
   "we invented within-list pruning."
2. **The clean WIN is scale-gated to N = 100k.** At N ≤ 50k the "all three nclusters" bar is not
   cleared (nc=1024 = 1.65× at 50k, 1.33× at 20k). The shippable claim is **scale-and-nclusters-
   resolved**, not universal: ≥ 2× at `nclusters ∈ {64,256}` from n ≈ 20–50k; the full sweep only at
   n = 100k. The decisive nc=1024/100k cell is a **knife-edge (2.03×)** — the crossover itself.
3. **`m = 16` is the tuned operating point.** `m = 8`'s coarser codes drop the ADC ceiling to ~0.3 →
   `R` blows up to 1000–1500 → re-rank cost erodes the win (it still wins at low nclusters but trails
   m=16 at high nclusters). Tuned PQ = `m=16`, as `nprobe` is tuned.
4. **Recall-floor tunability flatters PQ slightly.** Integer `nprobe` overshoots the 0.95 floor to
   0.957–0.970; PQ's finer `R` knob lands at 0.951–0.960. Part of PQ's edge is operating *exactly* at
   the floor while `nprobe` cannot. This is a genuine (if modest) PQ advantage — finer recall control
   — and the 2.5–3.4× margins at `nclusters ∈ {64,256}` dwarf the ~2–4% recall gap that drives it.
5. **The steelman mattered — a lot.** Early-abandon prunes **40–53%** of L2 dims and was the cheaper
   incumbent in *every* cell (e.g. 11,006 vs 23,232 at n=100k/nc=64). Against naive plain-L2 the PQ
   ratios would roughly **double** (~6×); reporting against the steelman keeps the headline honest at
   2–3.4×.

## The routing charge — an honest harness-bug catch

The first sweep **omitted routing from the cost ratio** — a bug in my own harness, since the frozen
accounting table charges `nclusters` centroid-evals to *both* contenders. It was decisive at high
`nclusters`: the n=50k/nc=1024 cell printed **2.24×** member-only but is **1.65×** once routing
(1024 evals) is folded into both costs. The pre-registered "no free routing" adversarial check caught
it against my own code; the authoritative table above charges routing throughout, and the harness now
prints **both** the member-only ratio (transparency) and the gate-deciding total. Recording the catch
is the point (cf. ADR-203's three deviations, ADR-205's PCA-control reversal).

## Consequences

**Positive (a real, shippable win — the first in the IVF-acceleration line).**
- **`ruvector-rairs::IvfFlat` should gain an `IVFPQ + exact-rerank` search path.** At matched
  recall@10 = 0.95 it cuts total member-eval cost 2–3.4× and wall-clock 3–5× in the sensible
  `nclusters ≈ √n` range from n ≈ 20–50k up; the payoff grows with scale. This is the first BET in
  the IVF line that *adds* shippable code rather than protecting the status quo (ADR-205).
- **Companion contrast to ADR-205/199.** Classical *exact* structures don't pay on embedding
  retrieval (graph separators — high treewidth, ADR-199; cluster bounds — redundant with `nprobe`,
  ADR-205). The *lossy-but-cheap* PQ distance with an exact re-rank **does** — because it attacks an
  axis `nprobe` leaves untouched. The kills sharpened *where* acceleration must come from; this is
  the where.

**Boundaries / honest scope.**
- **Scope: within-list PQ + rerank vs tuned `nprobe`, recall@10 = 0.95, 128-d arxiv.** The win is
  scale-gated (full sweep only at n=100k) and concentrated in `nclusters ≈ √n`. Not claimed: other
  recall targets, other corpora, or the over-clustered regime (nc=1024 below n≈100k).
- **It is IVFADC+refine, not a new method** — the contribution is the *measured, in-repo, steelman-
  and-routing-honest* demonstration that it beats `ruvector-rairs`'s current IVFFlat, with the regime
  mapped.

## Scoreboard

**3 WINS** (ADR-200/202 reuse+periodic; ADR-204 incremental high-recall tier; **ADR-206 PQ/IVFADC
within-list pruning, scale-gated**) / **4 KILLS** (ADR-199 CCH-on-embeddings; ADR-201 filtered-ANN
vs ACORN; ADR-203 KG-treewidth; ADR-205 IVF cluster-pruning vs `nprobe`).

## Next steps

1. **Productionize:** add an `IVFPQ + rerank` path to `ruvector-rairs::IvfFlat` (codebook training,
   `m`-byte codes, per-query ADC LUT, top-`R` exact rerank); default `m=16`, `R` auto-tuned to a
   recall SLA. The `PqIvf` kernel here is the reference.
2. **A coarse quantizer over centroids** would cut the `nclusters` routing charge that gates the
   high-`nclusters` win (HNSW-over-centroids, as FAISS `IVF…_HNSW` does) — would lift nc=1024 cleanly
   past 2× below n=100k. Different mechanism; a natural follow-on bet.
3. **OPQ / larger codebooks** (rotation before PQ) would raise the ~0.5 ADC ceiling, shrinking the
   re-rank `R` that currently carries recall — directly widens the win. Measurable on this harness.

## Alternatives considered

- **Pure ADC, no re-rank** — ceiling ~0.48–0.52 recall@10; cannot reach 0.95. Rejected (the re-rank
  is load-bearing).
- **`m = 8`** — coarser codes, ADC ceiling ~0.3, `R` up to 1500; wins at low nclusters but trails
  m=16. Retained only as the tuned-`m` sweep's loser.
- **Cluster-level triangle bound (ADR-205)** — redundant with `nprobe` (1.00×). The orthogonal
  within-list axis here is why PQ succeeds where that failed.
