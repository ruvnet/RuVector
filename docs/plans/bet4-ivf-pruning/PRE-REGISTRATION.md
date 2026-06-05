# BET 4 — Pre-Registration (FROZEN): LB-ordered branch-and-bound IVF probing vs tuned plain `IvfFlat`

**Status: FROZEN (2026-06-05, user-confirmed).** No gate, threshold, metric, dataset, or
control below may change after this commit. Deviations are limited to the explicitly
pre-authorised list at the end; any other change voids the run.

Thread: SepRAG (ruvnet/RuVector issue #534). This closes the BET 4 caveat left open by ADR-201
(#536): the region-pruning IVF kernel was built and validated *exact* there, but only ever run as
BET 2's mechanism **against ACORN** — never head-to-head against its own natural incumbent, **plain
IVF `nprobe` probing**. This is that head-to-head, on **unfiltered** ANN (no predicate — the
filtered question is BET 2, resolved NO-GO).

Independent of #535/#537/#539: this branch (`feat/seprag-bet4-ivf-pruning`) is cut off **clean
main**. The incumbent (`ruvector-rairs::IvfFlat`) is on main; the B&B kernel (which lives only on
the BET 2 branch) is **rebuilt self-contained** here, so the result is valid regardless of any
other PR's fate.

## Claim (one claim, one number)

> On unfiltered ANN over real **128-d** arxiv embeddings, **lower-bound-ordered branch-and-bound
> IVF probing** scans **≥ 2× fewer member distance-evals** than a **tuned plain `IvfFlat`
> `nprobe`**, at **matched recall@10**, **and wins on wall-clock**.

## Incumbent (tuned, in-repo — no straw man)

`ruvector-rairs::IvfFlat` (`crates/ruvector-rairs/src/ivf.rs`): k-means centroids + inverted lists;
`search(query, k, nprobe)` scans **all** members of the `nprobe` nearest-centroid lists, then
finalises top-k. Tuned = sweep `nclusters ∈ {64, 256, 1024}` × `nprobe ∈ [1, nclusters]` to its
best (recall, cost) frontier. **Both contenders share the same k-means centroids and seed** — only
the *probing strategy* differs, so the comparison isolates the strategy, not clustering luck.

## Contender (the bet — rebuilt standalone)

`BnBIvf` over the same centroids/lists:
- Precompute per-cluster radius `r_c = max_{v ∈ list_c} ‖v − centroid_c‖`.
- For a query `q`: compute `‖q − centroid_c‖` for all `c` (routing cost, charged); lower bound
  `LB(q,c) = max(0, ‖q − centroid_c‖ − r_c)`.
- Probe clusters in **ascending `LB`** order, maintaining a running k-th-best distance `τ`; scan a
  cluster's members (each a charged distance-eval), update `τ`; **break when `LB(c) ≥ τ`** (no
  unscanned cluster can contain a top-k point → provably done).
- **Exact** at full budget (recall → 1.0). A `max_probe` cap (probe at most that many clusters) is
  the approx knob used to hit a sub-1.0 recall target for the matched-recall comparison — the
  analogue of `nprobe`.

## Data

`target/m1-data/node-feat-100k.csv` — ogbn-arxiv 128-d node features (public, aligned, the same
corpus used by ADR-201/202/204). N-sweep at **20,000 and 100,000**. Queries: 200 held-out points.
Ground truth: brute-force exact L2 kNN@10 recomputed on the corpus.

## Metrics

- **Primary: member distance-evals at matched recall@10.** The count of query↔member L2
  evaluations (the dominant cost). Charged identically for both contenders. *Both* are additionally
  charged the `nclusters` query↔centroid routing evals (equal for both) and B&B's radius
  bookkeeping is build-time (reported separately, not hidden).
- **Secondary (honesty guard): wall-clock per query.** An eval win that **reverses on wall-clock**
  is reported as **"inconclusive," never WIN** (ADR-201 precedent).
- **Reported regardless: exact-regime pruning fraction** — the mean % of clusters B&B skips at
  recall → 1.0. The mechanistic explainer for whichever verdict lands.

## Matched-recall protocol

Pick recall target **R = 0.95**. Tune plain IVF `nprobe` (per `nclusters`) to the smallest value
reaching mean recall@10 ≥ R; record its member-evals. Cap `BnBIvf`'s `max_probe` to the smallest
value reaching ≥ R; record its member-evals. Compare. Repeat per `nclusters ∈ {64, 256, 1024}` and
per N ∈ {20k, 100k}. (Also report the **exact** regime R → 1.0: B&B full-budget vs `nprobe =
nclusters` full scan.)

## Gate (FROZEN)

| Verdict | Condition |
|---|---|
| **WIN** | member-scan reduction **≥ 2×** vs tuned `nprobe` at matched recall@10 (R = 0.95) **AND** wall-clock win **AND** holds across all three `nclusters` settings (at ≥ one N). |
| **KILL (NO-GO)** | reduction **< 1.5×** at matched recall **OR** wall-clock reverses. Interpretation: the triangle-inequality bound is too loose in 128-d (distance concentration) to pay. |
| **Qualified** | between 1.5× and 2×, or wins at some `nclusters`/N but not all → report as a **narrow/conditional edge** with the regime named (not a clean WIN). |
| **Report always** | exact-regime pruning fraction; the full (recall, member-evals, wall-clock) frontier per cell. |

## Controls (the teeth — both mandatory)

1. **Exact-vs-exact probe** (R → 1.0): `BnBIvf` full-budget vs `IvfFlat` `nprobe = nclusters`
   (full scan). Directly measures whether the LB bound prunes **at all** in 128-d. If ~0% of
   clusters are pruned here, that *mechanistically* predicts the KILL — and would make any
   matched-recall WIN suspect (must be reconciled).
2. **Low-dimensional control:** rerun the entire protocol on a **low-intrinsic-dim** input —
   PCA-project the arxiv features to **8-d** (retain the top-8 principal components). The bound is
   expected to be tight here, so `BnBIvf` **should WIN** the low-d control. This proves the kernel
   and harness are *sound* and isolates **high-d concentration** as the cause of any 128-d NO-GO —
   BET 4's analogue of BET 3's roadNet control and BET 1's stale-index control. If the kernel does
   **not** win even at 8-d, the implementation is suspect and the 128-d result is uninterpretable.

## Adversarial checks (pre-committed)

- **No free routing:** B&B is charged the `nclusters` centroid evals every query; the win must
  survive that charge (it is identical for plain IVF, so it cancels, but it is *counted*, not
  ignored).
- **Wall-clock guard** (above): eval win must not reverse on wall-clock.
- **Shared index:** identical centroids/seed/lists for both contenders; the *only* difference is
  the probe loop. No re-clustering between contenders.
- **Pruning-fraction reconciliation:** a matched-recall WIN with ~0% exact-regime pruning is
  internally inconsistent and must be explained before being reported as a WIN.

## Honest prior (stated before any run, per protocol)

I lean **NO-GO at 128-d.** Under distance concentration the per-cluster radius `r_c` tends to be
large relative to inter-centroid gaps, so `LB = max(0, d − r_c) ≈ 0` for most clusters → little
pruning → proving exactness scans nearly everything, costing more than a tuned `nprobe` that
accepts < 100% recall. That would be a clean kill, the IVF-level companion to ADR-199 (Euclidean
embedding geometry defeats classical pruning structures — separators there, triangle-inequality
cluster bounds here). A WIN would be a genuine shippable `IvfFlat` upgrade. Either outcome is a
tidy, **consumer-independent** finding — the reason this is the chosen next bet.

## Pre-authorised deviations (anything else voids the run)

- Substitute PCA-to-8-d with a synthetic low-d clustered set **only if** PCA is impractical to
  implement cleanly; the *role* (a tight-bound low-d control) is fixed.
- Reduce N from 100k to a smaller second scale if 100k brute-force truth is prohibitively slow,
  **provided** at least two distinct scales are reported and the larger is ≥ 50k.
- Adjust query count upward (≥ 200) for noise control; never below 200.
- Add `nclusters` settings; never drop one of {64, 256, 1024}.

## Plan

- **M0** — self-contained crate `crates/ruvector-bet4-ivf-bench` (deps: `ruvector-rairs`, `rand`):
  data loader, `BnBIvf` kernel, brute-force oracle; **gate test** `BnBIvf` full-budget == oracle
  (recall 1.0). clippy clean.
- **M1** — instrument member-eval + wall-clock counting on both contenders (shared index).
- **M2** — matched-recall sweep harness (`examples/ivf_pruning_sweep.rs`): the `nclusters` × N grid,
  exact-regime probe, frontier print.
- **M3** — low-d (PCA-8) control; adversarial reconciliation; verdict against this gate.
- **M4** — ADR-205 (WIN, NO-GO, or qualified — honest, ADR-199/201 precedent); one PR at M4 linked
  to #534; #534 scoreboard comment.

---

**Frozen.** Build starts at M0 against this document; the gate is not revisited.
