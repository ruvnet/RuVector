# BET 5 — Pre-Registration (FROZEN): PQ/IVFADC within-list pruning vs tuned plain `IvfFlat` `nprobe`

**Status: FROZEN (2026-06-05, user-confirmed).** No gate, threshold, metric, dataset, accounting
rule, or control below may change. The steelman incumbent (early-abandoned exact L2, user-confirmed)
is the verdict-setting PQ-free baseline. Deviations are limited to the pre-authorised list at the
end; any other change voids the run.

Thread: SepRAG (ruvnet/RuVector issue #534). This opens the **one lever ADR-205 left explicitly
open**: ADR-205 killed *cluster-level* triangle-inequality pruning vs tuned `nprobe` (structurally
redundant — the bound only skips far clusters `nprobe` already avoids). Its "Next steps #1" names the
different mechanism: **within-list pruning via product-quantized / IVFADC asymmetric distance.** This
is that bet.

Stacked on `feat/seprag-bet4-ivf-pruning` (PR #540) to **reuse the `ruvector-bet4-ivf-bench`
harness** (data loader, brute-force oracle, shared `ruvector-rairs` k-means substrate, sweep
skeleton). New module `src/pq.rs`, new example `examples/pq_pruning_sweep.rs`, new ADR-206. Valid
regardless of #540's merge fate (additive; depends only on `ruvector-rairs`, which is on main).

## Why this is NOT a re-run of ADR-205 (the mechanism is orthogonal, not redundant)

ADR-205's bound competed with `nprobe` on the **same axis** (which lists to scan) → redundant → 1.00×.
PQ competes on a **different axis**: `nprobe` decides *which* members to consider; PQ makes the cost
of *considering* a member cheaper (an `m`-entry table lookup-sum instead of a `D`-dim L2) **and**
lets a list be scanned approximately, deferring exact L2 to a small re-rank shortlist. There is no
redundancy with `nprobe`'s centroid cutoff. So a win is **not** structurally impossible here — the
question is purely empirical: does the cheaper-but-lossy per-member distance, plus its fixed
overheads, net out ahead of a tuned exact `nprobe` at matched recall, **at RuVector's scales**.

## Claim (one claim, one number)

> On unfiltered ANN over real **128-d** arxiv embeddings, **PQ/IVFADC within-list pruning**
> (approximate ADC scan of the `nprobe` lists + exact L2 re-rank of the top-`R` ADC candidates)
> reaches **matched recall@10 = 0.95** at **≥ 2× fewer full-L2-equivalent member-evals** than the
> strongest PQ-free incumbent, **and wins on wall-clock**, holding across `nclusters ∈ {64,256,1024}`
> at ≥ one scale `N ≥ 50k`.

## Incumbents (tuned, in-repo — and a steelman, no straw man)

Both share the **same k-means centroids/seed/lists** as the contender (only the within-list scan
differs), built over `ruvector-rairs::kmeans::train` — the same substrate as ADR-205.

1. **Plain `nprobe` full-L2** (the baseline, identical to ADR-205's incumbent; validated equal to
   `ruvector-rairs::IvfFlat`): scan all members of the `nprobe` nearest lists with exact `D`-dim L2.
2. **Steelman incumbent — `nprobe` + early-abandoned exact L2** (PQ-free *within-list pruning*):
   identical list selection, but each member's L2 is computed dim-by-dim and **abandoned** the
   instant the partial sum exceeds the current k-th-best `τ`. This is exact (no recall loss) and is
   the natural, free within-list pruning that needs no PQ. **The PQ contender must beat this**, not
   just naive full-L2 — rule #5 (steelman the incumbent so a kill is credible *and* a win is real).
   Cost is charged as **dims actually touched / D** full-L2-equivalents, so early abandonment gets
   full credit for the work it skips.

The verdict-setting incumbent is the **cheaper of the two** at matched recall (PQ must beat the best
PQ-free option available).

## Contender (the bet — `PqIvf`, rebuilt standalone over the shared index)

`PqIvf` over the same centroids/lists:
- **Train** `m` sub-quantizers: split each 128-d vector into `m` contiguous subvectors of `D/m` dims;
  train `2^nbits = 256` sub-centroids per subspace via `ruvector-rairs::kmeans::train` on the sliced
  subvectors (8-bit codes). Encode every corpus vector to its `m`-byte PQ code. **Build-time;
  reported separately, never hidden.**
- **Per query:** build the **ADC lookup table** — for each of the `m` subspaces, the L2² from the
  query subvector to all 256 sub-centroids (`m × 256` partial distances). **Charged per query** as
  `(m × 256 × (D/m)) / D = 256` full-L2-equivalents (the fixed overhead whose amortization is the
  whole bet — not hidden).
- **ADC scan:** for each member of the `nprobe` lists, approximate distance = sum of `m` table
  entries indexed by its code. **Charged `m / D` full-L2-equivalents per member.**
- **Exact re-rank:** take the top-`R` members by ADC distance and recompute exact `D`-dim L2 on
  them; return the top-k of those. **Charged `R` full-L2-equivalents** (one full L2 each).
- Knobs (the analogues of `nprobe`): `nprobe` (lists), `m ∈ {8, 16}` (sub-quantizers), `R` (re-rank
  pool). Tuned to the smallest cost reaching recall@10 ≥ 0.95, same as `nprobe` is tuned.

## Cost accounting (the honesty core — one unit, no free lunch)

**One unit = one full `D`-dim L2 = "1 member-eval-equivalent."** Everything converts to it:

| Operation | full-L2-equivalents |
|---|---|
| Plain full-L2 member | 1 |
| Early-abandoned L2 member | (dims touched) / D |
| Centroid routing (both, cancels but counted) | `nclusters` × 1 |
| PQ ADC table build (per query) | 256 (= `m`·256·(D/m)/D) |
| PQ ADC member scan | `m`/D |
| PQ exact re-rank member | 1 |

PQ's total = `256` (LUT) + `nprobe_members · m/D` (ADC) + `R` (re-rank). Incumbent's = `nprobe_members
· 1` (or less with early abandon). The fixed `256` LUT charge is what a small tuned working set must
overcome — **this is exactly the amortization question, and it is paid in full.**

## Data

`target/m1-data/node-feat-100k.csv` — ogbn-arxiv 128-d node features (public, aligned, same corpus as
ADR-201/202/204/205). N-sweep at **20,000 / 50,000 / 100,000** (three scales to *map the
amortization crossover* `n*`, not just pass/fail). Queries: 200 held-out points. Ground truth:
brute-force exact L2 kNN@10 on the corpus.

## Metrics

- **Primary: full-L2-equivalent member-evals at matched recall@10 = 0.95.** Per the table above.
- **Secondary (honesty guard): wall-clock per query.** An eval win that **reverses on wall-clock** is
  **"inconclusive," never WIN** (ADR-201/205 precedent). PQ's table-lookup inner loop has different
  cache behaviour than L2, so this guard has real teeth here.
- **Reported regardless:**
  - **Pure-ADC recall ceiling** (recall@10 of ADC ranking with **no** re-rank) per cell — how lossy
    PQ is on this data; the mechanistic explainer for the `R` it needs.
  - **`R` (re-rank pool) required** per cell to reach 0.95.
  - **Crossover `n*`** — the scale at which PQ overtakes the best incumbent (the amortization point).
  - **Early-abandon pruning fraction** — mean % of L2 dims the steelman skips (does exact within-list
    pruning work at all on concentrated 128-d?).

## Matched-recall protocol

Recall target **R₀ = 0.95**, k = 10. Per `nclusters ∈ {64,256,1024}` and per `N ∈ {20k,50k,100k}`:
tune plain/steelman `nprobe` to the smallest value reaching mean recall@10 ≥ 0.95; record evals.
Tune PQ `(nprobe, m, R)` to the smallest full-L2-equivalent cost reaching ≥ 0.95; record evals.
Compare PQ to the **cheaper** incumbent. (Also report exact regime: incumbent full-scan vs PQ at the
`R` that recovers ≥ 0.999.)

## Gate (to be FROZEN)

| Verdict | Condition |
|---|---|
| **WIN** | full-L2-equivalent reduction **≥ 2×** vs the best PQ-free incumbent at recall@10 = 0.95 **AND** wall-clock win **AND** holds across all three `nclusters` at ≥ one `N ≥ 50k`. |
| **KILL (NO-GO)** | reduction **< 1.5×** in every cell **OR** wall-clock reverses **OR** PQ cannot reach 0.95 recall at any tractable `R` (≤ `nprobe_members`; i.e. the quantization ceiling is too low to recover cheaply). |
| **Qualified** | between 1.5× and 2×, or wins at some `nclusters`/`N` but not all → report as a **scale/regime-conditional edge** with the crossover `n*` named (not a clean WIN). |
| **Report always** | pure-ADC recall ceiling; `R` per cell; crossover `n*`; early-abandon pruning fraction; the full (recall, eval, wall-clock) frontier per cell. |

## Controls (the teeth — both mandatory)

1. **Pure-ADC-recall probe (the mechanism control).** Measure ADC-only recall@10 (no re-rank) per
   cell. This isolates *how lossy* PQ is on 128-d arxiv. If ADC recall is already ≈ 0.95, PQ wins
   trivially (tiny `R`); if it is low, the re-rank `R` must carry recall and the win rides on whether
   `R` stays small — the explainer for whichever verdict lands. (Replaces ADR-205's PCA-8 control,
   whose role — *isolate the bound's tightness* — does not transfer; PQ's loss axis is quantization
   coarseness, measured directly here. See deviation note.)
2. **Early-abandon-vs-full-L2 control (the steelman is itself a control).** If early abandonment
   prunes ≈ 0% of dims on concentrated 128-d, that confirms the same distance-concentration that
   killed ADR-205's bound also defeats *exact* within-list pruning — isolating PQ's *lossy compute*
   as the only working within-list lever. If early abandonment prunes a lot, the steelman is strong
   and a PQ win is harder-earned.

## Adversarial checks (pre-committed)

- **No free LUT:** the `256`-equivalent ADC table build is charged **every query**; the win must
  survive it. (This is the amortization crux, not a footnote.)
- **No free codebook:** PQ codebook training is build-time, reported separately like ADR-205's radius
  bookkeeping — never folded into the per-query win.
- **Wall-clock guard:** eval win must not reverse on wall-clock (table-lookup cache effects are real).
- **Shared index:** identical centroids/seed/lists for all contenders; only the within-list scan
  differs. No re-clustering between contenders.
- **Re-rank honesty:** the `R` exact L2s are charged at full cost (1 each); a win cannot hide behind
  an uncharged re-rank.
- **Ceiling reconciliation:** a matched-recall WIN that requires `R` ≳ `nprobe_members` is not a
  win (PQ would be re-ranking the whole working set exactly — it has bought nothing); must be flagged.

## Honest prior (stated before any run, per protocol)

I lean **genuinely uncertain, with a slight WIN-at-scale lean** — the most honest reading of the
mechanics, and unlike ADR-205 this is *not* a foregone kill:

- **For a win:** PQ's per-member cost is ~`m/D` (≈ 1/8 at `m=16`) of full L2; the moment the `nprobe`
  working set is large (large `N`, or many lists), the `256`-equivalent LUT amortizes and the cheap
  ADC scan + small re-rank should undercut full-L2 `nprobe`. This is the textbook reason IVFPQ
  exists. A clean win would say "ruvector-rairs should add IVFPQ for large-`N` IVF" — a real,
  consumer-independent, *shippable* finding (the first WIN in the IVF-acceleration line).
- **For a kill / qualified:** two named risks. (a) **Amortization** — at moderate `N` (20k–50k) a
  *tuned* `nprobe` scans a *small* working set (it is tuned down to a few lists), so the fixed `256`
  LUT + re-rank `R` may not pay; the win could be purely asymptotic and *absent* at RuVector's
  scales. (b) **Concentration ceiling** — the same 128-d distance concentration that killed ADR-199
  /205 makes ADC ranking noisy (true neighbours scattered deep in ADC order), forcing a large `R` to
  recover 0.95; if `R` blows up, the re-rank cost erases the ADC saving → NO-GO, the IVFADC companion
  to "Euclidean embedding geometry defeats classical acceleration." I rate (b) the sharper risk.

Net: ~55% WIN at `N ≥ 50k`, with a real chance the crossover `n*` sits *above* RuVector's tested
scales (→ qualified) or that the concentration ceiling forces `R` too high (→ clean NO-GO). Either
outcome is a tidy, consumer-independent finding — the reason this is the chosen next bet.

## Pre-authorised deviations (anything else voids the run)

- Substitute the pure-ADC-recall control's role only if PQ training is impractical to implement
  cleanly; the *role* (measure PQ's quantization loss directly) is fixed.
- Reduce the largest `N` from 100k to ≥ 50k if 100k brute-force truth is prohibitively slow,
  **provided** at least three distinct scales spanning ≥ 4× are reported, the largest ≥ 50k.
- Adjust query count upward (≥ 200) for noise control; never below 200.
- Add `m` or `R` settings; never drop a required `nclusters ∈ {64,256,1024}`.
- If `m=16` and `m=8` bracket the same verdict, report both but the gate is read on the better `m`
  per cell (PQ is *tuned*, like `nprobe`).

## Plan

- **M0** — `src/pq.rs`: `PqIvf` (sub-quantizer training over shared k-means index, encode, ADC LUT,
  `search_adc_rerank`), early-abandon incumbent scan; **gate test** PQ@full-rerank == oracle
  (recall ≥ 0.999) + PQ shares centroids with `BnBIvf`/`IvfFlat`. clippy clean.
- **M1** — instrument full-L2-equivalent counting on all three contenders (shared index); pure-ADC
  recall probe.
- **M2** — matched-recall sweep `examples/pq_pruning_sweep.rs`: `nclusters` × `N` × `(m,R)` grid,
  crossover `n*`, frontier print.
- **M3** — controls (pure-ADC ceiling, early-abandon fraction); adversarial reconciliation; verdict
  against this gate.
- **M4** — ADR-206 (WIN / NO-GO / qualified — honest, ADR-199/201/205 precedent); one PR at M4
  stacked on #540, linked to #534; #534 scoreboard comment.

---

**Frozen.** Build starts at M0 against this document; the gate is not revisited.
