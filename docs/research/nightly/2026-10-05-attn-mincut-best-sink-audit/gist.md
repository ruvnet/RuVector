# Auditing a Min-Cut Attention Crate's Performance Claims: A Clean REJECT with a Root Cause

## Problem

`ruvector-attn-mincut` is a small Rust crate implementing "min-cut gated
attention": instead of applying softmax uniformly over all `Q.K^T` logits,
it builds a weighted directed graph from the logits and computes an s-t
min-cut (Dinic's algorithm) to prune "irrelevant" attention edges before
softmax. Its README carries a specific comparison table against dense
softmax attention:

| | Softmax Attention | Min-Cut Gated |
|---|---|---|
| KV-cache usage | Full | 15-40% reduction |
| Energy per sample | Baseline | 10-20% lower |
| Coherence | Reference | <1% degradation |

The crate had unit tests at `seq_len` 1-5 and no benchmark, example, or
integration test exercising these numbers at any realistic sequence length.

## Hypothesis

```text
Given synthetic Q/K/V attention inputs (deterministic seed, mild locality
bias, d=64) at seq_len in {32, 64, 128, 256} and lambda in {0.3, 0.5, 0.7},

when the shipped attn_mincut (fixed sink t=seq_len-1) is compared against
an eps-only elementwise-threshold ablation (identical eps clamp, no graph
cut at all) on the same logits,

then the min-cut step should remove >= 5 percentage points of edges beyond
the ablation, averaged over seq_len in {64,128,256} x lambda (H1),

and a best-sink variant (exhaustive search over every possible sink,
instead of the shipped fixed sink) should remove >= 3pp more than the
shipped variant at matched grid points (H2), showing the fixed-sink choice
specifically (not the min-cut concept) explains any H1 shortfall,

subject to: output cosine similarity to baseline softmax attention staying
>= 0.99 at seq_len=128, lambda=0.5 (the literal "<1% degradation" claim).
```

Thresholds were fixed in the benchmark source before running it once.

## Technical design

- **Ablation control** (`eps_only_keep_mask`): the elementwise `logit > eps`
  mask the crate already applies regardless of its cut outcome, isolated
  as its own function so the min-cut step's *marginal* contribution can be
  measured directly: `mincut_specific_pp = (eps_kept - variant_kept) / n`.
- **Candidate_A**: the shipped `attn_mincut`/`dynamic_min_cut`, unchanged.
- **Candidate_B** (`attn_mincut_best_sink`/`dynamic_min_cut_best_sink`,
  added this run): fixes the source at `0` (matching the shipped choice)
  but searches every sink `t in 1..seq_len` with Dinic's algorithm and
  keeps the globally cheapest cut found, instead of the shipped
  implementation's single fixed `t = seq_len - 1`. By construction this
  searches a superset of candidate_A's cuts, so `best.cut_cost <=
  fixed.cut_cost` always — verified as a unit test, not just asserted in
  prose.
- **Coherence scoring**: `ruvector-coherence::quality_check` (cosine
  similarity + L2 distance against baseline `attn_softmax` output) and
  `compare_attention_masks` (Jaccard/edge-flip comparison of keep-masks) —
  an existing, previously-unused-outside-its-own-tests crate in the same
  workspace, built for exactly this comparison.
- **Benchmark**: `examples/claim_audit_bench.rs`, a self-contained Rust
  binary with a tiny inline LCG for deterministic synthetic data (no new
  runtime dependency), timing via `std::time::Instant` with reps scaled
  down as `seq_len` grows, and an explicit frozen ACCEPT/REJECT printout.

## Actual benchmark evidence (release build, real run)

```text
 seq | lam  | variant      | edges kept/tot  | mincutPP |  cosine |      latency |   ratio
  32 | 0.5  | candidate_A  |    459/1024   |   0.00pp |  0.7566 |      97.51us |   1.40x
  32 | 0.5  | candidate_B  |    459/1024   |   0.00pp |  0.7566 |     939.15us |  13.53x
  64 | 0.5  | candidate_A  |   1801/4096   |   0.00pp |  0.6946 |     394.51us |   1.44x
  64 | 0.5  | candidate_B  |   1801/4096   |   0.00pp |  0.6946 |    5189.71us |  19.00x
 128 | 0.5  | candidate_A  |   6898/16384  |   0.00pp |  0.7513 |    1579.68us |   1.41x
 128 | 0.5  | candidate_B  |   6898/16384  |   0.00pp |  0.7513 |   29928.04us |  26.76x
 256 | 0.5  | candidate_A  |  27851/65536  |   0.00pp |  0.7297 |    6302.29us |   1.46x

H1: avg=0.000pp -> REJECT
H2: A_avg=0.000pp B_avg=0.000pp delta=0.000pp -> REJECT
Coherence gate (seq_len=128, lambda=0.5): cosine_sim=0.75134 (threshold 0.99) -> FAIL
Candidate_A slower than baseline on every tested grid point.

OVERALL ACCEPTANCE: REJECT
```

(Full grid, including lambda 0.3/0.7, and a supplementary root-cause probe
at small `seq_len`, in the [full nightly report](README.md).)

**At every tested `seq_len >= 32`, the min-cut step removed exactly zero
edges beyond the trivial elementwise threshold — for both the shipped
fixed-sink implementation and an exhaustive best-sink search.** Candidate_A
is 1.4-1.7x slower than dense softmax, not faster. Candidate_B is 13-27x
slower. Coherence sits at ~0.70-0.76 cosine similarity to baseline, far
below the claimed <1%-degradation bar.

## Root cause

A supplementary diagnostic (same binary, additional small `seq_len` points,
clearly separated from the frozen H1/H2 grid) traced this to the gate
condition itself:

```text
apply_cut  iff  cut_cost(s, t) <= lambda * mean_edge_weight
```

`cut_cost` sums the weights of edges crossing the cut — for a dense
attention graph this grows with `seq_len` (roughly with the cut vertex's
degree). `mean_edge_weight` is a single-edge weight scale that does not
grow with `seq_len` at all. The inequality gets *harder* to satisfy as
context grows, for any `lambda` in the crate's documented `[0, 1]` range.
The probe shows this directly: the gate fires at `seq_len=8` (3.12pp
pruned) and decays to needing `lambda=50` — 50x the documented maximum —
just to fire at all by `seq_len=64` (0.54pp). The crate's own unit tests,
at `seq_len` 1-5, sit comfortably inside the regime where the gate still
fires, which is exactly why this was never caught before.

## Limitations

- Synthetic data (seeded LCG + mild locality bias), not a real trained
  model's attention patterns — the root cause is a property of the gate
  formula and graph density, not of text-specific structure, so this is
  unlikely to overturn the finding but is not independently confirmed
  against real attention maps.
- Candidate_B's latency figures are single-shot (not averaged) given its
  `O(seq_len)`-Dinic-call cost — the direction and scale of the slowdown is
  unambiguous, but exact multipliers carry more noise than candidate_A's.
- No energy instrumentation exists in this repository; "energy per sample"
  is neither validated nor refuted here, only latency (a loose proxy) is
  measured, and the report says so explicitly rather than conflating them.
- `seq_len >= 256` untested for candidate_B — ruled out by its own measured
  scaling trend, not attempted.

## Production relevance

Not a reason to rip the crate out — it's a reason not to trust its claims
table, and a concrete three-step path to make the mechanism real: (1) fix
the gate to scale with cut size instead of being an absolute edge-weight
threshold, (2) swap the from-scratch Dinic solver for `ruvector-mincut`'s
hardened global min-cut (fixed for determinism in an earlier nightly,
2026-09-11, ADR-346) once it has a small-ephemeral-graph entry point, and
(3) re-run this exact harness against the fixed design before writing any
new number in a README. The crate's README was updated in this same change
to reflect what was actually measured.

## RuVector ecosystem implications

This connects three RuVector capabilities that hadn't previously been
exercised together: the attention-level min-cut crate, the coherence-
scoring crate built for exactly this comparison, and (as the recommended
fix) the hardened graph-level min-cut primitive from a recent, unrelated
nightly. It also demonstrates a reusable pattern for the ecosystem: an
ablation-controlled, frozen-hypothesis benchmark is enough to turn a vague
"is this claim real?" into a specific, falsifiable, and in this case
falsified, answer with a located root cause — the kind of evidence a
`ruFlo` continuous claims-audit workflow could run automatically across
every crate in the workspace carrying a quantitative claims table.

## Future direction

Build candidate_C on `ruvector-mincut`'s hardened backend with a scale-
aware gate, re-run this same harness, and only then revise the README's
claims table from measured numbers. If candidate_C still fails H1, the
min-cut-for-attention idea itself — not just this implementation — would
be the next thing to question.

## References

- `crates/ruvector-attn-mincut/` (source, this run's new tests/example).
- `crates/ruvector-coherence/src/{quality,comparison}.rs`.
- ADR-346 and `docs/research/nightly/2026-09-11-mincut-partition-determinism/`
  (the `ruvector-mincut` hardening this report recommends building on).
- Full report with complete grid, mermaid architecture diagram, and all
  38 nightly-template sections: [README.md](README.md).
