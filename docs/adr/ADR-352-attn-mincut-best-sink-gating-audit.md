# ADR-352: Min-Cut Attention Gating Does Not Prune Beyond Threshold at Realistic Scale

## Status

Accepted (as a research finding). Non-breaking, additive change to
`ruvector-attn-mincut` (two new public functions, one new public operator,
one function made public, zero API removals) plus a README correction. No
production code path depended on the audited claims before this change, so
there is no migration for downstream consumers.

## Context

`ruvector-attn-mincut` implements "min-cut gated attention": a graph is
built from `Q.K^T` attention logits and Dinic's max-flow algorithm computes
an s-t min-cut to prune edges before softmax, instead of applying softmax
densely. Its README advertised, relative to dense softmax: 15-40% KV-cache
reduction, 10-20% lower energy per sample, <1% coherence degradation. The
crate had unit tests only at `seq_len` 1-5 and no benchmark exercising any
of these numbers at a realistic sequence length.

This nightly run (`docs/research/nightly/2026-10-05-attn-mincut-best-sink-audit/README.md`)
built that benchmark. The full raw results, methodology, and root-cause
analysis are in that report; this ADR records the decision and its
rationale.

## Hypothesis

```text
Given synthetic Q/K/V attention inputs (deterministic seed, mild locality
bias, d=64) at seq_len in {32, 64, 128, 256} and lambda in {0.3, 0.5, 0.7},

when the shipped attn_mincut (candidate_A, fixed sink t=seq_len-1) is
compared against an eps-only elementwise-threshold ablation (no graph cut)
on identical logits,

then candidate_A's min-cut step should remove >= 5 percentage points of
edges beyond the ablation, averaged over seq_len in {64,128,256} x lambda
(H1); and a best-sink exhaustive variant (candidate_B) should remove >= 3pp
more than candidate_A at matched grid points (H2), isolating whether the
fixed-sink choice (vs. the min-cut concept itself) explains any shortfall;

subject to: candidate_A's output cosine similarity to baseline attn_softmax
remaining >= 0.99 at seq_len=128, lambda=0.5 (the README's literal claim).
```

Thresholds were fixed in the benchmark source (`examples/claim_audit_bench.rs`)
before it was run and not altered afterward.

## Decision

1. **Do not change the production-referenced implementation** (`attn_mincut`
   / `dynamic_min_cut`). The bug this audit found is a design property
   (threshold formula does not scale with graph size), not a localized
   defect with an obvious one-line fix, and this harness's rules forbid
   weakening acceptance criteria or silently changing the hypothesis to
   manufacture a pass. The correct fix (a scale-aware gate, see
   [Decision 4](#decision) below) is left for a future nightly.
2. **Add, but do not promote, `attn_mincut_best_sink` /
   `dynamic_min_cut_best_sink`** (candidate_B) as a tested research
   artifact. It is kept in the crate per this repository's rule that "a
   failed Darwin candidate must remain part of the lineage so future runs
   do not rediscover it blindly" — H2 (REJECT) is a real, reusable result:
   the fixed-sink choice is *not* why candidate_A under-performs, so a
   future researcher does not need to re-propose "just search more sinks."
3. **Make `eps_only_keep_mask` and `compute_logits` public API.** Both are
   small, pure, and now load-bearing for any future audit or Darwin
   candidate against this crate — hiding them again would make the next
   nightly reinvent them.
4. **Correct the crate's README** (`crates/ruvector-attn-mincut/README.md`)
   to replace the unverified claims table with this run's measured numbers
   and an explicit pointer to the full report, rather than leaving an
   unsubstantiated table in place. See the companion diff in this same
   change.
5. **Record a concrete production path** (scale-aware gate + migrate to
   `ruvector-mincut`'s hardened backend + re-run this harness) as "next
   research," not as work done in this change — scope discipline: this
   nightly's job was to audit the existing claim, not to also design and
   validate its replacement in the same run.

## Evidence

- `cargo test -p ruvector-attn-mincut --release`: 25/25 passed, including
  three new tests (`test_eps_only_mask_matches_simple_threshold`,
  `test_best_sink_cut_cost_le_fixed_sink`,
  `test_mincut_best_sink_shape_and_finite`).
- `cargo clippy -p ruvector-attn-mincut --release -- -D warnings`: clean.
- `cargo check --workspace` (pre-change baseline): green — no pre-existing
  breakage in the workspace to isolate from this change.
- `cargo run --release -p ruvector-attn-mincut --example claim_audit_bench`:
  real measured output (full transcript in the nightly report). Headline:
  H1 avg 0.000pp (REJECT), H2 delta 0.000pp (REJECT), coherence cosine
  0.751 vs. 0.99 threshold (FAIL), candidate_A 1.3-1.7x slower than
  baseline at every grid point, candidate_B 8-26x slower.
- Root-cause probe (same binary, additional diagnostic-only `seq_len`
  points): gate fires at `seq_len=8` (3.12pp) and requires `lambda=50`
  (50x the documented max) to fire at all by `seq_len=64` (0.54pp),
  confirming the threshold-scale-mismatch explanation rather than
  insufficient lambda tuning within the documented range.

## Consequences

- **Positive:** the crate's documentation now matches its measured
  behavior; the crate gained real test coverage of a previously-untested
  code path (the min-cut step's actual contribution, as opposed to its
  mere absence of panics); the ecosystem gained a reusable
  ablation-controlled benchmark pattern (`claim_audit_bench.rs`) that other
  crates with claims tables can copy; `ruvector-coherence`'s
  `quality_check`/`compare_attention_masks` got their first real caller
  outside their own unit tests.
- **Negative:** `ruvector-attn-mincut` currently delivers no measurable
  sparsity, latency, or coherence benefit over dense softmax at the sizes
  tested; any downstream code that had silently assumed the README's
  numbers (none found in this workspace at audit time) would need to stop
  assuming them.
- **Neutral:** the crate's existing scaffolding (graph construction,
  hysteresis, witness logging) is unaffected and continues to pass its
  own tests; this ADR does not touch it.

## Alternatives

- **Fix the gate formula in this same change.** Rejected for scope: a
  correct fix requires its own hypothesis, benchmark, and acceptance
  criteria (what should `lambda` mean under a scale-aware formula?), which
  belongs in its own nightly run (see [Next research](#rejection-criteria)),
  not bundled into an audit whose job was to test the existing claim as
  specified.
- **Wire `ruvector-mincut`'s hardened global min-cut in directly.**
  Rejected for scope this run: its API targets persistent,
  `DashMap`-backed agent-memory graphs, not small ephemeral per-call
  attention-logit matrices; a correct integration needs its own design
  work, not a rushed adapter.
- **Quietly lower the acceptance threshold (e.g. to 0.5pp) so H1 would
  pass.** Rejected categorically — this harness's rules explicitly forbid
  weakening a threshold to manufacture a pass, and doing so here would
  have hidden a real, measured 0.000pp effect behind a technically-true
  but misleading "ACCEPT."

## Implementation plan

Already landed in this change:
`crates/ruvector-attn-mincut/{src/mincut.rs,src/gating.rs,src/lib.rs,Cargo.toml,examples/claim_audit_bench.rs,README.md}`.
No further implementation is part of this ADR; see
[Next research](#rejection-criteria) for the follow-on work this finding
motivates.

## API shape

Additive only:

```rust
// mincut.rs
pub fn eps_only_keep_mask(logits: &[f32], eps: f32) -> Vec<bool>;
pub fn dynamic_min_cut_best_sink(
    logits: &[f32], seq_len: usize, lambda: f32, tau: usize, eps: f32,
) -> GatingResult;

// gating.rs
pub fn compute_logits(q: &[f32], k: &[f32], d: usize, seq_len: usize) -> Vec<f32>; // was private
pub fn attn_mincut_best_sink(
    q: &[f32], k: &[f32], v: &[f32], d: usize, seq_len: usize,
    lambda: f32, tau: usize, eps: f32,
) -> AttentionOutput;
```

No existing signature changed; no feature flag was needed (both new
functions are unconditionally compiled, matching the rest of the crate).

## Feature flags

None added. The crate remains flag-free.

## Benchmark evidence

See [Evidence](#evidence) above and the full nightly report for the
complete grid (all `seq_len` x `lambda` combinations) and the raw
benchmark transcript.

## Security

No new attack surface (pure functions, no I/O, no unsafe, no new runtime
dependency — `ruvector-coherence` is a dev-dependency only). See the
nightly report's [Security](../research/nightly/2026-10-05-attn-mincut-best-sink-audit/README.md#security)
section for the supply-chain-of-trust framing of unverified claims tables
generally.

## Governance

Establishes a precedent (second consecutive nightly to "attack an existing
unverified claim," per `CLAUDE.md` Step 1) that crates with quantitative
claims tables should ship the benchmark that produced them or mark the
table as a design target rather than a measured result.

## Failure modes

The dangerous failure mode this audit surfaced is a **silent no-op**: the
gate never errors or panics when it fails to fire, it simply falls through
to the elementwise threshold, so every existing small-scale unit test
passes while the mechanism contributes nothing at realistic scale. See the
nightly report's [Performance math](../research/nightly/2026-10-05-attn-mincut-best-sink-audit/README.md#performance-math-root-cause)
for the exact dimensional-mismatch mechanism.

## Migration

None required — no in-workspace consumer of `ruvector-attn-mincut`'s
gating functions was found depending on the audited claims.

## Rollback

Trivial: this change is purely additive to the crate's public API (plus a
README correction); reverting it restores the prior unverified-but-also-
unused state. No rollback is recommended — the added tests and benchmark
are strictly more information than existed before.

## Rejection criteria

This finding (H1/H2 REJECT) would itself be overturned by: a future
nightly re-running this exact harness against a redesigned, scale-aware
gate and finding `mincut_specific_pp` averaging >= 5pp at realistic
`seq_len` without sacrificing the coherence or latency bounds. Until then,
the README's corrected (measured, not aspirational) framing stands.

## Open questions

1. What is the right scale-aware gate formula — normalize by crossing-edge
   count, by `seq_len` directly, or switch to a quantile-of-candidate-cuts
   criterion? (See nightly report's Production path.)
2. Does `ruvector-mincut` need a new small-ephemeral-graph entry point, or
   can its existing API be used directly at attention-call latency and
   throughput (far higher QPS than its current agent-memory consumers)?
3. Is there a principled way to hit the README's <1% coherence target at
   all, given that the elementwise eps-threshold alone (independent of the
   mincut step) already removes ~55-58% of attention mass in this
   benchmark's synthetic data?
