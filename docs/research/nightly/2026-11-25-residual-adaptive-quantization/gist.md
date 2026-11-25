# Does "contradiction energy" tell you which vectors are safe to compress? We tested it. It doesn't.

## Problem

Agent-memory vector stores quantize uniformly, or not at all. The
appealing idea: use a graph-structural "coherence" signal to find
memories that disagree with their neighborhood — contradictory,
surprising, boundary-case memories — and keep those at full precision
while compressing everything else hard. Two things make this tempting
right now: sheaf-Laplacian coherence scoring already exists in this
codebase (`prime-radiant`'s `CoherenceEngine`, implementing the
`E(S) = sum_e w_e |r_e|^2` residual-energy formula), and growing
agent-memory stores are increasingly memory-bound.

Two earlier attempts at a structurally similar idea — gating
quantization on mutual-kNN mutuality, and on dynamic-mincut
conductance — were both rejected, each for a distinct reason (a budget
mismatch in the first case, signal collapse on non-partitioned data in
the second). Neither tested the actual sheaf-residual formula. This is
that test.

## Hypothesis

Rank memory vectors by per-node sheaf-residual energy on a kNN
consistency graph. Give the top 25% by that ranking 8-bit quantization;
give the rest 2 bits. Compare recall against the identical two-bucket
allocation with (a) the high-bit 25% chosen at random, and (b) the
high-bit 25% chosen by a much simpler local-density/outlier score — the
control a prior rejected experiment's own open questions flagged as
untested. All three spend an identical total byte budget, so none can
win by cheating on memory.

Pre-registered acceptance: the residual-gated policy must improve
recall on a held-out "boundary vector" subset (planted, ground-truth
labeled contradiction stand-ins) by >= 0.05 over random, without
dropping overall recall by more than 0.02.

## Technical design

`ruvector-residual-quant` is a self-contained, zero-dependency Rust
crate. The residual formula implemented is a legitimate but simplified
special case of ADR-CE-001: instead of independent per-endpoint
restriction maps `rho_u, rho_v`, it uses one shared random Gaussian
projection `A_e` per edge, so `r_e = A_e(x_u - x_v)`. This is flagged in
the crate's own doc comment, *before* running the benchmark, as likely
to make the aggregate residual energy correlate with plain squared
neighbor distance by Johnson-Lindenstrauss concentration — i.e., to
collapse into a noisy version of the density control it's being raced
against. That prediction is checked directly, not assumed.

The benchmark: 1500 synthetic 48-dim vectors in 6 Gaussian clusters,
8% planted as "boundary" vectors (midpoints between two random cluster
centers, standing in for a memory that's consistent with two different
narratives). A k=10 cosine kNN graph feeds both the residual-energy and
local-density scoring. Five independent seeds. Every compression
decision is recorded in a hash-chained witness log.

## Actual implementation

- Deterministic dataset generator with a known contradiction label,
  used only for evaluation, never visible to the scoring functions.
- Brute-force kNN graph construction.
- Per-edge random restriction-map residual scoring, aggregated per node.
- A local-density/outlier control score (mean squared distance to graph
  neighbors).
- Per-vector min-max scalar quantizer, 1-8 bits, with real packed-byte
  memory accounting.
- Three allocation policies, matched budget by construction.
- A hash-chained witness log with tamper detection.
- 15 unit tests, all passing; `cargo clippy` clean.

## Actual benchmark evidence

Five seeds, `cargo run --release -p ruvector-residual-quant --bin benchmark`:

```text
Residual-energy AUC vs. boundary label: 0.467 ± 0.072   (chance = 0.5)
Local-density AUC vs. boundary label:   0.529 ± 0.089   (chance = 0.5)
Pearson(residual_score, density_score): 0.865 ± 0.013

              overall_recall@10      boundary_recall@10     bytes
random        0.240 ± 0.012          0.793 ± 0.049           43500
residual_gated 0.206 ± 0.008         0.887 ± 0.022           43500
density_gated 0.198 ± 0.010          0.906 ± 0.024           43500
```

The residual signal is statistically indistinguishable from chance at
the one thing it needed to do (AUC 0.467, where 0.5 is a coin flip), and
measured to be almost the same signal as the density control it was
raced against (r = 0.865) — exactly the collapse flagged in advance.
Both gated policies did lift boundary-subset recall over random, by
reallocating bits toward any geometrically atypical vector (which
correlates with, but isn't identical to, the planted label) — but that
reallocation comes out of the majority's budget, and the majority
dominates overall recall, so overall recall fell below the
pre-registered floor for both. The acceptance gate failed.

## Limitations

Synthetic data only. n=1500, no scale test. No delete/update/concurrency
testing. One fixed bucket shape, no sweep. The shared-restriction-map
simplification was deliberately chosen over a calibrated per-endpoint
one, which may behave differently — untested here.

## Production relevance

None, as designed — not adopted. The crate is retained as a reusable
matched-budget benchmark harness: witness log, scoring diagnostics
(AUC, correlation), and the dataset generator are all directly usable
by whoever runs the next attempt in this family.

## RuVector ecosystem implications

Three independent, honestly-matched-budget-tested rejections now exist
for "graph-structural signal gates quantization bit-width" (mutual-kNN,
mincut, sheaf-residual), each with a distinct documented root cause.
The open question this run surfaces for the *next* attempt: is the
overall-recall regression caused by *which* signal picks the minority,
or is it structural to any two-bucket hi/lo reallocation scheme at this
shape, regardless of signal? A fourth arm — plain uniform allocation at
the same matched average bit-width, with no selection at all — would
answer that directly and wasn't run here (pre-registration locked the
design before results were seen).

## Future direction

1. Add the uniform-no-selection baseline to isolate scheme vs. signal.
2. Try calibrated, per-endpoint restriction maps instead of the
   shared-per-edge simplification.
3. Sweep bucket shape with a bounded Darwin search once (1) is answered.
4. Test on real embeddings with labeled contradictions before any
   further synthetic-only iteration.

## References

ADR-CE-001/002/011 (`docs/adr/coherence-engine/`); ADR-305 and ADR-353
(prior rejected attempts in this family); ADR-134 and ADR-347 (witness
ledger conventions); `crates/ruvector-residual-quant` (this
experiment's code); `docs/adr/ADR-362-residual-adaptive-quantization.md`
(full decision record).
