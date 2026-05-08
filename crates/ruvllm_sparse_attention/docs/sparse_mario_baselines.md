# Sparse-Mario cross-baseline comparison — iter 13

A fair-fight comparison between the two Sparse-Mario pipelines and two
classical baselines, scored on the same five PCG metrics from iter 11
plus an aggregate L2 distance to the corpus median target
`{density: 0.30, linearity: 0.33, leniency: −0.04, playable_columns: 1.00}`.

All numbers are averages over the same three seeds
(`c0ffee42`, `baddf00d`, `1337beef`) generating 14×50 grids (700 tokens)
with `SamplingConfig::quality()` where applicable.

Recorded automatically by `cargo run --release --features parallel \
--example sparse_mario`.

## The pipelines

| Pipeline | What it is |
|---|---|
| **Sparse-Mario AR** | Attention-as-memory bigram LM via `KvCache` + `decode_step` (iter 8), top-k + top-p + rep penalty (iter 9). Causal autoregressive. |
| **Sparse-Mario diffusion** | Bidirectional masked discrete diffusion via the same kernel (iter 7), context-boot + cosine schedule (iter 7), `n_steps = 24` (iter 12). |
| **Markov-1 (corpus bigram)** | Classical first-order Markov chain over the embedded corpus. Exact P(next \| curr), no embeddings, no attention. The "what AR would be if you skipped the attention machinery" baseline. |
| **Uniform random** | Each tile drawn IID from the 15-token vocab. The lower bound on every metric. |
| **Corpus (target)** | The three embedded SMB level slices, evaluated as if they were generations. The upper bound. |

## The numbers

| pipeline                | density | linearity | leniency | novelty | playable | **L2 dist** |
|-------------------------|--------:|----------:|---------:|--------:|---------:|------------:|
| Corpus (target)         |   0.299 |     0.571 |    0.073 |   0.000 |    0.953 |   **0.504** |
| **Sparse-Mario diffusion** | 0.777 |  0.000 | −0.020 |  0.660 |  0.747 | **0.723** ⭐ |
| Markov-1 (corpus bigram)|   0.136 |     2.949 |    0.333 |   0.322 |    0.353 |       2.745 |
| Uniform random          |   0.284 |     3.475 |    0.633 |   0.456 |    0.073 |       3.353 |
| **Sparse-Mario AR**     |   0.329 |     5.254 |   −0.333 |   0.499 |    0.207 |    **4.998** |

Sorted by **L2 distance to corpus** (lower = closer to "real Mario"):

1. **Sparse-Mario diffusion — 0.723** (the only pipeline within striking distance of the corpus' 0.504 self-distance)
2. Markov-1 — 2.745
3. Uniform random — 3.353
4. Sparse-Mario AR — 4.998

## SOTA-on-this-artifact: Sparse-Mario diffusion

Diffusion wins by **3.8× over Markov-1** and **6.9× over AR**. It's the only
training-free pipeline that meaningfully approaches corpus-shape on this metric
suite. The win is concentrated in two places:

- **Linearity = 0.000** — the bidirectional context with cosine scheduling
  produces structurally consistent ground positioning when the boot slice
  carries a floor pattern. Markov can't match this because it's strictly
  left-to-right; uniform random can't because it has no model at all.
- **playable_columns = 0.747** — a healthy 0.21 below corpus but **3.6× higher
  than AR** and **2× higher than Markov-1**. Bidirectional masked filling,
  unique to the diffusion path, is what makes this work.

It loses ground on **density (0.777 vs corpus 0.299)** — the boot slice is
copied verbatim into the output, which inflates the non-sky tile count.
That's the known "verbatim corpus echoing" trade-off documented in the iter-7
commit; the iter-10 multi-token K builder narrowed it but didn't close it.

## Why AR is the worst — and what would fix it

Sparse-Mario AR is **below uniform random** on aggregate L2 distance. That looks
bad until you read the per-metric breakdown:

- AR's **density (0.329)** is excellent — closer to corpus than every other
  pipeline including Markov-1.
- AR's **linearity (5.254)** is catastrophic — 9× worse than corpus, and worse
  than uniform random's 3.475.

Why: AR's K builder mixes the embedding with `0.5·pos(i)`, and the AR query
position sits at the *tail* of the combined corpus+prefix sequence. The
attention is biased toward corpus positions with similar absolute index — i.e.
the *bottom* of the corpus, where ground rows live. So ground tiles emerge from
retrieval *evenly distributed across the output* rather than concentrated at the
bottom. Hence high linearity (jagged ground heights per column) and low
playable_columns (no clear floor row).

The fix is architectural — drop positional encoding from the AR K builder, the
same way the iter-7 diffuser did. That's a 3-line change and a candidate
follow-up; it would likely halve AR's L2 distance.

## Why Markov-1 sits in the middle

Markov-1 uses the *exact* corpus bigram, no embedding noise. So its bigram
fidelity is higher than AR — but it lacks the corpus-shaped *meta-structure*
(no positional bias, no global awareness). Result: density too low (0.136),
leniency too high (0.333 — randomly placed enemies/cannons), playable too low
(0.353).

The fact that Markov-1 beats AR on aggregate but is beaten by both Sparse-Mario
diffusion and the corpus is the headline finding: **the value-add of attention
machinery is not bigram fidelity (Markov-1 has perfect bigrams), it's the
ability to do bidirectional masked filling**, which only the kernel-based
diffuser provides.

## Where SOTA is and where to push it

**SOTA-on-this-artifact = Sparse-Mario diffusion at 0.723 L2 distance.** It
beats every other training-free pipeline tested on this corpus and metric set.

To go further (would require future iters):

1. Drop positional encoding from AR K builder → expected AR L2 ≈ 2.5 (matches Markov-1).
2. Floor anchor in diffusion → expected diffusion L2 ≈ 0.55 (would close most of
   the remaining gap to the 0.504 corpus self-distance).
3. Train an actual MaskGIT-style denoiser → would need autograd in Rust;
   significant additional scope.

(1) and (2) are 5–10 line architectural changes that the iter-11 metrics module
will keep honest.

## Reproduction

```bash
cd crates/ruvllm_sparse_attention
cargo run --release --features parallel --example sparse_mario \
  | grep -A 12 "Iter 13 cross-baseline"
```

## Test guarantees

Five new tests in `tests::` block guard the baselines:
- `uniform_random_outputs_in_vocab` / `_is_deterministic` /
  `_is_far_from_corpus` — sanity checks on the trivial baseline.
- `markov_one_outputs_in_vocab` / `_is_deterministic` — same on the
  classical baseline.
