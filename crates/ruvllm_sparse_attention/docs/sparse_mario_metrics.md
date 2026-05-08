# Sparse-Mario PCG metrics — iter 11 baseline

Quantitative descriptors for the levels the Sparse-Mario example produces.
Five metrics from the standard PCG / MarioGAN evaluation literature, computed on
700-token (14×50) outputs from both pipelines plus the embedded VGLC corpus
levels for reference.

Recorded automatically by `cargo run --release --features parallel \
--example sparse_mario`.

## Metrics

| Metric | Definition | Higher = |
|---|---|---|
| **density** | non-sky / total tiles | denser, more structures |
| **linearity** | std-dev of topmost-ground row across columns | jaggier ground profile |
| **leniency** | (hostile + gaps − friendly) / cols | harder level |
| **novelty** | min normalised Hamming distance to any same-shape corpus window | further from corpus |
| **playable_columns** | fraction of columns with a ground tile in the lower third | walkable surface |

Hostile = enemies + cannons. Friendly = ?-blocks + coins. Gap = column with no `X`
anywhere.

## Baseline numbers

Three seeds × {AR, Diffusion}, plus the three embedded corpus slices for reference.
`SamplingConfig::quality()` (top_k=5, top_p=0.90, rep_penalty=1.7, window=24).

| config              | density | linearity | leniency | novelty | playable |
|---------------------|--------:|----------:|---------:|--------:|---------:|
| AR     `c0ffee42`   |   0.346 |     4.937 |   −0.480 |   0.491 |    0.180 |
| AR     `baddf00d`   |   0.317 |     5.668 |   −0.260 |   0.497 |    0.140 |
| AR     `1337beef`   |   0.324 |     5.158 |   −0.260 |   0.507 |    0.300 |
| DIFF   `c0ffee42`   |   0.390 |     0.000 |    0.000 |   0.619 |    0.000 |
| DIFF   `baddf00d`   |   0.709 |     0.000 |   −0.020 |   0.796 |    0.960 |
| DIFF   `1337beef`   |   0.864 |     0.000 |   −0.040 |   0.593 |    1.000 |
| CORPUS slice 0      |   0.239 |     0.000 |   −0.040 |   0.000 |    1.000 |
| CORPUS slice 1      |   0.357 |     0.325 |   −0.040 |   0.000 |    1.000 |
| CORPUS slice 2      |   0.301 |     1.388 |    0.300 |   0.000 |    0.860 |

## Reading the table

**Density.** AR sits in [0.32, 0.35] which matches the corpus band [0.24, 0.36]
nicely — the bigram retriever produces a sky/structure mix close to the training
distribution. Diffusion is volatile (0.39 → 0.86) because the boot slice's
content dominates and the rest of the grid fills with whatever the bidirectional
retrieval finds in similar contexts.

**Linearity.** This is the headline AR weakness. AR scores 4.9–5.7, **5–6× higher
than the corpus' 0.0–1.4**. Reason: AR places `X` tiles wherever the bigram chain
favours them, not concentrated near the bottom — so the "topmost ground per
column" lands all over the grid. Diffusion stays at 0.0 across all seeds because
when a column does have ground, it's typically at a uniform height (the boot
slice's row position propagates).

**Leniency.** Corpus is balanced (−0.04 to 0.30). AR overshoots toward
"friendly" (−0.48 to −0.26): the retrieval samples coins and ?-blocks more
often than a real level would because the rep-penalty pushes away from over-using
sky/ground. Diffusion lands closer to corpus.

**Novelty.** Both pipelines produce levels that are 49–80% different from any
corpus window — well clear of byte-copying. Diffusion has higher novelty (0.59–
0.80) than AR (0.49–0.51), which reads as: AR follows local bigram context more
faithfully (closer to corpus locally), diffusion generates structurally novel
combinations (boot + bidirectional retrieval mixes things up).

**Playable columns.** This is the headline diffusion weakness, **and the
single number to optimise next**. AR's 14–30% means most columns lack a ground
tile in the lower third — Mario falls through. Corpus is 86–100%. Diffusion
ranges 0–100% with no middle ground (boot slice placement decides the outcome).

## Implications for iter 12

The metric that's farthest from corpus across both pipelines is **linearity** for
AR and **playable_columns** for diffusion. Iter 12 should chase those:

- **AR**: bias the K builder toward positional encoding only at the
  bottom rows so retrieval prefers ground-region patterns there. Or post-process:
  pin the bottom row to `X` after generation.
- **Diffusion**: bias the boot slice toward corpus rows that contain the floor
  pattern. Or seed multiple bottom-aligned ground tiles before the cosine schedule
  starts. This is a small architectural change — boot strategy, not denoiser
  retraining.

Both fixes are 5–10 line tweaks. The metrics module shipped in iter 11 will
keep them honest — any change must move `playable_columns` toward 0.86+ and
`linearity` toward 1.0–2.0 without crashing density / novelty.

## How to reproduce

```bash
cd crates/ruvllm_sparse_attention
cargo run --release --features parallel --example sparse_mario \
  | grep -A 30 "PCG metrics baseline"
```

The four unit tests under `tests::metrics_*` in `examples/sparse_mario.rs`
guard each metric's definition (sky → density 0, corpus slice → novelty 0,
half-ground → density 0.5, flat floor → linearity 0).
