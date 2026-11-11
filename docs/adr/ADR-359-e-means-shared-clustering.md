# ADR-359: Shared e-means clustering for RuVector and ruv-FANN

- **Status**: Accepted; local implementation, registry publication pending
- **Date**: 2026-11-11
- **Updated**: 2026-11-11 (split shared release qualification)
- **Source**: Bessa et al., [e-means: outperforming k-means (in one-tenth the epochs)](https://arxiv.org/abs/2611.09501), [version 1](https://arxiv.org/html/2611.09501v1), 2026-11-06
- **Related**: [ruv-FANN ADR-013](https://github.com/ruvnet/ruv-FANN/blob/main/docs/adr/ADR-013-e-means-clustering.md)

## Release qualification

The shared package is promoted separately from dependent integrations; see
[the protected leaf release protocol](../e-means-release.md). The browser route
was corrected and its missing-route mutation fails. PQ/IVF validators inspect
typed-array storage without caller-defined iterators. Local qualification of
the archived-source WASM build passed native/MSRV, npm, installed-consumer,
workerd and real Chromium checks; registry publication remains subject to the
exact-source protected workflow and fresh public consumers.

## Context and scope

RuVector trains product-quantization codebooks with Lloyd k-means. ruv-FANN
consumers also need unsupervised clustering of input embeddings without changing
their supervised labels or neural optimizers. Maintaining independent clustering
implementations for Rust and JavaScript would make numerical fixes and paper
conformance diverge.

The paper proposes mini-batch EMA sufficient statistics plus a particular
initialization, schedule, cluster revival, and finalization recipe. Its reported
GPU results motivate implementation, but do not establish performance of this CPU
implementation. Its fitted hyperparameters are less reliable for tiny and
low-dimensional data, including many PQ subspaces. This release therefore adds
explicit e-means APIs while preserving existing training defaults and persisted
codebook formats.

## Decision and contracts

Create `ruvector-e-means` 0.1.0 as a separately publishable, MIT-licensed crate in
`crates/ruvector-e-means`, registered in the workspace. The native core requires
Rust 1.81, has no default features, and has no dependency on either consumer.
`serde` adds config/model serialization. `wasm` adds a thin wasm-bindgen bridge.
Pin wasm-bindgen and its build CLI to the existing workspace version 0.2.125 so
this feature does not upgrade other packages' WASM dependency stack.

The public core API is `fit(flat_f32_rows, dimensions, &Config) -> Result<Model>`
and exact `predict`. `Config` exposes clusters, epochs (default 3), optional batch
and peak-step overrides, seed (default 42), and finalization. `Model` returns flat
centroids, assignments, dimensions, squared-L2 MSE per vector, resolved batch and
step, update count, and number of revived clusters. Assignments are recomputed
against the returned centroids in original row order. Public/deserialized
codebooks are validated when passed to prediction.

Implement the paper's primary recipe:

1. Seeded ChaCha8 sampling without replacement for initial centroids; initialize
   counts to `1e-10` and sums to count times centroid (equation 4).
2. Resolve `B = 0.035 k^0.13 n^0.88 T^0.81`, divided by eight when
   `n > 2,000,000`, rounded to the nearest power of two and capped at `n`.
   Resolve `alpha0 = min(0.824 k^-0.46 B^0.90 n^-0.45 T^-0.11, 1)` using
   the effective batch size (equations 6–7, table 2).
3. Shuffle each epoch. Assign rows by exact squared L2 and accumulate per-cluster
   batch counts/sums. Update every cluster's statistics with equation 2 and
   the cosine step from equation 5, including its 0.1 floor and `t/M` denominator.
4. Every five batches revive counts below 0.1 times the mean count. Draw live
   donors proportional to count, without replacement. Split count in half and
   perturb child coordinates by alternating `1 +/- 1/128` (Appendix H).
5. Finalize using the mean of assignments recorded during the last epoch, or
   return the main EMA centroids. Recompute nearest assignments and MSE in a
   final exact pass.

Use f64 sums, counts, and distances, and f32 codebooks. Retain a negligible
pseudo-count for a temporarily empty cluster when a step of one would otherwise
erase its state. Clamp extreme split coordinates to finite f32. Final clusters
without recorded rows retain their EMA centroid. If dead slots outnumber live
donors, remaining slots wait until a later revival pass.

Two deliberate execution differences from the printed algorithm are explicit:
include short tail batches rather than dropping rows, with
`M = T * ceil(n/B)`; and recompute final nearest assignments/MSE rather than
reporting the loss of stale last-epoch assignments. Last-epoch mean is an average
of assignments made while centroids moved, not a separate Lloyd iteration, so we
do not promise monotonic improvement over the final EMA model.

`@ruvector/e-means` 0.1.0 in `npm/packages/e-means` ships that same Rust core as
portable WebAssembly, typed ESM APIs, and the matching generated loader.
`await init()` loads bundled bytes in Node and fetches them in a browser;
`fit(Float32Array, dimensions, options)` and `predict` run synchronously after
initialization. No JavaScript numerical fallback or Rust installation is needed
by an installed consumer. Training calls are bounded by caller-selected sizes
and epochs and can block the caller; applications should use workers for long
runs. The Rust crate itself does not spawn threads or fetch datasets.

## Consumer integration and release order

- RuVector core directly depends on the new crate and exposes
  `ProductQuantized::train_e_means(vectors, num_subspaces, &Config)`. It trains
  each subspace with the shared implementation and retains u8 codes and the
  current codebook serialization. f64 encode comparisons preserve nearest
  ordering for large finite f32 coordinates.
- The main `ruvector` npm package depends on exact `@ruvector/e-means` 0.1.0 and
  ships `ruvector/clustering.mjs` with matching ESM declarations. This explicit
  subpath avoids eagerly initializing WASM or restricting legacy package
  subpaths with a new exports map.
- ruv-FANN has an optional `clustering` Cargo feature with a version-only
  dependency on `ruvector-e-means` 0.1.0, re-exported types/functions, and
  `TrainingData<f32>::cluster_inputs`. It clusters inputs without mutating inputs
  or outputs. Its npm facade depends on exact `@ruvector/e-means` 0.1.0 and
  exposes `ruv-fann/clustering` without changing the root neural API.

Publish the new crate and npm package before releasing dependent consumer
versions. Before registry publication, qualify ruv-FANN with an explicit local
Cargo patch and install the packed npm dependency in isolated consumer tests:

```sh
cd ../ruv-FANN
cargo test --features clustering --test clustering_tests \
  --config 'patch.crates-io.ruvector-e-means.path="../ruvector/crates/ruvector-e-means"'
```

No sibling checkout path is committed in ruv-FANN's published dependency
contract. Its version-only dependency cannot resolve from a fresh registry
checkout until the new crate is published, even with the feature disabled.
Local lockfile qualification with a patch does not prove registry availability.
Do not claim publication based on a successful local build or archive check.

## Alternatives and consequences

- Copying the algorithm into both projects and npm would multiply maintenance
  and correctness risk. A single Rust implementation compiled to WASM keeps
  numerical behavior shared.
- Replacing all Lloyd defaults immediately risks quality regressions for small
  PQ subspaces outside the scaling laws' fitted domain. Explicit APIs allow
  evaluation with real embeddings and consumer-specific recall/latency gates.
- A GPU/PyTorch/Faiss runtime dependency would add substantial deployment cost.
  This CPU reference and portable WASM establish the contract first. GPU,
  low-precision, distributed/IVF assignment, slow-EMA finalization, arbitrary
  streaming updates, and wholesale replacement of all clustering call sites are
  outside this release.

The algorithm stores O(n + kd) scratch state and performs O(Tnkd) exact-distance
work plus a final O(nkd) scoring pass. Training is synchronous. Seeded shuffles
are stable across native/WASM; platform transcendental routines can introduce
small floating-point differences, so cross-target numerical checks use an
appropriate tolerance rather than a universal bit-identity claim.

## Acceptance and validation

| Requirement | Executable evidence |
|---|---|
| Paper coefficients, cosine schedule, weighted mass-preserving revival | `cargo test -p ruvector-e-means` |
| Mean/MSE oracle, all tail rows visited, reproducibility, separated clouds | crate `tests/clustering.rs` |
| Empty/ragged/nonfinite shapes, invalid configs, overflow, duplicate/zero/extreme inputs | crate and consumer Rust tests; npm WASM tests |
| PQ representation, serialization, all 256 u8 codes, extreme nearest comparison | `cargo test -p ruvector-core --no-default-features --test e_means_quantization` |
| ruv-FANN adapter and unchanged supervised state | patched `clustering_tests` plus existing library tests |
| Same Rust core used through typed npm facade | `npm test` and `npm run typecheck` in the new package; consumer facade tests |
| Complete installable npm artifact without repository paths | `npm run test:consumer` in the new package |
| Browser default fetch/initialization | new package's headless browser consumer test |

Build WASM with `npm run build` in the new package, using the pinned CLI. Its
prepack check executes the shipped artifact. Review and run the targeted checks
before release; this ADR does not certify the entire monorepo, GPU speedups, or
universal quality superiority. Rollback means use the retained Lloyd/neural APIs
or stop importing the clustering subpaths; existing persisted codebooks need no
migration.

### Initial local qualification on 2026-11-11 (historical)

- The shared crate passed 10 native tests with all features and strict Clippy.
  `cargo package --allow-dirty` built and verified the archive. Its native core
  also passed `cargo +1.81.0 check --locked` against the packaged manifest.
- Actual RuVector core passed 6 new integration tests and 36 existing
  quantization tests. ruv-FANN passed 3 adapter tests and 207 existing library
  tests with the local dependency patch.
- The final wasm-bindgen 0.2.125 artifact passed 9 Node numerical/API tests,
  offline installed-archive qualification, strict TypeScript, and both consumer
  facade checks. Packed ruv-FANN was also installed with the packed dependency
  and exercised through its public subpath.
- Browser execution remains unverified here: the browser consumer timed out,
  and independent Chromium `about:blank` startup also timed out, including a
  single-process baseline. The checked-in browser test fails on that timeout;
  no skip or simulated success is counted as a pass. Re-run it in a working
  Chromium environment before claiming browser qualification.

These are focused local checks. Neither registry publication nor full-monorepo
qualification was performed.

## Optimization qualification

Exact partial-distance pruning checks candidates every eight coordinates for
dimensions above 16, while small codebooks retain scalar distance evaluation.
Zero distance ends the search. These changes retain the scalar addition order
and lowest-index ties. Independent exhaustive-oracle tests cover diffuse and
separated inputs, extreme/subnormal coordinates, near ties, and dimensions with
incomplete final blocks. The optimization does not add approximate assignment
or change seeded shuffles, sufficient statistics, or finalizers.

Training uses sliced centroid rows, reuses dead/donor scratch capacity, skips
donor-list construction when no cluster is dead, and omits last-epoch statistics
for EMA finalization. PQ encoding computes each candidate distance once. The
WASM wrapper keeps JSON for the small config and returns vectors as owned typed
arrays, releasing its temporary Rust handle in `finally`. Tests check repeated
fits, error-path cleanup, and output survival after WASM memory growth.

The author's run on an AMD EPYC 9V74 environment, with five interleaved native
repetitions, recorded these median fit times for last-epoch-mean finalization.
The "Before" build is a pre-optimization version of this crate that is not in
the repository, so these ratios cannot be re-run from the tree:

| Synthetic input (n × d, clusters) | Before | After | Speedup |
|---|---:|---:|---:|
| Clustered embeddings (4096 × 128, 64) | 130.05 ms | 70.57 ms | 1.84× |
| Diffuse embeddings (2048 × 768, 32) | 203.61 ms | 177.48 ms | 1.15× |
| Duplicate/zero rows (2048 × 32, 64) | 13.35 ms | 2.53 ms | 5.29× |

All model fields and predictions matched the baseline bit for bit in eight
native scenarios. A 21-repeat 8D PQ follow-up measured 51.25→52.68 ms with
last-epoch mean and 50.28→50.83 ms with EMA; no PQ speedup is claimed. EMA mode
on the 768D fixture reduced peak additional requested heap from 721,664 to
525,312 bytes. Last-epoch-mean mode may retain a little more scratch capacity;
allocation-event reduction is distinct from peak-memory reduction.

In the same author run, Node 24 WASM median-of-seven fit times were 36.19→15.00 ms (2.41×) for an
8192 × 64, 16-cluster fixture and 84.09→61.57 ms (1.37×) for one million
scalar rows with one cluster. Both match baseline centroids, assignments and
MSE. Comparing JSON and typed output using the same optimized kernel measured
75.97→58.41 ms on the million-row fixture; this includes the whole fit and
materialization, not just conversion time.

Independent review reproduction (2026-11-11, ruvultra, Rust 1.99.1 with the
workspace release profile for both sides, one pinned core, five interleaved
rounds of five repetitions). The baseline is this tree with partial-distance
pruning disabled (`nearest` always takes the scalar path), so it isolates the
pruning change only. It does not include the other optimizations above:

| Synthetic input (n × d, clusters) | Pruning off | Pruning on | Speedup |
|---|---:|---:|---:|
| Clustered embeddings (4096 × 128, 64) | 57.55 ms | 36.40 ms | 1.58× |
| Diffuse embeddings (2048 × 768, 32) | 97.95 ms | 98.30 ms | 1.00× |
| 8D PQ (8192 × 8, 256) | 28.83 ms | 28.80 ms | 1.00× |
| Duplicate/zero rows (2048 × 32, 64) | 3.65 ms | 1.44 ms | 2.54× |

The npm `benchmark` 8192 × 64, 16-cluster fixture measured 17.0→7.5 ms
(2.26×) against a pruning-off WASM build. All eight native digests
(centroid bits and assignments) equal both the pruning-off build and the
digests recorded in the measurement artifact, and the WASM fixtures matched
with zero label or centroid differences. The shipped `.wasm` rebuilds
byte-for-byte from this source with Rust 1.99.1 and wasm-bindgen 0.2.125
(`--remap-path-prefix` set to the author's home directory). Treat the author
ratios above as that environment's results against an earlier build, and the
reproduction as the checked effect of pruning.

The [measurement artifact](../research/e-means/performance-2026-11-11.json)
records shapes, options, raw native samples, environment, allocation definitions,
and source/artifact checksums. The native benchmark example and npm `benchmark`
command reproduce current measurements. These synthetic CPU/WASM results do not
establish speedups on arbitrary datasets or the paper's GPU benchmarks. At that initial checkpoint browser
qualification and registry publication remained pending. The release qualification
section above records later browser execution; publication still requires the protected workflow.

Initial post-optimization validation passed 14 shared-crate tests, 6 PQ integration
tests, 20 existing core quantization tests, 207 ruv-FANN library tests and its
3 adapter tests. The npm runtime passed 13 numerical/ownership tests, installed
archive checks, both consumer facades and strict TypeScript. Strict Clippy,
crate archive verification, and the packaged native Rust 1.81 check also passed.
