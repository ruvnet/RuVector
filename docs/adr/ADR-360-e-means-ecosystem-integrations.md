# ADR-360: e-means ecosystem submodules, MetaHarness, and Cloudflare Workers

- **Status**: Accepted; local integration implemented, publication and deployment pending
- **Date**: 2026-11-11
- **Updated**: 2026-11-11 (leaf publication and dependency contract review)
- **Related**: [ADR-359](./ADR-359-e-means-shared-clustering.md)

## Outcome and acceptance contract

Make the shared e-means implementation usable for existing PQ/IVF indexes,
neural-input workflows, measured MetaHarness tuning, and Cloudflare Workers.
Integrations must invoke the real Rust/WASM kernel, preserve current training
defaults, expose explicit typed entrypoints, and run from installed npm archives.
Here, submodules mean Rust modules and npm export subpaths; no Git submodule
checkout is required.

Success requires numerical encode/decode and posting-list oracles; checked
restoration of persisted records; actual Darwin evolution with held-out scoring;
authenticated, resource-bounded Worker execution with statically imported WASM;
and real local workerd/R2 qualification. Hosted deployment, registry publication,
and live Vectorize mutation are outside this implementation request.

## Module boundaries

| Surface | Interface and responsibility |
|---|---|
| Rust `quantization` | `ProductQuantizer::fit`, checked `from_centroids`, byte `encode`/`decode`, private shape fields, validated serde restoration |
| Rust `ivf` | `IvfPartition::fit`, checked `from_model`/`from_parts`, stable offsets/indices, checked `posting` access |
| npm `/quantization` | `trainProductQuantizer`, `encodeProductQuantizer`, `decodeProductQuantizer`, validated owned JSON conversions |
| npm `/ivf` | `partition` and `postingList`, with owned typed arrays and stable input-row IDs |
| npm `/runtime` | Node-free `init(source)`, `fit`, `predict`; requires explicit bytes or a precompiled module |
| npm `/metaharness` | `createClusteringEvaluator` and bounded Node-only `tuneClustering`, using the actual Darwin numeric API |
| npm `/cloudflare` | Stateless Worker factory and explicit R2/Vectorize binding helpers |
| npm `/cloudflare/build` | Node-only local static-WASM builder, using esbuild; never deploys |
| npm `/wasm` | Raw artifact export for Wrangler's static WebAssembly import |

PQ decomposition and posting construction wrap the shared fitter and nearest
assignment. There is no second JavaScript clustering optimizer. All PQ codes are
row-major, one u8 per subspace; centroids are subspace-major. Rust and JavaScript
use the same numeric layout, but JS JSON has an explicit schema envelope. Remove
that envelope before native serde deserialization, or construct the native
quantizer with its checked `from_centroids` API. Do not assume arbitrary JSON
records from the two APIs are interchangeable.

## Dependent integrations (separate promotion)

- `ruvector-core::ProductQuantized::train_e_means` remains available from ADR-359.
- `ruvector-pq-search` adds `PqCodebook::train_e_means`,
  `IvfPqIndex::new_e_means`, and checked insertion/query wrappers. Existing PQ
  layouts and ADC search behavior remain in place.
- `ruvector-rairs` adds explicit training to `IvfFlat`, `RairsStrict`, and
  `RairsSeil`. Invalid training leaves state intact. Successful explicit
  retraining resets inserted vectors and IDs, as documented by these methods.
- `@ruvector/core`, `ruvector`, and `@ruvector/all` ship `clustering.mjs` and
  `clustering/{quantization,ivf,metaharness,cloudflare}.mjs`, with `.d.mts`
  declarations. Their existing package subpaths remain accessible.
- The current `@ruvector/wasm` source at `crates/ruvector-wasm` exposes
  `/clustering` and `/clustering/{quantization,ivf,metaharness,cloudflare}`.
  Historical same-name manifest copies are not release sources for this change.
- `ruv-fann` and `@ruv-fann/integrations` expose corresponding clustering
  subpaths. Existing WSP Cloudflare and neural MetaHarness entrypoints retain
  their own meanings.

Each npm consumer depends on exact `@ruvector/e-means` 0.1.1. Static
clustering re-exports require it as a normal dependency, including in
`@ruvector/core`; the package's Node 20 runtime requirement applies to these
entrypoints. The new modules do not eagerly initialize WASM or import Darwin
during ordinary SDK imports. These consumer changes are outside the leaf
publication commit and are promoted after the shared registries are verified.

Shared modules use f64 assignment arithmetic. Existing downstream ADC/scanning
APIs retain their existing f32 precision and legacy validation behavior; this
integration does not certify those algorithms for extreme f32 magnitudes or
silently replace their defaults.

## MetaHarness control flow and evidence

The evaluator synchronously copies training/holdout arrays and snapshots the
baseline, domain, and budgets before its first await. Holdout rows must be
nonempty and nonoverlapping with training rows. Each dispatch validates the
genome, reserves its work allowance, executes real WASM training, and separately
predicts and scores the holdout. Higher-is-better `primary = 1/(1+holdoutMse)`
matches Darwin's score contract. Records include actual training/holdout MSE,
fit/prediction timings, selected configuration, and resource estimates.

Only one evaluation runs at a time, with at most 32 admitted dispatches,
64 MiB of input snapshots, and a caller-selected allowance capped at 1e9
coordinate-scale work units. Admission includes centroid updates, tail batches,
and quadratic donor selection. Units are conservative estimates, not measured
instructions or a process-memory limit. Deadlines are cooperative; synchronous
fits and arbitrary host operations cannot be preempted.

`tuneClustering` dynamically loads `@metaharness/darwin` 0.13.0 or 0.13.1, validates
the installed identity/API, and calls its real `evolveNumeric` at concurrency 1.
It checks worst-case search cost before execution and matches returned archive
records to observed evaluations. Regressed/unmeasured winners cannot replace
the baseline recommendation. A temporary archive is removed in `finally`.
Receipts and returned scores are independent copies; `sourceApplied` is always
false. The holdout guides selection and is not an independent generalization
test. SDK version checking is metadata validation, not binary attestation.

The SDK is an optional peer, loaded only when tuning is invoked. Package tests
exercise both an installation without that peer and actual evolution with the
pinned peer installed.

## Cloudflare execution and authority boundaries

Wrangler imports `.wasm` as `WebAssembly.Module`; the Worker passes that static
module to the Node-free runtime. No Node filesystem loader, downloaded model,
runtime source compilation, or `instantiateStreaming` path is needed.

The Worker exposes public `/health` and authenticated POST `/fit` and `/predict`.
Operations fail closed when no token is configured, except an explicit trusted
local `allowUnauthenticated` opt-in. The example reads `E_MEANS_BEARER_TOKEN`
from the caller's binding configuration; credentials are not packaged.

Before entering WASM, the handler reads bounded UTF-8 JSON with a stream deadline
and validates exact fields, finite float32 values, shapes, options, and work.
Absolute caps include 1 MiB request bodies, 262,144 elements, 16,384 rows,
1,536 dimensions, 256 clusters, 32 epochs, and 1e8 estimated work units; defaults
are smaller. The work estimate includes updates and donor revival. These are
per-call bounds, not a guarantee against aggregate isolate exhaustion or a hard
CPU-time cap. Request vectors/models are not kept in global mutable state, and
HTTP responses expose no unexpected stack traces.

Storage operations are explicit helper calls on caller-owned bindings. HTTP
fitting does not automatically store or export anything:

- R2 stores only bounded centroid codebooks, not source vectors or assignments.
  Writes require an acknowledgement; reads validate streamed bytes and schema
  before constructing owned numeric arrays. PQ schema records are rejected
  because their subspace-major layout is not a single centroid set.
- Vectorize V2 exports validate the entire snapshot and identifier/namespace
  byte lengths before the first upsert. Batches are limited to 1,000 vectors
  and dimensions to 1,536. Every acknowledgement must carry a mutation ID.
  An acknowledgement does not prove immediate query visibility.
- Export failures report prior acknowledged rows/mutation IDs. The failed batch
  may also have applied; no transaction, rollback, retry, or deletion is implied.
  Use a versioned prefix/namespace for revised codebooks, since upserts do not
  remove stale IDs from an older, larger codebook.

Official Cloudflare references: [WASM bundling](https://developers.cloudflare.com/workers/wrangler/bundling/),
[Vectorize limits](https://developers.cloudflare.com/vectorize/platform/limits/),
[asynchronous mutations](https://developers.cloudflare.com/vectorize/get-started/intro/).
The connected Cloudflare documentation tool was used to verify these contracts.

## Packaging, validation, and rollback

Use the isolated package lock in `npm/packages/e-means` for integration tooling.
Miniflare 4.20260730.0 and esbuild 0.28.2 are development-only. Scoped overrides
select patched sharp 0.35.5 and undici 7.30.1; re-run workerd tests and the
lockfile-aware audit when updating them. The workspace lock records this tool
graph beneath the e-means workspace without upgrading unrelated dependency
versions. Whole-workspace installation remains separately constrained by the
repository's existing platform-specific workspace packages.

| Requirement | Check |
|---|---|
| Native layout, persisted-state validation, real index consumers | `cargo test -p ruvector-e-means --features serde -p ruvector-pq-search -p ruvector-rairs` |
| Existing shared core and optional WASM bridge | `cargo test -p ruvector-e-means --all-features` |
| JS numerical, ownership, holdout, budget and HTTP boundaries | `npm test` in the shared package |
| Installed package without/with actual optional Darwin SDK | `npm run test:consumer` |
| New entrypoints from actual consumer archives | `npm run test:ecosystem` |
| Real static WASM and R2 in local workerd | `npm run test:cloudflare` |
| Public declarations | `npm run typecheck` and installed consumer checks |
| Newly selected tool dependency graph | `npm audit --package-lock-only --workspaces=false` |

Consumer archive tests qualify the new clustering subpaths. Their unrelated
legacy root/prepack release builds are not qualified by this task. Vectorize
has binding-contract tests, not a claimed hosted index write. No Cloudflare
deployment or registry publication is inferred from passing local checks.

Release the shared crate/package first, bump affected consumer release versions,
and verify registry installs before publishing consumers. Local ruv-FANN Rust
checks use the explicit Cargo patch from ADR-359. Rollback means stop using the
new feature methods or subpaths; existing codebooks, neural APIs, and search
defaults require no migration. Storage helpers never delete external resources.

Initial integration qualification passed 50 targeted Rust tests, 44 shared npm tests, 8 FANN
facade tests, and 3 Rust FANN adapter tests, plus strict declarations, Clippy,
archive verification and the packaged Rust 1.81 check. Installed consumers
exercised the optional-SDK absence path, actual Darwin evolution, all four
RuVector facade packages, and static-WASM/R2 execution in real local workerd.
The isolated integration-tool lockfile audit reported zero known advisories
after the scoped overrides. [Qualification evidence](../research/e-means/integrations-qualification.json)
records versions, scope limits, and current lock/artifact digests.
