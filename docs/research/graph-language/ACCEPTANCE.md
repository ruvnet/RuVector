# RuGraph v0.1 acceptance evidence

Validated on 2026-10-16 against repository base
`46d816ddc4c2a8a06cd3613534a79b1d557d858b`.
This report covers the version 1 bounded language in [SPEC.md](SPEC.md).

## Result

Local acceptance passed: **90 tests**, formatting, strict targeted Clippy, ADR
number validation, packed npm client execution with TypeScript declarations,
and installed Python wheel execution and replay. No tests were skipped.

The implementation includes the text and JSON frontends, type/effect checks,
deterministic runtime, evidence and rule semantics, replay receipts, real
RuVector graph/KGE adapters, offline model lifecycle, CLI, process SDKs,
examples, eight ADRs and a scoped CI workflow. Package publication and default
branch merge are separate release actions.

## Reproduction

Install Rust 1.90.0 with rustfmt and Clippy, Node.js 24, Python 3.12, setuptools
and wheel. The package smoke script downloads TypeScript 5.9.3 through npm;
it does not publish a package. Run from the repository root:

```bash
bash scripts/test-graph-language.sh
bash scripts/smoke-graph-language-packages.sh
cargo +1.90.0 run --release --locked \
  --manifest-path crates/ruvector-language-native/Cargo.toml --example native_bench
```

The two acceptance scripts completed with exit code 0. Exact local versions:
rustc 1.90.0 (`1159e78c4`, 2025-09-14), Node.js 24.19.0, Python 3.12.14,
setuptools 84.0.0, wheel 0.48.0. Target: Linux x86_64, glibc 2.39.
The checked in workflow uses the same Rust, Python and Node major versions.
The dedicated remote acceptance workflow also passed at commit
`01d961f2d48528b3cf320d335adefa0f1245dc79`:
[Graph language run 38899871818](https://github.com/ruvnet/RuVector/actions/runs/38899871818).
Broader repository checks are separate from that successful feature gate.

## Requirement trace

| Requirement | Executed evidence | Result |
|---|---|---|
| Text/JSON parity; strict schema; limits; static types and effects | `tests/frontend.rs`, 16 tests | Pass |
| Four evidence states; explicit negative support; temporal freshness | `tests/runtime.rs`, part of 15 tests | Pass |
| Signed conjunction, chained derivation, source lineage, cycles | `tests/runtime.rs` | Pass |
| Deletion, expiry and alternate witness recomputation | `tests/runtime.rs` | Pass |
| Traversal and exact cosine ranking with deterministic ties | `tests/runtime.rs` | Pass |
| Host capability denial, budget exhaustion, provenance bounds | `tests/runtime.rs` | Pass |
| Prediction/evidence separation and provider response checks | Frontend and runtime tests | Pass |
| Hashes independent of nested object insertion order | `tests/canonical.rs`, 2 tests, serde `preserve_order` enabled | Pass |
| CLI strict input, preflight, integrity and replay behavior | Core `tests/cli.rs`, 5 tests | Pass |
| Real GraphDB capture and bounded borrowed traversal | Native `tests/native.rs`, part of 9 tests | Pass |
| HolE and RotatE score parity with direct native calls | Native `tests/native.rs` | Pass |
| Actual training changes weights; deterministic seed; native evaluation parity | Native `tests/native.rs` | Pass |
| Reject split leakage, unseen vocabulary, malformed models and metrics | Native `tests/native.rs` | Pass |
| Train → promote → evaluate → predict → replay, in real processes | Native `tests/cli.rs`, 1 test | Pass |
| JavaScript process API, replay, permission and stream boundaries | Node test runner, 11 tests | Pass |
| Python process API, full u64 replay, stream/thread cleanup | Python unittest, 11 tests | Pass |
| Published layout usability without publication | npm pack/install + strict TypeScript compile; Python wheel/install | Pass |
| Formatting, warning gate and ADR allocation | Both manifests: fmt; Clippy all targets/no deps with `-D warnings`; ADR check | Pass |

Core test paths are relative to `crates/ruvector-language`; native paths are
relative to `crates/ruvector-language-native`. The 70 total comprises 38 core,
10 native, 11 JavaScript and 11 Python tests. Artifact smoke checks are additional
to this count. Every newly added source and test file is below 500 lines.

## Independent review and regressions

The compiler, runtime and native implementation were built in separate work
streams, then cross reviewed against the shared specification. Integration and
review found and fixed:

1. Flattened operation deserialization incorrectly rejected valid JSON fields.
   Canonical JSON and combined request roundtrips now pass while unknown fields
   and duplicate effects remain rejected.
2. Nested property key order affected hashing when an embedding host enabled
   serde's insertion ordered map feature. Borrowed canonical serialization now
   sorts nested object keys and charges sorting work.
3. Graph capture cloned arbitrary payloads before validation. Borrowed fallible
   visitors now validate identities, depth and property bounds before copying.
4. Native prediction copied large candidate strings before the output limit.
   It now ranks score/index pairs and preflights output bytes before allocation.
5. Python buffered unbounded child output and JavaScript could round u64 values.
   Streaming limits and explicit numeric precision checks now fail closed.
6. Missing replay arguments were detected after request execution. Required
   flags are now checked before input or model loading.

For the native output regression, the identical 1,000 entity / 50 subject /
top 1,000 request with 512 byte identities fell from 90,312 KiB to 9,216 KiB
peak child RSS and returned `LIMIT` before candidate allocation. It is a
targeted synthetic regression, not a general memory usage guarantee.

## Performance evidence

Full machine, source hash, workload, measurements and previous runs are in
[native-benchmark.json](native-benchmark.json). The final run uses HolE,
32 dimensions, 256 entities, 8 subjects, top 5, five warmups and 30 samples.
Direct and adapter paths both rank score/index pairs and clone only top k.
Candidate scores and rankings match before timing starts.

| Path | Median µs | p95 µs |
|---|---:|---:|
| Direct native KGE | 406.690 | 580.892 |
| Native adapter | 418.458 | 1,520.381 |
| Full language runtime | 437.987 | 1,000.401 |

Observed median runtime overhead is **7.70%** for this workload. The shared build
environment showed substantial tail variance. The direct baseline omits
validation, permission checks and receipt hashing, so this does not establish
the proposed 10% equivalent work product gate. The proposed 30% application code
reduction has not been established. Both remain product evaluation targets.
Synthetic initialized weights measure execution only; toy training tests do not
establish useful real world prediction accuracy or transfer to unseen graphs.

## Dependency identity and review scope

| Lockfile | SHA256 |
|---|---|
| Core `Cargo.lock` | `2381cce2e1ec9e372714914533e335bb6ec6866a96e17c2a2ad85e51e7f8dbe0` |
| Native `Cargo.lock` | `3beba443fc7ade7b251411c4e471f5728eb270b0cf7bfb578751de36d227fe5c` |

Both lockfiles are committed. A followup CI repair changes only the root
`rustls` lock entry from 0.23.45 to 0.23.46 and its registry checksum. Cargo
downloaded and verified the patched archive; dependency requirements, features
and minimum Rust version are identical in the registry records. The patched
library compiled with the root ring/std/tls12/logging feature profile in an
isolated Cargo check. Dedicated advisory checks now cover both standalone
language lockfiles; root advisory and lockfile integrity jobs validate the root
repair in CI. This does not suppress advisory findings. The
existing graph crate receives two additive borrowed visitor methods; existing
APIs keep their behavior. [SECURITY-REVIEW.md](SECURITY-REVIEW.md) records the
threat boundary, scanner scope, fixed findings and advisory coverage limits.

## Remaining limits

Graph capture requires caller enforced exclusion of all writers. The snapshot
profile supports one label per node, binary edges and no edge properties or
hyperedges. Predictions are transductive scorer outputs, not probabilities.
Imported model metrics and observation sources require host trust. Receipts
provide checksums, not signatures. Cross platform FFT bit equality is not
guaranteed. JavaScript accepts only safe integers; Python and Rust support u64.

No claim is made that all unrelated RuVector crates pass, that every transitive
dependency is vulnerability free, or that performance and accuracy targets have
been met outside the measured workload. This release implements the specified
bounded language; broader research extensions remain separate work.

## Counterfactual audit advancement

2026-10-16 followup: 84 tests passed (48 core, 10 native, 13 JavaScript,
13 Python), including 10 new audit/oracle tests and four SDK tests. No skipped
tests. Both standalone workspaces passed formatting and strict all-target Clippy.
Packed npm TypeScript 5.9.3 and installed Python wheel checks now also execute
audits. The acceptance and packaging scripts both exited 0.

The generated tests compare 128 seeds at two clocks (256 cases) against an
independent status evaluator and reverse input order to require identical
receipts. Another 32 seeds compare two interventions each (64 ablations).
The reference application executed and emitted a receipt plus audit report.
No external predictive accuracy benchmark was run.

Synthetic agent memory medians, microseconds, Linux x86_64, Rust 1.90 release:

| Nodes | Status-only direct evaluator | Full runtime | Baseline plus 3 audit trials |
|---|---:|---:|---:|
| 128 | 19.74 | 440.66 | 1961.89 |
| 1024 | 163.46 | 3121.65 | 12640.27 |
| 4096 | 640.51 | 12467.28 | 49978.01 |

The direct evaluator omits validation, capabilities, resource accounting,
provenance and receipt hashing. This is a feature-cost comparison, not an
identical-feature speedup claim. All three sizes matched statuses and replayed
exactly. Five warmups and 30 measured samples per path; raw timing samples are in
[agent-memory-benchmark.json](agent-memory-benchmark.json). Timings exclude parsing,
process startup and graph capture. Shared-host variance limits generalization.
See [FRONTIER.md](FRONTIER.md) and ADR-GL-006 for sources, bounds and next gates.

## Immutable preparation followup

PreparedSnapshot is an additive Rust host API for repeated queries over a borrowed
immutable snapshot. Four new tests add 96 cold/warm query and clock comparisons,
invalid snapshot checks, capability revocation, budget equivalence and fresh
provider invocation. Complete receipts are equal, not only decision statuses.

Measured medians in microseconds, three alternating binary pairs, 30 samples per
path per pair, five warmups. The cold and prepared columns use the new binary;
preparation is measured independently:

| Nodes | Cold execution | Prepared execution | Preparation cost | Warm reduction |
|---|---:|---:|---:|---:|
| 128 | 454.22 | 124.30 | 303.85 | 72.64% |
| 1024 | 3226.86 | 272.81 | 2840.60 | 91.55% |
| 4096 | 13078.38 | 930.38 | 12181.73 | 92.89% |

These are warm query reductions on synthetic fixtures, excluding initial
preparation. Median estimates suggest amortization within two queries; do not
interpret that estimate as a guarantee. The retained indexes consume memory until
dropped; resident memory was not measured. No changes to process SDK caching or
audit intervention execution are included.

Cold old/new differences range from a 2.59% reduction to a 3.52% increase; audit
differences range from a 2.60% reduction to an 8.14% increase. The unchanged direct
control slowed 3.26% to 6.01%. This host does not establish zero regression or a
cold-path improvement. The API is opt-in; profile cold workloads on a dedicated
host before making a performance release claim.

Raw final results: [prepared-specialized-benchmark.json](prepared-specialized-benchmark.json).
The first prepared candidate with runtime path selection is retained in
[prepared-snapshot-benchmark.json](prepared-snapshot-benchmark.json); the shipped
implementation specializes the two paths at compile time. The rejected 8 KiB
hash buffer results are retained in [hash-buffer-benchmark.json](hash-buffer-benchmark.json).
That experiment overlapped other local checks and supports no performance claim.

Reproduce with `scripts/benchmark-rugraph-paired.py --before OLD_BINARY --after
NEW_BINARY --output NEW_REPORT.json`. Both binaries are the release
`agent_memory_bench` example. Baseline source is commit
`6834ddc78bf51a1a2b6332409d194a7a8b77ec0d`; the candidate is this change, Rust
1.90.0, Linux x86_64. Reports retain binary SHA256 and raw samples. The runner
requires identical receipt hashes, runtime work, audit work and intervention
deltas across every block. It intentionally sets no noisy CI timing threshold.

Final local gate: 88 tests passed (52 core, 10 native, 13 JavaScript, 13 Python),
with formatting and strict all-target Clippy passing in both workspaces.

## Exact selection and borrowed node sets

Followup baseline: `1c1f90115f76c01779bab595e406a10a2e5c3fd2`. Candidate partitions
exact cosine scores and sorts only the winning K, while intermediate node IDs
borrow from the immutable snapshot. It preserves historical work/byte charges,
all vector validation, tie ordering, owned public outputs and complete receipts.

Three alternating old/new binary pairs, 5 warmups and 30 samples per path per
pair; medians of block medians, in microseconds:

| Nodes | Previous prepared | New prepared | Reduction | New cold | Preparation |
|---|---:|---:|---:|---:|---:|
| 128 | 124.38 | 111.03 | 10.73% | 446.59 | 308.59 |
| 1024 | 285.19 | 173.60 | 39.13% | 3250.58 | 3063.81 |
| 4096 | 863.75 | 496.03 | 42.57% | 13474.92 | 12874.55 |

This is an additional reduction relative to the previous prepared API, not a
comparison against its old cold path. The new warm/cold difference at 4096 nodes
is 96.32%, excluding preparation. The unchanged status-only control moved less
than 1% at all sizes. Cold execution ranged from 4.68% faster to 0.06% slower;
audit timings ranged from 3.47% faster to 2.38% slower. No universal no-regression
or cold-speedup claim is made.

All old/new receipt hashes, runtime work, audit work and intervention deltas match.
Raw samples and binary hashes: [selection-borrowing-benchmark.json](selection-borrowing-benchmark.json).
Reproduce with the paired runner and release `agent_memory_bench` binaries.
The source fixture uses TWO dimensional vectors and top K=4. This measures node
allocation/ranking overhead, not production embedding throughput. Higher dimensions,
larger K, changed snapshots and model inference need separate measurements. Scoring
still examines every vector and retains O(N) scratch space. No RSS figure is claimed.

Two new tests compare exact selection against a separate full sort over 64 cases
(16 generated datasets times four K values), including ties, zero/missing/negative
and extreme finite vectors. A malformed candidate outside top K still fails.
Cold and prepared receipts agree in every case. See ADR-GL-008.

Final optimization gate: **90 tests passed** (54 core, 10 native, 13 JavaScript,
13 Python), both Rust workspaces passed fmt and strict Clippy, and packed npm
TypeScript plus installed Python wheel checks passed. No tests skipped.
