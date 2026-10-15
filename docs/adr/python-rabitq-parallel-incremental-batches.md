# ADR: Parallel encoding for incremental Python RaBitQ batches

## Status

Accepted for implementation; independently reviewed promotion remains required.

## Context

Issue #1141 describes a throughput gap between constructing an index and
incrementally appending the same vectors. The core bulk constructor already
rotates and packs vectors through Rayon. Python `RabitqIndex.add_batch`, used by
`Collection.insert_batch`, releases the GIL but calls the serial single-row
encoder repeatedly. GIL release alone does not parallelize that work.

A local screening used the existing native artifact built from
`07933730fac709457e0c5f41444c9a4adc505aee`, the dev profile with opt-level 0,
four Rayon workers, frozen 2,000 by 128 float32 Gaussian vectors, and five
measured runs after warmup. Median bulk construction was 156.957 ms; appending
1,900 rows to a prebuilt 100-row index was 417.029 ms. This is preliminary
same-host screening, not release performance or a reproduction of the issue's
historical 13x figure. Paired baseline/fixed measurements are required before
publication.

## Decision

Expose `RabitqPlusIndex::add_batch` using the constructor's existing encoding
and serial commit phases. Validate every row dimension before mutation, encode
against the existing seeded rotation, collect in indexed input order, reserve
storage, and append codes, norms, IDs and originals together. Construction
uses this same path. Native targets retain Rayon; wasm32 retains sequential
encoding. Empty and single-row batches avoid parallel scheduling.

The Python binding keeps its existing dtype, contiguity, dimension, finite
value, ID range and duplicate checks, then invokes the core batch method while
the GIL is released. Update its ID set only after successful insertion.

## Alternatives and consequences

Keeping the serial loop preserves a material batch throughput gap. Rebuilding
the whole index would rotate old rows again and change the cost of incremental
work. Parallel mutation of storage complicates ordering and consistency; the
chosen path parallelizes only independent encoding and keeps commit serial.

Encoded batches temporarily retain packed codes as well as the owned input
rows, as the existing bulk constructor does. No new dependency, persistence
format, ID policy, numerical estimator or rotation algorithm is introduced.
Allocator failures and panics are not converted into transactional errors.

## Verification contract

Core tests compare incremental, serial and bulk codes, norms, original rows,
nonsequential IDs and exact search scores for both rotation kinds and aligned
and nonaligned dimensions. A late invalid dimension must leave every storage
array unchanged. Actual native Python controls exercise empty, single-row and
multirow insertion, external IDs, search and retained originals, plus rejected
dimension, nonfinite, duplicate and overflowing IDs with successful reuse after
rejection. All six Python controls pass against the original native artifact;
they protect semantics rather than claim the performance defect as a failure.

## Observed validation

The formatted native source at `610d6fdca7de50d3caf21d01aa74a9cda02309bb`
passed the complete RaBitQ Rust suite (41 tests) and actual native Python suite
(301 passed, two skipped). The baseline preservation controls passed all six
cases without skips. These are semantic controls, not performance failures.

Paired original/fixed measurements used the same dev opt-level-0 build,
debug-info disabled, incremental compilation disabled, four Rayon workers,
identical frozen data and Python source, and alternating artifact order.
Each process warmed up both producer paths. Five primary rounds measured
2,000 by 128 float32 rows with a 100-row prefix. Median append time was
397.534 ms originally and 101.562 ms fixed: a measured 3.914x improvement on
this host. Full construction was 149.871 ms versus 149.018 ms. A two-row
append control (three rounds, 25 measured iterations per artifact) was
437.625 microseconds versus 274.333 microseconds. No release-profile speedup
is claimed. The original issue's historical 13x result is not reused.

Tests compare the retained index memory footprint exactly. Process peak RSS
was also recorded, but is observational whole-process memory, not an isolated
measurement of batch allocations. The temporary encoded-row storage tradeoff
above still applies. An initial timing run overlapped compiler-owner test
processes; its logs were retained and excluded. The reported paired run began
only after the owner confirmed the compiler and tests were idle.

The baseline native SHA256 was
`55270150fc7d789fd47469aa7dddcc831b6fcba50eadcf8c332bc38460983359`;
the fixed native SHA256 was
`e8e2a32a2eac1fbbcbaa04f7d3fd0cc413be7db6f3ca12983db6f11c5f80913b`.
The frozen primary NumPy file SHA256 was
`225ad6ccbfeaaa5aba460b9306cff64cf31e4bcf6f78dc3188e47c3d015f86a8`.
The final documentation commit retains the tested native input bytes.

To reproduce the public producer, create float32 data with
`np.random.default_rng(1141).standard_normal((2000, 128), dtype=np.float32)`.
Time `Collection.from_vectors(data, backend="rabitq", seed=42)` for build;
for append, construct the first 100 rows before timing and call
`collection.insert_batch(data[100:])`. Assert returned IDs are 100 through
1999 and the resulting count is 2,000. Warm up both paths, alternate order,
record individual timings and host load, and compare immutable artifacts built
with identical compiler settings. These measurements do not cover release
optimization, other dimensions, other hosts, or different thread counts.
