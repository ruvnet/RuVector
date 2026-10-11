# ADR: Parallel encoding for incremental Python RaBitQ batches

## Status

Proposed; native correctness and paired performance validation pending.

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

Publication requires core Rust tests, actual fixed native Python tests, and
paired frozen-data timings with the same build profile and thread settings.
