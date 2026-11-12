# Preserve external identities at Python RaBitQ export and persistence boundaries

## Context

RabitqPlusIndex.export_items intentionally returns row positions with vectors. The Python binding promises external IDs, but passed core row positions through both export_items and save_index. A sparse index could search with correct external IDs, then lose those IDs and associated Collection metadata after save/load. The same mistake affected Collection consumers of exported rows.

## Decision

The Python wrapper joins each exported vector with the corresponding external_ids entry, using one private helper shared by export and save. Preserve the underlying core row-position exporter contract and the existing disk format.

## Consequences

No additional vector copy is introduced beyond the existing owned core export. The wrapper performs a linear join over identities and vectors. Existing files already saved with substituted row IDs cannot recover their original identities; this change preserves identities for newly saved files. The related Collection decisions preserve those identities during deletion and vacuum.

## Validation

Actual current-source native bindings at the unchanged base expose three public regressions: sparse native export IDs, native save/load search identity and duplicate rejection, and Collection sparse-ID metadata round-trip. Seventy-three controls pass on that baseline. Tests use source-built macOS arm64 bindings and deterministic vectors. Build and runtime receipts are bound to the exact native source SHA; published-wheel version equality is not used as evidence of source equivalence.
