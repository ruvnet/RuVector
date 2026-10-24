# Bind Python collection sidecars to their persisted index

## Context

Collection.save replaces the index and metadata sidecar separately. A failure at the second rename can pair a new sidecar with old vectors. Equal dimensions and row counts previously let Collection.load silently serve this mixed pair, for both HNSW and RaBitQ.

## Decision

New sidecars store a SHA-256 digest of the complete persisted index and are replaced before the index. Load validates its representation and compares the index digest before backend reconstruction. Hashing uses bounded one-MiB reads and supports Python 3.9. Legacy sidecars without the field remain readable with their existing payload checks.

## Consequences

Save and load each read the complete index once more, adding IO and hashing latency. The metadata-first order ensures a failed second rename leaves a digest-bearing sidecar, including the first upgrade of legacy files. Existing legacy pairs without a digest remain readable but cannot establish their pairing. It does not roll back the two renames or guarantee safety against concurrent writers or authenticate metadata. Empty RaBitQ collections still require only their sidecar.

## Validation

Public save/load regression tests inject a second-rename failure for both backends with equal-sized and differently-sized old/new vector sets, including legacy upgrade. Legacy sidecar, ordinary round-trip, empty collection, malformed digest, and legacy non-finite payload controls retain their contracts. Python source is exercised with the published 0.1.2 native wheel; no fresh Rust source build or cross-platform result is claimed.
