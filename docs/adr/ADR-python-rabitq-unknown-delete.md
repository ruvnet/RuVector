# Ignore missing RaBitQ IDs before adding tombstones

## Context

Collection.delete previously tombstoned any integer for RaBitQ, while HNSW ignored missing IDs. Missing tombstones corrupted reported counts and could hide a later insert receiving that ID. Sparse caller IDs make a numeric range check insufficient.

## Decision

Track RaBitQ index identities in a Python set. Bulk construction already knows the IDs; a loaded collection derives the set from native export once on its first delete. Successful inserts and batch inserts update an existing cache, and vacuum replaces it with rebuilt identities. Missing IDs and an empty index are no-ops.

## Consequences

The set adds memory proportional to vector count. First deletion after loading copies native exported vectors once because the binding has no identity-only Python accessor; later deletes avoid those repeated copies. This depends on the native export/save external-ID correction included in this change. It does not promise transactional multi-row inserts.

## Validation

Public Collection tests cover missing sparse IDs, empty collections, future IDs followed by insertion, repeated deletion, batch insertion, persistence, and vacuum, with HNSW controls. Runtime tests use the separately hashed native artifact built whose native source files are identical to this revision.
