# Prepare Collection metadata before committing native rows

Status: proposed for review

## Problem and evidence

Collection.insert and insert_batch commit RaBitQ vectors before converting supplied metadata with dict(). A rejected conversion therefore raises to the caller after native rows have been added, while the Python next-ID counter has not advanced. A late invalid batch metadata entry can also leave metadata for an earlier, unacknowledged row. This differs from provider embedding-output validation in the LangChain adapter: it occurs directly at the Collection mutation boundary.

Six original public-consumer cases against the pristine source-built 07933730 native extension produced four failures and two valid-backend controls: single and batch insertion into both lazy and populated RaBitQ collections left native rows after rejecting metadata.

## Decision

Materialize the single metadata dictionary or the entire batch's dictionaries before native mutation, then reuse those copies for HNSW's native arguments and the Collection metadata map. Preserve None and empty dictionary behavior and valid mapping conversion. Rejected falsey non-dictionaries now fail before insertion. The copies are shallow, as before; this does not introduce a JSON schema or recursively freeze caller-owned values. from_vectors is unchanged because its failed constructor does not expose an existing Collection.

## Validation and limits

Ten final public-consumer tests cover the four failures, unchanged valid metadata on both backends, one conversion per supplied mapping, top-level caller-dictionary snapshots and None/empty dictionaries. The original baseline covers six tests; the four added controls are fixed-head validation. Full Python SDK tests run against the existing immutable source-built 07933730 extension, with optional integration dependencies already installed. No native source changes or fresh native build are claimed.

Preparing metadata is O(total metadata entries) with one temporary dictionary per non-None item. It prevents metadata-conversion failures from leaving unacknowledged rows; it does not promise rollback for arbitrary subsequent native errors, concurrent writers or nested metadata mutation. Existing backend-specific native metadata value validation remains unchanged.

## Changelog

Reject invalid Collection metadata before inserting vectors or advancing mutation state.
