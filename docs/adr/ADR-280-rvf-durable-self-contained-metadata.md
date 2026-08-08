# ADR-280: Durable Metadata for Self-Contained RVF Artifacts

- **Status**: Proposed
- **Date**: 2026-07-29
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-026, ADR-029, ADR-056, issue #704
- **Tags**: rvf, metadata, portability, filtered-search, cow, persistence

## Context

RVF is the canonical portable format for RuVector, but an RVF file cannot yet
carry its application metadata durably through the public SDK and runtime.
Vectors survive close and reopen; metadata does not.

The current implementation has most of the structural pieces:

- `SegmentType::Meta` (`0x07`) and `SegmentType::MetaIdx` (`0x0D`) are
  reserved.
- `SegmentWriter::write_meta_seg` exists but has no runtime caller.
- `RvfStore::ingest_batch` accepts metadata and installs it in an in-memory
  `MetadataStore`.
- create, open, read-only open, and COW-child construction initialize a new,
  empty `MetadataStore`.
- native search results expose identifiers and distance, but not stored
  metadata.

The public TypeScript SDK previously accepted metadata and silently dropped
it. The interim safety behavior introduced for issue #704 correctly rejects
non-empty metadata. That prevents corruption, but it does not deliver the
portable artifact promised by ADR-029.

A self-contained RVF artifact must be sufficient to move a merchant catalog,
agent memory, or other vector collection between processes and machines
without a sidecar database. The artifact must preserve:

1. the field dictionary and value types;
2. the metadata associated with every vector identifier;
3. file-level provenance, including embedding configuration;
4. metadata behavior across close/reopen, compaction, and COW branches; and
5. enough information to rebuild filtered-search indexes.

## Decision

RVF will make `META_SEG` authoritative durable state and treat `META_IDX_SEG`
as optional, derived acceleration.

### 1. `META_SEG` is the source of truth

Each committed metadata generation contains a versioned payload with:

- a schema dictionary mapping a file-local `field_id` to a canonical UTF-8
  field name and value type;
- file-level metadata in the reserved `rvf.*` namespace;
- per-vector records keyed by the RVF vector identifier;
- explicit field deletion and record deletion markers for delta generations;
  and
- a generation number and base-generation reference.

The logical v1 payload is:

```text
MetaHeader {
  magic: "RVFMETA\0"
  version: 1
  flags
  generation
  base_generation
  schema_count
  file_metadata_count
  record_count
}

SchemaEntry[] {
  field_id
  value_type
  flags
  name_len
  name_utf8
}

FileMetadataRecord[] {
  key_len
  key_utf8
  value_tag
  value_bytes
}

MetadataRecord[] {
  vector_id
  operation       // upsert | delete-record
  field_count
  FieldValue[]    // field_id + value-tag + value-bytes
}
```

The exact byte encoding, bounds, and canonical ordering will be specified in
`rvf-types` beside the Rust structs. Encoding must be deterministic: schema
entries sort by `field_id`, file metadata by canonical key, records by
`vector_id`, and fields within a record by `field_id`.

Supported v1 value types are UTF-8 string, bytes, signed integer, unsigned
integer, finite `f64`, and boolean. Nullability is a schema flag, not a
standalone value type. Arrays and nested objects are deferred; SDKs must
reject them rather than stringify them implicitly.

The wire tags distinguish four states:

| State | Wire representation | Snapshot/COW meaning |
|-------|---------------------|----------------------|
| Absent field | no `FieldValue` entry | In a delta, inherit the prior value; in a full snapshot, the field is absent |
| Null value | `ValueNull` | Replace the inherited value with null; valid only when the schema field is nullable |
| Delete field | `DeleteField` | Remove the current or inherited field so reconstructed output treats it as absent |
| Delete record | `MetadataRecord.operation = DeleteRecord` | Tombstone the complete current or inherited record |

Typed non-null values use distinct tags (`ValueString`, `ValueBytes`,
`ValueI64`, `ValueU64`, `ValueF64`, and `ValueBool`). An upsert with zero
fields represents an existing empty metadata record. Decoders reject null on
a non-nullable field and any value tag incompatible with the schema.

### 2. Field identifiers are file-local and lineage-stable

Field identifiers are not global identifiers and must not be compared across
unrelated RVF files.

- A root file assigns monotonically increasing `field_id` values.
- A COW child inherits the parent dictionary and never reuses an inherited
  identifier.
- New child fields append identifiers.
- Merging unrelated files resolves fields by canonical field name and
  compatible value type, then remaps identifiers into the destination
  dictionary.
- A same-name, incompatible-type merge is an explicit error unless the caller
  supplies a conversion policy.

This keeps files independently portable without imposing a global schema
registry while preserving stable identifiers throughout one COW lineage.

### 3. `META_IDX_SEG` is rebuildable

`META_IDX_SEG` may contain equality bitmaps, range indexes, or other
field-specific accelerators. It is never authoritative.

- Missing or corrupt metadata indexes do not make metadata unavailable.
- Readers fall back to evaluating filters against the restored
  `MetadataStore`.
- An index records the `META_SEG` generation it covers. A generation mismatch
  makes the index stale and therefore unusable.
- Compaction may discard and rebuild every metadata index.

The first implementation may use a linear scan. Persistence correctness is a
release requirement; index acceleration is not. Linear fallback is bounded
by configurable record, decoded-byte, and wall-clock budgets, supports
cooperative cancellation, and emits scanned-record, scanned-byte, elapsed,
fallback-reason, and stale-index metrics. Exceeding a budget returns
`FilterBudgetExceeded`; it never returns a silently partial result. Operators
may opt into a larger budget per request.

### 4. Metadata commits atomically with vector state

An ingest that includes metadata is visible only after all referenced vector
and metadata segments are complete and the manifest points to the new
generation.

Validation is against the resulting candidate vector snapshot: the previous
snapshot plus all vector inserts, replacements, and deletions in the atomic
batch. Metadata for a vector inserted by the same batch is valid. A metadata
upsert for a vector deleted by the batch is invalid, while a metadata record
tombstone paired with that vector deletion is valid.

The write sequence is:

1. validate the complete batch, including vector/metadata cardinality, field
   types, limits, and duplicate identifiers;
2. write vector, metadata, and optional index segments with content checksums
   and a completion footer covering the header and payload;
3. fsync every segment-bearing file;
4. serialize the complete new manifest to a temporary sibling file;
5. fsync the temporary manifest;
6. atomically rename the temporary manifest over the canonical manifest
   publication path;
7. fsync the containing directory.

The manifest references only segments whose completion footer and checksum
validate. A failure before atomic rename leaves the prior manifest
authoritative and new segments orphaned. A failure after rename but before
directory fsync is treated as uncommitted after recovery unless the new
manifest is durably discoverable and fully validates. A metadata failure must
not commit the corresponding vectors.

Implementations that embed the manifest inside one RVF file may use two
alternating manifest-root slots instead of a filesystem rename only if they
provide equivalent guarantees: generation-numbered slots, checksum and
completion footer, write to the inactive slot, file fsync, atomic selector
publication, second file fsync, and containing-directory fsync. In-place
overwrite of the active manifest is forbidden.

Every open path restores the latest manifest-reachable metadata generation
before serving filtered queries. Read-only open follows the same restoration
path.

### 5. COW branches inherit metadata and store deltas

A COW child observes its parent metadata snapshot plus child metadata deltas.
It does not copy the complete parent metadata payload.

- Parent metadata is resolved at the snapshot epoch recorded by the COW map.
- Child `META_SEG` generations contain upserts and tombstones relative to that
  snapshot.
- Child lookup applies child records over the inherited parent record.
- Child compaction may materialize a complete standalone metadata generation
  only when explicitly exporting or detaching the branch.
- Freezing a child freezes its metadata generation with its vector state.

This mirrors vector COW semantics and prevents metadata from becoming a
second, inconsistent branching system.

Replay complexity is bounded across both local deltas and inherited COW
ancestry:

- at most 64 deltas may occur between full metadata snapshots;
- at most 256 MiB of decoded metadata may be consumed while reconstructing
  one snapshot;
- writers materialize a full snapshot before either ceiling would be
  exceeded;
- readers detect repeated generations, ancestry cycles, missing bases,
  multiple parents, and non-monotonic generation numbers; and
- each delta generation must be exactly newer than the generation it names
  as its base.

An artifact that exceeds a ceiling or violates ancestry returns
`MetadataReplayLimitExceeded` or `MetadataAncestryInvalid` before serving
metadata or filtered queries. It is not partially reconstructed. Repair and
compaction tooling may operate with a separately authorized larger budget,
but normal open never expands these ceilings based on untrusted file content.

### 6. Public APIs expose durable metadata

All supported SDKs converge on these operations:

```text
ingest_batch(entries_with_metadata)
get_metadata(vector_id) -> metadata | not_found
search(query, options.include_metadata) -> hits
filter_search(query, filter) -> hits
```

Search does not include metadata by default, avoiding unnecessary decoding
and allocation. `include_metadata: true` returns the metadata from the same
committed snapshot as the hit list.

Filter declarations use canonical field names at public API boundaries.
The runtime resolves names to file-local identifiers and validates value
types. Low-level `field_id` APIs may remain for trusted tooling but are not
the portable SDK contract.

### 7. File-level provenance lives in metadata

The reserved `rvf.*` namespace records the canonical
`EmbeddingSpaceIdentity` defined by ADR-281:

```text
rvf.embedding.identity       // RFC 8785 canonical JSON bytes
rvf.embedding.space_id       // SHA-256 identity hash
rvf.schema.application
rvf.schema.version
```

The identity includes model and graph hashes, tokenizer hash, prompt-template
hash, pooling, normalization, truncation length, output dimension and dtype,
runtime revision, distance metric, role policy, prefix policy, and
`prefix_policy_version`. RVF does not maintain a smaller competing
provenance list.

Application fields may not use the `rvf.*` namespace. Provenance mismatch
rules from ADR-210 and ADR-281 remain in force: incompatible corpus mutation
and text embedding are disabled, while vector-only reads, inspection,
verification, and export remain available.

### 8. Artifact integrity binds vectors, metadata, schema, and provenance

Every committed manifest contains:

```text
artifact_state_commitment =
  SHA-256(
    "ruvector.artifact-state.v1\0"
    || manifest_generation
    || vector_generation
    || metadata_generation
    || vector_merkle_root
    || metadata_merkle_root
    || schema_sha256
    || embedding_space_id
    || parent_state_commitment_or_zero
  )
```

The manifest's integrity root and every artifact signature cover
`artifact_state_commitment`. A metadata generation cannot therefore be
replayed against a different vector generation, schema, embedding space, or
COW parent without verification failure. COW children bind the exact parent
state commitment and snapshot epoch they inherit.

Open verifies the complete binding before enabling metadata, filtered search,
text embedding, or corpus mutation. Verification failure leaves only
diagnostic inspection of raw segments available.

### 9. Resource limits are part of the format contract

Readers must bound field count, field-name length, record size, string/bytes
value length, total decoded metadata, and nesting depth (zero in v1). All
offset arithmetic is checked. Invalid UTF-8 field names, non-finite floats,
duplicate field identifiers, and records that do not match the resulting
committed vector snapshot are rejected as format errors.

Limits are configurable within implementation-defined hard ceilings and are
reported in inspection output.

## Compatibility and Migration

- RVF files without `META_SEG` remain valid and expose empty metadata.
- A manifest marks authoritative segment classes and required reader
  capabilities. A reader that encounters an unknown authoritative segment is
  read-only unless its rewrite path explicitly copies every opaque,
  manifest-reachable segment byte-for-byte, preserves its manifest metadata
  and ordering constraints, and includes it in the new integrity root.
- Compaction or migration that cannot prove opaque preservation is forbidden.
  Merely ignoring an unknown segment while rewriting known state does not
  satisfy ADR-029's forward-compatibility rule.
- The SDK's `MetadataNotSupported` rejection remains until persistence,
  reopen, and result-retrieval gates pass on native and WASM backends.
- No SDK may re-enable metadata ingestion on the strength of in-memory tests
  alone.
- Sidecar import is an explicit migration command that writes a new
  `META_SEG`; it never mutates a file without creating a new committed
  generation.

## Acceptance Criteria

1. Metadata with unequal field counts per vector survives close/reopen.
2. Strings, bytes, integers, finite floats, booleans, nullable values,
   absent fields, field deletions, and record tombstones round-trip with
   distinct COW behavior.
3. File-level metadata and the complete `EmbeddingSpaceIdentity` round-trip.
4. Filter results are identical before and after reopen.
5. `get_metadata` and metadata-bearing search hits return the committed
   values.
6. Crash injection after every commit-protocol write, fsync, rename, selector,
   and directory-fsync boundary exposes exactly the prior or new complete
   snapshot, never a mixed or torn snapshot.
7. Metadata for a vector inserted in the same transaction succeeds;
   metadata upsert for a vector deleted in that transaction fails; paired
   vector deletion and metadata tombstone succeeds.
8. A missing or corrupt `META_IDX_SEG` falls back to correct bounded linear
   filtering; cancellation and budget exhaustion return explicit errors and
   observability counters.
9. COW children inherit parent metadata, override fields, and tombstone
   records without changing the parent.
10. Delta replay materializes by 64 generations, refuses more than 256 MiB
    decoded data, and detects cycles, missing bases, and non-monotonic
    generations.
11. Compaction preserves opaque authoritative segments, the logical metadata
    snapshot, and embedding identity, or refuses to run.
12. Tampering independently with vector generation, metadata generation,
    schema, embedding identity, or COW parent breaks the artifact state
    commitment and signature verification.
13. Native, N-API, and WASM byte export/import pass the same fixture corpus.
14. Fuzz tests cover payload decoding, length arithmetic, duplicate fields,
    invalid types, ancestry, unknown authoritative segments, and resource
    ceilings.
15. Given one corpus, changing only the query template while retaining the
    model ID produces a new `embedding_space_id`, rejects cache reuse and
    corpus mutation, preserves vector-only reads, and requires a new
    ADR-282 experimental revision.

## Consequences

### Positive

- One RVF file becomes a complete, portable application artifact.
- Filter behavior no longer changes after restart.
- COW vector and metadata semantics remain aligned.
- Metadata indexes can evolve without changing the authoritative format.
- Embedding provenance travels with the vectors it governs.

### Negative

- Metadata adds format, migration, and resource-accounting complexity.
- Filtered search may initially be linear after open or index invalidation.
- COW lookup must resolve a metadata overlay as well as a vector overlay.
- Public SDKs must align value types and error behavior across native and
  WASM implementations.

## Alternatives Considered

- **Keep metadata in a sidecar database**: rejected because it breaks
  portability, atomicity, signing, COW lineage, and single-artifact delivery.
- **Make `META_IDX_SEG` authoritative**: rejected because indexes are
  implementation-specific and should remain safely rebuildable.
- **Use globally stable field identifiers**: rejected because it requires a
  global registry and makes independent artifact creation brittle.
- **Write a complete metadata snapshot on every mutation**: rejected because
  it defeats append-only efficiency and COW branching.
- **Return metadata in every search result**: rejected because it adds decode
  and allocation cost for callers that only need identifiers and distance.

## Implementation Surfaces

- `schemas/embedding-space-identity-v1.json` from ADR-281
- `crates/rvf/rvf-types`: versioned metadata wire types and validation
- `crates/rvf/rvf-runtime/src/filter.rs`: restored authoritative store
- `crates/rvf/rvf-runtime/src/store.rs`: open, ingest, query, COW, compaction
- `crates/rvf/rvf-runtime/src/write_path.rs`: committed metadata segments
- `crates/rvf/rvf-node`: N-API ingest and metadata retrieval
- `crates/rvf/rvf-wasm`: byte export/import and metadata APIs
- `npm/packages/rvf`: field-name-based portable SDK contract
