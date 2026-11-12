# ADR-353: RVF Stream Descriptor (RVSD) — verified streaming delivery (project Ofer)

- **Status**: Proposed (implemented on `feat/ofer-core`; section 10 records where the code refines this text)
- **Date**: 2026-10-26
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-033, ADR-284, ADR-285, ADR-286, ADR-287, ADR-354; RVM ADR-149, ADR-155, ADR-160
- **Tags**: rvf, streaming, integrity, signatures, progressive-index, wasm, ofer

## Context

An RVF reader cannot verify a prefix today. The manifest is found by tail scan, per-segment hashes in the header are attacker-recomputable (XXH3, or CRC32-rotation in `rvf-runtime`), segment signatures trail their payload, cover no file identity, and exist only for some segment types. A truncated file, a stale earlier version, a reordered or spliced file and a forged tail manifest are all indistinguishable from a valid file (research dossier, project Ofer, sections 1 and 6).

ADR-284 (Accepted) requires streaming validation of executable modules and ADR-287 (Proposed) describes it. ADR-155 keeps `rvfIdentity` = SHA-256 of the canonical RVF. Neither says how bytes are authenticated while they arrive. This ADR specifies that mechanism. It does not change the RVF segment format.

## Decision

Authority lives in a small signed **stream descriptor** that commits to every byte of an unchanged RVF **body**. The descriptor is delivered first (inline as the first stream frame, as a sidecar `name.rvf.rvsd`, or referenced by `descriptor_hash` from a package contract). A receiver verifies the descriptor, then accepts body bytes only as chunks whose hashes the descriptor commits to. Bytes that fail a hash are never released to any consumer.

Rejected: an in-file head segment (circular with `rvfIdentity`); per-segment signed headers (cannot detect truncation); hash chains alone (cannot detect truncation or a stale prefix); the QR-seed manifest as is (symmetric default signature, u32 offsets); full Bao encoding (new wire format and dependency; reserved as `hash_alg = 2`).

### 1. Descriptor layout (all integers little-endian; fixed layout, length-checked)

| Offset | Size | Field | Rule |
| --- | --- | --- | --- |
| 0x00 | 4 | `magic` | bytes `52 56 53 44` ("RVSD"); must not collide with `RVFS`, `RVM0`, `RVQS` |
| 0x04 | 2 | `desc_version` | `1`; any other value is rejected |
| 0x06 | 2 | `flags` | bit0 `chunked` (chunk tables present), bit1 `identity_present`, bit2 `require_complete_before_exec`; unknown bits rejected |
| 0x08 | 1 | `hash_alg` | `1` = SHA-256; `2` reserved (BLAKE3); other values rejected |
| 0x09 | 1 | `chunk_log2` | `0xFF` = one chunk per segment (segment-granular profile); otherwise `12..=30` (default 16 = 64 KiB) |
| 0x0A | 2 | `sig_alg` | `0` = Ed25519 (`verify_strict`); other values rejected |
| 0x0C | 4 | reserved | must be zero |
| 0x10 | 16 | `file_id` | random, unique per artifact lineage |
| 0x20 | 8 | `epoch` | monotone; receivers pin the highest accepted epoch per `file_id` |
| 0x28 | 8 | `body_len` | total body bytes |
| 0x30 | 32 | `rvf_identity` | SHA-256 of the body (ADR-155 identity); checked at completion |
| 0x50 | 32 | `policy_hash` | `capabilityPolicyHash` (ADR-155 section 5) |
| 0x70 | 4 | `seg_count` | |
| 0x74 | 4 | `table_len` | must equal `seg_count * 72` |
| 0x78 | 8 | `expires_ns` | `0` = no expiry; enforced only when the receiver has a trusted clock |
| 0x80 | 8 | `signer_key_id` | first 8 bytes of SHA-256(public key) |
| 0x88 | 8 | reserved | must be zero |
| 0x90 | 72 x n | `entries[seg_count]` | see below |
| end | 2 | `sig_len` | `64` for Ed25519 |
| end+2 | 64 | `signature` | Ed25519 over `"RVF-v1-stream-descriptor\0" || every preceding descriptor byte` |

Entry (72 bytes): `seg_id u64 | seg_type u8 | stage u8 | flags u16 | requires_mask u32 | body_offset u64 | stored_len u64 | uncompressed_len u64 | chunk_table_hash [32]`.

Correction to the research dossier: its sketch listed a 96-byte entry that also held a `seg_root`; `seg_root` is redundant with `chunk_table_hash`, so the entry is 72 bytes. 4,096 segments cost 288 KiB of table.

`stored_len` covers the **whole stored segment** (64-byte header, payload, signature footer, padding) so header fields that the legacy signature does not cover (`uncompressed_len`, `alignment_pad`) are authenticated by the descriptor. A trailing Level-0 root page is covered by a pseudo-entry with `seg_type = 0xFE` (descriptor-local; `0xF0..0xFF` is reserved in `rvf-types`).

### 2. Descriptor invariants (all checked before any body byte is accepted)

1. `len(descriptor) <= policy.max_descriptor_bytes` (default 1 MiB) and `seg_count <= policy.max_segments`; checked from the fixed header before reading the table.
2. Entries are sorted by `body_offset`, start at 0, are **contiguous with no gap and no overlap**, and end exactly at `body_len`. Every body byte belongs to exactly one entry, so nothing can be smuggled between segments.
3. `seg_id` values are unique; `stored_len >= 64` except for the `0xFE` pseudo-entry; `uncompressed_len` is bounded by policy.
4. Per-class totals (executable, model, index, state) and per-segment maxima satisfy the signed `SizePolicy` (ADR-284 1.8, ADR-287). Executables are bounded by policy, not by the 1 MiB `MAX_MODULE_SIZE` constant, which remains the executor backstop (ADR-155 section 4).
5. The signature verifies under a key in the receiver's trust set identified by `signer_key_id`; `epoch >= pin.epoch` where the pin is looked up by `(signer_key_id, file_id)` and carries the hash of the descriptor accepted at that epoch (an equal epoch requires an equal descriptor hash); `expires_ns` is honored when a clock exists; `policy_hash` equals the expected value when the caller pins one.
6. `checksum_algo` 0 or 1 in any header of an executable or index segment is rejected at consumption time (only SHA-256-grade commitments authenticate bytes).

### 3. Chunk commitments

- Leaf: `SHA-256( 0x00 || file_id || seg_id u64 || chunk_index u32 || chunk_len u64 || bytes )`.
- A segment has `n = 1` when `chunk_log2 = 0xFF` or `stored_len <= chunk_size`, else `n = ceil(stored_len / chunk_size)`; the last chunk is the short one.
- `n == 1`: `chunk_table_hash` is the leaf hash itself and no table frame exists.
- `n > 1`: `chunk_table_hash = SHA-256( 0x01 || file_id || seg_id u64 || n u32 || leaf_0 || ... || leaf_{n-1} )`. The sender provides the leaf list in a table frame; the receiver checks it against `chunk_table_hash` before trusting any leaf.
- Domain tags (`0x00` leaf, `0x01` table) and the binding of `file_id`, `seg_id`, `chunk_index` and `chunk_len` prevent splicing between files, reordering, duplication and truncation of a chunk.

Hash cost is negligible next to transport (measured on this machine: SHA-256 2.6 GB/s, BLAKE3 9.2 GB/s; a 1 GiB body costs about 0.4 s of SHA-256).

### 4. Frames (transport-neutral)

- `0x01 CHUNK`: `seg_id u64 | chunk_index u32 | len u32 | bytes`.
- `0x02 CHUNK_TABLE`: `seg_id u64 | n u32 | n * 32-byte leaf hashes`.
- The descriptor is delivered as `0x00 DESCRIPTOR | len u32 | bytes` when inline.
- A frame's declared sizes are checked against the descriptor before any allocation. At most one chunk per stream is buffered.

### 5. Receiver state machine

```
AwaitDescriptor --(invariants 1-5 hold)--> Ready            [witness DescriptorVerified]
AwaitDescriptor --(any failure)----------> Rejected         [witness DescriptorRejected(reason)]
Ready/Receiving: per segment  Pending -> TableOk (n>1) -> chunks verified -> SegmentVerified
Receiving --(every segment verified, bytes == body_len, identity matches)--> Complete
Receiving --(connection drop / timeout)--> Suspended (resumable at chunk granularity, never Complete)
any --(hash mismatch beyond retry budget, duplicate/out-of-order beyond budget, size lie)--> Poisoned (terminal)
```

- Frames are accepted only for the exact next expected chunk of a segment. An out-of-order (future index) chunk, a frame for an unknown segment and a hash mismatch are counted faults; exceeding the fault budget poisons the stream. An authentic retransmission is **not** a fault: a chunk with an index below the segment's next expected index whose bytes hash to the already verified leaf (and a byte-identical descriptor frame after admission) is an idempotent duplicate, ignored with event `Duplicate`, nothing re-released. Duplicates are bounded by a separate counter (`max_duplicate_frames`, default `4 * total chunks + 64`); exceeding it poisons with `StreamDuplicateFlood`. A below-index chunk whose bytes do not hash to the verified leaf is an ordinary hash-mismatch fault.
- `Poisoned` is terminal: consumers are told to revoke granted capabilities, terminate instances started from the stream and drop index layers; a witness `StreamPoisoned(seg_id, chunk_index, reason)` is emitted.
- The identity check (`SHA-256(body) == rvf_identity`) preserves ADR-155 identity. SHA-256 is sequential, so the verifier hashes only while chunks arrive contiguously in file order. A stream delivered in file order ends in `Complete` with the identity confirmed. Any other order (stage order, the point of M2) ends in the distinct terminal state `CompleteUnverified`: every chunk commitment is verified, but `rvf_identity` is **not** confirmed and must not be used as an identity until a storage-capable consumer re-reads its persisted body in file order and `finish_identity` succeeds (state becomes `Complete`; a mismatch makes it `Poisoned`). `finish_identity` does not trust the host: it re-reads the persisted body in bounded 64 KiB pieces, **re-verifies every chunk against the signed leaf commitments**, hashes the same pass for `rvf_identity`, and re-checks the pin; only then is the state `Complete`. A descriptor that claims another body's hash cannot be satisfied by supplying that body (`StreamChunkHashMismatch`; with matching chunks but a lying identity, `StreamIdentityMismatch`). The unsafe path is therefore not the default one: code that waits for `Complete` cannot skip the check.
- **Pin protocol.** Pins live in a `PinStore` (`get(signer_key_id, file_id)` and a compare-and-set `advance(signer_key_id, file_id, expected_current, new)` that never moves backwards; reference `MemoryPinStore`), keyed by `(signer_key_id, file_id)` with the descriptor hash mandatory. The pin is read at admission, **re-checked at completion** (automatically when the policy carries a store, and through `recheck_pin(&dyn PinStore)` before any derived release such as a sealed answer), and recorded only after `Complete` with the identity confirmed, by `commit_pin`, which is a compare-and-set against the pin seen at admission. A session superseded while it ran ends in `StreamStaleEpoch` (or `StreamEpochConflict`), never in `Complete`/`CompleteUnverified -> Complete`. One signer can never move another signer's pin.
- `Complete`, `CompleteUnverified` and `Suspended` are never overwritten by `Poisoned`; bytes arriving after a finished stream are refused (`StreamBadState`) without changing state. `Rejected` and `Poisoned` are terminal.

### 6. Consumption rules (no unverified bytes are ever used)

- A chunk is released to a consumer as a `VerifiedChunk` only after its leaf hash matches **and** every stage named in the segment's `requires_mask` (bit i = stage i) is fully verified. Authentication precedes parsing: parsers (index, wasmparser) only see released chunks.
- **Queries**: a `ProgressiveIndex` layer is used only once every segment of that layer is `SegmentVerified`; results carry a `QualityEnvelope` with `sealed = false` until `Complete`. A layer is never searched before its entry points are verified.
- **Executables**: bytes feed an incremental validator; executable memory is allocated and the module is started only when the whole segment is `SegmentVerified`, validation finished, and the module's length and SHA-256 match the verified executable identity (the existing `accepts_wasm` check). Monolithic modules cannot start before their last byte; components (ADR-287) may start once their own set is verified while later components still arrive. When `require_complete_before_exec` is set, nothing executes before `Complete`.
- Delivery order is the stage order chosen by the producer and signed in the table: stage 0 policy, capability and manifest segments; 1 Layer A; 2 executables and their data; 3 Layer B; 4 vectors; 5 Layer C and cold data.

### 7. Threat controls (from the dossier threat model)

| Threat | Control |
| --- | --- |
| Truncation | signed `body_len`, `seg_count`, contiguity; only `Complete` when all verified; partial results are marked `sealed = false` |
| Replay of an old valid version, or a session superseded while running | `(epoch, descriptor_hash)` pinned per `(signer_key_id, file_id)` in a compare-and-set `PinStore`, checked at admission and again at completion; optional `expires_ns`; clockless targets use a monotone counter anchored in the witness chain |
| Reorder, insert | exact-next-chunk rule; leaf binds `(file_id, seg_id, chunk_index, chunk_len)` |
| Replay / retransmit of authentic frames (also cross-tenant amplification on a shared package) | idempotent duplicates are not faults and release nothing; a separate duplicate cap (`StreamDuplicateFlood`) bounds the work; forged duplicates are hash-mismatch faults |
| Mid-stream substitution | per-chunk SHA-256 commitments; header `content_hash` is demoted to a cross-check |
| Splice between files signed by one key | `file_id` in every leaf |
| Size-lie resource exhaustion | limits from signed policy before allocation; one buffered chunk per stream; allocator-counting test |
| TOCTOU between verify and run | verify from verifier-owned memory; hand out `VerifiedChunk` tokens; copy the complete verified module once into partition-private memory; re-run `accepts_wasm` at start |
| Downgrade to non-streaming | a package that declares a stream profile is refused by the buffered loader unless signed policy permits; algorithm floor SHA-256 and `verify_strict` |
| Forged tail manifest | the stream path never uses tail scanning for authority |
| Parser bugs | the descriptor parser is the only parser of unauthenticated bytes: fixed layout, length-checked, fuzzed |

### 8. Profiles and milestones

- **M1 (segment-granular)**: `chunk_log2 = 0xFF`, one chunk per segment. Memory bound is the largest segment (policy-capped). Strict subset of M2: same parser and state machine, no table frames.
- **M2 (chunked)**: chunk tables, chunk-granular resume, stage-ordered delivery, progressive index ingest.
- **M3 (later)**: BLAKE3 (`hash_alg = 2`), ML-DSA-65, replay anchoring through the RVM witness chain.

### 9. API surface (proposed; absent today unless noted)

`rvf-types`: `StreamDescriptor`, `StreamEntry`, error codes beside `UnsignedManifest = 0x0800`. `rvf-crypto`: `sign_descriptor`, `verify_descriptor`, `leaf_hash`, `chunk_list_hash`. `rvf-wire`: `stream::{decode_descriptor, StreamVerifier::{new, push}}`, no_std plus alloc, push-style like `wasmparser::Parser::parse(data, eof)`. `rvf-runtime`: `StreamIngest` feeding `ProgressiveIndex`; `RvfStore::open_with_policy` (ADR-033, ADR-354). `rvf-server`: `GET /v1/stream/descriptor` and `GET /v1/stream/body` with stage-ordered frames (TCP types `0x05/0x85` are unused). RVM consumers are specified in RVM ADR-160.

### 10. Implementation notes (refinements recorded from the code; normative for interop)

Vectors: `crates/rvf/tests/vectors/rvsd-v1-vectors.json` with its README pins every byte below.

- **Errors**: `rvf-types` `ErrorCode` `0x0805..=0x081A` (`0x0818` reserved, unassigned; `0x081A` `StreamDuplicateFlood`), plus the existing `UnknownSigner` 0x0802 for a signer outside the trust set. The first refused frame reports its specific cause; poisoning by an exhausted fault budget keeps that cause.
- **Descriptor checks, in order**: size cap and fixed header; `seg_count` cap; exact length `0x90 + 72*seg_count + 66`; signature (signed message is the domain tag, a NUL byte, then every byte before the 64-byte signature including `sig_len`); expiry; expected `policy_hash`; epoch pin; table. The entry parser only sees signed bytes.
- **Extra descriptor rules**: `rvf_identity` is zero when `identity_present` is clear; at least one entry; `stage < 32`; a segment never requires its own stage and stage requirements are acyclic; `StreamEntry.flags` must be zero; the `0xFE` root-page pseudo-entry must be last, exactly 4096 bytes, `uncompressed_len` 0, and its first chunk must start with the Level-0 root magic.
- **Fault model**: a frame whose declared size contradicts the signed descriptor, a malformed table size, or a frame shorter than its header poisons the stream at once (`StreamFrameSizeLie`). Future-index, unknown-segment, unknown-type and chunk-before-table frames, a second chunk table, a different descriptor frame after admission, leaf or table hash mismatches (including a duplicate with different bytes), and a chunk whose `requires_mask` stages are not fully verified (`StreamRequirementsUnmet`; the chunk cannot be held because only one frame is buffered) are counted faults. Authentic duplicate chunks and a byte-identical repeated descriptor are idempotent (`Duplicate`, `DuplicateDescriptor`) and counted only against `max_duplicate_frames`; accepted chunk tables are retained to check duplicates of finished segments, so `max_table_bytes` (default 64 MiB) caps retained tables; exceeding `fault_budget` poisons. A refused frame never advances the expected chunk.
- **Algorithm floor**: after chunk 0 of an executable (Wasm, Kernel, eBPF) or index segment is verified, its segment header must carry the right magic and `seg_type`, and `checksum_algo` 0 or 1 poisons the stream (`StreamWeakChecksum`).
- **Memory**: the released `VerifiedChunk` borrows the frame passed to `push` (hashed in place); `FrameReader` buffers at most one frame, admitting it only after its declared size is checked against limits derived from the signed descriptor, with the total computed in checked `usize` (a 32-bit overflow is a size lie). Chunk tables held at once are capped by `max_table_bytes` (default 64 MiB); a table frame beyond the cap is refused before allocation.
- **Hashing**: SHA-256 is the portable `rvf_types::sha256` by default; the `fast-sha256` feature of `rvf-crypto` (on in `rvf-wire`) uses the `sha2` crate with runtime CPU detection. Outputs are identical. Measured on one machine: portable 0.45 GB/s, `sha2` 2.7 GB/s, verifier 1.35 GB/s (it hashes each chunk twice in file order: leaf and identity).
- **Epoch pin**: `EpochPin { epoch, descriptor_hash }`, hash mandatory, looked up by `(signer_key_id, file_id)` through the policy closure and/or the `PinStore` (see Pin protocol in section 5).
- **Resume**: `end_of_input` moves `Receiving` to `Suspended`; `resume` returns to `Receiving`; progress is kept per chunk.

## Acceptance

Each test fails closed and emits a witness. Shared known-answer vectors (`rvsd-v1-vectors.json`, generated here, consumed by `ruvnet/rvm`) pin the byte layout so the two independent implementations cannot drift.

- Any single-bit flip in the descriptor, a table frame or a chunk is rejected and nothing from the affected segment is released after the flip.
- Truncation at every byte offset never reaches `Complete`.
- Permutation, deletion or insertion of frames, and duplication with different bytes, never reach `Complete` with different bytes; authentic duplicates alone still reach `Complete` with the identical released set.
- An older epoch, or an equal epoch with a different descriptor hash, is rejected; a pin set for one signer does not constrain another signer's lineage.
- A validly hashed chunk from another file signed by the same key is rejected.
- A descriptor that lies about sizes is rejected before allocation (allocator-counting test).
- Whole-buffer verification and the stream verifier agree on accept and reject for a corpus, fed in file order (`Complete`) and in stage order (`CompleteUnverified`, then `finish_identity`).
- Descriptor decoder, frame state machine and chunk-split independence are fuzzed.

## Consequences

Positive: prefix and chunk authenticity; early rejection; bounded memory; resumability; existing `.rvf` files can be served streamed by generating a descriptor, with no rewrite.

Negative: a new signed artifact to produce, distribute and pin; descriptor generation needs a key; two implementations (`rvf-wire`, and `rvm-rvf` which restates the format) must stay in lock step, mitigated by shared vectors.

## What this ADR does not claim

It does not make RVM execute WASM: `ruvnet/rvm` has no execution engine today (RVM ADR-160). It does not change legacy files or readers. Benchmarks and measured numbers belong to the implementation, not to this ADR.

## Measured on the ruvector side (project Ofer, optimize pass)

Release builds, this workstation (32 threads, SHA-NI), load average 2-3, medians of at least 7 runs unless stated. 100k x 128 clustered vectors, 62 MB stream.

- **Query policy before Layer C (B2, recall@10 against exact ground truth).** `ProgressiveIndex` layers were worse than an exact scan over the vectors already verified; `StreamIngest::query` now scans the arrived vectors while the scan fits a latency bound (default 25 ms, calibrated from the measured per-vector cost) and reports `coverage_vectors`, `QueryPath::ArrivedScan` and `sealed = false`. Layers only: A 0.197, A+B 0.032, 40% of bytes 0.030, 60% 0.054, 80% 0.092, all vectors 0.106. Auto policy: 0.197, 0.197, 0.365, 0.601, 0.852, 1.000 (p95 latency 0.7, 0.7, 1.2, 2.1, 2.9-3.0, 3.4 ms). Layer C (0.953, 0.45 ms) answers once published.
- **Ingest (B9, 256 MiB of vectors).** Verifier alone 1.33 GB/s (two SHA-256 passes per byte in file order); `StreamIngest::feed` 757 -> 866 MB/s (+14%) by reserving arena room from the verified segment count and reusing the record buffer.
- **HTTP stream path (B10, loopback).** Raw download 12.1 GB/s; download + verify + ingest 743 MB/s (client bound). The idle-timeout rewrite (F2-05) briefly polled with sleeps and dropped raw download to 0.28 GB/s; it now parks on `send_timeout` and matches the pre-F2-05 12.2 GB/s.
- **SHA-256.** Every native consumer links `sha2` with runtime CPU detection: 2.70 GB/s with SHA-NI against 0.45 GB/s for the portable fallback, which only the wasm crates use.
