# rvm-witness

Append-only witness trail with keyed-BLAKE3 chain MACs (v2) and Merkle
segment sealing.

Implements ADR-134: every privileged action emits a witness record before
the mutation is committed. If emission fails, the mutation does not
proceed ("no witness, no mutation"). Records are stored in a fixed-capacity
ring buffer and linked by a MAC chain for tamper-evident auditing.

## Formats

- **v2 (current write format)** -- 96-byte records, 128-bit keyed-BLAKE3
  chain links, Merkle segment sealing. One keyed compression per append.
- **v1 (legacy, verify-only)** -- 64-byte records, 32-bit folded hash
  links. **Writing new v1 logs is removed.** v1 is frozen and retained
  only so existing logs keep verifying (`verify_chain`,
  `verify_chain_v1_with_head`) and so in-kernel rings can migrate
  incrementally. Serialized logs may contain a v1 prefix followed by v2
  records; `verify_log_bytes` dispatches on the version byte at offset 19
  of each record (v1 = `0`, v2 = `2`).

## v2 Record Layout (96 bytes, little-endian)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 8 | sequence (u64) |
| 8 | 8 | timestamp_ns (u64) |
| 16 | 1 | action_kind (u8) |
| 17 | 1 | proof_tier (u8) |
| 18 | 1 | flags (u8) |
| 19 | 1 | version (u8, always 2) |
| 20 | 4 | actor_partition_id (u32) |
| 24 | 8 | target_object_id (u64) |
| 32 | 4 | capability_hash (u32) |
| 36 | 8 | payload |
| 44 | 8 | aux (not chained) |
| 52 | 4 | reserved (zero) |
| 56 | 16 | prev_mac (predecessor's chain_mac) |
| 72 | 16 | chain_mac |
| 88 | 8 | pad (zero) |

`chain_mac = trunc128(BLAKE3_keyed(key, bytes[0..44] || prev_mac))`. The
60-byte MAC input fits one BLAKE3 block, so each append costs exactly one
keyed compression. The single MAC provides both self-integrity and the
chain link; forging either requires the chain key.

## Merkle Segment Sealing

Appends accumulate each record's `chain_mac` as a Merkle leaf (a memcpy).
`WitnessLogV2::seal_segment` computes the segment's Merkle root
(domain-separated: leaf = `BLAKE3(0x00 || seq || mac)`, node =
`BLAKE3(0x01 || l || r)`) and signs the chained seal digest
`BLAKE3(0x02 || root || first_seq || count || prev_seal_digest)` -- one
signature per segment (default 256 records), CT/QMDB-style, off the
per-record path. Sealed roots can be anchored externally;
`SegmentAccumulator::inclusion_proof` exports per-record Merkle inclusion
proofs verified by `verify_inclusion`.

Hardening (seal-time only, zero per-append cost):

- **Chained seals (R1)** -- each seal binds its predecessor's digest
  (genesis constant for the first), so append-only ordering of the whole
  sealed history is verifiable from seals alone via `verify_seal_chain`
  (`verify_seal_chain_binding` needs no key at all). Splice, reorder,
  omission, and cross-log transplant all break the binding.
- **Key ratchet (R4)** -- every seal ratchets the chain MAC key
  (`ratchet_chain_key`) and erases the old one, atomically with the
  seal. Compromise window = the current unsealed segment only; verifiers
  holding the initial key re-derive all epochs
  (`verify_chain_v2_ratcheted`).
- **Coverage policy (R6)** -- `CoveragePolicy::Strict` (via
  `WitnessLogV2::with_policy`) makes `try_append` return backpressure
  (`CoverageError`) instead of silently dropping Merkle coverage or
  overwriting unsealed records. Pre-existing constructors keep
  `BestEffort` counter behavior.

## v1 Record Layout (64 bytes, legacy)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 8 | sequence (u64) |
| 8 | 8 | timestamp_ns (u64) |
| 16 | 1 | action_kind (u8) |
| 17 | 1 | proof_tier (u8) |
| 18 | 2 | flags (u16) |
| 20 | 4 | actor_partition_id (u32) |
| 24 | 4 | target_object_id (u32) |
| 28 | 4 | capability_hash (u32) |
| 32 | 8 | payload (u64) |
| 40 | 8 | prev_hash (u64) |
| 48 | 8 | record_hash (u64) |
| 56 | 8 | aux (u64) |

## Key Types

- `WitnessLogV2<N, SEG>` -- v2 ring buffer with embedded segment accumulator
- `WitnessLog<N>` -- legacy v1 ring buffer (kernel-internal compatibility)
- `WitnessEmitter` -- builds records with auto-incrementing sequence and hash chain
- `WitnessRecord`, `WitnessRecordV2`, `ActionKind` -- record types (in `rvm-types`)
- `SegmentAccumulator`, `SealedSegment`, `MerkleProof` -- Merkle sealing
- `SegmentSealSigner`, `Blake3SealSigner` -- seal signing (see also
  `rvm_proof::SealSignerAdapter` for HMAC/Ed25519/TEE signers)
- `verify_chain_v2`, `verify_chain_v2_from` -- v2 chain verification
- `verify_log_bytes` -- versioned byte-stream verification (v1, v2, mixed)
- `v1_head_to_genesis` -- anchor a verified v1 head into a v2 genesis
- `verify_chain`, `ChainIntegrityError` -- legacy v1 verification
- `WitnessSigner`, `HmacWitnessSigner` -- legacy per-record signing (v1)

## Example

```rust
use rvm_types::WitnessRecordV2;
use rvm_witness::{Blake3SealSigner, WitnessLogV2, verify_inclusion, verify_seal};

let log = WitnessLogV2::<256>::new(); // default key: dev only
let mut r = WitnessRecordV2::zeroed();
r.target_object_id = 42;
log.append(r); // one keyed BLAKE3 compression

let signer = Blake3SealSigner::new([7u8; 32]);
let (sealed, acc) = log.seal_segment(&signer).unwrap();
assert!(verify_seal(&sealed, &signer));
let proof = acc.proof_for_sequence(0).unwrap();
assert!(verify_inclusion(&sealed.root, &acc.leaf(0).unwrap(), &proof));
```

## Performance

- Per-record append: one keyed BLAKE3 compression + bookkeeping
  (target < 1 us; see `benches/benches/witness.rs`)
- Seal: 256 leaf hashes + 255 node hashes + 1 signature per segment

## Design Constraints

- **DC-10**: Epoch-based witness batching (no per-switch records)
- **DC-15**: `#![no_std]`, `#![forbid(unsafe_code)]`, `#![deny(missing_docs)]`
- ADR-134 v2: 96-byte versioned record; keyed-BLAKE3 chain; v1 verify-only

## Workspace Dependencies

- `rvm-types`
- `blake3` (portable `pure` build, no_std)
- `spin`
