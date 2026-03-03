# ADR-058: RVF Hash Security Hardening and Optimization

**Status**: Accepted
**Date**: 2026-02-27
**Authors**: ruv.io, RuVector Architecture Team
**Deciders**: Architecture Review Board
**SDK**: Claude-Flow
**Relates to**: ADR-029 (RVF Canonical Format), ADR-042 (Security-RVF-AIDefence-TEE)

## Context

### Current Hash Implementation

The RVF wire format (`rvf-wire`) uses XXH3-128 as the sole content hash for all
segment integrity verification. The `checksum_algo` field in the 64-byte segment
header supports three values:

| Algo | Name | Status |
|------|------|--------|
| 0 | CRC32C | Deprecated — silently upgraded to XXH3-128 |
| 1 | XXH3-128 | Active — used for all operations |
| 2 | SHAKE-256 | Declared in enum but **never implemented** |

### Security Findings

A comprehensive review of `rvf-wire/src/hash.rs`, `rvf-types/src/checksum.rs`,
and the graph shard module (`ruvector-graph/src/distributed/shard.rs`) identified
six issues:

1. **Non-constant-time hash comparison (P1)**: `verify_content_hash` uses `==`
   on byte arrays. While XXH3-128 is not cryptographic, a timing side-channel
   could reveal partial hash values to an attacker probing segment files over a
   network interface. For defense-in-depth, verification should use
   constant-time comparison.

2. **SHAKE-256 declared but unimplemented (P2)**: `ChecksumAlgo::Shake256` (algo=2)
   exists in the enum and is accepted by `TryFrom<u8>`, but
   `compute_content_hash` ignores the algo parameter entirely — all paths route
   to XXH3-128. A writer could set `checksum_algo=2` in the header and it would
   silently verify against XXH3-128, creating a false sense of cryptographic
   integrity.

3. **Algo parameter ignored (P2)**: `compute_content_hash(_algo, data)` discards
   the algorithm selector. If a future writer uses algo=2, the verifier cannot
   detect the mismatch.

4. **No keyed/HMAC hash option (P3)**: The current scheme provides integrity
   (accidental corruption detection) but not authentication. For federated
   transfer scenarios (ADR-057), a keyed hash is needed to prevent a
   man-in-the-middle from replacing segment payloads while recomputing the hash.

5. **Dead CRC32C dependency (P3)**: `crc32c = "0.6"` remains in `Cargo.toml`
   even though CRC32C is fully deprecated. The `compute_crc32c` and
   `compute_crc32c_hash` functions are dead code.

6. **Graph shard uses XXH3-64 (P3)**: `ruvector-graph` shard routing uses
   `xxh3_64()` (64-bit). With 2^32 nodes the birthday bound gives ~50%
   collision probability for shard assignment. This is acceptable for current
   scale but noted for future growth.

### Performance Baseline

XXH3-128 on 1 MB payload: ~50 GB/s on AVX2 hardware (dominated by memory
bandwidth). No performance regression is expected from the changes below since
the hash function itself is not modified.

## Decision

### 1. Constant-Time Hash Verification

Replace the `==` comparison in `verify_content_hash` with a constant-time
byte-equality check using `subtle::ConstantTimeEq`. This eliminates the timing
side-channel at negligible cost (~2 ns overhead for 16 bytes).

### 2. Wire SHAKE-256 Implementation

Implement `compute_shake256_128(data)` using the `sha3` crate (already a
dependency of `rvf-crypto`). Route `algo=2` in `compute_content_hash` to this
implementation. This makes the `ChecksumAlgo::Shake256` enum variant truthful.

SHAKE-256 truncated to 128 bits provides:
- 128-bit collision resistance (same as XXH3-128)
- Post-quantum preimage resistance (vs XXH3's ~0 bits)
- ~300 MB/s throughput (vs XXH3's ~50 GB/s) — acceptable for
  security-sensitive segments where correctness matters more than speed

### 3. Honor the Algo Parameter

Make `compute_content_hash` dispatch on the algo value:
- 0 → XXH3-128 (CRC32C upgrade, backward compatible)
- 1 → XXH3-128
- 2 → SHAKE-256 (first 128 bits)
- other → XXH3-128 (fallback)

### 4. Remove Dead CRC32C Code

Remove `compute_crc32c()`, `compute_crc32c_hash()`, and the `crc32c` dependency.
Retain the `Crc32c = 0` enum variant for backward-compatible header parsing.

### 5. Add Keyed Hash Support (Algo=3)

Reserve `checksum_algo=3` for HMAC-SHAKE-256 (keyed integrity). Implementation
is deferred to a follow-up PR as it requires key management infrastructure.
Add the enum variant now so the wire format is forward-compatible.

## Consequences

### Positive

- Eliminates timing side-channel in hash verification
- SHAKE-256 segments can now be written and verified correctly
- Dead code removed, smaller dependency tree
- Wire format is forward-compatible with keyed hashing

### Negative

- `subtle` crate added as a dependency (~10 KB, widely audited)
- `sha3` crate added to `rvf-wire` (already in `rvf-crypto`)
- Writers that relied on the silent algo-mismatch behavior will now produce
  SHAKE-256 hashes when they set algo=2 (breaking change for any such writers,
  but none are known to exist)

### Risks

- The `subtle` crate's constant-time guarantees depend on the compiler not
  optimizing away the timing-safe operations. Rust's `subtle` v2.6+ uses
  inline-asm barriers on supported platforms.

## Implementation Plan

1. Add `subtle` and `sha3` dependencies to `rvf-wire/Cargo.toml`
2. Remove `crc32c` dependency and dead CRC32C functions
3. Implement `compute_shake256_128()` in `hash.rs`
4. Update `compute_content_hash()` to dispatch on algo
5. Update `verify_content_hash()` to use `subtle::ConstantTimeEq`
6. Add `HmacShake256 = 3` to `ChecksumAlgo` enum (reserved, no impl yet)
7. Update tests and benchmarks
8. Verify existing tests pass (no behavioral change for algo=0 and algo=1)

## Integration with mcp-brain-server

ADR-075 documents the integration of `rvf-crypto` (including the SHAKE-256 functions hardened by this ADR) into the Shared Brain server. The brain server's `verify.rs` module previously used inline `sha3::Shake256` calls; it now delegates to `rvf_crypto::shake256_256()` for content hashing and `rvf_crypto::create_witness_chain()` / `rvf_crypto::verify_witness_chain()` for tamper-evident audit trails. This ensures that the constant-time comparison and proper algo dispatch implemented here are used consistently across the stack.

See: [ADR-075 — Wire Full RVF AGI Stack into mcp-brain-server](ADR-075-rvf-agi-stack-brain-integration.md)
