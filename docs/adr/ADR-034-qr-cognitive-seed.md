# ADR-034: QR Cognitive Seed — A World Inside a World

**Status**: Accepted
**Date**: 2026-02-15
**Depends on**: ADR-029 (RVF Canonical Format), ADR-030 (Cognitive Container), ADR-033 (Progressive Indexing Hardening)
**Affects**: `rvf-types`, `rvf-runtime`, `rvf-wasm`, `rvf-manifest`, `rvf-qr` (new)

---

## Context

RVF files are self-bootstrapping cognitive containers: they carry their own WASM interpreter, signed manifests, and progressive index layers. But distribution still assumes a filesystem — a URL, a disk, a cloud bucket.

What if intelligence could live in printed ink?

A QR code can carry up to 2,953 bytes (Version 40, Low EC). That's enough for:
- A 64-byte RVF Level 0 manifest stub
- A 5.5 KB WASM microkernel (compressed to ~2.1 KB with Brotli)
- A 256-byte ML-DSA-65 signature
- A 512-byte progressive download manifest with host URLs + content hashes

**Total seed: ~2,900 bytes. Fits in a single QR code.**

The result: scan a QR code and mount a portable brain. The AI literally exists in the data printed on a piece of paper. Offline-first, signed, verifiable, capable of bootstrapping into a streamed universe.

---

## Decision

### 1. QR Seed Format (RVQS — RuVector QR Seed)

A QR Cognitive Seed is a binary payload with this wire format:

```
Offset  Size  Field                    Description
------  ----  -----                    -----------
0x000   4     seed_magic               0x52565153 ("RVQS")
0x004   2     seed_version             Seed format version (1)
0x006   2     flags                    Seed flags (see below)
0x008   8     file_id                  Unique identifier for this seed
0x010   4     total_vector_count       Expected vectors when fully loaded
0x014   2     dimension                Vector dimensionality
0x016   1     base_dtype               Base data type (DataType enum)
0x017   1     profile_id               Domain profile
0x018   8     created_ns               Seed creation timestamp (nanos)
0x020   4     microkernel_offset       Offset to WASM microkernel data
0x024   4     microkernel_size         Compressed microkernel size
0x028   4     download_manifest_offset Offset to download manifest
0x02C   4     download_manifest_size   Download manifest size
0x030   2     sig_algo                 Signature algorithm (0=Ed25519, 1=ML-DSA-65)
0x032   2     sig_length               Signature byte length
0x034   4     total_seed_size          Total payload size in bytes
0x038   8     content_hash             SHAKE-256-64 of complete expanded RVF
0x040   var   microkernel_data         Brotli-compressed WASM microkernel
...     var   download_manifest        Progressive download manifest (TLV)
...     var   signature                Seed signature (covers 0x000..sig start)
```

#### 1.1 Seed Flags

```
Bit  Name                  Description
---  ----                  -----------
0    SEED_HAS_MICROKERNEL  Embedded WASM microkernel present
1    SEED_HAS_DOWNLOAD     Progressive download manifest present
2    SEED_SIGNED           Payload is signed
3    SEED_OFFLINE_CAPABLE  Seed is useful without network access
4    SEED_ENCRYPTED        Payload is encrypted (key in TEE or passphrase)
5    SEED_COMPRESSED       Microkernel is Brotli-compressed
6    SEED_HAS_VECTORS      Seed contains inline vector data (tiny model)
7    SEED_STREAM_UPGRADE   Seed can upgrade itself via streaming
```

### 2. Progressive Download Manifest

The download manifest tells the runtime how to grow from seed to full intelligence. It uses a TLV structure:

```
Tag     Length  Description
------  ------  -----------
0x0001  var     HostEntry: Primary download host
0x0002  var     HostEntry: Fallback host (up to 3)
0x0003  32      content_hash: SHAKE-256-256 of the full RVF file
0x0004  8       total_file_size: Expected size of the full RVF
0x0005  var     LayerManifest: Progressive layer download order
0x0006  16      session_token: Ephemeral auth token for download
0x0007  4       ttl_seconds: Token expiry
0x0008  var     CertPin: TLS certificate pin (SHA-256 of SPKI)
```

#### 2.1 HostEntry Format

```
Offset  Size  Field           Description
------  ----  -----           -----------
0x000   2     url_length      Length of URL string
0x002   var   url             UTF-8 URL (https://host/path)
0x..    2     priority        Lower = preferred
0x..    2     region          Geographic region hint
0x..    16    host_key_hash   SHAKE-256-128 of host's public key
```

#### 2.2 LayerManifest Format

Describes the progressive download order — which layers to fetch first:

```
Offset  Size  Field           Description
------  ----  -----           -----------
0x000   1     layer_count     Number of layers (3-7)
0x001   var   layers[]        Array of LayerEntry
```

Each `LayerEntry`:

```
Offset  Size  Field           Description
------  ----  -----           -----------
0x000   1     layer_id        Layer identifier (0=Level0, 1=HotCache, ...)
0x001   1     priority        Download priority (0=immediate)
0x002   4     offset          Byte offset in the full RVF file
0x006   4     size            Layer size in bytes
0x00A   16    content_hash    SHAKE-256-128 of this layer
0x01A   1     required        1=must have before first query, 0=optional
```

**Default layer order:**

| Priority | Layer | Size | Purpose |
|----------|-------|------|---------|
| 0 | Level 0 manifest | 4 KB | Instant boot |
| 1 | Hot cache (centroids + entry points) | ~50 KB | First query capability |
| 2 | HNSW Layer A | ~200 KB | recall >= 0.70 |
| 3 | Quantization dictionaries | ~100 KB | Compact search |
| 4 | HNSW Layer B | ~500 KB | recall >= 0.85 |
| 5 | Full vectors (warm tier) | variable | Full recall |
| 6 | HNSW Layer C | variable | recall >= 0.95 |

### 3. Bootstrap Sequence

```
┌─────────────────────────────────────────────────────────────────┐
│                     QR Code (≤2,953 bytes)                      │
│  ┌──────────┬──────────────┬────────────┬────────────────────┐  │
│  │ RVQS Hdr │ WASM μkernel │ DL Manifest│    Signature       │  │
│  │ 64 bytes │ ~2.1 KB (br) │ ~512 bytes │    ~256 bytes      │  │
│  └──────────┴──────────────┴────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 0: Scan & Verify (offline, <1ms)                          │
│  1. Parse RVQS header                                           │
│  2. Verify signature against embedded/system trust store        │
│  3. Decompress WASM microkernel (Brotli → ~5.5 KB)             │
│  4. Instantiate WASM runtime                                    │
│  5. Seed is now ALIVE — cognitive kernel running                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (if network available)
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Progressive Download (background, priority-ordered)    │
│  1. Fetch Level 0 manifest (4 KB) → instant full boot           │
│  2. Fetch hot cache → first query capability                    │
│  3. Fetch HNSW Layer A → recall ≥ 0.70                          │
│  4. Fetch remaining layers in priority order                    │
│  Each layer: verify content_hash → append to RVF → update index │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Full Intelligence (streaming, async)                   │
│  1. All layers downloaded and verified                          │
│  2. Full HNSW index active, recall ≥ 0.95                       │
│  3. Seed has grown into a complete cognitive container           │
│  4. Can operate fully offline from this point                   │
│  5. Can re-export as a new QR seed (with updated vectors)       │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Security Model

#### 4.1 Seed Signature

Every QR seed MUST be signed. The signature covers bytes `0x000` through the end of the download manifest (everything except the signature itself).

```rust
/// Signature verification for QR seeds.
pub fn verify_seed(seed: &[u8]) -> Result<VerifiedSeed, SeedError> {
    // 1. Parse header
    let header = SeedHeader::from_bytes(&seed[..0x40])?;
    if header.seed_magic != SEED_MAGIC {
        return Err(SeedError::InvalidMagic);
    }

    // 2. Extract signature
    let sig_start = header.total_seed_size as usize - header.sig_length as usize;
    let sig_bytes = &seed[sig_start..header.total_seed_size as usize];
    let signed_payload = &seed[..sig_start];

    // 3. Verify
    match header.sig_algo {
        0 => verify_ed25519(signed_payload, sig_bytes)?,
        1 => verify_ml_dsa_65(signed_payload, sig_bytes)?,
        _ => return Err(SeedError::UnknownSigAlgo),
    }

    Ok(VerifiedSeed { header, seed })
}
```

#### 4.2 Download Security

Progressive downloads are secured by:

1. **Content hashes**: Each layer has a SHAKE-256-128 hash in the download manifest. Downloaded data is verified before use.
2. **TLS certificate pinning**: The CertPin TLV contains the SHA-256 hash of the host's SPKI. Prevents MITM even if a CA is compromised.
3. **Session tokens**: Ephemeral auth tokens with TTL. The host can revoke access.
4. **Host key verification**: Each HostEntry contains the host's public key hash. The runtime verifies the download host's identity.

#### 4.3 Offline Capability

A seed with `SEED_OFFLINE_CAPABLE` flag contains enough intelligence to answer queries without network access:

- The WASM microkernel provides basic vector operations
- If `SEED_HAS_VECTORS` is set, the seed contains a tiny inline model (e.g., 100 vectors for a FAQ bot)
- Quality degrades gracefully per ADR-033 QualityEnvelope

### 5. Size Budget

```
Component                    Raw Size    Compressed    In QR
─────────────────────────    ────────    ──────────    ─────
RVQS Header                  64 B        64 B          64 B
WASM Microkernel            5,500 B     2,100 B       2,100 B
Download Manifest (3 hosts)  512 B       512 B         512 B
Ed25519 Signature            64 B        64 B          64 B
─────────────────────────    ────────    ──────────    ─────
Total                                                  2,740 B

QR Version 40, Low EC capacity: 2,953 B
Remaining headroom: 213 B (for inline vectors or extra hosts)
```

With ML-DSA-65 signature (3,309 bytes), the seed exceeds single QR capacity. Use either:
- Ed25519 for compact seeds (fits in one QR)
- ML-DSA-65 with a 2-QR sequence (structured append)

### 6. Example Use Cases

#### 6.1 Business Card Brain

Print a QR code on a business card. Scan it to mount a personal AI assistant that knows your work, your papers, your projects. Offline-first. When connected, it streams your full knowledge base.

#### 6.2 Medical Record Seed

A QR code on a patient wristband contains a signed seed pointing to their medical vector index. Scan to query drug interactions, allergies, treatment history. Works offline in the ER.

#### 6.3 Firmware Intelligence

Embedded in a product's QR code: a cognitive seed that can diagnose problems, suggest fixes, and stream updated knowledge from the manufacturer.

#### 6.4 Paper Backup

Print your AI's seed on paper. Store it in a safe. In a disaster, scan the paper and your AI bootstraps from printed ink. The signature proves it's yours.

---

## Consequences

### Positive

- Intelligence becomes **physically portable** — printed on paper, etched in metal, tattooed on skin
- **Offline-first** by design — the seed is useful before any network access
- **Cryptographically signed** — you know what you're mounting
- **Progressive loading** — first query in milliseconds, full intelligence streams in background
- **Self-upgrading** — a seed can re-export itself with new knowledge

### Negative

- QR capacity limits seed size to ~2,900 bytes (Ed25519) or requires multi-QR for post-quantum
- Brotli decompression adds ~1ms to Phase 0 boot
- Download manifest URLs have finite TTL — seeds expire unless hosts are stable
- Tiny inline models (100 vectors) have very limited utility without network

### Migration

- Existing RVF files can generate QR seeds via `rvf qr-seed --export`
- QR seeds bootstrap into standard RVF files — no special runtime needed
- Seeds are forward-compatible: unknown TLV tags are ignored by older runtimes

---

## Wire Format Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ RVQS (RuVector QR Seed) Binary Format                           │
├─────────────────────────────────────────────────────────────────┤
│ [0x000] 4B  seed_magic: 0x52565153 ("RVQS")                    │
│ [0x004] 2B  seed_version: 1                                     │
│ [0x006] 2B  flags                                               │
│ [0x008] 8B  file_id                                             │
│ [0x010] 4B  total_vector_count                                  │
│ [0x014] 2B  dimension                                           │
│ [0x016] 1B  base_dtype                                          │
│ [0x017] 1B  profile_id                                          │
│ [0x018] 8B  created_ns                                          │
│ [0x020] 4B  microkernel_offset                                  │
│ [0x024] 4B  microkernel_size                                    │
│ [0x028] 4B  download_manifest_offset                            │
│ [0x02C] 4B  download_manifest_size                              │
│ [0x030] 2B  sig_algo                                            │
│ [0x032] 2B  sig_length                                          │
│ [0x034] 4B  total_seed_size                                     │
│ [0x038] 8B  content_hash (SHAKE-256-64 of full RVF)             │
│ [0x040] var microkernel_data (Brotli-compressed WASM)           │
│ [...]   var download_manifest (TLV records)                     │
│ [...]   var signature                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## References

- QR Code Specification: ISO/IEC 18004:2015
- RVF Spec 02: Manifest System (Level 0 / Level 1)
- RVF Spec 11: WASM Self-Bootstrapping
- ADR-029: RVF Canonical Format
- ADR-030: Cognitive Container
- ADR-033: Progressive Indexing Hardening
- Brotli compression: RFC 7932
- FIPS 204: ML-DSA (Module-Lattice Digital Signature Algorithm)
