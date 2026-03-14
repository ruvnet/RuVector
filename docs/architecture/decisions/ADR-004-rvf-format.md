# ADR-004: RVF Cognitive Container Format

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-03-13 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | Architecture Review Board |
| **Supersedes** | - |
| **Related** | ADR-003-mcp-protocol, ADR-005-cross-platform-bindings |

## 1. Context

### 1.1 The Vector Storage Problem

Modern AI applications require storing and distributing vector embeddings with:

| Requirement | Challenge |
|-------------|-----------|
| Portability | Embeddings must work across platforms (native, WASM, mobile) |
| Provenance | Track model, version, timestamp, and author |
| Integrity | Detect tampering and verify authenticity |
| Executability | Include compute kernels for self-contained inference |
| Efficiency | Minimize storage and transmission overhead |

### 1.2 Existing Formats Limitations

| Format | Portability | Provenance | Integrity | Compute | Efficiency |
|--------|-------------|------------|-----------|---------|------------|
| NumPy .npy | Poor (Python-only) | None | None | None | Good |
| HDF5 | Medium | Some | None | None | Good |
| Safetensors | Good | Limited | CRC only | None | Excellent |
| ONNX | Good | Model-level | Weak | Inference | Medium |
| **RVF** | **Excellent** | **Full** | **Cryptographic** | **WASM** | **Excellent** |

### 1.3 The "Cognitive" Concept

A cognitive container is more than data storage - it contains:

1. **Memory**: The vector data itself (embeddings, indexes, quantization)
2. **Identity**: Cryptographic signatures and provenance chain
3. **Behavior**: Executable WASM kernels for self-contained compute
4. **Witness**: Audit trail of accesses, transformations, and attestations

## 2. Decision

### 2.1 RVF (RuVector Format) as Canonical Format

We adopt RVF as the universal format for vector data with a segment-based architecture:

```
+------------------+------------------+------------------+
|   FILE HEADER    |   SEG 1 HEADER   |   SEG 1 PAYLOAD  |
|    (64 bytes)    |   (64 bytes)     |   (variable)     |
+------------------+------------------+------------------+
|   SEG 2 HEADER   |   SEG 2 PAYLOAD  |   ...            |
|   (64 bytes)     |   (variable)     |                  |
+------------------+------------------+------------------+
```

### 2.2 File Header (64 bytes)

```rust
#[repr(C)]
pub struct RvfHeader {
    pub magic: [u8; 4],           // "RVF\x00"
    pub version: u16,             // Format version (currently 2)
    pub flags: u16,               // File-level flags
    pub segment_count: u32,       // Number of segments
    pub total_size: u64,          // Total file size in bytes
    pub created_at: u64,          // Unix timestamp (microseconds)
    pub modified_at: u64,         // Unix timestamp (microseconds)
    pub manifest_offset: u64,     // Offset to MANIFEST_SEG
    pub manifest_size: u32,       // Size of manifest segment
    pub root_hash: [u8; 32],      // Merkle root of all segments
}
```

### 2.3 Segment Header (64 bytes)

```rust
#[repr(C)]
pub struct SegmentHeader {
    pub magic: [u8; 4],           // Segment magic (per type)
    pub seg_type: u8,             // Segment type enum
    pub compression: u8,          // Compression algorithm
    pub flags: u16,               // Segment flags
    pub offset: u64,              // Offset in file
    pub compressed_size: u64,     // Size on disk
    pub uncompressed_size: u64,   // Size after decompression
    pub checksum: [u8; 32],       // SHA-256 of payload
    pub reserved: [u8; 8],        // Future use
}
```

### 2.4 Segment Types

| Type | Code | Magic | Description |
|------|------|-------|-------------|
| MANIFEST | 0x01 | RVMF | JSON manifest with metadata |
| VECTORS | 0x02 | RVVE | Dense vector data |
| HNSW | 0x03 | RVHN | HNSW index structure |
| QUANTIZED | 0x04 | RVQT | Quantized vectors (scalar/PQ/binary) |
| CODEBOOK | 0x05 | RVCB | PQ codebook data |
| METADATA | 0x06 | RVMD | Per-vector metadata (JSON) |
| SPARSE | 0x07 | RVSP | Sparse vector data (CSR format) |
| CRYPTO | 0x08 | RVCY | Cryptographic signatures |
| WASM | 0x09 | RVWM | WASM microkernel |
| WITNESS | 0x0A | RVWT | Audit trail / attestations |
| GRAPH | 0x0B | RVGR | Graph structure (edges, types) |
| HOT | 0x0C | RVHT | Hot tier vectors (uncompressed) |
| DELTA | 0x0D | RVDL | Incremental updates |

### 2.5 Manifest Segment Schema

```json
{
  "$schema": "https://ruvector.dev/schemas/rvf-manifest-v2.json",
  "version": "2.0",
  "id": "rvf:a1b2c3d4e5f6...",
  "name": "my-knowledge-base",
  "description": "Embeddings for documentation corpus",

  "vectors": {
    "dimension": 384,
    "count": 1000000,
    "dtype": "f32",
    "metric": "cosine"
  },

  "provenance": {
    "model": "text-embedding-3-small",
    "model_version": "2024-01-01",
    "created_by": "ruv.io",
    "created_at": "2026-03-13T10:00:00Z",
    "source": "corpus://docs.ruvector.dev"
  },

  "segments": [
    { "type": "VECTORS", "offset": 128, "size": 1536000000 },
    { "type": "HNSW", "offset": 1536000128, "size": 45000000 },
    { "type": "CRYPTO", "offset": 1581000128, "size": 2048 },
    { "type": "WASM", "offset": 1581002176, "size": 5632 }
  ],

  "indexes": {
    "primary": {
      "type": "hnsw",
      "params": { "M": 32, "ef_construction": 200 }
    }
  },

  "quantization": {
    "type": "none",
    "original_dtype": "f32"
  },

  "crypto": {
    "algorithm": "ML-DSA-65",
    "public_key": "...",
    "signatures": ["..."]
  }
}
```

## 3. WASM Microkernel

### 3.1 Purpose

The WASM segment (WASM_SEG) contains a portable compute kernel enabling:

1. **Self-contained search**: Query vectors without external runtime
2. **Cross-platform**: Same binary runs native, browser, edge
3. **Verified compute**: Hash-pinned, tamper-proof execution
4. **Hot updates**: Replace kernel without rebuilding index

### 3.2 WASM Interface (WIT)

```wit
package ruvector:kernel@0.1.0;

interface vectors {
    record search-result {
        id: string,
        score: float32,
    }

    /// Search k nearest neighbors
    search: func(query: list<float32>, k: u32) -> list<search-result>;

    /// Get vector by ID
    get: func(id: string) -> option<list<float32>>;

    /// Compute distance between two vectors
    distance: func(a: list<float32>, b: list<float32>) -> float32;
}

interface index {
    /// Add vector to index
    insert: func(id: string, vector: list<float32>) -> result<_, string>;

    /// Remove vector from index
    delete: func(id: string) -> result<_, string>;

    /// Get index statistics
    stats: func() -> index-stats;

    record index-stats {
        count: u64,
        dimension: u32,
        memory-bytes: u64,
    }
}

world rvf-kernel {
    export vectors;
    export index;
}
```

### 3.3 Kernel Size Budget

Target: <8 KB compressed WASM

| Component | Size |
|-----------|------|
| HNSW search | ~2 KB |
| Distance functions | ~1 KB |
| Vector decompression | ~1 KB |
| Memory management | ~500 B |
| Interface glue | ~500 B |
| **Total** | **~5.5 KB** |

## 4. Cryptographic Signatures

### 4.1 Signing Scheme

We use **ML-DSA-65** (FIPS 204) for post-quantum security:

| Property | Value |
|----------|-------|
| Algorithm | Module-Lattice Digital Signatures |
| Security Level | NIST Level 3 (~128-bit classical) |
| Public Key | 1,952 bytes |
| Signature | 3,309 bytes |
| Sign Time | ~100us |
| Verify Time | ~80us |

### 4.2 Signature Coverage

```rust
pub struct CryptoSegment {
    pub algorithm: SignatureAlgorithm,
    pub public_key: Vec<u8>,
    pub signatures: Vec<SegmentSignature>,
}

pub struct SegmentSignature {
    pub segment_type: u8,
    pub segment_hash: [u8; 32],  // SHA-256 of segment payload
    pub signature: Vec<u8>,      // ML-DSA-65 signature
    pub signed_at: u64,
    pub signer_id: String,
}
```

### 4.3 Witness Chain

The WITNESS_SEG maintains an append-only log:

```rust
pub struct WitnessEntry {
    pub sequence: u64,
    pub timestamp: u64,
    pub event_type: WitnessEventType,
    pub actor: String,
    pub payload_hash: [u8; 32],
    pub previous_hash: [u8; 32],  // Hash chain
    pub attestation: Option<TeeAttestation>,
}

pub enum WitnessEventType {
    Created,
    Modified,
    Accessed,
    Verified,
    Transferred,
    TeeExecution { enclave_id: String },
}
```

## 5. Compression

### 5.1 Compression Options

| Algorithm | Code | Ratio | Decode Speed | Use Case |
|-----------|------|-------|--------------|----------|
| None | 0x00 | 1.0x | N/A | Hot vectors |
| LZ4 | 0x01 | 1.5-2x | 4 GB/s | General |
| Zstd | 0x02 | 2-4x | 1.5 GB/s | Storage |
| Brotli | 0x03 | 3-5x | 500 MB/s | Network |

### 5.2 Tiered Compression Strategy

```
HOT_SEG:      None (immediate access, frequently queried)
VECTORS:      LZ4 (fast decode, ~1.8x compression)
HNSW:         Zstd (good compression, one-time decode)
WITNESS:      Zstd (archival, append-only)
WASM:         Brotli (network transfer, one-time)
METADATA:     Zstd (rarely accessed, high ratio)
```

## 6. Implementation

### 6.1 Reader API

```rust
pub struct RvfReader {
    file: MmapFile,
    header: RvfHeader,
    segment_index: HashMap<u8, Vec<SegmentHeader>>,
}

impl RvfReader {
    /// Open RVF file (memory-mapped)
    pub fn open(path: &Path) -> Result<Self>;

    /// Get manifest
    pub fn manifest(&self) -> Result<Manifest>;

    /// Load vectors (lazily decompresses)
    pub fn vectors(&self) -> Result<VectorIterator>;

    /// Load HNSW index
    pub fn hnsw_index(&self) -> Result<HnswIndex>;

    /// Verify all signatures
    pub fn verify(&self) -> Result<VerificationResult>;

    /// Execute WASM kernel
    pub fn kernel(&self) -> Result<RvfKernel>;

    /// Search using embedded kernel
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
}
```

### 6.2 Writer API

```rust
pub struct RvfWriter {
    segments: Vec<PendingSegment>,
    manifest: ManifestBuilder,
}

impl RvfWriter {
    pub fn new() -> Self;

    /// Add dense vectors
    pub fn add_vectors(
        &mut self,
        vectors: impl Iterator<Item = (String, Vec<f32>)>,
        compression: Compression,
    ) -> &mut Self;

    /// Add HNSW index
    pub fn add_hnsw(&mut self, index: &HnswIndex) -> &mut Self;

    /// Sign with private key
    pub fn sign(&mut self, key: &SigningKey) -> &mut Self;

    /// Add WASM kernel
    pub fn add_kernel(&mut self, wasm_bytes: &[u8]) -> &mut Self;

    /// Write to file
    pub fn write(&self, path: &Path) -> Result<()>;
}

// Builder pattern example
RvfWriter::new()
    .add_vectors(embeddings.iter(), Compression::Lz4)
    .add_hnsw(&index)
    .add_kernel(include_bytes!("kernel.wasm"))
    .sign(&my_key)
    .write("output.rvf")?;
```

## 7. Consequences

### 7.1 Benefits

1. **Self-Contained**: Single file contains data, index, and compute
2. **Verifiable**: Cryptographic signatures ensure integrity
3. **Portable**: WASM kernel runs anywhere
4. **Auditable**: Witness chain tracks all operations
5. **Efficient**: Tiered compression optimizes storage/speed
6. **Future-Proof**: Post-quantum signatures (ML-DSA-65)

### 7.2 Costs

1. **Complexity**: Multiple segment types to support
2. **Overhead**: Header/manifest add ~10 KB minimum
3. **Build Dependency**: WASM compilation toolchain needed

### 7.3 Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Safetensors | No index, no compute, no provenance |
| ONNX | Model-focused, not vector-focused |
| Parquet | No index support, column-oriented |
| Custom binary | Need all features of RVF anyway |
| ZIP container | No streaming, no random access |

### 7.4 File Size Analysis

Example: 1M 384-dim vectors

| Component | Raw Size | Compressed | Ratio |
|-----------|----------|------------|-------|
| Vectors (f32) | 1.5 GB | ~850 MB (LZ4) | 1.76x |
| HNSW index | 90 MB | ~45 MB (Zstd) | 2x |
| Manifest | ~2 KB | ~1 KB | - |
| Crypto segment | ~8 KB | ~8 KB | - |
| WASM kernel | ~6 KB | ~4 KB | - |
| **Total** | **1.59 GB** | **~895 MB** | **1.78x** |

## 8. Related Decisions

- **ADR-001-core-simd-strategy**: SIMD used in WASM kernel distance functions
- **ADR-005-cross-platform-bindings**: RVF kernel runs on all platforms
- **ADR-007-differential-privacy**: PII stripping before embedding in RVF

## 9. References

1. RVF Specification: `/crates/rvf/docs/specification.md`
2. WASM Kernel: `/crates/rvf-kernel/`
3. ML-DSA-65 (FIPS 204): https://csrc.nist.gov/pubs/fips/204/final
4. LZ4 Specification: https://lz4.github.io/lz4/

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-13 | Architecture Team | Initial decision record |
