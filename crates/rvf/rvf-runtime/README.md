<p align="center"><img src="https://raw.githubusercontent.com/ruvnet/ruvector/main/assets/ruvector/crates/rvf-runtime.svg" alt="rvf-runtime flow" width="100%"></p>

# rvf-runtime

RuVector Format runtime providing the RvfStore API, background compaction, and streaming I/O.

## Overview

`rvf-runtime` is the main entry point for applications that want to read and write RVF files:

- **RvfStore** -- high-level API for storing and retrieving vectors
- **HNSW query path** -- queries route through the persisted HNSW index when one is available
- **Compaction** -- background merge of segments to reclaim space
- **Streaming I/O** -- append-only writes with configurable flush policy

## Usage

```toml
[dependencies]
rvf-runtime = "0.5"
rvf-types = "0.3"
```

Create a store, query it, delete, and let the compaction policy decide when to
compact (`should_compact` / `compact_if_needed`, spec 10 §7 thresholds):

```rust
use rvf_runtime::compaction::CompactionDecision;
use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};
use rvf_types::RvfError;

fn main() -> Result<(), RvfError> {
    let path = std::env::temp_dir().join("quickstart.rvf");
    let _ = std::fs::remove_file(&path);

    // Create a 4-dimensional store and ingest three vectors
    let mut store = RvfStore::create(&path, RvfOptions { dimension: 4, ..Default::default() })?;
    let vectors = [[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    store.ingest_batch(&refs, &[1, 2, 3], None)?;

    // Nearest neighbours
    let hits = store.query(&[0.9, 0.1, 0.0, 0.0], 2, &QueryOptions::default())?;
    println!("top hit: id {} (distance {:.3})", hits[0].id, hits[0].distance);

    // Delete, then let the compaction policy decide whether to compact
    store.delete(&[2, 3])?;
    match store.should_compact() {
        CompactionDecision::None => println!("no compaction needed"),
        decision => println!("compaction advised: {decision:?}"),
    }
    if let Some(result) = store.compact_if_needed()? {
        println!("compacted: {result:?}");
    }

    store.close()?;
    Ok(())
}
```

## Features

- `std` (default) -- enable `std` I/O support
- `wasm` -- enable WASM-compatible runtime paths

## Query Path

`RvfStore::query` routes through the persisted HNSW index when an INDEX_SEG
is present in the file; otherwise it falls back to an exact brute-force scan.

- The index is persisted as an INDEX_SEG with a self-delimiting ID-mapping
  trailer (`"RVIX"` magic). Readers that only understand the plain INDEX_SEG
  codec ignore the trailer.
- Index rebuilds after ingest/delete are **non-blocking**: queries serve from
  the exact scan until the new index commits.
- `QueryOptions::force_exact` forces the exact scan even when an index is
  available (ground-truth comparison, benchmarking).
- Result ordering uses deterministic `(distance, id)` tie-breaking.

### RaBitQ opt-in (`QueryOptions::rabitq`)

Setting `rabitq: true` enables a two-stage path: a 1-bit-code candidate scan
(~32x smaller than f32) followed by an exact f32 rescore of the oversampled
candidates (`rabitq_oversample`, default 4x). v1 serves the L2 metric only;
other metrics and filtered/COW queries fall back to the default routing.

### In-Memory Vector Slab

In-memory vectors are stored in one contiguous row-major slab with an
id -> ordinal map (no per-vector heap allocation). Removals tombstone in
place; slots are reclaimed during compaction.

### Measured Performance

Environment: Windows x64, criterion release builds, 100k vectors x 64 dims, k=10.

| Benchmark | Baseline | Measured | Quality |
|-----------|----------|----------|---------|
| k-NN query via HNSW index | 21.7 ms (brute force) | **1.51 ms** | recall@10 0.968 |
| Brute-force scan (contiguous slab) | 24.5 ms (per-vector heap allocs) | **3.8 ms** | exact |
| Cold open (slab layout) | — | **-21.5%** open time | — |
| RaBitQ two-stage query (opt-in) | f32 codes | 32x code compression | recall@10 0.972 |

## Lineage Derivation

`RvfStore` supports DNA-style derivation chains where a parent store produces child stores with provenance linkage.

### `derive()` Method

Creates a child store that records this store as its parent. The child gets a new `file_id`, inherits dimensions and options, and records the parent's manifest hash for later verification:

```rust
use rvf_runtime::{RvfStore, options::RvfOptions};
use rvf_types::DerivationType;
use std::path::Path;

let parent = RvfStore::create(Path::new("parent.rvf"), options)?;
let child = parent.derive(
    Path::new("child.rvf"),
    DerivationType::Filter,
    None, // inherit parent options
)?;
assert_eq!(child.lineage_depth(), 1);
// Since 0.5.4 the type is recorded (lineage segment + witness entry):
assert_eq!(child.derivation_type()?, Some(DerivationType::Filter));
```

`derive()` records lineage only: the child starts empty for every type (use
`branch()` for a copy-on-write child that reads through to its parent).
`RvfStore::merge(&[&a, &b], path, None)` records several parents
(`DerivationType::Merge`, one `LineageRecord` per parent); the merged read path
is still being designed (#1289).

### Application segments

`append_segment(ty, key, value)`, `read_segment(ty, key)` and
`list_segments(ty)` store application records in the file as segments of a
`CustomSegmentType` (0xC0..=0xEF). They are checksummed, listed in the
manifest (and so covered by a signed store's root page), witnessed, and
carried through compaction; the latest value per key wins (#1290).

### FileIdentity Accessors

| Method | Return | Description |
|--------|--------|-------------|
| `file_id()` | `&[u8; 16]` | This file's unique identifier |
| `parent_id()` | `&[u8; 16]` | Parent file's identifier (zeros if root) |
| `lineage_depth()` | `u32` | Derivation depth (0 for root files) |
| `file_identity()` | `&FileIdentity` | Full 68-byte identity struct |

### Extension Aliasing and Domain Profiles

RVF files can use domain-specific extensions that are automatically detected on `create()` and `open()`:

| Extension | Domain Profile | Optimized For |
|-----------|---------------|---------------|
| `.rvf` | Generic | General-purpose vectors |
| `.rvdna` | RVDNA | Genomic sequence embeddings |
| `.rvtext` | RVText | Language model embeddings |
| `.rvgraph` | RVGraph | Graph/network node embeddings |
| `.rvvis` | RVVision | Image/vision model embeddings |

When a child is derived with `derive()`, the child's extension also controls its domain profile. For example, deriving a `.rvdna` child from a `.rvf` parent automatically sets the child's profile to RVDNA.

### FIDI Magic Marker

When `FileIdentity` is present (non-zero `file_id`), the manifest segment includes a 4-byte FIDI magic marker trailer followed by the 68-byte `FileIdentity`. This ensures backward compatibility: old readers that do not recognize the FIDI marker simply stop parsing the manifest payload at the expected end and ignore the trailing bytes.

## Computational Container

`rvf-runtime` provides low-level write-path support for the two computational container segment types defined in [ADR-030](../../../docs/adr/ADR-030-rvf-computational-container.md): KERNEL_SEG (`0x0E`) and EBPF_SEG (`0x0F`).

### Internal Write-Path API

The `SegmentWriter` exposes internal methods for writing computational container segments:

- `write_kernel_seg()` -- writes a KERNEL_SEG containing a 128-byte `KernelHeader`, a compressed kernel image, and an optional kernel command line.
- `write_ebpf_seg()` -- writes an EBPF_SEG containing a 64-byte `EbpfHeader`, eBPF program bytecode (ELF object), and optional BTF data.

These are `pub(crate)` methods used by the segment codec layer. Public `embed_kernel()` / `extract_kernel()` and `embed_ebpf()` / `extract_ebpf()` convenience methods on `RvfStore` are planned for Phase 2 and Phase 3 of the computational container implementation but are not yet available.

### Unknown Segment Preservation

During compaction, `rvf-runtime` preserves segments with unknown or unrecognized types. This means KERNEL_SEG and EBPF_SEG payloads written by newer tooling are retained even when compaction is performed by a runtime version that predates the computational container feature. The compactor copies unknown segments verbatim to the compacted output.

### Example: Writing a Test Stub Kernel Segment

```rust
use std::io::Cursor;

// Build a 128-byte KernelHeader (raw bytes for now; typed struct planned)
let mut kernel_header = [0u8; 128];
// Magic: "RVKN" (0x52564B4E) little-endian
kernel_header[0..4].copy_from_slice(&0x52564B4E_u32.to_le_bytes());
// header_version = 1
kernel_header[4..6].copy_from_slice(&1_u16.to_le_bytes());
// arch = 0x00 (x86_64)
kernel_header[6] = 0x00;
// kernel_type = 0xFE (TestStub)
kernel_header[7] = 0xFE;
// kernel_flags: HAS_QUERY_API (bit 4) | HAS_ADMIN_API (bit 6)
kernel_header[8..12].copy_from_slice(&0x0050_u32.to_le_bytes());
// min_memory_mb = 32
kernel_header[12..16].copy_from_slice(&32_u32.to_le_bytes());

let fake_image = b"test-kernel-stub";
let cmdline = b"console=ttyS0";

// Use the internal SegmentWriter (not public API)
// let (seg_id, offset) = writer.write_kernel_seg(
//     &mut output, &kernel_header, fake_image, Some(cmdline),
// )?;
```

### Planned Public API (Not Yet Implemented)

The following methods are planned for `RvfStore`:

```rust
// Embed a kernel image into the store's .rvf file
// store.embed_kernel(kernel_image, kernel_config)?;

// Extract the kernel image from an existing .rvf file
// let (header, image) = store.extract_kernel()?;

// Embed an eBPF program
// store.embed_ebpf(program_elf, ebpf_config)?;

// Extract the eBPF program
// let (header, elf) = store.extract_ebpf()?;
```

## License

MIT OR Apache-2.0

---

Built by [rUv](https://ruv.io) · [ruv.io](https://ruv.io) · [Source](https://github.com/ruvnet/ruvector/tree/main/crates/rvf/rvf-runtime)
