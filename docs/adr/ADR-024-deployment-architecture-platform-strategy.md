# ADR-024: Deployment Architecture & Platform Strategy

| Field | Value |
|-------|-------|
| **Status** | Proposed |
| **Date** | 2026-02-11 |
| **Authors** | RuVector Architecture Team |
| **Reviewers** | - |
| **Version** | 0.2 |
| **Supersedes** | - |
| **Superseded by** | - |
| **Related** | ADR-005 (WASM Runtime), ADR-003 (SIMD Optimization), ADR-001 (Core Architecture) |

## 1. Context

### 1.1 Problem Statement

The RuVector DNA Analyzer must operate across a spectrum of deployment targets -- from clinical HPC clusters processing thousands of whole genomes daily, to a single browser tab running a point-of-care variant caller at a remote field clinic with no internet. The existing codebase already produces native binaries, WASM modules (18+ `*-wasm` crates), and Node.js bindings (6+ `*-node` crates), but there is no unified deployment strategy that specifies how these artifacts compose into platform-specific pipelines, what capabilities each platform gains or loses, or how cross-platform data flows are coordinated.

### 1.2 Decision Drivers

- Genomics data is privacy-sensitive (HIPAA, GDPR, local regulations); compute-local-to-data is a regulatory advantage.
- Nanopore and short-read sequencers are increasingly deployed outside traditional lab settings.
- FPGA-accelerated basecalling is transitioning from proprietary to open toolchains.
- The MCP protocol enables AI assistants to invoke analysis tools, creating a new interaction paradigm for genomic interpretation.
- A single Rust codebase compiled to multiple targets is already the project's core architectural bet; this ADR formalizes the deployment topology around that bet.

## 2. Platform Matrix

The following matrix defines every supported deployment target, mapping each to its use case, the crate graph it activates, and its concrete constraints.

| Platform | Use Case | Crate Surface | SIMD | Memory | Storage | Network | Limitations |
|----------|----------|---------------|------|--------|---------|---------|-------------|
| **Native x86_64** | Clinical labs, HPC, cloud | Full workspace (80 crates) | AVX-512, AVX2, SSE4.2 | Unbounded | mmap + redb | Full | Server required |
| **Native ARM64** | Edge devices, Apple Silicon, Graviton | Full workspace | NEON, SVE/SVE2 | 4-64 GB typical | mmap + redb | Full | Lower single-core clock |
| **Native RISC-V** | Future processors, academic | Full workspace (feature-gated) | V-extension (draft) | Variable | mmap + redb | Full | Early ecosystem, limited SIMD |
| **WASM Browser** | Point-of-care, education, personal genomics | `ruvector-wasm`, `ruvector-delta-wasm`, `ruvector-attention-wasm`, subset modules | SIMD128 (wasm) | ~4 GB (browser limit) | IndexedDB, OPFS | fetch/WebSocket | No filesystem, no threads without SharedArrayBuffer, cold start |
| **WASM Edge** (Cloudflare Workers, Fastly Compute) | Distributed low-latency analysis, API gateway | `ruvector-wasm` (slim), `ruvector-router-wasm` | SIMD128 | 128 MB (worker limit) | None (stateless) / KV store | HTTP | Cold start (~50ms), no long-running processes, 30s CPU time |
| **WASM WASI** (Wasmtime, WasmEdge) | Sandboxed server-side, plugin host | Full workspace via WASI Preview 2 | SIMD128 + host SIMD via imports | Configurable | WASI filesystem | WASI sockets | Host-dependent capabilities |
| **FPGA** (Xilinx Alveo, Intel Agilex) | Real-time basecalling, kmer hashing | `ruvector-fpga-transformer`, custom bitstream | Custom datapath | HBM (4-32 GB) | Host DMA | PCIe Gen4/5 | Fixed function per bitstream, long synthesis |
| **GPU** (CUDA, ROCm, Metal) | Neural network inference, batch variant calling | `ruvector-attention` + GPU backend | Tensor cores, matrix cores | 8-80 GB VRAM | Host-mapped | PCIe / NVLink | Requires driver stack, batch-oriented |

### 2.1 Platform Tier Definitions

**Tier 1 (Primary)**: Native x86_64, Native ARM64, WASM Browser. Full CI, release binaries, performance regression testing on every merge.

**Tier 2 (Supported)**: WASM Edge, WASM WASI, GPU (CUDA), Confidential Computing (SEV-SNP / TDX / Nitro). CI on nightly builds, release artifacts published but with "beta" designation.

**Tier 3 (Experimental)**: Native RISC-V, FPGA, GPU (ROCm, Metal), Edge TPU/NPU, Unikernel. Community-maintained, CI on demand.

## 3. WASM DNA Analyzer: Browser-Based Variant Calling

### 3.1 Architecture

The browser deployment compiles the analysis pipeline into a layered WASM module set, loaded progressively to minimize time-to-first-interaction.

```
Browser Tab
+-------------------------------------------------------------------+
|  Main Thread (UI)                                                  |
|  +-------------------------------------------------------------+  |
|  | React/Svelte App                                             |  |
|  | - File picker (FASTQ/BAM/VCF)                               |  |
|  | - Genome browser visualization                               |  |
|  | - Variant table with filtering                               |  |
|  +-------------------------------------------------------------+  |
|       |  postMessage            |  postMessage                     |
|       v                        v                                   |
|  +------------------+   +------------------+   +-----------------+ |
|  | Worker 0         |   | Worker 1         |   | Worker N        | |
|  | (Coordinator)    |   | (Analysis)       |   | (Analysis)      | |
|  |                  |   |                  |   |                 | |
|  | ruvector-wasm    |   | ruvector-wasm    |   | ruvector-wasm   | |
|  | (core: 2 MB)     |   | (core: 2 MB)     |   | (core: 2 MB)    | |
|  | + router-wasm    |   | + attention-wasm |   | + gnn-wasm      | |
|  |                  |   |                  |   |                 | |
|  +-------+----------+   +-------+----------+   +--------+--------+ |
|          |                       |                       |          |
|          +-----------+-----------+-----------+------------+         |
|                      |                                              |
|                      v                                              |
|  +-------------------------------------------------------------+  |
|  | SharedArrayBuffer: Reference Genome (hg38, ~750 MB quant.)  |  |
|  +-------------------------------------------------------------+  |
|                      |                                              |
|                      v                                              |
|  +-------------------------------------------------------------+  |
|  | IndexedDB / OPFS                                             |  |
|  | - Cached reference genome chunks                             |  |
|  | - User genome data                                           |  |
|  | - Analysis results (VCF)                                     |  |
|  | - Delta checkpoints (ruvector-delta-wasm)                    |  |
|  +-------------------------------------------------------------+  |
+-------------------------------------------------------------------+
```

### 3.2 Progressive Module Loading

Rather than loading the entire analysis suite upfront, modules are fetched on demand based on the user's workflow.

| Load Phase | Module | Compressed Size | Trigger |
|------------|--------|-----------------|---------|
| **Phase 0** (immediate) | `ruvector-wasm` core (VectorDB + HNSW) | ~2 MB | Page load |
| **Phase 1** (on file open) | `ruvector-router-wasm` (variant routing) | ~800 KB | User opens FASTQ/BAM |
| **Phase 2** (on analysis start) | `ruvector-attention-wasm` (neural caller) | ~3 MB | "Run Analysis" button |
| **Phase 3** (on demand) | `ruvector-gnn-wasm` (graph neural net) | ~2.5 MB | Structural variant mode |
| **Phase 4** (on demand) | `ruvector-delta-wasm` (sync engine) | ~600 KB | "Share Results" action |

Total worst-case download: approximately 9 MB compressed. Each module is independently cacheable via Service Worker with content-hash URLs.

### 3.3 Privacy-First Computation Model

All genomic computation occurs within the browser sandbox. The architecture enforces this structurally:

1. **No server-side data path**: The WASM modules operate on `Float32Array` and `Uint8Array` buffers allocated within the WASM linear memory or JavaScript heap. No API endpoint receives raw genomic data.
2. **SharedArrayBuffer for reference genome**: The reference genome (hg38) is downloaded once, stored in IndexedDB, and mapped into a `SharedArrayBuffer` accessible by all Web Workers. This avoids per-worker copies and stays within the ~4 GB browser memory budget.
3. **IndexedDB persistence**: Analysis state, intermediate results, and delta checkpoints persist locally across sessions. The `ruvector-delta-wasm` crate's `DeltaStream` and `JsDeltaWindow` types (shown in the existing codebase at `/home/user/ruvector/crates/ruvector-delta-wasm/src/lib.rs`) provide event-sourced state management with compaction.
4. **Optional encrypted export**: Results can be exported as encrypted VCF files using SubtleCrypto, shared via peer-to-peer WebRTC, or uploaded to a user-chosen endpoint. The system never mandates server contact.

### 3.4 Web Worker Parallelism

Worker count is determined at runtime via `navigator.hardwareConcurrency`. The coordinator (Worker 0) partitions the genome into regions and distributes work:

```
Regions = chromosome_boundaries(reference)
Workers = navigator.hardwareConcurrency - 1  // reserve 1 for UI
Chunks  = distribute(Regions, Workers, strategy=balanced_by_complexity)

for each Worker w:
    w.postMessage({ type: "analyze", regions: Chunks[w], config })

// Results stream back via postMessage as each region completes
// Coordinator merges and deduplicates calls at region boundaries
```

The `ruvector-wasm` crate already supports `Arc<Mutex<CoreVectorDB>>` internally (see `/home/user/ruvector/crates/ruvector-wasm/src/lib.rs`, line 191), which is safe within a single WASM instance. Cross-worker coordination uses `postMessage` with `Transferable` objects for zero-copy buffer passing.

### 3.5 Quantized Reference Genome

A full hg38 reference is approximately 3.1 GB uncompressed. For browser deployment, the reference is quantized and chunked:

- **2-bit encoding**: Each nucleotide (A, C, G, T) is stored in 2 bits, reducing hg38 to ~775 MB.
- **Block compression**: 64 KB blocks with LZ4 decompression in WASM, yielding ~350 MB on-disk in IndexedDB.
- **On-demand decompression**: Only active regions are decompressed into the SharedArrayBuffer working set.
- **Content-addressed chunks**: Each 1 MB chunk is addressed by SHA-256 hash, enabling incremental download and validation.

### 3.6 WebGPU for Browser-Based Neural Inference

WebGPU replaces the legacy WebGL compute path with proper GPU compute shaders accessible directly from the browser. For genomic workloads that are inherently parallel -- k-mer embedding, batch variant scoring, attention-based basecalling -- WebGPU provides 10-100x speedup over WASM SIMD128 by dispatching work to the device GPU.

#### 3.6.1 Architecture

```
Browser Tab (WebGPU-accelerated)
+----------------------------------------------------------------------+
|  Main Thread                                                          |
|  +---------------------------------------------------------------+   |
|  | navigator.gpu.requestAdapter()                                 |   |
|  | -> GPUDevice                                                   |   |
|  +-------+-------------------------------------------------------+   |
|          |                                                            |
|          v                                                            |
|  +-------+-------------------------------------------------------+   |
|  | WebGpuBackend (ruvector-attention-wasm)                        |   |
|  |                                                                |   |
|  |  +------------------+  +------------------+  +--------------+  |   |
|  |  | Compute Pipeline |  | Compute Pipeline |  | Compute      |  |   |
|  |  | kmer_embed.wgsl  |  | flash_attn.wgsl  |  | Pipeline     |  |   |
|  |  | (k-mer embed)    |  | (Flash Attention)|  | score.wgsl   |  |   |
|  |  +--------+---------+  +--------+---------+  | (variant     |  |   |
|  |           |                     |             |  scoring)    |  |   |
|  |           v                     v             +------+-------+  |   |
|  |  +------------------------------------------------------+     |   |
|  |  | GPU Storage Buffers                                    |     |   |
|  |  | - Reference genome tiles (up to 2-4 GB on discrete)    |     |   |
|  |  | - Quantized model weights (INT8/FP16)                  |     |   |
|  |  | - K-mer embedding table                                |     |   |
|  |  | - Variant score accumulator                            |     |   |
|  |  +------------------------------------------------------+     |   |
|  +---------------------------------------------------------------+   |
|          |                                                            |
|          | SharedArrayBuffer interop                                  |
|          v                                                            |
|  +-------+-------------------------------------------------------+   |
|  | WASM Workers (existing pipeline)                               |   |
|  | - Pre/post-processing in WASM SIMD                             |   |
|  | - Coordinate GPU dispatch from worker threads                  |   |
|  | - Fallback to WASM SIMD when WebGPU unavailable               |   |
|  +---------------------------------------------------------------+   |
+----------------------------------------------------------------------+
```

#### 3.6.2 WGSL Compute Shaders for Genomics

Custom kernels are authored in WGSL (WebGPU Shading Language) and compiled at runtime by the browser's GPU driver:

| Kernel | Workgroup Size | Dispatch | Operation |
|--------|---------------|----------|-----------|
| `kmer_embed.wgsl` | (256, 1, 1) | ceil(sequence_len / 256) | Hash k-mers into embedding space, 256 k-mers per workgroup |
| `flash_attn.wgsl` | (128, 1, 1) | (num_heads, ceil(seq_len / 128), 1) | Tiled Flash Attention with shared memory for Q*K^T, online softmax |
| `variant_score.wgsl` | (64, 1, 1) | ceil(num_variants / 64) | Batch variant quality scoring against reference embeddings |
| `distance_matrix.wgsl` | (16, 16, 1) | (ceil(N/16), ceil(M/16), 1) | Pairwise cosine distance for HNSW neighbor search on GPU |

#### 3.6.3 GPU Memory Management

Storage buffers allow loading substantial data into GPU-accessible memory:

| Data | Size (hg38) | GPU Memory Strategy |
|------|-------------|---------------------|
| Reference genome (2-bit encoded) | ~775 MB | Tile into 64 MB chunks, stream on demand |
| Model weights (INT8 quantized) | ~50 MB | Persistent storage buffer, loaded once |
| K-mer embedding table (k=21) | ~128 MB | Persistent storage buffer |
| Working set (per-region) | ~16 MB | Double-buffered for overlap of compute and transfer |

Modern discrete GPUs (NVIDIA RTX 4090: 24 GB, Apple M3 Max: 48 GB unified) can hold the entire quantized reference genome in GPU memory. Integrated GPUs share system RAM, so the tiling strategy provides graceful degradation.

#### 3.6.4 Interop: WASM Workers and WebGPU Compute

The `SharedArrayBuffer` bridge enables zero-copy data flow between WASM worker threads and the WebGPU compute pipeline:

1. WASM worker writes input data to a `SharedArrayBuffer` region.
2. Main thread creates a `GPUBuffer` mapped to the same memory (via `mappedAtCreation` or `mapAsync`).
3. GPU compute shader processes the buffer.
4. Results are read back into a `SharedArrayBuffer` region accessible by WASM workers.

This avoids the double-copy penalty (WASM -> JS -> GPU -> JS -> WASM) that plagues naive WebGL approaches.

#### 3.6.5 Browser Compatibility

| Browser | WebGPU Status | Minimum Version | Notes |
|---------|---------------|-----------------|-------|
| Chrome | Stable | 113+ (May 2023) | Full compute shader support |
| Firefox | Stable | 121+ (Dec 2023) | Nightly had it earlier; stable since 121 |
| Safari | Stable | 18+ (Sep 2024) | Metal backend; some WGSL limitations |
| Edge | Stable | 113+ (same as Chrome) | Chromium-based, identical support |

#### 3.6.6 Performance Projections

| Metric | WASM SIMD128 | WebGPU (Integrated) | WebGPU (Discrete) |
|--------|-------------|---------------------|-------------------|
| Variant scoring throughput | 50 variants/sec | 500 variants/sec | 5,000 variants/sec |
| K-mer embedding (1M k-mers) | 200 ms | 20 ms | 2 ms |
| Flash Attention (seq=1024, heads=8) | 150 ms | 15 ms | 1.5 ms |
| HNSW search (100K vectors, top-10) | 80 ms | 12 ms | 3 ms |

#### 3.6.7 Implementation

The `WebGpuBackend` is implemented in `ruvector-attention-wasm` as a feature-gated module:

```
ruvector-attention-wasm/
  src/
    lib.rs                  # Existing WASM SIMD backend
    webgpu/
      mod.rs                # WebGpuBackend: GPUDevice init, pipeline cache
      kernels/
        kmer_embed.wgsl     # K-mer embedding compute shader
        flash_attn.wgsl     # Flash Attention with tiled shared memory
        variant_score.wgsl  # Batch variant quality scoring
        distance.wgsl       # HNSW distance computation on GPU
      buffer_pool.rs        # GPU buffer pool with double-buffering
      interop.rs            # SharedArrayBuffer <-> GPUBuffer bridge
```

Feature gate: `--features webgpu`. Falls back to WASM SIMD when `navigator.gpu` is unavailable.

### 3.7 WebNN for Hardware-Accelerated Neural Networks in Browser

The Web Neural Network (WebNN) API provides direct access to dedicated machine learning accelerators -- NPUs, GPUs, and optimized CPU paths -- from within the browser, enabling power-efficient inference for genomic neural networks.

#### 3.7.1 Architecture

```
Browser Tab (WebNN-accelerated)
+----------------------------------------------------------------------+
|  ruvector-wasm: WebNnInference                                        |
|  +----------------------------------------------------------------+  |
|  | navigator.ml.createContext(devicePreference)                    |  |
|  |     |                                                          |  |
|  |     +---> MLContext (NPU preferred)                            |  |
|  |     |        |                                                 |  |
|  |     |        v                                                 |  |
|  |     |    MLGraphBuilder                                        |  |
|  |     |    +-- buildGraph(onnx_model)                            |  |
|  |     |    +-- MLGraph (compiled, hardware-specific)             |  |
|  |     |                                                          |  |
|  |     +--- Fallback chain:                                       |  |
|  |          1. NPU (Apple Neural Engine / Qualcomm Hexagon /      |  |
|  |             Intel NPU)  -> 10x lower power than GPU            |  |
|  |          2. GPU (WebGPU backend) -> high throughput             |  |
|  |          3. CPU (WASM SIMD) -> universal fallback              |  |
|  +----------------------------------------------------------------+  |
+----------------------------------------------------------------------+
```

#### 3.7.2 NPU Hardware Landscape

| NPU | TOPS (INT8) | Power | Platform | Availability |
|-----|-------------|-------|----------|-------------|
| Apple Neural Engine (M1-M4) | 15.8-38 | ~5 W | macOS Safari, iOS | Available now |
| Qualcomm Hexagon DSP (8cx Gen 3) | 26+ | ~3 W | Windows on ARM | Snapdragon X Elite laptops |
| Intel NPU (Meteor Lake+) | 10-11 | ~3 W | Windows/Linux | 14th gen Core+ |
| AMD XDNA (Ryzen AI) | 10-16 | ~4 W | Windows/Linux | Ryzen 7040+ series |

#### 3.7.3 Model Format Pipeline

```
Training (PyTorch)
      |
      v
  ONNX export (opset 18+)
      |
      v
  ONNX Runtime quantization (INT8/INT4)
      |
      v
  WebNN graph compilation (browser-side)
      |
      v
  Hardware-specific execution plan
  (NPU firmware / GPU microcode / CPU vectorized)
```

#### 3.7.4 Genomic Use Cases

| Model | Parameters | INT8 Size | NPU Inference | GPU Inference | WASM Fallback |
|-------|-----------|-----------|---------------|---------------|---------------|
| Basecalling (transformer) | 5M | 5 MB | 0.8 ms/read | 1.2 ms/read | 15 ms/read |
| Variant quality (CNN) | 2M | 2 MB | 0.3 ms/variant | 0.5 ms/variant | 5 ms/variant |
| Methylation (attention) | 8M | 8 MB | 1.5 ms/window | 2.0 ms/window | 25 ms/window |

NPU inference runs at approximately 10x lower power consumption than GPU, making it ideal for battery-powered field devices running browser-based analysis.

#### 3.7.5 Implementation

The `WebNnInference` module in `ruvector-wasm` provides automatic backend selection:

```
ruvector-wasm/
  src/
    lib.rs                  # Existing core
    webnn/
      mod.rs                # WebNnInference: MLContext creation, fallback chain
      graph_builder.rs      # ONNX -> WebNN graph compilation
      backend_select.rs     # Automatic NPU -> GPU -> CPU fallback
      quantize.rs           # Runtime INT8/INT4 quantization for WebNN
```

Feature gate: `--features webnn`. Browser support: Chrome 122+ (with flag), full stable support expected 2025-2026. The fallback chain ensures functionality on all browsers regardless of WebNN availability.

## 4. Edge Computing for Field Genomics

### 4.1 Deployment Topology: Nanopore + Laptop

```
+--------------------+        USB3        +---------------------------+
| Oxford Nanopore    | ----- FAST5/POD5 ---> | Laptop (ARM64/x86_64) |
| MinION / Flongle   |                    |                           |
+--------------------+                    |  ruvector-cli             |
                                          |  +-- basecaller (FPGA    |
                                          |  |   or CPU fallback)    |
                                          |  +-- variant caller      |
                                          |  +-- annotation engine   |
                                          |  +-- local web UI        |
                                          |      (WASM analyzer)     |
                                          |                           |
                                          |  ruvector-delta-core      |
                                          |  +-- offline journal     |
                                          |  +-- sync queue          |
                                          +----------+----------------+
                                                     |
                                            (when connected)
                                                     |
                                                     v
                                          +---------------------------+
                                          |  Central Lab Server       |
                                          |  ruvector-server (REST)   |
                                          |  ruvector-cluster         |
                                          |  ruvector-replication     |
                                          +---------------------------+
```

### 4.2 Offline-First Architecture

The field deployment operates under the assumption that network connectivity is intermittent or absent. The architecture enforces this through three mechanisms:

**Local-complete pipeline**: The `ruvector-cli` binary includes the full analysis pipeline. No network call is required to progress from raw signal to annotated VCF. The CLI binary is statically linked and self-contained (~50 MB with all features, ~15 MB stripped for ARM64).

**Delta-based synchronization**: When connectivity returns, the `ruvector-delta-core` crate synchronizes results incrementally. Rather than transferring complete VCF files, only `VectorDelta` objects are transmitted. The existing `HybridEncoding` in `ruvector-delta-wasm` (line 202-206) provides efficient serialization. For a typical variant calling session producing 4-5 million variants, delta sync reduces transfer from ~500 MB to ~12 MB by transmitting only changed positions.

**Conflict resolution**: The `ruvector-replication` crate provides vector clocks (`VectorClock`) and last-write-wins (`LastWriteWins`) conflict resolution strategies (see `/home/user/ruvector/crates/ruvector-replication/src/lib.rs`, line 38). When multiple field laptops analyze overlapping regions, the central server reconciles using configurable merge strategies.

### 4.3 Compressed Reference Genomes

For field deployment where storage is constrained:

| Genome | Uncompressed | Quantized (2-bit + index) | With annotations |
|--------|-------------|---------------------------|------------------|
| Human (hg38) | 3.1 GB | 775 MB | 950 MB |
| Malaria (Pf3D7) | 23 MB | 6 MB | 12 MB |
| SARS-CoV-2 | 30 KB | 8 KB | 45 KB |
| Custom panel (targeted) | Variable | Variable | < 100 MB typical |

The quantized human reference at 950 MB with annotations fits comfortably on any modern laptop, eliminating the need for network access to reference data.

### 4.4 Edge TPU / NPU Deployment for Point-of-Care Genomics

Dedicated neural processing hardware enables complete basecalling and variant calling pipelines to run within strict power budgets on mobile and embedded devices, enabling true point-of-care genomic analysis.

#### 4.4.1 Hardware Targets

```
Point-of-Care Device
+----------------------------------------------------------------------+
|                                                                      |
|  +-------------------+   USB/PCIe   +-----------------------------+  |
|  | Nanopore MinION   | -----------> | Embedded Host               |  |
|  | (raw signal)      |              |                             |  |
|  +-------------------+              |  ruvector-cli               |  |
|                                     |  +-- NpuBackend             |  |
|                                     |      |                      |  |
|                   +-----------------+------+------------------+   |  |
|                   |                 |                          |   |  |
|            +------v------+  +------v--------+  +------v------+|  |  |
|            | Edge TPU    |  | Mobile NPU    |  | CPU         ||  |  |
|            | (Coral USB) |  | (ANE/Hexagon) |  | (fallback)  ||  |  |
|            | 4 TOPS INT8 |  | 15-73 TOPS    |  | NEON SIMD   ||  |  |
|            | 2W power    |  | 3-5W power    |  | 5-15W       ||  |  |
|            +-------------+  +---------------+  +-------------+|  |  |
|                   +-----------------+--------------------------+  |  |
|                                     |                             |  |
|                                     v                             |  |
|                              Annotated VCF                        |  |
|                              (battery-powered, <5W total)         |  |
+----------------------------------------------------------------------+
```

#### 4.4.2 NPU Hardware Specifications

| Hardware | TOPS (INT8) | Power | Form Factor | Interface | Best For |
|----------|-------------|-------|-------------|-----------|----------|
| Google Coral Edge TPU | 4 | 2 W | USB dongle / M.2 / PCIe | ONNX via TFLite delegate | Embedded Linux devices |
| Apple Neural Engine (M1+) | 15.8 | ~5 W | Integrated SoC | Core ML / ONNX Runtime | macOS/iOS tablets in clinic |
| Qualcomm Hexagon DSP/HVX (8 Gen 3) | 73+ | ~5 W | Integrated SoC | QNN / ONNX Runtime | Android tablets, mobile |
| Intel Movidius (Myriad X) | 1 | 1.5 W | USB stick | OpenVINO | Ultra-low-power embedded |
| MediaTek APU (Dimensity 9300) | 46 | ~4 W | Integrated SoC | NeuroPilot / ONNX Runtime | Android devices |

#### 4.4.3 Model Quantization Strategy

Different NPUs require different quantization levels for optimal performance:

| NPU Target | Quantization | Weight Size (basecaller) | Accuracy Loss | Throughput |
|------------|-------------|-------------------------|---------------|-----------|
| Edge TPU | INT8 (full) | 5 MB | <0.5% | 800 reads/sec |
| Apple ANE | INT8 (mixed FP16 attention) | 7 MB | <0.2% | 2,000 reads/sec |
| Hexagon HVX | INT4 weights / INT8 activations | 3 MB | <1.0% | 3,000 reads/sec |
| CPU fallback | FP32 | 20 MB | 0% (baseline) | 50 reads/sec |

#### 4.4.4 ONNX Runtime Execution Provider Abstraction

The `NpuBackend` in `ruvector-fpga-transformer` unifies all NPU targets through ONNX Runtime's execution provider (EP) interface:

```
ruvector-fpga-transformer/
  src/
    backend/
      mod.rs              # TransformerBackend trait (existing)
      npu/
        mod.rs            # NpuBackend: ONNX EP abstraction layer
        coral.rs          # Google Coral Edge TPU EP
        coreml.rs         # Apple Core ML EP (ANE)
        qnn.rs            # Qualcomm QNN EP (Hexagon)
        openvino.rs       # Intel OpenVINO EP (Movidius/NPU)
        auto_select.rs    # Runtime hardware detection + EP selection
```

EP selection at runtime: `NpuBackend::auto_detect()` probes available hardware and selects the highest-throughput EP. The same ONNX model runs on all backends; only the EP changes.

#### 4.4.5 Power Budget Analysis

Complete basecalling pipeline power consumption for battery-powered field deployment:

| Component | Power Draw | Duration (per WGS) | Energy |
|-----------|-----------|-------------------|--------|
| Nanopore MinION | 2.5 W | 48 hours | 120 Wh |
| NPU inference (basecalling) | 3 W | 6 hours | 18 Wh |
| CPU (alignment + calling) | 8 W | 4 hours | 32 Wh |
| Display + I/O | 3 W | 10 hours | 30 Wh |
| **Total** | **<5 W average** | **~48 hours** | **~200 Wh** |

A 200 Wh battery (typical laptop) can power a complete whole-genome sequencing and analysis session in the field without mains power.

## 5. FPGA Pipeline for Basecalling

### 5.1 Architecture

The `ruvector-fpga-transformer` crate (at `/home/user/ruvector/crates/ruvector-fpga-transformer/src/lib.rs`) provides the software interface to FPGA-accelerated inference. The architecture separates the concern of model execution from hardware specifics through the `TransformerBackend` trait:

```
                         +-----------------------------+
                         |  ruvector-fpga-transformer   |
                         |  Engine                      |
                         |  +-- load_artifact()         |
                         |  +-- infer()                 |
                         |  +-- CoherenceGate            |
                         +------+-------+---------+-----+
                                |       |         |
                   +------------+   +---+---+  +--+----------+
                   |                |       |  |              |
            +------v------+ +------v--+ +--v--v------+ +-----v-------+
            | FpgaPcie    | | FpgaDaemon| | NativeSim | | WasmSim     |
            | (pcie feat) | | (daemon)  | | (native)  | | (wasm feat) |
            |             | |           | |           | |             |
            | DMA ring    | | Unix sock | | Pure Rust | | wasm_bindgen|
            | BAR0/BAR1   | | /gRPC     | | simulator | | browser sim |
            +-------------+ +-----------+ +-----------+ +-------------+
                   |              |
                   v              v
            +----------------------------+
            | FPGA Hardware              |
            | +-- Convolution engine     |
            | +-- Multi-head attention   |
            | +-- CTC decoder           |
            | +-- Quantized matmul      |
            | +-- LUT-based softmax     |
            +----------------------------+
```

### 5.2 Basecalling Pipeline Stages

The FPGA implements a fixed-function pipeline for nanopore basecalling:

| Stage | Operation | FPGA Implementation | Throughput Target |
|-------|-----------|--------------------|--------------------|
| 1 | Signal normalization | Streaming mean/variance, INT16 | Line rate |
| 2 | Convolution layers | Systolic array, INT8 weights | 10 TOPS |
| 3 | Multi-head attention | Custom attention kernel with early exit | 5 TOPS |
| 4 | CTC decode | Beam search with hardware prefix tree | 100 Mbases/s |
| 5 | Quality scoring | LUT-based Phred computation | Line rate |

The existing `FixedShape` type (at `/home/user/ruvector/crates/ruvector-fpga-transformer/src/types.rs`, line 58) constrains all dimensions at model-load time, enabling the FPGA synthesis tool to generate optimized datapaths. The `QuantSpec` type carries INT4/INT8 quantization metadata that maps directly to FPGA arithmetic units.

### 5.3 Performance Targets

| Metric | GPU Baseline (A100) | FPGA Target (Alveo U250) | Speedup |
|--------|--------------------|-----------------------------|---------|
| Basecalling throughput | 1 Gbases/s | 10 Gbases/s | 10x |
| Latency per read (1000 bp) | 2 ms | 0.2 ms | 10x |
| Power consumption | 300 W | 75 W | 4x better |
| Batch requirement | 32+ reads | 1 read (streaming) | Real-time capable |

### 5.4 Programmable Pipeline: Model Updates Without Hardware Changes

The `ModelArtifact` system (defined in `/home/user/ruvector/crates/ruvector-fpga-transformer/src/artifact/`) enables model updates without FPGA re-synthesis:

1. **Artifact format**: Signed bundles containing quantized weights, shape metadata, and optional FPGA bitstream.
2. **Weight-only update**: When the model architecture is unchanged, only new `QuantizedTensor` weights are loaded via DMA. The FPGA datapath is reused. Latency: ~200 ms.
3. **Bitstream update**: When architectural changes are needed (new layer types, different attention mechanism), a new bitstream is loaded via partial reconfiguration. Latency: ~2 seconds.
4. **Ed25519 signature verification**: Every artifact is cryptographically signed. The `verify` module in the artifact subsystem validates signatures before any weights reach the FPGA.

### 5.5 Wire Protocol

Communication between host and FPGA uses the binary protocol defined in `/home/user/ruvector/crates/ruvector-fpga-transformer/src/backend/mod.rs`:

- Magic: `0x5256_5846` ("RVXF")
- 24-byte request header (`RequestFrame`) with sequence length, model dimension, vocabulary size, model ID, and flags
- 14-byte response header (`ResponseFrame`) with status, latency, cycles, and gate decision
- CRC32 integrity checking on all frames
- DMA ring buffer with 16 slots of 64 KB each for the PCIe backend (`PcieConfig` defaults)

## 6. Distributed Analysis via ruvector-cluster

### 6.1 Cluster Topology

```
                        +-------------------+
                        |  Load Balancer    |
                        |  (L7 / gRPC-LB)  |
                        +--------+----------+
                                 |
                 +---------------+---------------+
                 |               |               |
          +------v------+ +-----v-------+ +-----v-------+
          | Node 1      | | Node 2      | | Node 3      |
          | (Leader)    | | (Follower)  | | (Follower)  |
          |             | |             | |             |
          | ruvector-   | | ruvector-   | | ruvector-   |
          |  server     | |  server     | |  server     |
          | ruvector-   | | ruvector-   | | ruvector-   |
          |  cluster    | |  cluster    | |  cluster    |
          | ruvector-   | | ruvector-   | | ruvector-   |
          |  raft       | |  raft       | |  raft       |
          |             | |             | |             |
          | Shards:     | | Shards:     | | Shards:     |
          | [0,5,10,..] | | [1,6,11,..] | | [2,7,12,..] |
          +------+------+ +------+------+ +------+------+
                 |               |               |
                 +-------+-------+-------+-------+
                         |               |
                  +------v------+ +------v------+
                  | Replica A   | | Replica B   |
                  | (async)     | | (async)     |
                  +-------------+ +-------------+
```

### 6.2 Genome Sharding Strategy

The `ConsistentHashRing` in `ruvector-cluster` (at `/home/user/ruvector/crates/ruvector-cluster/src/shard.rs`, line 16) uses 150 virtual nodes per physical node for balanced distribution. For genome analysis, sharding follows a domain-aware strategy:

| Sharding Dimension | Strategy | Rationale |
|--------------------|----------|-----------|
| By chromosome | Range-based (chr1-22, X, Y, MT = 25 shards) | Locality for structural variant calling |
| By sample | Hash-based (jump consistent hash) | Even distribution across nodes |
| By analysis stage | Pipeline-based (align -> call -> annotate) | Stage-specific resource allocation |

For a whole-genome cohort study of 1000 samples across a 5-node cluster:

- 64 shards (default `ClusterConfig.shard_count`), replication factor 3
- Each node holds ~38 shard replicas (64 * 3 / 5)
- Chromosome-aware routing ensures all data for a given chromosome on a given sample co-locates

### 6.3 Consensus and State Management

The `ruvector-raft` crate (at `/home/user/ruvector/crates/ruvector-raft/src/lib.rs`) provides Raft consensus for distributed metadata:

- **What Raft manages**: Cluster membership, shard assignments, analysis pipeline state, schema metadata. NOT the genomic data itself.
- **What the data plane manages**: Genomic vectors flow through the `ShardRouter` directly, bypassing consensus for read operations. Writes go through the leader for ordering.
- **Failover**: The `ruvector-replication` crate's `FailoverManager` with `FailoverPolicy` handles primary promotion. Split-brain prevention uses the Raft quorum: a partition with fewer than `min_quorum_size` nodes (default 2) becomes read-only.

### 6.4 REST/gRPC API

The `ruvector-server` crate (at `/home/user/ruvector/crates/ruvector-server/src/lib.rs`) exposes an axum-based REST API on port 6333 (default). For DNA analysis, the API surface extends to:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health`, `/ready` | GET | Liveness and readiness probes |
| `/collections` | CRUD | Manage genome collections (samples, panels) |
| `/collections/{name}/points` | POST | Insert variant vectors |
| `/collections/{name}/search` | POST | Similarity search (find related variants) |
| `/analysis/submit` | POST | Submit analysis job (FASTQ -> VCF pipeline) |
| `/analysis/{id}/status` | GET | Job progress and status |
| `/analysis/{id}/results` | GET | Stream results as they complete |

Compression (`CompressionLayer`) and CORS (`CorsLayer`) are enabled by default.

## 7. MCP Integration: AI-Powered Genomic Interpretation

### 7.1 Architecture

The `mcp-gate` crate (at `/home/user/ruvector/crates/mcp-gate/src/lib.rs`) provides an MCP server that currently exposes coherence gate tools. For the DNA Analyzer, this extends to expose genomic analysis as MCP tools callable by AI assistants:

```
+------------------+       JSON-RPC/stdio       +-------------------+
| AI Assistant     | <========================> | mcp-gate          |
| (Claude, etc.)   |                            |                   |
|                  |    tools/call:              | Tools:            |
| "What variants  |    - permit_action          | - permit_action   |
|  in BRCA1 are   |    - get_receipt            | - get_receipt     |
|  pathogenic?"   |    - replay_decision        | - replay_decision |
|                  |    - search_variants  [NEW] | - search_variants |
|                  |    - annotate_variant [NEW] | - annotate_variant|
|                  |    - run_pipeline    [NEW]  | - run_pipeline    |
+------------------+                            +--------+----------+
                                                         |
                                                         v
                                               +---------+----------+
                                               | ruvector-core      |
                                               | ruvector-server    |
                                               | Analysis Pipeline  |
                                               +--------------------+
```

### 7.2 Genomic MCP Tools

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `search_variants` | Gene name, region, filters | Matching variant vectors with clinical annotations | "Find all ClinVar pathogenic variants in TP53" |
| `annotate_variant` | Chromosome, position, ref, alt | Functional impact, population frequency, clinical significance | "What is the impact of chr17:7674220 G>A?" |
| `run_pipeline` | FASTQ/BAM reference, analysis parameters | Job ID, streaming status | "Analyze this patient's exome against hg38" |
| `compare_samples` | Two sample IDs, region | Delta vectors showing differences | "How do these tumor/normal samples differ in chr9?" |

### 7.3 Coherence Gate for Genomic Decisions

The existing `TileZero` coherence gate (re-exported by `mcp-gate`) provides a safety layer for AI-driven genomic interpretation:

- **permit_action** with `action_type: "clinical_interpretation"` requires higher coherence thresholds than exploratory queries.
- **Witness receipts** create an auditable trail of every AI-assisted interpretation, critical for clinical compliance.
- **Replay capability** allows regulatory review of any AI-generated interpretation by deterministically replaying the decision with its original context.

## 8. Platform-Specific Optimization Strategies

### 8.1 x86_64 Optimizations

- **AVX-512 distance calculations**: The `simsimd` dependency (workspace `Cargo.toml`, line 96) auto-detects and uses the widest SIMD available. For 384-dimensional variant embeddings, AVX-512 processes 16 floats per cycle.
- **Memory-mapped storage**: `memmap2` (line 94) provides zero-copy access to genome indices. For a 64-shard cluster node holding 200 GB of variant data, mmap avoids loading the entire dataset into RAM.
- **Release profile**: `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `panic = "abort"` (workspace `Cargo.toml`, lines 151-156) produce maximally optimized binaries.

### 8.2 ARM64 Optimizations

- **NEON SIMD**: Automatic fallback from AVX-512 to NEON via `simsimd` runtime detection. NEON processes 4 floats per cycle (128-bit registers).
- **SVE/SVE2** (Graviton3+, Apple M4+): Scalable vector extension with variable-width registers (128-2048 bit). The `ruvector-core` distance functions are written to auto-vectorize under SVE.
- **Static linking**: ARM64 field deployments use `RUSTFLAGS="-C target-feature=+neon"` and static musl linking for a single self-contained binary.

### 8.3 WASM Optimizations

- **SIMD128**: The `detect_simd()` function in `ruvector-wasm` (line 426) detects WASM SIMD support. When available, distance calculations use `v128` operations providing 4x throughput over scalar.
- **Streaming compilation**: WASM modules use `WebAssembly.compileStreaming()` for parallel download and compilation.
- **Memory management**: The `MAX_VECTOR_DIMENSIONS` constant (line 98, set to 65536) prevents allocation bombs. Vector dimensions are validated before any WASM memory allocation.
- **wasm-opt**: All WASM modules pass through Binaryen's `wasm-opt -O3 --enable-simd` in the release build pipeline.

### 8.4 FPGA Optimizations

- **Zero-allocation hot path**: The `Engine::infer()` method (line 170) performs no heap allocations during inference. All buffers are pre-allocated at `load_artifact()` time.
- **INT4/INT8 quantization**: The `QuantSpec` type carries explicit quantization metadata. INT4 weights halve memory bandwidth requirements on the FPGA datapath.
- **LUT-based softmax**: The `LUT_SOFTMAX` flag (protocol flags, line 90) triggers hardware lookup-table softmax, avoiding expensive exponential computation.
- **Early exit**: The `EARLY_EXIT` flag (line 92) enables the coherence gate to terminate inference early when confidence exceeds a threshold, saving cycles.

### 8.5 RISC-V Vector Extensions (RVV 1.0)

RISC-V Vector Extensions version 1.0 (ratified 2023) bring variable-length vector processing to the RISC-V ISA, providing a future-proof SIMD target for RuVector's compute-intensive kernels.

#### 8.5.1 Vector-Length-Agnostic (VLA) Programming Model

Unlike fixed-width SIMD (AVX-512 = 512 bits, NEON = 128 bits), RVV 1.0 uses a vector-length-agnostic (VLA) model where the same binary runs optimally on any hardware vector width:

```
// Pseudocode: VLA distance computation
// Same binary runs on VLEN=128 (minimal) through VLEN=1024 (high-end)

fn cosine_distance_rvv(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    let mut i = 0;
    while i < a.len() {
        // vsetvl: hardware determines how many elements to process
        // based on VLEN and remaining count
        let vl = vsetvl(a.len() - i, SEW::F32);
        let va = vle32(&a[i..], vl);
        let vb = vle32(&b[i..], vl);
        dot   = vfredsum(vfmul(va, vb, vl), dot, vl);
        norm_a = vfredsum(vfmul(va, va, vl), norm_a, vl);
        norm_b = vfredsum(vfmul(vb, vb, vl), norm_b, vl);
        i += vl;
    }
    1.0 - dot / (norm_a.sqrt() * norm_b.sqrt())
}
```

#### 8.5.2 RISC-V Hardware Landscape

| Processor | VLEN | Cores | Clock | Throughput (est.) | Status |
|-----------|------|-------|-------|-------------------|--------|
| SiFive X280 | 512-bit | 4 | 2.0 GHz | ~2x ARM NEON equiv. | Available (dev boards) |
| SiFive P870 | 256-bit | 8 | 2.5 GHz | ~1.5x ARM NEON equiv. | Sampling 2025 |
| Tenstorrent Ascalon | 512-bit | 128 | 3.0 GHz | ~4x Graviton3 equiv. | Expected 2025-2026 |
| Alibaba Xuantie C920 | 256-bit | 16 | 2.5 GHz | ~1x Graviton2 equiv. | Available (China market) |
| Ventana Veyron V2 | 256-bit | 192 | 3.6 GHz | Server-class | Expected 2025-2026 |

#### 8.5.3 Rust Compilation Strategy

RuVector's SIMD kernels compile to RVV through two paths:

1. **Portable SIMD (`core::simd`)**: Rust's portable SIMD API auto-targets RVV when compiling with `target_feature = "v"`. The same `ruvector-core` distance functions that produce AVX-512 on x86 and NEON on ARM emit RVV instructions on RISC-V.

2. **Explicit RVV intrinsics**: For kernels requiring RVV-specific features (e.g., strided loads for k-mer processing, segment loads for interleaved genomic data), the `core::arch::riscv64` intrinsics provide direct access.

```
# Build for RISC-V with vector extensions
RUSTFLAGS="-C target-feature=+v" cargo build --release --target riscv64gc-unknown-linux-gnu
```

Feature gate in `ruvector-core`:
```rust
#[cfg(target_feature = "v")]
mod rvv_kernels;
```

#### 8.5.4 Performance Projection: Genomic Workloads on RVV

| Kernel | ARM NEON (128-bit) | RVV VLEN=256 | RVV VLEN=512 | RVV VLEN=1024 |
|--------|-------------------|-------------|-------------|---------------|
| Cosine distance (384-dim) | 96 cycles | 48 cycles | 24 cycles | 12 cycles |
| K-mer hashing (batch 1024) | 4,096 cycles | 2,048 cycles | 1,024 cycles | 512 cycles |
| HNSW neighbor scan | 1.2 ms/query | 0.6 ms/query | 0.3 ms/query | 0.15 ms/query |

The VLA model means RuVector binaries compiled today will automatically exploit wider vector widths in future RISC-V processors without recompilation.

## 9. Confidential Computing Deployment

Confidential computing enables processing of sensitive patient genomic data in untrusted environments (public cloud, multi-tenant infrastructure) by encrypting data in use -- not just at rest and in transit.

### 9.1 Architecture

```
+----------------------------------------------------------------------+
| Public Cloud Infrastructure (untrusted)                              |
|                                                                      |
|  +----------------------------------------------------------------+  |
|  | Confidential VM (encrypted memory, attestable)                  |  |
|  |                                                                |  |
|  |  +----------------------------------------------------------+  |  |
|  |  | ruvector-server + ruvector-cluster                        |  |  |
|  |  | (unmodified Rust binaries — no code changes required)     |  |  |
|  |  |                                                          |  |  |
|  |  | Patient FASTQ --> basecall --> align --> call --> VCF     |  |  |
|  |  |                                                          |  |  |
|  |  | Only VCF summary + aggregate stats leave the enclave     |  |  |
|  |  +----------------------------------------------------------+  |  |
|  |                                                                |  |
|  |  Attestation Report                                            |  |
|  |  +-- Hardware root of trust (TPM / PSP / SGX)                 |  |
|  |  +-- Measurement of VM image (hash of boot chain)             |  |
|  |  +-- Runtime measurement (loaded binary hashes)               |  |
|  |  +-- Signed by hardware vendor (AMD / Intel / AWS)            |  |
|  +----------------------------------------------------------------+  |
|          |                                                            |
|          | Attestation verification                                   |
|          v                                                            |
|  +----------------------------------------------------------------+  |
|  | Attestation Verifier (customer-controlled)                      |  |
|  | +-- Verify hardware attestation signature                      |  |
|  | +-- Check binary measurement against known-good hash           |  |
|  | +-- Authorize data release to enclave                          |  |
|  | +-- Audit log of all attestation events                        |  |
|  +----------------------------------------------------------------+  |
+----------------------------------------------------------------------+
```

### 9.2 Technology Matrix

| Technology | Vendor | Isolation Level | Code Changes | Performance Overhead | Cloud Availability |
|------------|--------|----------------|-------------|---------------------|-------------------|
| AMD SEV-SNP | AMD | Full VM encryption + attestation | None (unmodified binaries) | ~5% (memory encryption) | AWS, Azure, GCP |
| Intel TDX | Intel | Trust Domain (VM-level) | None (unmodified binaries) | ~5-8% (memory encryption) | Azure, GCP |
| AWS Nitro Enclaves | AWS | Isolated VM partition | Minimal (vsock communication) | ~3% (vsock overhead) | AWS only |
| Intel SGX | Intel | Process-level enclave | Significant (SDK required) | ~10-20% (page-level encryption) | Azure, on-prem |
| ARM CCA | ARM | Realm (VM-level) | None (unmodified binaries) | ~5% (estimated) | Expected 2025-2026 |

### 9.3 Attestation Flow for Genomic Data Processing

```
Attestation Timeline:

  Hospital             Attestation            Confidential VM
  Data Owner           Verifier              (Cloud)
     |                    |                       |
     |                    |    1. Boot + measure   |
     |                    |    <----attest_report--|
     |                    |                       |
     |                    | 2. Verify HW sig      |
     |                    | 3. Check binary hash   |
     |                    | 4. Issue auth token    |
     |                    |----auth_token-------->|
     |                    |                       |
     | 5. Encrypted genome upload                 |
     |---encrypted_fastq--+---------------------->|
     |                    |                       |
     |                    |  6. Decrypt + analyze  |
     |                    |  (in encrypted memory) |
     |                    |                       |
     |  7. Receive results only (VCF summary)     |
     |<---vcf_summary----+-----------------------|
     |                    |                       |
     | 8. Raw genome never leaves enclave         |
     |                    |                       |
```

### 9.4 Deployment Configuration

The `ConfidentialRuntime` deployment target requires no changes to existing RuVector binaries. Deployment is handled through infrastructure configuration:

| Parameter | AMD SEV-SNP | Intel TDX | AWS Nitro Enclaves |
|-----------|------------|-----------|-------------------|
| VM image | Standard `ruvector-server` Docker image | Standard Docker image | Enclave image (EIF) built from Docker |
| Memory encryption | Automatic (hardware) | Automatic (hardware) | Automatic (Nitro hypervisor) |
| Attestation | `/dev/sev-guest` | `/dev/tdx-guest` | `nitro-cli describe-enclaves` |
| Max memory | Instance limit | Instance limit | Parent instance - 512 MB |
| Network | Standard (encrypted at memory level) | Standard | vsock only (no external network) |

### 9.5 Cost Analysis

| Configuration | Instance Type (AWS) | Cost/hr | Overhead vs Non-Confidential | WGS/day |
|---------------|--------------------|---------|-----------------------------|---------|
| Standard | m6i.4xlarge | $0.768 | baseline | ~30 |
| SEV-SNP | m6a.4xlarge (SEV enabled) | $0.691 | ~5% perf, -10% cost (AMD) | ~28 |
| Nitro Enclave | m5.4xlarge + enclave | $0.768 | ~3% perf | ~29 |
| TDX | m7i.4xlarge (TDX enabled) | $0.806 | ~8% perf | ~27 |

The cost overhead for confidential computing is marginal (3-8% performance) while providing hardware-attested guarantees that raw genomic data is never exposed to the cloud operator.

## 10. Cross-Platform Data Flow

### 10.1 End-to-End Flow: Field to Cloud

```
  Field Laptop          |  Transit          |  Central Lab
  (ARM64, offline)      |  (delta sync)     |  (x86_64 cluster)
                        |                   |
  Nanopore -> basecall  |                   |
  -> align -> call      |                   |
  -> annotate           |                   |
  -> local VCF + deltas |                   |
       |                |                   |
       | (connectivity) |                   |
       +--------------->| ruvector-delta    |
                        | (compressed VectorDelta) |
                        +------------------>| merge via
                        |                   | ruvector-replication
                        |                   | -> cluster-wide
                        |                   |    variant database
                        |                   |
                        |                   | AI assistant queries
                        |                   | via mcp-gate
                        |                   | -> clinical report
```

### 10.2 Data Format Compatibility

All platforms produce and consume the same serialized formats:

| Data Type | Format | Crate | Cross-Platform |
|-----------|--------|-------|---------------|
| Vector embeddings | `Float32Array` / `Vec<f32>` | `ruvector-core` | All platforms |
| Delta updates | `HybridEncoding` (sparse + dense) | `ruvector-delta-core` / `ruvector-delta-wasm` | All platforms |
| Model artifacts | Signed bundle (manifest + weights + bitstream) | `ruvector-fpga-transformer::artifact` | All platforms (FPGA bitstream optional) |
| API payloads | JSON (REST) / Protobuf (gRPC) | `ruvector-server` | All platforms with network |
| MCP messages | JSON-RPC 2.0 over stdio | `mcp-gate` | All platforms with stdio |

## 11. Deployment Topologies

### 11.1 Single Laptop (Field Genomics)

```
Components: ruvector-cli (static binary)
Storage: Local filesystem (mmap)
Network: None required
Capacity: ~10 whole genomes / day (ARM64), ~30 / day (x86_64)
```

### 11.2 Clinical Lab (3-5 Nodes)

```
Components: ruvector-server + ruvector-cluster + ruvector-raft
Storage: NVMe SSDs, mmap
Network: 10 GbE / 25 GbE
Capacity: ~500 whole genomes / day
Fault tolerance: 1 node failure (replication factor 3)
```

### 11.3 Research HPC (50+ Nodes)

```
Components: ruvector-cluster (64+ shards) + FPGA accelerators + GPU nodes
Storage: Parallel filesystem (Lustre/GPFS) + local NVMe
Network: InfiniBand / 100 GbE
Capacity: ~10,000 whole genomes / day
Specialization: FPGA nodes for basecalling, GPU nodes for deep learning, CPU nodes for alignment
```

### 11.4 Global Edge Network

```
Components: WASM Edge workers (Cloudflare/Fastly) + central API
Storage: Edge KV for cached references, central cluster for results
Network: CDN-accelerated
Use case: Low-latency variant lookup API, privacy-preserving query routing
Capacity: 100,000+ queries/second globally
```

### 11.5 Browser-Only (Personal Genomics)

```
Components: ruvector-wasm + ruvector-delta-wasm + ruvector-attention-wasm
Storage: IndexedDB / OPFS
Network: Optional (reference genome download, result sharing)
Capacity: 1 exome / session, panel analysis in < 30 seconds
Privacy: All computation local, no data leaves the browser
```

### 11.6 Unikernel Deployment for Minimal Attack Surface

A unikernel compiles the entire RuVector analysis pipeline into a single-address-space OS image that boots directly on a hypervisor or bare metal. There is no shell, no SSH daemon, no package manager, no unnecessary kernel subsystems -- only the genomic analysis binary and the minimal kernel runtime it requires.

#### 11.6.1 Architecture

```
Traditional VM                          Unikernel
+---------------------------+           +---------------------------+
| Application               |           | ruvector-server           |
| (ruvector-server)         |           | (linked with unikernel    |
+---------------------------+           |  runtime library)         |
| Libraries (glibc, etc.)  |           |                           |
+---------------------------+           | Minimal network stack     |
| Linux Kernel              |           | Minimal memory allocator  |
| (~20M lines, 100+ syscalls|           | Minimal scheduler         |
|  used by application)     |           | (~10K lines TCB)          |
+---------------------------+           +---------------------------+
| Hypervisor (KVM/Xen)     |           | Hypervisor (KVM/Xen)     |
+---------------------------+           +---------------------------+

TCB: ~20,000,000 lines                 TCB: ~10,000 lines
Boot time: 30+ seconds                  Boot time: <100 ms
Attack surface: large                   Attack surface: minimal
```

#### 11.6.2 Unikernel Framework Options

| Framework | Language | Rust Support | Network Stack | Boot Time | Maturity |
|-----------|---------|-------------|--------------|-----------|----------|
| Unikraft | C/Rust | Native (Kraft build) | lwIP / custom | ~10 ms | Production (Linux Foundation) |
| NanoVMs (Ops/Nanos) | C | Binary-compatible (no recompile) | Full POSIX network | ~50 ms | Production |
| Hermit | Rust | Native | smoltcp | ~20 ms | Research/experimental |
| RustyHermit | Rust | Native | smoltcp | ~20 ms | Research/experimental |

#### 11.6.3 Deployment Configuration

```
# Build unikernel image with Unikraft
kraft build --target ruvector-server --plat qemu --arch x86_64

# Or with NanoVMs (no recompile, uses existing Linux binary)
ops build /usr/local/bin/ruvector-server -c config.json

# Result: single bootable image (~60 MB)
# Contains: ruvector-server + minimal kernel + network stack + TLS
```

| Parameter | Traditional VM | Unikernel |
|-----------|---------------|-----------|
| Image size | ~500 MB (distroless) to ~2 GB (Ubuntu) | ~60 MB |
| Boot time | 30-120 seconds | 10-100 ms |
| Memory footprint | ~200 MB baseline | ~30 MB baseline |
| Open ports | 22 (SSH), 6333 (API), potentially others | 6333 (API) only |
| Kernel CVEs applicable | All Linux kernel CVEs | ~0 (custom minimal kernel) |
| Trusted computing base | ~20M lines (kernel) + ~5M lines (userspace) | ~10K lines |
| Scaling (cold start) | 30+ seconds | <100 ms (instant autoscale) |

#### 11.6.4 Genomic Use Cases

- **Serverless variant calling**: Boot a unikernel per analysis job, process, emit results, terminate. No persistent state on the server. Cold start <100 ms means autoscaling is nearly instant.
- **Air-gapped clinical deployment**: Ship a single bootable USB image containing the entire variant calling pipeline. No OS to patch, no shell for attackers to exploit, no lateral movement possible.
- **Compliance-friendly**: The minimal TCB (~10K lines) is auditable by a single security team. Regulatory bodies (FDA, CE-IVD) can review the entire trusted base.

## 12. Multi-Region Federated Deployment

For genomic analysis at national or international scale, data sovereignty regulations (GDPR Article 44, HIPAA, China PIPL, Australia My Health Records Act) require that patient genome data remain within the jurisdiction where it was collected. The federated deployment architecture enables cross-region collaboration on genomic insights without moving raw data across borders.

### 12.1 Architecture

```
+----------------------------------------------------------------------+
|                        Global Federated Mesh                          |
|                                                                      |
|  Region A (EU - Frankfurt)          Region B (US - Virginia)         |
|  +----------------------------+     +----------------------------+   |
|  | ruvector-cluster (3 nodes) |     | ruvector-cluster (5 nodes) |   |
|  | ruvector-raft (intra)      |     | ruvector-raft (intra)      |   |
|  |                            |     |                            |   |
|  | Patient genomes: EU only   |     | Patient genomes: US only   |   |
|  | GDPR Article 44 compliant  |     | HIPAA compliant            |   |
|  |                            |     |                            |   |
|  | FederatedMesh coordinator  |     | FederatedMesh coordinator  |   |
|  +------------+---------------+     +---------------+------------+   |
|               |                                     |                |
|               | Secure aggregation only              |                |
|               | (model gradients, summary stats)     |                |
|               |                                     |                |
|               +----------------+--------------------+                |
|                                |                                     |
|                                v                                     |
|               +--------------------------------+                     |
|               | Cross-Region Aggregator        |                     |
|               | (no raw genome data)           |                     |
|               |                                |                     |
|               | - Federated model updates       |                     |
|               | - Aggregate allele frequencies  |                     |
|               | - Cross-cohort summary stats    |                     |
|               +--------------------------------+                     |
|                                |                                     |
|               +----------------+--------------------+                |
|               |                                     |                |
|  +------------+---------------+     +---------------+------------+   |
|  | Region C (APAC - Tokyo)    |     | Region D (AUS - Sydney)    |   |
|  | ruvector-cluster (3 nodes) |     | ruvector-cluster (2 nodes) |   |
|  | ruvector-raft (intra)      |     | ruvector-raft (intra)      |   |
|  |                            |     |                            |   |
|  | Patient genomes: JP only   |     | Patient genomes: AU only   |   |
|  | APPI compliant             |     | My Health Records Act      |   |
|  +----------------------------+     +----------------------------+   |
+----------------------------------------------------------------------+
```

### 12.2 Data Sovereignty Enforcement

The `FederatedMesh` coordinator enforces data residency at the routing layer:

| Data Type | Cross-Region Transfer | Enforcement Mechanism |
|-----------|----------------------|----------------------|
| Raw FASTQ/BAM | NEVER | `ShardRouter` tags all genomic data with origin region; cross-region routes are blocked at the network layer |
| Individual VCF | NEVER | VCF files are flagged with jurisdiction metadata; export requires attestation of destination compliance |
| Aggregate allele frequencies | ALLOWED | Differential privacy (epsilon=1.0) applied before cross-region transfer |
| Model weight updates | ALLOWED | Secure aggregation protocol: each region contributes encrypted gradient; aggregator decrypts only the sum |
| Reference genome data | ALLOWED | Public data (hg38, ClinVar) is replicated freely across all regions |
| Analysis pipeline code | ALLOWED | Binary artifacts are signed and distributed globally |

### 12.3 Consistency Model

| Data Class | Intra-Region Consistency | Cross-Region Consistency |
|------------|-------------------------|--------------------------|
| Patient genomic data | Strong (Raft consensus) | N/A (data never leaves region) |
| Analysis results (VCF) | Strong (Raft consensus) | N/A (results stay with data) |
| Reference genome data | Strong (Raft consensus) | Eventual (async replication, typically <1 hour) |
| Aggregate statistics | Strong within region | Eventual (federated aggregation on schedule, typically daily) |
| Model weights | Strong within region | Eventual (federated learning rounds, typically weekly) |
| Cluster metadata | Strong (Raft consensus) | Eventual (gossip protocol for topology discovery) |

### 12.4 Deployment Topology Details

**Intra-region**: Hub-and-spoke topology. The Raft leader serves as the hub; followers replicate from the leader. This provides strong consistency for all genomic operations within a jurisdiction.

**Cross-region**: Peer-to-peer mesh topology. Each region's `FederatedMesh` coordinator communicates directly with peers in other regions. No single point of failure for cross-region aggregation. If one region goes offline, others continue independently.

### 12.5 Implementation

```
ruvector-cluster/
  src/
    federation/
      mod.rs              # FederatedMesh: cross-region coordinator
      sovereignty.rs       # Data residency enforcement + tagging
      aggregation.rs       # Secure aggregation for model updates
      privacy.rs           # Differential privacy for aggregate stats
      topology.rs          # Cross-region peer discovery (gossip)

ruvector-raft/
  src/
    lib.rs                # Existing Raft (used for intra-region consensus)

ruvector-replication/
  src/
    federation.rs         # Cross-region eventual consistency
    conflict.rs           # Existing conflict resolution (extended for federation)
```

### 12.6 Cross-Region Performance

| Operation | Intra-Region Latency | Cross-Region Latency | Notes |
|-----------|---------------------|---------------------|-------|
| Variant search | <10 ms | N/A (local only) | All queries against local data |
| Analysis job submission | <50 ms | N/A (local only) | Jobs run on local cluster |
| Aggregate frequency query | <10 ms (cached) | 100-500 ms (live) | Cached aggregates updated daily |
| Federated model update | N/A | 5-30 minutes per round | Depends on model size and region count |
| Reference genome sync | N/A | <1 hour for full sync | Content-addressed chunks, incremental |

## 13. Build and Release Strategy

### 13.1 Artifact Matrix

| Target | Build Command | Output | Size |
|--------|--------------|--------|------|
| Linux x86_64 | `cargo build --release` | `ruvector-cli` | ~25 MB |
| Linux ARM64 | `cross build --release --target aarch64-unknown-linux-musl` | `ruvector-cli` | ~20 MB |
| macOS ARM64 | `cargo build --release --target aarch64-apple-darwin` | `ruvector-cli` | ~22 MB |
| Linux RISC-V | `cross build --release --target riscv64gc-unknown-linux-gnu` | `ruvector-cli` | ~28 MB |
| WASM (browser) | `wasm-pack build --release --target web` | `*.wasm` + JS glue | ~2 MB (core) |
| WASM (browser + WebGPU) | `wasm-pack build --release --target web --features webgpu` | `*.wasm` + WGSL shaders | ~2.5 MB (core + shaders) |
| WASM (Node) | `wasm-pack build --release --target nodejs` | `*.wasm` + JS glue | ~2.5 MB (core) |
| WASM (edge) | `wasm-pack build --release --target web --features slim` | `*.wasm` | ~800 KB |
| npm (Node.js bindings) | `napi build --release` | `*.node` | ~15 MB |
| Docker | `docker build -t ruvector .` | Container image | ~50 MB (distroless) |
| Docker (confidential) | `docker build -t ruvector:confidential --target confidential .` | Container image (SEV/TDX ready) | ~55 MB |
| Unikernel | `ops build target/release/ruvector-server` | Bootable image | ~60 MB |
| FPGA bitstream | `make synthesis BOARD=alveo_u250` | `.xclbin` | ~30 MB |

### 13.2 CI Matrix

Every PR runs Tier 1 targets. Nightly builds include Tier 2 (including confidential computing attestation tests). Release builds include all tiers. RISC-V builds use QEMU emulation in CI until native RISC-V CI runners are available.

## 14. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Browser memory limit blocks whole-genome analysis | High | Medium | Streaming analysis with region windowing; progressive result construction |
| WASM SIMD not available in older browsers | Medium | Low | Scalar fallback path; feature detection via `detect_simd()` |
| FPGA synthesis time blocks rapid iteration | High | Medium | NativeSim and WasmSim backends for development; FPGA only for production deployment |
| Edge worker cold start exceeds latency SLA | Medium | Medium | Pre-warming via cron triggers; keep-alive requests; minimal WASM module size |
| Cross-platform delta incompatibility | Low | High | `HybridEncoding` is platform-independent; fuzz-tested across all targets |
| SharedArrayBuffer disabled by browser security policy | Medium | High | Fallback to per-worker copies with memory pressure monitoring; COOP/COEP headers documented |
| WebGPU unavailable on target browser | Medium | Low | Automatic fallback to WASM SIMD; WebGPU is enhancement, not requirement |
| WebNN API not yet stable across browsers | High | Low | Triple fallback chain (NPU -> GPU -> CPU); WebNN is optional acceleration |
| Confidential VM attestation failure | Low | High | Pre-deployment attestation dry-run; fallback to standard VM with audit log |
| RISC-V ecosystem immaturity | High | Low | RISC-V is Tier 3 (experimental); x86_64 and ARM64 remain primary targets |
| Unikernel networking limitations | Medium | Medium | NanoVMs provides full POSIX network stack; thorough integration testing |
| Federated aggregation introduces privacy risk | Low | Critical | Differential privacy with configurable epsilon; formal privacy analysis required before deployment |
| NPU model quantization degrades accuracy | Medium | Medium | Per-NPU calibration datasets; accuracy regression testing in CI for each quantization level |

## 15. Decision

We adopt the multi-platform deployment architecture described above, with the following key commitments:

1. **Single codebase, multiple targets**: The existing pattern of `*-wasm`, `*-node`, and native crates continues. No platform-specific forks.
2. **Progressive capability**: Each platform gets the maximum capability its constraints allow, degrading gracefully.
3. **Privacy by architecture**: Browser and edge deployments are structurally incapable of leaking genomic data to servers.
4. **FPGA as acceleration, not dependency**: The `TransformerBackend` trait ensures every pipeline runs on CPU; FPGA is a 10x acceleration option, not a requirement.
5. **Delta-first synchronization**: All cross-node and cross-platform data exchange uses `ruvector-delta-*` for bandwidth efficiency.
6. **MCP as the AI integration surface**: Genomic analysis is exposed as MCP tools, enabling AI assistants to interpret results within the coherence gate's safety framework.
7. **GPU compute in the browser**: WebGPU provides 100x acceleration for neural inference in browser deployments, with WASM SIMD as universal fallback.
8. **NPU-aware inference**: WebNN and native ONNX Runtime execution providers enable power-efficient inference on dedicated ML accelerators across all form factors.
9. **Confidential computing for sensitive data**: AMD SEV-SNP, Intel TDX, and AWS Nitro Enclaves provide hardware-attested isolation for processing patient genomes in public cloud, with <8% performance overhead.
10. **Data sovereignty by design**: The federated deployment architecture enforces jurisdictional data residency at the routing layer, enabling international collaboration without cross-border genome data transfer.
11. **Minimal attack surface options**: Unikernel deployment reduces the trusted computing base by 2000x compared to traditional Linux VMs, suitable for air-gapped clinical and regulatory environments.
12. **Future-proof ISA support**: RISC-V Vector Extensions (RVV 1.0) with vector-length-agnostic code ensures RuVector benefits automatically from wider vector hardware as it ships.

## 16. Consequences

**Positive**: The architecture enables RuVector DNA Analyzer to serve clinical labs, field researchers, personal genomics users, and AI-powered interpretation pipelines from a single Rust codebase. The progressive loading strategy keeps browser deployments fast. The FPGA pipeline provides a clear path to 10x throughput. The MCP integration positions the system for AI-native genomics workflows. WebGPU acceleration brings near-native performance to browser deployments. Confidential computing opens the public cloud market for sensitive genomic workloads without compromising patient privacy. Federated deployment enables international genomic collaboration within regulatory constraints. Edge NPU support enables battery-powered field analysis at <5W. Unikernel deployment provides a compliance-friendly minimal-surface option for regulated environments.

**Negative**: Maintaining 80+ crates across 9+ targets increases CI complexity and build times. FPGA synthesis remains a bottleneck for hardware iteration. Browser memory limits constrain whole-genome analysis to streaming approaches. The coherence gate adds latency to MCP-mediated interpretations. WebGPU shader development requires WGSL expertise. Confidential computing attestation adds operational complexity to deployment. Federated learning rounds introduce latency for cross-region model convergence. NPU model quantization requires per-hardware calibration and accuracy validation. Unikernel deployment limits debugging capabilities (no shell, no SSH). RISC-V remains pre-production for server workloads.

**Neutral**: The platform tier system (Tier 1/2/3) acknowledges that not all targets receive equal investment, aligning engineering effort with user demand. Confidential computing and federated deployment are Tier 2, reflecting growing but not yet universal demand. Edge NPU, unikernel, and RISC-V are Tier 3, positioning for future hardware trends without over-investing today.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2026-02-11 | Initial proposal: core platform matrix, WASM browser, edge, FPGA, cluster, MCP |
| 0.2 | 2026-02-11 | SOTA enhancements: WebGPU browser inference, WebNN NPU acceleration, confidential computing (SEV-SNP/TDX/Nitro), Edge TPU/NPU deployment, RISC-V Vector Extensions (RVV 1.0), unikernel deployment, multi-region federated architecture |
