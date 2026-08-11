# ruvector-tiered-quant: Access-Pattern-Driven Per-Vector Precision Tiering

A new RuVector crate that assigns each stored vector to one of three precision tiers
(f32 / u8 / 1-bit) based on runtime access-frequency counters. A periodic `compact()`
call promotes hot vectors to f32 and demotes cold vectors to binary.

## Why this matters

Agent memory workloads have a Zipfian access distribution. Recent memories are accessed
constantly; archived knowledge is rarely touched. Today's vector databases compress
everything uniformly. This crate adapts encoding precision per vector — no existing
production system (Qdrant, Milvus, FAISS, Pinecone, LanceDB) does this.

## Results (Linux x86_64, n=10k, dims=128, k=10)

**Clustered workload (agent memory scenario):**
- HotWarmCold: recall=0.961, 568 QPS, **3.3 MB** (vs FlatU8 recall=0.961, 470 QPS, 11.0 MB)
- Same recall as FlatU8, **3.3× less memory**, **21% faster**.

**Uniform random (worst case, 60% cold tier):**
- HotWarmCold: recall=0.411 — honest lower bound for 1-bit BQ at 128-dim.

## Key engineering: cross-tier distance normalization

Hamming distances ∈ [0,1]. Euclidean distances for 128-dim vectors ∈ [0,~11].
Without scaling, cold vectors always win (their distances are tiny). Fix:

```rust
let hamming_norm = bits as f32 / (packed.len() as f32 * 64.0);
hamming_norm * (dims as f32 * 4.0 / 3.0).sqrt() / 0.5
```

## Crate surface

```rust
let mut idx = HotWarmColdIndex::new(128, /*hot_threshold=*/20, /*warm_threshold=*/5);
idx.insert(id, vec);
idx.access(id); // called by your query path
idx.compact();  // run on a timer or memory-pressure signal
let hits = idx.query(&query, 10);
let stats = idx.stats(); // hot/warm/cold counts + compression_ratio
```

## Files

- `crates/ruvector-tiered-quant/src/lib.rs` — trait, FlatF32/FlatU8/HWC indexes
- `crates/ruvector-tiered-quant/src/tier.rs` — TierKind, VectorRecord, TierStats
- `crates/ruvector-tiered-quant/src/quantizer.rs` — ScalarQuantizer, BinaryQuantizer
- `crates/ruvector-tiered-quant/src/dataset.rs` — deterministic LCG dataset generation
- `crates/ruvector-tiered-quant/src/bin/benchmark.rs` — three-suite benchmark binary
- `docs/adr/ADR-273-tiered-vector-quantization.md` — full decision record

## Phase 2 (Production hardening)

- f32 write-once arena for lossless cold→warm promotion
- Hysteresis band to prevent tier thrashing
- Async `compact()` as a ruFlo timer node
- HNSW graph integration via `ruvector-hnsw-repair`
