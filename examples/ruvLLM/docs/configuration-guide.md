# Configuration Guide

Every key in `config/example.toml`, what it does, and the common tuning
patterns that come up in deployments.

The configuration file has eight sections; six are documented in detail
below. Each section corresponds to one of the modules described in
[System Architecture](system-architecture.md).

## File Layout

```toml
[system]    # process-level: device class, memory ceiling, concurrency
[embedding] # embedding.rs: dimension, tokenization, batching
[memory]    # memory.rs: HNSW index params, persistence, write-back
[router]    # router.rs: FastGRNN dimensions, sparsity, confidence
[inference] # inference.rs: model variants, quantization, KV cache
[learning]  # learning.rs + sona/: replay, EWC, training cadence
```

## `[system]`

Process-level settings. Set these first; many of the per-section caps
derive from them.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `device_class` | string | host-dependent | One of `edge`, `desktop`, `server`. Tunes which inference backends and quantization paths get exercised. |
| `max_memory_mb` | int | `8192` | Hard ceiling for the process. The HNSW store, embedding cache, and inference KV cache all fit under this. Set to about 80 percent of available RAM. |
| `max_concurrent_requests` | int | `10` | Maximum inflight `/query` calls. Bound chosen so the SIMD pool and Candle backend stay below saturation. |
| `data_dir` | path | `./data` | Where persistent state lives. Used as default parent for `[memory].db_path`. Must be writable by the service user. |

## `[embedding]`

Configures `embedding.rs` (LRU plus tokenizer).

| Key | Type | Default | Purpose |
|---|---|---|---|
| `dimension` | int | `768` | Embedding vector width. Must match `[router].input_dim`'s upstream projection and `[memory]` HNSW vector size. |
| `max_tokens` | int | `512` | Truncation limit on tokenization input. Anything past this is dropped before embedding. |
| `batch_size` | int | `8` | Number of tokenization requests batched into a single CPU pass when concurrent requests collide. |

## `[memory]`

Configures `memory.rs` (HNSW vector store from `ruvector-core`).

| Key | Type | Default | Purpose |
|---|---|---|---|
| `db_path` | path | under `data_dir` | On-disk location of the HNSW store. Survives restarts when the `storage` feature is on. |
| `hnsw_m` | int | `16` | Maximum graph connectivity per node. Higher means better recall, more memory, slower insert. |
| `hnsw_ef_construction` | int | `100` | Build-time search width. Higher means better graph, slower insert. Spent once. |
| `hnsw_ef_search` | int | `64` | Query-time search width. Higher means better recall, slower search. The most-tuned knob in production. |
| `max_nodes` | int | `1000000` | Hard cap on total stored vectors. Hitting this triggers eviction. |
| `writeback_batch_size` | int | `100` | How many inserts are coalesced before hitting disk. |
| `writeback_interval_ms` | int | `1000` | How often the write-back task flushes pending inserts. |

## `[router]`

Configures `router.rs` (FastGRNN gated routing).

| Key | Type | Default | Purpose |
|---|---|---|---|
| `input_dim` | int | `128` | Router input width. Embeddings (768-D) are projected down to this. |
| `hidden_dim` | int | `64` | FastGRNN hidden state width. Bigger means more expressive, slower forward. |
| `sparsity` | float | `0.9` | Fraction of weights pinned to zero on the hot path. Higher means faster forward, less capacity. |
| `rank` | int | `8` | Low-rank decomposition dimension for the recurrent weight matrix. |
| `confidence_threshold` | float | `0.7` | Below this, the orchestrator takes the extended-context fallback path (see [System Architecture](system-architecture.md)). |

## `[inference]`

Configures `inference.rs` and (under `real-inference`) `inference_real.rs`.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `models` | string array | tiny, small, medium, large | Available model variants. The router decides which to dispatch on per request. |
| `quantization` | string | `q4` | Weight quantization. One of `q8`, `q4`, `binary`, or `fp16`. Lower precision means less memory, possibly less accuracy. |
| `max_context` | int | `8192` | Maximum context length passed to the inference backend. |
| `max_loaded_models` | int | `2` | How many model variants live in memory at once. The rest are loaded on demand. |
| `kv_cache_size` | int | `1024` | Per-session KV cache slot count. Multiplies by `max_concurrent_requests` for total budget. |

## `[learning]`

Configures `learning.rs` and the SONA subsystem in `sona/`.

| Key | Type | Default | Purpose |
|---|---|---|---|
| `enabled` | bool | `true` | Master switch for all learning loops. When `false`, trajectories are dropped and no replay/EWC happens. |
| `quality_threshold` | float | `0.7` | Trajectories scoring below this are not replayed. Aligns with `[router].confidence_threshold` by default. |
| `replay_capacity` | int | `10000` | Replay buffer size. Beyond this, oldest trajectories are evicted. |
| `batch_size` | int | `32` | Mini-batch size for the EWC++ training pass. |
| `learning_rate` | float | `0.001` | Learning rate for LoRA adapter updates. |
| `ewc_lambda` | float | `0.4` | Strength of the EWC++ penalty term. Higher means stronger anchoring to prior knowledge (less plasticity). |
| `training_interval_ms` | int | `3600000` | How often the consolidation loop runs. Default is one hour. |
| `min_samples` | int | `100` | Minimum replay-buffer fill before consolidation runs. Prevents premature low-data updates. |

The detailed semantics of EWC++, MicroLoRA vs. BaseLoRA, and the
ReasoningBank are in [SONA Overview](SONA/00-OVERVIEW.md) and the
chapter sequence under `docs/SONA/`.

## Common Tuning Patterns

### HNSW: Recall vs. Speed

The `hnsw_ef_search` parameter dominates query-time recall and latency.

| Goal | Setting | Trade |
|---|---|---|
| Lowest latency | `ef_search = 32` | Recall drops; some near-neighbors missed. |
| Balanced (default) | `ef_search = 64` | Good recall at single-digit microsecond search. |
| High-recall offline | `ef_search = 128 to 256` | 2 to 4 times slower, recall approaches exact. |

`hnsw_m` and `hnsw_ef_construction` are build-time. Raise them when index
quality matters more than disk-write throughput; they are cheap to spend
once if your write rate is moderate. Pair `m=32, ef_construction=200` for
a high-quality index that costs more memory but searches as fast as the
default.

### EWC lambda: Stability vs. Plasticity

`ewc_lambda` sets the EWC++ penalty strength. The trade-off is between
remembering old skills (high lambda) and adapting to new ones (low lambda).

| Setting | Behavior |
|---|---|
| `ewc_lambda = 0.0` | Pure plasticity. Catastrophic forgetting is possible. |
| `ewc_lambda = 0.4` (default) | Balanced. Stable for general workloads. |
| `ewc_lambda = 1.0+` | Strong anchoring. The base barely shifts; new patterns mostly land in MicroLoRA only. |

If you see drift on a long-running deployment (responses on common
queries get worse over time), raise lambda. If new domains never seem to
"stick", lower it.

### Quantization: Memory vs. Accuracy

The `quantization` choice intersects with `[system].max_memory_mb` and the
deployment target.

| Choice | Memory factor | Accuracy | Where |
|---|---|---|---|
| `fp16` | 1.0 | best | Workstation with plenty of RAM |
| `q8` (INT8) | 0.5 | small loss | Server default, ESP32-S3 with PSRAM |
| `q4` | 0.25 | moderate loss | Default for tight server budgets, plain ESP32 |
| `binary` | 0.125 | substantial loss | ESP32 with very tight RAM, accuracy-tolerant tasks |

When in doubt, start at `q4` and step up to `q8` if accuracy benchmarks
regress. ESP32 always ends up at `q8`, `q4`, or `binary`; `fp16` does
not fit. See [Deployment Guide](deployment-guide.md) for the ESP32
build commands.

### Concurrency Sizing

`[system].max_concurrent_requests` and `[inference].max_loaded_models`
are tightly coupled.

- Rule of thumb: each loaded model variant uses `kv_cache_size` times
  context-token-bytes per inflight request. Multiply by
  `max_concurrent_requests` to get the total KV-cache footprint.
- Symptom, latency spikes under load: lower `max_concurrent_requests`
  before raising `max_loaded_models`.
- Symptom, low CPU/GPU utilization: raise `max_concurrent_requests`
  by 50 percent, watch the latency p95 in `/stats`. Stop when p95
  starts to drift.

### Replay Buffer Sizing

`[learning].replay_capacity` should be sized so that consolidation runs
on a representative window of recent traffic.

- Daily volume V (queries that pass `quality_threshold`).
- Consolidation cadence equals `training_interval_ms`.
- A useful default is `replay_capacity` approximately
  `2 * V * (training_interval_ms / 1day)` so each consolidation sees
  roughly two windows of traffic.

If `min_samples` is never reached and consolidation never fires, lower
`min_samples` or `quality_threshold`. If consolidation always fires on
the same data, raise `replay_capacity`.

### Edge Profile (`device_class = "edge"`)

Recommended overrides for an edge / ESP32-class deployment:

```toml
[system]
device_class = "edge"
max_memory_mb = 256
max_concurrent_requests = 1

[memory]
hnsw_m = 8
hnsw_ef_construction = 50
hnsw_ef_search = 32
max_nodes = 10000

[router]
input_dim = 64
hidden_dim = 32
sparsity = 0.95

[inference]
models = ["tiny"]
quantization = "q4"
max_context = 1024
max_loaded_models = 1
kv_cache_size = 64

[learning]
enabled = false
```

The actual ESP32 firmware uses a compiled-in equivalent rather than a
TOML file, but the same trade-offs apply. See
[Deployment Guide](deployment-guide.md) for `esp32-flash` build commands.

### Server Profile (`device_class = "server"`)

For a moderate production server:

```toml
[system]
device_class = "server"
max_memory_mb = 16384
max_concurrent_requests = 32

[memory]
hnsw_m = 32
hnsw_ef_construction = 200
hnsw_ef_search = 96
max_nodes = 5000000

[router]
input_dim = 128
hidden_dim = 64
sparsity = 0.9

[inference]
quantization = "q8"
max_context = 8192
max_loaded_models = 4
kv_cache_size = 2048

[learning]
enabled = true
quality_threshold = 0.75
replay_capacity = 100000
training_interval_ms = 1800000
```

Pair this profile with `cargo build --release --features
"server,real-inference,parallel,metrics,storage"` from
[Deployment Guide](deployment-guide.md).

## Reloading Configuration

The TOML is read once at process start. Changing a value requires a
restart. There is no SIGHUP reload — by design, since the HNSW index
parameters and the embedding dimension cannot change without rebuilding
the store.

## See also

- [System Architecture](system-architecture.md)
- [Deployment Guide](deployment-guide.md)
- [API Reference](api-reference.md)
- [SONA Overview](SONA/00-OVERVIEW.md)
