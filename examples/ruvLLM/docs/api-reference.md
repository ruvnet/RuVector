# API Reference

The HTTP surface exposed by `ruvllm-server` (under the `server` feature),
the public Rust library API, and a brief note on the Node.js bindings.

## HTTP API

`ruvllm-server` is an Axum application. All endpoints accept and return
JSON unless noted. There are five endpoints.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + readiness probe |
| POST | `/query` | Run a query through the orchestrator |
| GET | `/stats` | Runtime stats (HNSW size, replay buffer fill, etc.) |
| POST | `/feedback` | Record feedback against a prior response |
| POST | `/session` | Open or resume a session |

### `GET /health`

Liveness check. Returns 200 OK when the server is up and the orchestrator
has finished initializing (HNSW loaded, base model — if `real-inference` —
ready).

**Response (200):**

```json
{
  "status": "ok",
  "uptime_ms": 123456,
  "version": "x.y.z"
}
```

A non-200 (typically 503) means the server is up but not ready; load
balancers should treat that as out-of-rotation. Once initialization is
complete it transitions to 200 and stays there.

### `POST /query`

The main entry point. Submits a query through the full orchestrator
pipeline (embedding → memory → router → attention → inference → trajectory
emission).

**Request body:**

```json
{
  "text": "What is the orchestration latency budget?",
  "session_id": "optional-uuid",
  "context": ["optional", "prior", "snippets"],
  "max_tokens": 256
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | The user-facing prompt. |
| `session_id` | string | no | Reuse a session opened via `/session`; affects which trajectory + adapter context is used. |
| `context` | string[] | no | Caller-supplied additional context. Augments, does not replace, retrieved memory. |
| `max_tokens` | int | no | Caps generation length; defaults to `[inference].max_context`-derived value. |

**Response (200):**

```json
{
  "text": "Sub-millisecond. P50 ~0.06 ms, P95 ~0.08 ms.",
  "confidence": 0.91,
  "sources": [
    { "id": "node-12", "score": 0.87 },
    { "id": "node-44", "score": 0.81 }
  ],
  "latency_ms": 0.07,
  "session_id": "uuid-if-provided-or-anonymous"
}
```

`confidence` is the router's output. `sources` are the HNSW neighbors that
contributed to the attended representation. `latency_ms` is wall-clock for
the orchestration path, not including inference.

**Error responses.** Every error has the shape
`{ "error": "code", "message": "...", "request_id": "..." }`. Codes follow
the `Error` enum in `src/error.rs` (see [Code Standards](code-standards.md)).

### `GET /stats`

Snapshot of internal counters. Cheap to call; useful for dashboards in
addition to the Prometheus scrape (the `metrics` feature) which gives the
full time series.

**Response (200):**

```json
{
  "memory": {
    "hnsw_node_count": 12345,
    "hnsw_ef_search": 64,
    "writeback_pending": 0
  },
  "router": {
    "confidence_p50": 0.84,
    "confidence_p95": 0.97
  },
  "learning": {
    "replay_buffer_size": 7321,
    "last_consolidation_ms_ago": 1820000
  },
  "inflight_requests": 2
}
```

The exact set of fields evolves with new metrics. Only the top-level keys
(`memory`, `router`, `learning`, `inflight_requests`) are part of the
stable contract.

### `POST /feedback`

Records feedback against a prior response. Drives the `learning.rs` replay
buffer when the configured `quality_threshold` is met.

**Request body:**

```json
{
  "session_id": "uuid",
  "request_id": "from-prior-query",
  "score": 0.85,
  "label": "good",
  "comment": "optional free text"
}
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | string | yes | Must match the session the original `/query` used. |
| `request_id` | string | yes | Identifier returned with the original response (also surfaced in error envelopes). |
| `score` | float | no | 0.0–1.0; if absent, derived from `label`. |
| `label` | string | no | One of `good`, `bad`, `neutral`. |
| `comment` | string | no | Stored alongside the trajectory; not used for scoring. |

**Response (202):**

```json
{ "accepted": true }
```

Feedback is processed asynchronously by `learning.rs`. A 202 means it was
queued; whether it ends up in the replay buffer depends on the
`[learning].quality_threshold` configuration.

### `POST /session`

Opens a session, optionally resuming a prior one. Sessions are how the
server scopes per-user adapter context, trajectory state, and routing
history.

**Request body:**

```json
{
  "resume": "optional-prior-session-id",
  "metadata": { "user": "alice" }
}
```

**Response (200):**

```json
{
  "session_id": "uuid",
  "resumed": false,
  "expires_at": "2026-05-09T14:00:00Z"
}
```

`resumed: true` means the server found and reattached to the prior session
state. `resumed: false` means it created a fresh session (either because
no `resume` was provided, or because the prior id had expired).

## Library API (Rust)

The crate exposes a small public surface from `lib.rs`. The canonical
struct is `RuvLLM`.

### `struct RuvLLM`

A configured, running orchestrator. Holds the embedding cache, HNSW
memory, router, attention, inference dispatcher, and learning subsystem.

**Construction:**

```rust
use ruvllm::{RuvLLM, Config};

let cfg = Config::from_path("config/example.toml")?;
let llm = RuvLLM::new(cfg).await?;
```

**Key methods (representative — see rustdoc for full list):**

| Method | Purpose |
|---|---|
| `RuvLLM::new(config) -> Result<Self>` | Wire up subsystems and load the HNSW store. |
| `llm.query(req) -> Result<Response>` | The hot path. Mirrors `POST /query`. |
| `llm.feedback(req) -> Result<()>` | Mirrors `POST /feedback`. |
| `llm.stats() -> Stats` | Mirrors `GET /stats`. |
| `llm.session_open(meta) -> SessionId` | Mirrors `POST /session`. |
| `llm.shutdown() -> Result<()>` | Flush the HNSW writeback queue and stop background loops cleanly. |

Internally, the orchestrator chains the modules described in
[System Architecture](system-architecture.md). Public methods always return
typed errors via the `Error` enum (`thiserror`); see
[Code Standards](code-standards.md).

### Subsystem Types (re-exports)

For callers who want fine-grained access (e.g. embedding without running
the full pipeline):

- `Embedding` — from `embedding.rs`. `embed(text) -> Vec<f32>`.
- `Memory` — from `memory.rs`. `search(vec, k) -> Vec<Hit>`.
- `Router` — from `router.rs`. `route(features) -> Decision`.
- `Inference` — from `inference.rs`. `dispatch(prompt, context) -> Response`.

These are `pub` so you can build alternative pipelines, but the canonical
flow goes through `RuvLLM::query`.

### Configuration

`Config` mirrors the TOML structure documented in
[Configuration Guide](configuration-guide.md). It implements `serde::Deserialize`
so you can build it from any source (TOML, JSON, env).

```rust
use ruvllm::Config;

// From file
let cfg = Config::from_path("config.toml")?;

// From a string
let cfg: Config = toml::from_str(include_str!("config.toml"))?;
```

### Errors

Every fallible function returns `Result<T, ruvllm::Error>`. The enum is
defined in `src/error.rs` with `thiserror`. Variants cover I/O, config,
HNSW, inference, and learning failures. Wrap or downcast as needed; the
HTTP server already maps each variant onto an HTTP status.

## Node.js Bindings (`napi` feature)

When the `napi` feature is enabled, the crate compiles as a `cdylib` that
Node.js can load directly. The bindings live in `src/napi.rs` and expose
a thin async wrapper around `RuvLLM::query`. Detailed JS-side examples
are out of scope for this reference; consult `napi.rs` for the function
surface, and the `napi-rs` documentation for build mechanics.

Typical use:

```ts
import { RuvLLM } from "ruvllm";

const llm = await RuvLLM.fromConfig("./config.toml");
const res = await llm.query({ text: "hello" });
console.log(res.text, res.confidence);
```

## Versioning

- Crate version is in `Cargo.toml`.
- HTTP endpoints carry no version prefix today; breaking shape changes
  are introduced on major version bumps with a path prefix (`/v2/...`)
  added at that time.
- Library API follows SemVer.

## See also

- [System Architecture](system-architecture.md)
- [Configuration Guide](configuration-guide.md)
- [Deployment Guide](deployment-guide.md)
- [SONA API Reference](SONA/09-API-REFERENCE.md)
