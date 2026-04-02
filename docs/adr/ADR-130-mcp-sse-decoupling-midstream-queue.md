# ADR-130: MCP SSE Decoupling via Midstream Queue Architecture

## Status

**Deployed** (2026-04-02) — Phases 1-3 complete. SSE decoupled to `mcp.pi.ruv.io` (`ruvbrain-sse` Cloud Run service). Sparsifier skip for >5M edge graphs. Response queue drain implemented.

## Date

2026-03-29

## Context

pi.ruv.io has experienced three outages in two days, all traced to the same root cause: **MCP SSE transport sharing the Cloud Run concurrency pool with the REST API**. Each SSE connection holds an open HTTP stream on a Cloud Run request slot indefinitely. When MCP clients disconnect and reconnect in loops (IDE restarts, network blips, SSE polyfill reconnects), they create a reconnect storm that exhausts all concurrency slots, returning 429 to every request — including health checks, scheduler jobs, and the REST API.

### Failure Timeline

| Date | Incident | Root Cause |
|------|----------|------------|
| 2026-03-28 AM | 504 Gateway Timeout | Scheduler thundering herd saturated CPU |
| 2026-03-28 PM | 503 "Service is disabled" | GFE marked service unavailable after cascading timeouts |
| 2026-03-29 AM | OOM crash → partial data load | 5,898 memories + 12.5M edge graph exceeded 2GB |
| 2026-03-29 PM | 429 Rate Exceeded (all endpoints) | SSE reconnect storm consumed all concurrency slots |

### Current Architecture (broken)

```
MCP Clients ──SSE──┐
Health Checks ─────┤
Scheduler Jobs ────┤── Cloud Run (single service, shared concurrency)
REST API ──────────┤     ruvbrain: 2 CPU, 4GB, concurrency=250
Browser/UI ────────┘
```

All traffic types compete for the same concurrency slots. SSE connections are long-lived (minutes to hours). REST requests are short-lived (milliseconds). Mixing them on the same concurrency pool is fundamentally broken at scale.

## Decision

### Split into three decoupled services with a Rust midstream queue

```
                    ┌─────────────────────┐
MCP Clients ──SSE──▶│  ruvbrain-sse        │──push──▶┌──────────────┐
                    │  (Cloud Run)         │         │              │
                    │  Concurrency: 500    │◀─poll───│  Midstream    │
                    │  CPU: 1, Mem: 512MB  │         │  Queue        │
                    └─────────────────────┘         │  (in-process) │
                                                     │              │
Health/Scheduler ──▶┌─────────────────────┐         │  Ring buffer  │
REST API ──────────▶│  ruvbrain-api        │──push──▶│  per session  │
Browser/UI ────────▶│  (Cloud Run)         │         │              │
                    │  Concurrency: 80     │◀─poll───│              │
                    │  CPU: 2, Mem: 4GB    │         └──────────────┘
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
Scheduler Jobs ───▶│  ruvbrain-worker     │
                    │  (Cloud Run Jobs)    │
                    │  GPU: L4 (optional)  │
                    │  Timeout: 1hr        │
                    └─────────────────────┘
```

### Service Separation

| Service | Purpose | Concurrency | CPU | Memory | Cost/month |
|---------|---------|-------------|-----|--------|------------|
| `ruvbrain-api` | REST API, health, status | 80 | 2 | 4 GB | ~$30 |
| `ruvbrain-sse` | MCP SSE transport only | 500 | 1 | 512 MB | ~$10 |
| `ruvbrain-worker` | Scheduler jobs (train, drift, transfer) | 1 | 2 | 4 GB | ~$15 (job-based) |

### Midstream Queue (Rust, in-process)

The queue bridges SSE and API services. It runs inside `ruvbrain-api` as a Rust module — no external dependencies (no Pub/Sub, no Redis).

```rust
/// Midstream message queue for SSE decoupling.
/// Each MCP session gets a bounded ring buffer.
/// API writes responses into the buffer; SSE service polls via internal endpoint.
pub struct MidstreamQueue {
    /// Session ID → bounded ring buffer of JSON-RPC responses
    sessions: DashMap<String, SessionBuffer>,
    /// Maximum sessions before evicting oldest idle
    max_sessions: usize,
    /// Maximum messages per session buffer
    buffer_capacity: usize,
}

pub struct SessionBuffer {
    messages: VecDeque<String>,
    created_at: Instant,
    last_poll: Instant,
    capacity: usize,
}

impl MidstreamQueue {
    pub fn new(max_sessions: usize, buffer_capacity: usize) -> Self {
        Self {
            sessions: DashMap::new(),
            max_sessions,
            buffer_capacity,
        }
    }

    /// Called by API when processing a JSON-RPC request for a session.
    /// Pushes the response into the session's ring buffer.
    pub fn push(&self, session_id: &str, message: String) -> Result<(), QueueError> {
        let mut entry = self.sessions
            .entry(session_id.to_string())
            .or_insert_with(|| SessionBuffer::new(self.buffer_capacity));
        entry.push(message);
        Ok(())
    }

    /// Called by SSE service to drain pending messages for a session.
    /// Returns all buffered messages and clears the buffer.
    pub fn drain(&self, session_id: &str) -> Vec<String> {
        if let Some(mut entry) = self.sessions.get_mut(session_id) {
            entry.last_poll = Instant::now();
            entry.messages.drain(..).collect()
        } else {
            Vec::new()
        }
    }

    /// Evict sessions idle for > timeout (called periodically).
    pub fn evict_idle(&self, timeout: Duration) {
        let now = Instant::now();
        self.sessions.retain(|_, buf| now.duration_since(buf.last_poll) < timeout);
    }
}
```

### SSE Service Protocol

The `ruvbrain-sse` service is a thin proxy:

1. **Client connects** via `GET /sse` → SSE service creates session, sends `endpoint` event
2. **Client sends** JSON-RPC via `POST /messages?sessionId=X` → SSE service forwards to `ruvbrain-api` internal endpoint
3. **API processes** request, pushes response into midstream queue
4. **SSE service polls** `ruvbrain-api` at `/internal/queue/drain?sessionId=X` every 100ms (or uses Server-Sent Events from API → SSE via internal endpoint)
5. **SSE service streams** response to client

The SSE service has **no business logic** — it only manages WebSocket/SSE transport. All brain logic stays in `ruvbrain-api`.

### Worker Service (Scheduler Isolation)

Heavy scheduler jobs move to `ruvbrain-worker` as Cloud Run Jobs:

| Job | Current | New |
|-----|---------|-----|
| `brain-train` | POST to API (every 5m) | Cloud Run Job (every 5m) — reads Firestore directly |
| `brain-transfer` | POST to API (every 30m) | Cloud Run Job (every 30m) — reads Firestore directly |
| `brain-attractor` | POST to API (every 20m) | Cloud Run Job (every 20m) |
| `brain-graph` | POST to API (hourly) | Cloud Run Job (hourly) — rebuilds graph, writes back |
| `brain-drift` | POST to API (every 15m) | Stays on API (lightweight, read-only) |
| `brain-cleanup` | POST to API (daily) | Cloud Run Job (daily) |

Workers read from and write to Firestore directly. They share the same Rust crates (`mcp-brain-server::store`, `mcp-brain-server::graph`, `sona`) compiled into a separate binary.

### Internal API Endpoints (not exposed externally)

Added to `ruvbrain-api` for SSE service communication:

```
POST /internal/queue/push    — SSE service forwards JSON-RPC here
GET  /internal/queue/drain   — SSE service polls for responses
POST /internal/session/create — SSE service registers new session
DELETE /internal/session/:id  — SSE service cleans up on disconnect
```

These are authenticated via an internal service account token (Cloud Run service-to-service auth).

## Implementation Plan

### Phase 1: SSE Connection Limiting (immediate, no new services)

Add server-side SSE connection limits to the existing monolith to stop the bleeding:

```rust
/// Maximum concurrent SSE connections per instance
const MAX_SSE_CONNECTIONS: usize = 50;
/// SSE idle timeout — disconnect clients that haven't sent a message in 5 minutes
const SSE_IDLE_TIMEOUT: Duration = Duration::from_secs(300);
/// Backoff header sent on 429 to slow reconnect storms
const SSE_RETRY_AFTER: u32 = 10; // seconds
```

1. Track active SSE count with `AtomicUsize`
2. Reject new SSE connections with `429 + Retry-After: 10` when at capacity
3. Add idle timeout — disconnect SSE sessions with no activity for 5 minutes
4. Add exponential backoff hint in SSE retry field

### Phase 2: Midstream Queue Module (week 1)

Implement `MidstreamQueue` as a module in `crates/mcp-brain-server/src/midstream_queue.rs`:

1. Ring buffer per session (capacity: 64 messages)
2. Idle session eviction (5 min timeout)
3. Max 200 concurrent sessions
4. Internal drain endpoint for future SSE service

### Phase 3: Service Split — **DEPLOYED 2026-04-02**

1. ~~Create `ruvbrain-sse` Dockerfile~~ **Done** — `Dockerfile.sse`, thin proxy binary
2. ~~Create `ruvbrain-worker` binary~~ Deferred (scheduler jobs stay on API for now)
3. ~~Deploy both alongside existing `ruvbrain-api`~~ **Done** — `ruvbrain-sse` Cloud Run service
4. ~~Update Cloud Scheduler~~ Deferred
5. ~~Update DNS/domain mapping~~ **Done** — `mcp.pi.ruv.io` CNAME via Cloudflare → `ghs.googlehosted.com`

**Deployment details:**
- SSE proxy: `ruvbrain-sse` Cloud Run service, concurrency=200, timeout=3600s, 1 min-instance
- Domain: `mcp.pi.ruv.io` (Cloudflare CNAME, Google-managed SSL cert)
- Root `/` and `/sse` both serve SSE (root is canonical)
- Main server SSE limited to 5 connections via `MAX_SSE` env var
- Sparsifier skip for >5M edge graphs prevents CPU starvation on startup
- All code references updated from `pi.ruv.io/sse` → `mcp.pi.ruv.io`

### Phase 4: Worker Migration (week 3)

1. Convert `brain-train`, `brain-transfer`, `brain-attractor`, `brain-graph` to direct Firestore workers
2. Remove `/v1/pipeline/optimize` endpoint from API (workers don't need it)
3. Add write-back protocol: worker writes results to Firestore, API reads on next request

## SSE Connection Limiting (Phase 1 — ship immediately)

This is the minimum fix to stop outages while the full decoupling is built:

```rust
// In AppState:
pub sse_connections: Arc<AtomicUsize>,

// In sse_handler:
async fn sse_handler(State(state): State<AppState>) -> Result<Sse<...>, (StatusCode, String)> {
    let current = state.sse_connections.load(Ordering::Relaxed);
    if current >= MAX_SSE_CONNECTIONS {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            "SSE connection limit reached. Retry-After: 10".into(),
        ));
    }
    state.sse_connections.fetch_add(1, Ordering::Relaxed);

    // ... existing SSE logic ...

    // On stream close:
    state.sse_connections.fetch_sub(1, Ordering::Relaxed);
}
```

## Alternatives Considered

1. **Google Cloud Pub/Sub**: External message queue between SSE and API. Adds latency (~50ms), cost ($0.40/million messages), and operational complexity. The in-process ring buffer is simpler and faster for this scale.

2. **Cloud Run WebSocket support**: Cloud Run supports WebSockets but with the same concurrency model. Doesn't solve the slot exhaustion problem.

3. **Separate domain for SSE** (`mcp.pi.ruv.io`): **Adopted.** Routes SSE to `ruvbrain-sse` Cloud Run service via Cloudflare CNAME → `ghs.googlehosted.com`. The SSE proxy is a thin binary with no business logic — it forwards JSON-RPC to the API and polls `/internal/queue/drain` for responses.

4. **Cloud Run min-instances scaling**: Set `min-instances=3` to absorb SSE load across more instances. Increases cost 3x without solving the architectural issue. Band-aid.

5. **Move to GKE**: Full Kubernetes with separate deployments per concern. Correct architecture but massive operational overhead for a single-developer project.

6. **Redis Pub/Sub**: External broker. Adds a managed service dependency. Overkill for session-scoped message passing where the in-process queue suffices.

## Cost Impact

| Component | Current | After Split |
|-----------|---------|-------------|
| API (ruvbrain) | ~$40/mo (2 CPU, 4GB, always-on) | ~$30/mo (same but less load) |
| SSE service | included above | ~$10/mo (1 CPU, 512MB) |
| Worker jobs | included above | ~$15/mo (on-demand, job-based) |
| **Total** | **~$40/mo** | **~$55/mo** |

$15/month increase for elimination of all concurrency-related outages.

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| SSE ↔ API latency | 100ms polling adds latency to MCP responses | Use internal SSE stream instead of polling; fallback to 50ms poll |
| Worker ↔ API cache coherence | Worker writes to Firestore; API has stale in-memory cache | API refreshes from Firestore on cache miss; add TTL to cached data |
| Increased deploy complexity | 3 services instead of 1 | Shared Dockerfile base; single `deploy_all.sh` script |
| SSE service statelessness | Session affinity needed for SSE reconnects | Cloud Run session affinity on SSE service |

## References

- [ADR-066: SSE MCP Transport](./ADR-066-sse-mcp-transport.md)
- [ADR-077: Midstream Brain Integration](./ADR-077-midstream-brain-integration.md)
- [Cloud Run concurrency model](https://cloud.google.com/run/docs/about-concurrency)
- [MCP SSE Transport specification](https://modelcontextprotocol.io/docs/concepts/transports#sse)
