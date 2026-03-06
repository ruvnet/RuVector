# ADR-064: Pi Brain Infrastructure & Landing Page

**Status**: Accepted, Deployed
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud), ADR-060 (Shared Brain Capabilities), ADR-066 (SSE MCP Transport)

## 1. Context

The Shared Brain (ADR-059, ADR-060) requires a production deployment surface that is both functional (serving REST and MCP endpoints) and discoverable (a public-facing landing page that communicates what the brain is). A bare API with no human-readable entry point creates adoption friction. Developers who visit the domain should immediately understand the system and how to connect.

Google Cloud Run provides a serverless container platform with automatic TLS, scaling to zero, and custom domain mapping. Custom domains (pi.ruv.io, the Unicode variant) make the brain addressable under a memorable, branded namespace. A landing page with a Three.js visualization gives the project a distinctive identity rooted in the Foundation (Asimov) sci-fi aesthetic.

## 2. Decision

Deploy `mcp-brain-server` as a Cloud Run service (`ruvbrain`) in `us-central1`. Map custom domains through Google Cloud's domain verification and Cloudflare DNS. Embed static HTML landing pages at compile time using `include_str!` so the binary is fully self-contained with no runtime file I/O for static content.

## 3. Architecture

### 3.1 Cloud Run Service

| Property | Value |
|----------|-------|
| Service name | `ruvbrain` |
| Region | `us-central1` |
| Image | Multi-stage Dockerfile: `rust:1.85-bookworm` builder, `debian:bookworm-slim` runtime |
| Port | 8080 (configured via `PORT` env var) |
| Scaling | 0-10 instances, 2 CPU / 2Gi RAM per instance |
| Concurrency | Default (80 requests per instance) |
| Startup | `mcp-brain-server` binary, tracing to stderr |

The Dockerfile uses a two-stage build. The builder stage compiles the full workspace with `cargo build --release -p mcp-brain-server`. The runtime stage copies only the binary and installs `ca-certificates` for HTTPS outbound calls to Firestore and GCS.

### 3.2 Custom Domains

| Domain | Type | Target |
|--------|------|--------|
| `pi.ruv.io` | CNAME | `ghs.googlehosted.com` |
| `xn--1xa.ruv.io` (pi.ruv.io) | CNAME | `ghs.googlehosted.com` |

DNS is managed through Cloudflare. The CNAME records point to Google's hosted services endpoint, which handles TLS termination and routes to the Cloud Run service. Google Cloud domain mapping verifies ownership and provisions managed TLS certificates automatically.

### 3.3 Persistence Layer

**Firestore**: Stores brain memories, page metadata, WASM node metadata, contributor reputations, and LoRA federation state. The `FirestoreClient` hydrates an in-memory cache on startup via `load_from_firestore()`. Writes are dual-written to both local cache and Firestore. When `FIRESTORE_URL` is not set, the server operates in local-only mode for development.

**Google Cloud Storage (GCS)**: Stores RVF containers and WASM node binaries. Content-addressed storage for WASM nodes uses `gs://ruvector-brain-{region}/nodes/sha256/{hash}.wasm`. The `GcsClient` handles uploads and signed URL generation.

### 3.4 Landing Page

The root route (`/`) serves a Three.js Prime Radiant visualization. The page renders an interactive 3D particle system representing the brain's knowledge topology. The aesthetic follows a Foundation (Asimov) sci-fi theme: dark background, blue-white particle fields, Encyclopedia Galactica typography.

Implementation: `include_str!("../static/index.html")` embeds the HTML at compile time. The handler returns the static content with `Content-Type: text/html; charset=utf-8` and `Cache-Control: public, max-age=300`. No runtime filesystem access required.

### 3.5 Origin Story

The `/origin` route serves an animated narrative page explaining the brain's purpose and design philosophy. Same embed pattern: `include_str!("../static/origin.html")`. Same cache headers.

### 3.6 Static Embed Pattern

```rust
async fn landing_page() -> (StatusCode, [(HeaderName, &'static str); 2], &'static str) {
    (
        StatusCode::OK,
        [
            (CONTENT_TYPE, "text/html; charset=utf-8"),
            (CACHE_CONTROL, "public, max-age=300"),
        ],
        include_str!("../static/index.html"),
    )
}
```

This pattern has three advantages:
1. **Zero runtime I/O**: No file reads, no path resolution, no directory traversal risk
2. **Single binary deployment**: The container image contains one executable with no static file dependencies
3. **Compile-time verification**: If the HTML file is missing, the build fails immediately

## 4. Implementation

### 4.1 Router Structure

The `create_router()` function in `routes.rs` constructs the full axum Router:

```rust
Router::new()
    .route("/", get(landing_page))
    .route("/origin", get(origin_page))
    .route("/v1/health", get(health))
    // ... 14 REST endpoints (ADR-060)
    // ... 6 Brainpedia endpoints (ADR-062)
    // ... 5 WASM node endpoints (ADR-063)
    .route("/sse", get(sse_handler))
    .route("/messages", post(messages_handler))
    .layer(CorsLayer::new()
        .allow_origin([
            "https://brain.ruv.io",
            "https://pi.ruv.io",
            "https://ruvbrain-875130704813.us-central1.run.app",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]))
    .layer(TraceLayer::new_for_http())
    .layer(RequestBodyLimitLayer::new(1_048_576))  // 1MB
```

### 4.2 Health Endpoint

`GET /v1/health` reports service identity, version, uptime, and persistence mode:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "domain": "pi.ruv.io",
  "uptime_seconds": 3600,
  "persistence_mode": "firestore"
}
```

### 4.3 Application State

The `AppState` struct holds all shared service state:

| Field | Type | Purpose |
|-------|------|---------|
| `store` | `Arc<FirestoreClient>` | Memory and page persistence |
| `gcs` | `Arc<GcsClient>` | Binary artifact storage |
| `graph` | `Arc<RwLock<KnowledgeGraph>>` | Mincut topology graph |
| `rate_limiter` | `Arc<RateLimiter>` | BudgetTokenBucket rate limiting |
| `ranking` | `Arc<RwLock<RankingEngine>>` | Search result ranking |
| `cognitive` | `Arc<RwLock<CognitiveEngine>>` | DentateGyrus pattern separation |
| `drift` | `Arc<RwLock<DriftMonitor>>` | Embedding centroid drift tracking |
| `aggregator` | `Arc<ByzantineAggregator>` | LoRA weight aggregation |
| `domain_engine` | `Arc<RwLock<DomainExpansionEngine>>` | Cross-domain transfer |
| `sona` | `Arc<RwLock<SonaEngine>>` | Embedding feature extraction |
| `lora_federation` | `Arc<RwLock<LoraFederationStore>>` | Federated LoRA state |
| `nonce_store` | `Arc<NonceStore>` | Challenge nonce replay protection |
| `sessions` | `Arc<DashMap<String, mpsc::Sender>>` | SSE session management |

### 4.4 Deployment Flow

1. Build container image from workspace root using the `crates/mcp-brain-server/Dockerfile`
2. Push to Google Artifact Registry
3. Deploy to Cloud Run with `gcloud run deploy ruvbrain`
4. Map custom domains via Google Cloud Console
5. Configure Cloudflare CNAME records pointing to `ghs.googlehosted.com`
6. Verify TLS certificate provisioning

### 4.5 Endpoint Summary

The deployed server exposes the following route groups:

| Group | Routes | Source ADR |
|-------|--------|-----------|
| Landing pages | `/`, `/origin` | ADR-064 (this document) |
| Infrastructure | `/v1/health`, `/v1/challenge` | ADR-059 |
| Memories (CRUD) | `/v1/memories`, `/v1/memories/search`, `/v1/memories/list`, `/v1/memories/:id`, `/v1/memories/:id/vote` | ADR-060 |
| Transfer & Monitoring | `/v1/transfer`, `/v1/drift`, `/v1/partition`, `/v1/status` | ADR-060 |
| LoRA Federation | `/v1/lora/latest`, `/v1/lora/submit` | ADR-060 |
| Training | `/v1/training/preferences` | ADR-061 |
| Brainpedia | `/v1/pages`, `/v1/pages/:id`, `/v1/pages/:id/deltas`, `/v1/pages/:id/evidence`, `/v1/pages/:id/promote` | ADR-062 |
| WASM Nodes | `/v1/nodes`, `/v1/nodes/:id`, `/v1/nodes/:id/wasm`, `/v1/nodes/:id/revoke` | ADR-063 |
| MCP SSE | `/sse`, `/messages` | ADR-066 |

Total: 27 routes serving REST API, MCP SSE transport, and static HTML.

### 4.6 Security Layers

The server applies the following security measures at the infrastructure level:

1. **TLS termination**: Handled by Google Cloud Run / Cloudflare edge. The server itself listens on plain HTTP on port 8080.
2. **CORS**: Explicit origin allowlist (5 origins). No wildcards.
3. **Request body limit**: 1MB via tower-http `RequestBodyLimitLayer`. Prevents memory exhaustion from oversized payloads.
4. **Rate limiting**: `BudgetTokenBucket` per contributor. 100 writes/hour, 1000 reads/hour. Stale buckets evicted every 1000 operations.
5. **Challenge nonces**: Single-use, 5-minute TTL. Required for write operations.
6. **PII stripping**: 12-pattern detection on all incoming text fields (filesystem paths, API keys, tokens, email patterns).
7. **Embedding verification**: NaN, Inf, and magnitude checks on all incoming vectors.
8. **Ed25519 signature verification**: For RVF container and WASM node integrity.
9. **SHAKE-256 witness chains**: For provenance verification on all operations.

### 4.7 Tracing and Observability

The server uses `tracing_subscriber` with `EnvFilter` for log level control. Logs are written to stderr (Cloud Run captures stderr as structured logs). The `TraceLayer` middleware logs HTTP request/response metadata including method, path, status code, and latency.

The health endpoint provides basic liveness/readiness signaling. Cloud Run uses this for automatic health checks and instance lifecycle management.

## 5. Consequences

### Positive

- **Single binary deployment**: No static file serving complexity, no CDN configuration needed for the landing page
- **Zero-cost at rest**: Cloud Run scales to zero when idle, no persistent infrastructure cost
- **Branded access**: `pi.ruv.io` is memorable and communicates the mathematical/scientific identity
- **Self-documenting**: Visiting the domain explains the system without needing external documentation
- **Unified surface**: REST API, MCP SSE, and landing pages all served from the same binary on the same port

### Negative

- **HTML changes require recompilation**: Modifying the landing page or origin story requires a full cargo build and redeploy. This is acceptable because these pages change infrequently.
- **Single region**: `us-central1` only. Multi-region would require Cloud Run service mesh or global load balancer. Deferred until latency becomes a concern.
- **Cloud vendor lock-in**: Firestore and GCS are Google-specific. The `FirestoreClient` and `GcsClient` abstractions limit blast radius, but migration would require new storage implementations.

### Neutral

- CORS allows five specific origins. New frontends require a code change and redeploy.
- Request body limit is 1MB globally. WASM node uploads are constrained to this limit, matching the ADR-063 spec.
- The `read_only` flag on `AppState` enables emergency read-only mode without redeployment, controlled via an atomic boolean.
