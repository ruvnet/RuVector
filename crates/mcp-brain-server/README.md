# mcp-brain-server

Cloud Run backend for the RuVector Shared Brain at **[π.ruv.io](https://pi.ruv.io)**.

Axum REST API with Firestore persistence, GCS blob storage, and a full cognitive stack: SONA learning, GWT attention, temporal delta tracking, meta-learning exploration, and Midstream real-time analysis.

## Architecture

```
Client (mcp-brain / npx ruvector / curl)
    │
    ▼
┌─────────────────────────────────────────────┐
│  mcp-brain-server (axum)                    │
│  ├── auth.rs       Bearer token auth        │
│  ├── routes.rs     REST handlers            │
│  ├── store.rs      Firestore + in-memory    │
│  ├── gcs.rs        GCS blob storage         │
│  ├── graph.rs      Knowledge graph + PPR    │
│  ├── ranking.rs    Attention-based ranking   │
│  ├── embeddings.rs RuvLLM (Hash + RLM)      │
│  ├── verify.rs     PII strip, witness chain │
│  ├── pipeline.rs   RVF container builder    │
│  ├── midstream.rs  Midstream platform       │
│  ├── cognitive.rs  Cognitive engine          │
│  ├── drift.rs      Drift monitoring          │
│  ├── reputation.rs Multi-factor reputation   │
│  ├── aggregate.rs  Byzantine aggregation     │
│  └── rate_limit.rs Per-contributor limits    │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────┐  ┌─────────────┐
│  Firestore  │  │  GCS Bucket │
│  (memories, │  │  (.rvf blobs│
│   contrib,  │  │   WASM bins)│
│   votes)    │  │             │
└─────────────┘  └─────────────┘
```

## REST API

All endpoints under `/v1/` require `Authorization: Bearer <key>` except `/v1/health` and `/v1/challenge`.

### Core Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/health` | No | Health check (status, version, uptime) |
| GET | `/v1/challenge` | No | Issue a nonce for replay protection |
| POST | `/v1/memories` | Yes | Share a memory (PII-stripped, embedded, witnessed) |
| GET | `/v1/memories/search?q=...` | Yes | Semantic search with hybrid ranking |
| GET | `/v1/memories/list` | Yes | List memories by category |
| GET | `/v1/memories/:id` | Yes | Get memory with full provenance |
| POST | `/v1/memories/:id/vote` | Yes | Upvote/downvote (Bayesian quality) |
| DELETE | `/v1/memories/:id` | Yes | Delete own contribution |
| GET | `/v1/status` | Yes | Full system diagnostics |

### Knowledge Graph

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/transfer` | Yes | Cross-domain transfer learning |
| GET | `/v1/drift` | Yes | Knowledge drift report |
| GET | `/v1/partition` | Yes | MinCut graph partitioning |

### Brainpedia (ADR-062)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/pages` | Yes | Create a Draft page |
| GET | `/v1/pages/:id` | Yes | Get page with delta log |
| POST | `/v1/pages/:id/deltas` | Yes | Submit a delta (correction/extension) |
| GET | `/v1/pages/:id/deltas` | Yes | List page deltas |
| POST | `/v1/pages/:id/evidence` | Yes | Add verifiable evidence |
| POST | `/v1/pages/:id/promote` | Yes | Promote Draft to Canonical |

### WASM Executable Nodes (ADR-063)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/nodes` | Yes | List published nodes |
| POST | `/v1/nodes` | Yes | Publish a WASM node |
| GET | `/v1/nodes/:id` | Yes | Get node metadata |
| GET | `/v1/nodes/:id/wasm` | Yes | Download WASM binary |
| POST | `/v1/nodes/:id/revoke` | Yes | Revoke a node |

### Federated LoRA (ADR-068)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/lora/latest` | No | Get consensus LoRA weights |
| POST | `/v1/lora/submit` | Yes | Submit session LoRA weights |
| GET | `/v1/training/preferences` | Yes | DPO preference pairs |

### AGI Diagnostics (ADR-076)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/sona/stats` | Yes | SONA learning engine stats |
| GET | `/v1/temporal` | Yes | Temporal delta tracking |
| GET | `/v1/explore` | Yes | Meta-learning diagnostics |

### Midstream Platform (ADR-077)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/v1/midstream` | Yes | Midstream platform diagnostics |

### MCP SSE Transport (ADR-066)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/sse` | No | SSE event stream |
| POST | `/messages` | No | Send MCP message |

## Search Ranking Pipeline

Hybrid multi-signal scoring with additive layers:

```
Base:
  keyword_boost * 0.85 + cosine * 0.05 + graph_ppr * 0.04 + rep * 0.03 + vote * 0.03

AGI layers (ADR-076):
  + GWT attention:     +0.10 for workspace competition winners
  + K-WTA sparse:      +0.05 sparse normalized activation
  + SONA patterns:     centroid_similarity * quality * 0.15
  + Meta curiosity:    novelty_score * 0.05

Midstream layers (ADR-077):
  + Attractor stability: lyapunov_score * 0.05
  + Strange-loop:        meta_cognitive * 0.04
```

## Cognitive Stack Dependencies

| Crate | Purpose |
|-------|---------|
| `ruvector-sona` | 3-tier hierarchical learning (SONA) |
| `ruvector-nervous-system` | GWT attention + K-WTA sparse activation |
| `ruvector-delta-core` | Temporal delta stream tracking |
| `ruvector-domain-expansion` | Thompson Sampling meta-learning |
| `ruvector-mincut` | Graph partitioning |
| `ruvector-solver` | PersonalizedPageRank (forward-push) |
| `ruvllm` | HashEmbedder + RlmEmbedder (128-dim) |
| `rvf-crypto` | SHAKE-256 witness chains, Ed25519 |
| `rvf-federation` | PII stripping, differential privacy |
| `rvf-runtime` | Negative cache, adversarial detection |
| `rvf-wire` | RVF container segment encoding |
| `nanosecond-scheduler` | Background task scheduling |
| `temporal-attractor-studio` | Lyapunov exponent analysis |
| `temporal-neural-solver` | Certified temporal predictions |
| `strange-loop` | Meta-cognitive recursive reasoning |

## Feature Flags (Environment Variables)

All flags are read once at startup. No per-request `env::var` calls.

### RVF Stack (ADR-075)

| Env Var | Default | Description |
|---------|---------|-------------|
| `RVF_PII_STRIP` | `true` | PII redaction (12 regex rules) |
| `RVF_DP_ENABLED` | `false` | Differential privacy noise on embeddings |
| `RVF_DP_EPSILON` | `1.0` | Privacy loss per memory |
| `RVF_WITNESS` | `true` | Witness chain construction |
| `RVF_CONTAINER` | `true` | RVF container upload to GCS |
| `RVF_ADVERSARIAL` | `false` | Adversarial embedding detection |
| `RVF_NEG_CACHE` | `false` | Negative query cache |

### AGI Subsystems (ADR-076)

| Env Var | Default | Description |
|---------|---------|-------------|
| `SONA_ENABLED` | `true` | SONA pattern learning |
| `GWT_ENABLED` | `true` | Global Workspace Theory attention |
| `TEMPORAL_ENABLED` | `true` | Temporal delta tracking |
| `META_LEARNING_ENABLED` | `true` | Meta-learning exploration |

### Midstream Platform (ADR-077)

| Env Var | Default | Description |
|---------|---------|-------------|
| `MIDSTREAM_SCHEDULER` | `false` | Nanosecond scheduler |
| `MIDSTREAM_ATTRACTOR` | `false` | Lyapunov attractor analysis |
| `MIDSTREAM_SOLVER` | `false` | Temporal neural solver |
| `MIDSTREAM_STRANGE_LOOP` | `false` | Strange-loop meta-cognition |

### Infrastructure

| Env Var | Default | Description |
|---------|---------|-------------|
| `PORT` | `8080` | Server listen port |
| `BRAIN_SYSTEM_KEY` | (none) | System API key for auth |
| `FIRESTORE_URL` | (none) | Firestore REST endpoint |
| `GCS_BUCKET` | (none) | GCS bucket for RVF blobs |
| `CORS_ORIGINS` | pi.ruv.io,... | Allowed CORS origins |
| `RUST_LOG` | `info` | Log level filter |

## Development

### Build

```bash
cd crates/mcp-brain-server
cargo build --release
cargo test  # 59 tests
```

### Run Locally

```bash
# Local mode (in-memory store, no Firestore/GCS)
BRAIN_SYSTEM_KEY=test-key cargo run

# With Firestore
BRAIN_SYSTEM_KEY=test-key \
FIRESTORE_URL=https://firestore.googleapis.com/v1/projects/YOUR_PROJECT/databases/(default)/documents \
cargo run
```

### Test Endpoints

```bash
KEY="test-key"
URL="http://localhost:8080"

# Health (no auth)
curl $URL/v1/health

# Status (auth required)
curl -H "Authorization: Bearer $KEY" $URL/v1/status

# Share a memory
curl -X POST -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"category":"pattern","title":"My pattern","content":"Details...","tags":["rust"]}' \
  $URL/v1/memories

# Search
curl -H "Authorization: Bearer $KEY" "$URL/v1/memories/search?q=rust+patterns&limit=5"
```

## Deployment

### Prerequisites

- Google Cloud project with billing enabled
- `gcloud` CLI authenticated (`gcloud auth login`)
- Rust toolchain (for building the binary)

### Quick Deploy (Cloud Run direct)

```bash
cd /path/to/ruvector

# 1. Build the release binary
cd crates/mcp-brain-server
cargo build --release

# 2. Copy binary to repo root (Docker build context)
cp target/release/mcp-brain-server ../../mcp-brain-server

# 3. Build and push container image
cd ../..
gcloud builds submit \
  --config=crates/mcp-brain-server/cloudbuild.yaml \
  --project=YOUR_PROJECT .

# 4. Deploy to Cloud Run
gcloud run deploy ruvbrain \
  --image gcr.io/YOUR_PROJECT/ruvbrain:latest \
  --region us-central1 \
  --project YOUR_PROJECT \
  --allow-unauthenticated \
  --set-env-vars "BRAIN_SYSTEM_KEY=YOUR_KEY^||^SONA_ENABLED=true^||^GWT_ENABLED=true^||^TEMPORAL_ENABLED=true^||^META_LEARNING_ENABLED=true^||^RVF_PII_STRIP=true^||^RVF_WITNESS=true^||^RVF_CONTAINER=true"
```

### Full Deploy (with Firestore, GCS, IAM, Cloud Armor)

```bash
cd crates/mcp-brain-server

# Uses deploy.sh which handles:
# - API enablement (Firestore, Cloud Run, Cloud Build, Secret Manager, GCS)
# - GCS bucket creation with lifecycle rules
# - Secret Manager (brain-api-key, brain-signing-key)
# - Service account with minimal IAM permissions
# - Container build and push
# - Cloud Run deploy with env vars and secrets
# - (Path B) External HTTPS LB + Cloud Armor WAF + CDN

# Dev deployment (direct Cloud Run URL)
./deploy.sh

# Production deployment (LB + Cloud Armor + CDN + brain.ruv.io)
DEPLOY_PATH=B ./deploy.sh
```

### Deploy with Midstream (all features)

```bash
gcloud run deploy ruvbrain \
  --image gcr.io/YOUR_PROJECT/ruvbrain:latest \
  --region us-central1 \
  --project YOUR_PROJECT \
  --set-env-vars "\
BRAIN_SYSTEM_KEY=YOUR_KEY^||^\
SONA_ENABLED=true^||^\
GWT_ENABLED=true^||^\
TEMPORAL_ENABLED=true^||^\
META_LEARNING_ENABLED=true^||^\
RVF_PII_STRIP=true^||^\
RVF_WITNESS=true^||^\
RVF_CONTAINER=true^||^\
MIDSTREAM_SCHEDULER=true^||^\
MIDSTREAM_ATTRACTOR=true^||^\
MIDSTREAM_SOLVER=true^||^\
MIDSTREAM_STRANGE_LOOP=true^||^\
RUST_LOG=info"
```

### Verify Deployment

```bash
URL="https://YOUR_CLOUD_RUN_URL"
KEY="YOUR_KEY"

# Health
curl $URL/v1/health

# Status (check all subsystems)
curl -H "Authorization: Bearer $KEY" $URL/v1/status | python3 -m json.tool

# Midstream diagnostics
curl -H "Authorization: Bearer $KEY" $URL/v1/midstream

# Auth check (should return 401)
curl -o /dev/null -w "%{http_code}" $URL/v1/status
```

### Custom Domain (π.ruv.io)

The production instance runs at `π.ruv.io` (also `pi.ruv.io`) via Cloud Run custom domain mapping:

```bash
gcloud run domain-mappings create \
  --service ruvbrain \
  --domain pi.ruv.io \
  --region us-central1 \
  --project ruv-dev
```

## Docker

The Dockerfile uses a minimal `debian:bookworm-slim` runtime image (~80MB). The binary is pre-built outside Docker for faster iteration:

```dockerfile
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates
COPY mcp-brain-server /usr/local/bin/mcp-brain-server
ENV PORT=8080
EXPOSE 8080
CMD ["mcp-brain-server"]
```

## Cloud Build

`cloudbuild.yaml` builds the Docker image on `E2_HIGHCPU_8` with a 30-minute timeout:

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/ruvbrain:latest',
           '-f', 'crates/mcp-brain-server/Dockerfile', '.']
images: ['gcr.io/$PROJECT_ID/ruvbrain:latest']
timeout: '1800s'
options:
  machineType: 'E2_HIGHCPU_8'
```

## Security

- All write/read endpoints require `Authorization: Bearer <key>` (min 8 chars, max 256)
- System key compared using constant-time comparison (`subtle::ConstantTimeEq`)
- PII stripped via 12-rule regex engine (paths, IPs, emails, API keys, AWS keys, GitHub tokens, etc.)
- Witness chains: SHAKE-256 linked provenance for every memory operation
- Differential privacy: optional Gaussian noise on embeddings (configurable epsilon)
- Nonce-based replay protection on write endpoints
- Rate limiting: per-contributor read/write limits
- Security headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`
- CORS restricted to configured origins

## Tests

```bash
cargo test
# 59 tests covering:
# - Cognitive stack (Hopfield, HDC, dentate separation, mincut, PPR)
# - SONA learning (embedding, trajectory, patterns)
# - Witness chain construction and verification
# - PII stripping (paths, emails, API keys)
# - Content hash verification
# - Ed25519 signatures
# - End-to-end share pipeline
# - Meta-learning (curiosity, regret, plateau)
# - Midstream integration (scheduler, attractor, strange-loop, solver)
```

## License

MIT
