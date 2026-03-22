# ADR-118: Cost-Effective Common Crawl Strategy with Sparsifier-Aware Guardrails

**Status**: Accepted
**Date**: 2026-03-21
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: None
**Related**: ADR-115 (Common Crawl Integration), ADR-116 (Spectral Sparsifier), ADR-096 (Cloud Pipeline), ADR-117 (DragNES Dermatology)

## 1. Context

ADR-115 validated PiQ3 compression (8.68x, 99% recall) and defined a Common Crawl integration architecture. However, the initial cost estimates ($160-480/mo) assume always-on retrieval and Memorystore caching, which is premature for a brain with 1,554 memories.

**Current brain state**:
- 1,554 memories, 340K graph edges
- Sparsifier achieves 27x edge compression (340K -> 12K)
- `/v1/partition` times out: MinCut on 340K edges exceeds Cloud Run's 300s limit
- PiQ3 validated at 8.68x compression, 99% recall
- RlmEmbedder runs on CPU at 128-dim (no GPU or external API needed)

**Problem**: We need a crawl strategy that starts at $11-28/month and scales predictably, with hard guardrails to prevent cost overruns and graph explosion.

## 2. Decision

Implement a three-phase tiered crawl strategy with sparsifier-aware cost guardrails, starting with medical/dermatology domains at $11-28/month.

Key principles:
1. Start minimal, scale only after validating cost/value at each tier
2. Enforce hard daily limits on pages, memories, and Firestore writes
3. Run all partition/MinCut queries on sparsified graphs, never full graphs
4. Cache partition results with background recompute
5. Use RlmEmbedder (CPU, 128-dim) exclusively -- no external embedding APIs

## 3. Three-Phase Budget Model

### Phase 1: Medical Domain ($11-28/mo)

Target: 5K-15K new memories/month from medical and dermatology sources.

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| Cloud Run (crawl job, 30min/day) | $3-8 | Scale-to-zero, bursty pattern |
| Firestore (5K-15K writes + reads) | $2-5 | Document + subcollection ops |
| Cloud Scheduler (2 crawl + 1 partition + 1 cost report) | $0.10 | 4 scheduled jobs |
| GCS (compressed embeddings) | $0.50 | PiQ3-compressed, <1 GB total |
| CDX cache (SQLite on disk) | $0 | Local to Cloud Run instance, ephemeral |
| RlmEmbedder (CPU, 128-dim) | $0 | Runs in-process, no external API |
| Egress (internal only) | $0-5 | Minimal cross-region traffic |
| Cloud Monitoring + alerting | $0.50 | Free tier covers basic alerting |
| Buffer (20%) | $5-10 | Headroom for spikes |
| **Total** | **$11-28** | |

**Domains**: pubmed.ncbi.nlm.nih.gov, aad.org, jaad.org, nejm.org, lancet.com, bmj.com, dermnetnz.org, skincancer.org, melanoma.org

**Graduation criteria**: Recall >= 0.90 on medical domain queries, cost stable for 30 consecutive days, no partition timeouts.

### Phase 2: Academic + News ($73-108/mo)

Target: 50K-100K new memories/month.

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| Cloud Run (crawl, 2hrs/day) | $15-30 | Higher duty cycle |
| Firestore (50K-100K writes) | $20-30 | Scales with memory count |
| GCS (compressed, ~5-10 GB) | $2-5 | Growth from Phase 1 |
| CDX cache (small Redis or SQLite) | $5-15 | Persistent cache for larger query volume |
| Cloud Scheduler (6 jobs) | $0.30 | Additional domain crawlers |
| Egress | $5-10 | More cross-service traffic |
| Monitoring | $1 | Additional alert rules |
| Buffer (20%) | $15-20 | |
| **Total** | **$73-108** | |

**Additional domains**: arxiv.org, en.wikipedia.org, techcrunch.com, arstechnica.com, nature.com, science.org

**Graduation criteria**: Phase 1 metrics sustained, graph sharding design validated, budget approved by owner.

### Phase 3: Broad Web ($158-308/mo)

Target: 500K-1M new memories/month via WET segment processing.

| Item | Monthly Cost | Notes |
|------|-------------|-------|
| Cloud Run (crawl + ingest, 4hrs/day) | $40-80 | WET segment processing |
| Firestore (500K-1M writes) | $50-100 | Significant write volume |
| GCS (compressed, 50-100 GB) | $5-15 | Larger corpus |
| CDX cache (Redis 4 GiB) | $30-60 | High query volume |
| Cloud Scheduler (10+ jobs) | $0.50 | Multiple domain schedulers |
| Egress | $10-20 | |
| Monitoring | $2-3 | |
| Buffer (20%) | $25-50 | |
| **Total** | **$158-308** | |

**Graduation criteria**: Phase 2 metrics sustained, graph sharding operational, aggressive sparsification validated at 2M+ edges.

## 4. Cost Guardrails

Hard limits enforced at the application layer. These are not suggestions -- the crawl pipeline must check and enforce these before every operation.

```rust
/// Cost guardrails enforced by the crawl pipeline.
/// These prevent runaway spending by capping daily operations.
pub struct CostGuardrails {
    /// Maximum pages fetched from Common Crawl CDX per day.
    /// Prevents bandwidth and compute spikes.
    pub max_pages_per_day: u32,

    /// Maximum new memories created per day (after dedup + novelty filter).
    /// Controls Firestore write costs and graph growth rate.
    pub max_new_memories_per_day: u32,

    /// Edge count threshold that triggers aggressive sparsification.
    /// Prevents MinCut timeout by keeping partition graph small.
    pub max_graph_edges: u64,

    /// Hard cap on Firestore write operations per day.
    /// Direct cost control for the largest variable cost item.
    pub max_firestore_writes_per_day: u32,

    /// USD threshold that triggers budget alert via Cloud Monitoring.
    /// When monthly projection exceeds this, crawl jobs are paused.
    pub budget_alert_threshold_usd: f64,

    /// Novelty threshold for deduplication.
    /// Skip ingestion if cosine similarity to existing memory > (1 - threshold).
    /// Default 0.05 means skip if cosine > 0.95.
    pub novelty_threshold: f32,
}

impl CostGuardrails {
    /// Phase 1: Medical domain, minimal spend
    pub fn phase1() -> Self {
        Self {
            max_pages_per_day: 500,
            max_new_memories_per_day: 200,
            max_graph_edges: 500_000,
            max_firestore_writes_per_day: 5_000,
            budget_alert_threshold_usd: 30.0,
            novelty_threshold: 0.05,
        }
    }

    /// Phase 2: Academic + news, moderate spend
    pub fn phase2() -> Self {
        Self {
            max_pages_per_day: 5_000,
            max_new_memories_per_day: 2_000,
            max_graph_edges: 2_000_000,
            max_firestore_writes_per_day: 50_000,
            budget_alert_threshold_usd: 120.0,
            novelty_threshold: 0.05,
        }
    }

    /// Phase 3: Broad web, higher limits
    pub fn phase3() -> Self {
        Self {
            max_pages_per_day: 20_000,
            max_new_memories_per_day: 10_000,
            max_graph_edges: 5_000_000,
            max_firestore_writes_per_day: 200_000,
            budget_alert_threshold_usd: 350.0,
            novelty_threshold: 0.05,
        }
    }

    /// Returns true if the content should be skipped (too similar to existing)
    pub fn should_skip(&self, cosine_similarity: f32) -> bool {
        cosine_similarity > (1.0 - self.novelty_threshold)
    }
}
```

### Guardrail Enforcement Points

| Checkpoint | Guardrail | Action on Breach |
|-----------|-----------|-----------------|
| Before CDX query | `max_pages_per_day` | Skip crawl, log warning |
| Before memory creation | `max_new_memories_per_day` | Queue for next day |
| After memory creation | `max_graph_edges` | Trigger aggressive sparsification |
| Before Firestore write | `max_firestore_writes_per_day` | Buffer to next day |
| Hourly cost projection | `budget_alert_threshold_usd` | Pause all crawl jobs, alert owner |
| Before embedding comparison | `novelty_threshold` | Skip duplicate, log |

## 5. Sparsifier-Aware Graph Management

The spectral sparsifier (ADR-116) is critical for keeping partition queries fast and costs low.

### Edge Threshold Policy

| Edge Count | Action | Sparsifier Epsilon | Expected Sparse Edges |
|-----------|--------|-------------------|----------------------|
| < 100K | Partition on full graph | N/A | N/A |
| 100K - 500K | Partition on sparsified graph | 0.3 (default) | ~4K-18K |
| 500K - 2M | Aggressive sparsification | 0.5 | ~10K-40K |
| > 2M | Graph sharding + aggressive sparsify | 0.7 + shard | ~6K-30K per shard |

### Partition on Sparsified Graph

**Rule**: All MinCut/partition queries MUST use sparsified edges when `edge_count > 100K`.

The sparsifier preserves cut structure (spectral guarantee) while reducing edge count by 20-30x. This is not an approximation shortcut -- it is a mathematically justified reduction that preserves the properties MinCut needs.

**Current example**: 340K edges -> 12K sparsified (27x). MinCut on 12K edges completes in ~5-15 seconds vs timeout on 340K.

## 6. Partition Caching Strategy

### Problem
`/v1/partition` computes MinCut on every request. At 340K edges, this exceeds Cloud Run's 300-second timeout.

### Solution: Cache + Background Recompute

```rust
/// Cached partition result, stored in Firestore or in-memory.
/// Served directly on /v1/partition requests.
pub struct CachedPartition {
    /// The computed cluster assignments
    pub clusters: Vec<Cluster>,
    /// When this partition was computed
    pub computed_at: DateTime<Utc>,
    /// Cache TTL in seconds (default: 3600 = 1 hour)
    pub ttl_seconds: u64,
    /// Whether this was computed on the sparsified graph
    pub used_sparsified: bool,
    /// Number of edges used in computation
    pub edge_count: u64,
    /// Sparsifier epsilon used
    pub epsilon: f32,
}

impl CachedPartition {
    /// Check if cached result is still valid
    pub fn is_valid(&self) -> bool {
        let elapsed = Utc::now() - self.computed_at;
        elapsed.num_seconds() < self.ttl_seconds as i64
    }
}
```

### Request Flow

```
GET /v1/partition
  |
  v
[Cache hit + valid?] --yes--> Return cached result (< 10ms)
  |
  no
  |
  v
[Sparsified edges available?] --yes--> Compute on sparse (5-15s)
  |                                      |
  no                                     v
  |                                   Cache result, return
  v
[Edge count < 100K?] --yes--> Compute on full graph
  |
  no
  |
  v
Return 503 "Partition computing, try again in 60s"
  + Trigger background recompute
```

### Background Recompute Schedule

```yaml
# Cloud Scheduler: recompute partition hourly
- name: brain-partition-recompute
  schedule: "0 * * * *"
  target:
    uri: /v1/partition/recompute
    httpMethod: POST
  body: '{"use_sparsified": true, "timeout_seconds": 120}'
  retryConfig:
    retryCount: 2
    maxBackoffDuration: "60s"
```

## 7. Cloud Scheduler Configuration

### Phase 1 Jobs

```yaml
# Medical domain crawl - daily 2AM UTC
- name: brain-crawl-medical
  schedule: "0 2 * * *"
  target:
    uri: /v1/pipeline/crawl/ingest
    httpMethod: POST
  body: |
    {
      "domains": [
        "pubmed.ncbi.nlm.nih.gov",
        "aad.org",
        "jaad.org",
        "nejm.org",
        "lancet.com",
        "bmj.com"
      ],
      "limit": 500,
      "options": {
        "skip_duplicates": true,
        "compute_novelty": true,
        "novelty_threshold": 0.05,
        "guardrails": "phase1"
      }
    }
  retryConfig:
    retryCount: 1

# Dermatology-specific crawl - daily 3AM UTC
- name: brain-crawl-derm
  schedule: "0 3 * * *"
  target:
    uri: /v1/pipeline/crawl/ingest
    httpMethod: POST
  body: |
    {
      "domains": [
        "dermnetnz.org",
        "skincancer.org",
        "dermoscopy-ids.org",
        "melanoma.org",
        "bad.org.uk"
      ],
      "limit": 200,
      "options": {
        "skip_duplicates": true,
        "compute_novelty": true,
        "novelty_threshold": 0.05,
        "guardrails": "phase1"
      }
    }
  retryConfig:
    retryCount: 1

# Partition recompute - hourly
- name: brain-partition-recompute
  schedule: "0 * * * *"
  target:
    uri: /v1/partition/recompute
    httpMethod: POST
  body: '{"use_sparsified": true, "timeout_seconds": 120}'
  retryConfig:
    retryCount: 2

# Weekly cost report - Sunday 6AM UTC
- name: brain-cost-report
  schedule: "0 6 * * 0"
  target:
    uri: /v1/pipeline/cost/report
    httpMethod: POST
  body: '{"share_to_brain": true, "alert_if_over_budget": true}'
  retryConfig:
    retryCount: 1
```

## 8. Cost Monitoring

### /v1/pipeline/cost Endpoint

Returns current cost estimates and guardrail status.

```json
{
  "period": "current_month",
  "estimated_monthly_usd": 18.50,
  "breakdown": {
    "cloud_run_compute": 5.20,
    "firestore_ops": 3.10,
    "gcs_storage": 0.45,
    "scheduler": 0.10,
    "egress": 2.15,
    "other": 0.50
  },
  "guardrails": {
    "pages_today": 342,
    "pages_limit": 500,
    "memories_today": 187,
    "memories_limit": 200,
    "graph_edges": 352000,
    "edge_limit": 500000
  },
  "alerts": [],
  "phase": "phase1"
}
```

### Alert Escalation

| Condition | Action | Notification |
|----------|--------|-------------|
| Daily spend > $2 | Log warning | Cloud Monitoring -> Slack |
| Weekly spend > $15 | Reduce `max_pages_per_day` by 50% | Email alert |
| Monthly projection > `budget_alert_threshold_usd` | Pause all crawl jobs | Email + Slack + brain memory |
| Graph edges > 80% of `max_graph_edges` | Trigger aggressive sparsification | Log info |
| Graph edges > 100% of `max_graph_edges` | Pause memory creation, sparsify | Cloud Monitoring alert |

### Audit Trail

The weekly cost report (triggered by `brain-cost-report` scheduler job) is shared as a brain memory, creating an immutable audit trail of spending over time.

## 9. Anti-Patterns

| Anti-Pattern | Why It Fails | Cost Impact |
|-------------|-------------|------------|
| Download full WET segments in Phase 1 | Each segment is 100+ MB compressed; thousands per crawl | $1,000+/mo |
| Use external embedding APIs (OpenAI, Cohere) | Millions of embeddings at $0.0001-0.001 each | $500+/mo |
| Skip novelty filtering | Graph explodes with near-duplicate memories | Firestore + compute spiral |
| Run MinCut on full graph (>100K edges) | O(V*E*log V) exceeds Cloud Run 300s timeout | Timeout errors |
| Store raw HTML in Firestore | Average page is 50-100KB; Firestore charges per byte | $500+/mo at scale |
| Use GPU for RlmEmbedder | 128-dim HashEmbedder is CPU-efficient by design | $200+/mo unnecessary |
| Skip sparsification before partition | Full graph partition is ~100x slower than sparsified | Wasted compute |
| No daily caps | A bug or config error can drain budget in hours | Unbounded |

## 10. Acceptance Criteria

### Phase 1 Acceptance (Must Pass Before Phase 2)

| Criterion | Target | Measurement |
|----------|--------|-------------|
| Monthly cost | <= $28 | GCP billing report |
| Memories ingested/month | >= 5,000 | Brain memory count delta |
| Recall@10 on medical queries | >= 0.90 | Benchmark against uncompressed baseline |
| Partition latency (cached) | < 100ms | Cloud Run metrics |
| Partition latency (recompute) | < 30s | Scheduler job duration |
| No partition timeouts | 0 in 30 days | Cloud Run error logs |
| Cost guardrails enforced | All limits respected | Application logs |
| Novelty filter active | Skip rate > 20% | Pipeline metrics |
| Cost stable for 30 days | No single day > $2 | GCP billing |

### Phase 2 Acceptance

| Criterion | Target |
|----------|--------|
| Monthly cost | <= $108 |
| Memories ingested/month | >= 50,000 |
| Recall@10 across all domains | >= 0.90 |
| Graph edges managed by sparsifier | < 2M |
| Cost stable for 30 days | No single day > $5 |

### Phase 3 Acceptance

| Criterion | Target |
|----------|--------|
| Monthly cost | <= $308 |
| Memories ingested/month | >= 500,000 |
| Graph sharding operational | >= 2 shards |
| No partition timeouts | 0 in 30 days |
| Cost stable for 30 days | No single day > $15 |

## 11. References

- [ADR-115: Common Crawl Integration with Semantic Compression](./ADR-115-common-crawl-temporal-compression.md)
- [ADR-116: Spectral Sparsifier Brain Integration](./ADR-116-spectral-sparsifier-brain-integration.md)
- [ADR-096: Cloud Pipeline](./ADR-096-cloud-pipeline-realtime-optimization.md)
- [ADR-117: DragNES Dermatology Intelligence Platform](./ADR-117-dragnes-dermatology-intelligence-platform.md)
- [Cloud Run Pricing](https://cloud.google.com/run/pricing)
- [Firestore Pricing](https://cloud.google.com/firestore/pricing)
- [Cloud Scheduler Pricing](https://cloud.google.com/scheduler/pricing)
