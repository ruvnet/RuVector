# Common Crawl Phase 1 Benchmark -- Pipeline Validation Results

**Date:** 2026-03-22
**Branch:** `feature/dragnes`
**Target:** `https://pi.ruv.io`
**Methodology:** Pre/post inject comparison with search verification

---

## 1. Pre-Crawl Baseline

| Metric | Value |
|--------|-------|
| Total memories | 1,564 |
| Graph edges | 349,923 |
| Sparsifier compression | 27.3x (12,799 sparse edges) |
| Clusters | 20 |
| SONA patterns | 0 |
| LoRA epoch | 2 |
| Uptime | 5,983s |
| Pipeline messages processed | 0 |

## 2. Pipeline Endpoint Tests

### Crawl Stats (`/v1/pipeline/crawl/stats`)

- Adapter: `common_crawl`
- CDX cache: 0 entries, 0 hits, 0 misses (fresh state)
- Pages fetched/extracted: 0 (no crawl run yet)
- Web memory: 0 total memories, 0 link edges, 0 domains

### CDX Connectivity Test (`/v1/pipeline/crawl/test`)

| External Service | Status | Success | Body Length |
|-----------------|--------|---------|-------------|
| httpbin | 200 | Yes | 310 bytes |
| Internet Archive CDX | 200 | Yes | 104 bytes |
| Common Crawl data index | 200 | Yes | 9,330 bytes |
| Common Crawl CDX API | 0 | **No** | -- |

- CDX API direct query failed with `IncompleteMessage` after 3 attempts (3,265ms latency)
- Data index and Internet Archive CDX are reachable -- crawl pipeline can use IA CDX as fallback
- Total external test latency: 26,137ms

### Pipeline Metrics (`/v1/pipeline/metrics`)

- Messages received/processed/failed: 0/0/0
- Injections per minute: 0.0
- Optimization cycles: 0

## 3. Inject Test Results

### Single Inject

- **Endpoint:** `POST /v1/pipeline/inject`
- **Source:** `crawl-benchmark`
- **Title:** "Dermoscopic Features of Melanoma: A Systematic Review"
- **Content length:** ~615 characters
- **Tags:** melanoma, dermoscopy, diagnosis, ABCDE, skin-cancer
- **Category:** solution

| Result | Value |
|--------|-------|
| Status | 200 OK |
| Memory ID | `f97de695-6c33-4fbe-afbe-57fb571da10d` |
| Quality score | 0.5 |
| Graph edges added | 811 |
| Witness hash | `9c01cde4...c92773b` |
| **Latency** | **1,262ms** |

### Batch Inject (3 items)

- **Endpoint:** `POST /v1/pipeline/inject/batch`
- **Source:** `crawl-benchmark-batch`

| Item | Title | Category |
|------|-------|----------|
| 1 | Basal Cell Carcinoma Dermoscopy Patterns | solution |
| 2 | Actinic Keratosis: Pre-malignant Skin Lesion Recognition | solution |
| 3 | Dermatofibroma vs Melanoma: Differential Diagnosis | pattern |

| Result | Value |
|--------|-------|
| Status | 200 OK |
| Accepted | 3 |
| Rejected | 0 |
| Errors | [] |
| **Latency** | **2,778ms** |
| Per-item latency | ~926ms |

**Note:** Batch endpoint requires `source` field on each item (not just top-level). Initial attempt without per-item `source` returned HTTP 422.

## 4. Post-Inject Memory State

| Metric | Pre-Inject | Post-Inject | Delta |
|--------|-----------|-------------|-------|
| Total memories | 1,564 | 1,568 | +4 |
| Graph edges | 349,923 | 353,240 | +3,317 |
| Sparsifier edges | 12,799 | 12,924 | +125 |
| Sparsifier compression | 27.3x | 27.3x | unchanged |
| Clusters | 20 | 20 | unchanged |

### Pipeline Metrics After Inject

- Messages received: 4 (1 single + 3 batch)
- Messages processed: 4
- Messages failed: 0
- Injections per minute: 0.04
- **Processing success rate: 100%**

### Graph Growth Analysis

- Average edges per inject: 829 edges/memory
- Edge-to-memory ratio: 225.5 (353,240 / 1,568)
- Sparsifier absorbed 125 new sparse edges (3.8% of raw edges added)

## 5. Search Verification

### Query: "melanoma dermoscopy ABCDE"

| Rank | Title | Score | Category |
|------|-------|-------|----------|
| 1 | **Dermoscopic Features of Melanoma: A Systematic Review** | 1.723 | solution |
| 2 | DrAgnes Specialist Agent Implementation -- Full Code Spec | 1.608 | architecture |
| 3 | DrAgnes Orchestrator Agent -- Second-Opinion Routing Design | 1.471 | solution |

Injected article ranked **#1** with highest score. Tags correctly indexed.

### Query: "basal cell carcinoma arborizing vessels"

| Rank | Title | Score | Category |
|------|-------|-------|----------|
| 1 | **Basal Cell Carcinoma Dermoscopy Patterns** | 1.658 | solution |
| 2 | DrAgnes Phase 2: BCC Specialist + Keratosis Specialist | 1.479 | solution |
| 3 | HAM10000 Dataset Analysis: Class Distribution and Imbalance | 1.318 | pattern |

Injected article ranked **#1**. Semantic search correctly associates "arborizing vessels" with BCC content.

## 6. Latency Summary

| Operation | Latency | Notes |
|-----------|---------|-------|
| Status endpoint | <100ms | Fast health check |
| Crawl stats | <100ms | In-memory counters |
| CDX connectivity test | 26,137ms | 3 external services + CDX retry |
| Single inject | 1,262ms | Includes embedding + graph linking |
| Batch inject (3 items) | 2,778ms | ~926ms per item |
| Search query | <200ms | HNSW vector search |

## 7. Cost Estimate Validation

Based on observed inject latency and the pi.ruv.io infrastructure (Fly.io deployment):

| Metric | Value |
|--------|-------|
| Inject throughput | ~1.0 item/sec (single), ~1.08 items/sec (batch) |
| Estimated 1K articles | ~17 min (batch mode) |
| Estimated 10K articles | ~2.8 hours (batch mode) |
| Compute cost (Fly.io shared-cpu-1x) | ~$0.003/hr |
| Est. cost for 10K article ingest | ~$0.01 compute |
| Graph storage growth per article | ~829 edges |
| Est. graph at 10K articles | ~8.6M edges |

## 8. Findings and Recommendations

### What works well

1. **Pipeline inject** -- both single and batch endpoints function correctly with 100% success rate
2. **Semantic search** -- injected content is immediately searchable and ranks correctly
3. **Graph integration** -- each memory automatically creates ~829 edges for knowledge linking
4. **Sparsifier** -- maintains 27.3x compression ratio as new data is added
5. **Batch efficiency** -- batch mode achieves ~17% latency improvement per item vs single inject

### Issues discovered

1. **CDX API connectivity** -- Direct Common Crawl CDX API returns `IncompleteMessage` after 3 retries. Internet Archive CDX works as fallback.
2. **Batch schema** -- Batch inject requires `source` on each item (not documented at top level only). Returns 422 without it.

### Next steps

1. Investigate CDX API connection issue (may be transient or require different endpoint)
2. Run Phase 2: actual Common Crawl data extraction using IA CDX fallback
3. Test with larger batch sizes (50-100 items) to measure throughput ceiling
4. Trigger LoRA training cycle after sufficient new data ingestion
5. Monitor sparsifier compression ratio as memory count approaches 5K+
