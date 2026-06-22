# MetaHarness Integration for RuVector: Comprehensive Benchmark Suite Implementation Plan

**Author**: Claude Code MetaHarness Architect  
**Date**: 2026-06-21  
**Phase**: Phase 1 MVP (2026-06-21 to 2026-08-30)  
**Status**: In Development  

---

## Executive Summary

This document outlines the 5-phase implementation plan to integrate MetaHarness with RuVector's benchmark suite, enabling autonomous parameter optimization via Darwin Mode evolution against public leaderboard scores (ANN-Benchmarks, BEIR, VectorDBBench, MTEB).

**Key outcomes**:
- Phase 1: ANN-Benchmarks compatibility layer + single-dataset harness (4 weeks)
- Phase 2: Parameter sweep framework (3 weeks)
- Phase 3: BEIR + VectorDBBench integration (4 weeks)
- Phase 4: Darwin Mode evolution loop (3 weeks)
- Phase 5: MTEB embedding quality validation (2 weeks)

**Total**: 16 weeks, 8 concurrent agents, ~12K LOC across TypeScript + Rust.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Public Leaderboards (ANN-Benchmarks, BEIR, MTEB)       │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│  MetaHarness Darwin Mode Integration Layer              │
│  (scorePolicy.ts, mutationSurfaces.ts, configSchema.ts) │
└────┬──────────┬──────────────┬────────────┬─────────────┘
     │          │              │            │
┌────▼──┐  ┌────▼──────┐  ┌───▼──────┐  ┌──▼───────┐
│Phase 1│  │ Phase 2    │  │ Phase 3  │  │ Phase 4  │
│ HDF5  │  │ Parameter  │  │ BEIR +   │  │ Darwin   │
│Loader │  │ Sweep      │  │ VDBBench │  │ Mode     │
│SIFT/  │  │ (Grid+     │  │          │  │Evolution │
│GIST   │  │ Random)    │  │          │  │Loop      │
└────┬──┘  └────┬───────┘  └───┬──────┘  └──┬───────┘
     │          │              │            │
     └──────────┴──────────────┴────────────┘
            │
     ┌──────▼──────────────┐
     │  RuVector Core      │
     │  ─────────────────  │
     │  HNSW, RaBitQ,      │
     │  Matryoshka, PQ,    │
     │  Hybrid, LSM-ANN,   │
     │  ColBERT, DiskANN   │
     └─────────────────────┘
```

---

## Phase 1: ANN-Benchmarks Compatibility Layer (4 weeks)

**Goal**: Load SIFT1M, GIST1M, GloVe datasets; measure recall@10, QPS; single dataset benchmark harness.

### Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/benchmark/ann-datasets.ts` | 400 | HDF5 loader, dataset registry |
| `scripts/benchmark/single-dataset-harness.ts` | 600 | SIFT/GIST test runner, metric aggregation |
| `scripts/benchmark/baseline-configs.json` | 200 | RuVector module defaults (HNSW M=12, efConstruction=200, etc.) |
| `scripts/benchmark/result-formatter.ts` | 300 | CSV + JSON output, comparison tables |
| `.github/workflows/benchmark-smoke.yml` | 100 | Daily CI job (SIFT1M subset, 3 configs) |
| `crates/ruvector-bench/src/hdf5_loader.rs` | 350 | Rust HDF5 bindings (via hdf5 crate) |
| `docs/validation/smoke-baseline-2026-06.json` | 150 | Golden baseline for regression detection |

**Key APIs**:

```typescript
// ann-datasets.ts
interface Dataset {
  name: string;           // "sift1m", "gist1m", "glove-angular"
  dimension: number;      // 128, 960, 100
  train_size: number;     // 100k-1M
  test_size: number;      // 10k
  hdf5_url: string;       // download URL
  download_cache_dir: string;
}

async function loadDataset(ds: Dataset): Promise<{
  train: Float32Array[];
  test: Float32Array[];
  groundtruth: number[][];  // [test_size][100] nearest neighbor IDs
}>;

// single-dataset-harness.ts
interface BenchmarkConfig {
  module: string;          // "hnsw", "rabitq", "matryoshka"
  params: Record<string, any>;
  dataset: Dataset;
}

async function runBenchmark(config: BenchmarkConfig): Promise<{
  recall_at_k: number[];   // [1, 10, 100]
  qps: number;
  latency_p50_ms: number;
  latency_p99_ms: number;
  memory_mb: number;
  build_time_sec: number;
}>;
```

**CI Gate** (`.github/workflows/benchmark-smoke.yml`):
```yaml
- name: Smoke Benchmark
  run: |
    npm run benchmark:sift1m:smoke
    # Pass if recall@10 >= baseline * 0.98 (allow 2% regression)
    node scripts/benchmark/check-regression.js \
      --baseline docs/validation/smoke-baseline-2026-06.json \
      --tolerance 0.02
```

**Success Criteria**:
- Load SIFT1M in <30s
- Run 3 configs in <5min per config
- CSV output matches manual Python benchmark ±1%
- 0 regression on main branch

---

## Phase 2: Parameter Sweep Framework (3 weeks)

**Goal**: Grid + random search over index config space; identify Pareto frontier (recall vs QPS vs memory).

### Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/benchmark/sweep-config.json` | 150 | Grid definition (HNSW M∈[4,8,12,16,20,32], efConstruction∈[50,100,200,400]) |
| `scripts/benchmark/sweep-harness.ts` | 800 | Grid/random exploration, Pareto ranking |
| `scripts/benchmark/pareto-visualizer.ts` | 400 | 2D plots (recall vs QPS, memory vs latency) |
| `crates/ruvector-bench/src/grid_search.rs` | 500 | Parallel config evaluation (rayon) |
| `docs/benchmark-results/phase2-pareto-frontier.json` | 300 | Pareto archive per module |

**Sweep Grid**:

```json
{
  "sweep_spaces": {
    "hnsw": {
      "M": [4, 8, 12, 16, 20, 32],
      "efConstruction": [50, 100, 200, 400],
      "efSearch": [50, 100, 200]
    },
    "rabitq": {
      "bits": [1],
      "rotation": [true],
      "normalize": [true, false]
    },
    "matryoshka": {
      "full_dim": [768],
      "search_dims": [[64], [128, 256], [128, 256, 512]]
    },
    "pq": {
      "M": [8, 16, 32],
      "nbits": [4, 8]
    },
    "hybrid": {
      "sparse_weight": [0.2, 0.5, 0.8],
      "fusion_strategy": ["rrf", "linear", "dbsf"]
    }
  },
  "dataset": "sift1m",
  "sample_strategy": "grid",  // "grid" | "random" | "latin_hypercube"
  "sample_count": 50
}
```

**Key API**:

```typescript
// sweep-harness.ts
interface ParetoPoint {
  config: BenchmarkConfig;
  recall_at_10: number;
  qps: number;
  memory_mb: number;
  p99_ms: number;
  timestamp: string;
}

async function sweepConfigs(
  space: SweepSpace,
  dataset: Dataset,
  maxParallel?: number
): Promise<ParetoPoint[]>;

function rankPareto(points: ParetoPoint[]): {
  dominating: ParetoPoint[];      // non-dominated set
  dominated: ParetoPoint[];
  hypervolume: number;             // Pareto hypervolume
};
```

**Pareto Visualization**:
```html
<!-- pareto-frontier.html -->
<svg width="800" height="600">
  <!-- Scatter: X=recall@10, Y=QPS, bubble-size=memory -->
  <!-- Pareto frontier: red line connecting dominating points -->
  <!-- Hover: show config JSON -->
</svg>
```

**Success Criteria**:
- Identify 10-15 non-dominated configs per module
- Sweep completes in <2 hours (8 cores)
- Pareto frontier visually separates memory-optimized vs latency-optimized

---

## Phase 3: BEIR + VectorDBBench Integration (4 weeks)

**Goal**: Add retrieval benchmarks (11 BEIR datasets, VectorDBBench workloads); measure NDCG, MRR, MAP.

### Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/benchmark/beir-loader.ts` | 500 | BEIR dataset fetcher + corpus indexing |
| `scripts/benchmark/retrieval-harness.ts` | 700 | NDCG@10, MRR, MAP computation |
| `scripts/benchmark/vdb-bench-workloads.ts` | 400 | Insert rate, query latency, memory under workload |
| `crates/ruvector-bench/src/retrieval.rs` | 600 | Batch retrieval, recall@k histogram |
| `docs/benchmark-results/beir-baseline.json` | 250 | BEIR baselines (DPR, GTR, E5) |

**BEIR Datasets**:
```json
{
  "beir_datasets": [
    "trec-covid",      // 169K docs, 50 queries
    "nfcorpus",        // 323K docs, 323 queries
    "nq",              // 3.2M docs, 3.45K queries
    "scifact",         // 5.2K docs, 300 queries
    "trec-news",       // 595K docs, 60 queries
    "dbpedia",         // 4.6M docs, 400 queries
    "trec-web",        // 3.1M docs, 50 queries
    "fever",           // 5.4M docs, 6.8K queries
    "climate-fever",   // 5.4M docs, 1535 queries
    "arguana",         // 8.8K docs, 1406 queries
    "webis-touche2020" // 382K docs, 49 queries
  ],
  "metrics": ["ndcg@10", "mrr", "map", "recall@100"]
}
```

**Key API**:

```typescript
// beir-loader.ts
interface BEIRDataset {
  name: string;
  corpus: Document[];        // {id, text, metadata}
  queries: Query[];           // {id, text}
  qrels: Map<string, Map<string, number>>;  // {query_id -> {doc_id -> relevance}}
}

async function loadBEIRDataset(name: string): Promise<BEIRDataset>;

// retrieval-harness.ts
interface RetrievalMetrics {
  ndcg_at_k: number[];       // [10, 100, 1000]
  mrr: number;
  map: number;
  recall_at_k: number[];
  query_time_ms: number;
}

async function evaluateRetrieval(
  index: VectorIndex,
  dataset: BEIRDataset,
  k: number = 100
): Promise<RetrievalMetrics>;
```

**VectorDBBench Workloads**:
```json
{
  "workloads": [
    {
      "name": "insert-heavy",
      "insert_rate": 10000,      // docs/sec
      "query_rate": 1000,
      "duration_sec": 60,
      "k": 10
    },
    {
      "name": "query-heavy",
      "insert_rate": 100,
      "query_rate": 5000,
      "duration_sec": 60,
      "k": 100
    }
  ]
}
```

**Success Criteria**:
- BEIR indexing: 5M docs in <5 min
- NDCG@10 ≥ 0.45 on nq dataset (vs DPR baseline 0.49)
- VectorDBBench: sustain 5K QPS for 60 sec without OOM

---

## Phase 4: Darwin Mode Evolution Loop (3 weeks)

**Goal**: MetaHarness Darwin Mode autonomously evolves index configs to maximize composite score.

### Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/benchmark/darwin-score-policy.ts` | 300 | Score function composition |
| `scripts/benchmark/mutation-surfaces.ts` | 400 | Mutation definitions for all modules |
| `scripts/benchmark/darwin-harness.ts` | 600 | Main evolution loop, checkpoint strategy |
| `.github/workflows/darwin-evolution.yml` | 120 | Weekly evolution run |
| `docs/darwin/evolution-runs/` | per-run | Archive of all runs + winning configs |

**Score Function** (`darwin-score-policy.ts`):

```typescript
interface ScoringPolicy {
  baseline: {
    recall_at_10: number;    // 0.85
    qps: number;             // 50000
    memory_mb: number;        // 256
    latency_p99_ms: number;  // 5.0
  };
  weights: {
    recall: 0.4;
    qps: 0.3;
    memory: 0.2;
    latency: 0.1;
  };
}

function computeScore(metrics: BenchmarkMetrics, policy: ScoringPolicy): number {
  const recall_norm = metrics.recall_at_10 / policy.baseline.recall_at_10;
  const qps_norm = Math.log(metrics.qps / policy.baseline.qps);
  const mem_norm = 1 - (metrics.memory_mb / policy.baseline.memory_mb);
  const lat_norm = 1 - (metrics.latency_p99_ms / policy.baseline.latency_p99_ms);
  
  return (
    policy.weights.recall * recall_norm +
    policy.weights.qps * Math.max(0, qps_norm) +  // penalize slowdown
    policy.weights.memory * Math.max(0, mem_norm) +
    policy.weights.latency * Math.max(0, lat_norm)
  );
}
```

**Mutation Surfaces** (`mutation-surfaces.ts`):

```typescript
type MutationSurface = {
  module: string;
  param: string;
  type: "int" | "float" | "enum" | "boolean";
  range?: [number, number];
  options?: string[];
  mutation_ops: {
    add?: (v: any) => any;
    multiply?: (v: any) => any;
    swap?: (options: string[]) => string;
  };
};

const MUTATION_SURFACES: MutationSurface[] = [
  {
    module: "hnsw",
    param: "M",
    type: "int",
    range: [4, 32],
    mutation_ops: {
      add: (v) => Math.min(v + 2, 32),
      multiply: (v) => Math.max(Math.floor(v * 0.8), 4)
    }
  },
  {
    module: "hnsw",
    param: "efConstruction",
    type: "int",
    range: [50, 400],
    mutation_ops: {
      add: (v) => Math.min(v + 50, 400),
      multiply: (v) => Math.max(Math.floor(v * 1.2), 50)
    }
  },
  {
    module: "rabitq",
    param: "normalize",
    type: "boolean"
  },
  {
    module: "matryoshka",
    param: "search_dims",
    type: "enum",
    options: ["[64]", "[128]", "[256]", "[64,128]", "[128,256]", "[256,512]"]
  },
  // ... 15+ more surfaces across all modules
];
```

**Darwin Loop** (`darwin-harness.ts`):

```typescript
async function runDarwinEvolution(options: {
  dataset: Dataset;
  max_generations: number;
  population_size: number;
  mutation_rate: number;
  elite_fraction: number;
}): Promise<{
  generation: number;
  best_config: BenchmarkConfig;
  best_score: number;
  population: Array<{config, score}>;
  checkpoint: string;
}[]> {
  // 1. Initialize: Pareto frontier from Phase 2 + random mutations
  let population = [...phasePareto, ...randomMutations(options.population_size)];
  
  // 2. For each generation:
  for (let g = 0; g < options.max_generations; g++) {
    // a. Evaluate all configs
    const evaluated = await Promise.all(
      population.map(cfg => benchmarkAndScore(cfg))
    );
    
    // b. Rank by score, keep elite
    const sorted = evaluated.sort((a, b) => b.score - a.score);
    const elite = sorted.slice(0, Math.ceil(options.elite_fraction * population.size));
    
    // c. Mutate elite to create next generation
    const mutated = elite.flatMap(e =>
      Array(options.population_size / elite.length).fill(null).map(() =>
        mutateConfig(e.config, MUTATION_SURFACES)
      )
    );
    
    population = [...elite.map(e => e.config), ...mutated];
    
    // d. Checkpoint best config
    const best = sorted[0];
    console.log(`[G${g}] best_score=${best.score.toFixed(3)}, best_config=${JSON.stringify(best.config)}`);
    
    yield {
      generation: g,
      best_config: best.config,
      best_score: best.score,
      population: sorted.slice(0, 10),
      checkpoint: `generation-${g}.json`
    };
  }
}
```

**ADR-150 Compliance** (graceful degradation):

```typescript
// darwin-harness.ts
async function initDarwinMode(): Promise<void> {
  try {
    const Darwin = await import("@metaharness/darwin");
    log.info("MetaHarness Darwin Mode loaded");
    return Darwin;
  } catch (e) {
    if (e.code === "MODULE_NOT_FOUND") {
      log.warn("@metaharness/darwin not installed; skipping evolution");
      log.warn("Install via: npm install --optional @metaharness/darwin");
      return null;
    }
    throw e;
  }
}

async function runBenchmark(...) {
  const darwin = await initDarwinMode();
  if (!darwin) {
    // Fallback: run phase 2 grid search instead
    return sweepConfigs(...);
  }
  // Run Darwin evolution
  return runDarwinEvolution(...);
}
```

**Success Criteria**:
- Evolve to a config that beats baseline on 3 of 4 metrics
- Checkpoint every generation (JSON archive)
- Zero crashes on missing MetaHarness (graceful degradation)

---

## Phase 5: MTEB Embedding Quality Validation (2 weeks)

**Goal**: Validate embedding quality on MTEB benchmark (170K sentences, 15 retrieval tasks).

### Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/benchmark/mteb-loader.ts` | 300 | MTEB dataset fetcher |
| `scripts/benchmark/mteb-harness.ts` | 400 | STS evaluation, clustering scoring |
| `scripts/benchmark/embedding-quality.ts` | 350 | Vector similarity analysis |
| `docs/benchmark-results/mteb-baseline.json` | 150 | Baseline scores |

**MTEB Datasets**:
- Retrieval (15 datasets): trec-covid, scifact, nfcorpus, nq, ...
- STS (semantic textual similarity): 8 datasets
- Clustering: 11 datasets
- Reranking: 4 datasets

**Success Criteria**:
- All-MiniLM-L6-v2 on nq: NDCG@10 ≥ 0.45
- E5-large-v2 on nq: NDCG@10 ≥ 0.50
- Complete in <10 hours

---

## File Structure & Paths

```
ruvector/
├── scripts/benchmark/
│   ├── ann-datasets.ts                    (Phase 1, 400 lines)
│   ├── single-dataset-harness.ts          (Phase 1, 600 lines)
│   ├── baseline-configs.json              (Phase 1, 200 lines)
│   ├── result-formatter.ts                (Phase 1, 300 lines)
│   ├── check-regression.js                (Phase 1, 150 lines)
│   │
│   ├── sweep-config.json                  (Phase 2, 150 lines)
│   ├── sweep-harness.ts                   (Phase 2, 800 lines)
│   ├── pareto-visualizer.ts               (Phase 2, 400 lines)
│   │
│   ├── beir-loader.ts                     (Phase 3, 500 lines)
│   ├── retrieval-harness.ts               (Phase 3, 700 lines)
│   ├── vdb-bench-workloads.ts             (Phase 3, 400 lines)
│   │
│   ├── darwin-score-policy.ts             (Phase 4, 300 lines)
│   ├── mutation-surfaces.ts               (Phase 4, 400 lines)
│   ├── darwin-harness.ts                  (Phase 4, 600 lines)
│   │
│   ├── mteb-loader.ts                     (Phase 5, 300 lines)
│   ├── mteb-harness.ts                    (Phase 5, 400 lines)
│   ├── embedding-quality.ts               (Phase 5, 350 lines)
│   │
│   └── index.ts                           (master export, 50 lines)
│
├── crates/ruvector-bench/
│   ├── Cargo.toml
│   └── src/
│       ├── hdf5_loader.rs                 (Phase 1, 350 lines)
│       ├── grid_search.rs                 (Phase 2, 500 lines)
│       ├── retrieval.rs                   (Phase 3, 600 lines)
│       └── lib.rs
│
├── .github/workflows/
│   ├── benchmark-smoke.yml                (Phase 1, 100 lines)
│   ├── benchmark-sweep.yml                (Phase 2, 120 lines)
│   ├── benchmark-beir.yml                 (Phase 3, 140 lines)
│   └── darwin-evolution.yml               (Phase 4, 120 lines)
│
├── docs/validation/
│   ├── smoke-baseline-2026-06.json
│   └── manifests/
│       ├── 2026-06-21-sift1m.json
│       ├── 2026-06-21-beir-baseline.json
│       └── ...
│
├── docs/darwin/
│   ├── evolution-runs/
│   │   ├── 2026-07-10-run-1.json
│   │   ├── 2026-07-17-run-2.json
│   │   └── ...
│   └── best-configs-archive.json
│
└── docs/benchmark-results/
    ├── phase2-pareto-frontier.json
    ├── beir-baseline.json
    ├── mteb-baseline.json
    └── leaderboard-summary.html
```

---

## CI/CD Integration

### Daily Smoke Test
**File**: `.github/workflows/benchmark-smoke.yml`

```yaml
name: Benchmark Smoke Test
on:
  schedule:
    - cron: "0 6 * * *"  # 6 AM UTC daily
  workflow_dispatch:

jobs:
  smoke:
    runs-on: ubuntu-latest-16core
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
      
      - name: Install dependencies
        run: npm install
      
      - name: Download SIFT1M subset (100K)
        run: |
          curl -L https://ann-benchmarks.com/sift1m.hdf5 | head -c 100MB > sift1m-subset.hdf5
      
      - name: Run smoke benchmark (HNSW only)
        run: |
          npx ts-node scripts/benchmark/single-dataset-harness.ts \
            --dataset sift1m-subset \
            --modules hnsw,rabitq \
            --config baseline-configs.json \
            --output smoke-results.json
        timeout-minutes: 10
      
      - name: Check regression
        run: |
          node scripts/benchmark/check-regression.js \
            --baseline docs/validation/smoke-baseline-2026-06.json \
            --current smoke-results.json \
            --tolerance 0.02
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: smoke-results-${{ github.run_id }}
          path: smoke-results.json
      
      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const results = require('./smoke-results.json');
            const comment = `## Benchmark Smoke Test
            
            **SIFT1M (100K subset)**
            - HNSW: recall@10=${results.hnsw.recall_at_10.toFixed(3)}, QPS=${results.hnsw.qps.toFixed(0)}
            - RaBitQ: recall@10=${results.rabitq.recall_at_10.toFixed(3)}, QPS=${results.rabitq.qps.toFixed(0)}
            
            [Full results](https://github.com/ruvnet/ruvector/actions/runs/${{ github.run_id }})`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
```

### Weekly Parameter Sweep
**File**: `.github/workflows/benchmark-sweep.yml` (runs Phase 2)

```yaml
name: Weekly Parameter Sweep
on:
  schedule:
    - cron: "0 20 * * 0"  # Sunday 8 PM UTC
  workflow_dispatch:

jobs:
  sweep:
    runs-on: ubuntu-latest-32core
    timeout-minutes: 240  # 4 hours
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Rust
        uses: dtolnay/rust-toolchain@stable
      
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
      
      - name: Download full datasets
        run: |
          # Download to local cache, skip if cached
          npm run benchmark:download-datasets
      
      - name: Run sweep
        run: |
          npx ts-node scripts/benchmark/sweep-harness.ts \
            --config sweep-config.json \
            --parallel 8 \
            --output pareto-frontier-${{ github.run_id }}.json
      
      - name: Generate Pareto visualizations
        run: |
          npx ts-node scripts/benchmark/pareto-visualizer.ts \
            --input pareto-frontier-${{ github.run_id }}.json \
            --output pareto-frontier-${{ github.run_id }}.html
      
      - name: Commit results
        run: |
          git config user.email "bench@ruvector.local"
          git config user.name "Benchmark Bot"
          mv pareto-frontier-${{ github.run_id }}.json docs/benchmark-results/
          mv pareto-frontier-${{ github.run_id }}.html docs/benchmark-results/
          git add docs/benchmark-results/
          git commit -m "chore(bench): weekly parameter sweep $(date -u +%Y-%m-%d)"
          git push origin main
        if: always()
```

### BEIR & VectorDBBench (Phase 3)
**File**: `.github/workflows/benchmark-beir.yml`

```yaml
name: BEIR & VectorDBBench Benchmark
on:
  workflow_dispatch:
  schedule:
    - cron: "0 0 * * 1"  # Monday midnight UTC

jobs:
  beir:
    runs-on: ubuntu-latest-32core
    timeout-minutes: 480  # 8 hours
    steps:
      - uses: actions/checkout@v4
      
      - name: Download BEIR datasets
        run: npm run benchmark:download-beir
        timeout-minutes: 60
      
      - name: Run retrieval benchmark
        run: |
          npx ts-node scripts/benchmark/retrieval-harness.ts \
            --datasets nq,trec-covid,scifact \
            --modules hnsw,matryoshka,hybrid \
            --output beir-results-${{ github.run_id }}.json
      
      - name: Run VectorDBBench workloads
        run: |
          npx ts-node scripts/benchmark/vdb-bench-workloads.ts \
            --dataset nq \
            --config [insert-heavy,query-heavy] \
            --output vdb-results-${{ github.run_id }}.json
      
      - name: Store results
        run: |
          mkdir -p docs/validation/manifests
          mv beir-results-${{ github.run_id }}.json \
             docs/validation/manifests/beir-$(date -u +%Y-%m-%d).json
          mv vdb-results-${{ github.run_id }}.json \
             docs/validation/manifests/vdb-$(date -u +%Y-%m-%d).json
      
      - name: Create witness signature
        run: |
          npx ts-node scripts/benchmark/witness-signer.ts \
            --manifest docs/validation/manifests/beir-$(date -u +%Y-%m-%d).json \
            --sign-with /home/ruvultra/.ssh/id_ed25519
      
      - name: Commit & push
        run: |
          git config user.email "bench@ruvector.local"
          git config user.name "Benchmark Bot"
          git add docs/validation/manifests/
          git commit -m "chore(validation): beir+vdb benchmark $(date -u +%Y-%m-%d)"
          git push origin main
```

### Darwin Evolution (Phase 4)
**File**: `.github/workflows/darwin-evolution.yml`

```yaml
name: Darwin Mode Evolution
on:
  workflow_dispatch:
  schedule:
    - cron: "0 12 * * 3"  # Wednesday noon UTC (weekly)

jobs:
  darwin:
    runs-on: ubuntu-latest-32core
    timeout-minutes: 360  # 6 hours
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
      
      - name: Install MetaHarness Darwin
        run: |
          npm install --optional @metaharness/darwin
        continue-on-error: true  # OK if missing (ADR-150)
      
      - name: Run Darwin evolution
        run: |
          npx ts-node scripts/benchmark/darwin-harness.ts \
            --dataset sift1m \
            --generations 10 \
            --population-size 20 \
            --output darwin-run-${{ github.run_id }}.json
      
      - name: Extract best config
        run: |
          node -e "
            const run = require('./darwin-run-${{ github.run_id }}.json');
            const best = run.reduce((a,b) => a.best_score > b.best_score ? a : b);
            console.log('Best config (generation', best.generation + ')');
            console.log(JSON.stringify(best.best_config, null, 2));
            console.log('Score:', best.best_score.toFixed(4));
          "
      
      - name: Commit evolution history
        run: |
          mkdir -p docs/darwin/evolution-runs
          mv darwin-run-${{ github.run_id }}.json \
             docs/darwin/evolution-runs/$(date -u +%Y-%m-%d)-run-${{ github.run_number }}.json
          git add docs/darwin/evolution-runs/
          git commit -m "chore(darwin): evolution run $(date -u +%Y-%m-%d)"
          git push origin main
        if: success()
```

---

## Metrics & Success Gates

### Phase 1 Gate
- [ ] SIFT1M loads in <30s
- [ ] Single benchmark run takes <5 min per config
- [ ] CSV output within ±1% of manual Python baseline
- [ ] Smoke test passes daily with <2% regression tolerance

### Phase 2 Gate
- [ ] Grid sweep completes in <2 hours (8 cores)
- [ ] Identify 10-15 non-dominated Pareto configs
- [ ] Pareto frontier is visually correct (no crossing)
- [ ] Top 3 configs beat baseline on at least 2 metrics

### Phase 3 Gate
- [ ] BEIR indexing: 5M docs in <5 min per dataset
- [ ] NDCG@10 on NQ ≥ 0.45 (DPR baseline is 0.49)
- [ ] VectorDBBench: sustain 5K QPS for 60 sec without OOM
- [ ] All 11 BEIR datasets complete without timeout

### Phase 4 Gate
- [ ] Darwin evolution produces a config beating baseline on 3+ metrics
- [ ] Graceful degradation: if @metaharness/darwin missing, falls back to Phase 2
- [ ] 100% of evolution runs checkpointed to JSON
- [ ] Zero crashes on platform (macOS, Linux, Windows)

### Phase 5 Gate
- [ ] MTEB evaluation completes in <10 hours
- [ ] All-MiniLM-L6-v2 achieves ≥0.45 NDCG@10 on NQ
- [ ] E5-large-v2 achieves ≥0.50 NDCG@10 on NQ

---

## Effort Estimate

| Phase | Team | Weeks | Key Files | Risks |
|-------|------|-------|-----------|-------|
| **1** | 2 engineers | 4 | 7 TypeScript, 1 Rust | HDF5 library compatibility |
| **2** | 1 engineer | 3 | 3 TypeScript, 1 Rust | Grid explosion (need pruning) |
| **3** | 2 engineers | 4 | 5 TypeScript, 1 Rust | BEIR dataset size (26M docs total) |
| **4** | 1 engineer | 3 | 3 TypeScript | @metaharness/darwin API stability |
| **5** | 1 engineer | 2 | 3 TypeScript | MTEB evaluation infrastructure |
| **Total** | **8** | **16** | **21 TypeScript, 3 Rust** | **Dependency on MetaHarness** |

---

## Dependencies & Risks

### External Dependencies
- `hdf5` crate (Rust) — used for Phase 1 ANN-Benchmarks loading
- `@metaharness/darwin` (npm) — optional, Phase 4 only (ADR-150 compliance)
- BEIR corpus — 26M docs, ~200GB compressed (Phase 3)
- MTEB datasets — 170K sentences (Phase 5)

### Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| HDF5 library not available on CI | Medium | High | Ship pre-built binaries, fallback to Python subprocess |
| BEIR dataset download timeout | Medium | High | Cache in GCS, use CDN mirror |
| MetaHarness Darwin unstable | Low | High | Vendorize snapshot, version-pin with fallback |
| Parameter sweep explodes (>1000 configs) | Medium | Medium | Implement early pruning, random sampling instead of grid |
| CI job timeout on large runs | Medium | Medium | Increase timeout, split into multiple jobs |

---

## Rollout Timeline

```
2026-06-21 — Phase 1 kickoff (ANN-Benchmarks loader + smoke test)
2026-07-19 — Phase 1 complete, Phase 2 starts (grid sweep)
2026-08-09 — Phase 2 complete, Phase 3 starts (BEIR integration)
2026-09-06 — Phase 3 complete, Phase 4 starts (Darwin evolution)
2026-09-27 — Phase 4 complete, Phase 5 starts (MTEB validation)
2026-10-11 — Phase 5 complete, MVP launch
```

---

## Success Metrics (Post-MVP)

1. **Reproducibility**: All benchmark runs generate signed witness manifests (ADR-267)
2. **Autonomy**: Darwin Mode evolves at least 1 config/week that beats baseline
3. **Publication**: Submit SOTA results to ANN-Benchmarks, VectorDBBench leaderboards
4. **Adoption**: RuVector users run benchmarks via `npm run benchmark:all`
5. **SOTA Claims**: Claim SOTA in 3+ categories (recall@10, memory efficiency, latency)

---

## Appendix: ADR-150 Compliance Checklist

- [ ] All @metaharness/* packages in `optionalDependencies` only
- [ ] Darwin Mode imports wrapped in try-catch MODULE_NOT_FOUND
- [ ] Fallback to Phase 2 grid search if Darwin unavailable
- [ ] README includes installation: `npm install --optional @metaharness/darwin`
- [ ] CI smoke test runs without MetaHarness installed
- [ ] No hard dependency on @metaharness/* in main code paths

