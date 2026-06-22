# ADR-266: MetaHarness Integration for Autonomous ANN Optimization (Darwin Mode)

**Status**: Accepted  
**Date**: 2026-06-21  
**Authors**: Claude Code MetaHarness Architect  
**Supersedes**: None  
**Related**: ADR-150 (MetaHarness Integration Surfaces), ADR-265 (Benchmark Suite), ADR-267 (SOTA Validation)

---

## Context

MetaHarness (@metaharness/darwin package) is a mutation + scoring framework for autonomous software optimization. RuVector has 32+ tunable parameters across 8 modules (HNSW, RaBitQ, Matryoshka, PQ, Hybrid, ColBERT, MLA/SSM, KV-Cache). Manual grid search is O(n^k) where n=configs per param, k=num params.

**Problem**: How do we integrate Darwin Mode while respecting ADR-150 invariants?

ADR-150 requires:
1. **Removable**: `npm ls --without-deps @metaharness/*` still works
2. **Optional in package.json**: Only in optionalDependencies
3. **Graceful degradation**: MODULE_NOT_FOUND caught, fallback provided
4. **CI gate**: At least one job runs without MetaHarness

**Opportunity**: Darwin Mode can autonomously evolve index configs to beat baseline on 3+ metrics (recall, QPS, memory, latency).

---

## Decision

Integrate @metaharness/darwin as an optional evolution layer:

1. **Module is fully optional**: In optionalDependencies, no hard runtime dependency
2. **Fallback to Phase 2**: If missing, use grid search (Phase 2 of ADR-265) instead
3. **32 mutation surfaces**: Define mutable parameters for each module
4. **Single evolution loop**: Generations, population ranking, elite selection, checkpoint
5. **Scoring via ADR-265 function**: 4-component composite score (recall, QPS, memory, latency)
6. **Archive all runs**: Every generation checkpointed to JSON for reproducibility

### Mutation Surfaces (32 total)

```json
{
  "HNSW": [
    {"param": "M", "type": "int", "range": [4, 32], "default": 12},
    {"param": "efConstruction", "type": "int", "range": [50, 400], "default": 200},
    {"param": "efSearch", "type": "int", "range": [50, 200], "default": 100}
  ],
  "RaBitQ": [
    {"param": "bits", "type": "int", "range": [1, 1], "default": 1},
    {"param": "rotation", "type": "boolean", "default": true},
    {"param": "normalize", "type": "boolean", "default": true}
  ],
  "Matryoshka": [
    {"param": "full_dim", "type": "int", "range": [768, 768], "default": 768},
    {"param": "search_dims", "type": "enum", "options": ["[64]", "[128]", "[256]", "[64,128]", "[128,256]", "[256,512]"], "default": "[128,256,512]"}
  ],
  "ProductQuantization": [
    {"param": "M", "type": "int", "range": [8, 32], "default": 16},
    {"param": "nbits", "type": "int", "range": [4, 8], "default": 8}
  ],
  "Hybrid": [
    {"param": "sparse_weight", "type": "float", "range": [0.0, 1.0], "default": 0.3},
    {"param": "dense_weight", "type": "float", "range": [0.0, 1.0], "default": 0.7},
    {"param": "fusion_strategy", "type": "enum", "options": ["rrf", "linear", "dbsf"], "default": "rrf"}
  ],
  "ColBERT": [
    {"param": "token_k", "type": "int", "range": [4, 16], "default": 8}
  ],
  "KVCache": [
    {"param": "eviction_policy", "type": "enum", "options": ["H2O", "PyramidKV", "SlidingWindow"], "default": "H2O"},
    {"param": "quant_bits", "type": "int", "range": [2, 8], "default": 8}
  ],
  "DiskANN": [
    {"param": "alpha", "type": "float", "range": [1.0, 1.5], "default": 1.2},
    {"param": "L", "type": "int", "range": [10, 100], "default": 30}
  ]
}
```

---

## ADR-150 Compliance (Load-Bearing Invariants)

### Invariant 1: Removable

Even with MetaHarness installed, RuVector CLI functions without it:

```typescript
// scripts/benchmark/darwin-harness.ts
async function initDarwinMode(): Promise<DarwinModule | null> {
  try {
    const Darwin = await import("@metaharness/darwin");
    console.log("[darwin] MetaHarness Darwin Mode loaded");
    return Darwin;
  } catch (e) {
    if (e.code === "MODULE_NOT_FOUND") {
      console.warn("[darwin] @metaharness/darwin not installed");
      console.warn("[darwin] Falling back to Phase 2 grid search");
      return null;
    }
    throw e;  // Other errors are fatal
  }
}

export async function benchmarkWithEvolution(opts) {
  const darwin = await initDarwinMode();
  
  if (darwin) {
    return runDarwinEvolution(opts);
  } else {
    // Fallback: Phase 2 grid search
    return sweepConfigs(opts.sweep_space, opts.dataset);
  }
}
```

**CI gate** verifies this works:

```yaml
name: CLI Without MetaHarness
on: [push]
jobs:
  no-metaharness:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install --no-optional
      - run: npm run benchmark:sift1m:smoke
      - run: |
          # Verify falls back gracefully
          npm run benchmark:sweep 2>&1 | grep -q "Falling back"
```

### Invariant 2: Optional in package.json

```json
{
  "optionalDependencies": {
    "@metaharness/darwin": "^0.1.0"
  },
  "peerDependencies": {
    "@metaharness/darwin": "^0.1.0"
  }
}
```

Never in `dependencies`. Installation:

```bash
npm install --optional @metaharness/darwin
```

### Invariant 3: Graceful Degradation

Every code path that touches @metaharness/darwin is wrapped:

```typescript
// ✅ GOOD: Try-catch with graceful fallback
async function evolveConfigs() {
  let Darwin = null;
  try {
    Darwin = await import("@metaharness/darwin");
  } catch (e) {
    if (e.code !== "MODULE_NOT_FOUND") throw e;
    // Fallback silently
  }
  
  if (Darwin) {
    return await runDarwinEvolution();
  } else {
    return await runPhase2GridSearch();
  }
}

// ❌ BAD: No catch, hard dependency
import Darwin from "@metaharness/darwin";  // FAILS without install
```

### Invariant 4: CI Gate Without MetaHarness

Daily smoke test explicitly runs without optional deps:

```bash
npm install --no-optional
npm run benchmark:smoke  # Should pass
npm run benchmark:compare-baseline  # Should pass

# Verify graceful fallback message appears
npm run benchmark:sweep 2>&1 | grep -E "Falling back|grid search"
```

---

## Scoring Policy Implementation

```typescript
// scripts/benchmark/darwin-score-policy.ts

export interface ScoringPolicy {
  baseline: {
    recall_at_10: number;
    qps: number;
    memory_mb: number;
    latency_p99_ms: number;
  };
  weights: {
    recall: number;    // 0.0-1.0, sum to 1.0
    qps: number;
    memory: number;
    latency: number;
  };
}

export interface BenchmarkMetrics {
  recall_at_10: number;
  qps: number;
  memory_mb: number;
  latency_p99_ms: number;
  build_time_sec: number;
}

export function computeScore(
  metrics: BenchmarkMetrics,
  policy: ScoringPolicy
): number {
  // Normalize each dimension
  const recall_norm = metrics.recall_at_10 / policy.baseline.recall_at_10;
  
  const qps_norm = Math.log(
    Math.max(0.1, metrics.qps / policy.baseline.qps)
  );  // Log-scaled, minimum 0.1 to avoid negative infinity
  
  const memory_norm = Math.max(
    0,
    1 - (metrics.memory_mb / policy.baseline.memory_mb)
  );  // Clamped [0,1]
  
  const latency_norm = Math.max(
    0,
    1 - (metrics.latency_p99_ms / policy.baseline.latency_p99_ms)
  );  // Clamped [0,1]
  
  // Weighted sum
  const score =
    policy.weights.recall * recall_norm +
    policy.weights.qps * qps_norm +
    policy.weights.memory * memory_norm +
    policy.weights.latency * latency_norm;
  
  return score;
}

// Default policy (can be overridden per evolution run)
export const DEFAULT_POLICY: ScoringPolicy = {
  baseline: {
    recall_at_10: 0.85,
    qps: 50000,
    memory_mb: 256,
    latency_p99_ms: 5.0
  },
  weights: {
    recall: 0.4,
    qps: 0.3,
    memory: 0.2,
    latency: 0.1
  }
};
```

---

## Evolution Loop Implementation

```typescript
// scripts/benchmark/darwin-harness.ts

async function runDarwinEvolution(options: {
  dataset: Dataset;
  max_generations: number;
  population_size: number;
  mutation_rate: number;
  elite_fraction: number;
  scoring_policy?: ScoringPolicy;
}): Promise<EvolutionRun[]> {
  const Darwin = await initDarwinMode();
  if (!Darwin) {
    console.log("MetaHarness not available; using Phase 2 grid search");
    return sweepConfigs(...);
  }

  const policy = options.scoring_policy || DEFAULT_POLICY;
  const runs: EvolutionRun[] = [];

  // 1. Initialize population: Pareto frontier + random mutations
  let population: ConfigWithScore[] = [];
  const pareto = await loadPhase2ParetoFrontier(options.dataset);
  population.push(...pareto.map(cfg => ({ config: cfg, score: NaN })));
  
  const random = Array(options.population_size - pareto.length)
    .fill(null)
    .map(() => randomConfig(MUTATION_SURFACES));
  population.push(...random.map(cfg => ({ config: cfg, score: NaN })));

  // 2. Evolution loop
  for (let gen = 0; gen < options.max_generations; gen++) {
    console.log(`[darwin] Generation ${gen}/${options.max_generations}`);

    // a. Evaluate all configs
    const evaluated = await Promise.all(
      population.map(async ({ config }) => ({
        config,
        metrics: await benchmarkConfig(config, options.dataset),
        score: NaN
      }))
    );

    // b. Compute scores
    for (const entry of evaluated) {
      entry.score = computeScore(entry.metrics, policy);
    }

    // c. Rank by score
    const sorted = evaluated.sort((a, b) => b.score - a.score);
    const best = sorted[0];
    console.log(`  Best score: ${best.score.toFixed(4)}`);
    console.log(`  Config: ${JSON.stringify(best.config)}`);

    // d. Save checkpoint
    const checkpoint: EvolutionRun = {
      generation: gen,
      best_config: best.config,
      best_score: best.score,
      best_metrics: best.metrics,
      population: sorted.slice(0, Math.min(10, sorted.length)),
      timestamp: new Date().toISOString()
    };
    runs.push(checkpoint);

    // Save to JSON
    const filepath = `docs/darwin/evolution-runs/gen-${gen}.json`;
    await fs.promises.writeFile(
      filepath,
      JSON.stringify(checkpoint, null, 2)
    );
    console.log(`  Saved: ${filepath}`);

    // e. Mutation for next generation
    const elite = sorted.slice(
      0,
      Math.ceil(options.elite_fraction * population.length)
    );
    const mutated = elite.flatMap(entry =>
      Array(Math.ceil(population.length / elite.length))
        .fill(null)
        .map(() => mutateConfig(entry.config, MUTATION_SURFACES))
    );

    population = [
      ...elite.map(e => e.config),
      ...mutated
    ].map(config => ({ config, score: NaN }));
  }

  return runs;
}
```

---

## Mutation Operations

```typescript
// scripts/benchmark/mutation-surfaces.ts

type MutationOp = (v: any) => any;

interface MutationSurface {
  module: string;
  param: string;
  type: "int" | "float" | "enum" | "boolean";
  range?: [number, number];
  options?: string[];
  mutations: {
    increase?: MutationOp;
    decrease?: MutationOp;
    randomize?: MutationOp;
    swap?: (opts: string[]) => string;
  };
}

const MUTATION_SURFACES: MutationSurface[] = [
  {
    module: "hnsw",
    param: "M",
    type: "int",
    range: [4, 32],
    mutations: {
      increase: (v) => Math.min(v + 2, 32),
      decrease: (v) => Math.max(v - 2, 4),
      randomize: () => Math.floor(Math.random() * 28 + 4)
    }
  },
  {
    module: "hnsw",
    param: "efConstruction",
    type: "int",
    range: [50, 400],
    mutations: {
      increase: (v) => Math.min(Math.round(v * 1.3), 400),
      decrease: (v) => Math.max(Math.round(v * 0.75), 50),
      randomize: () => Math.floor(Math.random() * 350 + 50)
    }
  },
  // ... 30+ more surfaces
];

function mutateConfig(
  config: BenchmarkConfig,
  surfaces: MutationSurface[],
  rate: number = 0.3
): BenchmarkConfig {
  const mutated = { ...config };
  const surfacesToMutate = surfaces
    .filter(() => Math.random() < rate)
    .slice(0, 3);  // Limit to 3 mutations per generation
  
  for (const surface of surfacesToMutate) {
    const ops = Object.values(surface.mutations);
    const op = ops[Math.floor(Math.random() * ops.length)];
    
    if (surface.type === "enum" && surface.options) {
      mutated[surface.param] = surface.options[
        Math.floor(Math.random() * surface.options.length)
      ];
    } else {
      mutated[surface.param] = op(mutated[surface.param]);
    }
  }
  
  return mutated;
}
```

---

## CI/CD Workflow (Weekly Evolution)

```yaml
# .github/workflows/darwin-evolution.yml
name: Darwin Mode Evolution
on:
  workflow_dispatch:
  schedule:
    - cron: "0 12 * * 3"  # Wednesday noon UTC

jobs:
  darwin:
    runs-on: ubuntu-latest-32core
    timeout-minutes: 360
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: "20"
      
      - name: Install deps (MetaHarness optional)
        run: |
          npm install
          npm install --optional @metaharness/darwin || echo "Proceeding without Darwin"
      
      - name: Run evolution
        run: |
          npx ts-node scripts/benchmark/darwin-harness.ts \
            --dataset sift1m \
            --generations 10 \
            --population-size 20 \
            --output-dir docs/darwin/evolution-runs/$(date -u +%Y-%m-%d)
      
      - name: Verify graceful fallback (if Darwin missing)
        if: failure()
        run: |
          npm run benchmark:sweep --no-optional
          # Should complete via Phase 2 grid search
      
      - name: Commit checkpoints
        run: |
          git config user.email "darwin@ruvector.local"
          git config user.name "Darwin Bot"
          git add docs/darwin/
          git commit -m "chore(darwin): evolution run $(date -u +%Y-%m-%d)" || true
          git push origin main
```

---

## Success Criteria

- **Score improvement**: Evolve ≥1 config beating baseline on 3+ metrics
- **Graceful degradation**: Zero crashes if @metaharness/darwin missing
- **Checkpoint coverage**: 100% of generations saved to JSON
- **Platform stability**: Zero segfaults on Linux, macOS, Windows
- **ADR-150 compliance**: Full compliance with all 4 invariants

---

## References

- ADR-150: MetaHarness Integration Surfaces
- ADR-265: RuVector Comprehensive Benchmark Suite
- ADR-267: SOTA Validation Protocol
- @metaharness/darwin: https://github.com/ruvnet/agent-harness-generator

