# ADR-260: Darwin Mode as Evolutionary Substrate for MetaHarness

**Status:** Accepted  
**Date:** 2026-06-18  
**Supersedes:** Extends ADR-259 (ruvllm local mutator — point integration)  
**References:**
- ADR-258 — ruvllm GPU optimization (RDT/OpenMythos, RTX 5080)
- ADR-259 — ruvllm as local CodeGenerator backend for Darwin Mode
- [darwin-mode ADR-074](https://github.com/ruvnet/agent-harness-generator/blob/main/docs/adrs/ADR-074-darwin-ruvector-memory-ruflo-fabric.md) — ruvvector-memory-ruflo-fabric (upstream design)
- [darwin-mode ADR-085](https://github.com/ruvnet/agent-harness-generator/blob/main/docs/adrs/ADR-085.md) — LLM mutator (OpenRouterMutator)
- [darwin-mode ADR-144/149](https://github.com/ruvnet/agent-harness-generator/blob/main/docs/adrs/ADR-144.md) — SWE-bench Lite baseline + repair loop

---

## Executive Summary

Darwin Mode is the right evolutionary substrate for MetaHarness — and it is
**already wired for ruvvector** (darwin-mode ADR-074 "ruvvector-memory-ruflo-fabric")
and **already running real-LLM-on-real-code** (ADR-106, sandboxMode: 'agent').
This ADR documents how ruvllm + ruvvector close the remaining gaps in the live system:
(1) replace OpenRouter inference with GPU-local inference, (2) activate the SWE-bench
repair loop via ruvllm's recurrent depth, and (3) realise ADR-074's ruvvector archive
design against the actual ruvvector Rust/NAPI API.

---

## What Darwin Mode Actually Does (Corrected Understanding)

```
repo
  → profile      RepoProfile
  → baseline     generate 7 mutation-surface files
  → mutate       pick ONE surface, call CodeGenerator (OpenRouterMutator | RuvllmMutator)
  → sandbox      safety-inspect → run test command (no shell, no net, no secrets)
     └─ sandboxMode: 'real'  — repo test suite (shipped ADR-070)
     └─ sandboxMode: 'mock'  — surface-driven deterministic agent (ADR-102)
     └─ sandboxMode: 'agent' — real surface code in child process (SHIPPED ADR-106)
  → score        6-term base − hard penalty layer (frozen kernel)
  → archive      TREE (not single best branch), sampled whole-archive for parents
  → repeat
```

**Three empirically validated facts from the README that change the integration picture:**

### 1. `sandboxMode: 'agent'` is shipped and already on SWE-bench

Darwin Mode already runs on **SWE-bench Lite (full 300 instances)** with:
- Official `swebench` Docker harness (no cherry-picking)
- `deepseek-chat` at ~$0.01/instance ($0.4/Mtok)
- **7.7% resolve rate [5.2–11.2% 95% CI]** (ADR-144, open-loop single-shot)
- Localization improved file-recall 44.7% → 59.7% but resolve rate held flat (ADR-146)
- **Active lever: closed-loop repair (ADR-149)** — test feedback driving iterative patch refinement

This is the integration point that matters most for ruvllm.

### 2. The repair loop (ADR-149) is recurrent inference by another name

ADR-149's "closed-loop repair" structure:
```
patch → test → FAIL → feed back failure traces → re-patch → repeat
```

This is structurally identical to RDT's ACT loop:
```
hidden_state → halt_check → CONTINUE → next_loop_iteration → repeat
```

The distinction: RDT's loop is token-level, within a single forward pass.
ADR-149's loop is patch-level, across multiple inference calls. But both are
**adaptive depth computation** — spend more compute on harder instances.

**Connection:** `RuvllmMutator` + the repair loop = RDT serving as the inference
substrate for iterative code repair. The model's recurrent depth (ACT loops)
handles within-call reasoning; the repair loop handles across-call refinement.
For an agent working on a complex SWE-bench instance, this compounds: harder
patches get more ACT loops per call *and* more repair iterations.

### 3. DeepSeek-V3 wins the 15-model quality-per-dollar benchmark (ADR-085)

Darwin Mode's LLM benchmark (`bench/model-eval/`) evaluated 15 models × 6 languages.
DeepSeek-V3 at $0.4/Mtok tops quality-per-dollar. The local equivalent:
- **DeepSeek-Coder-V2 Q4_K_M** (33B GGUF) fits in 24 GB VRAM — runnable on RTX 5080
- At 300 ms/call (RTX 5080, ADR-258 optimizations) vs 500 ms median OpenRouter
- At $0/call vs $0.01/call (SWE-bench repair loop × 3 iterations = $0.03/instance → $0)

For a 300-instance SWE-bench Lite run with a 3-iteration repair loop: $9 → $0.

---

## ADR-074 Already Defines the ruvvector Integration

Darwin Mode's upstream ADR-074 ("ruvvector-memory-ruflo-fabric") already specifies:
- ruvvector as the population archive backend
- Behavioral embeddings per variant (one vector per scored variant)
- Cross-repo fleet archive via shared ruvvector node

This ADR **implements** ADR-074 against the ruvvector Rust/NAPI API as it actually exists
in `npm/packages/ruvector`. ADR-074 is the design; this is the build spec.

---

## Implementation: Three Components

### Component 1: `RuvllmMutator` (ADR-259, complete — ship it)

Already specified in ADR-259. Implements `CodeGenerator` interface.
Wire it as `--mutator ruvllm` in darwin-mode CLI. **No new work here.**

Operational command:
```bash
# Terminal 1
ruvllm serve --model ~/.cache/models/deepseek-coder-33b-q4.gguf --port 8080 --backend cuda

# Terminal 2 — darwin evolves the harness using local GPU
metaharness-darwin evolve /path/to/repo \
  --mutator ruvllm --ruvllm-url http://localhost:8080 \
  --sandbox agent --generations 5 --children 4
```

### Component 2: `RuvvectorArchive` (implements darwin-mode ADR-074)

New file: `packages/darwin-mode/src/archive-ruvvector.ts`

The darwin-mode archive stores `ArchiveRecord[]` in `archive.json`. The ruvvector
version indexes each record as an HNSW vector, enabling ANN-based parent selection
for the `quality-diversity` and `behavioral-diversity` strategies.

**Embedding scheme** (matches ADR-074 spec):
- Input: variant's `planner.ts` content (384-dim all-MiniLM-L6-v2 via ruvvector ONNX)
- Namespace: `darwin-variants/<repoHash>`
- Metadata: `{ score, generation, surface, parentId }`

```typescript
// packages/darwin-mode/src/archive-ruvvector.ts
// Implements darwin-mode ADR-074 against the ruvvector NAPI API.

import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Runtime-optional: darwin-mode core stays dependency-free.
// If @ruvector/ruvector is available, use HNSW; else fall back to filesystem.
let RuVector: any;
try { RuVector = require('@ruvector/ruvector'); } catch { /* fall back */ }

export class RuvvectorArchive {
  private db: any;           // RuVectorDB instance
  private namespace: string; // 'darwin-variants/<repoHash>'
  private available: boolean;

  constructor(repoHash: string, dbPath = '.ruvvector/darwin.db') {
    this.namespace = `darwin-variants/${repoHash}`;
    this.available = !!RuVector;
    if (this.available) {
      this.db = new RuVector.RuVectorDB({ path: dbPath, dimension: 384 });
    }
  }

  async upsert(record: ArchiveRecord, plannerContent: string): Promise<void> {
    if (!this.available) return;          // silent fallback — filesystem archive still writes
    const vector = await this.embed(plannerContent);
    await this.db.upsert(this.namespace, {
      id: record.variantId,
      vector,
      metadata: {
        score: record.score.finalScore,
        generation: record.generation,
        surface: record.mutatedSurface,
        parentId: record.parentId ?? null,
      },
    });
  }

  /** ANN search — O(log n) vs O(n) for exhaustive behavioural diversity selection. */
  async selectDiverse(k: number, queryContent: string): Promise<string[]> {
    if (!this.available) return [];
    const q = await this.embed(queryContent);
    const hits = await this.db.search(this.namespace, q, { limit: k * 3 });
    return greedyMaxDispersion(hits, k).map((h: any) => h.id);
  }

  async selectByScore(k: number): Promise<string[]> {
    if (!this.available) return [];
    const all = await this.db.list(this.namespace, {
      sortBy: 'metadata.score', order: 'desc', limit: k,
    });
    return all.map((r: any) => r.id);
  }

  private async embed(text: string): Promise<Float32Array> {
    return RuVector.embed(text);         // all-MiniLM-L6-v2, 384-dim, ONNX
  }
}

/** Greedy maximum-dispersion subset: pick the k points most spread apart. */
function greedyMaxDispersion(results: any[], k: number): any[] {
  if (results.length <= k) return results;
  const chosen = [results[0]];
  while (chosen.length < k) {
    let best = -1, bestDist = -Infinity;
    for (let i = 0; i < results.length; i++) {
      if (chosen.includes(results[i])) continue;
      const minD = Math.min(...chosen.map(c => cosineDist(c.vector, results[i].vector)));
      if (minD > bestDist) { bestDist = minD; best = i; }
    }
    if (best === -1) break;
    chosen.push(results[best]);
  }
  return chosen;
}
function cosineDist(a: Float32Array, b: Float32Array): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i]**2; nb += b[i]**2; }
  return 1 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}
```

**CLI integration** (additive flags):
```bash
metaharness-darwin evolve /path/to/repo \
  --mutator ruvllm \
  --archive ruvvector \          # activates RuvvectorArchive
  --db-path .ruvvector/darwin.db \
  --selection behavioral-diversity
```

### Component 3: RDT Depth Router for the Repair Loop

When `RuvllmMutator` drives the repair loop (ADR-149), the complexity of the patch
task correlates with how many repair iterations are needed. A depth signal from ruvllm
can route easy vs hard repairs:

```typescript
// In RuvllmMutator, during repair-loop mode:
const failureCount = input.failedTraces.length;
const patchComplexity = estimateComplexity(input.parentCode, input.failedTraces);

// Pass x-ruvllm-max-loops as a custom header (ruvllm-cli reads this)
// Low complexity → 4 ACT loops; high complexity → 12 ACT loops
const maxLoops = patchComplexity > 0.7 ? 12 : patchComplexity > 0.4 ? 8 : 4;
headers['x-ruvllm-max-loops'] = String(maxLoops);
```

For SWE-bench: easy instances (clear test failure, small diff) get fast inference;
hard instances (complex multi-file changes, ambiguous failures) get deeper reasoning.
This directly addresses the bottleneck ADR-146 identified: patch emission quality
on hard instances.

---

## Deep Review: Is Darwin Mode Useful in MetaHarness?

### Yes — and Here Is Why Each Layer Matters

**1. Darwin Mode closes the generation-to-improvement gap.**  
MetaHarness generates a harness; Darwin Mode improves it. This is the missing loop
in current metaharness deployments. `npx metaharness <name>` already produces `npm run evolve`
(ADR-147) — the scaffold is pre-wired.

**2. The repair loop (ADR-149) + ruvllm is the highest-ROI integration.**  
SWE-bench localization lifted file-recall 44.7% → 59.7% but resolve rate held flat.
ADR-146's conclusion: the bottleneck moved to patch emission. Iterative repair with a
local GPU model that adaptively deepens reasoning for hard instances is the measured
next lever. This is not speculative; it is the documented active work.

**3. The behavioral-diversity result (5/5 vs 0/5) justifies the ruvvector ANN archive.**  
Darwin Mode's ADR-105 showed greedy `score` selection fails 0/5 on deceptive epistatic
landscapes while `behavioral-diversity` succeeds 5/5. The behavioral-diversity selector
needs ANN search to be practical at fleet scale. ruvvector provides this natively.

**4. DeepSeek-V3 quality-per-dollar + ruvllm GPU = $0 inference.**  
The 15-model benchmark (ADR-085) found $0.4/Mtok frontier-quality. A Q4_K_M quantized
local equivalent served via ruvllm on RTX 5080 runs at ~300 ms/call and $0/call.
A 300-instance SWE-bench run with 3-iteration repair: $9 → $0.

**5. darwin-mode ADR-074 already specifies the ruvvector integration.**  
This is not new design work — it is implementation of an upstream ADR that was written
anticipating the ruvvector API. `RuvvectorArchive` above is a direct implementation of
ADR-074's spec against the `npm/packages/ruvector` bindings.

### What Darwin Mode Is Not

Darwin Mode is **not**:
- A model training system (no weight updates, no gradient descent)
- A replacement for RLHF or fine-tuning (it improves the harness, not the model)
- A general-purpose autonomous agent (the sandbox is deliberately constrained)

These are the right non-goals. The "frozen model, evolving harness" thesis is orthogonal to
ruvllm's "GPU-resident recurrent depth inference" thesis. They compose without conflict.

---

## SWE-bench Economics with ruvllm (Concrete)

| Config | Model | Cost/instance | Resolve rate | Active lever |
|--------|-------|--------------|-------------|-------------|
| Current baseline (ADR-144) | deepseek-chat via OpenRouter | $0.01 | 7.7% | — |
| + repair loop × 3 (ADR-149) | deepseek-chat via OpenRouter | $0.03 | *measuring* | test feedback |
| ruvllm local (this ADR) | deepseek-coder-33b Q4 (RTX 5080) | $0 | TBD | recurrent depth |
| ruvllm + repair loop (this ADR) | deepseek-coder-33b Q4 (RTX 5080) | $0 | TBD | depth × iterations |

The $0 path is the first time iterative repair becomes **cost-unlimited** — you can run
as many repair iterations as the RTX 5080 has GPU time for, without a token budget constraint.
This changes the optimization surface: instead of minimizing API calls, maximize repair quality.

---

## Consequences

| Dimension | Current (OpenRouter) | With ruvllm + ruvvector |
|-----------|---------------------|------------------------|
| Mutation cost | $0.15–$0.30/sweep | $0 |
| Repair loop cost | $0.01–$0.03/SWE-bench instance | $0 |
| Selection (100 variants) | O(n) scan | O(log n) ANN (ruvvector) |
| Cross-repo knowledge | Impossible | Fleet archive (darwin ADR-074) |
| Repair depth | Fixed 3 iterations | Adaptive (RDT loops + repair count) |
| Air-gap support | No | Yes |

---

## Implementation Plan

| Step | File | Effort | Dep |
|------|------|--------|-----|
| 1. `RuvllmMutator` | `darwin-mode/src/ruvllm-mutator.ts` | 80 LOC | ADR-259 |
| 2. CLI flags `--mutator ruvllm` | `darwin-mode/src/cli.ts` | 20 LOC | 1 |
| 3. `RuvvectorArchive` | `darwin-mode/src/archive-ruvvector.ts` | 120 LOC | — |
| 4. CLI flags `--archive ruvvector --db-path` | `darwin-mode/src/cli.ts` | 15 LOC | 3 |
| 5. Depth router header in `RuvllmMutator` | `ruvllm-mutator.ts` | 30 LOC | 1 |
| 6. ruvllm-cli reads `x-ruvllm-max-loops` header | `ruvllm-cli/bin/ruvllm.js` | 20 LOC | — |

Steps 1–2 directly enable the replacement of OpenRouter inference.  
Steps 3–4 implement darwin-mode ADR-074 against the real ruvvector API.  
Steps 5–6 connect the repair loop to RDT's adaptive depth.

---

## Acceptance Test

```bash
# Start ruvllm (RTX 5080, CUDA 13)
ruvllm serve --model ~/.cache/models/deepseek-coder-33b-q4.gguf --port 8080 --backend cuda

# Run full integration: local mutator + ruvvector archive + behavioral-diversity selection
metaharness-darwin evolve /path/to/test-agent-repo \
  --mutator ruvllm --ruvllm-url http://localhost:8080 \
  --archive ruvvector --db-path .ruvvector/darwin.db \
  --sandbox agent \
  --selection behavioral-diversity \
  --generations 5 --children 4

# Pass criteria:
# ✅ Generation 5 winner score > generation 1 winner by ≥ 0.05
# ✅ ruvvector .db contains 20+ indexed variants
# ✅ Zero OpenRouter API calls in network log
# ✅ Total elapsed < 120 s (RTX 5080)
# ✅ sandbox agent: real surface code executed in child process
```

---

## Alternatives Considered

**Use darwin-mode's filesystem archive + ruvvector as a sidecar index.**  
Dual-write introduces consistency hazards (archive.json and ruvvector can diverge on crash).
Clean replacement preferred.

**Use darwin-mode's own ADR-091 Poincaré embedding for behavioral diversity.**  
ADR-091 computes Poincaré-ball embeddings from traces. This is complementary: traces are
computed post-sandbox, embeddings are computed then. ruvvector stores both — the
pre-sandbox planner embedding (for parent selection before sandbox) and the post-sandbox
Poincaré vector (for behavioral diversity scoring after sandbox). Orthogonal, not competing.

**Upgrade to frontier model via OpenRouter for better SWE-bench resolve rate.**  
Per ADR-085's 15-model benchmark, quality-per-dollar peaks at DeepSeek-V3 ($0.4/Mtok),
not frontier models ($3–20/Mtok). A local quantized equivalent at $0/Mtok is always better
than $0.4/Mtok when RTX 5080 GPU time is available, and comparable in quality.
