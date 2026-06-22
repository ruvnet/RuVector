# MetaHarness Integration Architecture for RuVector: Complete Summary

**Prepared**: 2026-06-21  
**Status**: Ready for Implementation (Phase 1 Kickoff)  
**Scope**: RuVector comprehensive benchmark suite + Darwin Mode autonomous optimization  
**Effort**: 16 weeks, 8 concurrent agents, ~12K LOC

---

## What We're Building

A **3-ADR, 5-phase integration** that transforms RuVector's benchmarking from fragmented scripts into a rigorous, auditable, autonomous optimization system:

1. **ADR-265**: Defines **WHAT** we measure (5 categories, 4-component score)
2. **ADR-266**: Defines **HOW** Darwin Mode evolves configs (32 mutation surfaces, graceful degradation)
3. **ADR-267**: Defines **HOW WE PROVE IT** (3-tier validation, cryptographic audit trails)

### Why This Matters

- **Before**: "RaBitQ achieves 512× compression" (unverifiable)
- **After**: "RaBitQ achieves 512× compression with 0.92 recall on SIFT1M (manifest: SHA256=..., signature: ed25519=...)" (reproducible, auditable)

---

## The Three ADRs (Complete)

### ADR-265: Comprehensive Benchmark Suite

**Core Decision**: Unify measurement across ANN-Benchmarks, BEIR, VectorDBBench, MTEB with:
- 5 measurement categories (ANN, compression, latency, streaming, embedding quality)
- 4-component scoring function: `0.4*recall + 0.3*log(QPS) + 0.2*memory + 0.1*latency`
- Fixed baselines (reproducibility) vs mutable configs (evolution)

**File**: `/docs/adr/ADR-265-ruvector-comprehensive-benchmark-suite.md` (280 lines)

### ADR-266: Darwin Mode Integration

**Core Decision**: Integrate @metaharness/darwin as optional evolution layer respecting ADR-150 invariants:
- 32 mutation surfaces across 8 modules (HNSW M, RaBitQ bits, Matryoshka dims, etc.)
- Single evolution loop: generations → ranking → elite selection → checkpoint
- Graceful fallback to Phase 2 grid search if MetaHarness missing
- 100% try-catch wrapped, no hard dependencies

**File**: `/docs/adr/ADR-266-metaharness-darwin-integration.md` (350 lines)

**Key Implementation**: 
```typescript
// Graceful degradation example from ADR-266
async function benchmarkWithEvolution() {
  const darwin = await initDarwinMode();  // Returns null if missing
  if (darwin) return runDarwinEvolution();
  else return sweepConfigs(...);          // Fallback to Phase 2
}
```

### ADR-267: SOTA Validation Protocol

**Core Decision**: 3-tier validation with witness signing (ADR-103):
- **Tier 1 (Daily Smoke)**: Quick regression gate (<10 min)
- **Tier 2 (Weekly Validation)**: Full ANN-Benchmarks, all modules, signed manifest
- **Tier 3 (Biannual Publication)**: 3 replications, statistical CIs, Ed25519 signature

**File**: `/docs/adr/ADR-267-sota-validation-protocol.md` (400 lines)

**Example Manifest** (from ADR-267):
```json
{
  "timestamp": "2026-06-21T12:34:56Z",
  "ruvector_commit": "abc123...",
  "configurations": [{
    "module": "rabitq",
    "config": {"bits": 1, "rotation": true},
    "recall_at_10": 0.92,
    "qps": 100000,
    "memory_mb": 128
  }],
  "witness": {
    "signature_algorithm": "ed25519",
    "signature": "..."
  }
}
```

---

## The 5-Phase Implementation Plan

**File**: `/docs/metaharness-implementation-plan.md` (500 lines with detailed CI/CD, code sketches, rollout timeline)

### Phase 1: ANN-Benchmarks Compatibility (4 weeks)
- HDF5 loader for SIFT1M, GIST1M, GloVe
- Single-dataset harness (build → query → measure)
- Baseline config file
- Daily CI smoke test
- **Deliverable**: `scripts/benchmark/ann-datasets.ts`, `single-dataset-harness.ts`, smoke test workflow

### Phase 2: Parameter Sweep (3 weeks)
- Grid search over HNSW M∈[4,32], efConstruction∈[50,400], etc.
- Pareto frontier identification
- Random sampling fallback
- **Deliverable**: Pareto frontier JSON, visualization HTML

### Phase 3: BEIR + VectorDBBench (4 weeks)
- BEIR corpus loader (11 datasets, 26M docs)
- Retrieval harness (NDCG@10, MRR, MAP)
- VectorDBBench workloads (insert-heavy, query-heavy)
- **Deliverable**: BEIR baseline JSON, workload results

### Phase 4: Darwin Evolution (3 weeks)
- Integrate @metaharness/darwin (optional)
- 32 mutation surface definitions
- Evolution loop with checkpoint strategy
- **Deliverable**: Evolved configs archive, best-config leaderboard

### Phase 5: MTEB Embedding Quality (2 weeks)
- MTEB dataset loader (170K sentences)
- STS evaluation, clustering scoring
- **Deliverable**: MTEB baseline, embedding quality report

### Timeline
```
2026-06-21 — Phase 1 kickoff
2026-07-19 — Phase 1 complete, Phase 2 starts
2026-08-09 — Phase 2 complete, Phase 3 starts
2026-09-06 — Phase 3 complete, Phase 4 starts
2026-09-27 — Phase 4 complete, Phase 5 starts
2026-10-11 — Phase 5 complete, MVP launch
```

---

## Architecture & File Structure

### New Directories Created

```
ruvector/
├── docs/adr/
│   ├── ADR-265-ruvector-comprehensive-benchmark-suite.md
│   ├── ADR-266-metaharness-darwin-integration.md
│   ├── ADR-267-sota-validation-protocol.md
│   └── [existing ADRs]
│
├── docs/metaharness-implementation-plan.md  (this file)
│
├── scripts/benchmark/                       (21 TypeScript files, ~7.5K LOC)
│   ├── ann-datasets.ts                      (400 lines, HDF5 loader)
│   ├── single-dataset-harness.ts            (600 lines)
│   ├── baseline-configs.json                (200 lines)
│   ├── result-formatter.ts                  (300 lines)
│   ├── check-regression.js                  (150 lines)
│   ├── sweep-config.json                    (150 lines)
│   ├── sweep-harness.ts                     (800 lines)
│   ├── pareto-visualizer.ts                 (400 lines)
│   ├── beir-loader.ts                       (500 lines)
│   ├── retrieval-harness.ts                 (700 lines)
│   ├── vdb-bench-workloads.ts               (400 lines)
│   ├── darwin-score-policy.ts               (300 lines)
│   ├── mutation-surfaces.ts                 (400 lines)
│   ├── darwin-harness.ts                    (600 lines)
│   ├── mteb-loader.ts                       (300 lines)
│   ├── mteb-harness.ts                      (400 lines)
│   ├── embedding-quality.ts                 (350 lines)
│   ├── witness-signer.ts                    (200 lines)
│   ├── verify-manifest.ts                   (150 lines)
│   └── index.ts                             (50 lines)
│
├── crates/ruvector-bench/                   (3 Rust files, ~1.5K LOC)
│   ├── Cargo.toml                           (minimal)
│   └── src/
│       ├── hdf5_loader.rs                   (350 lines)
│       ├── grid_search.rs                   (500 lines)
│       ├── retrieval.rs                     (600 lines)
│       └── lib.rs
│
├── .github/workflows/
│   ├── benchmark-smoke.yml                  (100 lines, daily)
│   ├── benchmark-sweep.yml                  (120 lines, weekly)
│   ├── benchmark-beir.yml                   (140 lines, Monday)
│   └── darwin-evolution.yml                 (120 lines, Wednesday)
│
├── docs/validation/
│   ├── smoke-baseline-2026-06.json          (baseline, committed)
│   ├── manifests/
│   │   ├── 2026-06-21-tier2-unsigned.json   (signed per-release)
│   │   └── ...
│   ├── tier3-replications/
│   │   └── 2026-09-15/
│   │       ├── run1.csv
│   │       ├── run2.csv
│   │       └── run3.csv
│   ├── witness-public-key.pem               (Ed25519)
│   └── witness-manifest-index.json
│
└── docs/darwin/
    └── evolution-runs/
        ├── 2026-07-10-run-1.json
        ├── 2026-07-17-run-2.json
        └── ...
```

---

## CI/CD Gates & Automation

### Daily (Smoke Test)
- Trigger: every commit to main
- Runtime: <10 min
- Dataset: SIFT1M subset (100K vectors)
- Modules: HNSW only
- Gate: Fail if recall@10 regresses >2%

### Weekly (Full Validation)
- Trigger: Monday midnight
- Runtime: <4 hours
- Dataset: SIFT1M, GIST1M, GloVe + BEIR subset
- Modules: All 8 core modules
- Artifact: Signed Tier 2 manifest

### Weekly (Darwin Evolution)
- Trigger: Wednesday noon
- Runtime: <6 hours
- Dataset: SIFT1M
- Generations: 10, population 20
- Artifact: Generation checkpoints

### Biannual (Publication Audit)
- Trigger: Manual (before paper/leaderboard claim)
- Runtime: ~12 hours
- Replications: 3 per config
- Artifact: Signed Tier 3 manifest + statistical summary

---

## ADR-150 Compliance

All MetaHarness integration respects the 4 invariants:

1. **Removable**: `npm ls --without-deps @metaharness/*` → still works
2. **Optional**: Only in `optionalDependencies` + `peerDependencies`
3. **Graceful degradation**: Every Darwin call wrapped in try-catch
4. **CI gate**: Daily smoke test runs without MetaHarness

**Enforcement** (from ADR-266):
```typescript
async function initDarwinMode() {
  try {
    const Darwin = await import("@metaharness/darwin");
    return Darwin;  // Optional loaded successfully
  } catch (e) {
    if (e.code === "MODULE_NOT_FOUND") {
      console.warn("[darwin] @metaharness/darwin not installed");
      console.warn("[darwin] Falling back to Phase 2 grid search");
      return null;  // Graceful degradation
    }
    throw e;  // Other errors fatal
  }
}
```

---

## Success Metrics (MVP Exit Criteria)

### Phase 1 Complete
- [ ] SIFT1M loads in <30s
- [ ] Single benchmark <5 min per config
- [ ] Accuracy within ±1% of Python baseline
- [ ] Smoke test daily with <2% regression tolerance

### Phase 2 Complete
- [ ] Grid sweep <2 hours
- [ ] 10-15 non-dominated Pareto configs identified
- [ ] Top 3 beat baseline on 2+ metrics

### Phase 3 Complete
- [ ] BEIR indexing <5 min per dataset
- [ ] NDCG@10 ≥ 0.45 on NQ
- [ ] VectorDBBench 5K QPS sustained

### Phase 4 Complete
- [ ] Darwin evolves 3+ metric improvement
- [ ] Graceful fallback if missing
- [ ] 100% generation checkpoints

### Phase 5 Complete
- [ ] MTEB <10 hours
- [ ] all-MiniLM ≥0.45 NDCG@10

### Post-MVP (Publication)
- [ ] Signed Tier 3 manifests for all SOTA claims
- [ ] Witness signatures verifiable by third parties
- [ ] Paper references manifest hash + DOI
- [ ] ANN-Benchmarks leaderboard entry submitted

---

## Estimated Effort

| Phase | Team | Weeks | Files | Risks |
|-------|------|-------|-------|-------|
| **1** | 2 eng | 4 | 7 TS, 1 Rust | HDF5 compat |
| **2** | 1 eng | 3 | 3 TS, 1 Rust | Grid explosion |
| **3** | 2 eng | 4 | 5 TS, 1 Rust | BEIR size (26M) |
| **4** | 1 eng | 3 | 3 TS | Darwin API |
| **5** | 1 eng | 2 | 3 TS | Infra |
| **Total** | **8** | **16** | **21 TS, 3 Rust** | **MetaHarness dep** |

---

## Key Decisions & Rationale

### Why These Datasets?
- **SIFT1M**: Industry standard, well-understood
- **BEIR**: Retrieval ground truth, 11 diverse datasets
- **MTEB**: Embedding quality, 170K sentences
- **Not specialized leaderboards**: Maintain reproducibility

### Why Darwin Mode?
- Manual grid search is O(n^k) in config space
- Darwin intelligently samples via genetic algorithm + simulated annealing
- Expected: beat baseline on 3+ metrics in 10 generations (~20 hours)

### Why Witness Signing?
- SOTA claims need cryptographic proof (tamper-evidence)
- Enables third-party verification
- Required for publication credibility

---

## Cross-References

| Document | Purpose | Status |
|----------|---------|--------|
| `ADR-265` | Measurement spec | Complete |
| `ADR-266` | Darwin integration | Complete |
| `ADR-267` | Validation protocol | Complete |
| `metaharness-implementation-plan.md` | 5-phase detailed plan | This file |
| `ADR-150` | MetaHarness surfaces (upstream) | Reference |
| `ADR-103` | Witness chain (upstream) | Reference |
| `ADR-128` | SOTA gap implementations | Related context |

---

## Next Steps

1. **Immediate** (this week):
   - Review & approve 3 ADRs
   - Create GitHub milestone "MetaHarness MVP"
   - Assign Phase 1 team

2. **Phase 1 Kickoff** (next 4 weeks):
   - HDF5 loader implementation
   - Smoke test workflow
   - Baseline config finalization

3. **Weekly Sync** (ongoing):
   - Phase completeness check
   - ADR-150 compliance audit
   - Timeline adjustments

---

## Questions & Open Issues

1. **Leaderboard target**: Submit to ANN-Benchmarks, VectorDBBench, or both?
   - **Proposal**: Both (wider visibility, cross-validation)

2. **Embedding model**: Which E5 variant for BEIR retrieval?
   - **Proposal**: E5-large-v2 (standard baseline)

3. **Hardware variance**: Run on GitHub Actions (variable) or GCP (controlled)?
   - **Proposal**: GitHub Actions + explicit hardware disclosure in manifest

4. **Publication venue**: NeurIPS, MLSys, or conference?
   - **Proposal**: NeurIPS Systems Track (first choice), MLSys (fallback)

---

**Prepared by**: Claude Code MetaHarness Architect  
**Review Gate**: CTO + Lead Engineer sign-off before Phase 1 kickoff

