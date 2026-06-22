# MetaHarness Integration for RuVector: Quick Start

This directory contains the complete architecture for integrating MetaHarness Darwin Mode with RuVector's benchmark suite, enabling autonomous parameter optimization against public leaderboards (ANN-Benchmarks, BEIR, VectorDBBench, MTEB).

## Documents (Read in Order)

### 1. Executive Summary (Start Here)
**File**: `docs/METAHARNESS-ARCHITECTURE-SUMMARY.md`  
**Length**: 500 lines, ~20 min read  
**What it covers**: Entire project overview, 3 ADRs, 5 phases, effort estimate, success criteria

### 2. Three Architecture Decision Records (ADRs)

#### ADR-265: Comprehensive Benchmark Suite
**File**: `docs/adr/ADR-265-ruvector-comprehensive-benchmark-suite.md`  
**Length**: 280 lines  
**What it covers**: 
- What we measure (5 categories: ANN, compression, latency, streaming, embedding quality)
- How we score configs (4-component function: recall, QPS, memory, latency)
- Baseline anchors and mutable surfaces
- Why these datasets (SIFT1M, GIST1M, GloVe, BEIR, MTEB)

#### ADR-266: Darwin Mode Integration
**File**: `docs/adr/ADR-266-metaharness-darwin-integration.md`  
**Length**: 350 lines  
**What it covers**:
- How Darwin Mode evolves configs (32 mutation surfaces, genetic algorithm)
- ADR-150 compliance (graceful degradation if MetaHarness missing)
- Scoring policy implementation (TypeScript code)
- Evolution loop with checkpoint strategy
- CI/CD workflow (weekly evolution runs)

#### ADR-267: SOTA Validation Protocol
**File**: `docs/adr/ADR-267-sota-validation-protocol.md`  
**Length**: 400 lines  
**What it covers**:
- 3-tier validation (Tier 1: daily smoke, Tier 2: weekly full, Tier 3: publication audit)
- Witness signing with Ed25519 (cryptographic audit trails)
- Regression detection and SOTA claim rules
- File structure for manifests and replications

### 3. Detailed Implementation Plan
**File**: `docs/metaharness-implementation-plan.md`  
**Length**: 500 lines, detailed code sketches and CI/CD configs  
**What it covers**:
- All 5 phases with deliverables and success gates
- File structure (21 TypeScript, 3 Rust files)
- Effort breakdown (16 weeks, 8 agents)
- Rollout timeline (June 21 - Oct 11, 2026)
- Risk mitigation

## Quick Reference: The 5 Phases

```
Phase 1 (4w):  ANN-Benchmarks loader + smoke test
Phase 2 (3w):  Parameter sweep + Pareto frontier  
Phase 3 (4w):  BEIR + VectorDBBench integration
Phase 4 (3w):  Darwin Mode evolution loop
Phase 5 (2w):  MTEB embedding quality validation
─────────────────────────────────────────────
Total:  16 weeks, ~12K LOC
```

## Key Scoring Function

```
score = 0.4 * recall@10_norm 
      + 0.3 * log(QPS/baseline_QPS)
      + 0.2 * (1 - min(1, memory/baseline_memory))
      + 0.1 * (1 - min(1, p99_ms/baseline_p99_ms))
```

## ADR-150 Compliance (MetaHarness Removable)

All integration respects 4 invariants:

1. ✅ **Removable**: `npm ls --without-deps @metaharness/*` still works
2. ✅ **Optional**: Only in `optionalDependencies` + `peerDependencies`
3. ✅ **Graceful degradation**: Every Darwin call wrapped in try-catch → fallback to grid search
4. ✅ **CI gate**: Daily smoke test runs WITHOUT MetaHarness

Example graceful degradation:
```typescript
async function initDarwinMode() {
  try {
    return await import("@metaharness/darwin");
  } catch (e) {
    if (e.code === "MODULE_NOT_FOUND") {
      console.warn("[darwin] MetaHarness missing, using grid search");
      return null;
    }
    throw e;
  }
}
```

## File Structure

```
ruvector/
├── docs/adr/
│   ├── ADR-265-ruvector-comprehensive-benchmark-suite.md       (280 lines)
│   ├── ADR-266-metaharness-darwin-integration.md               (350 lines)
│   └── ADR-267-sota-validation-protocol.md                     (400 lines)
│
├── docs/metaharness-implementation-plan.md                      (500 lines)
├── docs/METAHARNESS-ARCHITECTURE-SUMMARY.md                    (500 lines)
├── METAHARNESS-README.md                                        (this file)
│
├── scripts/benchmark/                                            (21 TypeScript files, ~7.5K LOC)
│   ├── ann-datasets.ts                                          (400 lines)
│   ├── single-dataset-harness.ts                                (600 lines)
│   ├── sweep-harness.ts                                         (800 lines)
│   ├── darwin-harness.ts                                        (600 lines)
│   ├── beir-loader.ts                                           (500 lines)
│   ├── retrieval-harness.ts                                     (700 lines)
│   ├── mteb-harness.ts                                          (400 lines)
│   └── ... 14 more files
│
├── crates/ruvector-bench/
│   └── src/
│       ├── hdf5_loader.rs                                       (350 lines)
│       ├── grid_search.rs                                       (500 lines)
│       └── retrieval.rs                                         (600 lines)
│
├── .github/workflows/
│   ├── benchmark-smoke.yml                                      (daily)
│   ├── benchmark-sweep.yml                                      (weekly)
│   ├── benchmark-beir.yml                                       (weekly)
│   └── darwin-evolution.yml                                     (weekly)
│
└── docs/validation/
    ├── smoke-baseline-2026-06.json                              (baseline)
    ├── manifests/                                                (signed per-release)
    ├── tier3-replications/                                      (publication audits)
    ├── witness-public-key.pem                                   (Ed25519)
    └── witness-manifest-index.json
```

## Success Criteria (MVP)

**Phase 1**: SIFT1M in <30s, smoke test ±1% accuracy
**Phase 2**: 10-15 Pareto configs, grid sweep <2h
**Phase 3**: BEIR NDCG@10 ≥0.45 on NQ, VectorDBBench 5K QPS
**Phase 4**: Darwin evolves 3+ metric improvement
**Phase 5**: MTEB <10h, all-MiniLM ≥0.45 NDCG@10

**Post-MVP**: Signed Tier 3 manifests, ANN-Benchmarks submission

## Key Decisions

### Why These Datasets?
- **SIFT1M**: Industry standard, well-understood
- **BEIR**: Retrieval ground truth, 11 diverse datasets
- **MTEB**: Embedding quality, 170K sentences
- **Not specialized leaderboards**: Maintain reproducibility

### Why Darwin Mode?
- Manual grid search is O(n^k) in parameter space
- Darwin intelligently samples via genetic algorithm + simulated annealing
- Expected: beat baseline on 3+ metrics in 10 generations (~20 hours)

### Why Witness Signing?
- SOTA claims need cryptographic proof (tamper-evidence)
- Enables third-party verification
- Required for publication credibility

## Next Steps

1. **This week**: Review & approve 3 ADRs
2. **Next 4 weeks**: Phase 1 (HDF5 loader, smoke test)
3. **Ongoing**: Weekly sync on completion, ADR-150 compliance audit

## Team & Contacts

- **MetaHarness Architect**: Claude Code
- **Phase 1 Lead**: (TBD)
- **Darwin Integration Lead**: (TBD)
- **Validation Protocol Lead**: (TBD)

## References

- **ADR-150**: MetaHarness Integration Surfaces (upstream)
- **ADR-103**: Witness Chain (upstream)
- **ADR-128**: SOTA Gap Implementations (context)
- **ANN-Benchmarks**: https://github.com/erikbern/ann-benchmarks
- **BEIR**: https://github.com/beir-cellar/beir
- **VectorDBBench**: https://github.com/zilliztech/VectorDBBench
- **MTEB**: https://github.com/embeddings-benchmark/mteb

---

**Status**: Ready for Phase 1 Kickoff  
**Last Updated**: 2026-06-21  
**Prepared by**: Claude Code MetaHarness Architect

