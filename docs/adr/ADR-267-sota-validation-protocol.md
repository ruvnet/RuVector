# ADR-267: SOTA Validation Protocol for RuVector

**Status**: Accepted  
**Date**: 2026-06-21  
**Authors**: Claude Code MetaHarness Architect  
**Supersedes**: None  
**Related**: ADR-103 (Witness Chain), ADR-265 (Benchmark Suite), ADR-266 (Darwin Mode)

---

## Context

RuVector makes 10+ SOTA claims across vector search, compression, and embedding quality. Public claims (papers, leaderboards, marketing) require reproducible audit trails—not just numbers, but full provenance including:

- RuVector version & commit hash
- Exact index configuration
- Hardware environment (CPU cores, RAM, GPU)
- Dataset snapshot & ground truth
- Raw metrics + statistical confidence intervals
- Cryptographic signature for tamper-evidence

**Current State**: Benchmarks produce JSON results but no signed manifest. Third parties cannot verify claims.

**Problem**: Without SOTA validation protocol:
1. Claims unverifiable (can anyone reproduce?)
2. Regressions go undetected (no baseline snapshot)
3. Publications rejected by peer reviewers (missing provenance)
4. Marketing claims unreliable (no legal/scientific backing)

---

## Decision

Implement **3-tier SOTA validation protocol** with cryptographic audit trails (ADR-103 witness chain):

### Tier 1: Smoke Test (Daily CI)
- Single small dataset (SIFT1M subset, 100K vectors)
- 3 index configs (baseline, aggressive, memory-optimized)
- Pass/fail on regression threshold (≤2% recall loss)
- No artifact retention (just CI log)

### Tier 2: Validation Run (Per-release)
- Full ANN-Benchmarks (SIFT1M, GIST1M, GloVe, 1M vectors each)
- All RuVector modules tested
- CSV results + JSON manifest (unsigned)
- Stored in `docs/validation/manifests/`
- Triggers before npm publish

### Tier 3: Publication Audit (Biannual)
- Signed manifest (Ed25519) with full provenance
- Statistical analysis: 95% confidence intervals, cross-validation
- Published to research venues (NeurIPS, MLSys)
- Archived with permanent DOI

---

## Audit Record Schema (JSON)

```json
{
  "version": 1,
  "audit_tier": "tier-2",
  "timestamp": "2026-06-21T12:34:56Z",
  "ruvector": {
    "version": "0.2.32",
    "commit": "abc123def456...",
    "branch": "main"
  },
  "environment": {
    "platform": "Linux",
    "kernel": "6.17.0-20-generic",
    "cpu_cores": 16,
    "cpu_model": "AMD Ryzen 7950X",
    "memory_gb": 128,
    "gpu": "none"
  },
  "datasets": [
    {
      "name": "sift1m",
      "vectors": 1000000,
      "dimension": 128,
      "download_url": "http://ann-benchmarks.com/sift1m.hdf5",
      "download_sha256": "...",
      "base_path": "~/data/sift1m/"
    }
  ],
  "modules_tested": [
    "hnsw",
    "rabitq",
    "matryoshka",
    "pq",
    "hybrid",
    "diskann",
    "colbert",
    "mla"
  ],
  "configurations": [
    {
      "id": "hnsw-baseline",
      "module": "hnsw",
      "config": {
        "M": 12,
        "efConstruction": 200,
        "efSearch": 100
      },
      "metrics": {
        "recall_at_1": 0.99,
        "recall_at_10": 0.85,
        "recall_at_100": 0.78,
        "qps": 45000,
        "memory_mb": 256,
        "build_time_sec": 42.3,
        "latency_p50_ms": 0.22,
        "latency_p99_ms": 5.1,
        "latency_p99_9_ms": 12.3
      },
      "timestamps": {
        "build_started": "2026-06-21T12:34:56Z",
        "build_completed": "2026-06-21T12:35:38Z",
        "query_started": "2026-06-21T12:35:38Z",
        "query_completed": "2026-06-21T12:36:10Z"
      }
    }
  ],
  "baseline_comparison": {
    "baseline_ref": "ANN-Benchmarks 2026-Q2 leaderboard",
    "baseline_date": "2026-06-01",
    "baseline_entry": "HNSW M=16 efConstruction=400",
    "baseline_recall_at_10": 0.87,
    "our_recall_at_10": 0.85,
    "recall_gap": -0.02,
    "regression_detected": false,
    "regression_threshold": 0.02
  },
  "statistical_summary": {
    "tier": "tier-2",
    "replications": 1,
    "confidence_interval_95": {
      "recall_at_10": [0.84, 0.86],
      "qps": [44000, 46000]
    }
  },
  "witness": {
    "signature_algorithm": "ed25519",
    "public_key": "...",
    "signature": "...",
    "signed_fields": [
      "timestamp", "ruvector.commit", "configurations", "metrics"
    ]
  },
  "notes": "SIFT1M, 16 cores, no concurrent write traffic, baseline from public leaderboard",
  "publication": {
    "status": "draft",
    "venue": "NeurIPS 2026 Systems Track",
    "doi": null
  }
}
```

---

## Tier Definitions

### Tier 1: Smoke Test (Daily)

**Trigger**: Every commit to main

**Scope**: 
- Dataset: SIFT1M subset (100K vectors, first 100K rows of HDF5)
- Modules: HNSW only
- Configs: 1 default config
- Queries: 1000 random

**Artifact**: CI log only (no saved results)

**Pass Criteria**:
- Build completes in <5 min
- Recall@10 ≥ baseline * 0.98 (2% regression tolerance)
- No crashes

**On Failure**: Email alert, block PR merge

```yaml
# .github/workflows/benchmark-smoke.yml
jobs:
  smoke:
    runs-on: ubuntu-latest-8core
    timeout-minutes: 10
    steps:
      - name: Run SIFT1M smoke test
        run: npm run benchmark:sift1m:smoke
      
      - name: Check regression
        run: |
          node scripts/check-regression.js \
            --baseline docs/validation/smoke-baseline-2026-06.json \
            --tolerance 0.02
      
      - name: Report
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.checks.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              head_sha: context.sha,
              name: "Benchmark Smoke Test",
              conclusion: "failure",
              output: { title: "Regression detected", summary: "..." }
            });
```

### Tier 2: Validation Run (Per-release)

**Trigger**: Before npm publish + weekly GitHub Actions

**Scope**:
- Datasets: SIFT1M, GIST1M, GloVe (1M vectors each)
- Modules: All 8 core modules
- Configs: 5-10 per module (grid-selected or Pareto frontier)
- Queries: 10K per dataset

**Artifact**: Unsigned JSON manifest + CSV

**Pass Criteria**:
- All 8 modules tested
- NDCG@10 on retrieval ≥ 0.45 (if using E5-large-v2)
- No module regresses >2% on recall
- Build time <4 hours total

**On Failure**: Halt release, investigate

```bash
# Pre-publish hook in CI
npm run benchmark:tier2 --output-dir docs/validation/manifests/
# Manifest stored as: docs/validation/manifests/2026-06-21-tier2-unsigned.json
git add docs/validation/manifests/
npm publish
```

### Tier 3: Publication Audit (Biannual)

**Trigger**: Manual, before paper submission or major leaderboard claim

**Scope**:
- Datasets: SIFT1M, GIST1M, GloVe + BEIR NQ + MTEB STS
- Modules: All 10 modules
- Configs: Darwin-evolved best configs + manual experts
- Replications: 3 runs per config (confidence intervals)
- Queries: 10K per dataset

**Artifact**: Signed manifest (Ed25519) + cross-validation report

**Pass Criteria**:
- 95% confidence intervals overlap with published SOTA
- No regression vs Tier 2 baseline
- Witness signature verifies (no tampering)
- All raw data in `docs/validation/tier3-replications/`

**Publication Checklist**:
- [ ] Witness manifest signed & archived
- [ ] Raw CSV for all replications committed
- [ ] Statistical analysis (mean, std dev, CIs) documented
- [ ] SOTA claim rule satisfied (beat 3 of top-5 on leaderboard)
- [ ] Paper references manifest DOI
- [ ] Submission includes witness signature in appendix

---

## SOTA Claim Rules

A module claims SOTA in a category only if it:
1. **Beats top-3** on public leaderboard (ANN-Benchmarks, VectorDBBench, or BEIR)
2. **Has signed Tier 3 manifest** with full provenance
3. **Includes witness signature** in any publication
4. **Configuration is reproducible** (full config in manifest)
5. **Hardware disclosed** (CPU model, cores, RAM, GPU if used)

Example valid SOTA claim:
```
RaBitQ achieves 0.92 recall@10 with 512× compression on SIFT1M
(see manifest: https://github.com/ruvnet/ruvector/blob/main/docs/validation/manifests/2026-06-21-rabitq-sota.json)
Signature: ed25519 ABC123...XYZ
```

Example invalid claim (missing components):
```
RaBitQ achieves 0.92 recall on SIFT1M
[❌ No manifest, no witness, no config, no hardware disclosed]
```

---

## Regression Detection

**Daily CI regression threshold**: ≤2% loss allowed (smoke test)
**Weekly validation threshold**: ≤1% loss allowed
**Publication threshold**: Must improve or ≤0.5% loss

If regression detected:

1. **Smoke test fails**: Block PR merge
2. **Weekly validation fails**: Alert maintainers, investigate commits
3. **Publication regression**: Retract SOTA claim or revise paper

```typescript
// scripts/check-regression.ts
function checkRegression(
  baseline: BenchmarkMetrics,
  current: BenchmarkMetrics,
  tolerance: number = 0.02
): { pass: boolean; deltas: Record<string, number> } {
  const deltas = {
    recall_at_10: (baseline.recall_at_10 - current.recall_at_10) / baseline.recall_at_10,
    qps: (current.qps - baseline.qps) / baseline.qps,
    memory: (current.memory_mb - baseline.memory_mb) / baseline.memory_mb
  };
  
  const pass = 
    deltas.recall_at_10 <= tolerance &&
    deltas.qps >= -tolerance &&  // slower is OK (within tolerance)
    deltas.memory >= -0.5;        // memory slower OK (up to 50%)
  
  return { pass, deltas };
}
```

---

## Witness Signing (ADR-103)

Each Tier 2+ manifest is signed with Ed25519 private key at `~/.ssh/ruvector-witness-key`:

```typescript
// scripts/witness-signer.ts
import { readFileSync } from "fs";
import { createPrivateKey } from "crypto";

async function signManifest(manifest: AuditRecord): Promise<string> {
  const key = createPrivateKey({
    key: readFileSync("~/.ssh/ruvector-witness-key", "utf8"),
    format: "pem",
    type: "pkcs8"
  });
  
  const fieldsToSign = [
    manifest.timestamp,
    manifest.ruvector.commit,
    JSON.stringify(manifest.configurations),
    JSON.stringify(manifest.baseline_comparison)
  ].join("|");
  
  const sig = createSign("sha256")
    .update(fieldsToSign)
    .sign(key, "hex");
  
  return sig;
}
```

**Verification** (anyone can verify):

```bash
# Public key published in repo
cat docs/validation/witness-public-key.pem

# Verify signature
node scripts/verify-manifest.ts \
  --manifest docs/validation/manifests/2026-06-21-tier2.json \
  --public-key docs/validation/witness-public-key.pem
# Output: Signature valid (no tampering detected)
```

---

## File Structure

```
docs/validation/
├── smoke-baseline-2026-06.json          (Tier 1 baseline, committed)
├── manifests/
│   ├── 2026-06-21-tier2-unsigned.json   (Tier 2, signed before publish)
│   ├── 2026-07-10-tier2-unsigned.json
│   └── 2026-09-15-tier3-rabitq-sota.json (Tier 3, signed for publication)
├── tier3-replications/
│   ├── 2026-09-15-run1.csv
│   ├── 2026-09-15-run2.csv
│   └── 2026-09-15-run3.csv
├── witness-public-key.pem               (Ed25519 public key)
└── witness-manifest-index.json          (List of all signed manifests)
```

---

## CI/CD Integration

### Tier 2 (Weekly Validation)

```yaml
name: Tier 2 Validation
on:
  schedule:
    - cron: "0 0 * * 1"  # Monday midnight
  workflow_dispatch:

jobs:
  tier2:
    runs-on: ubuntu-latest-32core
    timeout-minutes: 240
    steps:
      - name: Download datasets
        run: npm run benchmark:download-datasets
      
      - name: Run Tier 2 benchmark
        run: npm run benchmark:tier2
      
      - name: Sign manifest
        run: |
          node scripts/witness-signer.ts \
            --manifest benchmark-results.json \
            --output docs/validation/manifests/$(date -u +%Y-%m-%d)-tier2.json
      
      - name: Check regression
        run: |
          node scripts/check-regression.js \
            --baseline docs/validation/manifests/baseline-tier2.json \
            --current docs/validation/manifests/$(date -u +%Y-%m-%d)-tier2.json \
            --tolerance 0.01
      
      - name: Commit
        run: |
          git add docs/validation/manifests/
          git commit -m "chore(validation): tier2 run $(date -u +%Y-%m-%d)"
          git push
```

### Tier 3 (Manual Publication)

```bash
#!/bin/bash
# scripts/run-tier3-audit.sh

echo "Running Tier 3 publication audit..."

# 1. Run 3 replications
for i in 1 2 3; do
  echo "Replication $i/3"
  npm run benchmark:tier3 --output-dir tier3-run-$i
done

# 2. Generate statistical summary
node scripts/analyze-replications.ts tier3-run-* > tier3-analysis.json

# 3. Sign all manifests
for manifest in tier3-run-*/manifest.json; do
  node scripts/witness-signer.ts --manifest "$manifest"
done

# 4. Archive to docs/validation/tier3-replications/
mkdir -p docs/validation/tier3-replications/$(date -u +%Y-%m-%d)
mv tier3-run-* docs/validation/tier3-replications/$(date -u +%Y-%m-%d)/

# 5. Commit
git add docs/validation/tier3-replications/
git commit -m "chore(validation): tier3 publication audit $(date -u +%Y-%m-%d)"

echo "Tier 3 audit complete. Ready for publication."
```

---

## Success Criteria

- **Tier 1**: Daily CI gate working, 0 false positives on regression
- **Tier 2**: Pre-release manifests signed, stored in version control
- **Tier 3**: Publication claims verifiable, witness signatures valid, 95% CIs documented

---

## References

- ADR-103: Witness Chain for Cryptographic Verification
- ADR-265: RuVector Comprehensive Benchmark Suite
- ADR-266: MetaHarness Darwin Mode Integration
- ANN-Benchmarks: https://github.com/erikbern/ann-benchmarks
- VectorDBBench: https://github.com/zilliztech/VectorDBBench

