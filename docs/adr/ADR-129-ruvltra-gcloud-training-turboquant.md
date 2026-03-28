# ADR-129: RuvLTRA Model Training & TurboQuant Optimization on Google Cloud

## Status

Accepted — Phase 1 (calibration) deployed and executing. Governance and release gates implemented.

## Date

2026-03-28

## Implementation Status (2026-03-28)

| Phase | Status | Details |
|-------|--------|---------|
| **Infrastructure** | **Deployed** | Docker image `gcr.io/ruv-dev/ruvltra-training:latest` (torch 2.5.1+cu124, libgomp, gguf, peft, trl) |
| **Cloud Run Jobs** | **3 deployed** | `ruvltra-calibration`, `ruvltra-nightly-train`, `ruvltra-benchmark` (all L4 GPU) |
| **Cloud Schedulers** | **2 enabled** | Nightly 03:00 UTC, Weekly benchmark Mon 06:00 UTC |
| **Phase 1: Calibration** | **Complete** | All 4 models calibrated on L4 GPU. TQ profiles + benchmarks uploaded to HuggingFace. Results: 75.4 tok/s (small), 62.6 tok/s (medium), 67.1 tok/s (claude-code) |
| **Phase 2: SFT** | **Executing** | LoRA SFT running on L4 GPU (rank-16, 2 epochs, lr=2e-5). Corpus: 230 records, 530K tokens |
| **Phase 3: Benchmarks** | **Executing** | Release gate automation tested. L4 GPU benchmark job running. Calibration benchmarks complete for all 4 models |
| **Phase 4: Publishing** | **Complete** | TurboQuant sidecar configs + benchmark results uploaded to all 4 HF models. Model card READMEs updated with benchmark tables |
| **Tooling** | **ruvllm-native** | Uses RuvltraQuantizer + TurboQuantProfile (Rust), gguf + llama-cpp-python (Python). No llama.cpp source compilation. |

## Context

RuvLTRA models (0.5B-3B parameters) are the purpose-built LLMs powering Claude Code integrations via RuvLLM. The current published models (`ruv/ruvltra-claude-code`, `ruv/ruvltra`, `ruv/ruvltra-medium`, `ruv/ruvltra-small`) have accumulated 8,281 HuggingFace downloads but haven't been retrained since their initial release. Meanwhile, significant new capabilities have been implemented:

1. **TurboQuant** (1,483 lines) — 2-4 bit asymmetric per-channel KV-cache quantization with 6-8x memory reduction
2. **WET Processing** — Common Crawl data pipeline (`brain-wet-daily`) extracting training-relevant web content
3. **Brain Knowledge** — pi.ruv.io brain with 3,870+ memories and 4.7M+ graph edges of accumulated knowledge
4. **v2.1.0 SOTA modules** — FlashAttention-3, Graph RAG, MLA, Mamba SSM, DiskANN, ColBERT, OPQ

### Available GCloud Infrastructure

| Resource | Details |
|----------|---------|
| **Project** | `ruv-dev` |
| **Billing** | `Generative Ai` account (active) |
| **GPUs** | GB200 (192GB), B200 (180GB), H200 (141GB), H100 (80GB), A100 (80GB/40GB), L4 (24GB), T4 (16GB) |
| **TPUs** | v3, v3p, v5l, v5lp, v5p, v6e |
| **Existing finetuning service** | `phi4-finetuning-gpu` — Cloud Run, L4 GPU, 8 CPU, 32GB RAM, HF_TOKEN configured |
| **Scheduler** | 21 active jobs including `brain-train` (every 5min), `brain-transfer` (30min), `brain-wet-daily` |
| **Secrets** | HuggingFace token, Anthropic key, Google AI key, brain keys |
| **Artifact Registry** | `ruvector` Docker repo in us-central1 |
| **Vertex AI** | Enabled, no current jobs |

### Current Model Artifacts

| Model | Parameters | Quants | Downloads | Status |
|-------|-----------|--------|-----------|--------|
| ruvltra-claude-code | Fine-tuned | Q4K, Q5K, Q8, imatrix | 7,615 | Production |
| ruvltra | 0.5B | Q4K, Q5K, Q8, FP16 | 560 | Production |
| ruvltra-medium | 3B | Q4K, Q5K, Q8, FP16 | 74 | Production |
| ruvltra-small | 494M | Q4K, Q5K, Q8, FP16 | 32 | Production |

## Decision

### Phase 1: imatrix Recalibration + TurboQuant KV Profiling (Week 1)

**Goal**: Produce improved GGUF quantizations with code-focused imatrix calibration, and generate TurboQuant KV-cache configuration profiles per model.

**Important**: TurboQuant operates at **runtime on KV-cache and embeddings** — it is not a weight quantization format. It is data-oblivious (no training, no codebooks). The optimization here is:
1. Better imatrix calibration → better base GGUF quantizations
2. Per-model TurboQuant KV profiles → optimal bit-width per attention layer at runtime

#### 1.1 imatrix Recalibration

Generate new importance matrices using RuvLTRA-specific calibration data:
- Code generation tasks (HumanEval, MBPP)
- Agent routing examples (Claude Flow dataset, 2,700+ examples)
- Claude Code instruction-following (ADR corpus, v2.1.0 code)

Produce updated GGUF variants with code-optimized imatrix:

| Variant | Format | Size (3B) | Use Case |
|---------|--------|-----------|----------|
| `Q4_K_M` (recalibrated) | Standard GGUF | ~2.1 GB | **Default — production** |
| `Q5_K_M` (recalibrated) | Standard GGUF | ~2.5 GB | Higher quality |
| `Q8_0` (recalibrated) | Standard GGUF | ~3.5 GB | Quality-first |
| `Q2_K` (recalibrated) | Standard GGUF | ~1.0 GB | Edge/mobile |

#### 1.2 TurboQuant KV-Cache Profiling

Profile each model's attention patterns to determine optimal per-layer TurboQuant configuration:

```bash
# Cloud Run Job: imatrix + TurboQuant profiling
gcloud run jobs create ruvltra-calibration \
  --image=gcr.io/ruv-dev/ruvltra-training:latest \
  --cpu=8 --memory=32Gi --gpu=1 --gpu-type=nvidia-l4 \
  --region=us-central1 \
  --set-secrets=HF_TOKEN=huggingface-token:latest \
  --max-retries=1 --task-timeout=3600s \
  --command="python3,calibrate_and_profile.py"
```

**Outputs**:
1. New imatrix files for each model
2. TurboQuant runtime config: recommended bits per layer, eviction policy, QJL settings
3. Perplexity delta report: standard KV vs TQ3 vs TQ4 per layer

### Phase 2: WET-Augmented LoRA Fine-Tuning (Week 2-3)

**Goal**: LoRA fine-tune RuvLTRA models on curated data from brain knowledge + WET (Common Crawl WARC/WET extraction) processing + new v2.1.0 documentation.

**Note**: Full pre-training is not in scope. The existing Rust training infrastructure supports LoRA adapters (rank 2-32) and embedding fine-tuning. For full SFT/DPO, we use Python (transformers + trl + peft) on Vertex AI.

#### 2.1 Training Data Sources

| Source | Records | Content | Pipeline |
|--------|---------|---------|----------|
| **Brain memories** | 3,870+ | Architecture patterns, solutions, conventions, debug knowledge | `pi.ruv.io/v1/memories/list` |
| **WET extraction** | ~50K pages | Rust/ML/vector-DB documentation from Common Crawl | `brain-wet-daily` scheduler |
| **Claude Flow routing** | 2,700+ | Claude-style training examples (existing HF dataset) | `ruvnet/claude-flow-routing` |
| **v2.1.0 code** | 8,577 lines | TurboQuant, Graph RAG, FlashAttention-3, DiskANN implementations | Git history |
| **ADR corpus** | 129 docs | Architectural decisions with rationale | `docs/adr/` |

#### 2.2 Dataset Governance

**Record schema** — every training record must contain:

```json
{
  "id": "uuid",
  "source": "brain|wet|claude-routing|code|adr",
  "text": "...",
  "license": "apache-2.0|mit|cc-by-4.0|public-domain",
  "quality_score": 0.0-1.0,
  "provenance": "url or commit hash",
  "created_at": "ISO-8601",
  "content_hash": "sha256"
}
```

**Source allowlist**: Only these sources may contribute training data:

| Source | License | Allowed |
|--------|---------|---------|
| Brain memories (pi.ruv.io) | Apache-2.0 (project-owned) | Yes |
| Common Crawl WET | CC-BY-4.0 / robots.txt compliant | Yes, with content filter |
| Claude Flow routing dataset | Apache-2.0 (HF: `ruvnet/claude-flow-routing`) | Yes |
| RuVector source code | MIT (project-owned) | Yes |
| ADR corpus | MIT (project-owned) | Yes |
| HumanEval / MBPP | MIT | Eval only — never train |
| SWE-Bench | MIT | Eval only — never train |

**Deduplication**: Content-hash dedup (SHA-256) at record level. Fuzzy dedup via MinHash (Jaccard > 0.8 = duplicate). Run before any train/eval split.

**Eval contamination check**: After dedup, compute 13-gram overlap between training set and all eval sets (HumanEval, MBPP, SWE-Bench Lite, routing eval). Any record with >50% 13-gram overlap with an eval instance is removed from training and flagged. Contamination report is a required Phase 3 artifact.

**Quality scoring**: Each record gets a quality score (0.0-1.0):
- Brain memories: score = memory confidence (from brain API)
- WET pages: score = content relevance classifier (>0.6 to include)
- Claude routing: score = 1.0 (curated dataset)
- Code: score = 1.0 (project-owned, reviewed)
- ADRs: score = 1.0 (project-owned)

Records with quality_score < 0.5 are excluded. Final corpus statistics (count, token count, source distribution, quality histogram) are logged as a Phase 2 artifact.

#### 2.3 Data Processing Pipeline

```
WET segments → CommonCrawlAdapter → Dedup (bloom) → Content filter
                                                          ↓
Brain memories → /v1/memories/search → Category filter → Merge
                                                          ↓
Claude dataset → HF download → Format validation → Unified corpus
                                                          ↓
                                                    SFT/DPO split
                                                    (80/20 train/eval)
```

#### 2.4 Training Configuration

**Infrastructure**:
- **Phase 2a (SFT)**: Vertex AI Custom Job, 1x A100-80GB, 4-8 hours
- **Phase 2b (DPO)**: Vertex AI Custom Job, 1x A100-80GB, 2-4 hours
- **Estimated cost**: ~$30-50 per full training run (A100 at $3.67/hr)

**Hyperparameters (SFT)**:

| Parameter | RuvLTRA-Small (0.5B) | RuvLTRA-Medium (3B) |
|-----------|---------------------|---------------------|
| Learning rate | 2e-5 | 1e-5 |
| Batch size | 16 | 8 |
| Epochs | 3 | 2 |
| LoRA rank | 16 | 32 |
| LoRA alpha | 32 | 64 |
| LoRA targets | Q,K,V,O,Gate,Up | Q,K,V,O,Gate,Up |
| Max seq length | 4096 | 8192 |
| Warmup ratio | 0.05 | 0.03 |
| Weight decay | 0.01 | 0.01 |
| Gradient checkpointing | Yes | Yes |

**Hyperparameters (DPO)**:

| Parameter | Value |
|-----------|-------|
| Beta | 0.1 |
| Learning rate | 5e-6 |
| Epochs | 1 |
| Max prompt length | 1024 |
| Max completion length | 2048 |

### Phase 3: Benchmarking & Validation (Week 3-4)

#### 3.1 Benchmark Suite

| Benchmark | Metric | Current Baseline | Target |
|-----------|--------|-----------------|--------|
| **Code generation** | pass@1 on HumanEval | TBD | >50% (0.5B), >65% (3B) |
| **Agent routing** | Accuracy on routing dataset | 80% | >85% |
| **TurboQuant quality** | Perplexity degradation | N/A | <0.5% at 4-bit, <1% at 3-bit |
| **Inference speed** | tok/s on M4 Pro | 88-135 | >100 (0.5B), >60 (3B) |
| **Memory** | Peak VRAM with TQ3 KV | N/A | <2GB (0.5B), <4GB (3B) |
| **Long context** | Perplexity at 32K tokens | N/A | <15 PPL (3B with TQ3) |
| **SWE-Bench Lite** | Resolution rate | TBD | >10% (0.5B), >20% (3B) |

#### 3.2 Release Gate — Ship/No-Ship Criteria

A model version is **approved for publishing** only if ALL of the following pass:

| Gate | Criterion | Measurement |
|------|-----------|-------------|
| **G1: Code quality** | HumanEval pass@1 improves by ≥5 percentage points over current baseline, or ≥45% (0.5B) / ≥55% (3B) absolute | `eval_humaneval.py` |
| **G2: Routing no-regression** | Agent routing accuracy ≥ current baseline (80%). Must not regress. | `eval_routing.py` on held-out routing eval set |
| **G3: General no-regression** | Perplexity on Wikitext-2 does not increase by >5% vs current baseline | `eval_perplexity.py` |
| **G4: TurboQuant memory** | TQ3 KV compression ≥ 8x with perplexity delta < 1% | `turbo_quant_bench` |
| **G5: Long context** | Perplexity at 16K tokens (3B model) < 20 PPL with TQ3 | `eval_long_context.py` |
| **G6: Contamination** | Zero eval contamination detected (13-gram check passes) | Phase 2 contamination report |
| **G7: Inference speed** | tok/s ≥ 80 (0.5B) / ≥ 40 (3B) on reference hardware (M4 Pro or L4 GPU) | `e2e_bench` |

If any gate fails, the model is **not published**. The team must either fix the issue and re-run, or document the regression and get explicit approval to ship with a known deficit.

**Gate evaluation is automated** via `scripts/release_gate.py` which runs all checks and produces a single PASS/FAIL verdict with per-gate details.

#### 3.3 Ablation Matrix

Each improvement must be measured in isolation to attribute impact:

| Run | imatrix Recal | LoRA SFT | DPO | TQ Runtime | Purpose |
|-----|:---:|:---:|:---:|:---:|---------|
| A (baseline) | | | | | Current published model |
| B | ✓ | | | | Isolate imatrix improvement |
| C | ✓ | ✓ | | | Isolate SFT impact |
| D | ✓ | ✓ | ✓ | | Isolate DPO impact |
| E | ✓ | ✓ | ✓ | ✓ | Full pipeline (ship candidate) |

Each run is evaluated on the same held-out eval set. Results are recorded in `reports/ablation-{date}.json`. The ablation report is a required Phase 3 artifact.

#### 3.4 TurboQuant-Specific Benchmarks

```rust
// Run from crates/ruvllm
cargo bench --bench turbo_quant_bench

// Benchmarks included:
// - compress_batch/128d, 256d, 512d, 1024d
// - decompress_batch
// - inner_product_asymmetric vs inner_product_asymmetric_optimized
// - kv_cache_tier push/get throughput
// - embedding_store search latency
```

| Benchmark | Expected Result |
|-----------|----------------|
| Compress 1M KV vectors (128d, 3-bit) | <500ms |
| Asymmetric inner product (batch 1000) | <1ms |
| KV-cache tier push (per entry) | <10µs |
| Embedding store search (10K vectors, top-10) | <5ms |

#### 3.5 Automated Benchmark Pipeline

```yaml
# Cloud Scheduler: weekly benchmark
gcloud scheduler jobs create http ruvltra-benchmark-weekly \
  --schedule="0 6 * * 1" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/ruv-dev/jobs/ruvltra-benchmark:run" \
  --location=us-central1
```

### Phase 4: Publishing (Week 4)

#### 4.1 Model Publishing Pipeline

```
Train → Merge LoRA → Convert GGUF → TurboQuant calibrate → Benchmark
                                           ↓
                         Q4_K_M, Q5_K_M, Q8_0 (standard)
                         Q4_K_M-TQ3, Q4_K_M-TQ4 (TurboQuant-optimized)
                                           ↓
                                    Upload to HuggingFace
                                    Update model cards
                                    Notify via Resend email
```

#### 4.2 Model Card Updates

Each model card will include:
- TurboQuant benchmark results (compression ratio, perplexity delta)
- Training data sources and sizes
- SWE-Bench and HumanEval scores
- Recommended `ruvllm` configuration
- Memory footprint comparison (standard vs TurboQuant KV)

#### 4.3 Versioning

| Model | Current | After Training |
|-------|---------|---------------|
| ruvltra-claude-code | v1.0 | v2.0-tq |
| ruvltra | v1.0 | v2.0-tq |
| ruvltra-medium | v1.0 | v2.0-tq |
| ruvltra-small | v1.0 | v2.0-tq |

## Nightly Continuous Learning Loop

Beyond the initial 4-phase training, a nightly pipeline continuously improves the models using fresh brain learnings from pi.ruv.io.

### Schedule

| Job | Schedule | What It Does |
|-----|----------|-------------|
| `brain-train` | Every 5 min | Brain memory optimization (existing) |
| `brain-wet-daily` | Daily 05:00 UTC | Common Crawl WET extraction (existing) |
| `ruvltra-nightly-train` | Daily 03:00 UTC | **NEW** — incremental LoRA from brain learnings → validate → push to HF |
| `ruvltra-benchmark-weekly` | Monday 06:00 UTC | Automated benchmark + release gate check |

### Nightly Pipeline Flow

```
03:00 UTC — ruvltra-nightly-train fires
     │
     ├─ [1] Export brain learnings (last 24h) + ADR corpus
     │     └─ Skip if < 10 records
     ├─ [2] Contamination check (13-gram)
     ├─ [3] Incremental LoRA training (rank-8, 1 epoch, lr=1e-5)
     ├─ [4] Release gate check (G1-G7)
     │     └─ Block publishing if any gate fails
     └─ [5] Push to HuggingFace (only if gates pass)
```

### Safety

- **Minimum data threshold**: Skips if < 10 records (prevents training on noise)
- **Release gates**: All 7 gates must pass before publishing
- **Incremental only**: Rank-8 LoRA, 1 epoch — small updates, not full retraining
- **7-day retention**: Old runs auto-cleaned
- **Daily cost**: ~$4 (L4 GPU × ~2hr, only on days with sufficient data)
- **Monthly cost**: ~$60-90

### Implementation

- Script: `scripts/training/nightly_train.sh`
- Cloud Run Job: `ruvltra-nightly-train` (L4 GPU, 8 CPU, 32GB RAM, 2hr timeout)
- Deployed via: `scripts/training/deploy_training.sh` (Step 6-7)

## Rollback Plan

If fine-tuning degrades model quality (any release gate fails after publishing):

1. **Immediate**: Revert HuggingFace model files to previous commit. HF supports git-based rollback:
   ```bash
   # Revert to previous version
   git -C /path/to/hf-clone revert HEAD
   huggingface-cli upload ruv/ruvltra-medium . --commit-message "Rollback: v2.0-tq regressed on G3"
   ```

2. **Model versioning**: All GGUF files are tagged with `v1.0` (current) and `v2.0-tq` (new). Both versions remain downloadable. The `latest` tag only moves to `v2.0-tq` after all gates pass. Users pinning `v1.0` are unaffected.

3. **Registry rollback**: Update `crates/ruvllm/src/hub/registry.rs` to point default downloads back to v1.0 GGUF files. Publish `ruvllm` patch release.

4. **npm rollback**: Publish `@ruvector/ruvllm` patch with reverted model defaults and updated README noting the rollback.

5. **Post-mortem**: File a GitHub issue documenting which gate failed, root cause, and what changes are needed before the next attempt.

## TurboQuant Serving Plan

TurboQuant is a runtime technique, not a GGUF artifact. The serving integration works as follows:

### Config Discovery

```
model.gguf (standard GGUF, unchanged)
model.turboquant.json (TurboQuant runtime config, new file)
```

The `.turboquant.json` file is a sidecar published alongside the GGUF on HuggingFace:

```json
{
  "version": 1,
  "default_bits": "3.5",
  "default_eviction": "h2o",
  "use_qjl": true,
  "per_layer_config": {
    "layer_0": {"bits": "4.0", "reason": "high entropy, needs precision"},
    "layer_1": {"bits": "3.5"},
    "layer_23": {"bits": "3.0", "reason": "low entropy, safe to compress"}
  }
}
```

### Runtime Loading in RuvLLM

`ruvllm` loads TurboQuant config in this order:
1. User-provided `TurboQuantConfig` (explicit override)
2. `.turboquant.json` sidecar file next to the GGUF (auto-discovered)
3. Default config (3.5-bit, H2O eviction, QJL enabled)

This requires a small addition to the model loading path in `crates/ruvllm/src/gguf/`. The sidecar is optional — models without it use the default config.

### Implementation Steps

1. Add `TurboQuantProfile` struct to `crates/ruvllm/src/quantize/turbo_quant.rs`
2. Add sidecar loading to `GgufLoader` in `crates/ruvllm/src/gguf/`
3. Generate `.turboquant.json` files during Phase 1 profiling
4. Upload sidecar files alongside GGUF to HuggingFace
5. Update `ModelDownloader` to also fetch the sidecar

## Cost Estimate

**Note**: This covers initial experimental compute only (happy-path GPU time). It does not include failed runs, repeated calibration, data cleaning retries, GCS storage, network egress, or engineer time. Budget 2-3x for realistic end-to-end cost.

| Phase | Resource | Duration | Cost |
|-------|----------|----------|------|
| TurboQuant calibration | L4 GPU (Cloud Run) | 2 hours | ~$4 |
| SFT training (0.5B) | A100-80GB (Vertex AI) | 4 hours | ~$15 |
| SFT training (3B) | A100-80GB (Vertex AI) | 8 hours | ~$30 |
| DPO training (both) | A100-80GB (Vertex AI) | 4 hours | ~$15 |
| GGUF conversion | L4 GPU (Cloud Run) | 1 hour | ~$2 |
| Benchmarking | L4 GPU (Cloud Run) | 2 hours | ~$4 |
| **Total** | | **~21 hours** | **~$70** |

Weekly benchmark runs add ~$4/week (~$16/month).

## Current Gaps Identified

| Gap | Description | Resolution |
|-----|-------------|------------|
| **No GPU compute provisioned** | All GCloud is CPU-only Cloud Run except `phi4-finetuning-gpu` (L4) | Phase 1-2 provision GPU via Cloud Run Jobs and Vertex AI |
| **TurboQuant has no GGUF format** | TurboQuant is runtime-only (KV-cache/embeddings), no GGUF serialization | Ship TQ runtime configs alongside standard GGUF files |
| **Model checksums not set** | Registry `checksum` fields are `None` for all models | Compute and set SHA256 during Phase 4 publishing |
| **WET pipeline is brain-only** | `CommonCrawlAdapter` feeds brain memories, not model training | Export WET-processed content as training corpus in Phase 2 |
| **No full-model fine-tuning in Rust** | Rust training covers LoRA/embedding-level only | Use Python (transformers + peft) on Vertex AI for SFT/DPO |
| **WASM Pi-Quant incomplete** | ADR-090 Phase 4 (PiQ WASM export) listed as "In Progress" | Track separately, not blocking this ADR |

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Catastrophic forgetting during SFT | Model loses general ability | EWC++ regularization (SONA integration), eval after each epoch |
| WET data quality | Noisy training data degrades model | Content filtering, dedup, quality scoring before inclusion |
| TurboQuant calibration mismatch | Quantized model quality drops | A/B test against standard quantization on eval set |
| GPU quota limits | Training job fails | Use preemptible instances, retry logic, L4 fallback |
| HuggingFace token scope | Upload fails | Verify write scope before training pipeline starts |

## Tooling Decision: ruvllm-native over llama.cpp

The calibration and quantization pipeline uses **ruvllm-native tooling** rather than compiling llama.cpp from source:

| Component | Previous (llama.cpp) | Current (ruvllm-native) |
|-----------|---------------------|------------------------|
| GGUF quantization | `llama-quantize` binary (requires CUDA source build) | `RuvltraQuantizer` (Rust) + `gguf` Python package |
| Model loading | `llama-imatrix` binary | `llama-cpp-python` (prebuilt CUDA wheel) |
| KV-cache profiling | N/A | `TurboQuantCompressor` + `TurboQuantProfile` (Rust) |
| Docker build time | 20+ min (CUDA kernel compilation) | ~5 min (pip install prebuilt wheels) |

**Rationale**: Building llama.cpp from source requires CUDA toolkit compilation of Flash Attention kernels (~100 CUDA template files), which exceeds Cloud Build timeout limits. The ruvllm ecosystem already provides all needed quantization, profiling, and inference capabilities via `RuvltraQuantizer`, `TurboQuantCompressor`, and the `gguf` Python library.

## Alternatives Considered

1. **Vertex AI Model Garden**: Pre-built fine-tuning pipelines, but no TurboQuant integration and limited model architecture support.
2. **GKE with GPU node pool**: More flexible but higher operational complexity. Cloud Run jobs are simpler for batch workloads.
3. **TPU training**: Better cost/perf for large models, but RuvLTRA models (0.5B-3B) are small enough that A100 is sufficient and simpler.
4. **External training providers** (Lambda, RunPod): Cheaper GPU hours but no integration with existing GCloud secrets, scheduler, and Artifact Registry.
5. **llama.cpp source build**: Full CUDA compilation for `llama-imatrix` and `llama-quantize`. Rejected due to 20+ minute build times that exceed Cloud Build timeout, and redundancy with existing ruvllm tooling.

## Next Steps

### Pre-Approval (before marking Accepted)
1. [ ] Review and finalize dataset governance rules (Section 2.2)
2. [ ] Confirm release gate thresholds with stakeholders (Section 3.2)
3. [ ] Validate HuggingFace token has write scope for all 4 model repos

### Phase 1 (Week 1)
4. [ ] Build `gcr.io/ruv-dev/ruvltra-training:latest` Docker image
5. [ ] Run imatrix recalibration with code-focused calibration data
6. [ ] Generate TurboQuant per-layer profiles and `.turboquant.json` sidecars
7. [ ] Benchmark recalibrated GGUFs against baseline (ablation run B)

### Phase 2 (Week 2-3)
8. [ ] Export brain memories and WET-processed data as training corpus
9. [ ] Run eval contamination check on training corpus
10. [ ] Create Vertex AI custom training job template
11. [ ] Run SFT training (ablation run C)
12. [ ] Run DPO training (ablation run D)

### Phase 3 (Week 3-4)
13. [ ] Run full ablation matrix (runs A-E)
14. [ ] Evaluate all release gates (G1-G7)
15. [ ] Produce contamination report and ablation report
16. [ ] Implement `scripts/release_gate.py` automation

### Phase 4 (Week 4)
17. [ ] Produce new GGUF variants + sidecar configs
18. [ ] Publish to HuggingFace with updated model cards
19. [ ] Update `ruvllm` model registry with checksums
20. [ ] Set up weekly benchmark scheduler job
21. [ ] Publish `ruvllm` and `@ruvector/ruvllm` with sidecar loading support

## Existing Training Infrastructure

| Component | Location | What It Does |
|-----------|----------|--------------|
| MicroLoRA training | `crates/ruvllm/src/lora/training.rs` | Per-request LoRA with EWC++ regularization |
| Adapter trainer | `crates/ruvllm/src/lora/adapters/trainer.rs` | Synthetic Claude dataset training |
| Pretrain pipeline | `crates/ruvllm/src/claude_flow/pretrain_pipeline.rs` | 4-phase: Bootstrap, Synthetic, Reinforce, Consolidate |
| TS training | `npm/packages/ruvllm/src/training.ts` | Full pipeline with LR scheduling, early stopping, EWC |
| Contrastive fine-tune | `npm/packages/ruvllm/scripts/training/contrastive-finetune.js` | Triplet loss router training |
| Brain LoRA training | `scripts/train-lora.py` | Federated LoRA with Byzantine-tolerant aggregation |
| 15-agent swarm | `scripts/swarm_train_15.sh` | Parallel discovery + training from 15 data sources |
| Weight quantization | `crates/ruvllm/src/quantize/ruvltra_quant.rs` | Q4_K_M, Q5_K_M, Q8_0, PiQ3, PiQ2 GGUF export |
| TurboQuant (runtime) | `crates/ruvllm/src/quantize/turbo_quant.rs` | 2-4 bit KV-cache/embedding compression |
| Benchmarks | `crates/ruvllm/benches/` | 13 benchmark files covering all subsystems |

## References

- [TurboQuant implementation](../../crates/ruvllm/src/quantize/turbo_quant.rs)
- [KV-Cache management](../../crates/ruvllm/src/kv_cache.rs)
- [WET processing pipeline](../../crates/mcp-brain-server/src/pipeline.rs)
- [ADR-128: SOTA Gap Implementations](./ADR-128-sota-gap-implementations.md)
- [v2.1.0 Release](https://github.com/ruvnet/RuVector/releases/tag/v2.1.0)
- [phi4-finetuning-gpu service](https://console.cloud.google.com/run/detail/us-central1/phi4-finetuning-gpu/revisions?project=ruv-dev) — existing template
- [ADR-049: Verified Training Pipeline](./ADR-049-verified-training-pipeline.md)
- [ADR-090: Ultra-Low-Bit QAT & Pi-Quantization](./ADR-090-ultra-low-bit-qat-pi-quantization.md)
- [ADR-093: Daily Discovery Brain Training](./ADR-093-daily-discovery-brain-training.md)
- [Federated LoRA training script](../../scripts/train-lora.py)
- [15-agent swarm training](../../scripts/swarm_train_15.sh)
- [RuvLTRA model registry](../../crates/ruvllm/src/hub/registry.rs)
