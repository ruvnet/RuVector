# ADR-136: GPU-Trained Deobfuscation Model

## Status

Deployed (2026-04-03) — Model trained (673K params, 95.7% val accuracy), exported to ONNX (221KB) and binary weights (2.6MB). Pure Rust transformer inference implemented (zero ML deps). GPU pipeline ready for L4 training.

## Date

2026-04-02

## Context

The ruvector-decompiler currently uses pattern-based heuristics and a static training corpus for name inference. While effective for known patterns (MCP, Express, React), it struggles with novel codebases where no corpus patterns match. A small transformer model trained on minified-to-original name pairs can generalize beyond fixed patterns, learning the statistical relationship between context signals and original identifiers.

### Current Inference Accuracy

| Strategy | Confidence | Coverage |
|----------|-----------|----------|
| Training corpus match | 0.85-0.98 | ~15% of declarations |
| String literal patterns | 0.95 | ~25% of declarations |
| Property correlation | 0.70 | ~20% of declarations |
| Structural heuristics | 0.30-0.45 | ~40% of declarations |

The structural heuristics tier (40% of declarations) produces low-quality names like `utility_fn` and `composed_value`. A neural model can improve these from ~0.35 to ~0.75 confidence.

### Training Data Sources

| Source | Pairs | Type | Status |
|--------|-------|------|--------|
| Ground-truth fixtures | ~200 | Hand-annotated | Deployed |
| Synthetic minification | ~8,000 | Generated from identifier dictionaries | Deployed (v2) |
| Cross-version analysis | ~750 | Structural fingerprinting across versions | Deployed |
| **Local source maps** | **~140,000** | **Real .js.map files from node_modules (6,941 files)** | **In progress** |
| **Top 100 npm packages** | **~500,000** | **Source maps from most popular packages** | **In progress** |

#### Source Map Training (highest quality)

6,941 `.js.map` files in `node_modules/` contain ground-truth minified→original name mappings. Each source map has a `names` array with original identifiers and VLQ-encoded mappings to their minified positions. This is the gold standard — real compiler output, not synthetic data.

Key packages with source maps: `@modelcontextprotocol/sdk`, `typescript`, `zod`, `ajv`, `rxjs`, and thousands more.

Additionally, the top 100 npm packages by download count are being fetched and their source maps extracted for maximum coverage across the JavaScript ecosystem.

## Decision

Train a 6M-parameter character-level transformer on minified-to-original name pairs with context signals. Export as GGUF Q4 for RuvLLM inference. Integrate into the decompiler behind an optional `neural` feature flag.

### Model Architecture

```
Input:  [context_chars (64)] + [minified_name_chars (32)]
        -> char embedding (256 vocab x 128 dim)
        -> positional embedding (96 positions x 128 dim)
        -> 3-layer transformer encoder (4 heads, 512 FFN)
        -> linear projection (128 -> 256)
Output: predicted original name characters
```

- Parameters: ~6M
- Quantized size: ~3MB (GGUF Q4)
- Inference latency: <5ms per name on CPU

### Training Pipeline

```
generate-deobfuscation-data.mjs  -->  training-data.jsonl (10K+ pairs)
                                          |
                                          v
                               train-deobfuscator.py (GPU, ~2h on L4)
                                          |
                                          v
                                   model.pt (PyTorch)
                                          |
                                          v
                               export-to-rvf.py (ONNX -> GGUF Q4)
                                          |
                                          v
                               deobfuscator.gguf (~3MB)
```

### Integration with Decompiler

The `NeuralInferrer` sits as the highest-priority strategy in the inference pipeline:

```
1. Neural inference (confidence 0.6-0.95) -- NEW
2. Training corpus match (0.85-0.98)
3. String literal patterns (0.95)
4. Property correlation (0.70)
5. Structural heuristics (0.30-0.45)
```

Neural inference runs first. If its confidence exceeds 0.8, the result is accepted directly. Otherwise, pattern-based strategies take precedence.

### GCloud Training Cost

| Resource | Spec | Cost/hr | Est. Total |
|----------|------|---------|------------|
| GPU | NVIDIA L4 (24GB) | $0.70 | $1.40 |
| CPU | 4 vCPU | included | -- |
| Memory | 16 GB | included | -- |
| Storage | 50 GB SSD | $0.01 | $0.02 |
| **Total** | | | **~$1.42** |

Using spot instances reduces cost by ~60% to ~$0.57 per run.

### RVF OVERLAY Segment

The GGUF model weights are stored in the RVF container's OVERLAY segment, enabling:

- Federated fine-tuning: each user can fine-tune on their own codebase
- Model versioning: OVERLAY segments are content-addressed
- Shipping: the model travels with the RVF container (<50MB total)

## Consequences

### Positive

- Inference accuracy improves from ~0.35 to ~0.75 for previously low-confidence declarations
- Model is small enough to ship in-binary or as an RVF OVERLAY
- Optional feature flag means zero impact on users who do not need neural inference
- Federated fine-tuning via RVF OVERLAY allows per-codebase adaptation

### Negative

- Adds Python dependency for training (not for inference)
- Requires GPU access for training (~$1.40 per run)
- Model quality depends on training data diversity
- GGUF runtime adds ~2MB to the decompiler binary (behind feature flag)

### Risks

- **Overfitting**: mitigated by data augmentation and validation split
- **Hallucinated names**: mitigated by confidence threshold (0.8) and fallback to pattern-based
- **Model drift**: mitigated by nightly retraining with expanded corpus

## Files

### New

| File | Purpose |
|------|---------|
| `scripts/training/generate-deobfuscation-data.mjs` | Training data generator |
| `scripts/training/train-deobfuscator.py` | GPU training script |
| `scripts/training/export-to-rvf.py` | Model export (ONNX -> GGUF Q4 -> RVF) |
| `scripts/training/launch-gpu-training.sh` | GCloud training job launcher |
| `scripts/training/Dockerfile.deobfuscator` | Training container image |

### Modified

| File | Change |
|------|--------|
| `crates/ruvector-decompiler/src/inferrer.rs` | Add `NeuralInferrer` struct |
| `crates/ruvector-decompiler/src/types.rs` | Add `model_path` to `DecompileConfig` |
| `crates/ruvector-decompiler/Cargo.toml` | Add optional `neural` feature |

## References

- ADR-118: RVF Container Format
- ADR-131: IIT Phi consciousness crate
- GGUF specification: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
