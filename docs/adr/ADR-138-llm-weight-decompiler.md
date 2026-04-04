# ADR-138: LLM Model Weight Decompiler

## Status

Implemented (2026-04-03) -- GGUF and Safetensors format decompilation with architecture inference, tokenizer extraction, and witness chain provenance.

## Date

2026-04-03

## Context

The `ruvector-decompiler` crate (ADR-135) currently decompiles JavaScript bundles into reconstructed modules using MinCut graph partitioning. Model weight files (GGUF, Safetensors, ONNX) are a parallel "compiled artifact" problem: they bundle architecture decisions, tokenizer data, quantization choices, and layer topology into opaque binary blobs.

Researchers and engineers need to inspect model files without loading them into a runtime. Existing tools (llama.cpp's `gguf-py`, HuggingFace's `safetensors` library) provide basic inspection but lack:

- Unified cross-format analysis
- Architecture reconstruction from tensor shapes alone
- GQA/MQA detection from Q/K projection dimension ratios
- Cryptographic witness chains for model provenance
- Integration with the existing ruvector decompiler pipeline

### Prior Art

- `ruvllm` crate has a GGUF v3 parser at `crates/ruvllm/src/gguf/parser.rs`
- The parser handles header, metadata KV, and tensor info extraction
- However, `ruvllm` carries heavy dependencies (candle, tokenizers, ndarray, etc.) unsuitable for a lightweight decompiler

## Decision

Add a `model` feature flag to `ruvector-decompiler` that enables LLM weight file decompilation. The implementation:

1. **Self-contained GGUF parser** -- copies the minimal parsing logic from `ruvllm` to avoid heavy transitive dependencies (candle, tokenizers, HuggingFace hub, etc.)
2. **Safetensors parser** -- reads the JSON header (8-byte length prefix + JSON) without external dependencies
3. **Architecture inference** -- reconstructs model architecture from GGUF metadata keys and tensor shape analysis
4. **Tokenizer extraction** -- reads vocabulary from GGUF metadata arrays
5. **Quantization detection** -- maps GGUF quant types to human-readable format names with bits-per-weight calculation
6. **Witness chain** -- SHA3-256 Merkle chain over tensor metadata for provenance

### Architecture

```
npx ruvector decompile --model llama-3-8b.gguf
       |
       v
+----------------------------+
| model_decompiler.rs        |
|  - decompile_model()       |  auto-detect format
|  - decompile_gguf()        |  GGUF v2/v3
|  - decompile_safetensors() |  Safetensors JSON header
+----------------------------+
       |
       v
+----------------------------+
| Architecture Inference     |
|  - GGUF metadata keys      |  general.architecture, etc.
|  - Tensor shape analysis   |  embed [V,H], attn [H,H]
|  - GQA detection           |  Q vs K projection dims
+----------------------------+
       |
       v
+----------------------------+
| Output                     |
|  - ModelDecompileResult    |  structured result
|  - CLI pretty print        |  human-readable summary
|  - JSON output             |  machine-readable
|  - Witness chain           |  SHA3-256 provenance
+----------------------------+
```

### Module Structure

```
crates/ruvector-decompiler/
  src/
    model_decompiler.rs     # Top-level model decompilation API
    model_types.rs          # Domain types (ModelArchitecture, LayerInfo, etc.)
    model_gguf.rs           # Self-contained GGUF parser + architecture inference
    model_safetensors.rs    # Safetensors JSON header parser
    lib.rs                  # Updated with `model` feature gate
```

### Why Copy the GGUF Parser

The `ruvllm` crate's GGUF parser is self-contained (~470 lines) but `ruvllm` itself pulls in `candle-core`, `candle-nn`, `tokenizers`, `hf-hub`, `ndarray`, `tokio`, and platform-specific GPU dependencies. Adding `ruvllm` as a dependency would increase compile time by 2-5 minutes and bloat the binary. The parser logic is stable (GGUF v3 format is frozen) and unlikely to diverge.

## Consequences

### Positive

- Unified model inspection across GGUF and Safetensors formats
- Zero-dependency parsing (no runtime, no GPU libraries)
- Reuses existing witness chain infrastructure
- Feature-gated so JS decompiler users pay no cost
- CLI integration via existing `npx ruvector decompile` command

### Negative

- Duplicated GGUF parser code (must be kept in sync manually if format changes)
- No tensor data reading (metadata and shapes only -- sufficient for decompilation)
- ONNX support deferred (requires protobuf parsing)

### Risks

- GGUF v4 format changes would require updating both parsers
- Safetensors format is simple enough that format changes are unlikely

## References

- ADR-135: MinCut Decompiler with Witness Chains
- ADR-136: GPU-Trained Deobfuscation Model
- ADR-137: npm Decompiler CLI and MCP
- GGUF format spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Safetensors format: https://huggingface.co/docs/safetensors/
