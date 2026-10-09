# @ruvector/ruvllm

[![npm version](https://img.shields.io/npm/v/@ruvector/ruvllm.svg)](https://www.npmjs.com/package/@ruvector/ruvllm)
[![Downloads](https://img.shields.io/npm/dm/@ruvector/ruvllm)](https://www.npmjs.com/package/@ruvector/ruvllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/ruvnet/ruvector?style=social)](https://github.com/ruvnet/ruvector)

**Self-learning LLM runtime for Node.js** — GGUF inference, TurboQuant KV-cache compression (6-8x memory savings), SONA adaptive learning, FlashAttention, speculative decoding, and SIMD-optimized kernels. Built in Rust, runs everywhere.

> Inference at **88-135 tok/s** on M4 Pro | **<1ms** SONA adaptation | **6-8x** KV-cache compression via TurboQuant

## Installation

```bash
npm install @ruvector/ruvllm
```

## Quick Start

```typescript
import { RuvLLM, RuvLLMConfig } from '@ruvector/ruvllm';

// Initialize with default configuration
const llm = new RuvLLM();

// Or with custom configuration
const llm = new RuvLLM({ embeddingDim: 384, learningEnabled: true });

// Routing, memory and embeddings (no model needed)
const decision = llm.route('Explain quantum computing');
llm.addMemory('RuvLLM routes queries with FastGRNN', { source: 'docs' });
const hits = llm.searchMemory('how are queries routed?', 3);
const vec = llm.embed('Explain quantum computing');
```

## Text Generation (GGUF)

Text comes only from a model you load. `loadModel()` runs a local GGUF file
on the native candle backend (CPU; Metal on macOS builds with the `metal`
feature):

```typescript
const llm = new RuvLLM();
llm.loadModel('./qwen2.5-0.5b-instruct-q4_k_m.gguf', { maxContext: 4096 });

const out = llm.chat([{ role: 'user', content: 'What is the capital of France?' }], {
  maxTokens: 64,
  temperature: 0,
});
// { text: 'The capital of France is Paris.', promptTokens: 15,
//   completionTokens: 8, finishReason: 'stop' }
```

- **Architectures**: GGUF `llama`, `mistral`, `qwen2` (Qwen2.5) and `qwen3`.
  Any other architecture (phi, gemma, qwen3.5, ...) fails to load with an error
  naming it.
- **Tokenizer**: a `tokenizer.json` next to the GGUF, else the tokenizer
  embedded in the GGUF. The embedded one works for byte-level BPE (`gpt2`)
  vocabularies such as Qwen and Llama 3; GGUFs with a SentencePiece vocabulary
  (Llama 2, Mistral, TinyLlama) need the `tokenizer.json`, otherwise loading
  fails with `RUVLLM_TOKENIZER_MISSING`.
- **`chat()`** applies the model's chat template, read from the GGUF's
  embedded `tokenizer.chat_template` (ChatML for Qwen). A `qwen2`/`qwen3` GGUF
  whose template is not ChatML (e.g. DeepSeek-R1-Distill-Qwen) fails to load
  with `RUVLLM_MODEL_LOAD_FAILED` rather than run under the wrong format.
- **`generate()`** continues the prompt verbatim; **`query()`** routes, then
  answers as a single chat message.
- **Usage**: `promptTokens`/`completionTokens` are counted with the model's
  tokenizer; `finishReason` is `stop` (end-of-sequence token or a stop
  sequence) or `length` (`maxTokens` reached or the context is full).
- **Blocking**: loading and generation are synchronous and block the event
  loop; run them in a worker thread if that matters.

Without a loaded model, `generate()`, `query()`, `chat()` and
`generateDetailed()` throw `RUVLLM_NO_LANGUAGE_MODEL`; they never return text
that is not model output. For demos and tests, `new RuvLLM({ allowPlaceholder:
true })` (or `RUVLLM_ALLOW_MOCK=1`) makes `generate()`/`query()` return the
labelled `PLACEHOLDER_TEXT` instead, with a one-time warning.

### Native binary requirements

Model loading needs a native binary built with the `candle` feature;
`supportsModelLoading()` reports whether the installed one was.

| Native binary | `loadModel()` |
|---------------|---------------|
| Built from this source (`npm run build:native`) | Works |
| `@ruvector/ruvllm-<platform>` 3.x (built by the release workflow with `candle`) | Works |
| `@ruvector/ruvllm-<platform>` 2.0.x | Throws `RUVLLM_NATIVE_TOO_OLD`: they predate model loading |
| Built with `--features napi` only (no `candle`) | Throws `RUVLLM_NO_INFERENCE_BACKEND` |
| None for your platform | Throws `RUVLLM_NATIVE_UNAVAILABLE` |

To use a binary you built, point `RUVLLM_NATIVE_PATH` at it (it takes
precedence over the installed platform package):

```bash
npm run build:native   # writes ruvllm.<platform>.node (needs a Rust toolchain)
RUVLLM_NATIVE_PATH=$PWD/ruvllm.linux-x64-gnu.node node app.js
```

Until the 3.x platform packages are on npm, build the binary from source as
above, or use the ruvllm CLI (`ruvllm serve <model>`, an OpenAI-compatible
server that exits if the model fails to load).

| Error code | Meaning |
|------------|---------|
| `RUVLLM_NO_LANGUAGE_MODEL` | Generation was requested with no model loaded |
| `RUVLLM_NATIVE_UNAVAILABLE` | No native binary could be loaded |
| `RUVLLM_NATIVE_TOO_OLD` | The native binary predates model loading (2.0.x platform packages) |
| `RUVLLM_NO_INFERENCE_BACKEND` | The native binary was built without the `candle` backend |
| `RUVLLM_MODEL_NOT_FOUND` | The model path does not exist |
| `RUVLLM_MODEL_LOAD_FAILED` | The file could not be loaded (unsupported architecture, not GGUF, ...) |
| `RUVLLM_TOKENIZER_MISSING` | Weights loaded but no usable tokenizer was found |

## What's New in v2.5

| Feature | Description |
|---------|-------------|
| **TurboQuant KV-Cache** | 2-4 bit asymmetric quantization with per-channel scale/zero-point — 6-8x memory reduction, <0.5% perplexity loss |
| **TurboQuant Embedding Store** | Quantized vector storage with compressed search — 10-30x memory savings |
| **H2O / PyramidKV Eviction** | Intelligent cache eviction policies for long-context inference |
| **Optimized Inner Product** | Asymmetric distance on quantized data — skip decompression for 2-4x faster search |
| **RuvLTRA Models** | Purpose-built 0.5B & 3B models for Claude Flow |
| **Task-Specific LoRA** | 5 pre-trained adapters (coder, researcher, security, architect, reviewer) |
| **HuggingFace Hub** | Download/upload models directly |
| **Adapter Merging** | TIES, DARE, SLERP strategies |
| **HNSW Routing** | 150x faster semantic matching |
| **Evaluation Harness** | SWE-Bench testing with 5 ablation modes |
| **mistral-rs Backend** | Production serving with PagedAttention, X-LoRA, ISQ |

## TurboQuant — KV-Cache Compression

Reduce inference memory by 6-8x with <0.5% quality loss:

```typescript
import { simd } from '@ruvector/ruvllm/simd';

// TurboQuant compresses KV-cache entries at 2-4 bit precision
// with per-channel asymmetric quantization (scale + zero-point).
// Eviction policies (H2O, Sliding Window, PyramidKV) keep the
// most important tokens in cache during long-context generation.

// Supported bit widths: 2-bit (32x), 3-bit (10.7x), 4-bit (8x), 8-bit (4x)
```

| Bits | Compression | Perplexity Loss | Use Case |
|------|-------------|-----------------|----------|
| 2-bit | 32x | ~2% | Maximum compression, edge devices |
| 3-bit | 10.7x | <1% | Balanced — recommended for most uses |
| 4-bit | 8x | <0.5% | High quality, long-context inference |
| 8-bit | 4x | ~0% | Baseline quantization |

## CLI Usage

```bash
# Query a model (this package's `ruvllm` bin; needs --model, see above)
ruvllm query "What is machine learning?" --model ./qwen2.5-0.5b-instruct-q4_k_m.gguf

# The commands below are the Rust ruvllm CLI (crate ruvllm-cli)
# Stream output
ruvllm query --stream "Write a poem"

# Download a model
ruvllm download ruvector/ruvltra-small-q4km

# Benchmark
ruvllm bench ./models/model.gguf

# Run evaluation (SWE-Bench)
ruvllm eval --model ./models/model.gguf --subset lite --max-tasks 50
```

## API Reference

### RuvLLM Class

```typescript
class RuvLLM {
  constructor(config?: RuvLLMConfig);

  // Language model (see "Text Generation" above)
  loadModel(path: string, options?: LoadModelOptions): LoadedModelInfo;
  unloadModel(): void;
  isModelLoaded(): boolean;
  modelInfo(): LoadedModelInfo | null;
  supportsModelLoading(): boolean; // native binary can load models

  // Generation; all throw RUVLLM_NO_LANGUAGE_MODEL without a loaded model
  chat(messages: ChatMessage[], config?: GenerationConfig): GenerationResult;
  generateDetailed(prompt: string, config?: GenerationConfig): GenerationResult;
  generate(prompt: string, config?: GenerationConfig): string;
  query(text: string, config?: GenerationConfig): QueryResponse; // routed, + token usage

  // Routing, memory and embeddings
  route(text: string): RoutingDecision;
  addMemory(content: string, metadata?: Record<string, unknown>): MemoryId;
  searchMemory(text: string, k?: number): MemoryResult[];
  embed(text: string): Embedding;

  // Not in this package: token streaming (StreamingGenerator chunks a
  // finished generate() result), mistral-rs backends.

  // Get SONA learning stats
  sonaStats(): SonaStats | null;

  // Adapt on feedback
  adapt(input: Float32Array, quality: number): void;
}
```

### Configuration

```typescript
interface RuvLLMConfig {
  modelPath?: string;              // GGUF to load in the constructor (throws if it cannot be loaded)
  modelOptions?: LoadModelOptions; // { maxContext?: number } for modelPath
  allowPlaceholder?: boolean;      // generate()/query() return PLACEHOLDER_TEXT instead of
                                   // throwing when no model is loaded (default: false)
  strict?: boolean;                // Throw instead of warning for unsupported options (`backend`)
  // ...plus routing/memory options (embeddingDim, hnswM, ...), see types.ts
}
```

### Generation Config and Result

```typescript
interface GenerationConfig {
  maxTokens?: number;          // default 256
  temperature?: number;        // default 0.7; 0 = greedy
  topP?: number;               // default 0.9, in (0, 1]
  topK?: number;               // default 50; 0 disables
  repetitionPenalty?: number;  // default 1.1
  stopSequences?: string[];    // stop text is not included in the output
  seed?: number;               // reproducible sampling at temperature > 0
}

interface GenerationResult {
  text: string;
  promptTokens: number;        // after the chat template, for chat()
  completionTokens: number;    // includes an end-of-sequence token that ended generation
  finishReason: 'stop' | 'length' | 'cancelled';
}
```

## SIMD Module

For direct access to optimized SIMD kernels:

```typescript
import { simd } from '@ruvector/ruvllm/simd';

// Dot product
const result = simd.dotProduct(vecA, vecB);

// Matrix multiplication
const output = simd.matmul(matrix, vector);

// Flash Attention
const attended = simd.flashAttention(query, key, value, scale);

// RMS Normalization
simd.rmsNorm(hidden, weights, epsilon);
```

## Performance (M4 Pro)

| Operation | Performance |
|-----------|-------------|
| Inference | 88-135 tok/s |
| Flash Attention | 320µs (seq=2048) |
| HNSW Search | 17-62µs |
| SONA Adapt | <1ms |
| Evaluation | 5 ablation modes |

## Evaluation Harness

> **Not in this npm package**: `EvaluationHarness` and `AblationMode` are
> not exported by `@ruvector/ruvllm`. The sketch below shows the intended API.

Run model evaluations with SWE-Bench integration:

```typescript
import { RuvLLM, EvaluationHarness, AblationMode } from '@ruvector/ruvllm';

const harness = new EvaluationHarness({
  modelPath: './models/model.gguf',
  enableHnsw: true,
  enableSona: true,
});

// Run single evaluation
const result = await harness.evaluate(
  'Fix the null pointer exception',
  'def process(data): return data.split()',
  AblationMode.Full
);

console.log(`Success: ${result.success}, Quality: ${result.qualityScore}`);

// Run ablation study (Baseline, RetrievalOnly, AdaptersOnly, R+A, Full)
const report = await harness.runAblationStudy(tasks);
for (const [mode, metrics] of Object.entries(report.modeMetrics)) {
  console.log(`${mode}: ${metrics.successRate * 100}% success`);
}
```

## mistral-rs Backend (Production Serving)

For production deployments with 10-100+ concurrent users, use the mistral-rs backend:

```typescript
import { RuvLLM, MistralBackend, PagedAttentionConfig } from '@ruvector/ruvllm';

// Configure for production serving
const backend = new MistralBackend({
  // PagedAttention: 5-10x more concurrent users
  pagedAttention: {
    blockSize: 16,
    maxBlocks: 4096,
    gpuMemoryFraction: 0.9,
    prefixCaching: true,
  },
  // X-LoRA: Per-token adapter routing
  xlora: {
    adapters: ['./adapters/coder', './adapters/researcher'],
    topK: 2,
  },
  // ISQ: Runtime quantization
  isq: {
    bits: 4,
    method: 'awq',
  },
});

const llm = new RuvLLM({ backend });
await llm.loadModel('mistralai/Mistral-7B-Instruct-v0.2');

// Serve multiple concurrent requests
const response = await llm.query('Write production code');
```

> **Not in this npm package**: `MistralBackend` is not exported, the
> `backend` option is ignored (with a `RUVLLM_UNSUPPORTED_OPTION` warning), and
> `loadModel()` takes a local GGUF path, not a Hub id. mistral-rs serving is a
> feature of the Rust crate (`mistral-rs` feature).

## Supported Models

GGUF files loadable with `loadModel()` (architectures llama, mistral, qwen2, qwen3):

- **RuvLTRA-Small** (494M) - Q4K, Q5K, Q8
- **RuvLTRA-Medium** (3B) - Q4K, Q5K, Q8
- **Qwen 2.5 / Qwen 3** (0.5B-72B)
- **Llama 3.x** (8B-70B)
- **Mistral** (7B-22B)

Phi-3, Gemma-2 and Qwen3.5 GGUFs are not supported: loading fails with an
error naming the architecture.

## Platform Support

The table covers routing, memory and embeddings. Model loading additionally
needs a native binary built with the `candle` feature (see "Native binary
requirements").

| Platform | Architecture | Status |
|----------|--------------|--------|
| macOS | arm64 (M1-M4) | ✅ Full support |
| macOS | x64 | ✅ Supported |
| Linux | x64 | ✅ Supported |
| Linux | arm64 | ✅ Supported |
| Windows | x64 | ✅ Supported |

## Related Packages

- [@ruvector/core](https://www.npmjs.com/package/@ruvector/core) - Vector operations
- [@ruvector/sona](https://www.npmjs.com/package/@ruvector/sona) - SONA learning engine
- [@ruvector/ruvector](https://www.npmjs.com/package/@ruvector/ruvector) - Full Ruvector SDK

## Links

- [GitHub Repository](https://github.com/ruvnet/ruvector)
- [API Documentation](https://docs.rs/ruvllm)
- [Crate (Rust)](https://crates.io/crates/ruvllm)

## License

MIT OR Apache-2.0
