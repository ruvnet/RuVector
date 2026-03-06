# @ruvector/ruvllm-wasm

[![npm](https://img.shields.io/npm/v/@ruvector/ruvllm-wasm.svg)](https://www.npmjs.com/package/@ruvector/ruvllm-wasm)
[![License](https://img.shields.io/crates/l/ruvllm-wasm.svg)](https://github.com/ruvnet/ruvector/blob/main/LICENSE)

Browser-compatible LLM inference runtime with WebAssembly. Semantic routing, adaptive learning, KV cache management, and chat template formatting — directly in the browser, no server required.

## Features

- **KV Cache Management** — Two-tier cache (FP32 tail + u8 quantized store) for efficient token storage
- **Memory Pooling** — Arena allocator + buffer pool for minimal allocation overhead
- **Chat Templates** — Llama3, Mistral, Qwen, ChatML, Phi, Gemma format support
- **HNSW Semantic Router** — 150x faster pattern matching with bidirectional graph search
- **MicroLoRA** — Sub-millisecond model adaptation (rank 1-4)
- **SONA Instant Learning** — EMA quality tracking + adaptive rank adjustment
- **Web Workers** — Parallel inference with SharedArrayBuffer detection
- **Full TypeScript** — Complete `.d.ts` type definitions for all exports

## Install

```bash
npm install @ruvector/ruvllm-wasm
```

## Quick Start

```javascript
import init, {
  RuvLLMWasm,
  ChatTemplateWasm,
  ChatMessageWasm,
  HnswRouterWasm,
  healthCheck
} from '@ruvector/ruvllm-wasm';

// Initialize WASM module
await init();

// Verify module loaded
console.log(healthCheck()); // true

// Format chat conversations
const template = ChatTemplateWasm.llama3();
const messages = [
  ChatMessageWasm.system("You are a helpful assistant."),
  ChatMessageWasm.user("What is WebAssembly?"),
];
const prompt = template.format(messages);

// Semantic routing with HNSW
const router = new HnswRouterWasm(384, 1000);
router.addPattern(new Float32Array(384).fill(0.1), "coder", "code tasks");
const result = router.route(new Float32Array(384).fill(0.1));
console.log(result.name, result.score); // "coder", 1.0
```

## API

### Core Types

| Type | Description |
|------|-------------|
| `RuvLLMWasm` | Main inference engine with KV cache + buffer pool |
| `GenerateConfig` | Generation parameters (temperature, top_k, top_p, repetitionPenalty) |
| `KvCacheWasm` | Two-tier KV cache for token management |
| `InferenceArenaWasm` | O(1) bump allocator for inference temporaries |
| `BufferPoolWasm` | Pre-allocated buffer pool (1KB-256KB size classes) |

### Chat Templates

```javascript
// Auto-detect from model ID
const template = ChatTemplateWasm.detectFromModelId("meta-llama/Llama-3-8B");
// Or use directly
const template = ChatTemplateWasm.mistral();
const prompt = template.format([
  ChatMessageWasm.system("You are helpful."),
  ChatMessageWasm.user("Hello!"),
]);
```

Supported: `llama3()`, `mistral()`, `chatml()`, `phi()`, `gemma()`, `custom(name, pattern)`

### HNSW Semantic Router

```javascript
const router = new HnswRouterWasm(384, 1000); // dimensions, max_patterns
router.addPattern(embedding, "agent-name", "metadata");
const result = router.route(queryEmbedding);
console.log(result.name, result.score);

// Persistence
const json = router.toJson();
const restored = HnswRouterWasm.fromJson(json);
```

### MicroLoRA Adaptation

```javascript
const config = new MicroLoraConfigWasm();
config.rank = 2;
config.inFeatures = 384;
config.outFeatures = 384;

const lora = new MicroLoraWasm(config);
const adapted = lora.apply(inputVector);
lora.adapt(new AdaptFeedbackWasm(0.9)); // quality score
```

### SONA Instant Learning

```javascript
const config = new SonaConfigWasm();
config.hiddenDim = 384;
const sona = new SonaInstantWasm(config);

const result = sona.instantAdapt(inputVector, 0.85); // quality
console.log(result.applied, result.qualityEma);

sona.recordPattern(embedding, "agent", true); // success pattern
const suggestion = sona.suggestAction(queryEmbedding);
```

### Parallel Inference (Web Workers)

```javascript
import { ParallelInference, feature_summary } from '@ruvector/ruvllm-wasm';

console.log(feature_summary()); // browser capability report

const engine = await new ParallelInference(4); // 4 workers
const result = await engine.matmul(a, b, m, n, k);
engine.terminate();
```

## Build from Source

```bash
# Install prerequisites
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Release build (workaround for Rust 1.91 codegen bug)
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 CARGO_PROFILE_RELEASE_LTO=off \
  wasm-pack build crates/ruvllm-wasm --target web --scope ruvector --release

# Dev build
wasm-pack build crates/ruvllm-wasm --target web --scope ruvector --dev

# With WebGPU support
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=256 CARGO_PROFILE_RELEASE_LTO=off \
  wasm-pack build crates/ruvllm-wasm --target web --scope ruvector --release -- --features webgpu
```

## Browser Compatibility

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | 57+ | Full support |
| Edge | 79+ | Full support |
| Firefox | 52+ | Full support |
| Safari | 11+ | Full support |

Optional enhancements:
- **SharedArrayBuffer**: Requires `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`
- **WebGPU**: Available with `webgpu` feature flag (Chrome 113+)

## Size

~435 KB release WASM (~178 KB gzipped)

## Related

- [`@ruvector/ruvllm`](https://www.npmjs.com/package/@ruvector/ruvllm) — Node.js LLM orchestration
- [`ruvector`](https://www.npmjs.com/package/ruvector) — Full RuVector CLI + MCP tools
- [ADR-084](../../docs/adr/ADR-084-ruvllm-wasm-publish.md) — Build documentation and known limitations

## License

MIT
