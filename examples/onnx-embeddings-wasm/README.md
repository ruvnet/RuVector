# RuVector ONNX Embeddings WASM

[![npm version](https://img.shields.io/npm/v/ruvector-onnx-embeddings-wasm.svg)](https://www.npmjs.com/package/ruvector-onnx-embeddings-wasm)
[![crates.io](https://img.shields.io/crates/v/ruvector-onnx-embeddings-wasm.svg)](https://crates.io/crates/ruvector-onnx-embeddings-wasm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-654FF0?logo=webassembly&logoColor=white)](https://webassembly.org/)
[![SIMD](https://img.shields.io/badge/SIMD-128bit-green)](https://webassembly.org/roadmap/)

> **Portable embedding generation with SIMD acceleration and parallel workers**

Generate text embeddings directly in browsers, Cloudflare Workers, Deno, Node.js, and any WASM runtime. Built with [Tract](https://github.com/sonos/tract) for pure Rust ONNX inference.

## Features

| Feature | Description |
|---------|-------------|
| 🌐 **Browser Support** | Generate embeddings client-side, no server needed |
| ⚡ **SIMD Acceleration** | WASM SIMD128 for vectorized operations |
| 🚀 **Parallel Workers** | Multi-threaded batch processing (3.8x speedup) |
| 🏢 **Edge Computing** | Deploy to Cloudflare Workers, Vercel Edge, Deno Deploy |
| 📦 **Zero Dependencies** | Single WASM binary, no native modules |
| 🤗 **HuggingFace Models** | Pre-configured URLs for popular models |
| 🔄 **Auto Caching** | Browser Cache API for instant reloads |
| 🎯 **Same API** | Compatible with native `ruvector-onnx-embeddings` |

## Installation

```bash
npm install ruvector-onnx-embeddings-wasm
```

The package works from both ES modules and CommonJS on Node.js 18+, and ships the
`--target web` glue for browsers and workers (`ruvector-onnx-embeddings-wasm/web`).
Versions 0.1.0-0.1.2 could not be loaded with either `import` or `require`; use 0.1.3 or later.

| Entry | Resolves to | Init |
|-------|-------------|------|
| `import` on Node | ESM wrapper over the CommonJS glue | ready on import |
| `require` on Node | CommonJS glue | ready on require |
| `/web`, browsers, bundlers | `--target web` ESM glue | `await init()` or `initSync({ module })` |
| `/wasm` | the `.wasm` binary | — |

## Quick Start

### Node.js core API (ESM or CommonJS, no model download)

```javascript
// const { version, cosineSimilarity, normalizeL2 } = require('ruvector-onnx-embeddings-wasm');
import { version, cosineSimilarity, normalizeL2 } from 'ruvector-onnx-embeddings-wasm';

console.log(version());
console.log(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([1, 0]))); // 1
console.log(normalizeL2(new Float32Array([3, 4]))); // Float32Array [0.6, 0.8]
```

### Node.js (Sequential)

```javascript
import { createEmbedder, similarity, embed } from 'ruvector-onnx-embeddings-wasm/loader';

// One-liner similarity
const score = await similarity("I love dogs", "I adore puppies");
console.log(score); // ~0.85

// One-liner embedding
const embedding = await embed("Hello world");
console.log(embedding.length); // 384

// Full control
const embedder = await createEmbedder('bge-small-en-v1.5');
const emb1 = embedder.embedOne("First text");
const emb2 = embedder.embedOne("Second text");
```

### Node.js (Parallel - 3.8x faster)

```javascript
import { ParallelEmbedder } from 'ruvector-onnx-embeddings-wasm/parallel';

// Initialize with worker threads
const embedder = new ParallelEmbedder({ numWorkers: 4 });
await embedder.init('all-MiniLM-L6-v2');

// Batch embed with parallel processing
const texts = [
  "Machine learning is transforming technology",
  "Deep learning uses neural networks",
  "Natural language processing understands text",
  "Computer vision analyzes images"
];
const embeddings = await embedder.embedBatch(texts);

// Compute similarity
const sim = await embedder.similarity("I love Rust", "Rust is great");
console.log(sim); // ~0.85

// Cleanup
await embedder.shutdown();
```

### Browser (ES Modules)

```html
<script type="module">
import init, { WasmEmbedder } from 'https://unpkg.com/ruvector-onnx-embeddings-wasm/ruvector_onnx_embeddings_wasm.js';
import { createEmbedder } from 'https://unpkg.com/ruvector-onnx-embeddings-wasm/loader.js';

// Initialize WASM
await init();

// Create embedder (downloads model automatically)
const embedder = await createEmbedder('all-MiniLM-L6-v2');

// Generate embeddings
const embedding = embedder.embedOne("Hello, world!");
console.log("Dimension:", embedding.length); // 384

// Compute similarity
const sim = embedder.similarity("I love Rust", "Rust is great");
console.log("Similarity:", sim.toFixed(4)); // ~0.85
</script>
```

### Cloudflare Workers

```javascript
import { initSync, WasmEmbedder } from 'ruvector-onnx-embeddings-wasm/web';
import wasmModule from 'ruvector-onnx-embeddings-wasm/wasm'; // Wrangler imports .wasm as a WebAssembly.Module

initSync({ module: wasmModule });

export default {
  async fetch(request, env) {
    // Load model from R2 or KV
    const modelBytes = await env.MODELS.get('model.onnx', 'arrayBuffer');
    const tokenizerJson = await env.MODELS.get('tokenizer.json', 'text');

    const embedder = new WasmEmbedder(
      new Uint8Array(modelBytes),
      tokenizerJson
    );

    const { text } = await request.json();
    const embedding = embedder.embedOne(text);

    return Response.json({
      embedding: Array.from(embedding),
      dimension: embedding.length
    });
  }
};
```

## Available Models

| Model | Dimension | Size | Speed | Quality | Best For |
|-------|-----------|------|-------|---------|----------|
| **all-MiniLM-L6-v2** ⭐ | 384 | 23MB | ⚡⚡⚡ | ⭐⭐⭐ | Default, fast |
| **all-MiniLM-L12-v2** | 384 | 33MB | ⚡⚡ | ⭐⭐⭐⭐ | Better quality |
| **bge-small-en-v1.5** | 384 | 33MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | State-of-the-art |
| **bge-base-en-v1.5** | 768 | 110MB | ⚡ | ⭐⭐⭐⭐⭐ | Best quality |
| **e5-small-v2** | 384 | 33MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Search/retrieval |
| **gte-small** | 384 | 33MB | ⚡⚡⚡ | ⭐⭐⭐⭐ | Multilingual |

## Performance

### Sequential vs Parallel (Node.js)

| Batch Size | Sequential | Parallel (4 workers) | Speedup |
|------------|------------|----------------------|---------|
| 4 texts | 1,573ms | 410ms | **3.83x** |
| 8 texts | 3,105ms | 861ms | **3.61x** |
| 12 texts | 4,667ms | 1,235ms | **3.78x** |

*Tested on 16-core machine with all-MiniLM-L6-v2*

### Environment Benchmarks

| Environment | Mode | Throughput | Latency |
|-------------|------|------------|---------|
| Node.js 20 | Sequential | ~2.5 texts/sec | ~390ms |
| Node.js 20 | Parallel (4w) | ~9.7 texts/sec | ~103ms |
| Chrome (M1 Mac) | Sequential | ~50 texts/sec | ~20ms |
| Firefox (M1 Mac) | Sequential | ~45 texts/sec | ~22ms |
| Cloudflare Workers | Sequential | ~30 texts/sec | ~33ms |
| Deno | Sequential | ~75 texts/sec | ~13ms |

*Browser benchmarks with smaller inputs; Node.js with full model warmup*

### SIMD Support

WASM SIMD128 is enabled by default and provides:
- Smaller binary size (180KB reduction)
- Vectorized tensor operations
- Supported in Chrome 91+, Firefox 89+, Safari 16.4+, Node.js 16+

```javascript
import { simd_available } from 'ruvector-onnx-embeddings-wasm';
console.log('SIMD enabled:', simd_available()); // true
```

## API Reference

### ModelLoader

```javascript
import { ModelLoader, MODELS, DEFAULT_MODEL } from 'ruvector-onnx-embeddings-wasm/loader';

// List available models
console.log(ModelLoader.listModels());

// Load with progress
const loader = new ModelLoader({
  cache: true,
  onProgress: ({ loaded, total, percent }) => console.log(`${percent}%`)
});

const { modelBytes, tokenizerJson, config } = await loader.loadModel('all-MiniLM-L6-v2');
```

### WasmEmbedder

```typescript
class WasmEmbedder {
  constructor(modelBytes: Uint8Array, tokenizerJson: string);

  static withConfig(
    modelBytes: Uint8Array,
    tokenizerJson: string,
    config: WasmEmbedderConfig
  ): WasmEmbedder;

  embedOne(text: string): Float32Array;
  embedBatch(texts: string[]): Float32Array;
  similarity(text1: string, text2: string): number;

  dimension(): number;
  maxLength(): number;
}
```

### WasmEmbedderConfig

```typescript
class WasmEmbedderConfig {
  constructor();
  setMaxLength(length: number): WasmEmbedderConfig;
  setNormalize(normalize: boolean): WasmEmbedderConfig;
  setPooling(strategy: number): WasmEmbedderConfig;
  // 0=Mean, 1=Cls, 2=Max, 3=MeanSqrtLen, 4=LastToken
}
```

### ParallelEmbedder (Node.js only)

```typescript
class ParallelEmbedder {
  constructor(options?: { numWorkers?: number });

  init(modelName?: string): Promise<void>;
  embedOne(text: string): Promise<Float32Array>;
  embedBatch(texts: string[]): Promise<number[][]>;
  similarity(text1: string, text2: string): Promise<number>;
  shutdown(): Promise<void>;
}
```

### Utility Functions

```typescript
function cosineSimilarity(a: Float32Array, b: Float32Array): number;
function normalizeL2(embedding: Float32Array): Float32Array;
function version(): string;
function simd_available(): boolean;
```

### Convenience Functions

```typescript
// One-liner embedding
async function embed(text: string | string[], modelName?: string): Promise<Float32Array>;

// One-liner similarity
async function similarity(text1: string, text2: string, modelName?: string): Promise<number>;

// Create configured embedder
async function createEmbedder(modelName?: string): Promise<WasmEmbedder>;
```

## Pooling Strategies

| Value | Strategy | Description |
|-------|----------|-------------|
| 0 | **Mean** | Average all tokens (default, recommended) |
| 1 | **Cls** | Use [CLS] token only (BERT-style) |
| 2 | **Max** | Max pooling across tokens |
| 3 | **MeanSqrtLen** | Mean normalized by sqrt(length) |
| 4 | **LastToken** | Last token (decoder models) |

## Comparison: Native vs WASM

| Aspect | Native (`ort`) | WASM (`tract`) |
|--------|----------------|----------------|
| Speed | ⚡⚡⚡ Native | ⚡⚡ ~2-3x slower |
| Browser | ❌ | ✅ |
| Edge Workers | ❌ | ✅ |
| Parallel | Multi-process | Worker threads |
| GPU | CUDA, TensorRT | ❌ |
| Bundle Size | ~50MB | ~7.4MB |
| SIMD | AVX2/AVX-512 | SIMD128 |
| Portability | Platform-specific | Universal |

**Use native** for: servers, high throughput, GPU acceleration
**Use WASM** for: browsers, edge, portability, simpler deployment

## Building from Source

```bash
# Install wasm-pack
cargo install wasm-pack

# Build the npm package (both glues, one SIMD .wasm) into ./pkg, then test the packed tarball
./scripts/build-npm.sh
node scripts/test-pack.mjs            # EMBED=1 also downloads a model and embeds

# Individual wasm-pack targets:
# Build for Node.js with SIMD
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target nodejs --release

# Build for web with SIMD
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target web --release

# Build for bundlers (webpack, vite) with SIMD
RUSTFLAGS='-C target-feature=+simd128' wasm-pack build --target bundler --release

# Build without SIMD (for older browsers)
wasm-pack build --target web --release
```

## Use Cases

### Semantic Search

```javascript
import { createEmbedder, cosineSimilarity } from 'ruvector-onnx-embeddings-wasm/loader';

const embedder = await createEmbedder();

// Index documents
const docs = ["Rust is fast", "Python is easy", "JavaScript runs everywhere"];
const embeddings = docs.map(d => embedder.embedOne(d));

// Search
const query = embedder.embedOne("Which language is performant?");
const scores = embeddings.map((e, i) => ({
  doc: docs[i],
  score: cosineSimilarity(query, e)
}));
scores.sort((a, b) => b.score - a.score);
console.log(scores[0]); // { doc: "Rust is fast", score: 0.82 }
```

### Batch Processing with Parallel Workers

```javascript
import { ParallelEmbedder } from 'ruvector-onnx-embeddings-wasm/parallel';

const embedder = new ParallelEmbedder({ numWorkers: 4 });
await embedder.init();

// Process large datasets efficiently
const documents = loadDocuments(); // Array of 1000+ texts
const batchSize = 100;

for (let i = 0; i < documents.length; i += batchSize) {
  const batch = documents.slice(i, i + batchSize);
  const embeddings = await embedder.embedBatch(batch);
  await saveEmbeddings(embeddings);
}

await embedder.shutdown();
```

### RAG (Retrieval-Augmented Generation)

```javascript
// Build knowledge base
const knowledge = [
  "RuVector is a vector database",
  "Embeddings capture semantic meaning",
  // ... more docs
];
const knowledgeEmbeddings = knowledge.map(k => embedder.embedOne(k));

// Retrieve relevant context for LLM
function getContext(query, topK = 3) {
  const queryEmb = embedder.embedOne(query);
  const scores = knowledgeEmbeddings.map((e, i) => ({
    text: knowledge[i],
    score: cosineSimilarity(queryEmb, e)
  }));
  return scores.sort((a, b) => b.score - a.score).slice(0, topK);
}
```

### Text Clustering

```javascript
const texts = [
  "Machine learning is amazing",
  "Deep learning uses neural networks",
  "I love pizza",
  "Italian food is delicious"
];

const embeddings = texts.map(t => embedder.embedOne(t));
// Use k-means or hierarchical clustering on embeddings
```

## Browser Compatibility

| Browser | SIMD | Status |
|---------|------|--------|
| Chrome 91+ | ✅ | Full support |
| Firefox 89+ | ✅ | Full support |
| Safari 16.4+ | ✅ | Full support |
| Edge 91+ | ✅ | Full support |
| Node.js 16+ | ✅ | Full support |
| Deno | ✅ | Full support |
| Cloudflare Workers | ✅ | Full support |

## Related Packages

| Package | Runtime | Use Case |
|---------|---------|----------|
| [ruvector-onnx-embeddings](https://crates.io/crates/ruvector-onnx-embeddings) | Native | High-performance servers |
| **ruvector-onnx-embeddings-wasm** | WASM | Browsers, edge, portable |

## Changelog

### v0.1.2
- Added `ParallelEmbedder` for multi-threaded batch processing (3.8x speedup)
- Worker threads support for Node.js environments

### v0.1.1
- Enabled WASM SIMD128 for vectorized operations
- Added `simd_available()` function
- Reduced binary size by 180KB

### v0.1.0
- Initial release
- HuggingFace model loader with caching
- Browser and Node.js support
- 6 pre-configured models

## License

MIT License - See [LICENSE](../../LICENSE) for details.

---

<p align="center">
  <b>Part of the RuVector ecosystem</b><br>
  High-performance vector operations in Rust
</p>
