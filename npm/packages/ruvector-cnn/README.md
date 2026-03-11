# @ruvector/cnn

[![npm version](https://img.shields.io/npm/v/@ruvector/cnn.svg)](https://www.npmjs.com/package/@ruvector/cnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CNN feature extraction for image embeddings** - SIMD-optimized, pure Rust/WASM.

Part of the [RuVector](https://github.com/ruvnet/ruvector) ecosystem.

## Features

- **SIMD-Optimized**: Uses WASM SIMD128 for 4-8x speedup on supported browsers
- **Contrastive Learning**: Built-in InfoNCE (SimCLR) and Triplet loss
- **Pure Rust/WASM**: No native dependencies, runs everywhere
- **MobileNet-style**: Efficient architectures optimized for CPU/WASM
- **TypeScript Support**: Full type definitions included

## Installation

```bash
npm install @ruvector/cnn
# or
yarn add @ruvector/cnn
# or
pnpm add @ruvector/cnn
```

## Quick Start

```javascript
const { init, CnnEmbedder, InfoNCELoss, SimdOps } = require('@ruvector/cnn');

// Initialize WASM module
await init();

// Create embedder
const embedder = new CnnEmbedder({
  embeddingDim: 512,
  normalize: true
});

// Extract embedding from image (RGB, no alpha)
const imageData = new Uint8Array(224 * 224 * 3);
const embedding = embedder.extract(imageData, 224, 224);
console.log('Embedding shape:', embedding.length); // 512

// Compute similarity between two images
const embedding2 = embedder.extract(imageData2, 224, 224);
const similarity = embedder.cosineSimilarity(embedding, embedding2);
console.log('Similarity:', similarity); // -1 to 1
```

## Contrastive Learning

### InfoNCE Loss (SimCLR style)

```javascript
const { init, InfoNCELoss } = require('@ruvector/cnn');
await init();

const loss = new InfoNCELoss(0.1); // temperature = 0.1

// Embeddings for 4 pairs: [view1_0, view1_1, view1_2, view1_3, view2_0, view2_1, view2_2, view2_3]
const embeddings = new Float32Array(8 * 512); // 8 embeddings of dim 512
// ... fill with actual embeddings

const lossValue = loss.forward(embeddings, 4, 512);
console.log('InfoNCE Loss:', lossValue);
```

### Triplet Loss

```javascript
const { init, TripletLoss } = require('@ruvector/cnn');
await init();

const loss = new TripletLoss(1.0); // margin = 1.0

const anchors = new Float32Array(batch * 512);
const positives = new Float32Array(batch * 512);
const negatives = new Float32Array(batch * 512);

const lossValue = loss.forward(anchors, positives, negatives, 512);
console.log('Triplet Loss:', lossValue);
```

## SIMD Operations

Low-level SIMD-optimized operations for custom networks:

```javascript
const { init, SimdOps, LayerOps } = require('@ruvector/cnn');
await init();

// Dot product
const a = new Float32Array([1, 2, 3, 4]);
const b = new Float32Array([5, 6, 7, 8]);
const dot = SimdOps.dotProduct(a, b); // 70

// Activations (in-place)
const data = new Float32Array([-1, 0, 1, 7]);
SimdOps.relu(data);   // [0, 0, 1, 7]
SimdOps.relu6(data);  // [0, 0, 1, 6]

// L2 normalize
SimdOps.l2Normalize(data);

// Batch normalization
LayerOps.batchNorm(input, gamma, beta, mean, variance, 1e-5);

// Global average pooling
const pooled = LayerOps.globalAvgPool(input, channels, height * width);
```

## Browser Usage

```html
<script type="module">
import { init, CnnEmbedder } from 'https://unpkg.com/@ruvector/cnn';

await init();
const embedder = new CnnEmbedder();

// Get image from canvas
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, 224, 224);

// Convert RGBA to RGB
const rgb = new Uint8Array(224 * 224 * 3);
for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
  rgb[j] = imageData.data[i];
  rgb[j + 1] = imageData.data[i + 1];
  rgb[j + 2] = imageData.data[i + 2];
}

const embedding = embedder.extract(rgb, 224, 224);
console.log('Embedding:', embedding);
</script>
```

## API Reference

### `CnnEmbedder`

| Method | Description |
|--------|-------------|
| `constructor(config?)` | Create embedder with optional config |
| `extract(imageData, width, height)` | Extract embedding from RGB image |
| `cosineSimilarity(a, b)` | Compute cosine similarity |
| `embeddingDim` | Get embedding dimension |

### `InfoNCELoss`

| Method | Description |
|--------|-------------|
| `constructor(temperature?)` | Create loss with temperature (default: 0.1) |
| `forward(embeddings, batchSize, dim)` | Compute loss for pairs |
| `temperature` | Get temperature parameter |

### `TripletLoss`

| Method | Description |
|--------|-------------|
| `constructor(margin?)` | Create loss with margin (default: 1.0) |
| `forward(anchors, positives, negatives, dim)` | Compute triplet loss |
| `margin` | Get margin parameter |

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Embedding extraction (224x224) | ~5ms | With SIMD128 |
| InfoNCE loss (batch=32, dim=512) | ~1ms | |
| Dot product (512-d) | ~0.01ms | SIMD optimized |

## Related Packages

- [`ruvector`](https://www.npmjs.com/package/ruvector) - Core vector operations
- [`@ruvector/attention`](https://www.npmjs.com/package/@ruvector/attention) - Attention mechanisms
- [`@ruvector/gnn`](https://www.npmjs.com/package/@ruvector/gnn) - Graph neural networks

## License

MIT OR Apache-2.0
