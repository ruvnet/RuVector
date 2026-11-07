# @ruvector/attention

High-performance attention mechanisms for Node.js, powered by Rust.

## Features

- **Scaled Dot-Product Attention**: Classic attention mechanism with optional scaling
- **Multi-Head Attention**: Parallel attention heads for richer representations
- **Flash Attention**: Memory-efficient attention with block-wise computation
- **Linear Attention**: O(N) complexity attention using kernel approximations
- **Hyperbolic Attention**: Attention in hyperbolic space for hierarchical data
- **Mixture-of-Experts (MoE) Attention**: Dynamic expert routing for specialized attention

## Installation

```bash
npm install @ruvector/attention
```

## Usage

### Basic Dot-Product Attention

```javascript
const { DotProductAttention } = require('@ruvector/attention');

const attention = new DotProductAttention(512, 1.0);
const query = new Float32Array([/* ... */]);
const keys = [new Float32Array([/* ... */])];
const values = [new Float32Array([/* ... */])];

const output = attention.compute(query, keys, values);
```

### Multi-Head Attention

```javascript
const { MultiHeadAttention } = require('@ruvector/attention');

const mha = new MultiHeadAttention(512, 8); // 512 dim, 8 heads
const output = mha.compute(query, keys, values);

// Async version for large computations
```

### Flash Attention

```javascript
const { FlashAttention } = require('@ruvector/attention');

const flash = new FlashAttention(512, 64); // 512 dim, 64 block size
const output = flash.compute(query, keys, values);
```

### Flat inputs (fastest)

Every mechanism also has `computeFlat(query, keys, values)`, where `keys` and
`values` are single row-major `Float32Array`s of `n * dim` floats. Only three
typed arrays cross into native code, so it avoids the per-key cost of
`Float32Array[]`:

```javascript
const keysFlat = new Float32Array(n * 384);   // row i at offset i * 384
const valuesFlat = new Float32Array(n * 384);
const output = flash.computeFlat(query, keysFlat, valuesFlat);
```

Measured in 2.3.0 (dim 384, x86-64, median µs per call, `FlashAttention(384, 64)`
and `MultiHeadAttention(384, 8)`), against a plain JS loop over the same data:

| case | plain JS | `compute` | `computeFlat` |
|---|---|---|---|
| flash, 20 keys | 8.6 | 5.6 | 3.9 |
| multi-head, 20 keys | 24.0 | 6.7 | 5.0 |
| flash, 200 keys | 115.9 | 43.2 | 31.3 |
| flash, 1000 keys | 412.9 | 204.5 | 155.8 |

### Hyperbolic Attention

```javascript
const { HyperbolicAttention } = require('@ruvector/attention');

const hyperbolic = new HyperbolicAttention(512, -1.0); // negative curvature
const output = hyperbolic.compute(query, keys, values);
```

### Mixture-of-Experts Attention

```javascript
const { MoEAttention } = require('@ruvector/attention');

const moe = new MoEAttention({
  dim: 512,
  numExperts: 8,
  topK: 2,
  expertCapacity: 1.25
});

const output = moe.compute(query, keys, values);
const expertUsage = moe.getExpertUsage();
```

### Training

```javascript
const { Trainer, AdamOptimizer } = require('@ruvector/attention');

// Configure training
const trainer = new Trainer({
  learningRate: 0.001,
  batchSize: 32,
  numEpochs: 100,
  weightDecay: 0.01,
  gradientClip: 1.0,
  warmupSteps: 1000
});

// Training step
const loss = trainer.trainStep(inputs, targets);

// Get metrics
const metrics = trainer.getMetrics();
console.log(`Loss: ${metrics.loss}, LR: ${metrics.learningRate}`);

// Custom optimizer
const optimizer = new AdamOptimizer(0.001, 0.9, 0.999, 1e-8);
const updatedParams = optimizer.step(gradients);
```

### Batch Processing

```javascript
const { BatchProcessor, parallelAttentionCompute } = require('@ruvector/attention');

// Batch processor for efficient batching
const processor = new BatchProcessor({
  batchSize: 32,
  numWorkers: 4,
  prefetch: true
});

const results = await processor.processBatch(queries, keys, values);
const throughput = processor.getThroughput();

// Parallel computation with automatic worker management
const results = await parallelAttentionCompute(
  'multi-head',
  queries,
  keys,
  values,
  4 // number of workers
);
```

## API Reference

### Classes

#### `DotProductAttention`
- `constructor(dim: number, scale?: number)`
- `compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array`
- `computeFlat(query: Float32Array, keys: Float32Array, values: Float32Array): Float32Array`

#### `MultiHeadAttention`
- `constructor(dim: number, numHeads: number)`
- `compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array`
- `computeFlat(query: Float32Array, keys: Float32Array, values: Float32Array): Float32Array`

#### `FlashAttention`
- `constructor(dim: number, blockSize: number)`
- `compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array`
- `computeFlat(query: Float32Array, keys: Float32Array, values: Float32Array): Float32Array`

#### `LinearAttention`
- `constructor(dim: number, numFeatures: number)`
- `compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array`
- `computeFlat(query: Float32Array, keys: Float32Array, values: Float32Array): Float32Array`

#### `HyperbolicAttention`
- `constructor(dim: number, curvature: number)`
- `compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array`
- `computeFlat(query: Float32Array, keys: Float32Array, values: Float32Array): Float32Array`

#### `MoEAttention`
- `constructor(config: MoEConfig)`
- `compute(query: Float32Array, keys: Float32Array[], values: Float32Array[]): Float32Array`
- `computeFlat(query: Float32Array, keys: Float32Array, values: Float32Array): Float32Array`
- `getExpertUsage(): number[]`

#### `Trainer`
- `constructor(config: TrainingConfig)`
- `trainStep(inputs: Float32Array[], targets: Float32Array[]): number`
- `trainStepAsync(inputs: Float32Array[], targets: Float32Array[]): Promise<number>`
- `getMetrics(): TrainingMetrics`

#### `AdamOptimizer`
- `constructor(learningRate: number, beta1?: number, beta2?: number, epsilon?: number)`
- `step(gradients: Float32Array[]): Float32Array[]`
- `getLearningRate(): number`
- `setLearningRate(lr: number): void`

#### `BatchProcessor`
- `constructor(config: BatchConfig)`
- `processBatch(queries: Float32Array[], keys: Float32Array[][], values: Float32Array[][]): Promise<Float32Array[]>`
- `getThroughput(): number`

### Functions

#### `parallelAttentionCompute`
```typescript
function parallelAttentionCompute(
  attentionType: string,
  queries: Float32Array[],
  keys: Float32Array[][],
  values: Float32Array[][],
  numWorkers?: number
): Promise<Float32Array[]>
```

#### `version`
Returns the package version string.

## Performance

This package uses Rust under the hood for optimal performance:
- Zero-copy data transfer where possible
- SIMD optimizations for vector operations
- Multi-threaded batch processing
- Memory-efficient attention mechanisms

## Platform Support

Pre-built binaries are provided for:
- macOS (x64, ARM64)
- Linux (x64, ARM64, musl)
- Windows (x64, ARM64)

## License

MIT OR Apache-2.0
