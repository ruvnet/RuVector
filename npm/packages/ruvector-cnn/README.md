# @ruvector/cnn

[![npm version](https://img.shields.io/npm/v/@ruvector/cnn.svg)](https://www.npmjs.com/package/@ruvector/cnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Turn images into searchable vectors** — runs in browsers, no backend needed.

## What Does This Do?

This package converts images into numbers (called "embeddings") that describe what's in the picture. Similar images produce similar numbers.

**Use it to:**
- Build "find similar images" features
- Group photos by what they show
- Create visual search for products
- Compare images without AI APIs

```javascript
import { init, CnnEmbedder } from '@ruvector/cnn';

await init();
const embedder = new CnnEmbedder();

// Turn an image into numbers
const numbers = embedder.extract(imagePixels, 224, 224);

// Compare two images (1.0 = identical, 0 = unrelated)
const similarity = embedder.cosineSimilarity(numbers1, numbers2);
```

## Why Use This?

| What You Get | Why It Matters |
|--------------|----------------|
| Runs in the browser | No server costs, instant results |
| ~5ms per image | Fast enough for real-time |
| ~900KB download | Small enough for any website |
| No API calls | Works offline, no per-image fees |
| Training included | Teach it your own categories |

## Installation

```bash
npm install @ruvector/cnn
```

## How to Use It

### 1. Extract Image Features

```javascript
import { init, CnnEmbedder } from '@ruvector/cnn';

// Start the engine (do this once)
await init();

// Create the feature extractor
const embedder = new CnnEmbedder({
  embeddingDim: 512,  // How many numbers per image
  normalize: true      // Make comparisons easier
});

// Get pixels from an image (RGB, no transparency)
// Each pixel has 3 values: red, green, blue (0-255)
const pixels = new Uint8Array(224 * 224 * 3);

// Turn pixels into 512 numbers that describe the image
const features = embedder.extract(pixels, 224, 224);
console.log('Got', features.length, 'numbers'); // 512
```

### 2. Compare Two Images

```javascript
const features1 = embedder.extract(image1Pixels, 224, 224);
const features2 = embedder.extract(image2Pixels, 224, 224);

// How similar are they? (1.0 = same, 0 = different, -1 = opposite)
const score = embedder.cosineSimilarity(features1, features2);

if (score > 0.8) {
  console.log('These images are very similar!');
} else if (score > 0.5) {
  console.log('These images have some things in common');
} else {
  console.log('These images are different');
}
```

### 3. Find the Most Similar Image

```javascript
// Your collection of images (already converted to features)
const catalog = [
  { name: 'red-shoe.jpg', features: embedder.extract(redShoePixels, 224, 224) },
  { name: 'blue-bag.jpg', features: embedder.extract(blueBagPixels, 224, 224) },
  { name: 'red-dress.jpg', features: embedder.extract(redDressPixels, 224, 224) },
];

// User uploads a photo
const userPhoto = embedder.extract(uploadedPixels, 224, 224);

// Find the best match
let bestMatch = null;
let bestScore = -1;

for (const item of catalog) {
  const score = embedder.cosineSimilarity(userPhoto, item.features);
  if (score > bestScore) {
    bestScore = score;
    bestMatch = item.name;
  }
}

console.log('Best match:', bestMatch, 'Score:', bestScore);
```

### 4. Get Pixels from a Canvas

```javascript
// If you have an image in a canvas element
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

// Get the pixel data
const imageData = ctx.getImageData(0, 0, 224, 224);

// Canvas gives RGBA (4 values per pixel), we need RGB (3 values)
const rgb = new Uint8Array(224 * 224 * 3);
for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
  rgb[j] = imageData.data[i];       // Red
  rgb[j + 1] = imageData.data[i + 1]; // Green
  rgb[j + 2] = imageData.data[i + 2]; // Blue
  // Skip alpha (imageData.data[i + 3])
}

const features = embedder.extract(rgb, 224, 224);
```

## Training (Teaching It Your Categories)

You can train the model to be better at recognizing your specific images.

### Contrastive Training (SimCLR style)

Show it pairs of images that should match:

```javascript
import { init, InfoNCELoss, CnnEmbedder } from '@ruvector/cnn';

await init();

const embedder = new CnnEmbedder();
const trainer = new InfoNCELoss(0.1); // Lower = stricter matching

// Get features for your training pairs
// Pairs: image1 and image1_different_angle should match
const image1 = embedder.extract(photo1, 224, 224);
const image1_alt = embedder.extract(photo1_rotated, 224, 224);

// Pack into one array: [view1s..., view2s...]
const batch = new Float32Array(2 * 512);
batch.set(image1, 0);
batch.set(image1_alt, 512);

const loss = trainer.forward(batch, 1, 512);
console.log('Loss:', loss); // Lower is better
```

### Triplet Training

Show it: "A is similar to B, but different from C"

```javascript
import { init, TripletLoss, CnnEmbedder } from '@ruvector/cnn';

await init();

const embedder = new CnnEmbedder();
const trainer = new TripletLoss(1.0); // margin

// Anchor: the reference image
// Positive: should be similar to anchor
// Negative: should be different from anchor
const anchor = embedder.extract(redShoePhoto, 224, 224);
const positive = embedder.extract(redShoePhoto2, 224, 224); // Same shoe
const negative = embedder.extract(blueBagPhoto, 224, 224);   // Different item

const loss = trainer.forward(
  new Float32Array(anchor),
  new Float32Array(positive),
  new Float32Array(negative),
  512
);
console.log('Loss:', loss);
```

## Fast Math Operations

If you're building custom features, these are optimized:

```javascript
import { init, SimdOps, LayerOps } from '@ruvector/cnn';

await init();

// Dot product (sum of element-wise multiplication)
const a = new Float32Array([1, 2, 3, 4]);
const b = new Float32Array([5, 6, 7, 8]);
const result = SimdOps.dotProduct(a, b); // 70

// ReLU: set negative values to 0
const data = new Float32Array([-1, 0, 1, 7]);
SimdOps.relu(data); // [0, 0, 1, 7]

// ReLU6: clamp between 0 and 6
SimdOps.relu6(data); // [0, 0, 1, 6]

// L2 normalize (make length = 1)
SimdOps.l2Normalize(data);
```

## Performance

| What | How Long | Notes |
|------|----------|-------|
| Extract features (224×224 image) | ~5ms | With SIMD |
| Compare two images | ~0.01ms | Just math |
| Training step | ~1ms | Per batch |
| First load | ~100ms | Downloads WASM |

## Browser Support

Works in all modern browsers with WebAssembly:
- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

For best speed, use browsers with SIMD128 support:
- Chrome 91+
- Firefox 89+
- Safari 16.4+

## Troubleshooting

**"init() takes too long"**
- Normal: First call downloads ~900KB WASM file
- Fix: Call init() early, before user needs results

**"Images look wrong"**
- Check: Images must be 224×224 pixels
- Check: Pixel format is RGB (3 values per pixel, not RGBA)
- Check: Values are 0-255, not 0-1

**"Similarity scores are all low"**
- Try: Set `normalize: true` in CnnEmbedder options
- Check: Are your images actually similar?

## API Reference

### CnnEmbedder

```typescript
new CnnEmbedder(options?: {
  embeddingDim?: number;  // Default: 512
  normalize?: boolean;    // Default: true
})

.extract(pixels: Uint8Array, width: number, height: number): Float32Array
.cosineSimilarity(a: Float32Array, b: Float32Array): number
.embeddingDim: number
```

### InfoNCELoss

```typescript
new InfoNCELoss(temperature?: number)  // Default: 0.1

.forward(embeddings: Float32Array, batchSize: number, dim: number): number
.temperature: number
```

### TripletLoss

```typescript
new TripletLoss(margin?: number)  // Default: 1.0

.forward(anchors, positives, negatives: Float32Array, dim: number): number
.margin: number
```

## Related Packages

- [`ruvector`](https://www.npmjs.com/package/ruvector) — Core vector database
- [`@ruvector/attention`](https://www.npmjs.com/package/@ruvector/attention) — AI attention layers
- [`@ruvector/gnn`](https://www.npmjs.com/package/@ruvector/gnn) — Graph neural networks

## License

MIT OR Apache-2.0
