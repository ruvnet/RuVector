/**
 * @ruvector/cnn - CNN feature extraction for image embeddings
 *
 * SIMD-optimized image embedding extraction using contrastive learning.
 *
 * @example
 * ```javascript
 * const { CnnEmbedder, InfoNCELoss, SimdOps } = require('@ruvector/cnn');
 *
 * // Create embedder
 * const embedder = new CnnEmbedder({ embeddingDim: 512, normalize: true });
 *
 * // Extract embedding from image data
 * const imageData = new Uint8Array(224 * 224 * 3); // RGB image
 * const embedding = embedder.extract(imageData, 224, 224);
 *
 * // Compute similarity
 * const sim = embedder.cosineSimilarity(embedding1, embedding2);
 * ```
 */

'use strict';

let wasm = null;
let initialized = false;

/**
 * Initialize the WASM module
 * @returns {Promise<void>}
 */
async function init() {
  if (initialized) return;

  if (typeof window !== 'undefined') {
    // Browser environment
    const wasmModule = await import('./ruvector_cnn_wasm.js');
    await wasmModule.default();
    wasm = wasmModule;
  } else {
    // Node.js environment
    const fs = require('fs');
    const path = require('path');
    const wasmPath = path.join(__dirname, 'ruvector_cnn_wasm_bg.wasm');

    if (fs.existsSync(wasmPath)) {
      const wasmModule = require('./ruvector_cnn_wasm.js');
      const wasmBuffer = fs.readFileSync(wasmPath);
      await wasmModule.default(wasmBuffer);
      wasm = wasmModule;
    } else {
      throw new Error('WASM file not found. Run `npm run build` first.');
    }
  }

  initialized = true;
}

/**
 * CNN Embedder for extracting image features
 */
class CnnEmbedder {
  /**
   * Create a new CNN embedder
   * @param {Object} [config] - Configuration options
   * @param {number} [config.inputSize=224] - Input image size (square)
   * @param {number} [config.embeddingDim=512] - Output embedding dimension
   * @param {boolean} [config.normalize=true] - L2 normalize embeddings
   */
  constructor(config = {}) {
    if (!initialized) {
      throw new Error('Module not initialized. Call init() first.');
    }

    const wasmConfig = new wasm.EmbedderConfig();
    wasmConfig.input_size = config.inputSize || 224;
    wasmConfig.embedding_dim = config.embeddingDim || 512;
    wasmConfig.normalize = config.normalize !== false;

    this._inner = new wasm.WasmCnnEmbedder(wasmConfig);
    this._embeddingDim = wasmConfig.embedding_dim;
  }

  /**
   * Extract embedding from image data
   * @param {Uint8Array} imageData - RGB image data (row-major, no alpha)
   * @param {number} width - Image width
   * @param {number} height - Image height
   * @returns {Float32Array} - Embedding vector
   */
  extract(imageData, width, height) {
    const result = this._inner.extract(imageData, width, height);
    return new Float32Array(result);
  }

  /**
   * Compute cosine similarity between two embeddings
   * @param {Float32Array} a - First embedding
   * @param {Float32Array} b - Second embedding
   * @returns {number} - Similarity in [-1, 1]
   */
  cosineSimilarity(a, b) {
    return this._inner.cosine_similarity(a, b);
  }

  /**
   * Get the embedding dimension
   * @returns {number}
   */
  get embeddingDim() {
    return this._embeddingDim;
  }
}

/**
 * InfoNCE loss for contrastive learning (SimCLR style)
 */
class InfoNCELoss {
  /**
   * Create InfoNCE loss
   * @param {number} [temperature=0.1] - Temperature parameter
   */
  constructor(temperature = 0.1) {
    if (!initialized) {
      throw new Error('Module not initialized. Call init() first.');
    }
    this._inner = new wasm.WasmInfoNCELoss(temperature);
  }

  /**
   * Compute loss for embedding pairs
   * @param {Float32Array} embeddings - Flattened [2N, D] array
   * @param {number} batchSize - N (number of pairs)
   * @param {number} dim - D (embedding dimension)
   * @returns {number} - Loss value
   */
  forward(embeddings, batchSize, dim) {
    return this._inner.forward(embeddings, batchSize, dim);
  }

  /**
   * Get temperature parameter
   * @returns {number}
   */
  get temperature() {
    return this._inner.temperature;
  }
}

/**
 * Triplet loss for metric learning
 */
class TripletLoss {
  /**
   * Create triplet loss
   * @param {number} [margin=1.0] - Margin parameter
   */
  constructor(margin = 1.0) {
    if (!initialized) {
      throw new Error('Module not initialized. Call init() first.');
    }
    this._inner = new wasm.WasmTripletLoss(margin);
  }

  /**
   * Compute triplet loss
   * @param {Float32Array} anchors - Anchor embeddings [N, D]
   * @param {Float32Array} positives - Positive embeddings [N, D]
   * @param {Float32Array} negatives - Negative embeddings [N, D]
   * @param {number} dim - Embedding dimension D
   * @returns {number} - Loss value
   */
  forward(anchors, positives, negatives, dim) {
    return this._inner.forward(anchors, positives, negatives, dim);
  }

  /**
   * Get margin parameter
   * @returns {number}
   */
  get margin() {
    return this._inner.margin;
  }
}

/**
 * SIMD-optimized operations
 */
const SimdOps = {
  /**
   * Compute dot product of two vectors
   * @param {Float32Array} a
   * @param {Float32Array} b
   * @returns {number}
   */
  dotProduct(a, b) {
    if (!initialized) throw new Error('Module not initialized');
    return wasm.SimdOps.dot_product(a, b);
  },

  /**
   * Apply ReLU activation in-place
   * @param {Float32Array} data
   */
  relu(data) {
    if (!initialized) throw new Error('Module not initialized');
    wasm.SimdOps.relu(data);
  },

  /**
   * Apply ReLU6 activation in-place
   * @param {Float32Array} data
   */
  relu6(data) {
    if (!initialized) throw new Error('Module not initialized');
    wasm.SimdOps.relu6(data);
  },

  /**
   * L2 normalize a vector in-place
   * @param {Float32Array} data
   */
  l2Normalize(data) {
    if (!initialized) throw new Error('Module not initialized');
    wasm.SimdOps.l2_normalize(data);
  }
};

/**
 * Layer operations for building custom networks
 */
const LayerOps = {
  /**
   * Apply batch normalization
   * @param {Float32Array} input - Input data (modified in-place)
   * @param {Float32Array} gamma - Scale parameter
   * @param {Float32Array} beta - Shift parameter
   * @param {Float32Array} mean - Running mean
   * @param {Float32Array} variance - Running variance
   * @param {number} [epsilon=1e-5] - Numerical stability
   */
  batchNorm(input, gamma, beta, mean, variance, epsilon = 1e-5) {
    if (!initialized) throw new Error('Module not initialized');
    wasm.LayerOps.batch_norm(input, gamma, beta, mean, variance, epsilon);
  },

  /**
   * Apply global average pooling
   * @param {Float32Array} input - Input tensor [C, H, W]
   * @param {number} channels - Number of channels
   * @param {number} spatialSize - H * W
   * @returns {Float32Array} - Output [C]
   */
  globalAvgPool(input, channels, spatialSize) {
    if (!initialized) throw new Error('Module not initialized');
    return new Float32Array(wasm.LayerOps.global_avg_pool(input, channels, spatialSize));
  }
};

module.exports = {
  init,
  CnnEmbedder,
  InfoNCELoss,
  TripletLoss,
  SimdOps,
  LayerOps
};
