/**
 * @ruvector/cnn - CNN feature extraction for image embeddings
 */

/**
 * Initialize the WASM module
 */
export function init(): Promise<void>;

/**
 * Configuration for CNN embedder
 */
export interface EmbedderConfig {
  /** Input image size (square), default: 224 */
  inputSize?: number;
  /** Output embedding dimension, default: 512 */
  embeddingDim?: number;
  /** L2 normalize embeddings, default: true */
  normalize?: boolean;
}

/**
 * CNN Embedder for extracting image features
 */
export class CnnEmbedder {
  /**
   * Create a new CNN embedder
   * @param config - Configuration options
   */
  constructor(config?: EmbedderConfig);

  /**
   * Extract embedding from image data
   * @param imageData - RGB image data (row-major, no alpha)
   * @param width - Image width
   * @param height - Image height
   * @returns Embedding vector
   */
  extract(imageData: Uint8Array, width: number, height: number): Float32Array;

  /**
   * Compute cosine similarity between two embeddings
   * @param a - First embedding
   * @param b - Second embedding
   * @returns Similarity in [-1, 1]
   */
  cosineSimilarity(a: Float32Array, b: Float32Array): number;

  /**
   * Get the embedding dimension
   */
  readonly embeddingDim: number;
}

/**
 * InfoNCE loss for contrastive learning (SimCLR style)
 */
export class InfoNCELoss {
  /**
   * Create InfoNCE loss
   * @param temperature - Temperature parameter (default: 0.1)
   */
  constructor(temperature?: number);

  /**
   * Compute loss for embedding pairs
   * @param embeddings - Flattened [2N, D] array where (i, i+N) are positive pairs
   * @param batchSize - N (number of pairs)
   * @param dim - D (embedding dimension)
   * @returns Loss value
   */
  forward(embeddings: Float32Array, batchSize: number, dim: number): number;

  /**
   * Get temperature parameter
   */
  readonly temperature: number;
}

/**
 * Triplet loss for metric learning
 */
export class TripletLoss {
  /**
   * Create triplet loss
   * @param margin - Margin parameter (default: 1.0)
   */
  constructor(margin?: number);

  /**
   * Compute triplet loss
   * @param anchors - Anchor embeddings [N, D]
   * @param positives - Positive embeddings [N, D]
   * @param negatives - Negative embeddings [N, D]
   * @param dim - Embedding dimension D
   * @returns Loss value
   */
  forward(
    anchors: Float32Array,
    positives: Float32Array,
    negatives: Float32Array,
    dim: number
  ): number;

  /**
   * Get margin parameter
   */
  readonly margin: number;
}

/**
 * SIMD-optimized operations
 */
export namespace SimdOps {
  /**
   * Compute dot product of two vectors
   */
  function dotProduct(a: Float32Array, b: Float32Array): number;

  /**
   * Apply ReLU activation in-place
   */
  function relu(data: Float32Array): void;

  /**
   * Apply ReLU6 activation in-place
   */
  function relu6(data: Float32Array): void;

  /**
   * L2 normalize a vector in-place
   */
  function l2Normalize(data: Float32Array): void;
}

/**
 * Layer operations for building custom networks
 */
export namespace LayerOps {
  /**
   * Apply batch normalization
   */
  function batchNorm(
    input: Float32Array,
    gamma: Float32Array,
    beta: Float32Array,
    mean: Float32Array,
    variance: Float32Array,
    epsilon?: number
  ): void;

  /**
   * Apply global average pooling
   * @param input - Input tensor [C, H, W]
   * @param channels - Number of channels
   * @param spatialSize - H * W
   * @returns Output [C]
   */
  function globalAvgPool(
    input: Float32Array,
    channels: number,
    spatialSize: number
  ): Float32Array;
}
