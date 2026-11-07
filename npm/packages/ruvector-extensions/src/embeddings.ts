/**
 * @fileoverview Comprehensive embeddings integration module for ruvector-extensions
 * Supports multiple providers: OpenAI, Cohere, Anthropic, and local HuggingFace models
 *
 * @module embeddings
 * @author ruv.io Team <info@ruv.io>
 * @license MIT
 *
 * @example
 * ```typescript
 * // OpenAI embeddings
 * const openai = new OpenAIEmbeddings({ apiKey: 'sk-...' });
 * const embeddings = await openai.embedTexts(['Hello world', 'Test']);
 *
 * // Auto-insert into VectorDB
 * await embedAndInsert(db, openai, [
 *   { id: '1', text: 'Hello world', metadata: { source: 'test' } }
 * ]);
 * ```
 */

import { createRequire } from 'node:module';

// The package is ESM ("type": "module"), so the CommonJS `require` global does
// not exist; build one to load the optional, synchronously-constructed SDKs.
const require = createRequire(import.meta.url);

// VectorDB type will be used as any for maximum compatibility
type VectorDB = any;

// ============================================================================
// Core Types and Interfaces
// ============================================================================

/**
 * Configuration for retry logic
 */
export interface RetryConfig {
  /** Maximum number of retry attempts */
  maxRetries: number;
  /** Initial delay in milliseconds before first retry */
  initialDelay: number;
  /** Maximum delay in milliseconds between retries */
  maxDelay: number;
  /** Multiplier for exponential backoff */
  backoffMultiplier: number;
}

/**
 * Result of an embedding operation
 */
export interface EmbeddingResult {
  /** The generated embedding vector */
  embedding: number[];
  /** Index of the text in the original batch */
  index: number;
  /** Optional token count used */
  tokens?: number;
}

/**
 * Batch result with embeddings and metadata
 */
export interface BatchEmbeddingResult {
  /** Array of embedding results */
  embeddings: EmbeddingResult[];
  /** Total tokens used (if available) */
  totalTokens?: number;
  /** Provider-specific metadata */
  metadata?: Record<string, unknown>;
}

/**
 * Error details for failed embedding operations
 */
export interface EmbeddingError {
  /** Error message */
  message: string;
  /** Original error object */
  error: unknown;
  /** Index of the text that failed (if applicable) */
  index?: number;
  /** Whether the error is retryable */
  retryable: boolean;
}

/**
 * Document to embed and insert into VectorDB
 */
export interface DocumentToEmbed {
  /** Unique identifier for the document */
  id: string;
  /** Text content to embed */
  text: string;
  /** Optional metadata to store with the vector */
  metadata?: Record<string, unknown>;
}

// ============================================================================
// Abstract Base Class
// ============================================================================

/**
 * Abstract base class for embedding providers
 * All embedding providers must extend this class and implement its methods
 */
export abstract class EmbeddingProvider {
  protected retryConfig: RetryConfig;

  /**
   * Creates a new embedding provider instance
   * @param retryConfig - Configuration for retry logic
   */
  constructor(retryConfig?: Partial<RetryConfig>) {
    this.retryConfig = {
      maxRetries: 3,
      initialDelay: 1000,
      maxDelay: 10000,
      backoffMultiplier: 2,
      ...retryConfig,
    };
  }

  /**
   * Get the maximum batch size supported by this provider
   */
  abstract getMaxBatchSize(): number;

  /**
   * Get the dimension of embeddings produced by this provider
   */
  abstract getDimension(): number;

  /**
   * Embed a single text string
   * @param text - Text to embed
   * @returns Promise resolving to the embedding vector
   */
  async embedText(text: string): Promise<number[]> {
    const result = await this.embedTexts([text]);
    return result.embeddings[0].embedding;
  }

  /**
   * Embed multiple texts with automatic batching
   * @param texts - Array of texts to embed
   * @returns Promise resolving to batch embedding results
   */
  abstract embedTexts(texts: string[]): Promise<BatchEmbeddingResult>;

  /**
   * Execute a function with retry logic
   * @param fn - Function to execute
   * @param context - Context description for error messages
   * @returns Promise resolving to the function result
   */
  protected async withRetry<T>(
    fn: () => Promise<T>,
    context: string
  ): Promise<T> {
    let lastError: unknown;
    let delay = this.retryConfig.initialDelay;

    for (let attempt = 0; attempt <= this.retryConfig.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;

        // Check if error is retryable
        if (!this.isRetryableError(error)) {
          throw this.createEmbeddingError(error, context, false);
        }

        if (attempt < this.retryConfig.maxRetries) {
          await this.sleep(delay);
          delay = Math.min(
            delay * this.retryConfig.backoffMultiplier,
            this.retryConfig.maxDelay
          );
        }
      }
    }

    throw this.createEmbeddingError(
      lastError,
      `${context} (after ${this.retryConfig.maxRetries} retries)`,
      false
    );
  }

  /**
   * Determine if an error is retryable
   * @param error - Error to check
   * @returns True if the error should trigger a retry
   */
  protected isRetryableError(error: unknown): boolean {
    if (error instanceof Error) {
      const message = error.message.toLowerCase();
      // Rate limits, timeouts, and temporary server errors are retryable
      return (
        message.includes('rate limit') ||
        message.includes('timeout') ||
        message.includes('503') ||
        message.includes('429') ||
        message.includes('connection')
      );
    }
    return false;
  }

  /**
   * Create a standardized embedding error
   * @param error - Original error
   * @param context - Context description
   * @param retryable - Whether the error is retryable
   * @returns Formatted error object
   */
  protected createEmbeddingError(
    error: unknown,
    context: string,
    retryable: boolean
  ): EmbeddingError {
    const message = error instanceof Error ? error.message : String(error);
    return {
      message: `${context}: ${message}`,
      error,
      retryable,
    };
  }

  /**
   * Sleep for a specified duration
   * @param ms - Milliseconds to sleep
   */
  protected sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Split texts into batches based on max batch size
   * @param texts - Texts to batch
   * @returns Array of text batches
   */
  protected createBatches(texts: string[]): string[][] {
    const batches: string[][] = [];
    const batchSize = this.getMaxBatchSize();

    for (let i = 0; i < texts.length; i += batchSize) {
      batches.push(texts.slice(i, i + batchSize));
    }

    return batches;
  }
}

// ============================================================================
// OpenAI Embeddings Provider
// ============================================================================

/**
 * Configuration for OpenAI embeddings
 */
export interface OpenAIEmbeddingsConfig {
  /** OpenAI API key */
  apiKey: string;
  /** Model name (default: 'text-embedding-3-small') */
  model?: string;
  /** Embedding dimensions (only for text-embedding-3-* models) */
  dimensions?: number;
  /** Organization ID (optional) */
  organization?: string;
  /** Custom base URL (optional) */
  baseURL?: string;
  /** Retry configuration */
  retryConfig?: Partial<RetryConfig>;
}

/**
 * OpenAI embeddings provider
 * Supports text-embedding-3-small, text-embedding-3-large, and text-embedding-ada-002
 */
export class OpenAIEmbeddings extends EmbeddingProvider {
  private config: {
    apiKey: string;
    model: string;
    organization?: string;
    baseURL?: string;
    dimensions?: number;
  };
  private openai: any;

  /**
   * Creates a new OpenAI embeddings provider
   * @param config - Configuration options
   * @throws Error if OpenAI SDK is not installed
   */
  constructor(config: OpenAIEmbeddingsConfig) {
    super(config.retryConfig);

    this.config = {
      apiKey: config.apiKey,
      model: config.model || 'text-embedding-3-small',
      organization: config.organization,
      baseURL: config.baseURL,
      dimensions: config.dimensions,
    };

    try {
      // Dynamic import to support optional peer dependency
      const OpenAI = require('openai');
      this.openai = new OpenAI({
        apiKey: this.config.apiKey,
        organization: this.config.organization,
        baseURL: this.config.baseURL,
      });
    } catch (error) {
      throw new Error(
        'OpenAI SDK not found. Install it with: npm install openai'
      );
    }
  }

  getMaxBatchSize(): number {
    // OpenAI supports up to 2048 inputs per request
    return 2048;
  }

  getDimension(): number {
    // Return configured dimensions or default based on model
    if (this.config.dimensions) {
      return this.config.dimensions;
    }

    switch (this.config.model) {
      case 'text-embedding-3-small':
        return 1536;
      case 'text-embedding-3-large':
        return 3072;
      case 'text-embedding-ada-002':
        return 1536;
      default:
        return 1536;
    }
  }

  async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
    if (texts.length === 0) {
      return { embeddings: [] };
    }

    const batches = this.createBatches(texts);
    const allResults: EmbeddingResult[] = [];
    let totalTokens = 0;

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      const baseIndex = batchIndex * this.getMaxBatchSize();

      const response = await this.withRetry(
        async () => {
          const params: any = {
            model: this.config.model,
            input: batch,
          };

          if (this.config.dimensions) {
            params.dimensions = this.config.dimensions;
          }

          return await this.openai.embeddings.create(params);
        },
        `OpenAI embeddings for batch ${batchIndex + 1}/${batches.length}`
      );

      totalTokens += response.usage?.total_tokens || 0;

      for (const item of response.data) {
        allResults.push({
          embedding: item.embedding,
          index: baseIndex + item.index,
          tokens: response.usage?.total_tokens,
        });
      }
    }

    return {
      embeddings: allResults,
      totalTokens,
      metadata: {
        model: this.config.model,
        provider: 'openai',
      },
    };
  }
}

// ============================================================================
// Cohere Embeddings Provider
// ============================================================================

/**
 * Configuration for Cohere embeddings
 */
export interface CohereEmbeddingsConfig {
  /** Cohere API key */
  apiKey: string;
  /** Model name (default: 'embed-english-v3.0') */
  model?: string;
  /** Input type: 'search_document', 'search_query', 'classification', or 'clustering' */
  inputType?: 'search_document' | 'search_query' | 'classification' | 'clustering';
  /** Truncate input text if it exceeds model limits */
  truncate?: 'NONE' | 'START' | 'END';
  /** Retry configuration */
  retryConfig?: Partial<RetryConfig>;
}

/**
 * Cohere embeddings provider
 * Supports embed-english-v3.0, embed-multilingual-v3.0, and other Cohere models
 */
export class CohereEmbeddings extends EmbeddingProvider {
  private config: {
    apiKey: string;
    model: string;
    inputType?: 'search_document' | 'search_query' | 'classification' | 'clustering';
    truncate?: 'NONE' | 'START' | 'END';
  };
  private cohere: any;

  /**
   * Creates a new Cohere embeddings provider
   * @param config - Configuration options
   * @throws Error if Cohere SDK is not installed
   */
  constructor(config: CohereEmbeddingsConfig) {
    super(config.retryConfig);

    this.config = {
      apiKey: config.apiKey,
      model: config.model || 'embed-english-v3.0',
      inputType: config.inputType,
      truncate: config.truncate,
    };

    try {
      // Dynamic import to support optional peer dependency
      const { CohereClient } = require('cohere-ai');
      this.cohere = new CohereClient({
        token: this.config.apiKey,
      });
    } catch (error) {
      throw new Error(
        'Cohere SDK not found. Install it with: npm install cohere-ai'
      );
    }
  }

  getMaxBatchSize(): number {
    // Cohere supports up to 96 texts per request
    return 96;
  }

  getDimension(): number {
    // Cohere v3 models produce 1024-dimensional embeddings
    if (this.config.model.includes('v3')) {
      return 1024;
    }
    // Earlier models use different dimensions
    return 4096;
  }

  async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
    if (texts.length === 0) {
      return { embeddings: [] };
    }

    const batches = this.createBatches(texts);
    const allResults: EmbeddingResult[] = [];

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      const baseIndex = batchIndex * this.getMaxBatchSize();

      const response = await this.withRetry(
        async () => {
          const params: any = {
            model: this.config.model,
            texts: batch,
          };

          if (this.config.inputType) {
            params.inputType = this.config.inputType;
          }

          if (this.config.truncate) {
            params.truncate = this.config.truncate;
          }

          return await this.cohere.embed(params);
        },
        `Cohere embeddings for batch ${batchIndex + 1}/${batches.length}`
      );

      for (let i = 0; i < response.embeddings.length; i++) {
        allResults.push({
          embedding: response.embeddings[i],
          index: baseIndex + i,
        });
      }
    }

    return {
      embeddings: allResults,
      metadata: {
        model: this.config.model,
        provider: 'cohere',
      },
    };
  }
}

// ============================================================================
// Anthropic Embeddings Provider
// ============================================================================

/**
 * Configuration for Anthropic embeddings via Voyage AI
 */
export interface AnthropicEmbeddingsConfig {
  /** Anthropic API key */
  apiKey: string;
  /** Model name (default: 'voyage-2') */
  model?: string;
  /** Input type for embeddings */
  inputType?: 'document' | 'query';
  /** Retry configuration */
  retryConfig?: Partial<RetryConfig>;
}

/**
 * Anthropic embeddings provider using Voyage AI
 * Anthropic partners with Voyage AI for embeddings
 */
export class AnthropicEmbeddings extends EmbeddingProvider {
  private config: {
    apiKey: string;
    model: string;
    inputType?: 'document' | 'query';
  };
  private anthropic: any;

  /**
   * Creates a new Anthropic embeddings provider
   * @param config - Configuration options
   * @throws Error if Anthropic SDK is not installed
   */
  constructor(config: AnthropicEmbeddingsConfig) {
    super(config.retryConfig);

    this.config = {
      apiKey: config.apiKey,
      model: config.model || 'voyage-2',
      inputType: config.inputType,
    };

    try {
      const Anthropic = require('@anthropic-ai/sdk');
      this.anthropic = new Anthropic({
        apiKey: this.config.apiKey,
      });
    } catch (error) {
      throw new Error(
        'Anthropic SDK not found. Install it with: npm install @anthropic-ai/sdk'
      );
    }
  }

  getMaxBatchSize(): number {
    // Process in smaller batches for Voyage API
    return 128;
  }

  getDimension(): number {
    // Voyage-2 produces 1024-dimensional embeddings
    return 1024;
  }

  async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
    if (texts.length === 0) {
      return { embeddings: [] };
    }

    const batches = this.createBatches(texts);
    const allResults: EmbeddingResult[] = [];

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      const baseIndex = batchIndex * this.getMaxBatchSize();

      // Note: As of early 2025, Anthropic uses Voyage AI for embeddings
      // This is a placeholder for when official API is available
      const response = await this.withRetry(
        async () => {
          // Use Voyage AI API through Anthropic's recommended integration
          const httpResponse = await fetch('https://api.voyageai.com/v1/embeddings', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${this.config.apiKey}`,
            },
            body: JSON.stringify({
              input: batch,
              model: this.config.model,
              input_type: this.config.inputType || 'document',
            }),
          });

          if (!httpResponse.ok) {
            const error = await httpResponse.text();
            throw new Error(`Voyage API error: ${error}`);
          }

          return await httpResponse.json() as { data: Array<{ embedding: number[] }> };
        },
        `Anthropic/Voyage embeddings for batch ${batchIndex + 1}/${batches.length}`
      );

      for (let i = 0; i < response.data.length; i++) {
        allResults.push({
          embedding: response.data[i].embedding,
          index: baseIndex + i,
        });
      }
    }

    return {
      embeddings: allResults,
      metadata: {
        model: this.config.model,
        provider: 'anthropic-voyage',
      },
    };
  }
}

// ============================================================================
// HuggingFace Local Embeddings Provider
// ============================================================================

/**
 * Configuration for HuggingFace local embeddings
 */
export interface HuggingFaceEmbeddingsConfig {
  /** Model name or path (default: 'sentence-transformers/all-MiniLM-L6-v2') */
  model?: string;
  /** Device to run on: 'cpu' or 'cuda' */
  device?: 'cpu' | 'cuda';
  /** Normalize embeddings to unit length */
  normalize?: boolean;
  /** Batch size for processing */
  batchSize?: number;
  /** Retry configuration */
  retryConfig?: Partial<RetryConfig>;
}

/**
 * HuggingFace local embeddings provider
 * Runs embedding models locally using transformers.js
 */
export class HuggingFaceEmbeddings extends EmbeddingProvider {
  private config: {
    model: string;
    normalize: boolean;
    batchSize: number;
  };
  private pipeline: any;
  private initialized: boolean = false;

  /**
   * Creates a new HuggingFace local embeddings provider
   * @param config - Configuration options
   */
  constructor(config: HuggingFaceEmbeddingsConfig = {}) {
    super(config.retryConfig);

    this.config = {
      model: config.model || 'Xenova/all-MiniLM-L6-v2',
      normalize: config.normalize !== false,
      batchSize: config.batchSize || 32,
    };
  }

  getMaxBatchSize(): number {
    return this.config.batchSize;
  }

  getDimension(): number {
    // all-MiniLM-L6-v2 produces 384-dimensional embeddings
    // This should be determined dynamically based on model
    return 384;
  }

  /**
   * Initialize the embedding pipeline
   */
  private async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Dynamic import of transformers.js
      const { pipeline } = await import('@xenova/transformers');

      this.pipeline = await pipeline(
        'feature-extraction',
        this.config.model
      );

      this.initialized = true;
    } catch (error) {
      throw new Error(
        'Transformers.js not found or failed to load. Install it with: npm install @xenova/transformers'
      );
    }
  }

  async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
    if (texts.length === 0) {
      return { embeddings: [] };
    }

    await this.initialize();

    const batches = this.createBatches(texts);
    const allResults: EmbeddingResult[] = [];

    for (let batchIndex = 0; batchIndex < batches.length; batchIndex++) {
      const batch = batches[batchIndex];
      const baseIndex = batchIndex * this.getMaxBatchSize();

      const embeddings = await this.withRetry(
        async () => {
          const output = await this.pipeline(batch, {
            pooling: 'mean',
            normalize: this.config.normalize,
          });

          // Convert tensor to array
          return output.tolist();
        },
        `HuggingFace embeddings for batch ${batchIndex + 1}/${batches.length}`
      );

      for (let i = 0; i < embeddings.length; i++) {
        allResults.push({
          embedding: embeddings[i],
          index: baseIndex + i,
        });
      }
    }

    return {
      embeddings: allResults,
      metadata: {
        model: this.config.model,
        provider: 'huggingface-local',
      },
    };
  }
}

// ============================================================================
// Lattice WASM Embeddings Provider
// ============================================================================

/**
 * Configuration for lattice-embed-wasm local embeddings
 */
export interface LatticeWasmEmbeddingsConfig {
  /** Model name (default: 'minilm'). See the supported-model list in the class doc. */
  model?: string;
  /** Retry configuration */
  retryConfig?: Partial<RetryConfig>;
}

/**
 * Known lattice-embed-wasm models and their output dimensionality.
 *
 * `bge-small` is an asymmetric retriever (see
 * {@link LATTICE_WASM_QUERY_INSTRUCTIONS} below); `minilm` is genuinely
 * symmetric (contrastive training on raw text, no prefix on either side).
 */
const LATTICE_WASM_MODEL_DIMENSIONS: Record<string, number> = {
  minilm: 384,
  'bge-small': 384,
};

/**
 * Query-side retrieval-instruction prefixes, keyed by the canonical model
 * name from {@link LATTICE_WASM_MODEL_DIMENSIONS}. Mirrors
 * `lattice_embed::EmbeddingModel::query_instruction()` for
 * `BgeSmallEnV15` (`crates/embed/src/model.rs` in ohdearquant/lattice) and
 * `LatticeEmbedding::embed_query` in `ruvector-core`'s own Lattice provider
 * (`crates/ruvector-core/src/embeddings.rs`) -- BGE-v1.5 is an asymmetric
 * retriever: the query side gets this instruction prefix, the passage side
 * stays raw. `minilm` has no entry here (symmetric, no prefix).
 */
const LATTICE_WASM_QUERY_INSTRUCTIONS: Record<string, string> = {
  'bge-small': 'Represent this sentence for searching relevant passages: ',
};

/**
 * Applies the model's query-side retrieval-instruction prefix, if it has
 * one. A pure function (exported, in addition to being used internally by
 * {@link LatticeWasmEmbeddings.embedQuery}) so the query/passage asymmetry
 * is unit-testable without depending on the optional
 * `@khive-ai/lattice-embed-wasm` peer package.
 *
 * @param model - A canonical model name from
 *   {@link LATTICE_WASM_MODEL_DIMENSIONS} (e.g. as returned by
 *   {@link LatticeWasmEmbeddings.getModel}), not a raw user-supplied alias.
 */
export function applyLatticeWasmQueryPrefix(model: string, text: string): string {
  const prefix = LATTICE_WASM_QUERY_INSTRUCTIONS[model];
  return prefix ? `${prefix}${text}` : text;
}

/**
 * Normalizes a user-supplied lattice-embed-wasm model identifier to one of
 * the canonical keys in {@link LATTICE_WASM_MODEL_DIMENSIONS} ('minilm' or
 * 'bge-small'). Accepts the same alias surface lattice-embed's own
 * `EmbeddingModel::from_str` accepts for these two model families --
 * case/underscore-insensitive, HuggingFace ids, short names (see
 * `crates/embed/src/model.rs` in ohdearquant/lattice) -- so a model id
 * valid for the Rust `LatticeEmbedding` provider in `ruvector-core` is also
 * valid here (#662). Returns `undefined` for anything else.
 */
export function normalizeLatticeWasmModel(model: string): string | undefined {
  const normalized = model.toLowerCase().trim().replace(/_/g, '-').replace('baai/', '');

  switch (normalized) {
    case 'bge-small-en-v1.5':
    case 'bge-small-en':
    case 'bge-small':
    case 'small':
      return 'bge-small';
    case 'all-minilm-l6-v2':
    case 'minilm':
    case 'all-minilm':
    case 'sentence-transformers/all-minilm-l6-v2':
      return 'minilm';
    default:
      return undefined;
  }
}

/**
 * Local embeddings provider backed by `@khive-ai/lattice-embed-wasm`, a
 * pure-Rust BERT-family text embedder compiled to WebAssembly.
 *
 * Node-only: the wasm adapter resolves model weights via `node:fs`, so this
 * provider does not run in a browser. Single-text only: the wasm package
 * exposes no batch API, so {@link getMaxBatchSize} honestly reports `1`
 * rather than simulating a larger batch. Output embeddings are
 * L2-normalized (matching the `native` lattice-embed path's guarantee --
 * `BertModel::encode`/`encode_batch` call `l2_normalize` unconditionally,
 * see `crates/inference/src/model/bert.rs` in ohdearquant/lattice).
 *
 * `bge-small` is an asymmetric retriever: {@link embedQuery} applies the
 * BGE-v1.5 retrieval-instruction prefix, {@link embedText}/{@link embedTexts}
 * do not (passage side, matching `LatticeEmbedding::embed`/`embed_query` in
 * ruvector-core's Rust provider and `lattice-embed`'s own
 * `query_instruction()`/`document_instruction()` split). `minilm` is
 * genuinely symmetric, so its query and passage embeddings are identical.
 *
 * This is a convenience local-embedder option at parity with the existing
 * `HuggingFaceEmbeddings` (transformers.js) local path above, not a faster
 * or higher-quality alternative to it.
 */
export class LatticeWasmEmbeddings extends EmbeddingProvider {
  private model: string;
  private dimension: number;

  /**
   * Creates a new lattice-embed-wasm local embeddings provider
   * @param config - Configuration options
   * @throws Error if the configured model is not one lattice-embed-wasm supports
   */
  constructor(config: LatticeWasmEmbeddingsConfig = {}) {
    super(config.retryConfig);

    const requested = config.model || 'minilm';
    const model = normalizeLatticeWasmModel(requested);

    if (model === undefined) {
      throw new Error(
        `Unknown lattice-embed-wasm model "${requested}". Supported models: ${Object.keys(
          LATTICE_WASM_MODEL_DIMENSIONS
        ).join(', ')} (aliases accepted, e.g. "bge-small-en-v1.5", "BAAI/bge-small-en-v1.5").`
      );
    }

    this.model = model;
    this.dimension = LATTICE_WASM_MODEL_DIMENSIONS[model];
  }

  getMaxBatchSize(): number {
    // lattice-embed-wasm embeds one text per call; there is no batch API to
    // wrap, so this honestly reports 1 rather than faking a larger batch.
    return 1;
  }

  getDimension(): number {
    return this.dimension;
  }

  /**
   * The canonical model name this provider resolved to ('minilm' or
   * 'bge-small'), regardless of which accepted alias the caller passed to
   * the constructor.
   */
  getModel(): string {
    return this.model;
  }

  /**
   * Embed **query** text, applying the model's retrieval-instruction prefix
   * if it uses one (currently only `bge-small`; `minilm` is symmetric and
   * this passes the text through unchanged). Mirrors
   * `LatticeEmbedding::embed_query` in ruvector-core's Rust provider and
   * `lattice-embed`'s own `EmbeddingModel::query_instruction()`. Use
   * {@link embedText}/{@link embedTexts} for the passage/document side (no
   * prefix applied there).
   */
  async embedQuery(text: string): Promise<number[]> {
    const prefixed = applyLatticeWasmQueryPrefix(this.model, text);
    const result = await this.embedTexts([prefixed]);
    return result.embeddings[0].embedding;
  }

  async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
    if (texts.length === 0) {
      return { embeddings: [] };
    }

    let lattice: any;
    try {
      // Dynamic import to support optional peer dependency. The specifier is
      // read from a variable (rather than passed as a literal) so
      // TypeScript does not attempt to statically resolve module types for
      // a peer that may not be installed.
      const specifier = '@khive-ai/lattice-embed-wasm';
      lattice = await import(specifier);
    } catch (error) {
      throw new Error(
        'lattice-embed-wasm not found. Install it with: npm install @khive-ai/lattice-embed-wasm'
      );
    }

    // No batch API: embed one text at a time. Node caches the dynamic
    // import above by specifier, so repeated calls do not re-import.
    const allResults: EmbeddingResult[] = [];

    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];

      const vector = await this.withRetry(async () => {
        const result = await lattice.embed(text, this.model);

        if (result === null) {
          // For an explicitly-selected provider, a null result is a real
          // failure (unknown model, unsupported over wasm, or weights that
          // could not be resolved/verified) -- never a silent skip.
          throw new Error(
            `lattice-embed-wasm returned null embedding for model "${this.model}": the model is unknown, unsupported over wasm, or its weights could not be resolved`
          );
        }

        return result as Float32Array;
      }, `lattice-embed-wasm embedding for text ${i + 1}/${texts.length}`);

      allResults.push({
        embedding: Array.from(vector),
        index: i,
      });
    }

    return {
      embeddings: allResults,
      metadata: {
        model: this.model,
        provider: 'lattice-wasm',
      },
    };
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Embed texts and automatically insert them into a VectorDB
 *
 * @param db - VectorDB instance to insert into
 * @param provider - Embedding provider to use
 * @param documents - Documents to embed and insert
 * @param options - Additional options
 * @returns Promise resolving to array of inserted vector IDs
 *
 * @example
 * ```typescript
 * const openai = new OpenAIEmbeddings({ apiKey: 'sk-...' });
 * const db = new VectorDB({ dimension: 1536 });
 *
 * const ids = await embedAndInsert(db, openai, [
 *   { id: '1', text: 'Hello world', metadata: { source: 'test' } },
 *   { id: '2', text: 'Another document', metadata: { source: 'test' } }
 * ]);
 *
 * console.log('Inserted vector IDs:', ids);
 * ```
 */
export async function embedAndInsert(
  db: VectorDB,
  provider: EmbeddingProvider,
  documents: DocumentToEmbed[],
  options: {
    /** Whether to overwrite existing vectors with same ID */
    overwrite?: boolean;
    /** Progress callback */
    onProgress?: (current: number, total: number) => void;
  } = {}
): Promise<string[]> {
  if (documents.length === 0) {
    return [];
  }

  // Verify dimension compatibility
  const dbDimension = (db as any).dimension || db.getDimension?.();
  const providerDimension = provider.getDimension();

  if (dbDimension && dbDimension !== providerDimension) {
    throw new Error(
      `Dimension mismatch: VectorDB expects ${dbDimension} but provider produces ${providerDimension}`
    );
  }

  // Extract texts
  const texts = documents.map(doc => doc.text);

  // Generate embeddings
  const result = await provider.embedTexts(texts);

  // Insert vectors
  const insertedIds: string[] = [];

  for (let i = 0; i < documents.length; i++) {
    const doc = documents[i];
    const embedding = result.embeddings.find(e => e.index === i);

    if (!embedding) {
      throw new Error(`Missing embedding for document at index ${i}`);
    }

    // Insert or update vector
    if (options.overwrite) {
      await db.upsert({
        id: doc.id,
        values: embedding.embedding,
        metadata: doc.metadata,
      });
    } else {
      await db.insert({
        id: doc.id,
        values: embedding.embedding,
        metadata: doc.metadata,
      });
    }

    insertedIds.push(doc.id);

    // Call progress callback
    if (options.onProgress) {
      options.onProgress(i + 1, documents.length);
    }
  }

  return insertedIds;
}

/**
 * Embed a query and search for similar documents in VectorDB
 *
 * @param db - VectorDB instance to search
 * @param provider - Embedding provider to use
 * @param query - Query text to search for
 * @param options - Search options
 * @returns Promise resolving to search results
 *
 * @example
 * ```typescript
 * const openai = new OpenAIEmbeddings({ apiKey: 'sk-...' });
 * const db = new VectorDB({ dimension: 1536 });
 *
 * const results = await embedAndSearch(db, openai, 'machine learning', {
 *   topK: 5,
 *   threshold: 0.7
 * });
 *
 * console.log('Found documents:', results);
 * ```
 */
export async function embedAndSearch(
  db: VectorDB,
  provider: EmbeddingProvider,
  query: string,
  options: {
    /** Number of results to return */
    topK?: number;
    /** Minimum similarity threshold (0-1) */
    threshold?: number;
    /** Metadata filter */
    filter?: Record<string, unknown>;
  } = {}
): Promise<any[]> {
  // Generate query embedding
  const queryEmbedding = await provider.embedText(query);

  // Search VectorDB
  const results = await db.search({
    vector: queryEmbedding,
    topK: options.topK || 10,
    threshold: options.threshold,
    filter: options.filter,
  });

  return results;
}

// ============================================================================
// Exports
// ============================================================================

export default {
  // Base class
  EmbeddingProvider,

  // Providers
  OpenAIEmbeddings,
  CohereEmbeddings,
  AnthropicEmbeddings,
  HuggingFaceEmbeddings,
  LatticeWasmEmbeddings,

  // Helper functions
  embedAndInsert,
  embedAndSearch,
};
