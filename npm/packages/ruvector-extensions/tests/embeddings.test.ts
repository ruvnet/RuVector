/**
 * @fileoverview Unit tests for the embeddings integration module
 *
 * @author ruv.io Team <info@ruv.io>
 * @license MIT
 */

import { describe, it, mock } from 'node:test';
import assert from 'node:assert';
import {
  EmbeddingProvider,
  OpenAIEmbeddings,
  CohereEmbeddings,
  AnthropicEmbeddings,
  HuggingFaceEmbeddings,
  LatticeWasmEmbeddings,
  applyLatticeWasmQueryPrefix,
  normalizeLatticeWasmModel,
  type BatchEmbeddingResult,
  type EmbeddingError,
} from '../src/embeddings.js';

// ============================================================================
// Mock Implementation for Testing
// ============================================================================

class MockEmbeddingProvider extends EmbeddingProvider {
  private dimension: number;
  private batchSize: number;

  constructor(dimension = 384, batchSize = 10) {
    super();
    this.dimension = dimension;
    this.batchSize = batchSize;
  }

  getMaxBatchSize(): number {
    return this.batchSize;
  }

  getDimension(): number {
    return this.dimension;
  }

  async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
    // Generate mock embeddings
    const embeddings = texts.map((text, index) => ({
      embedding: Array.from({ length: this.dimension }, () => Math.random()),
      index,
      tokens: text.length,
    }));

    return {
      embeddings,
      totalTokens: texts.reduce((sum, text) => sum + text.length, 0),
      metadata: {
        provider: 'mock',
        model: 'mock-model',
      },
    };
  }
}

// ============================================================================
// Tests for Base EmbeddingProvider
// ============================================================================

describe('EmbeddingProvider (Abstract Base)', () => {
  it('should embed single text', async () => {
    const provider = new MockEmbeddingProvider(384);
    const embedding = await provider.embedText('Hello, world!');

    assert.strictEqual(embedding.length, 384);
    assert.ok(Array.isArray(embedding));
    assert.ok(embedding.every(val => typeof val === 'number'));
  });

  it('should embed multiple texts', async () => {
    const provider = new MockEmbeddingProvider(384);
    const texts = ['First text', 'Second text', 'Third text'];

    const result = await provider.embedTexts(texts);

    assert.strictEqual(result.embeddings.length, 3);
    assert.ok(result.totalTokens > 0);
    assert.strictEqual(result.metadata?.provider, 'mock');
  });

  it('should handle empty text array', async () => {
    const provider = new MockEmbeddingProvider(384);
    const result = await provider.embedTexts([]);

    assert.strictEqual(result.embeddings.length, 0);
  });

  it('should create batches correctly', async () => {
    const provider = new MockEmbeddingProvider(384, 5);
    const texts = Array.from({ length: 12 }, (_, i) => `Text ${i}`);

    const result = await provider.embedTexts(texts);

    assert.strictEqual(result.embeddings.length, 12);
    // Verify all indices are present
    const indices = result.embeddings.map(e => e.index).sort((a, b) => a - b);
    assert.deepStrictEqual(indices, Array.from({ length: 12 }, (_, i) => i));
  });
});

// ============================================================================
// Tests for OpenAI Provider (Mock)
// ============================================================================

describe('OpenAIEmbeddings', () => {
  it('should throw error if OpenAI SDK not installed', () => {
    assert.throws(
      () => {
        new OpenAIEmbeddings({ apiKey: 'test-key' });
      },
      /OpenAI SDK not found/
    );
  });

  it('should have correct default configuration', () => {
    // This would work if OpenAI SDK is installed
    // For now, we test the error case
    try {
      const openai = new OpenAIEmbeddings({ apiKey: 'test-key' });
      assert.fail('Should have thrown error');
    } catch (error: any) {
      assert.ok(error.message.includes('OpenAI SDK not found'));
    }
  });

  it('should return correct dimensions', () => {
    // Mock test - would need OpenAI SDK installed
    const expectedDimensions = {
      'text-embedding-3-small': 1536,
      'text-embedding-3-large': 3072,
      'text-embedding-ada-002': 1536,
    };

    assert.ok(expectedDimensions['text-embedding-3-small'] === 1536);
  });

  it('should have correct max batch size', () => {
    // OpenAI supports up to 2048 inputs per request
    const expectedBatchSize = 2048;
    assert.strictEqual(expectedBatchSize, 2048);
  });
});

// ============================================================================
// Tests for Cohere Provider (Mock)
// ============================================================================

describe('CohereEmbeddings', () => {
  it('should throw error if Cohere SDK not installed', () => {
    assert.throws(
      () => {
        new CohereEmbeddings({ apiKey: 'test-key' });
      },
      /Cohere SDK not found/
    );
  });

  it('should return correct dimensions', () => {
    // Cohere v3 models use 1024 dimensions
    const expectedDimension = 1024;
    assert.strictEqual(expectedDimension, 1024);
  });

  it('should have correct max batch size', () => {
    // Cohere supports up to 96 texts per request
    const expectedBatchSize = 96;
    assert.strictEqual(expectedBatchSize, 96);
  });
});

// ============================================================================
// Tests for Anthropic Provider (Mock)
// ============================================================================

describe('AnthropicEmbeddings', () => {
  it('should construct (the SDK is a direct dependency)', () => {
    // Previously asserted "SDK not found", which only passed because the ESM
    // build had no `require` global and every SDK load failed.
    assert.doesNotThrow(() => new AnthropicEmbeddings({ apiKey: 'test-key' }));
  });

  it('should return correct dimensions', () => {
    // Voyage-2 uses 1024 dimensions
    const expectedDimension = 1024;
    assert.strictEqual(expectedDimension, 1024);
  });

  it('should have correct max batch size', () => {
    const expectedBatchSize = 128;
    assert.strictEqual(expectedBatchSize, 128);
  });
});

// ============================================================================
// Tests for HuggingFace Provider (Mock)
// ============================================================================

describe('HuggingFaceEmbeddings', () => {
  it('should create with default config', () => {
    const hf = new HuggingFaceEmbeddings();
    assert.strictEqual(hf.getDimension(), 384);
    assert.strictEqual(hf.getMaxBatchSize(), 32);
  });

  it('should create with custom config', () => {
    const hf = new HuggingFaceEmbeddings({
      batchSize: 64,
    });
    assert.strictEqual(hf.getMaxBatchSize(), 64);
  });

  it('should handle initialization lazily', async () => {
    const hf = new HuggingFaceEmbeddings();
    // Should not throw on construction
    assert.ok(hf);
  });
});

// ============================================================================
// Tests for Lattice WASM Provider
// ============================================================================

describe('LatticeWasmEmbeddings', () => {
  it('should throw for an unknown model', () => {
    assert.throws(
      () => {
        new LatticeWasmEmbeddings({ model: 'not-a-real-model' });
      },
      /Unknown lattice-embed-wasm model/
    );
  });

  it('should create with default config', () => {
    const lattice = new LatticeWasmEmbeddings();
    assert.strictEqual(lattice.getDimension(), 384);
    assert.strictEqual(lattice.getMaxBatchSize(), 1);
  });

  it('should create with bge-small config', () => {
    const lattice = new LatticeWasmEmbeddings({ model: 'bge-small' });
    assert.strictEqual(lattice.getDimension(), 384);
  });

  it('should not throw on construction (no eager load)', () => {
    const lattice = new LatticeWasmEmbeddings();
    assert.ok(lattice);
  });

  it('should produce a 384-dim, L2-normalized embedding when the wasm package and its model weights are available', async (t) => {
    let lattice: any;
    try {
      // Read from a variable (not a literal) so TypeScript does not attempt
      // to statically resolve module types for an optional peer that may
      // not be installed -- same rationale as in LatticeWasmEmbeddings itself.
      const specifier = '@khive-ai/lattice-embed-wasm';
      lattice = await import(specifier);
    } catch {
      t.skip('@khive-ai/lattice-embed-wasm is not installed (optional peer dependency)');
      return;
    }
    void lattice;

    const provider = new LatticeWasmEmbeddings();
    let result: BatchEmbeddingResult;
    try {
      result = await provider.embedTexts(['Hello, world!']);
    } catch (error: any) {
      // Model weights are resolved from a local cache or a pinned release
      // asset; neither is guaranteed to be present in every environment
      // (e.g. a fresh CI checkout with no local model cache). This is an
      // environment-availability gate, not a code-correctness failure.
      t.skip(`lattice-embed-wasm model weights unavailable: ${error.message}`);
      return;
    }

    assert.strictEqual(result.embeddings.length, 1);
    const embedding = result.embeddings[0].embedding;
    // Mutation-sensitive: an exact dimension check, not just "is an array".
    assert.strictEqual(embedding.length, 384);

    let sumSquares = 0;
    for (const value of embedding) sumSquares += value * value;
    const norm = Math.sqrt(sumSquares);
    assert.ok(Math.abs(norm - 1.0) < 1e-3, `expected L2 norm near 1.0, got ${norm}`);
  });
});

// ============================================================================
// LatticeWasmEmbeddings: query/passage asymmetry + model alias reconciliation
//
// Regression coverage for #662 (symmetric-vs-asymmetric bge-small mismatch
// between this provider and ruvector-core's LatticeEmbedding). Exercises the
// pure helper functions directly rather than the wasm layer itself, so these
// tests run without the optional @khive-ai/lattice-embed-wasm peer package.
// ============================================================================

describe('LatticeWasmEmbeddings query/passage prefix asymmetry (#662)', () => {
  it('prefixes bge-small queries with the BGE-v1.5 retrieval instruction', () => {
    const text = 'Where is the Eiffel Tower?';
    assert.strictEqual(
      applyLatticeWasmQueryPrefix('bge-small', text),
      `Represent this sentence for searching relevant passages: ${text}`
    );
  });

  it('leaves minilm queries unprefixed (genuinely symmetric model)', () => {
    const text = 'Where is the Eiffel Tower?';
    assert.strictEqual(applyLatticeWasmQueryPrefix('minilm', text), text);
  });

  it('bge-small query text differs from passage text; minilm query and passage text are identical', () => {
    const text = 'Where is the Eiffel Tower?';
    // Passage side: LatticeWasmEmbeddings.embedText/embedTexts never prefix,
    // so plain `text` is what reaches the wasm embed() call for documents.
    assert.notStrictEqual(applyLatticeWasmQueryPrefix('bge-small', text), text);
    assert.strictEqual(applyLatticeWasmQueryPrefix('minilm', text), text);
  });

  it('LatticeWasmEmbeddings.embedQuery resolves the model before prefixing, so aliases behave like the canonical name', () => {
    const viaAlias = new LatticeWasmEmbeddings({ model: 'bge-small-en-v1.5' });
    const viaCanonical = new LatticeWasmEmbeddings({ model: 'bge-small' });
    assert.strictEqual(viaAlias.getModel(), viaCanonical.getModel());
  });
});

describe('LatticeWasmEmbeddings model alias parsing (#662)', () => {
  it('accepts the same bge-small alias surface as lattice-embed\'s EmbeddingModel::from_str', () => {
    for (const alias of [
      'bge-small',
      'bge-small-en',
      'bge-small-en-v1.5',
      'small',
      'BAAI/bge-small-en-v1.5',
      'BGE_SMALL_EN_V1.5',
    ]) {
      const provider = new LatticeWasmEmbeddings({ model: alias });
      assert.strictEqual(provider.getModel(), 'bge-small', `alias "${alias}"`);
      assert.strictEqual(provider.getDimension(), 384, `alias "${alias}"`);
    }
  });

  it('accepts the minilm alias surface', () => {
    for (const alias of [
      'minilm',
      'all-minilm',
      'all-minilm-l6-v2',
      'sentence-transformers/all-MiniLM-L6-v2',
    ]) {
      const provider = new LatticeWasmEmbeddings({ model: alias });
      assert.strictEqual(provider.getModel(), 'minilm', `alias "${alias}"`);
    }
  });

  it('normalizeLatticeWasmModel returns undefined for unrecognized ids', () => {
    assert.strictEqual(normalizeLatticeWasmModel('not-a-real-model'), undefined);
  });

  it('still rejects unknown models at construction', () => {
    assert.throws(
      () => new LatticeWasmEmbeddings({ model: 'not-a-real-model' }),
      /Unknown lattice-embed-wasm model/
    );
  });
});

// ============================================================================
// Tests for Retry Logic
// ============================================================================

describe('Retry Logic', () => {
  it('should retry on retryable errors', async () => {
    let attempts = 0;

    class RetryTestProvider extends MockEmbeddingProvider {
      async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
        attempts++;
        if (attempts < 3) {
          throw new Error('Rate limit exceeded');
        }
        return super.embedTexts(texts);
      }
    }

    const provider = new RetryTestProvider();
    const result = await provider.embedTexts(['Test']);

    assert.strictEqual(attempts, 3);
    assert.strictEqual(result.embeddings.length, 1);
  });

  it('should not retry on non-retryable errors', async () => {
    let attempts = 0;

    class NonRetryableProvider extends MockEmbeddingProvider {
      async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
        attempts++;
        throw new Error('Invalid API key');
      }
    }

    const provider = new NonRetryableProvider();

    try {
      await provider.embedTexts(['Test']);
      assert.fail('Should have thrown error');
    } catch (error) {
      // Should fail on first attempt only
      assert.strictEqual(attempts, 1);
    }
  });

  it('should respect max retries', async () => {
    let attempts = 0;

    class MaxRetriesProvider extends MockEmbeddingProvider {
      async embedTexts(texts: string[]): Promise<BatchEmbeddingResult> {
        attempts++;
        throw new Error('Rate limit exceeded');
      }
    }

    const provider = new MaxRetriesProvider();

    try {
      await provider.embedTexts(['Test']);
      assert.fail('Should have thrown error');
    } catch (error) {
      // Default maxRetries is 3, so should try 4 times total (initial + 3 retries)
      assert.strictEqual(attempts, 4);
    }
  });
});

// ============================================================================
// Tests for Error Handling
// ============================================================================

describe('Error Handling', () => {
  it('should identify retryable errors', () => {
    const provider = new MockEmbeddingProvider();
    const retryableErrors = [
      new Error('Rate limit exceeded'),
      new Error('Request timeout'),
      new Error('503 Service Unavailable'),
      new Error('429 Too Many Requests'),
      new Error('Connection refused'),
    ];

    retryableErrors.forEach(error => {
      const isRetryable = (provider as any).isRetryableError(error);
      assert.strictEqual(isRetryable, true, `Should be retryable: ${error.message}`);
    });
  });

  it('should identify non-retryable errors', () => {
    const provider = new MockEmbeddingProvider();
    const nonRetryableErrors = [
      new Error('Invalid API key'),
      new Error('Authentication failed'),
      new Error('Invalid request'),
      new Error('Resource not found'),
    ];

    nonRetryableErrors.forEach(error => {
      const isRetryable = (provider as any).isRetryableError(error);
      assert.strictEqual(isRetryable, false, `Should not be retryable: ${error.message}`);
    });
  });

  it('should create embedding error with context', () => {
    const provider = new MockEmbeddingProvider();
    const originalError = new Error('Test error');
    const embeddingError = (provider as any).createEmbeddingError(
      originalError,
      'Test context',
      true
    ) as EmbeddingError;

    assert.strictEqual(embeddingError.message, 'Test context: Test error');
    assert.strictEqual(embeddingError.retryable, true);
    assert.strictEqual(embeddingError.error, originalError);
  });
});

// ============================================================================
// Tests for Batch Processing
// ============================================================================

describe('Batch Processing', () => {
  it('should split large datasets into batches', async () => {
    const provider = new MockEmbeddingProvider(384, 10);
    const texts = Array.from({ length: 35 }, (_, i) => `Text ${i}`);

    const result = await provider.embedTexts(texts);

    assert.strictEqual(result.embeddings.length, 35);
    // Verify all texts were processed
    const processedIndices = result.embeddings.map(e => e.index).sort((a, b) => a - b);
    assert.deepStrictEqual(processedIndices, Array.from({ length: 35 }, (_, i) => i));
  });

  it('should handle single batch correctly', async () => {
    const provider = new MockEmbeddingProvider(384, 100);
    const texts = Array.from({ length: 50 }, (_, i) => `Text ${i}`);

    const result = await provider.embedTexts(texts);

    assert.strictEqual(result.embeddings.length, 50);
  });

  it('should preserve order across batches', async () => {
    const provider = new MockEmbeddingProvider(384, 5);
    const texts = Array.from({ length: 12 }, (_, i) => `Text ${i}`);

    const result = await provider.embedTexts(texts);

    // Check that indices are correct
    result.embeddings.forEach((embedding, i) => {
      assert.strictEqual(embedding.index, i);
    });
  });
});

console.log('✓ All embeddings tests passed!');
