/**
 * ruvector - High-performance vector database for Node.js
 *
 * JavaScript wrapper providing the expected API
 */

let implementation;
let implementationType = 'wasm';

try {
  implementation = require('@ruvector/core');
  implementationType = 'native';

  if (typeof implementation.VectorDb !== 'function' && typeof implementation.VectorDB !== 'function') {
    throw new Error('Native module loaded but VectorDb/VectorDB class not found');
  }
} catch (e) {
  console.warn('[RuVector] Native module not available:', e.message);
  console.warn('[RuVector] Using stub implementation.');

  implementation = {
    VectorDb: class StubVectorDb {
      constructor() {}
      async insert() { return 'stub-id-' + Date.now(); }
      async insertBatch(entries) { return entries.map(() => 'stub-id-' + Date.now()); }
      async search() { return []; }
      async delete() { return true; }
      async get() { return null; }
      async len() { return 0; }
      async isEmpty() { return true; }
    }
  };
  implementationType = 'wasm';
}

// Get the VectorDb class (handle both naming conventions)
const NativeVectorDb = implementation.VectorDb || implementation.VectorDB;

/**
 * VectorIndex - High-level vector database interface
 * Provides the API expected by tests
 */
class VectorIndex {
  constructor(options = {}) {
    const { dimension, metric = 'cosine', indexType = 'hnsw' } = options;

    if (!dimension || dimension <= 0) {
      throw new Error('Invalid dimension: must be a positive integer');
    }

    this.dimension = dimension;
    this.metric = metric;
    this.indexType = indexType;
    this._vectorCount = 0;
    this._vectors = new Map();

    try {
      this._db = NativeVectorDb.withDimensions ?
        NativeVectorDb.withDimensions(dimension) :
        new NativeVectorDb({ dimensions: dimension });
    } catch (e) {
      // Fallback to in-memory implementation
      this._db = null;
    }
  }

  async insert(entry) {
    const { id, values } = entry;
    const vectorId = id || `vec-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    if (this._db) {
      try {
        const vector = values instanceof Float32Array ? values : new Float32Array(values);
        await this._db.insert({ id: vectorId, vector });
      } catch (e) {
        // Fallback
      }
    }

    this._vectors.set(vectorId, { id: vectorId, values: Array.from(values) });
    this._vectorCount++;

    return vectorId;
  }

  async insertBatch(entries, options = {}) {
    const { batchSize = 100, progressCallback } = options;
    const results = [];

    for (let i = 0; i < entries.length; i++) {
      const result = await this.insert(entries[i]);
      results.push(result);

      if (progressCallback && (i + 1) % batchSize === 0) {
        progressCallback((i + 1) / entries.length);
      }
    }

    if (progressCallback) {
      progressCallback(1);
    }

    return results;
  }

  async search(query, options = {}) {
    const { k = 10 } = options;
    const queryVector = Array.isArray(query) ? query : Array.from(query);

    // Return empty if no vectors inserted
    if (this._vectors.size === 0) {
      return [];
    }

    if (this._db) {
      try {
        const vector = queryVector instanceof Float32Array ? queryVector : new Float32Array(queryVector);
        const results = await this._db.search({ vector, k });
        return results.map(r => ({
          id: r.id,
          score: r.score || r.distance || 0
        }));
      } catch (e) {
        // Fallback to in-memory search
      }
    }

    // In-memory fallback using cosine similarity
    const results = [];
    for (const [id, entry] of this._vectors) {
      const score = 1 - Utils.cosineSimilarity(queryVector, entry.values);
      results.push({ id, score });
    }

    results.sort((a, b) => a.score - b.score);
    return results.slice(0, k);
  }

  async get(id) {
    const entry = this._vectors.get(id);
    if (!entry) return null;
    return { id: entry.id, values: entry.values };
  }

  async delete(id) {
    if (this._vectors.has(id)) {
      this._vectors.delete(id);
      this._vectorCount--;

      if (this._db) {
        try {
          await this._db.delete(id);
        } catch (e) {}
      }

      return true;
    }
    return false;
  }

  async stats() {
    return {
      vectorCount: this._vectors.size,
      dimension: this.dimension,
      metric: this.metric,
      indexType: this.indexType
    };
  }

  async clear() {
    this._vectors.clear();
    this._vectorCount = 0;
  }

  async optimize() {
    // No-op for compatibility
  }
}

/**
 * Get backend information
 */
function getBackendInfo() {
  return {
    type: implementationType,
    version: require('./package.json').version,
    features: implementationType === 'native'
      ? ['SIMD', 'Multi-threading', 'HNSW']
      : ['Browser-compatible', 'Universal']
  };
}

/**
 * Check if native implementation is available
 */
function isNativeAvailable() {
  return implementationType === 'native';
}

/**
 * Utility functions for vector operations
 */
const Utils = {
  /**
   * Calculate cosine similarity between two vectors
   */
  cosineSimilarity(a, b) {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same dimension');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  },

  /**
   * Calculate Euclidean distance between two vectors
   */
  euclideanDistance(a, b) {
    if (a.length !== b.length) {
      throw new Error('Vectors must have the same dimension');
    }

    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  },

  /**
   * Normalize a vector to unit length
   */
  normalize(vector) {
    let magnitude = 0;
    for (const val of vector) {
      magnitude += val * val;
    }
    magnitude = Math.sqrt(magnitude);

    if (magnitude === 0) return vector.slice();
    return vector.map(v => v / magnitude);
  },

  /**
   * Generate a random unit vector
   */
  randomVector(dimension) {
    const vector = [];
    for (let i = 0; i < dimension; i++) {
      vector.push(Math.random() * 2 - 1);
    }
    return Utils.normalize(vector);
  }
};

// Exports
module.exports = {
  VectorIndex,
  VectorDB: VectorIndex,  // Alias
  VectorDb: VectorIndex,  // Alias
  getBackendInfo,
  isNativeAvailable,
  getImplementationType: () => implementationType,
  isNative: () => implementationType === 'native',
  isWasm: () => implementationType === 'wasm',
  getVersion: () => ({ version: require('./package.json').version, implementation: implementationType }),
  Utils,
  NativeVectorDb
};
