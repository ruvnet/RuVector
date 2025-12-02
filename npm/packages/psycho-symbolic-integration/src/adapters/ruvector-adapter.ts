/**
 * Ruvector Integration Adapter for Psycho-Symbolic Reasoner
 *
 * Combines vector database capabilities with symbolic reasoning:
 * - Store knowledge graphs as vector embeddings
 * - Semantic search across reasoning results
 * - Hybrid symbolic-vector queries
 * - Memory persistence for reasoning sessions
 */

import { PsychoSymbolicReasoner } from 'psycho-symbolic-reasoner';

/**
 * LRU Cache for embeddings with memory limit
 * Prevents unbounded cache growth and memory leaks
 * Max size: 1000 entries (~6MB assuming 6KB per embedding)
 */
class LRUCache<K, V> {
  private cache: Map<K, V>;
  private maxSize: number;

  constructor(maxSize: number = 1000) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    if (!this.cache.has(key)) return undefined;

    // Move to end (most recently used)
    const value = this.cache.get(key)!;
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    // Remove if exists to reinsert at end
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }

    // Evict oldest if at capacity
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, value);
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }
}

export interface RuvectorConfig {
  dbPath: string;
  collectionName?: string;
  embeddingDimensions?: number;
  enableSemanticCache?: boolean;
}

export interface KnowledgeGraphEmbedding {
  id: string;
  nodeData: any;
  embedding: number[];
  metadata: {
    nodeType: string;
    relationships: string[];
    properties: Record<string, any>;
  };
}

export interface SemanticQueryResult {
  nodes: any[];
  score: number;
  reasoning: {
    symbolicMatch: number;
    semanticMatch: number;
    combinedScore: number;
  };
}

export class RuvectorAdapter {
  private reasoner: PsychoSymbolicReasoner;
  private vectorDB: any; // Ruvector instance (optional peer dependency)
  private config: RuvectorConfig;
  private embeddingCache: LRUCache<string, number[]>;
  private available: boolean = false;

  constructor(reasoner: PsychoSymbolicReasoner, config: RuvectorConfig) {
    this.reasoner = reasoner;
    this.config = config;
    // LRU cache with 1000 entry limit (~6MB max, prevents memory leaks)
    this.embeddingCache = new LRUCache(1000);
    this.detectAvailability();
  }

  /**
   * Detect if Ruvector is available
   */
  private detectAvailability(): void {
    try {
      // Dynamic import to handle optional dependency
      // @ts-ignore - optional peer dependency
      const { Ruvector } = require('ruvector');
      this.available = true;
    } catch {
      this.available = false;
      console.warn('Ruvector not available. Install with: npm install ruvector');
    }
  }

  /**
   * Check if adapter is available
   */
  isAvailable(): boolean {
    return this.available;
  }

  /**
   * Initialize vector database
   */
  async initialize(): Promise<void> {
    if (!this.available) {
      throw new Error('Ruvector is not available');
    }

    // @ts-ignore
    const { Ruvector } = require('ruvector');
    this.vectorDB = new Ruvector({
      path: this.config.dbPath,
      dimensions: this.config.embeddingDimensions || 768
    });

    await this.vectorDB.initialize();
  }

  /**
   * Store knowledge graph nodes as vectors
   */
  async storeKnowledgeGraph(knowledgeBase: any): Promise<void> {
    if (!this.available) {
      console.warn('Ruvector not available, skipping vector storage');
      return;
    }

    const embeddings: KnowledgeGraphEmbedding[] = [];

    for (const node of knowledgeBase.nodes) {
      // Generate embedding for node (using simple hash-based approach)
      // In production, use actual embedding model
      const embedding = await this.generateEmbedding(node);

      embeddings.push({
        id: node.id,
        nodeData: node,
        embedding,
        metadata: {
          nodeType: node.type,
          relationships: this.getNodeRelationships(node.id, knowledgeBase.edges),
          properties: node.properties || {}
        }
      });
    }

    // Batch insert to vector DB
    for (const emb of embeddings) {
      await this.vectorDB.insert({
        id: emb.id,
        vector: emb.embedding,
        metadata: emb.metadata
      });
    }
  }

  /**
   * Hybrid query: combine symbolic reasoning with vector search
   */
  async hybridQuery(query: string, options: {
    symbolicWeight?: number;
    vectorWeight?: number;
    maxResults?: number;
  } = {}): Promise<SemanticQueryResult[]> {
    const symbolicWeight = options.symbolicWeight || 0.6;
    const vectorWeight = options.vectorWeight || 0.4;
    const maxResults = options.maxResults || 10;

    // Perform symbolic reasoning
    const symbolicResults = await this.reasoner.queryGraph({
      pattern: query,
      maxResults,
      includeInference: true
    });

    if (!this.available) {
      // Return only symbolic results if vector DB not available
      return symbolicResults.nodes.map((node: any) => ({
        nodes: [node],
        score: symbolicWeight,
        reasoning: {
          symbolicMatch: 1.0,
          semanticMatch: 0.0,
          combinedScore: symbolicWeight
        }
      }));
    }

    // Perform vector search
    const queryEmbedding = await this.generateEmbedding({ text: query });
    const vectorResults = await this.vectorDB.search(queryEmbedding, {
      limit: maxResults
    });

    // Combine results
    const combinedResults: SemanticQueryResult[] = [];
    const nodeMap = new Map();

    // Add symbolic results
    for (const node of symbolicResults.nodes) {
      nodeMap.set(node.id, {
        nodes: [node],
        score: 0,
        reasoning: {
          symbolicMatch: 1.0,
          semanticMatch: 0.0,
          combinedScore: 0
        }
      });
    }

    // Merge with vector results
    for (const result of vectorResults) {
      const nodeId = result.id;
      if (nodeMap.has(nodeId)) {
        const existing = nodeMap.get(nodeId);
        existing.reasoning.semanticMatch = result.score;
        existing.reasoning.combinedScore =
          (symbolicWeight * existing.reasoning.symbolicMatch) +
          (vectorWeight * result.score);
      } else {
        nodeMap.set(nodeId, {
          nodes: [result.metadata],
          score: result.score,
          reasoning: {
            symbolicMatch: 0.0,
            semanticMatch: result.score,
            combinedScore: vectorWeight * result.score
          }
        });
      }
    }

    // Sort by combined score
    return Array.from(nodeMap.values())
      .sort((a, b) => b.reasoning.combinedScore - a.reasoning.combinedScore)
      .slice(0, maxResults);
  }

  /**
   * Store reasoning session in vector memory
   */
  async storeReasoningSession(sessionId: string, results: any): Promise<void> {
    if (!this.available) return;

    const embedding = await this.generateEmbedding(results);
    await this.vectorDB.insert({
      id: `session_${sessionId}`,
      vector: embedding,
      metadata: {
        type: 'reasoning_session',
        timestamp: Date.now(),
        results
      }
    });
  }

  /**
   * Retrieve similar reasoning sessions
   */
  async findSimilarSessions(query: any, limit: number = 5): Promise<any[]> {
    if (!this.available) return [];

    const embedding = await this.generateEmbedding(query);
    return await this.vectorDB.search(embedding, { limit });
  }

  /**
   * Generate embedding for content (simplified version)
   * In production, use proper embedding model
   */
  private async generateEmbedding(content: any): Promise<number[]> {
    const text = JSON.stringify(content);
    const cacheKey = text.substring(0, 100); // Cache based on first 100 chars

    if (this.embeddingCache.has(cacheKey)) {
      return this.embeddingCache.get(cacheKey)!;
    }

    // Simple hash-based embedding (replace with actual model in production)
    const dims = this.config.embeddingDimensions || 768;
    const embedding = new Array(dims).fill(0);

    for (let i = 0; i < text.length; i++) {
      const idx = text.charCodeAt(i) % dims;
      embedding[idx] += 1;
    }

    // Normalize
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    const normalized = embedding.map(val => val / (magnitude || 1));

    this.embeddingCache.set(cacheKey, normalized);
    return normalized;
  }

  /**
   * Get relationships for a node
   */
  private getNodeRelationships(nodeId: string, edges: any[]): string[] {
    return edges
      .filter(edge => edge.from === nodeId || edge.to === nodeId)
      .map(edge => `${edge.from}-${edge.relationship}-${edge.to}`);
  }

  /**
   * Clear embedding cache
   */
  clearCache(): void {
    this.embeddingCache.clear();
  }

  /**
   * Get cache statistics
   */
  getCacheStats() {
    return {
      size: this.embeddingCache.size,
      available: this.available
    };
  }
}
