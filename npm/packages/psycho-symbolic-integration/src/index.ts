/**
 * psycho-symbolic-integration
 *
 * Unified integration layer combining:
 * - psycho-symbolic-reasoner: Ultra-fast symbolic AI reasoning (0.3ms queries)
 * - ruvector: High-performance vector database
 * - agentic-synth: AI-powered synthetic data generation
 *
 * This package enables:
 * 1. Reasoning-guided synthetic data generation
 * 2. Vector-enhanced symbolic queries
 * 3. Psychological validation of generated data
 * 4. Goal-oriented planning for data strategies
 */

import { PsychoSymbolicReasoner } from 'psycho-symbolic-reasoner';
import { AgenticSynth } from '@ruvector/agentic-synth';
import { RuvectorAdapter } from './adapters/ruvector-adapter.js';
import { AgenticSynthAdapter } from './adapters/agentic-synth-adapter.js';

export { RuvectorAdapter, AgenticSynthAdapter };

export interface IntegratedSystemConfig {
  // Psycho-Symbolic Reasoner config
  reasoner?: {
    enableGraphReasoning?: boolean;
    enableAffectExtraction?: boolean;
    enablePlanning?: boolean;
    logLevel?: 'debug' | 'info' | 'warn' | 'error';
  };

  // Agentic-Synth config
  synth?: {
    provider?: 'gemini' | 'openrouter';
    apiKey?: string;
    model?: string;
    cache?: {
      enabled?: boolean;
      maxSize?: number;
    };
  };

  // Ruvector config (optional)
  vector?: {
    dbPath?: string;
    collectionName?: string;
    dimensions?: number;
    enableSemanticCache?: boolean;
  };
}

/**
 * Integrated Psycho-Symbolic System
 *
 * Combines all three packages into a unified interface for:
 * - Intelligent data generation
 * - Fast symbolic reasoning
 * - Vector-based semantic search
 */
export class IntegratedPsychoSymbolicSystem {
  public reasoner: PsychoSymbolicReasoner;
  public synth: AgenticSynth;
  public ruvectorAdapter?: RuvectorAdapter;
  public synthAdapter: AgenticSynthAdapter;

  private initialized: boolean = false;

  constructor(config: IntegratedSystemConfig = {}) {
    // Initialize psycho-symbolic reasoner
    this.reasoner = new PsychoSymbolicReasoner({
      enableGraphReasoning: config.reasoner?.enableGraphReasoning ?? true,
      enableAffectExtraction: config.reasoner?.enableAffectExtraction ?? true,
      enablePlanning: config.reasoner?.enablePlanning ?? true,
      logLevel: config.reasoner?.logLevel || 'info'
    });

    // Initialize agentic-synth
    this.synth = new AgenticSynth({
      provider: config.synth?.provider || 'gemini',
      apiKey: config.synth?.apiKey || process.env.GEMINI_API_KEY,
      model: config.synth?.model,
      cacheStrategy: config.synth?.cache?.enabled ? 'memory' : 'none',
      maxCacheSize: config.synth?.cache?.maxSize
    });

    // Initialize adapters
    this.synthAdapter = new AgenticSynthAdapter(this.reasoner, this.synth);

    if (config.vector) {
      this.ruvectorAdapter = new RuvectorAdapter(this.reasoner, {
        dbPath: config.vector.dbPath || './data/psycho-vector.db',
        collectionName: config.vector.collectionName || 'psycho-knowledge',
        embeddingDimensions: config.vector.dimensions || 768,
        enableSemanticCache: config.vector.enableSemanticCache ?? true
      });
    }
  }

  /**
   * Initialize all components
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    console.log('🚀 Initializing Integrated Psycho-Symbolic System...');

    // Initialize reasoner
    await this.reasoner.initialize();
    console.log('✅ Psycho-Symbolic Reasoner initialized');

    // Initialize vector adapter if available
    if (this.ruvectorAdapter?.isAvailable()) {
      await this.ruvectorAdapter.initialize();
      console.log('✅ Ruvector adapter initialized');
    }

    this.initialized = true;
    console.log('✨ System ready!');
  }

  /**
   * Generate synthetic data with psychological reasoning
   *
   * Example:
   * ```typescript
   * const result = await system.generateIntelligently('structured', {
   *   count: 100,
   *   schema: { name: 'string', age: 'number' }
   * }, {
   *   targetSentiment: { score: 0.7, emotion: 'happy' },
   *   userPreferences: ['I prefer concise data', 'Focus on quality over quantity']
   * });
   * ```
   */
  async generateIntelligently(
    type: 'timeseries' | 'events' | 'structured',
    baseOptions: any,
    psychoConfig: any = {}
  ): Promise<any> {
    if (!this.initialized) {
      await this.initialize();
    }

    return await this.synthAdapter.generateWithPsychoGuidance(
      type,
      baseOptions,
      psychoConfig
    );
  }

  /**
   * Perform hybrid reasoning query (symbolic + vector)
   *
   * Example:
   * ```typescript
   * const results = await system.intelligentQuery(
   *   'Find activities that reduce stress',
   *   { symbolicWeight: 0.6, vectorWeight: 0.4 }
   * );
   * ```
   */
  async intelligentQuery(
    query: string,
    options: {
      symbolicWeight?: number;
      vectorWeight?: number;
      maxResults?: number;
    } = {}
  ): Promise<any> {
    if (!this.initialized) {
      await this.initialize();
    }

    if (this.ruvectorAdapter?.isAvailable()) {
      return await this.ruvectorAdapter.hybridQuery(query, options);
    } else {
      // Fallback to pure symbolic reasoning
      return await this.reasoner.queryGraph({
        pattern: query,
        maxResults: options.maxResults || 10,
        includeInference: true
      });
    }
  }

  /**
   * Load knowledge base into both symbolic and vector stores
   */
  async loadKnowledgeBase(knowledgeBase: any): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    // Load into symbolic reasoner
    await this.reasoner.loadKnowledgeBase(knowledgeBase);

    // Store in vector database if available
    if (this.ruvectorAdapter?.isAvailable()) {
      await this.ruvectorAdapter.storeKnowledgeGraph(knowledgeBase);
    }
  }

  /**
   * Analyze text for sentiment and preferences
   */
  async analyzeText(text: string): Promise<{
    sentiment: any;
    preferences: any;
  }> {
    if (!this.initialized) {
      await this.initialize();
    }

    const [sentiment, preferences] = await Promise.all([
      this.reasoner.extractSentiment(text),
      this.reasoner.extractPreferences(text)
    ]);

    return { sentiment, preferences };
  }

  /**
   * Plan data generation strategy using GOAP
   */
  async planDataGeneration(goal: string, constraints: any): Promise<any> {
    if (!this.initialized) {
      await this.initialize();
    }

    return await this.synthAdapter.planGenerationStrategy(goal, constraints);
  }

  /**
   * Get system statistics and insights
   */
  getSystemInsights(): any {
    return {
      initialized: this.initialized,
      components: {
        reasoner: 'psycho-symbolic-reasoner',
        synth: 'agentic-synth',
        vector: this.ruvectorAdapter?.isAvailable() ? 'ruvector' : 'not-available'
      },
      adapters: {
        synthHistory: this.synthAdapter.getGenerationInsights(),
        vectorCache: this.ruvectorAdapter?.getCacheStats() || null
      }
    };
  }

  /**
   * Shutdown and cleanup
   */
  async shutdown(): Promise<void> {
    if (this.ruvectorAdapter) {
      this.ruvectorAdapter.clearCache();
    }
    this.synthAdapter.clearHistory();
    this.initialized = false;
  }
}

/**
 * Factory function for quick initialization
 */
export function createIntegratedSystem(config: IntegratedSystemConfig = {}): IntegratedPsychoSymbolicSystem {
  return new IntegratedPsychoSymbolicSystem(config);
}

/**
 * Quick start with defaults
 */
export async function quickStart(apiKey?: string): Promise<IntegratedPsychoSymbolicSystem> {
  const system = createIntegratedSystem({
    synth: {
      provider: 'gemini',
      apiKey: apiKey || process.env.GEMINI_API_KEY,
      cache: { enabled: true }
    },
    reasoner: {
      enableGraphReasoning: true,
      enableAffectExtraction: true,
      enablePlanning: true
    }
  });

  await system.initialize();
  return system;
}
