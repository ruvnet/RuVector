/**
 * agentic-synth - AI-powered synthetic data generation
 *
 * @packageDocumentation
 */

import 'dotenv/config';
import {
  SynthConfig,
  SynthConfigSchema,
  GeneratorOptions,
  TimeSeriesOptions,
  EventOptions,
  GenerationResult,
  ModelProvider,
  DataType
} from './types';
import { TimeSeriesGenerator } from './generators/timeseries';
import { EventGenerator } from './generators/events';
import { StructuredGenerator } from './generators/structured';
import { CacheManager } from './cache/index';

function defaultApiKey(provider: ModelProvider): string | undefined {
  return provider === 'openrouter'
    ? process.env.OPENROUTER_API_KEY
    : process.env.GEMINI_API_KEY;
}

// OpenRouter has no fixed default: the router picks its top-priority route.
function defaultModel(provider: ModelProvider): string | undefined {
  return provider === 'gemini' ? 'gemini-2.5-flash' : undefined;
}

/**
 * Main AgenticSynth class for data generation
 */
export class AgenticSynth {
  private config: SynthConfig;
  private timeSeriesGen: TimeSeriesGenerator;
  private eventGen: EventGenerator;
  private structuredGen: StructuredGenerator;

  constructor(config: Partial<SynthConfig> = {}) {
    // Validate and merge config
    const provider = config.provider ?? 'gemini';
    const defaultConfig: SynthConfig = {
      provider,
      apiKey: defaultApiKey(provider),
      model: defaultModel(provider),
      cacheStrategy: 'memory',
      cacheTTL: 3600,
      maxRetries: 3,
      timeout: 30000,
      streaming: false,
      automation: false,
      vectorDB: false
    };

    this.config = SynthConfigSchema.parse({ ...defaultConfig, ...config });

    // Initialize generators
    this.timeSeriesGen = new TimeSeriesGenerator(this.config);
    this.eventGen = new EventGenerator(this.config);
    this.structuredGen = new StructuredGenerator(this.config);
  }

  /**
   * Generate time-series data
   */
  async generateTimeSeries<T = unknown>(
    options: Partial<TimeSeriesOptions> = {}
  ): Promise<GenerationResult<T>> {
    return this.timeSeriesGen.generate<T>(options as TimeSeriesOptions);
  }

  /**
   * Generate event data
   */
  async generateEvents<T = unknown>(
    options: Partial<EventOptions> = {}
  ): Promise<GenerationResult<T>> {
    return this.eventGen.generate<T>(options as EventOptions);
  }

  /**
   * Generate structured data
   */
  async generateStructured<T = unknown>(
    options: Partial<GeneratorOptions> = {}
  ): Promise<GenerationResult<T>> {
    return this.structuredGen.generate<T>(options as GeneratorOptions);
  }

  /**
   * Generate data by type
   */
  async generate<T = unknown>(
    type: DataType,
    options: Partial<GeneratorOptions> = {}
  ): Promise<GenerationResult<T>> {
    switch (type) {
      case 'timeseries':
        return this.generateTimeSeries<T>(options as TimeSeriesOptions);
      case 'events':
        return this.generateEvents<T>(options as EventOptions);
      case 'structured':
      case 'json':
        return this.generateStructured<T>(options);
      default:
        throw new Error(`Unsupported data type: ${type}`);
    }
  }

  /**
   * Generate with streaming
   */
  async *generateStream<T = unknown>(
    type: DataType,
    options: Partial<GeneratorOptions> = {}
  ): AsyncGenerator<T, void, unknown> {
    const generator = this.getGenerator(type);
    yield* generator.generateStream<T>(options as GeneratorOptions);
  }

  /**
   * Generate multiple batches in parallel
   */
  async generateBatch<T = unknown>(
    type: DataType,
    batchOptions: Partial<GeneratorOptions>[],
    concurrency: number = 3
  ): Promise<GenerationResult<T>[]> {
    const generator = this.getGenerator(type);
    return generator.generateBatch<T>(batchOptions as GeneratorOptions[], concurrency);
  }

  /**
   * Get generator for data type
   */
  private getGenerator(type: DataType) {
    switch (type) {
      case 'timeseries':
        return this.timeSeriesGen;
      case 'events':
        return this.eventGen;
      case 'structured':
      case 'json':
        return this.structuredGen;
      default:
        throw new Error(`Unsupported data type: ${type}`);
    }
  }

  /**
   * Configure instance
   */
  configure(config: Partial<SynthConfig>): void {
    const next = { ...this.config, ...config };
    // Switching provider without a new key: the old key belongs to the old
    // provider and must not be sent to the new one.
    if (config.apiKey === undefined && next.provider !== this.config.provider) {
      next.apiKey = defaultApiKey(next.provider);
    }
    if (config.model === undefined && next.provider !== this.config.provider) {
      next.model = defaultModel(next.provider);
    }
    this.config = SynthConfigSchema.parse(next);

    // Recreate generators with new config
    this.timeSeriesGen = new TimeSeriesGenerator(this.config);
    this.eventGen = new EventGenerator(this.config);
    this.structuredGen = new StructuredGenerator(this.config);
  }

  /**
   * Get current configuration
   */
  getConfig(): SynthConfig {
    return { ...this.config };
  }
}

/**
 * Create a new AgenticSynth instance
 */
export function createSynth(config?: Partial<SynthConfig>): AgenticSynth {
  return new AgenticSynth(config);
}

// Export types and utilities
export * from './types';
export * from './generators/index';
export * from './cache/index';
export * from './routing/index';

// Default export
export default AgenticSynth;
