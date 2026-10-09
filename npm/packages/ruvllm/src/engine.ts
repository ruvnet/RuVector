/**
 * RuvLLM Engine - Main orchestrator for self-learning LLM
 */

import {
  RuvLLMConfig,
  GenerationConfig,
  QueryResponse,
  RoutingDecision,
  MemoryResult,
  MemoryId,
  RuvLLMStats,
  Feedback,
  Embedding,
  BatchQueryRequest,
  BatchQueryResponse,
  ChatMessage,
  FinishReason,
  GenerationResult,
  LoadModelOptions,
  LoadedModelInfo,
} from './types';

import {
  getNativeModule,
  NativeEngine,
  NativeConfig,
  NativeGenConfig,
  NativeGenerationResult,
  NativeModelInfo,
} from './native';

/**
 * Convert JS config to native config format
 */
function toNativeConfig(config?: RuvLLMConfig): NativeConfig | undefined {
  if (!config) return undefined;

  return {
    embedding_dim: config.embeddingDim,
    router_hidden_dim: config.routerHiddenDim,
    hnsw_m: config.hnswM,
    hnsw_ef_construction: config.hnswEfConstruction,
    hnsw_ef_search: config.hnswEfSearch,
    learning_enabled: config.learningEnabled,
    quality_threshold: config.qualityThreshold,
    ewc_lambda: config.ewcLambda,
  };
}

/**
 * Convert JS generation config to native format
 */
function toNativeGenConfig(config?: GenerationConfig): NativeGenConfig | undefined {
  if (!config) return undefined;

  // camelCase: napi-rs ignores snake_case keys on `#[napi(object)]` inputs.
  return {
    maxTokens: config.maxTokens,
    temperature: config.temperature,
    topP: config.topP,
    topK: config.topK,
    repetitionPenalty: config.repetitionPenalty,
    stopSequences: config.stopSequences,
    seed: config.seed,
  };
}

function toGenerationResult(r: NativeGenerationResult): GenerationResult {
  return {
    text: r.text,
    promptTokens: r.promptTokens,
    completionTokens: r.completionTokens,
    finishReason: r.finishReason as FinishReason,
  };
}

function toModelInfo(i: NativeModelInfo): LoadedModelInfo {
  return {
    name: i.name,
    architecture: i.architecture,
    numParameters: i.numParameters,
    vocabSize: i.vocabSize,
    hiddenSize: i.hiddenSize,
    numLayers: i.numLayers,
    maxContextLength: i.maxContextLength,
    quantization: i.quantization ?? null,
    memoryUsageBytes: i.memoryUsageBytes,
    chatTemplate: i.chatTemplate ?? null,
  };
}

const warned = new Set<string>();

/** Warn once per process (or throw when `strict`). */
function notice(code: string, message: string, strict: boolean | undefined): void {
  if (strict) throw codedError(code, message);
  if (warned.has(code)) return;
  warned.add(code);
  const g = globalThis as unknown as {
    process?: { emitWarning?: (m: string, o?: { code?: string }) => void };
    console?: { warn?: (m: string) => void };
  };
  if (typeof g.process?.emitWarning === 'function') g.process.emitWarning(message, { code });
  else g.console?.warn?.(`${code}: ${message}`);
}

/** An Error whose message starts with, and whose `code` is, `code`. */
function codedError(code: string, message: string): Error & { code: string } {
  const err = new Error(`${code}: ${message}`) as Error & { code: string };
  err.code = code;
  return err;
}

/**
 * Call into the native engine, giving its errors the same `code` as the
 * wrapper's own: native messages start with `RUVLLM_<CODE>:`, but napi-rs
 * sets `code` to a generic status such as `GenericFailure`.
 */
function callNative<T>(f: () => T): T {
  try {
    return f();
  } catch (e) {
    const match = /^(RUVLLM_[A-Z_]+):/.exec(e instanceof Error ? e.message : '');
    if (match) (e as { code?: string }).code = match[1];
    throw e;
  }
}

function platformKey(): string {
  const p = (globalThis as { process?: { platform?: string; arch?: string } }).process;
  return `${p?.platform ?? 'unknown'}-${p?.arch ?? 'unknown'}`;
}

function envAllowsPlaceholder(): boolean {
  const p = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return p?.env?.RUVLLM_ALLOW_MOCK === '1';
}

/** Returned by generate()/query() only with `allowPlaceholder` / `RUVLLM_ALLOW_MOCK=1`. */
export const PLACEHOLDER_TEXT =
  '[RuvLLM placeholder: no language model is loaded; this text is not model output]';

/**
 * RuvLLM - Self-learning LLM orchestrator
 *
 * Combines SONA adaptive learning with HNSW memory, FastGRNN routing, and
 * GGUF inference through the native candle backend.
 *
 * Text comes only from a model loaded with `loadModel()` (or the `modelPath`
 * option). Without one, `generate()`, `query()`, `chat()` and
 * `generateDetailed()` throw `RUVLLM_NO_LANGUAGE_MODEL`.
 *
 * @example
 * ```typescript
 * import { RuvLLM } from '@ruvector/ruvllm';
 *
 * const llm = new RuvLLM();
 * llm.loadModel('./qwen2.5-0.5b-instruct-q4_k_m.gguf');
 *
 * const out = llm.chat([{ role: 'user', content: 'What is machine learning?' }], {
 *   maxTokens: 128,
 * });
 * console.log(out.text, out.finishReason, out.completionTokens);
 *
 * // Routed query, answered by the loaded model
 * const response = llm.query('What is machine learning?');
 * llm.feedback({ requestId: response.requestId, rating: 5 });
 * ```
 */
export class RuvLLM {
  private native: NativeEngine | null = null;
  private nativeVersion: string | null = null;
  private config: RuvLLMConfig;
  private allowPlaceholder: boolean;

  // Fallback state for when native module is not available
  private fallbackState = {
    memory: new Map<number, { content: string; embedding: number[]; metadata: Record<string, unknown> }>(),
    nextId: 1,
    queryCount: 0,
  };

  /**
   * Create a new RuvLLM instance
   */
  constructor(config?: RuvLLMConfig) {
    this.config = config ?? {};
    this.allowPlaceholder = this.config.allowPlaceholder === true || envAllowsPlaceholder();
    const extra = this.config as Record<string, unknown>;
    if (extra.backend !== undefined) {
      notice(
        'RUVLLM_UNSUPPORTED_OPTION',
        '`backend` is not supported by @ruvector/ruvllm and is ignored: models loaded with ' +
          'loadModel() run on the built-in candle backend.',
        this.config.strict,
      );
    }

    const mod = getNativeModule();
    if (mod) {
      try {
        this.native = new mod.RuvLLMEngine(toNativeConfig(config));
        this.nativeVersion = mod.version();
      } catch {
        // Silently fall back to JS implementation
      }
    }

    if (this.config.modelPath !== undefined) {
      this.loadModel(this.config.modelPath, this.config.modelOptions);
    }
  }

  /**
   * Whether the native binary can load and run a language model. False
   * without a native binary, for binaries that predate `loadModel` (the 2.0.x
   * `@ruvector/ruvllm-<platform>` packages), and for binaries built without
   * the `candle` inference backend.
   */
  supportsModelLoading(): boolean {
    const native = this.native;
    if (typeof native?.loadModel !== 'function') return false;
    // Builds that predate this probe all had the backend (they were local
    // `npm run build:native` builds, which enable `candle`).
    return typeof native.supportsModelLoading === 'function' ? native.supportsModelLoading() : true;
  }

  /** Error for a binary that has `loadModel` but no inference backend. */
  private noBackendError(): Error {
    return codedError(
      'RUVLLM_NO_INFERENCE_BACKEND',
      `the installed native binary (version ${this.nativeVersion ?? 'unknown'}) was built ` +
        'without the `candle` inference backend and cannot load models. Rebuild it with ' +
        '`npm run build:native` (which enables `candle`) and set RUVLLM_NATIVE_PATH to it, ' +
        'or use the ruvllm CLI (`ruvllm serve <model>`).',
    );
  }

  /**
   * Load a GGUF model (or a directory holding one) for `generate()`,
   * `chat()` and `query()`. Supported GGUF architectures: llama, mistral,
   * qwen2, qwen3; others fail with an error naming the architecture. The
   * tokenizer comes from a `tokenizer.json` beside the file, else from the
   * GGUF itself. Replaces any loaded model; if loading fails, no model is
   * loaded. Synchronous: blocks the event loop while the weights load.
   *
   * @throws `RUVLLM_NATIVE_UNAVAILABLE` without a native binary,
   *   `RUVLLM_NATIVE_TOO_OLD` when the binary predates model loading,
   *   `RUVLLM_NO_INFERENCE_BACKEND` when it was built without `candle`,
   *   `RUVLLM_MODEL_NOT_FOUND` / `RUVLLM_MODEL_LOAD_FAILED` from the backend.
   */
  loadModel(path: string, options?: LoadModelOptions): LoadedModelInfo {
    if (typeof path !== 'string' || path.length === 0) {
      throw new TypeError('loadModel: path must be a non-empty string');
    }
    if (!this.native) {
      throw codedError(
        'RUVLLM_NATIVE_UNAVAILABLE',
        `no native RuvLLM binary could be loaded for ${platformKey()}, so models cannot be ` +
          'loaded. Install the matching @ruvector/ruvllm-<platform> package, or build one ' +
          '(`npm run build:native`) and set RUVLLM_NATIVE_PATH to it.',
      );
    }
    if (typeof this.native.loadModel !== 'function') {
      throw codedError(
        'RUVLLM_NATIVE_TOO_OLD',
        `the installed native binary (version ${this.nativeVersion ?? 'unknown'}) cannot load ` +
          'models: it predates loadModel(). The @ruvector/ruvllm-<platform> 2.0.x packages are ' +
          'such builds. Install the 3.x platform package, or build the binary from source ' +
          '(`npm run build:native`, which enables the `candle` feature) and set ' +
          'RUVLLM_NATIVE_PATH to it, or use the ruvllm CLI (`ruvllm serve <model>`).',
      );
    }
    if (!this.supportsModelLoading()) throw this.noBackendError();
    const native = this.native;
    return toModelInfo(callNative(() => native.loadModel!(path, options)));
  }

  /**
   * Unload the language model and free its memory
   */
  unloadModel(): void {
    this.native?.unloadModel?.();
  }

  /**
   * Whether a language model is loaded
   */
  isModelLoaded(): boolean {
    return this.native?.isModelLoaded?.() ?? false;
  }

  /**
   * The loaded language model, or null
   */
  modelInfo(): LoadedModelInfo | null {
    const info = this.native?.modelInfo?.();
    return info ? toModelInfo(info) : null;
  }

  /**
   * Route `text`, then answer it with the loaded model (as one user message
   * under the model's chat template).
   *
   * @throws `RUVLLM_NO_LANGUAGE_MODEL` when no model is loaded, unless
   *   `allowPlaceholder` is set (then `text` is `PLACEHOLDER_TEXT`).
   */
  query(text: string, config?: GenerationConfig): QueryResponse {
    if (!this.isModelLoaded()) {
      const placeholder = this.placeholderOrThrow();
      this.fallbackState.queryCount++;
      const route = this.route(text);
      return {
        text: placeholder,
        confidence: route.confidence,
        model: route.model,
        contextSize: route.contextSize,
        latencyMs: 0,
        requestId: `placeholder-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      };
    }

    const native = this.native!;
    const result = callNative(() => native.query(text, toNativeGenConfig(config)));
    return {
      text: result.text,
      confidence: result.confidence,
      model: result.model,
      contextSize: result.contextSize,
      latencyMs: result.latencyMs,
      requestId: result.requestId,
      promptTokens: result.promptTokens,
      completionTokens: result.completionTokens,
      finishReason: result.finishReason as FinishReason | undefined,
    };
  }

  /**
   * Continue `prompt` verbatim with the loaded model (no chat template; use
   * `chat()` for instruction-tuned models). Returns the generated text.
   *
   * @throws `RUVLLM_NO_LANGUAGE_MODEL` when no model is loaded, unless
   *   `allowPlaceholder` is set (then it returns `PLACEHOLDER_TEXT`).
   */
  generate(prompt: string, config?: GenerationConfig): string {
    if (!this.isModelLoaded()) return this.placeholderOrThrow();
    return this.generateDetailed(prompt, config).text;
  }

  /**
   * Like `generate()`, with token counts (from the model's tokenizer) and
   * the finish reason. Always throws `RUVLLM_NO_LANGUAGE_MODEL` without a
   * loaded model.
   */
  generateDetailed(prompt: string, config?: GenerationConfig): GenerationResult {
    const native = this.requireModel();
    return toGenerationResult(
      callNative(() => native.generate(prompt, toNativeGenConfig(config))) as NativeGenerationResult,
    );
  }

  /**
   * Run the loaded model on a conversation, formatted with the model's chat
   * template (ChatML for Qwen, Llama 3 / Mistral formats for those models).
   * Always throws `RUVLLM_NO_LANGUAGE_MODEL` without a loaded model.
   */
  chat(messages: ChatMessage[], config?: GenerationConfig): GenerationResult {
    const native = this.requireModel();
    return toGenerationResult(callNative(() => native.chat!(messages, toNativeGenConfig(config))));
  }

  private requireModel(): NativeEngine {
    if (!this.isModelLoaded()) throw this.noModelError();
    return this.native!;
  }

  private noModelError(): Error {
    let fix: string;
    if (!this.native) {
      fix =
        `no native RuvLLM binary is available for ${platformKey()}, so no model can be loaded ` +
        '(see RUVLLM_NATIVE_UNAVAILABLE in the README)';
    } else if (typeof this.native.loadModel !== 'function') {
      fix =
        `the installed native binary (version ${this.nativeVersion ?? 'unknown'}) cannot load ` +
        'models (see RUVLLM_NATIVE_TOO_OLD in the README)';
    } else if (!this.supportsModelLoading()) {
      fix =
        `the installed native binary (version ${this.nativeVersion ?? 'unknown'}) was built ` +
        'without the candle inference backend, so it cannot load models (see ' +
        'RUVLLM_NO_INFERENCE_BACKEND in the README)';
    } else {
      fix = 'call loadModel(<path to a GGUF file>) first';
    }
    return codedError(
      'RUVLLM_NO_LANGUAGE_MODEL',
      `no language model is loaded, so there is no model output to return: ${fix}. Routing, ` +
        'memory and embeddings work without a model. For an HTTP server use the ruvllm CLI ' +
        '(`ruvllm serve <model>`).',
    );
  }

  /** Throw `RUVLLM_NO_LANGUAGE_MODEL`, or return the labelled placeholder when opted in. */
  private placeholderOrThrow(): string {
    if (!this.allowPlaceholder) throw this.noModelError();
    notice(
      'RUVLLM_PLACEHOLDER_TEXT',
      'no language model is loaded: generate()/query() are returning a labelled placeholder, ' +
        'not model output, because allowPlaceholder (or RUVLLM_ALLOW_MOCK=1) is set.',
      false,
    );
    return PLACEHOLDER_TEXT;
  }

  /**
   * Get routing decision for a query
   */
  route(text: string): RoutingDecision {
    if (this.native) {
      const result = this.native.route(text);
      return {
        model: result.model as any,
        contextSize: result.contextSize ?? result.context_size,
        temperature: result.temperature,
        topP: result.topP ?? result.top_p,
        confidence: result.confidence,
      };
    }

    // Fallback
    return {
      model: 'M700',
      contextSize: 512,
      temperature: 0.7,
      topP: 0.9,
      confidence: 0.5,
    };
  }

  /**
   * Search memory for similar content
   */
  searchMemory(text: string, k = 10): MemoryResult[] {
    if (this.native) {
      const results = this.native.searchMemory(text, k);
      return results.map(r => {
        // The native side reports `distance` (smaller is closer). MemoryResult
        // documents `score` as a similarity, so convert rather than passing a
        // distance through under a similarity's name. Older builds that already
        // emit `score` are used as-is.
        const anyR = r as any;
        const score =
          typeof anyR.score === 'number'
            ? anyR.score
            : typeof anyR.distance === 'number'
              ? 1 / (1 + Math.max(0, anyR.distance))
              : 0;
        return {
          id: r.id,
          score,
          content: r.content,
          metadata: JSON.parse(r.metadata || '{}'),
        };
      });
    }

    // Fallback - simple search
    return Array.from(this.fallbackState.memory.entries())
      .slice(0, k)
      .map(([id, data]) => ({
        id,
        score: 0.5,
        content: data.content,
        metadata: data.metadata,
      }));
  }

  /**
   * Add content to memory
   */
  addMemory(content: string, metadata?: Record<string, unknown>): MemoryId {
    if (this.native) {
      return this.native.addMemory(content, metadata ? JSON.stringify(metadata) : undefined);
    }

    // Fallback
    const id = this.fallbackState.nextId++;
    this.fallbackState.memory.set(id, {
      content,
      embedding: this.embed(content),
      metadata: metadata ?? {},
    });
    return id;
  }

  /**
   * Provide feedback for learning
   */
  feedback(fb: Feedback): boolean {
    if (this.native) {
      return this.native.feedback(fb.requestId, fb.rating, fb.correction);
    }
    return false;
  }

  /**
   * Get engine statistics
   */
  stats(): RuvLLMStats {
    if (this.native) {
      const s = this.native.stats();
      // Map native stats (snake_case) to TypeScript interface (camelCase)
      // Handle both old and new field names for backward compatibility
      return {
        totalQueries: (s as any).totalQueries ?? s.total_queries ?? 0,
        memoryNodes: (s as any).memoryNodes ?? s.memory_nodes ?? 0,
        patternsLearned:
          (s as any).trainingSteps ??
          s.patterns_learned ??
          (s as any).training_steps ??
          0,
        avgLatencyMs: (s as any).avgLatencyMs ?? s.avg_latency_ms ?? 0,
        cacheHitRate: (s as any).cacheHitRate ?? s.cache_hit_rate ?? 0,
        routerAccuracy: (s as any).routerAccuracy ?? s.router_accuracy ?? 0.5,
      };
    }

    // Fallback
    return {
      totalQueries: this.fallbackState.queryCount,
      memoryNodes: this.fallbackState.memory.size,
      patternsLearned: 0,
      avgLatencyMs: 1.0,
      cacheHitRate: 0.0,
      routerAccuracy: 0.5,
    };
  }

  /**
   * Force router learning cycle
   */
  forceLearn(): string {
    if (this.native) {
      return this.native.forceLearn();
    }
    return 'Learning not available in fallback mode';
  }

  /**
   * Get embedding for text
   */
  embed(text: string): Embedding {
    if (this.native) {
      return this.native.embed(text);
    }

    // Fallback - simple hash-based embedding
    const dim = this.config.embeddingDim ?? 768;
    const embedding = new Array(dim).fill(0);

    for (let i = 0; i < text.length; i++) {
      const idx = (text.charCodeAt(i) * (i + 1)) % dim;
      embedding[idx] += 0.1;
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0)) || 1;
    return embedding.map(x => x / norm);
  }

  /**
   * Compute similarity between two texts
   */
  similarity(text1: string, text2: string): number {
    if (this.native) {
      // f32 accumulation can land just outside the mathematical range — an
      // identical pair measures 1.0000001 — which turns Math.acos(sim) into
      // NaN downstream. Cosine similarity is defined on [-1, 1], so clamp.
      const raw = this.native.similarity(text1, text2);
      return Math.min(1, Math.max(-1, raw));
    }

    // Fallback - cosine similarity
    const emb1 = this.embed(text1);
    const emb2 = this.embed(text2);

    let dot = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < emb1.length; i++) {
      dot += emb1[i] * emb2[i];
      norm1 += emb1[i] * emb1[i];
      norm2 += emb2[i] * emb2[i];
    }

    const denom = Math.sqrt(norm1) * Math.sqrt(norm2);
    const similarity = denom > 0 ? dot / denom : 0;
    // Clamp to [0, 1] to handle floating point errors
    return Math.max(0, Math.min(1, similarity));
  }

  /**
   * Check if SIMD is available
   */
  hasSimd(): boolean {
    if (this.native) {
      return this.native.hasSimd();
    }
    return false;
  }

  /**
   * Get SIMD capabilities
   */
  simdCapabilities(): string[] {
    if (this.native) {
      return this.native.simdCapabilities();
    }
    return ['Scalar (fallback)'];
  }

  /**
   * Batch query multiple prompts
   */
  batchQuery(request: BatchQueryRequest): BatchQueryResponse {
    const start = Date.now();
    const responses = request.queries.map(q => this.query(q, request.config));
    return {
      responses,
      totalLatencyMs: Date.now() - start,
    };
  }

  /**
   * Check if native module is loaded
   */
  isNativeLoaded(): boolean {
    return this.native !== null;
  }
}
