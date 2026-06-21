/**
 * Router configuration for Tiny Dancer neural routing
 */
export interface RouterConfig {
  /** Path to the FastGRNN model file (safetensors format) */
  modelPath: string;
  /** Confidence threshold for routing decisions (0.0 to 1.0, default: 0.85) */
  confidenceThreshold?: number;
  /** Maximum uncertainty before falling back (0.0 to 1.0, default: 0.15) */
  maxUncertainty?: number;
  /** Enable circuit breaker for fault tolerance (default: true) */
  enableCircuitBreaker?: boolean;
  /** Number of failures before circuit opens (default: 5) */
  circuitBreakerThreshold?: number;
  /** Enable quantization for memory efficiency (default: true) */
  enableQuantization?: boolean;
  /** Optional database path for persistence */
  databasePath?: string;
}

/**
 * Candidate for routing evaluation
 */
export interface Candidate {
  /** Unique identifier for the candidate */
  id: string;
  /** Embedding vector (Float32Array or number[]) */
  embedding: Float32Array | number[];
  /** Optional metadata as JSON string */
  metadata?: string;
  /** Creation timestamp (Unix epoch milliseconds) */
  createdAt?: number;
  /** Number of times this candidate was accessed */
  accessCount?: number;
  /** Historical success rate (0.0 to 1.0) */
  successRate?: number;
}

/**
 * Routing request containing query and candidates
 */
export interface RoutingRequest {
  /** Query embedding to route */
  queryEmbedding: Float32Array | number[];
  /** Candidates to evaluate for routing */
  candidates: Candidate[];
  /** Optional request metadata as JSON string */
  metadata?: string;
}

/**
 * Individual routing decision for a candidate
 */
export interface RoutingDecision {
  /** ID of the candidate */
  candidateId: string;
  /** Confidence score (0.0 to 1.0) */
  confidence: number;
  /** Whether to use lightweight/fast model */
  useLightweight: boolean;
  /** Uncertainty estimate (0.0 to 1.0) */
  uncertainty: number;
}

/**
 * Response from a routing operation
 */
export interface RoutingResponse {
  /** Ranked routing decisions */
  decisions: RoutingDecision[];
  /** Total inference time in microseconds */
  inferenceTimeUs: number;
  /** Number of candidates processed */
  candidatesProcessed: number;
  /** Feature engineering time in microseconds */
  featureTimeUs: number;
}

/**
 * Tiny Dancer neural router for intelligent AI agent routing
 *
 * @example
 * ```typescript
 * import { Router } from '@ruvector/tiny-dancer';
 *
 * const router = new Router({
 *   modelPath: './models/fastgrnn.safetensors',
 *   confidenceThreshold: 0.85,
 *   enableCircuitBreaker: true
 * });
 *
 * const response = await router.route({
 *   queryEmbedding: new Float32Array([0.1, 0.2, ...]),
 *   candidates: [
 *     { id: 'gpt4', embedding: new Float32Array([...]) },
 *     { id: 'claude', embedding: new Float32Array([...]) }
 *   ]
 * });
 *
 * console.log('Best route:', response.decisions[0].candidateId);
 * ```
 */
export class Router {
  /**
   * Create a new neural router
   * @param config Router configuration
   */
  constructor(config: RouterConfig);

  /**
   * Route a request through the neural routing system
   * @param request Routing request with query and candidates
   * @returns Promise resolving to routing decisions
   */
  route(request: RoutingRequest): Promise<RoutingResponse>;

  /**
   * Hot-reload the model from disk
   * @returns Promise resolving when reload is complete
   */
  reloadModel(): Promise<void>;

  /**
   * Check circuit breaker status
   * @returns true if circuit is closed (healthy), false if open
   */
  circuitBreakerStatus(): boolean | null;
}

/**
 * Get the version of the Tiny Dancer library
 */
export function version(): string;

/**
 * Test function to verify bindings are working
 */
export function hello(): string;

/**
 * One DRACO training row: a query embedding and the quality each model achieved
 * on it. Matches the `{ embedding, scores }` shape `@metaharness/router` consumes.
 */
export interface DracoRowJs {
  /** Query embedding (used directly as the model's input features). */
  embedding: number[];
  /** model id → quality achieved on this query (0..1). */
  scores: Record<string, number>;
}

/** Options for {@link trainRouter}. */
export interface TrainRouterOptions {
  /** Where to write the trained `.safetensors` model. */
  outputPath: string;
  /** Input feature dimension (must equal the embedding length). */
  inputDim: number;
  /** Hidden dimension (default 12). */
  hiddenDim?: number;
  /** Training epochs (default 40). */
  epochs?: number;
  /** Learning rate (default 0.05). */
  learningRate?: number;
  /** DRACO label tolerance: cheap model is "good enough" within this of the best (default 0.05). */
  tolerance?: number;
}

/** Result of {@link trainRouter}. */
export interface TrainRouterResult {
  epochsRun: number;
  trainLoss: number;
  trainAccuracy: number;
  valAccuracy: number;
  modelPath: string;
  modelBytes: number;
}

/**
 * Train a FastGRNN router from a DRACO dataset and write it to a `.safetensors`
 * file consumable by `new Router({ modelPath })`.
 *
 * @example
 * ```javascript
 * const res = await trainRouter(rows, { haiku: 1, opus: 15 }, {
 *   outputPath: './router.safetensors', inputDim: 8, epochs: 40,
 * });
 * const router = new Router({ modelPath: res.modelPath });
 * ```
 */
export function trainRouter(
  rows: DracoRowJs[],
  prices: Record<string, number>,
  options: TrainRouterOptions
): Promise<TrainRouterResult>;

/**
 * Score a query embedding with a trained FastGRNN model (raw forward pass).
 *
 * Loads the `.safetensors` produced by {@link trainRouter} and runs the model
 * directly on `embedding` (which must match the model's `inputDim`). Returns the
 * sigmoid output in 0..1 — high means "the cheap model is good enough" (route to
 * the cheaper model); low means route to a stronger model. This is the inference
 * path that matches `trainRouter`; it does not run `Router`'s feature engineering.
 *
 * @example
 * ```javascript
 * const s = await score('./router.safetensors', queryEmbedding);
 * const useCheap = s >= 0.5;
 * ```
 */
export function score(modelPath: string, embedding: number[]): Promise<number>;
