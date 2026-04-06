/**
 * Router Wrapper - Semantic router for AI agent intent matching
 *
 * Wraps @ruvector/router for vector-based intent classification.
 * Perfect for hooks to route tasks to the right agent.
 */
export declare function isRouterAvailable(): boolean;
export interface Route {
    name: string;
    utterances: string[];
    metadata?: Record<string, any>;
}
export interface RouteMatch {
    route: string;
    score: number;
    metadata?: Record<string, any>;
}
/**
 * Semantic Router for agent task routing
 */
export declare class SemanticRouter {
    private inner;
    private routes;
    constructor(options?: {
        dimension?: number;
        threshold?: number;
    });
    /**
     * Set the embedder function for converting text to vectors.
     * Required before match() can accept string input.
     */
    setEmbedder(embedder: (text: string) => Promise<Float32Array>): void;
    /**
     * Add a route with example utterances (sync, requires pre-computed embedding)
     */
    addRoute(name: string, utterances: string[], metadata?: Record<string, any>, embedding?: Float32Array | number[]): void;
    /**
     * Add a route with automatic embedding computation (requires setEmbedder)
     */
    addRouteAsync(name: string, utterances: string[], metadata?: Record<string, any>): Promise<void>;
    /**
     * Add multiple routes at once
     */
    addRoutes(routes: Route[]): void;
    /**
     * Match input to best route (async, accepts string if embedder is set, or Float32Array)
     */
    match(input: string | Float32Array): Promise<RouteMatch | null>;
    /**
     * Get top-k route matches (async)
     */
    matchTopK(input: string | Float32Array, k?: number): Promise<RouteMatch[]>;
    /**
     * Match with a pre-computed embedding (synchronous)
     */
    matchWithEmbedding(embedding: Float32Array, k?: number): RouteMatch[];
    /**
     * Get all registered routes
     */
    getRoutes(): Route[];
    /**
     * Remove a route
     */
    removeRoute(name: string): boolean;
    /**
     * Clear all routes
     */
    clear(): void;
}
/**
 * Create a pre-configured agent router for hooks
 */
export declare function createAgentRouter(): SemanticRouter;
export default SemanticRouter;
//# sourceMappingURL=router-wrapper.d.ts.map