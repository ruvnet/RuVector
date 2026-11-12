/* tslint:disable */
/* eslint-disable */

/**
 * WASM wrapper for DeterministicLocalKCut
 *
 * Implements the deterministic local k-cut algorithm from arXiv:2512.13105:
 * - Uses 4-color coding (red-blue, green-yellow)
 * - Greedy forest packing for edge classification
 * - Color-coded DFS for cut enumeration
 */
export class WasmLocalKCut {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Delete an edge
     */
    deleteEdge(u: bigint, v: bigint): void;
    /**
     * Insert an edge
     */
    insertEdge(u: bigint, v: bigint, weight: number): void;
    /**
     * Create a new LocalKCut structure
     *
     * # Arguments
     * * `lambda_max` - Maximum cut value to consider
     * * `volume_bound` - Maximum volume to explore (nu parameter)
     * * `beta` - Cut depth parameter (typically 2)
     */
    constructor(lambda_max: bigint, volume_bound: number, beta: number);
    /**
     * Get number of edges
     */
    numEdges(): number;
    /**
     * Get number of vertices (approximate)
     */
    numVertices(): number;
    /**
     * Query local cuts from a source vertex
     *
     * Returns array of { cut_value, vertices } objects
     */
    query(source: bigint): any;
}

/**
 * WASM wrapper for DynamicMinCut
 */
export class WasmMinCut {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Batch delete multiple edges
     *
     * # Arguments
     * * `edges` - JavaScript array of [u, v] tuples
     *
     * # Returns
     * The final minimum cut value
     *
     * # Example
     * ```javascript
     * const edges = [[0, 1], [1, 2]];
     * const cutValue = mincut.batchDelete(edges);
     * ```
     */
    batchDelete(edges: any): number;
    /**
     * Batch insert multiple edges
     *
     * # Arguments
     * * `edges` - JavaScript array of [u, v, weight] tuples
     *
     * # Returns
     * The final minimum cut value
     *
     * # Example
     * ```javascript
     * const edges = [[0, 1, 1.0], [1, 2, 2.0], [2, 3, 1.5]];
     * const cutValue = mincut.batchInsert(edges);
     * ```
     */
    batchInsert(edges: any): number;
    /**
     * Bulk insertion from Uint32Array endpoint pairs and Float64Array weights.
     * wasm-bindgen copies these slices into WASM memory once per call.
     */
    batchInsertTyped(endpoints: Uint32Array, weights: Float64Array): number;
    /**
     * Clear all edges from the graph
     */
    clear(): void;
    /**
     * Get the cut edges as JSON array
     *
     * # Returns
     * JavaScript array of edge objects: [{ u, v, weight }, ...]
     *
     * # Example
     * ```javascript
     * const edges = mincut.cutEdges();
     * edges.forEach(e => console.log(`Edge ${e.u}-${e.v}: ${e.weight}`));
     * ```
     */
    cutEdges(): any;
    /**
     * Delete an edge from the graph
     *
     * # Arguments
     * * `u` - Source vertex
     * * `v` - Target vertex
     *
     * # Returns
     * The new minimum cut value after deletion
     */
    deleteEdge(u: bigint, v: bigint): number;
    /**
     * Create from edges array: [[u, v, weight], ...]
     *
     * # Arguments
     * * `edges` - JavaScript array of [u, v, weight] tuples
     *
     * # Example
     * ```javascript
     * const edges = [[0, 1, 1.5], [1, 2, 2.0]];
     * const mincut = WasmMinCut.fromEdges(edges);
     * ```
     */
    static fromEdges(edges: any): WasmMinCut;
    /**
     * Insert an edge into the graph
     *
     * # Arguments
     * * `u` - Source vertex
     * * `v` - Target vertex
     * * `weight` - Edge weight
     *
     * # Returns
     * The new minimum cut value after insertion
     */
    insertEdge(u: bigint, v: bigint, weight: number): number;
    /**
     * Check if the graph is connected
     *
     * # Returns
     * `true` if there is a path between all vertex pairs
     */
    isConnected(): boolean;
    /**
     * Get the current minimum cut value
     *
     * # Returns
     * The sum of edge weights in the minimum cut
     */
    minCutValue(): number;
    /**
     * Create a new empty minimum cut structure
     */
    constructor();
    /**
     * Get the number of edges in the graph
     */
    numEdges(): number;
    /**
     * Get the number of vertices in the graph
     */
    numVertices(): number;
    /**
     * Get the partition as JSON: { "s": [...], "t": [...] }
     *
     * # Returns
     * JavaScript object with two arrays: `s` and `t` containing vertex IDs
     *
     * # Example
     * ```javascript
     * const { s, t } = mincut.partition();
     * console.log("S partition:", s);
     * console.log("T partition:", t);
     * ```
     */
    partition(): any;
    /**
     * Get comprehensive statistics as JSON
     *
     * # Returns
     * JavaScript object with:
     * - `num_vertices`: Number of vertices
     * - `num_edges`: Number of edges
     * - `min_cut_value`: Current minimum cut value
     * - `is_connected`: Whether graph is connected
     * - `num_operations`: Total operations performed
     *
     * # Example
     * ```javascript
     * const stats = mincut.stats();
     * console.log(`Graph has ${stats.num_vertices} vertices and ${stats.num_edges} edges`);
     * console.log(`Minimum cut value: ${stats.min_cut_value}`);
     * ```
     */
    stats(): any;
    /**
     * Set an edge weight, inserting a missing edge; invalid weights make no changes.
     *
     * # Arguments
     * * `u` - Source vertex
     * * `v` - Target vertex
     * * `new_weight` - New edge weight
     *
     * # Returns
     * The new minimum cut value after update
     */
    updateEdge(u: bigint, v: bigint, new_weight: number): number;
}

/**
 * WASM wrapper for MinCutWrapper
 *
 * High-level API combining all paper algorithms:
 * - O(log n) instance management
 * - ThreeLevelHierarchy decomposition
 * - LocalKCut discovery
 * - Connectivity curve analysis for boundary validation
 */
export class WasmMinCutWrapper {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Compute edge-connectivity degradation curve
     *
     * # Arguments
     * * `ranked_edges` - Array of [u, v, score] ranked by cut-likelihood
     * * `k_max` - Maximum edges to remove
     *
     * # Returns
     * Array of { k, min_cut } showing degradation
     */
    connectivityCurve(ranked_edges: any, k_max: number): any;
    /**
     * Get current logical time
     */
    currentTime(): bigint;
    /**
     * Delete an edge (timestamp auto-incremented)
     */
    deleteEdge(u: bigint, v: bigint): void;
    /**
     * Compute detector quality score
     *
     * # Arguments
     * * `ranked_edges` - Array of [u, v, score]
     * * `true_cut_size` - Known size of true minimum cut
     *
     * # Returns
     * Quality score from 0.0 (poor) to 1.0 (perfect)
     */
    detectorQuality(ranked_edges: any, true_cut_size: number): number;
    /**
     * Find elbow point in connectivity curve
     *
     * Returns { k, drop } or null if no elbow found
     */
    static findElbow(curve: any): any;
    /**
     * Insert an edge (timestamp auto-incremented)
     */
    insertEdge(u: bigint, v: bigint): void;
    /**
     * Get local cuts from a source vertex
     */
    localCuts(source: bigint, lambda_max: bigint): any;
    /**
     * Create a new MinCutWrapper
     */
    constructor();
    /**
     * Get number of active instances
     */
    numInstances(): number;
    /**
     * Query the minimum cut value
     */
    query(): number;
    /**
     * Query with LocalKCut certification
     *
     * Returns { cut_value, certified } object
     */
    queryWithCertification(source: bigint): any;
}

/**
 * Directed integer-cost routing. Arc IDs are input positions. Run in a Worker.
 */
export class WasmRoadRouter {
    free(): void;
    [Symbol.dispose](): void;
    nearest(lat: number, lon: number, radius_m: number): any;
    constructor(nodes: number, endpoints: Uint32Array, costs: Uint32Array, turns: Uint32Array);
    prepare(landmarks: Uint32Array, budget: number): void;
    route(source: number, target: number, use_landmarks: boolean, budget: number): any;
    setCoordinates(lat_lon: Float64Array): void;
    /**
     * u32::MAX closes an arc. Other costs must be <= 1e9.
     */
    update(ids: Uint32Array, costs: Uint32Array): void;
}

/**
 * RuField-aware route planner. Verify live event receipts before ingest.
 */
export class WasmRuFieldRouter {
    free(): void;
    [Symbol.dispose](): void;
    activeNodes(): number;
    bindCell(x: number, y: number, z: number, node: number): void;
    bindZone(zone: string, node: number): void;
    expire(now_ns: number): number;
    ingestRuField(json: string, verified: boolean, now_ns: number): any;
    constructor(nodes: number, endpoints: Uint32Array, costs: Uint32Array, turns: Uint32Array, max_penalty: number, close_at_millionths: number, ttl_ns: number, max_lateness_ns: number);
    route(source: number, target: number, use_landmarks: boolean, budget: number): any;
}

/**
 * WASM wrapper for ThreeLevelHierarchy
 *
 * Implements the 3-level decomposition from arXiv:2512.13105:
 * - Level 0: Expanders (φ-expander subgraphs)
 * - Level 1: Preclusters (groups of expanders)
 * - Level 2: Clusters (top-level grouping with mirror cuts)
 */
export class WasmThreeLevelHierarchy {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Build the complete 3-level hierarchy
     *
     * Must be called after inserting edges to compute the decomposition.
     */
    build(): void;
    /**
     * Delete an edge from the graph
     */
    deleteEdge(u: bigint, v: bigint): void;
    /**
     * Get the global minimum cut estimate
     */
    globalMinCut(): number;
    /**
     * Insert an edge into the graph
     */
    insertEdge(u: bigint, v: bigint, weight: number): void;
    /**
     * Create a new hierarchy with default configuration
     */
    constructor();
    /**
     * Get hierarchy statistics as JSON
     */
    stats(): any;
    /**
     * Get all vertices as JSON array
     */
    vertices(): any;
    /**
     * Create hierarchy with custom expansion parameter φ
     */
    static withPhi(phi: number): WasmThreeLevelHierarchy;
}

/**
 * Get version information
 */
export function getVersion(): string;

/**
 * Initialize the WASM module (call once at startup)
 *
 * This sets up panic hooks for better error messages in the browser console.
 */
export function init(): void;
