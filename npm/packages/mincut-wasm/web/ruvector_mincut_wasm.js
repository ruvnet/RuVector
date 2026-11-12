/* @ts-self-types="./ruvector_mincut_wasm.d.ts" */

/**
 * WASM wrapper for DeterministicLocalKCut
 *
 * Implements the deterministic local k-cut algorithm from arXiv:2512.13105:
 * - Uses 4-color coding (red-blue, green-yellow)
 * - Greedy forest packing for edge classification
 * - Color-coded DFS for cut enumeration
 */
export class WasmLocalKCut {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmLocalKCutFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmlocalkcut_free(ptr, 0);
    }
    /**
     * Delete an edge
     * @param {bigint} u
     * @param {bigint} v
     */
    deleteEdge(u, v) {
        wasm.wasmlocalkcut_deleteEdge(this.__wbg_ptr, u, v);
    }
    /**
     * Insert an edge
     * @param {bigint} u
     * @param {bigint} v
     * @param {number} weight
     */
    insertEdge(u, v, weight) {
        wasm.wasmlocalkcut_insertEdge(this.__wbg_ptr, u, v, weight);
    }
    /**
     * Create a new LocalKCut structure
     *
     * # Arguments
     * * `lambda_max` - Maximum cut value to consider
     * * `volume_bound` - Maximum volume to explore (nu parameter)
     * * `beta` - Cut depth parameter (typically 2)
     * @param {bigint} lambda_max
     * @param {number} volume_bound
     * @param {number} beta
     */
    constructor(lambda_max, volume_bound, beta) {
        const ret = wasm.wasmlocalkcut_new(lambda_max, volume_bound, beta);
        this.__wbg_ptr = ret;
        WasmLocalKCutFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get number of edges
     * @returns {number}
     */
    numEdges() {
        const ret = wasm.wasmlocalkcut_numEdges(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get number of vertices (approximate)
     * @returns {number}
     */
    numVertices() {
        const ret = wasm.wasmlocalkcut_numVertices(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Query local cuts from a source vertex
     *
     * Returns array of { cut_value, vertices } objects
     * @param {bigint} source
     * @returns {any}
     */
    query(source) {
        const ret = wasm.wasmlocalkcut_query(this.__wbg_ptr, source);
        return takeObject(ret);
    }
}
if (Symbol.dispose) WasmLocalKCut.prototype[Symbol.dispose] = WasmLocalKCut.prototype.free;

/**
 * WASM wrapper for DynamicMinCut
 */
export class WasmMinCut {
    static __wrap(ptr) {
        const obj = Object.create(WasmMinCut.prototype);
        obj.__wbg_ptr = ptr;
        WasmMinCutFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMinCutFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmincut_free(ptr, 0);
    }
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
     * @param {any} edges
     * @returns {number}
     */
    batchDelete(edges) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmmincut_batchDelete(retptr, this.__wbg_ptr, addHeapObject(edges));
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {any} edges
     * @returns {number}
     */
    batchInsert(edges) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmmincut_batchInsert(retptr, this.__wbg_ptr, addHeapObject(edges));
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Bulk insertion from Uint32Array endpoint pairs and Float64Array weights.
     * wasm-bindgen copies these slices into WASM memory once per call.
     * @param {Uint32Array} endpoints
     * @param {Float64Array} weights
     * @returns {number}
     */
    batchInsertTyped(endpoints, weights) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(endpoints, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArrayF64ToWasm0(weights, wasm.__wbindgen_export);
            const len1 = WASM_VECTOR_LEN;
            wasm.wasmmincut_batchInsertTyped(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Clear all edges from the graph
     */
    clear() {
        wasm.wasmmincut_clear(this.__wbg_ptr);
    }
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
     * @returns {any}
     */
    cutEdges() {
        const ret = wasm.wasmmincut_cutEdges(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Delete an edge from the graph
     *
     * # Arguments
     * * `u` - Source vertex
     * * `v` - Target vertex
     *
     * # Returns
     * The new minimum cut value after deletion
     * @param {bigint} u
     * @param {bigint} v
     * @returns {number}
     */
    deleteEdge(u, v) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmmincut_deleteEdge(retptr, this.__wbg_ptr, u, v);
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {any} edges
     * @returns {WasmMinCut}
     */
    static fromEdges(edges) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmmincut_fromEdges(retptr, addHeapObject(edges));
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return WasmMinCut.__wrap(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
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
     * @param {bigint} u
     * @param {bigint} v
     * @param {number} weight
     * @returns {number}
     */
    insertEdge(u, v, weight) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmmincut_insertEdge(retptr, this.__wbg_ptr, u, v, weight);
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Check if the graph is connected
     *
     * # Returns
     * `true` if there is a path between all vertex pairs
     * @returns {boolean}
     */
    isConnected() {
        const ret = wasm.wasmmincut_isConnected(this.__wbg_ptr);
        return ret !== 0;
    }
    /**
     * Get the current minimum cut value
     *
     * # Returns
     * The sum of edge weights in the minimum cut
     * @returns {number}
     */
    minCutValue() {
        const ret = wasm.wasmmincut_minCutValue(this.__wbg_ptr);
        return ret;
    }
    /**
     * Create a new empty minimum cut structure
     */
    constructor() {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmmincut_new(retptr);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0;
            WasmMinCutFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * Get the number of edges in the graph
     * @returns {number}
     */
    numEdges() {
        const ret = wasm.wasmmincut_numEdges(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Get the number of vertices in the graph
     * @returns {number}
     */
    numVertices() {
        const ret = wasm.wasmmincut_numVertices(this.__wbg_ptr);
        return ret >>> 0;
    }
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
     * @returns {any}
     */
    partition() {
        const ret = wasm.wasmmincut_partition(this.__wbg_ptr);
        return takeObject(ret);
    }
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
     * @returns {any}
     */
    stats() {
        const ret = wasm.wasmmincut_stats(this.__wbg_ptr);
        return takeObject(ret);
    }
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
     * @param {bigint} u
     * @param {bigint} v
     * @param {number} new_weight
     * @returns {number}
     */
    updateEdge(u, v, new_weight) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmmincut_updateEdge(retptr, this.__wbg_ptr, u, v, new_weight);
            var r0 = getDataViewMemory0().getFloat64(retptr + 8 * 0, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            var r3 = getDataViewMemory0().getInt32(retptr + 4 * 3, true);
            if (r3) {
                throw takeObject(r2);
            }
            return r0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) WasmMinCut.prototype[Symbol.dispose] = WasmMinCut.prototype.free;

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
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmMinCutWrapperFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmmincutwrapper_free(ptr, 0);
    }
    /**
     * Compute edge-connectivity degradation curve
     *
     * # Arguments
     * * `ranked_edges` - Array of [u, v, score] ranked by cut-likelihood
     * * `k_max` - Maximum edges to remove
     *
     * # Returns
     * Array of { k, min_cut } showing degradation
     * @param {any} ranked_edges
     * @param {number} k_max
     * @returns {any}
     */
    connectivityCurve(ranked_edges, k_max) {
        const ret = wasm.wasmmincutwrapper_connectivityCurve(this.__wbg_ptr, addHeapObject(ranked_edges), k_max);
        return takeObject(ret);
    }
    /**
     * Get current logical time
     * @returns {bigint}
     */
    currentTime() {
        const ret = wasm.wasmmincutwrapper_currentTime(this.__wbg_ptr);
        return BigInt.asUintN(64, ret);
    }
    /**
     * Delete an edge (timestamp auto-incremented)
     * @param {bigint} u
     * @param {bigint} v
     */
    deleteEdge(u, v) {
        wasm.wasmmincutwrapper_deleteEdge(this.__wbg_ptr, u, v);
    }
    /**
     * Compute detector quality score
     *
     * # Arguments
     * * `ranked_edges` - Array of [u, v, score]
     * * `true_cut_size` - Known size of true minimum cut
     *
     * # Returns
     * Quality score from 0.0 (poor) to 1.0 (perfect)
     * @param {any} ranked_edges
     * @param {number} true_cut_size
     * @returns {number}
     */
    detectorQuality(ranked_edges, true_cut_size) {
        const ret = wasm.wasmmincutwrapper_detectorQuality(this.__wbg_ptr, addHeapObject(ranked_edges), true_cut_size);
        return ret;
    }
    /**
     * Find elbow point in connectivity curve
     *
     * Returns { k, drop } or null if no elbow found
     * @param {any} curve
     * @returns {any}
     */
    static findElbow(curve) {
        const ret = wasm.wasmmincutwrapper_findElbow(addHeapObject(curve));
        return takeObject(ret);
    }
    /**
     * Insert an edge (timestamp auto-incremented)
     * @param {bigint} u
     * @param {bigint} v
     */
    insertEdge(u, v) {
        wasm.wasmmincutwrapper_insertEdge(this.__wbg_ptr, u, v);
    }
    /**
     * Get local cuts from a source vertex
     * @param {bigint} source
     * @param {bigint} lambda_max
     * @returns {any}
     */
    localCuts(source, lambda_max) {
        const ret = wasm.wasmmincutwrapper_localCuts(this.__wbg_ptr, source, lambda_max);
        return takeObject(ret);
    }
    /**
     * Create a new MinCutWrapper
     */
    constructor() {
        const ret = wasm.wasmmincutwrapper_new();
        this.__wbg_ptr = ret;
        WasmMinCutWrapperFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get number of active instances
     * @returns {number}
     */
    numInstances() {
        const ret = wasm.wasmmincutwrapper_numInstances(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Query the minimum cut value
     * @returns {number}
     */
    query() {
        const ret = wasm.wasmmincutwrapper_query(this.__wbg_ptr);
        return ret;
    }
    /**
     * Query with LocalKCut certification
     *
     * Returns { cut_value, certified } object
     * @param {bigint} source
     * @returns {any}
     */
    queryWithCertification(source) {
        const ret = wasm.wasmmincutwrapper_queryWithCertification(this.__wbg_ptr, source);
        return takeObject(ret);
    }
}
if (Symbol.dispose) WasmMinCutWrapper.prototype[Symbol.dispose] = WasmMinCutWrapper.prototype.free;

/**
 * Directed integer-cost routing. Arc IDs are input positions. Run in a Worker.
 */
export class WasmRoadRouter {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmRoadRouterFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmroadrouter_free(ptr, 0);
    }
    /**
     * @param {number} lat
     * @param {number} lon
     * @param {number} radius_m
     * @returns {any}
     */
    nearest(lat, lon, radius_m) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmroadrouter_nearest(retptr, this.__wbg_ptr, lat, lon, radius_m);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} nodes
     * @param {Uint32Array} endpoints
     * @param {Uint32Array} costs
     * @param {Uint32Array} turns
     */
    constructor(nodes, endpoints, costs, turns) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(endpoints, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArray32ToWasm0(costs, wasm.__wbindgen_export);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passArray32ToWasm0(turns, wasm.__wbindgen_export);
            const len2 = WASM_VECTOR_LEN;
            wasm.wasmroadrouter_new(retptr, nodes, ptr0, len0, ptr1, len1, ptr2, len2);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0;
            WasmRoadRouterFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {Uint32Array} landmarks
     * @param {number} budget
     */
    prepare(landmarks, budget) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(landmarks, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasmroadrouter_prepare(retptr, this.__wbg_ptr, ptr0, len0, budget);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} source
     * @param {number} target
     * @param {boolean} use_landmarks
     * @param {number} budget
     * @returns {any}
     */
    route(source, target, use_landmarks, budget) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmroadrouter_route(retptr, this.__wbg_ptr, source, target, use_landmarks, budget);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {Float64Array} lat_lon
     */
    setCoordinates(lat_lon) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArrayF64ToWasm0(lat_lon, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasmroadrouter_setCoordinates(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * u32::MAX closes an arc. Other costs must be <= 1e9.
     * @param {Uint32Array} ids
     * @param {Uint32Array} costs
     */
    update(ids, costs) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(ids, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArray32ToWasm0(costs, wasm.__wbindgen_export);
            const len1 = WASM_VECTOR_LEN;
            wasm.wasmroadrouter_update(retptr, this.__wbg_ptr, ptr0, len0, ptr1, len1);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) WasmRoadRouter.prototype[Symbol.dispose] = WasmRoadRouter.prototype.free;

/**
 * RuField-aware route planner. Verify live event receipts before ingest.
 */
export class WasmRuFieldRouter {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmRuFieldRouterFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmrufieldrouter_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    activeNodes() {
        const ret = wasm.wasmrufieldrouter_activeNodes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {number} node
     */
    bindCell(x, y, z, node) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmrufieldrouter_bindCell(retptr, this.__wbg_ptr, x, y, z, node);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {string} zone
     * @param {number} node
     */
    bindZone(zone, node) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(zone, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasmrufieldrouter_bindZone(retptr, this.__wbg_ptr, ptr0, len0, node);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            if (r1) {
                throw takeObject(r0);
            }
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} now_ns
     * @returns {number}
     */
    expire(now_ns) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmrufieldrouter_expire(retptr, this.__wbg_ptr, now_ns);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return r0 >>> 0;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {string} json
     * @param {boolean} verified
     * @param {number} now_ns
     * @returns {any}
     */
    ingestRuField(json, verified, now_ns) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passStringToWasm0(json, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.wasmrufieldrouter_ingestRuField(retptr, this.__wbg_ptr, ptr0, len0, verified, now_ns);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} nodes
     * @param {Uint32Array} endpoints
     * @param {Uint32Array} costs
     * @param {Uint32Array} turns
     * @param {number} max_penalty
     * @param {number} close_at_millionths
     * @param {number} ttl_ns
     * @param {number} max_lateness_ns
     */
    constructor(nodes, endpoints, costs, turns, max_penalty, close_at_millionths, ttl_ns, max_lateness_ns) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(endpoints, wasm.__wbindgen_export);
            const len0 = WASM_VECTOR_LEN;
            const ptr1 = passArray32ToWasm0(costs, wasm.__wbindgen_export);
            const len1 = WASM_VECTOR_LEN;
            const ptr2 = passArray32ToWasm0(turns, wasm.__wbindgen_export);
            const len2 = WASM_VECTOR_LEN;
            wasm.wasmrufieldrouter_new(retptr, nodes, ptr0, len0, ptr1, len1, ptr2, len2, max_penalty, close_at_millionths, ttl_ns, max_lateness_ns);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            this.__wbg_ptr = r0;
            WasmRuFieldRouterFinalization.register(this, this.__wbg_ptr, this);
            return this;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
    /**
     * @param {number} source
     * @param {number} target
     * @param {boolean} use_landmarks
     * @param {number} budget
     * @returns {any}
     */
    route(source, target, use_landmarks, budget) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            wasm.wasmrufieldrouter_route(retptr, this.__wbg_ptr, source, target, use_landmarks, budget);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var r2 = getDataViewMemory0().getInt32(retptr + 4 * 2, true);
            if (r2) {
                throw takeObject(r1);
            }
            return takeObject(r0);
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) WasmRuFieldRouter.prototype[Symbol.dispose] = WasmRuFieldRouter.prototype.free;

/**
 * WASM wrapper for ThreeLevelHierarchy
 *
 * Implements the 3-level decomposition from arXiv:2512.13105:
 * - Level 0: Expanders (φ-expander subgraphs)
 * - Level 1: Preclusters (groups of expanders)
 * - Level 2: Clusters (top-level grouping with mirror cuts)
 */
export class WasmThreeLevelHierarchy {
    static __wrap(ptr) {
        const obj = Object.create(WasmThreeLevelHierarchy.prototype);
        obj.__wbg_ptr = ptr;
        WasmThreeLevelHierarchyFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        WasmThreeLevelHierarchyFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wasmthreelevelhierarchy_free(ptr, 0);
    }
    /**
     * Build the complete 3-level hierarchy
     *
     * Must be called after inserting edges to compute the decomposition.
     */
    build() {
        wasm.wasmthreelevelhierarchy_build(this.__wbg_ptr);
    }
    /**
     * Delete an edge from the graph
     * @param {bigint} u
     * @param {bigint} v
     */
    deleteEdge(u, v) {
        wasm.wasmthreelevelhierarchy_deleteEdge(this.__wbg_ptr, u, v);
    }
    /**
     * Get the global minimum cut estimate
     * @returns {number}
     */
    globalMinCut() {
        const ret = wasm.wasmthreelevelhierarchy_globalMinCut(this.__wbg_ptr);
        return ret;
    }
    /**
     * Insert an edge into the graph
     * @param {bigint} u
     * @param {bigint} v
     * @param {number} weight
     */
    insertEdge(u, v, weight) {
        wasm.wasmthreelevelhierarchy_insertEdge(this.__wbg_ptr, u, v, weight);
    }
    /**
     * Create a new hierarchy with default configuration
     */
    constructor() {
        const ret = wasm.wasmthreelevelhierarchy_new();
        this.__wbg_ptr = ret;
        WasmThreeLevelHierarchyFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Get hierarchy statistics as JSON
     * @returns {any}
     */
    stats() {
        const ret = wasm.wasmthreelevelhierarchy_stats(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Get all vertices as JSON array
     * @returns {any}
     */
    vertices() {
        const ret = wasm.wasmthreelevelhierarchy_vertices(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Create hierarchy with custom expansion parameter φ
     * @param {number} phi
     * @returns {WasmThreeLevelHierarchy}
     */
    static withPhi(phi) {
        const ret = wasm.wasmthreelevelhierarchy_withPhi(phi);
        return WasmThreeLevelHierarchy.__wrap(ret);
    }
}
if (Symbol.dispose) WasmThreeLevelHierarchy.prototype[Symbol.dispose] = WasmThreeLevelHierarchy.prototype.free;

/**
 * Get version information
 * @returns {string}
 */
export function getVersion() {
    let deferred1_0;
    let deferred1_1;
    try {
        const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
        wasm.getVersion(retptr);
        var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
        var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
        deferred1_0 = r0;
        deferred1_1 = r1;
        return getStringFromWasm0(r0, r1);
    } finally {
        wasm.__wbindgen_add_to_stack_pointer(16);
        wasm.__wbindgen_export4(deferred1_0, deferred1_1, 1);
    }
}

/**
 * Initialize the WASM module (call once at startup)
 *
 * This sets up panic hooks for better error messages in the browser console.
 */
export function init() {
    wasm.init();
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_fdd633d4bb5dd76a: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return addHeapObject(ret);
        },
        __wbg_Number_c4bdf66bb78f7977: function(arg0) {
            const ret = Number(getObject(arg0));
            return ret;
        },
        __wbg_String_8564e559799eccda: function(arg0, arg1) {
            const ret = String(getObject(arg1));
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_bigint_get_as_i64_d9e915702856f831: function(arg0, arg1) {
            const v = getObject(arg1);
            const ret = typeof(v) === 'bigint' ? v : undefined;
            getDataViewMemory0().setBigInt64(arg0 + 8 * 1, isLikeNone(ret) ? BigInt(0) : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_boolean_get_edaed31a367ce1bd: function(arg0) {
            const v = getObject(arg0);
            const ret = typeof(v) === 'boolean' ? v : undefined;
            return isLikeNone(ret) ? 0xFFFFFF : ret ? 1 : 0;
        },
        __wbg___wbindgen_debug_string_8a447059637473e2: function(arg0, arg1) {
            const ret = debugString(getObject(arg1));
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_in_4990f46af709e33c: function(arg0, arg1) {
            const ret = getObject(arg0) in getObject(arg1);
            return ret;
        },
        __wbg___wbindgen_is_bigint_90b5ccfe67c78460: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'bigint';
            return ret;
        },
        __wbg___wbindgen_is_function_acc5528be2b923f2: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_object_0beba4a1980d3eea: function(arg0) {
            const val = getObject(arg0);
            const ret = typeof(val) === 'object' && val !== null;
            return ret;
        },
        __wbg___wbindgen_is_string_1fca8072260dd261: function(arg0) {
            const ret = typeof(getObject(arg0)) === 'string';
            return ret;
        },
        __wbg___wbindgen_is_undefined_721f8decd50c87a3: function(arg0) {
            const ret = getObject(arg0) === undefined;
            return ret;
        },
        __wbg___wbindgen_jsval_eq_4e8c38722cb8ff51: function(arg0, arg1) {
            const ret = getObject(arg0) === getObject(arg1);
            return ret;
        },
        __wbg___wbindgen_jsval_loose_eq_4b9aba9e5b3c4582: function(arg0, arg1) {
            const ret = getObject(arg0) == getObject(arg1);
            return ret;
        },
        __wbg___wbindgen_number_get_1cc01dd708740256: function(arg0, arg1) {
            const obj = getObject(arg1);
            const ret = typeof(obj) === 'number' ? obj : undefined;
            getDataViewMemory0().setFloat64(arg0 + 8 * 1, isLikeNone(ret) ? 0 : ret, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, !isLikeNone(ret), true);
        },
        __wbg___wbindgen_string_get_71bb4348194e31f0: function(arg0, arg1) {
            const obj = getObject(arg1);
            const ret = typeof(obj) === 'string' ? obj : undefined;
            var ptr1 = isLikeNone(ret) ? 0 : passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            var len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg___wbindgen_throw_ea4887a5f8f9a9db: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_call_8e98ed2f3c86c4b5: function() { return handleError(function (arg0, arg1) {
            const ret = getObject(arg0).call(getObject(arg1));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_done_b62d4a7d2286852a: function(arg0) {
            const ret = getObject(arg0).done;
            return ret;
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_export4(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_get_9a29be2cb383ed9a: function() { return handleError(function (arg0, arg1) {
            const ret = Reflect.get(getObject(arg0), getObject(arg1));
            return addHeapObject(ret);
        }, arguments); },
        __wbg_get_unchecked_54a4374c38e08460: function(arg0, arg1) {
            const ret = getObject(arg0)[arg1 >>> 0];
            return addHeapObject(ret);
        },
        __wbg_get_with_ref_key_6412cf3094599694: function(arg0, arg1) {
            const ret = getObject(arg0)[getObject(arg1)];
            return addHeapObject(ret);
        },
        __wbg_instanceof_ArrayBuffer_2a7bb09fee70c2da: function(arg0) {
            let result;
            try {
                result = getObject(arg0) instanceof ArrayBuffer;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_instanceof_Uint8Array_f080092dc70f5d58: function(arg0) {
            let result;
            try {
                result = getObject(arg0) instanceof Uint8Array;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_isArray_145a34fd0a38d37b: function(arg0) {
            const ret = Array.isArray(getObject(arg0));
            return ret;
        },
        __wbg_isSafeInteger_a3389a198582f5f6: function(arg0) {
            const ret = Number.isSafeInteger(getObject(arg0));
            return ret;
        },
        __wbg_iterator_cc47ba25a2be735a: function() {
            const ret = Symbol.iterator;
            return addHeapObject(ret);
        },
        __wbg_length_589238bdcf171f0e: function(arg0) {
            const ret = getObject(arg0).length;
            return ret;
        },
        __wbg_length_c6054974c0a6cdb9: function(arg0) {
            const ret = getObject(arg0).length;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return addHeapObject(ret);
        },
        __wbg_new_2e117a478906f062: function() {
            const ret = new Object();
            return addHeapObject(ret);
        },
        __wbg_new_3444eb7412549f0b: function() {
            const ret = new Map();
            return addHeapObject(ret);
        },
        __wbg_new_36e147a8ced3c6e0: function() {
            const ret = new Array();
            return addHeapObject(ret);
        },
        __wbg_new_81880fb5002cb255: function(arg0) {
            const ret = new Uint8Array(getObject(arg0));
            return addHeapObject(ret);
        },
        __wbg_next_0c4066e251d2eff9: function() { return handleError(function (arg0) {
            const ret = getObject(arg0).next();
            return addHeapObject(ret);
        }, arguments); },
        __wbg_next_402fa10b59ab20c3: function(arg0) {
            const ret = getObject(arg0).next;
            return addHeapObject(ret);
        },
        __wbg_prototypesetcall_d721637c7ca66eb8: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), getObject(arg2));
        },
        __wbg_set_6be42768c690e380: function(arg0, arg1, arg2) {
            getObject(arg0)[takeObject(arg1)] = takeObject(arg2);
        },
        __wbg_set_9a1d61e17de7054c: function(arg0, arg1, arg2) {
            const ret = getObject(arg0).set(getObject(arg1), getObject(arg2));
            return addHeapObject(ret);
        },
        __wbg_set_dc601f4a69da0bc2: function(arg0, arg1, arg2) {
            getObject(arg0)[arg1 >>> 0] = takeObject(arg2);
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = getObject(arg1).stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_export, wasm.__wbindgen_export2);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_value_49f783bb59765962: function(arg0) {
            const ret = getObject(arg0).value;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000002: function(arg0) {
            // Cast intrinsic for `I64 -> Externref`.
            const ret = arg0;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000004: function(arg0) {
            // Cast intrinsic for `U64 -> Externref`.
            const ret = BigInt.asUintN(64, arg0);
            return addHeapObject(ret);
        },
        __wbindgen_object_clone_ref: function(arg0) {
            const ret = getObject(arg0);
            return addHeapObject(ret);
        },
        __wbindgen_object_drop_ref: function(arg0) {
            takeObject(arg0);
        },
    };
    return {
        __proto__: null,
        "./ruvector_mincut_wasm_bg.js": import0,
    };
}

const WasmLocalKCutFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmlocalkcut_free(ptr, 1));
const WasmMinCutFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmincut_free(ptr, 1));
const WasmMinCutWrapperFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmmincutwrapper_free(ptr, 1));
const WasmRoadRouterFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmroadrouter_free(ptr, 1));
const WasmRuFieldRouterFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmrufieldrouter_free(ptr, 1));
const WasmThreeLevelHierarchyFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wasmthreelevelhierarchy_free(ptr, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function debugString(val) {
    // primitive types
    const type = typeof val;
    if (type == 'number' || type == 'boolean' || val == null) {
        return  `${val}`;
    }
    if (type == 'string') {
        return `"${val}"`;
    }
    if (type == 'symbol') {
        const description = val.description;
        if (description == null) {
            return 'Symbol';
        } else {
            return `Symbol(${description})`;
        }
    }
    if (type == 'function') {
        const name = val.name;
        if (typeof name == 'string' && name.length > 0) {
            return `Function(${name})`;
        } else {
            return 'Function';
        }
    }
    // objects
    if (Array.isArray(val)) {
        const length = val.length;
        let debug = '[';
        if (length > 0) {
            debug += debugString(val[0]);
        }
        for(let i = 1; i < length; i++) {
            debug += ', ' + debugString(val[i]);
        }
        debug += ']';
        return debug;
    }
    // Test for built-in
    const builtInMatches = /\[object ([^\]]+)\]/.exec(toString.call(val));
    let className;
    if (builtInMatches && builtInMatches.length > 1) {
        className = builtInMatches[1];
    } else {
        // Failed to match the standard '[object ClassName]'
        return toString.call(val);
    }
    if (className == 'Object') {
        // we're a user defined class or Object
        // JSON.stringify avoids problems with cycles, and is generally much
        // easier than looping through ownProperties of `val`.
        try {
            return 'Object(' + JSON.stringify(val) + ')';
        } catch (_) {
            return 'Object';
        }
    }
    // errors
    if (val instanceof Error) {
        return `${val.name}: ${val.message}\n${val.stack}`;
    }
    // TODO we could test for more things here, like `Set`s and `Map`s.
    return className;
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export3(addHeapObject(e));
    }
}

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('ruvector_mincut_wasm_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
