/* tslint:disable */
/* eslint-disable */

/**
 * Minimal DAG structure for WASM
 * Self-contained with no external dependencies beyond wasm-bindgen
 */
export class WasmDag {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Add edge from -> to
     * Returns false if creates cycle (simple check)
     */
    add_edge(from: number, to: number): boolean;
    /**
     * Add a node with operator type and cost
     * Returns node ID
     */
    add_node(op: number, cost: number): number;
    /**
     * Compute attention scores for nodes
     * mechanism: 0=topological, 1=critical_path, 2=uniform
     */
    attention(mechanism: number): Float32Array;
    /**
     * Find critical path (longest path by cost)
     * Returns JSON: {"path": [node_ids], "cost": total}
     */
    critical_path(): any;
    /**
     * Get number of edges
     */
    edge_count(): number;
    /**
     * Deserialize from bytes
     */
    static from_bytes(data: Uint8Array): WasmDag;
    /**
     * Deserialize from JSON
     */
    static from_json(json: string): WasmDag;
    /**
     * Create new empty DAG
     */
    constructor();
    /**
     * Get number of nodes
     */
    node_count(): number;
    /**
     * Serialize to bytes (bincode format)
     */
    to_bytes(): Uint8Array;
    /**
     * Serialize to JSON
     */
    to_json(): string;
    /**
     * Topological sort using Kahn's algorithm
     * Returns node IDs in topological order
     */
    topo_sort(): Uint32Array;
}
