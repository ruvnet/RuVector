//! WASM bindings for `ruvector-sparsifier`.
//!
//! Provides [`WasmSparsifier`] for dynamic spectral graph sparsification
//! in the browser or any WASM runtime.

use wasm_bindgen::prelude::*;

use ruvector_sparsifier::{
    AdaptiveGeoSpar, SparseGraph, SparsifierConfig,
    traits::Sparsifier,
};

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

/// Initialise the WASM module (sets up the panic hook for better error messages).
#[wasm_bindgen]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Return the crate version.
#[wasm_bindgen]
pub fn version() -> String {
    ruvector_sparsifier::VERSION.to_string()
}

/// Return the default configuration as a JSON string.
#[wasm_bindgen]
pub fn default_config() -> String {
    serde_json::to_string_pretty(&SparsifierConfig::default()).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// WasmSparseGraph
// ---------------------------------------------------------------------------

/// A lightweight sparse graph for building input data.
#[wasm_bindgen]
pub struct WasmSparseGraph {
    inner: SparseGraph,
}

#[wasm_bindgen]
impl WasmSparseGraph {
    /// Create a new empty graph with `n` vertices.
    #[wasm_bindgen(constructor)]
    pub fn new(n: u32) -> Self {
        Self {
            inner: SparseGraph::with_capacity(n as usize),
        }
    }

    /// Add an undirected edge.
    #[wasm_bindgen(js_name = addEdge)]
    pub fn add_edge(&mut self, u: u32, v: u32, weight: f64) -> Result<(), JsValue> {
        self.inner
            .insert_edge(u as usize, v as usize, weight)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Remove an edge.
    #[wasm_bindgen(js_name = removeEdge)]
    pub fn remove_edge(&mut self, u: u32, v: u32) -> Result<(), JsValue> {
        self.inner
            .delete_edge(u as usize, v as usize)
            .map(|_| ())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Degree of vertex `u`.
    pub fn degree(&self, u: u32) -> u32 {
        self.inner.degree(u as usize) as u32
    }

    /// Number of undirected edges.
    #[wasm_bindgen(js_name = numEdges)]
    pub fn num_edges(&self) -> u32 {
        self.inner.num_edges() as u32
    }

    /// Number of vertices.
    #[wasm_bindgen(js_name = numVertices)]
    pub fn num_vertices(&self) -> u32 {
        self.inner.num_vertices() as u32
    }

    /// Serialise the edge list to JSON.
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        let edges: Vec<(usize, usize, f64)> = self.inner.edges().collect();
        serde_json::to_string(&edges).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// WasmSparsifier
// ---------------------------------------------------------------------------

/// Dynamic spectral graph sparsifier for WASM.
///
/// Maintains a compressed shadow graph that preserves the Laplacian energy
/// of the full graph within `(1 ± epsilon)`.
#[wasm_bindgen]
pub struct WasmSparsifier {
    inner: AdaptiveGeoSpar,
}

#[wasm_bindgen]
impl WasmSparsifier {
    /// Create a new sparsifier with the given JSON configuration.
    ///
    /// Pass `"{}"` or `default_config()` for defaults.
    #[wasm_bindgen(constructor)]
    pub fn new(config_json: &str) -> Result<WasmSparsifier, JsValue> {
        let config: SparsifierConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("invalid config: {e}")))?;
        Ok(Self {
            inner: AdaptiveGeoSpar::new(config),
        })
    }

    /// Build a sparsifier from a JSON edge list: `[[u, v, weight], ...]`.
    #[wasm_bindgen(js_name = buildFromEdges)]
    pub fn build_from_edges(
        edges_json: &str,
        config_json: &str,
    ) -> Result<WasmSparsifier, JsValue> {
        let edges: Vec<(usize, usize, f64)> = serde_json::from_str(edges_json)
            .map_err(|e| JsValue::from_str(&format!("invalid edges: {e}")))?;
        let config: SparsifierConfig = serde_json::from_str(config_json)
            .map_err(|e| JsValue::from_str(&format!("invalid config: {e}")))?;

        let graph = SparseGraph::from_edges(&edges);
        let spar = AdaptiveGeoSpar::build(&graph, config)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self { inner: spar })
    }

    /// Insert an edge into the full graph and update the sparsifier.
    #[wasm_bindgen(js_name = insertEdge)]
    pub fn insert_edge(&mut self, u: u32, v: u32, weight: f64) -> Result<(), JsValue> {
        self.inner
            .insert_edge(u as usize, v as usize, weight)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Delete an edge from the full graph and update the sparsifier.
    #[wasm_bindgen(js_name = deleteEdge)]
    pub fn delete_edge(&mut self, u: u32, v: u32) -> Result<(), JsValue> {
        self.inner
            .delete_edge(u as usize, v as usize)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Handle an embedding move for a node (JSON: `[[neighbor, weight], ...]`).
    #[wasm_bindgen(js_name = updateEmbedding)]
    pub fn update_embedding(
        &mut self,
        node: u32,
        old_neighbors_json: &str,
        new_neighbors_json: &str,
    ) -> Result<(), JsValue> {
        let old: Vec<(usize, f64)> = serde_json::from_str(old_neighbors_json)
            .map_err(|e| JsValue::from_str(&format!("invalid old_neighbors: {e}")))?;
        let new: Vec<(usize, f64)> = serde_json::from_str(new_neighbors_json)
            .map_err(|e| JsValue::from_str(&format!("invalid new_neighbors: {e}")))?;

        self.inner
            .update_embedding(node as usize, &old, &new)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Run a spectral audit and return the result as JSON.
    pub fn audit(&self) -> String {
        let result = self.inner.audit();
        serde_json::to_string(&result).unwrap_or_default()
    }

    /// Get the current sparsifier edges as JSON.
    #[wasm_bindgen(js_name = sparsifierEdges)]
    pub fn sparsifier_edges(&self) -> String {
        let edges: Vec<(usize, usize, f64)> = self.inner.sparsifier().edges().collect();
        serde_json::to_string(&edges).unwrap_or_default()
    }

    /// Get the statistics as JSON.
    pub fn stats(&self) -> String {
        serde_json::to_string(self.inner.stats()).unwrap_or_default()
    }

    /// Get the current compression ratio.
    #[wasm_bindgen(js_name = compressionRatio)]
    pub fn compression_ratio(&self) -> f64 {
        self.inner.compression_ratio()
    }

    /// Rebuild the sparsifier around specific nodes (JSON array of u32).
    #[wasm_bindgen(js_name = rebuildLocal)]
    pub fn rebuild_local(&mut self, nodes_json: &str) -> Result<(), JsValue> {
        let nodes: Vec<usize> = serde_json::from_str(nodes_json)
            .map_err(|e| JsValue::from_str(&format!("invalid nodes: {e}")))?;
        self.inner
            .rebuild_local(&nodes)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Full reconstruction of the sparsifier.
    #[wasm_bindgen(js_name = rebuildFull)]
    pub fn rebuild_full(&mut self) -> Result<(), JsValue> {
        self.inner
            .rebuild_full()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Number of vertices in the full graph.
    #[wasm_bindgen(js_name = numVertices)]
    pub fn num_vertices(&self) -> u32 {
        self.inner.full_graph().num_vertices() as u32
    }

    /// Number of edges in the full graph.
    #[wasm_bindgen(js_name = numEdges)]
    pub fn num_edges(&self) -> u32 {
        self.inner.full_graph().num_edges() as u32
    }

    /// Number of edges in the sparsifier.
    #[wasm_bindgen(js_name = sparsifierNumEdges)]
    pub fn sparsifier_num_edges(&self) -> u32 {
        self.inner.sparsifier().num_edges() as u32
    }
}
