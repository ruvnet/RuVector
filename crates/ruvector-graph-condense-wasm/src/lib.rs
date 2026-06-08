//! WASM bindings for `ruvector-graph-condense`.
//!
//! Exposes the structure-preserving condenser and the trained differentiable
//! min-cut condenser to JavaScript / the browser / edge runtimes, so a graph can
//! be condensed into a small deployable artifact client-side. Built without the
//! `parallel` (Rayon) feature, since `wasm32-unknown-unknown` has no threads.
//!
//! Graphs are passed as flat typed arrays from JS (`src`, `dst`, `w`: parallel
//! arrays, one entry per undirected edge; `features`: row-major `n × dim` `f32`
//! embeddings). Results are returned as JSON (a serialised `CondensedGraph`).

use ruvector_graph_condense::{
    CondenseConfig, CondenseMethod, DiffCutConfig, GraphCondenser, NodeFeatures,
};
use ruvector_mincut::DynamicGraph;
use wasm_bindgen::prelude::*;

fn build(
    n: u32,
    src: &[u32],
    dst: &[u32],
    w: &[f32],
    features: &[f32],
    dim: u32,
) -> Result<(DynamicGraph, NodeFeatures), String> {
    let n = n as usize;
    let dim = dim as usize;
    if src.len() != dst.len() || src.len() != w.len() {
        return Err("src/dst/w length mismatch".into());
    }
    if features.len() != n * dim {
        return Err(format!(
            "features length {} != n*dim {}",
            features.len(),
            n * dim
        ));
    }
    let g = DynamicGraph::new();
    let mut f = NodeFeatures::new(dim, 0);
    for v in 0..n {
        f.set_embedding(v as u64, features[v * dim..(v + 1) * dim].to_vec())
            .map_err(|e| e.to_string())?;
        g.add_vertex(v as u64);
    }
    for i in 0..src.len() {
        let _ = g.insert_edge(src[i] as u64, dst[i] as u64, w[i] as f64);
    }
    Ok((g, f))
}

fn run(config: CondenseConfig, args: BuildArgs) -> String {
    match build(args.n, args.src, args.dst, args.w, args.features, args.dim) {
        Ok((g, f)) => match GraphCondenser::new(config).condense(&g, &f) {
            Ok(c) => serde_json::to_string(&c).unwrap_or_else(|e| err_json(&e.to_string())),
            Err(e) => err_json(&e.to_string()),
        },
        Err(e) => err_json(&e),
    }
}

struct BuildArgs<'a> {
    n: u32,
    src: &'a [u32],
    dst: &'a [u32],
    w: &'a [f32],
    features: &'a [f32],
    dim: u32,
}

fn err_json(msg: &str) -> String {
    format!(
        "{{\"error\":{}}}",
        serde_json::to_string(msg).unwrap_or_default()
    )
}

/// Condense with the default structure-preserving `WeakBoundary` method.
/// Returns a JSON `CondensedGraph` (or `{"error": "..."}`).
#[wasm_bindgen]
pub fn condense_weak(
    n: u32,
    src: &[u32],
    dst: &[u32],
    w: &[f32],
    features: &[f32],
    dim: u32,
) -> String {
    run(
        CondenseConfig::default(),
        BuildArgs {
            n,
            src,
            dst,
            w,
            features,
            dim,
        },
    )
}

/// Condense with the trained differentiable min-cut method (Adam + warm-start).
#[wasm_bindgen]
pub fn condense_diffmincut(
    n: u32,
    src: &[u32],
    dst: &[u32],
    w: &[f32],
    features: &[f32],
    dim: u32,
    num_clusters: u32,
) -> String {
    let cfg = CondenseConfig {
        method: CondenseMethod::DiffMinCut(DiffCutConfig {
            num_clusters: num_clusters.max(1) as usize,
            ..Default::default()
        }),
        normalize_centroids: false,
    };
    run(
        cfg,
        BuildArgs {
            n,
            src,
            dst,
            w,
            features,
            dim,
        },
    )
}

/// Crate version (handy for cache-busting a deployed bundle).
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
