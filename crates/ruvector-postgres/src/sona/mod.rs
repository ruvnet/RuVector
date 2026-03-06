//! Sona self-learning module — Micro-LoRA trajectories and EWC++ for PostgreSQL.

pub mod operators;

use dashmap::DashMap;
use ruvector_sona::{SonaConfig, SonaEngine};
use std::sync::Arc;

/// Cache key includes dimension so different-dim inputs get separate engines.
fn engine_key(table_name: &str, dim: u32) -> String {
    format!("{}::{}", table_name, dim)
}

/// Global Sona engine state per table+dimension.
static SONA_ENGINES: once_cell::sync::Lazy<DashMap<String, Arc<SonaEngine>>> =
    once_cell::sync::Lazy::new(DashMap::new);

/// Default dimension when none is specified (e.g., for stats queries).
const DEFAULT_DIM: u32 = 256;

/// Get or create a SonaEngine for a given table with default dimension.
pub fn get_or_create_engine(table_name: &str) -> Arc<SonaEngine> {
    get_or_create_engine_with_dim(table_name, DEFAULT_DIM)
}

/// Get or create a SonaEngine for a given table and embedding dimension.
pub fn get_or_create_engine_with_dim(table_name: &str, dim: u32) -> Arc<SonaEngine> {
    let key = engine_key(table_name, dim);
    SONA_ENGINES
        .entry(key)
        .or_insert_with(|| {
            Arc::new(SonaEngine::with_config(SonaConfig {
                hidden_dim: dim as usize,
                embedding_dim: dim as usize,
                ..Default::default()
            }))
        })
        .value()
        .clone()
}
