//! Point operations endpoints

use crate::{error::Error, state::AppState, Result};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post, put},
    Json, Router,
};
use ruvector_core::{SearchQuery, SearchResult, VectorEntry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Point upsert request
#[derive(Debug, Deserialize)]
pub struct UpsertPointsRequest {
    /// Points to upsert
    pub points: Vec<VectorEntry>,
}

/// Search request
#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results to return
    #[serde(default = "default_limit")]
    pub k: usize,
    /// Optional score threshold
    pub score_threshold: Option<f32>,
    /// Optional metadata filters
    pub filter: Option<HashMap<String, serde_json::Value>>,
}

fn default_limit() -> usize {
    10
}

/// Maximum `k` accepted in a search request. Prevents memory exhaustion from
/// maliciously large result-set allocations.
const MAX_K: usize = 10_000;
/// Maximum number of points accepted in a single upsert batch.
const MAX_UPSERT_BATCH: usize = 10_000;
/// Maximum dimension accepted for a query vector.
const MAX_VECTOR_DIM: usize = 65_536;

/// Search response
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    /// Search results
    pub results: Vec<SearchResult>,
}

/// Upsert response
#[derive(Debug, Serialize)]
pub struct UpsertResponse {
    /// IDs of upserted points
    pub ids: Vec<String>,
}

/// Create point routes
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/collections/:name/points", put(upsert_points))
        .route("/collections/:name/points/search", post(search_points))
        .route("/collections/:name/points/:id", get(get_point))
}

/// Upsert points into a collection
///
/// PUT /collections/:name/points
async fn upsert_points(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertPointsRequest>,
) -> Result<impl IntoResponse> {
    // Security: guard against oversized batches that could exhaust memory.
    if req.points.len() > MAX_UPSERT_BATCH {
        return Err(Error::InvalidRequest(format!(
            "batch size {} exceeds maximum of {}",
            req.points.len(),
            MAX_UPSERT_BATCH
        )));
    }

    let db = state
        .get_collection(&name)
        .ok_or_else(|| Error::CollectionNotFound(name.clone()))?;

    let ids = db.insert_batch(req.points).map_err(Error::Core)?;

    Ok((StatusCode::OK, Json(UpsertResponse { ids })))
}

/// Search for similar points
///
/// POST /collections/:name/points/search
async fn search_points(
    State(state): State<AppState>,
    Path(name): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse> {
    // Security: guard against k=0, oversized k, and zero/oversized query vectors.
    if req.k == 0 {
        return Ok(Json(SearchResponse { results: vec![] }));
    }
    if req.k > MAX_K {
        return Err(Error::InvalidRequest(format!(
            "k={} exceeds maximum of {}",
            req.k, MAX_K
        )));
    }
    if req.vector.is_empty() {
        return Err(Error::InvalidRequest(
            "query vector must not be empty".into(),
        ));
    }
    if req.vector.len() > MAX_VECTOR_DIM {
        return Err(Error::InvalidRequest(format!(
            "query vector dimension {} exceeds maximum of {}",
            req.vector.len(),
            MAX_VECTOR_DIM
        )));
    }

    let db = state
        .get_collection(&name)
        .ok_or_else(|| Error::CollectionNotFound(name))?;

    let query = SearchQuery {
        vector: req.vector,
        k: req.k,
        filter: req.filter,
        ef_search: None,
    };

    let mut results = db.search(query).map_err(Error::Core)?;

    // Apply score threshold if provided
    if let Some(threshold) = req.score_threshold {
        results.retain(|r| r.score >= threshold);
    }

    Ok(Json(SearchResponse { results }))
}

/// Get a point by ID
///
/// GET /collections/:name/points/:id
async fn get_point(
    State(state): State<AppState>,
    Path((name, id)): Path<(String, String)>,
) -> Result<impl IntoResponse> {
    let db = state
        .get_collection(&name)
        .ok_or_else(|| Error::CollectionNotFound(name))?;

    let entry = db.get(&id).map_err(Error::Core)?;

    Ok(Json(entry))
}
