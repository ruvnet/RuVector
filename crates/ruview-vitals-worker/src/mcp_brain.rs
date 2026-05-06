//! Reusable router + types for the `ruview-mcp-brain-mini` binary.
//!
//! Wire shape mirrors the existing `mcp-brain-serve` REST surface so
//! both [`crate::brain::BrainClient`] and RuView's `brain_bridge.rs`
//! POST into it unchanged:
//!
//! ```text
//!   POST /memories    {category, content}
//!     -> 201 {id, category, content, content_hash, created_at}
//!
//!   GET /memories?category=X&limit=N&offset=M
//!     -> 200 {count, total, offset, memories: [...]}
//!
//!   GET /health  -> 200 "ok"
//! ```
//!
//! Pulled out of `src/bin/ruview-mcp-brain-mini.rs` so integration
//! tests can [`build_app`] + axum::serve in-process without a
//! subprocess (ADR-183 Tier 2 iter 12).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{DefaultBodyLimit, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

/// Default body cap. A vital-sign memory is ~200 B; 16 KiB is
/// generous headroom while bounding DoS via huge POSTs.
pub const DEFAULT_BODY_LIMIT_BYTES: usize = 16 * 1024;
/// Per-field caps applied inside `post_memory`.
pub const MAX_CATEGORY_LEN: usize = 256;
pub const MAX_CONTENT_LEN: usize = 8 * 1024;

/// One persisted memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub category: String,
    pub content: String,
    pub content_hash: String,
    pub created_at: u64,
}

#[derive(Debug, Deserialize)]
pub struct PostBody {
    pub category: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub category: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// In-memory store with optional JSONL append-only persistence.
#[derive(Default)]
pub struct Store {
    pub memories: Vec<Memory>,
    pub store_path: Option<PathBuf>,
}

impl Store {
    /// Build an empty store; if `path` is `Some` and the file exists,
    /// replay its lines into `memories`. Corrupt lines are skipped
    /// with a `tracing::warn!`.
    #[must_use]
    pub fn load(path: Option<PathBuf>) -> Self {
        let mut store = Self {
            memories: Vec::new(),
            store_path: path.clone(),
        };
        if let Some(p) = path {
            if p.exists() {
                if let Ok(contents) = std::fs::read_to_string(&p) {
                    let mut loaded = 0usize;
                    let mut skipped = 0usize;
                    for (i, line) in contents.lines().enumerate() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<Memory>(line) {
                            Ok(m) => {
                                store.memories.push(m);
                                loaded += 1;
                            }
                            Err(e) => {
                                tracing::warn!(line = i + 1, error = %e, "skip corrupt line");
                                skipped += 1;
                            }
                        }
                    }
                    tracing::info!(loaded, skipped, path = %p.display(), "restored from JSONL");
                }
            }
        }
        store
    }
}

/// Build the axum router with body limit applied.
#[must_use]
pub fn build_app(store: Arc<RwLock<Store>>, body_limit_bytes: usize) -> Router {
    Router::new()
        .route("/memories", get(list_memories).post(post_memory))
        .route("/health", get(health))
        .layer(DefaultBodyLimit::max(body_limit_bytes))
        .with_state(store)
}

async fn post_memory(
    State(store): State<Arc<RwLock<Store>>>,
    Json(body): Json<PostBody>,
) -> Result<(StatusCode, Json<Memory>), (StatusCode, Json<HashMap<String, String>>)> {
    if body.category.is_empty() || body.content.is_empty() {
        let mut err = HashMap::new();
        err.insert("error".into(), "category and content must be non-empty".into());
        return Err((StatusCode::BAD_REQUEST, Json(err)));
    }
    if body.category.len() > MAX_CATEGORY_LEN || body.content.len() > MAX_CONTENT_LEN {
        let mut err = HashMap::new();
        err.insert(
            "error".into(),
            format!(
                "category > {MAX_CATEGORY_LEN} B or content > {MAX_CONTENT_LEN} B rejected"
            ),
        );
        return Err((StatusCode::PAYLOAD_TOO_LARGE, Json(err)));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let content_hash = {
        let mut h = Sha256::new();
        h.update(body.content.as_bytes());
        format!("{:x}", h.finalize())
    };
    let id = {
        let mut h = Sha256::new();
        h.update(now.to_be_bytes());
        h.update(body.category.as_bytes());
        h.update(body.content.as_bytes());
        let hex = format!("{:x}", h.finalize());
        hex.chars().take(32).collect::<String>()
    };

    let memory = Memory {
        id,
        category: body.category,
        content: body.content,
        content_hash,
        created_at: now,
    };

    let mut g = store.write().await;
    if let Some(path) = &g.store_path {
        if let Ok(line) = serde_json::to_string(&memory) {
            let _ = append_line(path, &line);
        }
    }
    g.memories.push(memory.clone());
    drop(g);
    tracing::debug!(category = %memory.category, "POST /memories ok");
    Ok((StatusCode::CREATED, Json(memory)))
}

fn append_line(path: &std::path::Path, line: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")
}

async fn list_memories(
    State(store): State<Arc<RwLock<Store>>>,
    Query(q): Query<ListQuery>,
) -> Json<serde_json::Value> {
    let g = store.read().await;
    let limit = q.limit.unwrap_or(50).min(1000);
    let offset = q.offset.unwrap_or(0);
    let filtered: Vec<&Memory> = g
        .memories
        .iter()
        .rev()
        .filter(|m| q.category.as_ref().map_or(true, |c| &m.category == c))
        .skip(offset)
        .take(limit)
        .collect();
    Json(serde_json::json!({
        "count": filtered.len(),
        "total": g.memories.len(),
        "offset": offset,
        "memories": filtered,
    }))
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}
