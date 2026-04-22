//! mcp-brain-server-local: local private brain backend (ADR-SYS-0002)
//!
//! Standalone binary with SQLite + optional RVF storage, brute-force-cosine
//! vector search, and a minimal axum REST API for the ruvultra local stack.
//!
//! Build: cargo build --release --no-default-features --features local-all \
//!        -p mcp-brain-server --bin mcp-brain-server-local

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, LazyLock, Mutex};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

// ── AIDefence (inline Rust port) ────────────────────────────────────────────

/// Critical injection + PII patterns compiled once.
static THREAT_PATTERNS: LazyLock<Vec<(regex::Regex, &'static str, &'static str)>> = LazyLock::new(
    || {
        let patterns: Vec<(&str, &str, &str)> = vec![
            (
                r"(?i)ignore\s+(previous|all|above|any|the)(\s+\w+)*\s+(instructions?|prompts?|rules?|context)",
                "injection",
                "high",
            ),
            (
                r"(?i)disregard\s+(previous|all|above|the|your)(\s+\w+)*\s+(instructions?|prompts?|input)",
                "injection",
                "high",
            ),
            (
                r"(?i)forget\s+(everything|all|previous|your)",
                "injection",
                "high",
            ),
            (r"(?i)you\s+are\s+(now|actually)\s+", "injection", "high"),
            (
                r"(?i)pretend\s+(to\s+be|you're|you\s+are)",
                "injection",
                "high",
            ),
            (
                r"(?i)what\s+(is|are)\s+your\s+(system\s+)?prompt",
                "extraction",
                "high",
            ),
            (
                r"(?i)show\s+(me\s+)?your\s+(system\s+)?instructions",
                "extraction",
                "high",
            ),
            (r"(?i)DAN\s+(mode|prompt)", "jailbreak", "critical"),
            (
                r"(?i)bypass\s+(safety|security|filter)",
                "jailbreak",
                "critical",
            ),
            (
                r"(?i)remove\s+(all\s+)?restrictions",
                "jailbreak",
                "critical",
            ),
            (r"<script", "code_injection", "critical"),
            (r"(?i)javascript:", "code_injection", "critical"),
            (r"(?i)eval\s*\(", "code_injection", "high"),
        ];
        patterns
            .into_iter()
            .filter_map(|(p, cat, sev)| regex::Regex::new(p).ok().map(|r| (r, cat, sev)))
            .collect()
    },
);

static PII_PATTERNS: LazyLock<Vec<(regex::Regex, &'static str)>> = LazyLock::new(|| {
    let patterns: Vec<(&str, &str)> = vec![
        (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "email",
        ),
        (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", "ssn"),
        (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "credit_card"),
        (r"\b(sk-|api[_-]?key|token)[a-zA-Z0-9_-]{20,}\b", "api_key"),
    ];
    patterns
        .into_iter()
        .filter_map(|(p, t)| regex::Regex::new(p).ok().map(|r| (r, t)))
        .collect()
});

/// Scan text for threats. Returns (safe, threat_level, details_json).
fn aidefence_scan(text: &str) -> (bool, &'static str, serde_json::Value) {
    let mut threats = Vec::new();
    let mut max_severity = 0u8;

    for (pattern, category, severity) in THREAT_PATTERNS.iter() {
        if pattern.is_match(text) {
            let sev_num = match *severity {
                "critical" => 4,
                "high" => 3,
                "medium" => 2,
                "low" => 1,
                _ => 0,
            };
            if sev_num > max_severity {
                max_severity = sev_num;
            }
            threats.push(serde_json::json!({
                "type": category, "severity": severity,
            }));
        }
    }

    for (pattern, pii_type) in PII_PATTERNS.iter() {
        if pattern.is_match(text) {
            if max_severity < 2 {
                max_severity = 2;
            }
            threats.push(serde_json::json!({
                "type": "pii", "pii_type": pii_type, "severity": "medium",
            }));
        }
    }

    let level = match max_severity {
        4 => "critical",
        3 => "high",
        2 => "medium",
        1 => "low",
        _ => "none",
    };
    let safe = max_severity < 2; // block at medium and above
    (safe, level, serde_json::json!(threats))
}

// ── Configuration ────────────────────────────────────────────────────────────

const VERSION: &str = "0.2.0";
const DEFAULT_DB: &str = "/home/ruvultra/brain-data/brain.sqlite";
const DEFAULT_BLOBS: &str = "/home/ruvultra/brain-data/blobs";

fn db_path() -> String {
    std::env::var("RUVBRAIN_DB").unwrap_or_else(|_| DEFAULT_DB.to_string())
}

fn blob_dir() -> String {
    std::env::var("RUVBRAIN_BLOBS").unwrap_or_else(|_| DEFAULT_BLOBS.to_string())
}

fn store_mode() -> String {
    std::env::var("RUVBRAIN_STORE").unwrap_or_else(|_| "rvf".to_string())
}

// ── Vector Index (DiskANN-inspired) ──────────────────────────────────────────

/// Hybrid vector index: Vamana graph routing in memory, vectors on SSD.
///
/// At <10K vectors the full index fits in RAM (~15 MB) so we keep it in-memory
/// for speed. The Vamana graph (64 neighbors per node, u32 IDs = ~256 KB for
/// 1K nodes) would allow scaling to 1M+ vectors with <50 MB RAM by memory-
/// mapping the vectors from disk.
///
/// DiskANN paper: "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search
/// on a Single Node" (Subramanya et al., NeurIPS 2019).
struct VectorIndex {
    /// Node data: (id_hex, category, normalized_embedding)
    entries: Vec<(String, String, Vec<f32>)>,
    /// Vamana graph: neighbors[i] = list of neighbor indices for node i
    neighbors: Vec<Vec<u32>>,
    /// Medoid (entry point for greedy search)
    medoid: usize,
    dim: usize,
    /// Max neighbors per node (R in the paper)
    max_degree: usize,
    /// SSD vector file path (for future disk-backed mode)
    ssd_path: Option<String>,
}

impl VectorIndex {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            neighbors: Vec::new(),
            medoid: 0,
            dim: 0,
            max_degree: 64,
            ssd_path: None,
        }
    }

    /// Enable SSD-backed vector storage (for future scaling).
    #[allow(dead_code)]
    fn with_ssd(mut self, path: &str) -> Self {
        self.ssd_path = Some(path.to_string());
        self
    }

    fn insert(&mut self, id_hex: &str, category: &str, embedding: &[f32]) {
        if embedding.is_empty() {
            return;
        }
        if self.dim == 0 {
            self.dim = embedding.len();
        }

        // Pre-normalize
        let norm = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 1e-10 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            vec![0.0; embedding.len()]
        };

        let new_idx = self.entries.len() as u32;
        self.entries
            .push((id_hex.to_string(), category.to_string(), normalized));
        self.neighbors.push(Vec::new());

        // Connect to graph using Vamana-style greedy insert
        if self.entries.len() > 1 {
            self.vamana_insert(new_idx);
        }
    }

    /// Vamana greedy insert: find nearest neighbors via graph traversal,
    /// then add bidirectional edges with robust pruning.
    fn vamana_insert(&mut self, new_idx: u32) {
        let n = self.entries.len();
        if n <= 1 {
            return;
        }

        // For small graphs (<500), just connect to nearest neighbors directly
        if n < 500 {
            let query = self.entries[new_idx as usize].2.clone();
            let mut dists: Vec<(f32, u32)> = (0..new_idx)
                .map(|i| (self.dot(&query, &self.entries[i as usize].2), i))
                .collect();
            dists.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(self.max_degree);

            for &(_, neighbor) in &dists {
                self.neighbors[new_idx as usize].push(neighbor);
                let nbrs = &mut self.neighbors[neighbor as usize];
                if nbrs.len() < self.max_degree {
                    nbrs.push(new_idx);
                }
            }
            return;
        }

        // Greedy search from medoid to find candidates
        let query = self.entries[new_idx as usize].2.clone();
        let candidates = self.greedy_search_internal(&query, self.max_degree, new_idx);

        // Add edges with pruning
        let mut selected: Vec<u32> = Vec::new();
        for &(_, cand_idx) in &candidates {
            if selected.len() >= self.max_degree {
                break;
            }
            // Robust pruning: only add if candidate is closer than α * distance
            // to any already-selected neighbor (α = 1.2)
            let cand_dist = self.dot(&query, &self.entries[cand_idx as usize].2);
            let mut dominated = false;
            for &sel in &selected {
                let sel_to_cand = self.dot(
                    &self.entries[sel as usize].2,
                    &self.entries[cand_idx as usize].2,
                );
                if sel_to_cand > cand_dist * 1.2 {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                selected.push(cand_idx);
            }
        }

        self.neighbors[new_idx as usize] = selected.clone();
        for &neighbor in &selected {
            let nbrs = &mut self.neighbors[neighbor as usize];
            if nbrs.len() < self.max_degree {
                nbrs.push(new_idx);
            }
        }

        // Update medoid if new point is more central
        if new_idx as usize % 100 == 0 {
            self.update_medoid();
        }
    }

    fn update_medoid(&mut self) {
        if self.entries.len() < 10 {
            return;
        }
        // Sample-based medoid: pick the point closest to the mean of a sample
        let sample_size = self.entries.len().min(100);
        let step = self.entries.len() / sample_size;
        let mut mean = vec![0.0f32; self.dim];
        let mut count = 0;
        for i in (0..self.entries.len()).step_by(step.max(1)) {
            for (j, v) in self.entries[i].2.iter().enumerate() {
                mean[j] += v;
            }
            count += 1;
        }
        if count > 0 {
            for v in &mut mean {
                *v /= count as f32;
            }
        }

        let mut best = self.medoid;
        let mut best_dist = self.dot(&mean, &self.entries[best].2);
        for i in (0..self.entries.len()).step_by(step.max(1)) {
            let d = self.dot(&mean, &self.entries[i].2);
            if d > best_dist {
                best_dist = d;
                best = i;
            }
        }
        self.medoid = best;
    }

    /// Greedy search on the Vamana graph. Returns (score, index) pairs.
    fn greedy_search_internal(&self, query: &[f32], k: usize, exclude: u32) -> Vec<(f32, u32)> {
        use std::collections::BinaryHeap;
        use std::collections::HashSet;

        if self.entries.is_empty() {
            return Vec::new();
        }

        let mut visited = HashSet::new();
        let mut candidates: BinaryHeap<std::cmp::Reverse<(ordered_float::OrderedFloat<f32>, u32)>> =
            BinaryHeap::new();
        let mut results: BinaryHeap<(ordered_float::OrderedFloat<f32>, u32)> = BinaryHeap::new();

        // Start from medoid
        let start = self.medoid.min(self.entries.len() - 1);
        let start_dist = self.dot(query, &self.entries[start].2);
        visited.insert(start as u32);
        candidates.push(std::cmp::Reverse((
            ordered_float::OrderedFloat(-start_dist),
            start as u32,
        )));
        if start as u32 != exclude {
            results.push((ordered_float::OrderedFloat(-start_dist), start as u32));
        }

        let beam_width = (k * 8).max(40);

        while let Some(std::cmp::Reverse((_, current))) = candidates.pop() {
            if current as usize >= self.neighbors.len() {
                continue;
            }

            for &neighbor in &self.neighbors[current as usize] {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let dist = self.dot(query, &self.entries[neighbor as usize].2);

                if neighbor != exclude {
                    if results.len() < beam_width {
                        results.push((ordered_float::OrderedFloat(-dist), neighbor));
                    } else if let Some(&(worst, _)) = results.peek() {
                        if -dist < worst.0 {
                            results.pop();
                            results.push((ordered_float::OrderedFloat(-dist), neighbor));
                        }
                    }
                }

                candidates.push(std::cmp::Reverse((
                    ordered_float::OrderedFloat(-dist),
                    neighbor,
                )));
            }

            if visited.len() > beam_width * 4 {
                break;
            }
        }

        let mut out: Vec<(f32, u32)> = results.into_iter().map(|(d, idx)| (-d.0, idx)).collect();
        out.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(k);
        out
    }

    /// Public search: normalize query, search graph, return (score, id_hex).
    fn search(&self, query: &[f32], k: usize) -> Vec<(f64, String)> {
        if query.is_empty() || self.entries.is_empty() {
            return Vec::new();
        }

        let qnorm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if qnorm < 1e-10 {
            return Vec::new();
        }
        let q: Vec<f32> = query.iter().map(|x| x / qnorm).collect();

        // For small indexes (<2000), brute force is faster than graph traversal
        if self.entries.len() < 2000 {
            return self.brute_force_search(&q, k);
        }

        let results = self.greedy_search_internal(&q, k, u32::MAX);
        results
            .into_iter()
            .map(|(score, idx)| (score as f64, self.entries[idx as usize].0.clone()))
            .collect()
    }

    /// Brute force fallback for small indexes.
    fn brute_force_search(&self, q: &[f32], k: usize) -> Vec<(f64, String)> {
        let mut results: Vec<(f64, &str)> = self
            .entries
            .iter()
            .map(|(id, _, v)| {
                let dot: f64 = q
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                (dot, id.as_str())
            })
            .collect();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
            .into_iter()
            .map(|(s, id)| (s, id.to_string()))
            .collect()
    }

    #[inline]
    fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

// ── App State ────────────────────────────────────────────────────────────────

struct AppState {
    db: Mutex<Connection>,
    index: Mutex<VectorIndex>,
    blob_dir: String,
    mode: String,
}

type SharedState = Arc<AppState>;

// ── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"));
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(filter)
        .init();

    let path = db_path();
    let blobs = blob_dir();
    let mode = store_mode();

    // Ensure blob dir exists
    let _ = std::fs::create_dir_all(&blobs);

    let conn = Connection::open(&path)?;
    conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
    ensure_schema(&conn)?;

    // Build vector index from all memories with embeddings
    let mut index = VectorIndex::new();
    {
        let mut stmt = conn.prepare(
            "SELECT hex(id), category, embedding FROM memories WHERE embedding IS NOT NULL AND length(embedding) > 0"
        )?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let cat: String = row.get::<_, String>(1).unwrap_or_default();
            let blob: Vec<u8> = row.get(2)?;
            Ok((id, cat, blob))
        })?;
        for row in rows {
            if let Ok((id, cat, blob)) = row {
                let emb = bytes_to_f32(&blob);
                if !emb.is_empty() {
                    index.insert(&id.to_lowercase(), &cat, &emb);
                }
            }
        }
    }
    eprintln!("  vector index: {} memories pre-indexed", index.len());

    let state = Arc::new(AppState {
        db: Mutex::new(conn),
        index: Mutex::new(index),
        blob_dir: blobs.clone(),
        mode: mode.clone(),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/brain/info", get(brain_info))
        .route("/brain/store_mode", get(store_mode_handler))
        .route("/brain/index_stats", get(index_stats))
        .route("/brain/search", post(brain_search))
        .route("/brain/checkpoint", post(brain_checkpoint))
        .route("/brain/workload", get(brain_workload))
        .route(
            "/brain/export-pairs",
            get(brain_export_pairs_get).post(brain_export_pairs),
        )
        .route("/brain/training-stats", get(brain_training_stats))
        .route("/memories", get(list_memories))
        .route("/memories", post(create_memory))
        .route("/memories/:id", get(get_memory))
        .route("/security/scan", post(security_scan))
        .route("/security/status", get(security_status))
        .route("/preference_pairs", get(list_preference_pairs))
        .route("/preference_pairs", post(create_preference_pair))
        .route("/learning/stats", get(learning_stats))
        .with_state(state);

    let port: u16 = std::env::var("RUVBRAIN_PORT")
        .unwrap_or_else(|_| "9876".to_string())
        .parse()
        .unwrap_or(9876);
    let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{port}")).await?;
    eprintln!("mcp-brain-server-local listening on http://127.0.0.1:{port}");
    eprintln!("  mode:  {mode}");
    eprintln!("  db:    {path}");
    eprintln!("  blobs: {blobs}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = ctrl_c => {}, _ = terminate => {} }
}

// ── Schema ───────────────────────────────────────────────────────────────────

fn ensure_schema(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS memories (
            id BLOB PRIMARY KEY,
            category TEXT,
            content_hash TEXT,
            created_at INTEGER,
            embedding BLOB,
            quality REAL DEFAULT 0.5
        );
        CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
        CREATE INDEX IF NOT EXISTS idx_memories_created  ON memories(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memories_hash     ON memories(content_hash);

        CREATE TABLE IF NOT EXISTS preference_pairs (
            id BLOB PRIMARY KEY,
            chosen BLOB,
            rejected BLOB,
            direction TEXT,
            created_at INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_pairs_direction ON preference_pairs(direction);
        CREATE INDEX IF NOT EXISTS idx_pairs_created   ON preference_pairs(created_at DESC);

        CREATE TABLE IF NOT EXISTS votes (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            voter TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_votes_memory ON votes(memory_id);

        CREATE TABLE IF NOT EXISTS pages (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft',
            author TEXT NOT NULL,
            quality REAL NOT NULL DEFAULT 0.5,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS page_deltas (
            id TEXT PRIMARY KEY,
            page_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            evidence_type TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_deltas_page ON page_deltas(page_id);

        CREATE TABLE IF NOT EXISTS page_evidence (
            id TEXT PRIMARY KEY,
            page_id TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            description TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_evidence_page ON page_evidence(page_id);

        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            publisher TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            wasm_hash TEXT,
            wasm_size INTEGER DEFAULT 0,
            conformance TEXT,
            revoked INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS adapters (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            base_model TEXT,
            metrics TEXT,
            path TEXT,
            created_at INTEGER NOT NULL
        );",
    )
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn bytes_to_f32(blob: &[u8]) -> Vec<f32> {
    if blob.len() % 4 != 0 {
        return Vec::new();
    }
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn new_id() -> Vec<u8> {
    uuid::Uuid::new_v4().as_bytes().to_vec()
}

fn id_hex(blob: &[u8]) -> String {
    hex::encode(blob)
}

fn hex_to_id(s: &str) -> Option<Vec<u8>> {
    hex::decode(s).ok().filter(|v| v.len() == 16)
}

fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn content_hash(data: &str) -> String {
    use sha2::{Digest, Sha256};
    let h = Sha256::digest(data.as_bytes());
    hex::encode(h)
}

/// Write content to blob store: {blob_dir}/{hash[0:2]}/{hash[2:]}
fn blob_write(blob_dir: &str, hash: &str, content: &str) {
    if hash.len() < 4 {
        return;
    }
    let dir = format!("{}/{}", blob_dir, &hash[..2]);
    let _ = std::fs::create_dir_all(&dir);
    let path = format!("{}/{}", dir, &hash[2..]);
    let _ = std::fs::write(path, content);
}

/// Read content from blob store by content_hash
fn blob_read(blob_dir: &str, hash: &str) -> Option<String> {
    if hash.len() < 4 {
        return None;
    }
    let path = format!("{}/{}/{}", blob_dir, &hash[..2], &hash[2..]);
    std::fs::read_to_string(path).ok()
}

// ── Handlers ─────────────────────────────────────────────────────────────────

async fn health(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let backend = if st.mode == "rvf" { "sqlite" } else { &st.mode };
    Json(serde_json::json!({
        "status": "ok",
        "version": VERSION,
        "backend": backend,
        "mode": "local"
    }))
}

async fn brain_info(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let db = st.db.lock().unwrap();
    let mem_count: i64 = db
        .query_row("SELECT count(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);
    let pair_count: i64 = db
        .query_row("SELECT count(*) FROM preference_pairs", [], |r| r.get(0))
        .unwrap_or(0);
    Json(serde_json::json!({
        "version": VERSION,
        "db_path": db_path(),
        "blob_dir": blob_dir(),
        "memories_count": mem_count,
        "preference_pairs_count": pair_count,
    }))
}

async fn store_mode_handler(State(st): State<SharedState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "mode": st.mode,
        "version": VERSION,
    }))
}

async fn index_stats(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let idx = st.index.lock().unwrap();
    let mode = if idx.len() < 2000 {
        "brute_force"
    } else {
        "vamana_graph"
    };
    let graph_edges: usize = idx.neighbors.iter().map(|n| n.len()).sum();
    let avg_degree = if idx.len() > 0 {
        graph_edges as f64 / idx.len() as f64
    } else {
        0.0
    };
    let vec_ram_mb = (idx.len() * idx.dim * 4) as f64 / (1024.0 * 1024.0);
    let graph_ram_kb = (graph_edges * 4) as f64 / 1024.0;
    Json(serde_json::json!({
        "engine": "diskann_vamana",
        "mode": mode,
        "indexed_count": idx.len(),
        "dim": idx.dim,
        "max_degree": idx.max_degree,
        "avg_degree": format!("{:.1}", avg_degree),
        "graph_edges": graph_edges,
        "medoid": idx.medoid,
        "vector_ram_mb": format!("{:.2}", vec_ram_mb),
        "graph_ram_kb": format!("{:.1}", graph_ram_kb),
        "ssd_backed": idx.ssd_path.is_some(),
    }))
}

#[derive(Deserialize)]
struct SearchRequest {
    query: Option<String>,
    query_vector: Option<Vec<f32>>,
    k: Option<usize>,
}

async fn brain_search(
    State(st): State<SharedState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let k = req.k.unwrap_or(5).min(100);

    // Use provided vector, or embed the query text via the local embedder
    let query_vec = if let Some(ref v) = req.query_vector {
        v.clone()
    } else if let Some(ref q) = req.query {
        // Call local embedder service
        match embed_text(q).await {
            Ok(v) => v,
            Err(e) => {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({
                        "error": format!("embedder unavailable: {e}")
                    })),
                ));
            }
        }
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "missing 'query' or 'query_vector'"
            })),
        ));
    };

    // Request extra results from index to account for DB misses and dedup
    let fetch_k = (k * 3).max(10);
    let results = {
        let idx = st.index.lock().unwrap();
        idx.search(&query_vec, fetch_k)
    };

    // Fetch metadata for each result, deduplicate by content_hash
    let db = st.db.lock().unwrap();
    let mut items = Vec::new();
    let mut seen_hashes = std::collections::HashSet::new();
    for (score, id_hex) in &results {
        if let Ok(id_blob) = hex::decode(id_hex) {
            let row = db.query_row(
                "SELECT category, content_hash, created_at FROM memories WHERE id = ?1",
                [&id_blob],
                |r| {
                    Ok((
                        r.get::<_, String>(0).unwrap_or_default(),
                        r.get::<_, String>(1).unwrap_or_default(),
                        r.get::<_, i64>(2).unwrap_or(0),
                    ))
                },
            );
            if let Ok((category, hash, created_at)) = row {
                // Skip duplicates with same content
                if !seen_hashes.insert(hash.clone()) {
                    continue;
                }
                let content = blob_read(&st.blob_dir, &hash);
                let mut item = serde_json::json!({
                    "id": id_hex,
                    "score": score,
                    "category": category,
                    "content_hash": hash,
                    "created_at": created_at,
                });
                if let Some(c) = content {
                    item["content"] = serde_json::Value::String(c);
                }
                items.push(item);
            }
        }
    }

    // Truncate to requested k after dedup
    items.truncate(k);

    Ok(Json(serde_json::json!({
        "query_vector_dim": query_vec.len(),
        "results": items,
    })))
}

async fn embed_text(text: &str) -> Result<Vec<f32>, String> {
    let url = std::env::var("RUVBRAIN_EMBEDDER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:9877".to_string());

    let body = serde_json::json!({ "texts": [text] });
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .post(format!("{url}/embed"))
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    let vectors = data
        .get("vectors")
        .or_else(|| data.get("embeddings"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_array())
        .ok_or_else(|| "unexpected embedder response".to_string())?;

    Ok(vectors
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect())
}

async fn brain_checkpoint(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let db = st.db.lock().unwrap();
    let result = db.execute_batch("PRAGMA wal_checkpoint(PASSIVE);");
    Json(serde_json::json!({
        "ok": result.is_ok(),
        "error": result.err().map(|e| e.to_string()),
    }))
}

// ── Item 1: Auto-profile workload endpoint (ADR-SYS-0007) ───────────────────

#[derive(Serialize)]
struct WorkloadResponse {
    gpu_util: f64,
    cpu_load: f64,
    recommended_profile: String,
    reason: String,
}

async fn brain_workload() -> Json<WorkloadResponse> {
    let gpu_util = read_gpu_util().await;
    let (cpu_load, num_cores) = read_cpu_load();
    let hour = chrono::Local::now().hour();

    let (profile, reason) = decide_profile(gpu_util, cpu_load, num_cores, hour);

    Json(WorkloadResponse {
        gpu_util,
        cpu_load,
        recommended_profile: profile,
        reason,
    })
}

async fn read_gpu_util() -> f64 {
    // nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .trim()
            .parse::<f64>()
            .unwrap_or(0.0),
        _ => 0.0,
    }
}

fn read_cpu_load() -> (f64, usize) {
    let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let load_1m: f64 = loadavg
        .split_whitespace()
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let cores = num_cpus();
    (load_1m, cores)
}

fn num_cpus() -> usize {
    std::fs::read_to_string("/proc/cpuinfo")
        .map(|s| s.matches("processor\t:").count())
        .unwrap_or(1)
        .max(1)
}

use chrono::Timelike;

fn decide_profile(gpu_util: f64, cpu_load: f64, num_cores: usize, hour: u32) -> (String, String) {
    let cpu_threshold = num_cores as f64 * 0.8;
    let gpu_idle = gpu_util < 5.0;
    let after_hours = hour >= 22 || hour < 6;

    if gpu_util > 80.0 {
        (
            "gpu-train".into(),
            format!("GPU util {gpu_util:.0}% > 80% -- sustained training workload"),
        )
    } else if gpu_util >= 30.0 && gpu_util <= 80.0 {
        (
            "gpu-infer".into(),
            format!("GPU util {gpu_util:.0}% in 30-80% range -- inference/MPS beneficial"),
        )
    } else if cpu_load > cpu_threshold && gpu_idle {
        ("cpu-bulk".into(), format!(
            "CPU load {cpu_load:.1} > {cpu_threshold:.0} threshold, GPU idle -- CPU-bound batch work"
        ))
    } else if after_hours && gpu_idle && cpu_load < cpu_threshold * 0.3 {
        (
            "power-save".into(),
            format!(
                "After hours (hour={hour}), GPU idle, CPU load {cpu_load:.1} low -- power save"
            ),
        )
    } else if gpu_idle && cpu_load < cpu_threshold * 0.5 {
        (
            "interactive".into(),
            format!("Low utilization (GPU {gpu_util:.0}%, CPU {cpu_load:.1}) -- interactive mode"),
        )
    } else {
        (
            "default".into(),
            format!(
                "GPU {gpu_util:.0}%, CPU load {cpu_load:.1} -- no strong signal, using default"
            ),
        )
    }
}

// ── Item 2: DPO training data export (ADR-SYS-0004) ─────────────────────────

#[derive(Deserialize)]
struct ExportPairsQuery {
    format: Option<String>,
    limit: Option<usize>,
}

#[derive(Serialize)]
struct DpoRecord {
    chosen_text: String,
    rejected_text: String,
    chosen_embedding: Vec<f32>,
    rejected_embedding: Vec<f32>,
    direction: String,
    quality_delta: f64,
}

async fn brain_export_pairs_get(
    state: State<SharedState>,
    query: Query<ExportPairsQuery>,
) -> impl IntoResponse {
    brain_export_pairs_inner(state, query).await
}

async fn brain_export_pairs(
    state: State<SharedState>,
    query: Query<ExportPairsQuery>,
) -> impl IntoResponse {
    brain_export_pairs_inner(state, query).await
}

async fn brain_export_pairs_inner(
    State(st): State<SharedState>,
    Query(q): Query<ExportPairsQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(1000).min(10000);
    let jsonl = q.format.as_deref() == Some("jsonl");

    let db = st.db.lock().unwrap();
    let mut stmt = match db.prepare(
        "SELECT p.chosen, p.rejected, p.direction,
                mc.embedding, mc.quality, mc.content_hash,
                mr.embedding, mr.quality, mr.content_hash
         FROM preference_pairs p
         LEFT JOIN memories mc ON mc.id = p.chosen
         LEFT JOIN memories mr ON mr.id = p.rejected
         ORDER BY p.created_at DESC
         LIMIT ?1",
    ) {
        Ok(s) => s,
        Err(e) => {
            let body = serde_json::json!({"error": e.to_string()});
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                [("content-type", "application/json")],
                serde_json::to_string(&body).unwrap(),
            );
        }
    };

    let rows = stmt.query_map([limit as i64], |row| {
        let chosen_emb_blob: Vec<u8> = row.get::<_, Vec<u8>>(3).unwrap_or_default();
        let chosen_quality: f64 = row.get::<_, f64>(4).unwrap_or(0.5);
        let chosen_hash: String = row.get::<_, String>(5).unwrap_or_default();
        let rejected_emb_blob: Vec<u8> = row.get::<_, Vec<u8>>(6).unwrap_or_default();
        let rejected_quality: f64 = row.get::<_, f64>(7).unwrap_or(0.5);
        let rejected_hash: String = row.get::<_, String>(8).unwrap_or_default();
        let direction: String = row.get::<_, String>(2).unwrap_or_default();

        Ok(DpoRecord {
            chosen_text: chosen_hash,
            rejected_text: rejected_hash,
            chosen_embedding: bytes_to_f32(&chosen_emb_blob),
            rejected_embedding: bytes_to_f32(&rejected_emb_blob),
            direction,
            quality_delta: chosen_quality - rejected_quality,
        })
    });

    let records: Vec<DpoRecord> = match rows {
        Ok(r) => r.filter_map(|r| r.ok()).collect(),
        Err(e) => {
            let body = serde_json::json!({"error": e.to_string()});
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                [("content-type", "application/json")],
                serde_json::to_string(&body).unwrap(),
            );
        }
    };

    if jsonl {
        let mut out = String::new();
        for rec in &records {
            if let Ok(line) = serde_json::to_string(rec) {
                out.push_str(&line);
                out.push('\n');
            }
        }
        (
            StatusCode::OK,
            [("content-type", "application/x-ndjson")],
            out,
        )
    } else {
        let body = serde_json::to_string(&records).unwrap_or_else(|_| "[]".to_string());
        (StatusCode::OK, [("content-type", "application/json")], body)
    }
}

#[derive(Serialize)]
struct TrainingStats {
    total_pairs: i64,
    pairs_with_embeddings: i64,
    exportable: i64,
    mean_quality_delta: f64,
}

async fn brain_training_stats(State(st): State<SharedState>) -> Json<TrainingStats> {
    let db = st.db.lock().unwrap();

    let total: i64 = db
        .query_row("SELECT count(*) FROM preference_pairs", [], |r| r.get(0))
        .unwrap_or(0);

    let with_emb: i64 = db
        .query_row(
            "SELECT count(*) FROM preference_pairs p
         JOIN memories mc ON mc.id = p.chosen AND length(mc.embedding) > 0
         JOIN memories mr ON mr.id = p.rejected AND length(mr.embedding) > 0",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);

    let (exportable, mean_delta) = db
        .query_row(
            "SELECT count(*), coalesce(avg(mc.quality - mr.quality), 0.0)
         FROM preference_pairs p
         JOIN memories mc ON mc.id = p.chosen AND length(mc.embedding) > 0
         JOIN memories mr ON mr.id = p.rejected AND length(mr.embedding) > 0",
            [],
            |r| {
                Ok((
                    r.get::<_, i64>(0).unwrap_or(0),
                    r.get::<_, f64>(1).unwrap_or(0.0),
                ))
            },
        )
        .unwrap_or((0, 0.0));

    Json(TrainingStats {
        total_pairs: total,
        pairs_with_embeddings: with_emb,
        exportable,
        mean_quality_delta: (mean_delta * 1000.0).round() / 1000.0,
    })
}

// ── Existing endpoints ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ListQuery {
    limit: Option<usize>,
    offset: Option<usize>,
    category: Option<String>,
}

async fn list_memories(
    State(st): State<SharedState>,
    Query(q): Query<ListQuery>,
) -> Json<serde_json::Value> {
    let limit = q.limit.unwrap_or(20).min(500);
    let offset = q.offset.unwrap_or(0);
    let db = st.db.lock().unwrap();

    // Get total count for this query
    let total: i64 = match &q.category {
        Some(cat) => db
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE category = ?1",
                [cat],
                |r| r.get(0),
            )
            .unwrap_or(0),
        None => db
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap_or(0),
    };

    let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match &q.category {
        Some(cat) => (
            "SELECT hex(id), category, content_hash, created_at FROM memories WHERE category = ?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3".into(),
            vec![
                Box::new(cat.clone()) as Box<dyn rusqlite::types::ToSql>,
                Box::new(limit as i64),
                Box::new(offset as i64),
            ],
        ),
        None => (
            "SELECT hex(id), category, content_hash, created_at FROM memories ORDER BY created_at DESC LIMIT ?1 OFFSET ?2".into(),
            vec![
                Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>,
                Box::new(offset as i64),
            ],
        ),
    };

    let row_data: Vec<(String, String, String, i64)> = {
        let mut stmt = db.prepare(&sql).unwrap();
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params.iter().map(|p| p.as_ref()).collect();
        stmt.query_map(params_refs.as_slice(), |row| {
            Ok((
                row.get::<_, String>(0).unwrap_or_default().to_lowercase(),
                row.get::<_, String>(1).unwrap_or_default(),
                row.get::<_, String>(2).unwrap_or_default(),
                row.get::<_, i64>(3).unwrap_or(0),
            ))
        })
        .unwrap()
        .filter_map(|r| r.ok())
        .collect()
    };
    drop(db);

    let memories: Vec<serde_json::Value> = row_data
        .iter()
        .map(|(id, cat, hash, ts)| {
            let mut obj = serde_json::json!({
                "id": id,
                "category": cat,
                "content_hash": hash,
                "created_at": ts,
            });
            if let Some(c) = blob_read(&st.blob_dir, hash) {
                obj["content"] = serde_json::Value::String(c);
            }
            obj
        })
        .collect();

    Json(serde_json::json!({
        "count": memories.len(),
        "total": total,
        "offset": offset,
        "memories": memories,
    }))
}

#[derive(Deserialize)]
struct CreateMemoryRequest {
    category: String,
    content: String,
    #[serde(default)]
    embedding: Vec<f32>,
}

async fn create_memory(
    State(st): State<SharedState>,
    Json(req): Json<CreateMemoryRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    // AIDefence: scan content before storing
    let (safe, threat_level, threats) = aidefence_scan(&req.content);
    if !safe {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(serde_json::json!({
                "error": "content blocked by AIDefence",
                "threat_level": threat_level,
                "threats": threats,
            })),
        );
    }

    let id = new_id();
    let id_hex = id_hex(&id);
    let hash = content_hash(&req.content);
    let now = now_epoch();

    let emb_blob = if req.embedding.is_empty() {
        // Try to get embedding from local embedder
        match embed_text(&req.content).await {
            Ok(v) => f32_to_bytes(&v),
            Err(_) => Vec::new(),
        }
    } else {
        f32_to_bytes(&req.embedding)
    };

    // Write content to blob store
    blob_write(&st.blob_dir, &hash, &req.content);

    let db = st.db.lock().unwrap();
    let result = db.execute(
        "INSERT INTO memories (id, category, content_hash, created_at, embedding, quality)
         VALUES (?1, ?2, ?3, ?4, ?5, 0.5)",
        rusqlite::params![id, req.category, hash, now, emb_blob],
    );

    match result {
        Ok(_) => {
            // Add to index if we have an embedding
            if !emb_blob.is_empty() {
                let emb = bytes_to_f32(&emb_blob);
                let mut idx = st.index.lock().unwrap();
                idx.insert(&id_hex, &req.category, &emb);
            }
            (
                StatusCode::CREATED,
                Json(serde_json::json!({
                    "id": id_hex,
                    "content_hash": hash,
                    "created_at": now,
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string(),
            })),
        ),
    }
}

async fn get_memory(
    State(st): State<SharedState>,
    Path(id_str): Path<String>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let id_blob = hex_to_id(&id_str).ok_or(StatusCode::BAD_REQUEST)?;
    let db = st.db.lock().unwrap();
    let row_data = db.query_row(
        "SELECT hex(id), category, content_hash, created_at, quality FROM memories WHERE id = ?1",
        [&id_blob],
        |row| {
            Ok((
                row.get::<_, String>(0).unwrap_or_default().to_lowercase(),
                row.get::<_, String>(1).unwrap_or_default(),
                row.get::<_, String>(2).unwrap_or_default(),
                row.get::<_, i64>(3).unwrap_or(0),
                row.get::<_, f64>(4).unwrap_or(0.5),
            ))
        },
    ).map_err(|_| StatusCode::NOT_FOUND)?;
    drop(db);

    let content = blob_read(&st.blob_dir, &row_data.2);
    let mut obj = serde_json::json!({
        "id": row_data.0,
        "category": row_data.1,
        "content_hash": row_data.2,
        "created_at": row_data.3,
        "quality": row_data.4,
    });
    if let Some(c) = content {
        obj["content"] = serde_json::Value::String(c);
    }
    Ok(Json(obj))
}

#[derive(Deserialize)]
struct PairListQuery {
    limit: Option<usize>,
    direction: Option<String>,
}

async fn list_preference_pairs(
    State(st): State<SharedState>,
    Query(q): Query<PairListQuery>,
) -> Json<serde_json::Value> {
    let limit = q.limit.unwrap_or(20).min(500);
    let db = st.db.lock().unwrap();

    let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = match &q.direction {
        Some(dir) => (
            "SELECT hex(id), hex(chosen), hex(rejected), direction, created_at
             FROM preference_pairs WHERE direction = ?1
             ORDER BY created_at DESC LIMIT ?2"
                .into(),
            vec![
                Box::new(dir.clone()) as Box<dyn rusqlite::types::ToSql>,
                Box::new(limit as i64),
            ],
        ),
        None => (
            "SELECT hex(id), hex(chosen), hex(rejected), direction, created_at
             FROM preference_pairs
             ORDER BY created_at DESC LIMIT ?1"
                .into(),
            vec![Box::new(limit as i64) as Box<dyn rusqlite::types::ToSql>],
        ),
    };

    let mut stmt = db.prepare(&sql).unwrap();
    let params_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
    let rows = stmt
        .query_map(params_refs.as_slice(), |row| {
            Ok(serde_json::json!({
                "id": row.get::<_, String>(0).unwrap_or_default().to_lowercase(),
                "chosen_id": row.get::<_, String>(1).unwrap_or_default().to_lowercase(),
                "rejected_id": row.get::<_, String>(2).unwrap_or_default().to_lowercase(),
                "direction": row.get::<_, String>(3).unwrap_or_default(),
                "created_at": row.get::<_, i64>(4).unwrap_or(0),
            }))
        })
        .unwrap();

    let pairs: Vec<serde_json::Value> = rows.filter_map(|r| r.ok()).collect();
    Json(serde_json::json!({
        "count": pairs.len(),
        "pairs": pairs,
    }))
}

#[derive(Deserialize)]
struct CreatePairRequest {
    chosen_id: String,
    rejected_id: String,
    direction: String,
}

async fn create_preference_pair(
    State(st): State<SharedState>,
    Json(req): Json<CreatePairRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let chosen = match hex_to_id(&req.chosen_id) {
        Some(v) => v,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid chosen_id"})),
            )
        }
    };
    let rejected = match hex_to_id(&req.rejected_id) {
        Some(v) => v,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid rejected_id"})),
            )
        }
    };

    let id = new_id();
    let id_hex = id_hex(&id);
    let now = now_epoch();

    let db = st.db.lock().unwrap();
    let result = db.execute(
        "INSERT INTO preference_pairs (id, chosen, rejected, direction, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
        rusqlite::params![id, chosen, rejected, req.direction, now],
    );

    match result {
        Ok(_) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "id": id_hex,
                "created_at": now,
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": e.to_string(),
            })),
        ),
    }
}

// ── Security Endpoints ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ScanRequest {
    text: String,
}

async fn security_scan(Json(req): Json<ScanRequest>) -> Json<serde_json::Value> {
    let (safe, threat_level, threats) = aidefence_scan(&req.text);
    Json(serde_json::json!({
        "safe": safe,
        "threat_level": threat_level,
        "threats": threats,
        "patterns_loaded": THREAT_PATTERNS.len() + PII_PATTERNS.len(),
    }))
}

async fn security_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "guard": "active",
        "engine": "aidefence-rust-inline",
        "injection_patterns": THREAT_PATTERNS.len(),
        "pii_patterns": PII_PATTERNS.len(),
        "block_threshold": "medium",
        "protects": ["POST /memories", "POST /security/scan"],
    }))
}

// ── Learning Stats ──────────────────────────────────────────────────────────

async fn learning_stats(State(st): State<SharedState>) -> Json<serde_json::Value> {
    let db = st.db.lock().unwrap();

    let memories: i64 = db
        .query_row("SELECT count(*) FROM memories", [], |r| r.get(0))
        .unwrap_or(0);
    let pairs: i64 = db
        .query_row("SELECT count(*) FROM preference_pairs", [], |r| r.get(0))
        .unwrap_or(0);
    let votes: i64 = db
        .query_row("SELECT count(*) FROM votes", [], |r| r.get(0))
        .unwrap_or(0);
    let pages: i64 = db
        .query_row("SELECT count(*) FROM pages", [], |r| r.get(0))
        .unwrap_or(0);
    let nodes: i64 = db
        .query_row("SELECT count(*) FROM nodes", [], |r| r.get(0))
        .unwrap_or(0);
    let adapters: i64 = db
        .query_row("SELECT count(*) FROM adapters", [], |r| r.get(0))
        .unwrap_or(0);

    // Blob stats
    let blob_count = std::fs::read_dir(&st.blob_dir)
        .map(|d| d.count())
        .unwrap_or(0);
    let blob_bytes: u64 = std::fs::read_dir(&st.blob_dir)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .filter_map(|e| e.metadata().ok())
                .map(|m| m.len())
                .sum()
        })
        .unwrap_or(0);

    // RVF file stats
    let rvf_path = db_path().replace(".sqlite", ".rvf");
    let rvf_bytes = std::fs::metadata(&rvf_path).map(|m| m.len()).unwrap_or(0);
    // Rough segment count estimate from file size (each segment ~550 bytes avg)
    let rvf_segments = if rvf_bytes > 0 { rvf_bytes / 570 } else { 0 };

    // Schema version heuristic
    let schema_version = 3i64;

    Json(serde_json::json!({
        "memories_count": memories,
        "preference_pairs_count": pairs,
        "votes_count": votes,
        "pages_count": pages,
        "nodes_count": nodes,
        "adapters_count": adapters,
        "blob_count": blob_count,
        "blob_total_bytes": blob_bytes,
        "rvf_file_bytes": rvf_bytes,
        "rvf_segment_count": rvf_segments,
        "schema_version": schema_version,
        "store_mode": st.mode,
    }))
}
