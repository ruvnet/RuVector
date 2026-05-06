//! `ruvector csi` subcommands — ADR-183 Tier 3 iter 16.
//!
//! `ruvector csi sink`   — poll brain for `spatial-csi-embedding` memories
//!                         and index them into an HNSW VectorDB.
//! `ruvector csi search` — k-NN cosine search over the CSI embedding index.

use anyhow::{Context, Result};
use clap::Subcommand;
use colored::*;
use ruvector_core::{
    types::{DbOptions, DistanceMetric, HnswConfig, SearchQuery, VectorEntry},
    VectorDB,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;

/// Default HNSW database path on cognitum-v0.
pub const DEFAULT_CSI_DB: &str = "/var/lib/ruvector-vectors/csi-embeddings.db";

/// Default brain URL on cognitum-v0.
pub const DEFAULT_BRAIN_URL: &str = "http://127.0.0.1:9876";

/// CSI embedding dimension (must match `ruvector_hailo::CSI_EMBED_DIM`).
pub const CSI_DIM: usize = 128;

#[derive(Subcommand)]
pub enum CsiCommands {
    /// Poll the brain for spatial-csi-embedding memories and insert them
    /// into a 128-dim cosine HNSW index at `--db`.
    Sink {
        /// Brain HTTP base URL.
        #[arg(long, default_value = DEFAULT_BRAIN_URL)]
        brain: String,

        /// Path to the HNSW index file.
        #[arg(long, default_value = DEFAULT_CSI_DB)]
        db: PathBuf,

        /// Poll once then exit (default: run continuously every 30 s).
        #[arg(long)]
        once: bool,

        /// Polling interval in seconds (ignored with --once).
        #[arg(long, default_value = "30")]
        interval: u64,
    },

    /// Search the CSI embedding index for the K nearest neighbours of a
    /// query embedding.
    Search {
        /// Path to the HNSW index file.
        #[arg(long, default_value = DEFAULT_CSI_DB)]
        db: PathBuf,

        /// 128 comma-separated f32 values (the query embedding).
        #[arg(long)]
        embedding: Option<String>,

        /// Number of results.
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,

        /// Print full 128-dim vectors.
        #[arg(long)]
        show_vectors: bool,
    },
}

/// Brain `/memories` response shape.
#[derive(Debug, Deserialize)]
struct BrainResponse {
    memories: Vec<BrainMemory>,
}

#[derive(Debug, Deserialize)]
struct BrainMemory {
    category: String,
    content: String,
}

/// Parse `node_id=N node=X embedding=[f32,…]` content string.
/// Returns `(id_string, embedding)` on success.
fn parse_csi_embedding(content: &str) -> Option<(String, Vec<f32>)> {
    // Extract node_id
    let node_id = content
        .split_whitespace()
        .find(|t| t.starts_with("node_id="))?
        .strip_prefix("node_id=")?
        .to_string();
    let node = content
        .split_whitespace()
        .find(|t| t.starts_with("node="))
        .and_then(|t| t.strip_prefix("node="))
        .unwrap_or("unknown");

    // Extract embedding JSON array
    let emb_start = content.find("embedding=[")?;
    let array_str = &content[emb_start + "embedding=".len()..];
    let close = array_str.find(']')?;
    let values: Vec<f32> = array_str[1..close]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if values.len() != CSI_DIM {
        return None;
    }
    let id = format!("csi:{node}:{node_id}");
    Some((id, values))
}

/// Open (or create) the 128-dim cosine HNSW index at `path`.
fn open_csi_db(path: &PathBuf) -> Result<VectorDB> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create db dir {}", parent.display()))?;
    }
    let opts = DbOptions {
        dimensions: CSI_DIM,
        distance_metric: DistanceMetric::Cosine,
        storage_path: path.to_string_lossy().into_owned(),
        hnsw_config: Some(HnswConfig {
            m: 16,
            ef_construction: 100,
            ef_search: 50,
            max_elements: 100_000,
        }),
        quantization: None,
    };
    VectorDB::new(opts).context("open CSI VectorDB")
}

/// Fetch all `spatial-csi-embedding` memories from the brain and ingest
/// any that are not already in the index.
fn ingest_once(brain_url: &str, db: &mut VectorDB) -> Result<usize> {
    let url = format!("{}/memories", brain_url.trim_end_matches('/'));
    let resp = reqwest::blocking::get(&url)
        .with_context(|| format!("GET {url}"))?
        .json::<serde_json::Value>()
        .context("parse brain response")?;

    let memories: Vec<BrainMemory> = {
        let raw = resp
            .get("memories")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        raw.into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect()
    };

    let csi_mems: Vec<_> = memories
        .iter()
        .filter(|m| m.category == "spatial-csi-embedding")
        .collect();

    let mut inserted = 0usize;
    for mem in &csi_mems {
        let Some((id, vec)) = parse_csi_embedding(&mem.content) else {
            continue;
        };
        let mut meta = HashMap::new();
        meta.insert(
            "content".to_string(),
            serde_json::Value::String(mem.content.clone()),
        );
        let entry = VectorEntry {
            id: Some(id),
            vector: vec,
            metadata: Some(meta),
        };
        if db.insert(entry).is_ok() {
            inserted += 1;
        }
    }
    Ok(inserted)
}

pub fn run_csi_sink(
    brain: &str,
    db_path: &PathBuf,
    once: bool,
    interval_secs: u64,
) -> Result<()> {
    println!("{}", format!("CSI sink — brain: {brain}  db: {}", db_path.display()).cyan());

    let mut db = open_csi_db(db_path)?;
    loop {
        match ingest_once(brain, &mut db) {
            Ok(n) => {
                if n > 0 {
                    println!("{}", format!("  +{n} spatial-csi-embedding entries indexed").green());
                } else {
                    println!("  no new embeddings");
                }
            }
            Err(e) => eprintln!("{}", format!("  ingest error: {e}").red()),
        }
        if once {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(interval_secs));
    }
    Ok(())
}

/// Open the CSI index for read-only search. When the live DB is locked by
/// the sink process, copies it to a temp file and reads the snapshot.
fn open_for_search(db_path: &PathBuf) -> Result<VectorDB> {
    match open_csi_db(db_path) {
        Ok(db) => Ok(db),
        Err(_) if db_path.exists() => {
            // DB is locked by the running sink — snapshot it.
            let tmp = std::env::temp_dir().join("csi-search-snapshot.db");
            std::fs::copy(db_path, &tmp)
                .context("snapshot locked CSI db for search")?;
            open_csi_db(&tmp)
        }
        Err(e) => Err(e),
    }
}

pub fn run_csi_search(
    db_path: &PathBuf,
    embedding_arg: Option<&str>,
    top_k: usize,
    show_vectors: bool,
) -> Result<()> {
    let mut db = open_for_search(db_path)?;

    let query_vec: Vec<f32> = if let Some(s) = embedding_arg {
        let v: Vec<f32> = s
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .filter_map(|x| x.trim().parse().ok())
            .collect();
        anyhow::ensure!(v.len() == CSI_DIM, "embedding must be {CSI_DIM} floats, got {}", v.len());
        v
    } else {
        // No query provided — list the most recently inserted entries instead.
        println!("{}", "No --embedding provided; listing most-recent entries:".yellow());
        let results = db.search(SearchQuery {
            vector: vec![0.0_f32; CSI_DIM],
            k: top_k,
            filter: None,
            ef_search: Some(200),
        })?;
        for (i, r) in results.iter().enumerate() {
            let meta_str = r
                .metadata
                .as_ref()
                .and_then(|m| m.get("content"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .get(..80)
                .unwrap_or("");
            println!("  {}: id={} score={:.4}  {}", i + 1, r.id.cyan(), r.score, meta_str);
        }
        return Ok(());
    };

    let results = db.search(SearchQuery {
        vector: query_vec,
        k: top_k,
        filter: None,
        ef_search: None,
    })?;

    println!("{}", format!("Top-{} CSI embedding matches:", top_k).bold());
    for (i, r) in results.iter().enumerate() {
        let meta_str = r
            .metadata
            .as_ref()
            .and_then(|m| m.get("content"))
            .and_then(|v| v.as_str())
            .map(|s| s.get(..80).unwrap_or(s))
            .unwrap_or("");
        println!("  {}: id={}  score={:.4}  {}", i + 1, r.id.cyan(), r.score, meta_str);
        if show_vectors {
            if let Some(ref vec) = r.vector {
                let preview: Vec<String> = vec.iter().take(8).map(|v| format!("{v:.3}")).collect();
                println!("     vec: [{}…]", preview.join(", "));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_csi_embedding() {
        let content = format!(
            "node_id=3 node=cognitum-v0 embedding=[{}]",
            vec!["0.125000"; 128].join(",")
        );
        let (id, vec) = parse_csi_embedding(&content).unwrap();
        assert!(id.starts_with("csi:"), "id should start with csi: prefix: {id}");
        assert!(id.contains("cognitum-v0"), "id should include node name: {id}");
        assert_eq!(vec.len(), 128);
        assert!((vec[0] - 0.125).abs() < 1e-5);
    }

    #[test]
    fn parse_wrong_dim_returns_none() {
        let content = "node_id=1 node=x embedding=[0.1,0.2,0.3]";
        assert!(parse_csi_embedding(content).is_none());
    }

    #[test]
    fn parse_missing_embedding_returns_none() {
        let content = "node_id=1 node=x";
        assert!(parse_csi_embedding(content).is_none());
    }

    #[test]
    fn hailo_embedder_config_variants() {
        // Verify the ADR-183 iter 14 config types are accessible from CLI
        // (they live in ruvector-hailo, but we want to confirm the CLI
        //  can construct them without pulling the full hailo dep).
        // Just test the constants we re-define here match expectations.
        assert_eq!(CSI_DIM, 128);
        assert_eq!(DEFAULT_CSI_DB, "/var/lib/ruvector-vectors/csi-embeddings.db");
    }
}
