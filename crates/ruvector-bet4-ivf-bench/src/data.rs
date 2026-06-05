//! Loader for the aligned ogbn-arxiv 128-d node-feature CSV (row `i` = node `i`), the same
//! public corpus used by ADR-201/202/204. Data lives under `target/m1-data/` (gitignored).

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Load up to `limit` rows of comma-separated f32 features. Blank lines are skipped. Each
/// returned row is one node's feature vector (all rows share the file's column count, 128 for
/// the arxiv features).
pub fn load_feat_csv<P: AsRef<Path>>(path: P, limit: usize) -> std::io::Result<Vec<Vec<f32>>> {
    let reader = BufReader::new(File::open(path)?);
    let mut out = Vec::with_capacity(limit);
    for line in reader.lines() {
        if out.len() >= limit {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Vec<f32> = line
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap_or(0.0))
            .collect();
        out.push(row);
    }
    Ok(out)
}
