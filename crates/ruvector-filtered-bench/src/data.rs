//! M0 — load aligned ogbn-arxiv features / labels / years.
//!
//! Row `i` of every file is node `i` (ogbn-arxiv node-index order), so the three
//! arrays align by position. Features are pre-extracted to plain CSV (128 comma-
//! separated f32 per line); labels/years are the gunzipped single-column files.
//!
//! One-time extraction (already done in `target/m1-data/`):
//! ```text
//! gunzip -kc target/m1-data/arxiv/raw/node-label.csv.gz > target/m1-data/node-label.csv
//! gunzip -kc target/m1-data/arxiv/raw/node_year.csv.gz  > target/m1-data/node-year.csv
//! # features: target/m1-data/node-feat-100k.csv (first 100k rows already extracted)
//! ```

use std::path::Path;

/// Default in-repo paths (relative to workspace root).
pub const FEAT_100K: &str = "target/m1-data/node-feat-100k.csv";
pub const LABELS: &str = "target/m1-data/node-label.csv";
pub const YEARS: &str = "target/m1-data/node-year.csv";

/// An aligned ogbn-arxiv slice: `feats[i]`, `labels[i]`, `years[i]` all describe node `i`.
#[derive(Clone)]
pub struct Dataset {
    pub feats: Vec<Vec<f32>>,
    pub labels: Vec<u32>,
    pub years: Vec<i32>,
    pub dim: usize,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.feats.len()
    }
    pub fn is_empty(&self) -> bool {
        self.feats.is_empty()
    }

    /// Load `max_n` aligned rows (capped by the shortest file). Panics on malformed
    /// input — this is a benchmark harness, not a service; failing loud is correct.
    pub fn load(
        feat_path: impl AsRef<Path>,
        label_path: impl AsRef<Path>,
        year_path: impl AsRef<Path>,
        max_n: usize,
    ) -> Dataset {
        let feats = read_feats(feat_path.as_ref(), max_n);
        let labels = read_ints(label_path.as_ref(), max_n);
        let years = read_ints(year_path.as_ref(), max_n);

        // Truncate all three to the common minimum so alignment is exact.
        let n = feats.len().min(labels.len()).min(years.len());
        let dim = feats.first().map(|v| v.len()).unwrap_or(0);
        assert!(n > 0, "empty dataset after load");
        assert!(
            feats.iter().take(n).all(|v| v.len() == dim),
            "ragged feature rows — dim must be constant"
        );

        Dataset {
            feats: feats.into_iter().take(n).collect(),
            labels: labels.into_iter().take(n).map(|v| v as u32).collect(),
            years: years.into_iter().take(n).map(|v| v as i32).collect(),
            dim,
        }
    }

    /// Convenience: load the standard in-repo 100k arxiv slice.
    pub fn load_arxiv(max_n: usize) -> Dataset {
        Dataset::load(FEAT_100K, LABELS, YEARS, max_n)
    }
}

fn read_feats(path: &Path, max_n: usize) -> Vec<Vec<f32>> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read features {}: {e}", path.display()));
    raw.lines()
        .take(max_n)
        .map(|line| {
            line.split(',')
                .map(|f| f.trim().parse::<f32>().expect("parse feature f32"))
                .collect::<Vec<f32>>()
        })
        .collect()
}

fn read_ints(path: &Path, max_n: usize) -> Vec<i64> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read ints {}: {e}", path.display()));
    raw.lines()
        .take(max_n)
        .map(|line| line.trim().parse::<i64>().expect("parse int"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_aligned_slice() {
        // Small slice keeps the test fast; skips cleanly if data isn't extracted.
        if !Path::new(FEAT_100K).exists() {
            eprintln!("skip: {FEAT_100K} not extracted");
            return;
        }
        let ds = Dataset::load_arxiv(2000);
        assert_eq!(ds.len(), 2000);
        assert_eq!(ds.labels.len(), 2000);
        assert_eq!(ds.years.len(), 2000);
        assert_eq!(ds.dim, 128);
        assert!(ds.labels.iter().all(|&l| l < 40), "arxiv has 40 subject labels");
        assert!(ds.years.iter().all(|&y| (1900..=2025).contains(&y)));
    }
}
