pub mod build;
pub mod error;
pub mod graph;
pub mod search;
pub mod vamana;

pub use error::{Result, SymphonyError};
pub use search::{FlatExactIndex, GraphExactIndex, SearchResult, SymphonyIndex};

/// Distance metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// Squared Euclidean distance (monotone for k-NN ranking).
    Euclidean,
    /// Cosine distance: 1 − cosine_similarity.
    Cosine,
}

/// Configuration for a SymphonyQG index.
#[derive(Debug, Clone)]
pub struct Config {
    /// Vector dimensionality. Must be a positive multiple of 8.
    pub dim: usize,
    /// Base neighbor count before BATCH_SIZE padding.
    pub m_base: usize,
    /// Number of candidates evaluated per vertex during construction.
    pub ef_construction: usize,
    pub metric: Metric,
    pub seed: u64,
    /// Optional Vamana α-pruning refinement (DiskANN, NeurIPS 2019).
    /// **Status: experimental (PR #428 iter-7).** Currently helps on
    /// uniform-random data and small/medium clustered corpora; can
    /// regress recall on large (n ≥ 10K) clustered corpora because the
    /// refinement uses the existing sampled-greedy graph as its
    /// candidate source — when that graph has poor recall (e.g. 21%
    /// at n=50K, ef=200), most candidates are wrong and α-pruning
    /// selects diverse-but-wrong neighbours.
    ///
    /// Implemented so far: forward α-prune, back-edge propagation
    /// (DiskANN §3.3), medoid entry-point selection.
    ///
    /// Not yet implemented: iterative refinement from a *random*
    /// initial graph (DiskANN's actual protocol — fixes the
    /// candidate-quality bootstrap problem); two-pass α=1.0 → α>1.0
    /// schedule.
    ///
    /// Validated on uniform-Gaussian data at n=3000 (+15pp recall);
    /// see `examples/vamana_measure.rs` for the empirical state.
    pub vamana: Option<vamana::VamanaConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dim: 128,
            m_base: 16,
            ef_construction: 200,
            metric: Metric::Euclidean,
            seed: 42,
            vamana: None,
        }
    }
}

impl Config {
    /// Hard validation — returns `Err` on configurations that would
    /// produce undefined behaviour or panic deep in the build/search path.
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 || self.dim % 8 != 0 {
            return Err(SymphonyError::InvalidConfig(format!(
                "dim must be > 0 and a multiple of 8; got {}",
                self.dim
            )));
        }
        if self.m_base == 0 {
            return Err(SymphonyError::InvalidConfig("m_base must be > 0".into()));
        }
        if self.ef_construction == 0 {
            return Err(SymphonyError::InvalidConfig(
                "ef_construction must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Soft validation — returns advisory warnings the caller may want
    /// to surface but that don't prevent the index from being built.
    /// Each entry is a single-line human-readable advisory.
    ///
    /// Currently advises:
    /// - `dim < 128`: 1-bit RaBitQ estimation noise σ ≈ sin(θ)/√D becomes
    ///   too high at low dimensionality; recall typically drops below
    ///   useful levels even with re-rank. Empirically (per ADR-193), D=64
    ///   produces estimator stddev ~0.18 on unit vectors and degrades
    ///   recall@10 by 5–10 points compared to D=128. SymphonyQG is not
    ///   the right kernel for low-D embeddings; consider GraphExact or
    ///   a quantization-free kernel instead.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.dim < 128 {
            out.push(format!(
                "dim={} is below the SymphonyQG sweet spot of 128. \
                 1-bit estimation noise σ ≈ sin(θ)/√D will produce noisy beam guidance; \
                 recall may drop 5–10 points vs the same graph at dim=128. \
                 If your embeddings are < 128-D consider GraphExact (skip the 1-bit estimator).",
                self.dim
            ));
        }
        out
    }
}

/// Build all three index variants from the same dataset.
///
/// Calls `Config::validate()` first and panics with a descriptive
/// message on hard-invalid input (zero dim, dim not a multiple of 8,
/// zero m_base, zero ef_construction). For fallible construction use
/// `try_build_all` which returns `Result`.
///
/// Shares one construction pass — the graph is cloned for the two
/// graph-based variants so both see identical edge topology.
pub fn build_all(
    vecs: &[Vec<f32>],
    config: &Config,
) -> (FlatExactIndex, GraphExactIndex, SymphonyIndex) {
    try_build_all(vecs, config)
        .expect("build_all: invalid Config — call Config::validate() first or use try_build_all")
}

/// Fallible variant of `build_all`. Returns `Err` on invalid config
/// instead of panicking. Soft warnings (e.g. `dim < 128`) are NOT
/// errors here — query them separately via `Config::warnings()` if
/// you want to surface them to the user.
pub fn try_build_all(
    vecs: &[Vec<f32>],
    config: &Config,
) -> Result<(FlatExactIndex, GraphExactIndex, SymphonyIndex)> {
    config.validate()?;
    let flat = FlatExactIndex::build(vecs, config);
    let graph = build::build(vecs, config);
    // Optional Vamana α-pruning refinement (NeurIPS 2019). Both index
    // variants benefit from the improved graph topology — keeping their
    // adjacency identical preserves the apples-to-apples comparison.
    let graph = if let Some(ref vcfg) = config.vamana {
        vamana::refine(graph, config, vcfg)
    } else {
        graph
    };
    let graph_clone = graph.clone();
    let graph_exact = GraphExactIndex::from_graph(graph, config.metric);
    let symphony = SymphonyIndex::from_graph(graph_clone, config.metric);
    Ok((flat, graph_exact, symphony))
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[test]
    fn validate_accepts_default() {
        assert!(Config::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_dim_zero() {
        let cfg = Config {
            dim: 0,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_dim_not_multiple_of_8() {
        let cfg = Config {
            dim: 7,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
        let cfg = Config {
            dim: 100,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_m_base_or_ef() {
        let cfg = Config {
            m_base: 0,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
        let cfg = Config {
            ef_construction: 0,
            ..Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn warnings_advises_low_dim() {
        let cfg = Config {
            dim: 64,
            ..Config::default()
        };
        let w = cfg.warnings();
        assert_eq!(w.len(), 1, "expected one warning at dim=64");
        assert!(w[0].contains("dim=64"));
    }

    #[test]
    fn warnings_silent_at_dim_128() {
        let cfg = Config {
            dim: 128,
            ..Config::default()
        };
        assert!(cfg.warnings().is_empty());
    }

    #[test]
    fn try_build_all_propagates_validation_error() {
        let cfg = Config {
            dim: 0,
            ..Config::default()
        };
        let dummy = vec![vec![0.0; 4]; 2];
        let r = try_build_all(&dummy, &cfg);
        assert!(matches!(r, Err(SymphonyError::InvalidConfig(_))));
    }
}
