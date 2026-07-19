//! Adaptive semantic memory tiering for RuVector agent stores.
//!
//! Vectors are assigned to one of three storage tiers—Hot, Warm, Cold—based on
//! a scoring function that combines access recency, semantic coherence, and
//! graph centrality.  Three pluggable scoring strategies are measured:
//!
//! | Strategy | Signal | Tier placement basis |
//! |----------|--------|----------------------|
//! | `AccessOnly` | access count | how often a vector has been queried |
//! | `Coherence` | intra-cluster L2 mean | how tightly clustered a vector is |
//! | `SemanticTemp` | recency + coherence + centrality | composite semantic temperature |
//!
//! The central claim: when the query workload concentrates on semantically
//! coherent clusters, `SemanticTemp` places those clusters in the hot tier
//! before access patterns can "teach" the system—solving the cold-start
//! placement problem for newly ingested agent memories.

pub mod dataset;
pub mod distance;
pub mod scorer;
pub mod store;
pub mod temperature;

pub use scorer::{AccessOnlyScorer, CoherenceScorer, SemanticTempScorer};
pub use store::TieredStore;
pub use temperature::TieringConfig;

/// Physical storage tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// In-memory, highest priority.  O(hot_n) scan.
    Hot,
    /// Memory-mapped or second-level cache.  O(warm_n) scan.
    Warm,
    /// Disk-resident (simulated in PoC by the cold flat array).
    Cold,
}

/// A single retrieved candidate.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: u64,
    /// Squared L2 distance to the query vector.
    pub dist_sq: f32,
    pub tier: Tier,
}

/// Trait implemented by all three tier scoring strategies.
pub trait Scorer: Send + Sync {
    /// Human-readable variant name used in benchmark output.
    fn name(&self) -> &'static str;

    /// Score a vector's metadata; higher score → hotter tier.
    fn score(&self, meta: &temperature::VectorMeta, current_epoch: u64, cfg: &TieringConfig)
        -> f32;

    /// Whether this scorer requires coherence to be pre-computed in metadata.
    fn needs_coherence(&self) -> bool;
}

/// Summary stats accumulated over a query batch.
#[derive(Default, Debug, Clone)]
pub struct BatchStats {
    pub hot_hits: u64,
    pub warm_hits: u64,
    pub cold_hits: u64,
    pub total_result_slots: u64,
}

impl BatchStats {
    pub fn record_result(&mut self, tier: Tier) {
        self.total_result_slots += 1;
        match tier {
            Tier::Hot => self.hot_hits += 1,
            Tier::Warm => self.warm_hits += 1,
            Tier::Cold => self.cold_hits += 1,
        }
    }

    /// Fraction of returned results found in the hot tier.
    pub fn hot_hit_rate(&self) -> f64 {
        if self.total_result_slots == 0 {
            return 0.0;
        }
        self.hot_hits as f64 / self.total_result_slots as f64
    }

    /// Fraction of returned results found in hot or warm tiers.
    pub fn hot_warm_hit_rate(&self) -> f64 {
        if self.total_result_slots == 0 {
            return 0.0;
        }
        (self.hot_hits + self.warm_hits) as f64 / self.total_result_slots as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dataset::generate_clustered, store::TieredStore, AccessOnlyScorer, CoherenceScorer,
        SemanticTempScorer, TieringConfig,
    };

    const DIMS: usize = 32;
    const N_VECS: usize = 400;
    const HOT_CAP: usize = 40; // 10 %
    const WARM_CAP: usize = 120; // 30 %

    fn config() -> TieringConfig {
        TieringConfig {
            hot_capacity: HOT_CAP,
            warm_capacity: WARM_CAP,
            ..TieringConfig::default()
        }
    }

    fn rng() -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(42)
    }

    /// All strategies must return exactly k results when k ≤ n_vecs.
    #[test]
    fn search_returns_k() {
        let mut rng = rng();
        let (vecs, _labels) = generate_clustered(N_VECS, DIMS, &mut rng);
        let cfg = config();
        let mut store = TieredStore::new(cfg.clone(), CoherenceScorer);
        for (id, v) in vecs.iter().enumerate() {
            store.insert(id as u64, v.clone());
        }
        store.evaluate_tiers(0);
        let q: Vec<f32> = vecs[0].clone();
        let results = store.search(&q, 10);
        assert_eq!(results.len(), 10, "search must return k results");
    }

    /// Tier counts must sum to total inserted.
    #[test]
    fn tier_counts_sum_to_total() {
        let mut rng = rng();
        let (vecs, _) = generate_clustered(N_VECS, DIMS, &mut rng);
        let cfg = config();
        let mut store = TieredStore::new(cfg, SemanticTempScorer);
        for (id, v) in vecs.iter().enumerate() {
            store.insert(id as u64, v.clone());
        }
        store.evaluate_tiers(0);
        let (h, w, c) = store.tier_counts();
        assert_eq!(h + w + c, N_VECS, "tiers must account for all vectors");
    }

    /// Hot tier must not exceed configured capacity.
    #[test]
    fn hot_tier_respects_capacity() {
        let mut rng = rng();
        let (vecs, _) = generate_clustered(N_VECS, DIMS, &mut rng);
        let cfg = config();
        let mut store = TieredStore::new(cfg.clone(), AccessOnlyScorer);
        for (id, v) in vecs.iter().enumerate() {
            store.insert(id as u64, v.clone());
        }
        store.evaluate_tiers(0);
        let (h, _, _) = store.tier_counts();
        assert!(h <= cfg.hot_capacity, "hot tier must not exceed capacity");
    }

    /// CoherenceScorer must place the tight cluster in the hot tier more often
    /// than AccessOnly when the tight cluster has never been accessed.
    #[test]
    fn coherence_beats_access_only_on_cold_start() {
        let mut rng = rng();
        let (vecs, labels) = generate_clustered(N_VECS, DIMS, &mut rng);
        let cfg = config();

        // --- AccessOnly: warmup with noise queries ---
        let mut access_store = TieredStore::new(cfg.clone(), AccessOnlyScorer);
        for (id, v) in vecs.iter().enumerate() {
            access_store.insert(id as u64, v.clone());
        }
        // Warm up with noise vectors (label == 2) as proxy queries
        let noise_query: Vec<f32> = vecs
            .iter()
            .zip(labels.iter())
            .find(|(_, &l)| l == 2)
            .map(|(v, _)| v.clone())
            .unwrap();
        for _ in 0..20 {
            let results = access_store.search(&noise_query, 5);
            for r in &results {
                access_store.record_access(r.id, 50);
            }
        }
        access_store.evaluate_tiers(50);

        // --- Coherence: no warmup, coherence drives placement ---
        let mut coh_store = TieredStore::new(cfg.clone(), CoherenceScorer);
        for (id, v) in vecs.iter().enumerate() {
            coh_store.insert(id as u64, v.clone());
        }
        coh_store.evaluate_tiers(0);

        // Evaluate: query the important cluster (label == 0)
        let tight_query: Vec<f32> = vecs
            .iter()
            .zip(labels.iter())
            .find(|(_, &l)| l == 0)
            .map(|(v, _)| v.clone())
            .unwrap();

        let mut access_hot = 0u64;
        let mut coh_hot = 0u64;
        let rounds = 10;

        for _ in 0..rounds {
            for r in access_store.search(&tight_query, 5) {
                if r.tier == Tier::Hot {
                    access_hot += 1;
                }
            }
            for r in coh_store.search(&tight_query, 5) {
                if r.tier == Tier::Hot {
                    coh_hot += 1;
                }
            }
        }

        assert!(
            coh_hot > access_hot,
            "CoherenceScorer ({coh_hot} hot hits) must beat AccessOnly ({access_hot}) \
             on cold-start important-cluster queries"
        );
    }

    /// SemanticTemp must place the tight cluster hotter than AccessOnly
    /// even when access patterns favour the noise cluster.
    #[test]
    fn semantic_temp_beats_access_only() {
        let mut rng = rng();
        let (vecs, labels) = generate_clustered(N_VECS, DIMS, &mut rng);
        let cfg = TieringConfig {
            hot_capacity: HOT_CAP,
            warm_capacity: WARM_CAP,
            ..TieringConfig::default()
        };

        let mut st_store = TieredStore::new(cfg.clone(), SemanticTempScorer);
        let mut ao_store = TieredStore::new(cfg.clone(), AccessOnlyScorer);

        for (id, v) in vecs.iter().enumerate() {
            st_store.insert(id as u64, v.clone());
            ao_store.insert(id as u64, v.clone());
        }
        // Warmup: 30 noise queries to inflate access counts for noise vectors
        let noise_q: Vec<f32> = vecs
            .iter()
            .zip(labels.iter())
            .find(|(_, &l)| l == 2)
            .map(|(v, _)| v.clone())
            .unwrap();
        for _ in 0..30 {
            for r in st_store.search(&noise_q, 5) {
                st_store.record_access(r.id, 30);
            }
            for r in ao_store.search(&noise_q, 5) {
                ao_store.record_access(r.id, 30);
            }
        }
        st_store.evaluate_tiers(30);
        ao_store.evaluate_tiers(30);

        let tight_q: Vec<f32> = vecs
            .iter()
            .zip(labels.iter())
            .find(|(_, &l)| l == 0)
            .map(|(v, _)| v.clone())
            .unwrap();

        let mut st_hot = 0u64;
        let mut ao_hot = 0u64;
        for _ in 0..20 {
            for r in st_store.search(&tight_q, 5) {
                if r.tier == Tier::Hot {
                    st_hot += 1;
                }
            }
            for r in ao_store.search(&tight_q, 5) {
                if r.tier == Tier::Hot {
                    ao_hot += 1;
                }
            }
        }
        assert!(
            st_hot > ao_hot,
            "SemanticTemp ({st_hot}) must beat AccessOnly ({ao_hot}) on cold-start tight-cluster queries"
        );
    }
}
