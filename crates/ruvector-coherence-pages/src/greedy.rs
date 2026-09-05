/// Greedy coherence page store: build pages by iteratively seeding from the most
/// central unassigned vector and greedily pulling in its nearest neighbors.
///
/// This produces pages with higher intra-page coherence than centroid clustering,
/// at the cost of O(N²) build time. Suitable for agent memory compaction where
/// build happens offline and query quality matters.
use crate::{dot, BuildStats, PageStore, SearchResult, VecPage};
use std::time::Instant;

pub struct GreedyCoherenceStore {
    pages: Vec<VecPage>,
    page_size: usize,
}

impl GreedyCoherenceStore {
    /// `page_size`: max vectors per page.
    /// `probe` is passed per-search call; this constructor just takes page_size.
    pub fn new(page_size: usize, _probe: usize) -> Self {
        Self {
            pages: Vec::new(),
            page_size,
        }
    }
}

impl PageStore for GreedyCoherenceStore {
    fn name(&self) -> &str {
        "greedy-coherence"
    }

    fn build_from(&mut self, data: Vec<(usize, Vec<f32>)>) -> BuildStats {
        let t = Instant::now();
        let n = data.len();
        if n == 0 {
            return BuildStats {
                page_count: 0,
                avg_coherence: 0.0,
                build_ms: 0,
            };
        }

        // Track which indices are still available.
        let mut available: Vec<bool> = vec![true; n];
        let mut remaining = n;
        let ps = self.page_size.max(1);

        while remaining > 0 {
            // Seed: pick the first available index.
            let seed_idx = available
                .iter()
                .position(|&a| a)
                .expect("remaining > 0 but no available index");
            available[seed_idx] = false;
            remaining -= 1;

            let seed_vec = data[seed_idx].1.clone();
            let seed_id = data[seed_idx].0;

            // Score all remaining vectors by similarity to seed.
            let mut scored: Vec<(usize, f32)> = (0..n)
                .filter(|&i| available[i])
                .map(|i| (i, dot(&seed_vec, &data[i].1)))
                .collect();

            // Sort descending; take up to page_size - 1 neighbors.
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let take = (ps - 1).min(scored.len());

            let mut page_vecs = vec![(seed_id, seed_vec)];
            for &(idx, _) in &scored[..take] {
                page_vecs.push((data[idx].0, data[idx].1.clone()));
                available[idx] = false;
                remaining -= 1;
            }

            self.pages.push(VecPage::from_vecs(page_vecs));
        }

        let avg_coherence = if self.pages.is_empty() {
            0.0
        } else {
            self.pages.iter().map(|p| p.coherence).sum::<f32>() / self.pages.len() as f32
        };

        BuildStats {
            page_count: self.pages.len(),
            avg_coherence,
            build_ms: t.elapsed().as_millis(),
        }
    }

    fn search(&self, query: &[f32], k: usize, probe: usize) -> SearchResult {
        let p = probe.min(self.pages.len());

        // Rank pages by centroid similarity.
        let mut page_scores: Vec<(usize, f32)> = self
            .pages
            .iter()
            .enumerate()
            .map(|(i, pg)| (i, dot(query, &pg.centroid)))
            .collect();
        page_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for &(pi, _) in &page_scores[..p] {
            for (id, v) in &self.pages[pi].vecs {
                candidates.push((*id, dot(query, v)));
            }
        }
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);

        SearchResult {
            ids: candidates.iter().map(|(id, _)| *id).collect(),
            scores: candidates.iter().map(|(_, s)| *s).collect(),
            pages_probed: p,
        }
    }

    fn page_count(&self) -> usize {
        self.pages.len()
    }

    fn avg_page_coherence(&self) -> f32 {
        if self.pages.is_empty() {
            return 0.0;
        }
        self.pages.iter().map(|p| p.coherence).sum::<f32>() / self.pages.len() as f32
    }
}
