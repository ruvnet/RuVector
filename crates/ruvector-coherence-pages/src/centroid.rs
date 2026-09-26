/// Centroid page store: cluster vectors into pages using k-means (Lloyd's algorithm).
///
/// Build: run K-means for `iters` iterations, assign each vector to nearest centroid.
/// Search: rank all page centroids by dot product with query, probe top-P pages.
use crate::{dot, normalize, BuildStats, PageStore, SearchResult, VecPage};
use std::time::Instant;

pub struct CentroidPageStore {
    pages: Vec<VecPage>,
    num_pages: usize,
    iters: usize,
}

impl CentroidPageStore {
    /// Create a store that builds `num_pages` pages with `iters` k-means iterations.
    pub fn new(num_pages: usize, iters: usize) -> Self {
        Self {
            pages: Vec::new(),
            num_pages,
            iters,
        }
    }
}

impl PageStore for CentroidPageStore {
    fn name(&self) -> &str {
        "centroid-pages"
    }

    fn build_from(&mut self, data: Vec<(usize, Vec<f32>)>) -> BuildStats {
        let t = Instant::now();
        let n = data.len();
        let dim = data[0].1.len();
        let k = self.num_pages.min(n);

        // Seed centroids from data with stride sampling for determinism.
        let stride = n / k;
        let mut centroids: Vec<Vec<f32>> = (0..k).map(|i| data[i * stride].1.clone()).collect();

        let mut assignments = vec![0usize; n];

        for _ in 0..self.iters {
            // Assignment step.
            for (i, (_, v)) in data.iter().enumerate() {
                let best = centroids
                    .iter()
                    .enumerate()
                    .map(|(ci, c)| (ci, dot(v, c)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(ci, _)| ci)
                    .unwrap_or(0);
                assignments[i] = best;
            }

            // Update step: recompute centroid as mean of assigned vectors.
            let mut new_centroids = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];
            for (i, (_, v)) in data.iter().enumerate() {
                let ci = assignments[i];
                for (c, x) in new_centroids[ci].iter_mut().zip(v.iter()) {
                    *c += x;
                }
                counts[ci] += 1;
            }
            for (ci, nc) in new_centroids.iter_mut().enumerate() {
                if counts[ci] > 0 {
                    let cnt = counts[ci] as f32;
                    for x in nc.iter_mut() {
                        *x /= cnt;
                    }
                    normalize(nc);
                    centroids[ci] = nc.clone();
                }
            }
        }

        // Build pages from final assignments.
        let mut page_vecs: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); k];
        for (i, (id, v)) in data.into_iter().enumerate() {
            page_vecs[assignments[i]].push((id, v));
        }

        // Handle any empty pages: assign to page 0 (shouldn't happen with stride seeding
        // unless k > n, which we guard against above).
        self.pages = page_vecs
            .into_iter()
            .filter(|pv| !pv.is_empty())
            .map(VecPage::from_vecs)
            .collect();

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

        // Rank pages by centroid dot product with query.
        let mut page_scores: Vec<(usize, f32)> = self
            .pages
            .iter()
            .enumerate()
            .map(|(i, pg)| (i, dot(query, &pg.centroid)))
            .collect();
        page_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Collect candidates from top-P pages.
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
