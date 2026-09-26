/// Baseline flat store: all vectors in a single "page", exhaustive linear scan.
///
/// Recall is always 1.0. Used as the baseline for latency and coherence comparison.
use crate::{dot, BuildStats, PageStore, SearchResult, VecPage};
use std::time::Instant;

#[derive(Default)]
pub struct FlatStore {
    page: Option<VecPage>,
}

impl PageStore for FlatStore {
    fn name(&self) -> &str {
        "flat"
    }

    fn build_from(&mut self, data: Vec<(usize, Vec<f32>)>) -> BuildStats {
        let t = Instant::now();
        let page = VecPage::from_vecs(data);
        let coherence = page.coherence;
        self.page = Some(page);
        BuildStats {
            page_count: 1,
            avg_coherence: coherence,
            build_ms: t.elapsed().as_millis(),
        }
    }

    fn search(&self, query: &[f32], k: usize, _probe: usize) -> SearchResult {
        let page = self.page.as_ref().expect("store not built");
        let mut scores: Vec<(usize, f32)> = page
            .vecs
            .iter()
            .map(|(id, v)| (*id, dot(query, v)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(k);
        SearchResult {
            ids: scores.iter().map(|(id, _)| *id).collect(),
            scores: scores.iter().map(|(_, s)| *s).collect(),
            pages_probed: 1,
        }
    }

    fn page_count(&self) -> usize {
        1
    }

    fn avg_page_coherence(&self) -> f32 {
        self.page.as_ref().map(|p| p.coherence).unwrap_or(0.0)
    }
}
