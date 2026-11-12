//! Traversal over u8 codes and the public search entry points.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{HnswIndex, Visited, NONE};
use crate::error::{IndexError, RerankError};
use crate::heap::{Cand, Hit};
use crate::metric::validate_vector;
use crate::quant::PreparedQuery;
use crate::rerank::{rerank, RerankFetch};

impl HnswIndex {
    /// Greedy descent on one upper layer.
    pub(crate) fn greedy(&self, pq: &PreparedQuery, mut cur: Cand, layer: usize) -> Cand {
        loop {
            let mut changed = false;
            for &nb in self.links(cur.id, layer) {
                if nb == NONE {
                    break;
                }
                let c = Cand {
                    d: pq.distance(self.code(nb)),
                    id: nb,
                };
                if c < cur {
                    cur = c;
                    changed = true;
                }
            }
            if !changed {
                return cur;
            }
        }
    }

    /// Beam search on one layer; ascending by `(distance, id)`.
    ///
    /// Every node routes, but with `live_only` tombstoned nodes never
    /// enter the result heap, so they cannot take result slots (hnswlib's
    /// scheme): the search keeps expanding until `ef` *live* results are
    /// held or the reachable graph is exhausted. Construction passes
    /// `false` — deleted nodes stay valid neighbours until a purge.
    pub(crate) fn search_layer(
        &self,
        pq: &PreparedQuery,
        eps: &[Cand],
        ef: usize,
        layer: usize,
        visited: &mut Visited,
        live_only: bool,
    ) -> Vec<Cand> {
        let keep = |id: u32| !live_only || !self.is_deleted(id);
        let mut frontier: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        let mut best: BinaryHeap<Cand> = BinaryHeap::with_capacity(ef.min(1024) + 1);
        for &e in eps {
            if !visited.test_set(e.id) {
                frontier.push(Reverse(e));
                if keep(e.id) {
                    best.push(e);
                }
            }
        }
        while let Some(Reverse(c)) = frontier.pop() {
            if best.len() >= ef && best.peek().is_some_and(|w| c > *w) {
                break;
            }
            for &nb in self.links(c.id, layer) {
                if nb == NONE {
                    break;
                }
                if visited.test_set(nb) {
                    continue;
                }
                let cand = Cand {
                    d: pq.distance(self.code(nb)),
                    id: nb,
                };
                if best.len() < ef || best.peek().is_some_and(|w| cand < *w) {
                    frontier.push(Reverse(cand));
                    if keep(nb) {
                        best.push(cand);
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
        }
        best.into_sorted_vec()
    }

    /// Up to `ef` live candidates by code distance (slot ids).
    fn candidates(&self, query: &[f32], ef: usize) -> Result<Vec<Cand>, IndexError> {
        validate_vector(self.quant.metric(), self.quant.dim(), query)?;
        if self.entry == NONE || ef == 0 || self.live == 0 {
            return Ok(Vec::new());
        }
        let pq = self.quant.prepare(query);
        let mut ep = Cand {
            d: pq.distance(self.code(self.entry)),
            id: self.entry,
        };
        for l in (1..=self.top as usize).rev() {
            ep = self.greedy(&pq, ep, l);
        }
        let mut visited = Visited::new(self.levels.len());
        Ok(self.search_layer(&pq, &[ep], ef, 0, &mut visited, true))
    }

    /// Approximate top-`k` live nodes by code distance with beam width
    /// `max(ef, k)` (L2 distances are squared). Returns `min(k, len())`
    /// hits whenever the live nodes are reachable.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<Hit>, IndexError> {
        let mut c = self.candidates(query, ef.max(k))?;
        c.truncate(k);
        Ok(c.into_iter()
            .map(|c| Hit {
                iid: self.iid(c.id),
                distance: c.d,
            })
            .collect())
    }

    /// Beam search with width `max(ef, k)` over codes, then an exact f32
    /// rerank of every live candidate through `fetch` (store iids); best `k`.
    pub fn search_rerank<F: RerankFetch>(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        fetch: &mut F,
    ) -> Result<Vec<Hit>, RerankError<F::Error>> {
        let c = self
            .candidates(query, ef.max(k))
            .map_err(RerankError::Invalid)?;
        let ids: Vec<u32> = c.iter().map(|c| self.iid(c.id)).collect();
        rerank(self.quant.metric(), query, &ids, k, fetch)
    }
}
