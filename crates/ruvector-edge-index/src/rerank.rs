//! Exact f32 rerank of code-space candidates.
//!
//! f32 vectors are not resident (ADR §6.1: "f32 rerank from SQLite"); the
//! store implements [`RerankFetch`] with paged
//! `SELECT iid, f32 FROM vectors WHERE iid IN (…)`. Batches never exceed
//! [`RERANK_BATCH`] ids, matching the ≤ 100 bound-parameter rule.

use crate::error::RerankError;
use crate::heap::{to_hits, Cand, Hit, TopK};
use crate::metric::{distance, Metric};

/// Maximum ids per fetch call.
pub const RERANK_BATCH: usize = 100;

/// Batched f32 vector source.
pub trait RerankFetch {
    /// Fetch error (e.g. a SQL failure).
    type Error;

    /// Look up `iids` (≤ [`RERANK_BATCH`]) and call `sink(iid, vector)` for
    /// each one found. Ids that no longer exist (deleted concurrently) are
    /// simply not reported.
    fn fetch(&mut self, iids: &[u32], sink: &mut dyn FnMut(u32, &[f32]))
        -> Result<(), Self::Error>;
}

/// Rerank `candidates` exactly and keep the best `k`, ascending by
/// `(distance, iid)`.
pub fn rerank<F: RerankFetch>(
    metric: Metric,
    query: &[f32],
    candidates: &[u32],
    k: usize,
    fetch: &mut F,
) -> Result<Vec<Hit>, RerankError<F::Error>> {
    let mut top = TopK::new(k);
    let mut bad = None;
    // (iid, reported) for the current batch: the sink only accepts ids that
    // were asked for, each at most once, whatever the fetch returns (a
    // duplicate-producing join, a deleted row still in SQLite, …).
    let mut want: Vec<(u32, bool)> = Vec::with_capacity(RERANK_BATCH);
    for batch in candidates.chunks(RERANK_BATCH) {
        want.clear();
        want.extend(batch.iter().map(|&i| (i, false)));
        want.sort_unstable();
        want.dedup_by_key(|w| w.0);
        fetch
            .fetch(batch, &mut |iid, v| {
                match want.binary_search_by_key(&iid, |w| w.0) {
                    Ok(p) if !want[p].1 => want[p].1 = true,
                    _ => return,
                }
                if v.len() != query.len() {
                    bad.get_or_insert(iid);
                    return;
                }
                top.push(Cand {
                    d: distance(metric, query, v),
                    id: iid,
                });
            })
            .map_err(RerankError::Fetch)?;
        if let Some(iid) = bad {
            return Err(RerankError::DimMismatch { iid });
        }
    }
    Ok(to_hits(top.into_sorted()))
}

/// [`RerankFetch`] over a row-major in-memory slab where row `i` is iid `i`
/// (tests, benches, and the M1 resident f32 slab).
#[derive(Debug, Clone, Copy)]
pub struct SliceFetch<'a> {
    data: &'a [f32],
    dim: usize,
    calls: usize,
    max_batch: usize,
}

impl<'a> SliceFetch<'a> {
    /// Wrap `data` (`len % dim == 0`).
    pub fn new(data: &'a [f32], dim: usize) -> Self {
        Self {
            data,
            dim,
            calls: 0,
            max_batch: 0,
        }
    }
    /// Number of fetch calls served.
    pub fn calls(&self) -> usize {
        self.calls
    }
    /// Largest batch requested.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }
}

impl RerankFetch for SliceFetch<'_> {
    type Error = std::convert::Infallible;

    fn fetch(
        &mut self,
        iids: &[u32],
        sink: &mut dyn FnMut(u32, &[f32]),
    ) -> Result<(), Self::Error> {
        self.calls += 1;
        self.max_batch = self.max_batch.max(iids.len());
        for &i in iids {
            let start = i as usize * self.dim;
            if let Some(row) = self.data.get(start..start + self.dim) {
                sink(i, row);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_and_orders() {
        let dim = 2;
        let data: Vec<f32> = (0..500).flat_map(|i| [i as f32, 0.0]).collect();
        let mut f = SliceFetch::new(&data, dim);
        let cands: Vec<u32> = (0..250).rev().chain([9999]).collect();
        let hits = rerank(Metric::L2, &[10.2, 0.0], &cands, 3, &mut f).unwrap();
        let ids: Vec<u32> = hits.iter().map(|h| h.iid).collect();
        assert_eq!(ids, vec![10, 11, 9]);
        assert_eq!((f.calls(), f.max_batch()), (3, RERANK_BATCH));
    }

    struct Bad;
    impl RerankFetch for Bad {
        type Error = &'static str;
        fn fetch(
            &mut self,
            iids: &[u32],
            sink: &mut dyn FnMut(u32, &[f32]),
        ) -> Result<(), &'static str> {
            if iids[0] == 0 {
                sink(0, &[1.0]);
                Ok(())
            } else {
                Err("sql")
            }
        }
    }

    /// Reports every requested row twice, plus a row nobody asked for.
    struct Sloppy<'a>(SliceFetch<'a>);
    impl RerankFetch for Sloppy<'_> {
        type Error = std::convert::Infallible;
        fn fetch(
            &mut self,
            iids: &[u32],
            sink: &mut dyn FnMut(u32, &[f32]),
        ) -> Result<(), Self::Error> {
            self.0.fetch(iids, sink)?;
            self.0.fetch(iids, sink)?;
            sink(0, &[10.2, 0.0]); // exact match, never a candidate
            Ok(())
        }
    }

    #[test]
    fn ignores_duplicate_and_unrequested_rows() {
        let data: Vec<f32> = (0..300).flat_map(|i| [i as f32, 0.0]).collect();
        let mut f = Sloppy(SliceFetch::new(&data, 2));
        let cands: Vec<u32> = (5..250).chain([7, 7]).collect();
        let hits = rerank(Metric::L2, &[10.2, 0.0], &cands, 4, &mut f).unwrap();
        let ids: Vec<u32> = hits.iter().map(|h| h.iid).collect();
        assert_eq!(ids, vec![10, 11, 9, 12]);
    }

    #[test]
    fn errors_are_typed() {
        assert_eq!(
            rerank(Metric::L2, &[0.0, 0.0], &[0], 1, &mut Bad),
            Err(RerankError::DimMismatch { iid: 0 })
        );
        assert_eq!(
            rerank(Metric::L2, &[0.0, 0.0], &[5], 1, &mut Bad),
            Err(RerankError::Fetch("sql"))
        );
    }
}
