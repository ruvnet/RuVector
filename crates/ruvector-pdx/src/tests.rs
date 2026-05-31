//! Integration tests for ruvector-pdx.
//!
//! All tests use real f32 arithmetic — no mocks, no stubs.
//! Correctness criterion: PdxFlatIndex and PdxPruneIndex must return
//! the same top-k result ids as RowMajorIndex (the exact baseline).

#[cfg(test)]
mod tests {
    use crate::{AnnIndex, PdxFlatIndex, PdxPruneIndex, RowMajorIndex};

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| {
                        let mut h = DefaultHasher::new();
                        (i * 1009 + d * 7 + seed as usize).hash(&mut h);
                        (h.finish() % 1000) as f32 / 500.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    fn exact_top_k(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                (i, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.total_cmp(&b.1));
        dists[..k].iter().map(|(i, _)| *i).collect()
    }

    fn ids(results: &[crate::index::SearchResult]) -> Vec<usize> {
        results.iter().map(|r| r.id).collect()
    }

    // ── layout tests ──────────────────────────────────────────────────────────

    #[test]
    fn pdx_block_push_and_col() {
        use crate::layout::PdxBlock;
        let mut block = PdxBlock::new(4, 3);
        block.push(0, &[1.0, 2.0, 3.0, 4.0]);
        block.push(1, &[5.0, 6.0, 7.0, 8.0]);
        // col(0) should be [1.0, 5.0]
        assert_eq!(block.col(0), &[1.0f32, 5.0]);
        // col(1) should be [2.0, 6.0]
        assert_eq!(block.col(1), &[2.0f32, 6.0]);
        // col(3) should be [4.0, 8.0]
        assert_eq!(block.col(3), &[4.0f32, 8.0]);
    }

    #[test]
    fn row_major_returns_full_block() {
        use crate::layout::RowBlock;
        let mut block = RowBlock::new(3, 2);
        block.push(0, &[1.0, 2.0, 3.0]);
        block.push(1, &[4.0, 5.0, 6.0]);
        assert_eq!(block.row(0), &[1.0f32, 2.0, 3.0]);
        assert_eq!(block.row(1), &[4.0f32, 5.0, 6.0]);
    }

    // ── exact-match correctness ───────────────────────────────────────────────

    fn correctness_harness(n: usize, dim: usize, k: usize, block_size: usize, seed: u64) {
        let corpus = make_corpus(n, dim, seed);
        let query = make_corpus(1, dim, seed + 1)[0].clone();
        let truth = exact_top_k(&corpus, &query, k);

        let mut row = RowMajorIndex::new(dim, block_size);
        let mut pdx = PdxFlatIndex::new(dim, block_size);
        let first_check = (dim / 8).max(4).min(dim);
        let mut pdxp = PdxPruneIndex::new(dim, block_size.min(64), first_check);

        for (i, v) in corpus.iter().enumerate() {
            row.add(i, v.clone()).unwrap();
            pdx.add(i, v.clone()).unwrap();
            pdxp.add(i, v.clone()).unwrap();
        }

        let row_ids = ids(&row.search(&query, k).unwrap());
        let pdx_ids = ids(&pdx.search(&query, k).unwrap());
        let pdxp_ids = ids(&pdxp.search(&query, k).unwrap());

        use std::collections::HashSet;
        let truth_set: HashSet<_> = truth.iter().copied().collect();
        let row_set: HashSet<_> = row_ids.iter().copied().collect();
        let pdx_set: HashSet<_> = pdx_ids.iter().copied().collect();
        let pdxp_set: HashSet<_> = pdxp_ids.iter().copied().collect();

        // Row-major must match exact ground truth (it IS exact)
        assert_eq!(row_set, truth_set, "RowMajorIndex diverged from exact (n={n}, D={dim})");

        // PDX flat must match RowMajor exactly (no approximation, same math)
        assert_eq!(pdx_set, row_set, "PdxFlatIndex diverged from RowMajorIndex (n={n}, D={dim})");

        // PdxPrune: recall@k must be 100% (pruning is exact — zero false negatives)
        let pdxp_recall = pdxp_set.intersection(&truth_set).count() as f64 / k as f64;
        assert!(
            pdxp_recall >= 1.0,
            "PdxPruneIndex recall@{k} = {pdxp_recall:.2} < 1.0 (n={n}, D={dim}, fc={first_check})"
        );
    }

    #[test]
    fn correctness_small_d32() { correctness_harness(100, 32, 5, 32, 1); }

    #[test]
    fn correctness_small_d128() { correctness_harness(200, 128, 10, 64, 2); }

    #[test]
    fn correctness_medium_d384() { correctness_harness(500, 384, 10, 64, 3); }

    #[test]
    fn correctness_k_equals_1() { correctness_harness(300, 64, 1, 32, 4); }

    #[test]
    fn correctness_block_boundary() {
        // n = 3 * block_size — tests that multi-block search works.
        correctness_harness(192, 64, 10, 64, 5);
    }

    // ── error handling ────────────────────────────────────────────────────────

    #[test]
    fn dim_mismatch_returns_error() {
        let mut idx = RowMajorIndex::new(4, 16);
        idx.add(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = idx.add(1, vec![1.0, 2.0]); // wrong dim
        assert!(result.is_err());
    }

    #[test]
    fn k_too_large_returns_error() {
        let mut idx = PdxFlatIndex::new(4, 16);
        idx.add(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let result = idx.search(&[0.0, 0.0, 0.0, 0.0], 10); // k > n
        assert!(result.is_err());
    }

    #[test]
    fn empty_index_returns_error() {
        let idx = PdxPruneIndex::new(4, 16, 2);
        let result = idx.search(&[0.0, 0.0, 0.0, 0.0], 1);
        assert!(result.is_err());
    }

    // ── memory accounting ─────────────────────────────────────────────────────

    #[test]
    fn memory_bytes_nonzero_after_add() {
        let mut idx = PdxFlatIndex::new(128, 64);
        for i in 0..128usize {
            idx.add(i, vec![i as f32; 128]).unwrap();
        }
        let mem = idx.memory_bytes();
        // At minimum: 128 vectors × 128 floats × 4 bytes = 65536 bytes
        assert!(mem >= 128 * 128 * 4, "memory_bytes too small: {mem}");
    }

    // ── distance kernel tests ─────────────────────────────────────────────────

    #[test]
    fn pdx_block_l2_matches_row_major() {
        use crate::layout::{PdxBlock, RowBlock};
        let dim = 8;
        let n = 4;
        let data: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..dim).map(|d| (i * dim + d) as f32).collect())
            .collect();
        let query: Vec<f32> = (0..dim).map(|d| d as f32 * 0.5).collect();

        let mut pdx = PdxBlock::new(dim, n);
        let mut row = RowBlock::new(dim, n);
        for (i, v) in data.iter().enumerate() {
            pdx.push(i, v);
            row.push(i, v);
        }

        // Compute L2 both ways and check they match.
        for i in 0..n {
            let mut pdx_l2 = 0.0f32;
            for d in 0..dim {
                let diff = query[d] - pdx.col(d)[i];
                pdx_l2 += diff * diff;
            }
            let mut row_l2 = 0.0f32;
            for (a, b) in query.iter().zip(row.row(i).iter()) {
                row_l2 += (a - b) * (a - b);
            }
            let diff = (pdx_l2 - row_l2).abs();
            assert!(
                diff < 1e-4,
                "L2 mismatch for vector {i}: pdx={pdx_l2}, row={row_l2}"
            );
        }
    }
}
