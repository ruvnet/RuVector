//! Ground-truth baseline: brute-force exact f32 scan over the whole corpus.

use crate::{sq_l2_f32, AnnVariant, Hit};

pub struct LinearFull {
    vectors: Vec<f32>,
    dims: usize,
    n: usize,
}

impl LinearFull {
    pub fn new(vectors: Vec<f32>, dims: usize) -> Self {
        let n = vectors.len() / dims;
        assert_eq!(vectors.len(), n * dims, "vector store length mismatch");
        LinearFull { vectors, dims, n }
    }

    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self.vectors[i * self.dims..(i + 1) * self.dims]
    }
}

impl AnnVariant for LinearFull {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut hits: Vec<Hit> = (0..self.n)
            .map(|i| Hit {
                id: i,
                dist: sq_l2_f32(query, self.row(i)),
            })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str {
        "LinearFull"
    }

    fn memory_bytes(&self) -> usize {
        self.vectors.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_self_as_nearest() {
        let vecs = vec![0.0, 0.0, 1.0, 1.0, 5.0, 5.0];
        let lf = LinearFull::new(vecs, 2);
        let hits = lf.search(&[1.0, 1.0], 1);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[0].dist, 0.0);
    }
}
