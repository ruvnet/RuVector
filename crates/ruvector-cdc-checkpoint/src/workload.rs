//! A synthetic, deterministic vector+graph index whose on-disk snapshot
//! shape approximates what `ruvector-snapshot` persists for an HNSW-backed
//! collection: a flat vector table plus per-node adjacency lists. It is not
//! wired to `ruvector-core`'s real index (that internal binary layout is
//! out of scope for a one-crate experiment), but the serialization is a
//! real, deterministic byte format and the churn model (small-percentage
//! insert/update/delete batches between checkpoints) matches how an
//! agent-memory collection is actually updated: a few new memories, a few
//! superseded ones deleted, a few re-scored/re-embedded in place.

/// Small, dependency-free PRNG (splitmix64) so the whole workload — vector
/// contents, adjacency, and the churn schedule — is reproducible from one
/// seed without pulling in the `rand` crate.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn next_range(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() as usize) % n
        }
    }
}

pub struct IndexState {
    pub dim: usize,
    pub degree: usize,
    /// `id -> vector`; a tombstoned id maps to `None` but keeps its slot,
    /// matching how a real index defers physical compaction.
    pub vectors: Vec<Option<Vec<f32>>>,
    pub adjacency: Vec<Vec<u32>>,
    next_id: u32,
    rng: Rng,
}

impl IndexState {
    pub fn build(n: usize, dim: usize, degree: usize, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut vectors = Vec::with_capacity(n);
        for _ in 0..n {
            vectors.push(Some((0..dim).map(|_| rng.next_f32()).collect()));
        }
        let mut adjacency = Vec::with_capacity(n);
        for i in 0..n {
            let mut neighbors = Vec::with_capacity(degree);
            for _ in 0..degree {
                neighbors.push(rng.next_range(n) as u32);
            }
            let _ = i;
            adjacency.push(neighbors);
        }
        Self {
            dim,
            degree,
            vectors,
            adjacency,
            next_id: n as u32,
            rng,
        }
    }

    /// One checkpoint round of churn: insert `n_insert` fresh vectors,
    /// tombstone `n_delete` existing (non-tombstoned) ones, and perturb
    /// `n_update` existing vectors' contents and adjacency in place —
    /// representative of new agent memories arriving, superseded memories
    /// being retired, and re-embedding/re-linking on access.
    pub fn churn(&mut self, n_insert: usize, n_delete: usize, n_update: usize) {
        for _ in 0..n_insert {
            let v: Vec<f32> = (0..self.dim).map(|_| self.rng.next_f32()).collect();
            self.vectors.push(Some(v));
            let n = self.vectors.len();
            let neighbors = (0..self.degree)
                .map(|_| self.rng.next_range(n) as u32)
                .collect();
            self.adjacency.push(neighbors);
            self.next_id += 1;
        }

        for _ in 0..n_delete {
            let idx = self.rng.next_range(self.vectors.len());
            self.vectors[idx] = None;
        }

        for _ in 0..n_update {
            let idx = self.rng.next_range(self.vectors.len());
            if self.vectors[idx].is_some() {
                let v: Vec<f32> = (0..self.dim).map(|_| self.rng.next_f32()).collect();
                self.vectors[idx] = Some(v);
                let n = self.vectors.len();
                self.adjacency[idx] = (0..self.degree)
                    .map(|_| self.rng.next_range(n) as u32)
                    .collect();
            }
        }
    }

    /// Deterministic little-endian byte layout: header (counts), then the
    /// vector table (a live tag byte + payload per slot), then adjacency
    /// lists. This is the "checkpoint blob" every variant chunks.
    pub fn serialize(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.vectors.len() * (self.dim * 4 + 1));
        out.extend_from_slice(&(self.vectors.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.degree as u32).to_le_bytes());

        for v in &self.vectors {
            match v {
                Some(vec) => {
                    out.push(1);
                    for f in vec {
                        out.extend_from_slice(&f.to_le_bytes());
                    }
                }
                None => out.push(0),
            }
        }
        for neighbors in &self.adjacency {
            for id in neighbors {
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialization_is_deterministic_for_a_fixed_seed() {
        let a = IndexState::build(100, 8, 4, 1).serialize();
        let b = IndexState::build(100, 8, 4, 1).serialize();
        assert_eq!(a, b);
    }

    #[test]
    fn churn_changes_the_serialized_blob() {
        let mut state = IndexState::build(1000, 16, 8, 2);
        let before = state.serialize();
        state.churn(10, 5, 10);
        let after = state.serialize();
        assert_ne!(before, after);
        // Insertions grow the vector table.
        assert!(after.len() > before.len());
    }
}
