//! Variant 1 — Naive K-means compactor (baseline).
//!
//! Runs Lloyd's K-means on the embeddings, replaces each cluster with its
//! centroid, and emits one [`WitnessRecord`] per cluster.

use crate::{
    centroid, cosine_sim, recall_clustered, CompactionResult, Compactor, MemoryEntry, MemoryStore,
    WitnessRecord,
};

/// Baseline compactor: K-means, then centroid substitution.
pub struct NaiveCompactor {
    pub max_iters: usize,
    pub seed: u64,
}

impl Default for NaiveCompactor {
    fn default() -> Self {
        Self {
            max_iters: 30,
            seed: 42,
        }
    }
}

impl Compactor for NaiveCompactor {
    fn name(&self) -> &'static str {
        "naive-kmeans"
    }

    fn compact(
        &self,
        store: &mut MemoryStore,
        target_ratio: f64,
        queries: &[Vec<f32>],
        k: usize,
    ) -> CompactionResult {
        let n = store.len();
        let target_k = ((n as f64) * target_ratio).round().max(1.0) as usize;
        let before: Vec<MemoryEntry> = store.entries.clone();

        let clusters = kmeans(&store.entries, target_k, self.max_iters, self.seed);

        let mut new_entries: Vec<MemoryEntry> = Vec::with_capacity(target_k);
        let mut witness: Vec<WitnessRecord> = Vec::new();
        let mut new_id = store.next_id;

        for cluster in &clusters {
            let embs: Vec<&[f32]> = cluster
                .iter()
                .map(|&i| before[i].embedding.as_slice())
                .collect();
            let c = centroid(&embs);
            let intra_sim = avg_intra_sim(&before, cluster);
            let merged_ids: Vec<u64> = cluster.iter().map(|&i| before[i].id).collect();
            witness.push(WitnessRecord {
                centroid_id: new_id,
                merged_ids,
                intra_sim,
            });
            new_entries.push(MemoryEntry {
                id: new_id,
                embedding: c,
                age: cluster.iter().map(|&i| before[i].age).max().unwrap_or(0),
                metadata: format!("centroid({})", cluster.len()),
            });
            new_id += 1;
        }
        store.entries = new_entries;
        store.next_id = new_id;

        let recall = recall_clustered(queries, &before, &store.entries, &witness, k);
        let compacted = store.len();
        CompactionResult {
            variant: self.name().to_string(),
            original_count: n,
            compacted_count: compacted,
            compaction_ratio: 1.0 - compacted as f64 / n as f64,
            recall_at_k: recall,
            duration_ms: 0,
            witness_records: witness,
        }
    }
}

/// Average pairwise cosine similarity within a cluster.
pub fn avg_intra_sim(entries: &[MemoryEntry], cluster: &[usize]) -> f32 {
    if cluster.len() < 2 {
        return 1.0;
    }
    let mut s = 0.0_f32;
    let mut pairs = 0usize;
    for ii in 0..cluster.len() {
        for jj in ii + 1..cluster.len() {
            s += cosine_sim(
                &entries[cluster[ii]].embedding,
                &entries[cluster[jj]].embedding,
            );
            pairs += 1;
        }
    }
    if pairs == 0 {
        1.0
    } else {
        s / pairs as f32
    }
}

/// Lloyd's K-means using cosine similarity as the affinity measure.
/// Returns cluster assignments as groups of original indices.
pub fn kmeans(entries: &[MemoryEntry], k: usize, max_iters: usize, seed: u64) -> Vec<Vec<usize>> {
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    let n = entries.len();
    let k = k.min(n);
    if k == 0 || n == 0 {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);

    let dim = entries[0].embedding.len();
    let mut centroids: Vec<Vec<f32>> = indices[..k]
        .iter()
        .map(|&i| entries[i].embedding.clone())
        .collect();

    let mut assignments: Vec<usize> = vec![0; n];

    for _ in 0..max_iters {
        let mut changed = false;
        for (i, entry) in entries.iter().enumerate() {
            let best = centroids
                .iter()
                .enumerate()
                .map(|(ci, c)| (ci, cosine_sim(&entry.embedding, c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci)
                .unwrap_or(0);
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0; dim]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for (i, &ci) in assignments.iter().enumerate() {
            let emb = &entries[i].embedding;
            for (j, &v) in emb.iter().enumerate() {
                sums[ci][j] += v;
            }
            counts[ci] += 1;
        }
        for (ci, sum) in sums.iter().enumerate() {
            let c = counts[ci].max(1) as f32;
            centroids[ci] = sum.iter().map(|&v| v / c).collect();
        }
    }

    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &ci) in assignments.iter().enumerate() {
        groups[ci].push(i);
    }
    groups.retain(|g| !g.is_empty());
    groups
}
