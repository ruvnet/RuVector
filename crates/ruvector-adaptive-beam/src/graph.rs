/// k-NN graph construction for AdaptiveBeamIndex.
///
/// Builds an exact k-nearest-neighbour graph via parallel exhaustive search.
/// Exact k-NN gives a navigable NSW structure without approximation artefacts,
/// making it the fairest baseline for measuring search-policy differences.
use crate::l2_sq;
use rayon::prelude::*;

/// Build an exact k-NN graph over `vectors`.
///
/// Returns `(adjacency_lists, entry_point_index)`.
/// Each node i connects to its `max_neighbors` nearest peers (excluding itself).
/// The entry point is the medoid — the vector closest to the data centroid —
/// which provides balanced graph traversal from any query direction.
pub fn build_knn_graph(vectors: &[Vec<f32>], max_neighbors: usize) -> (Vec<Vec<u32>>, u32) {
    let n = vectors.len();
    if n == 0 {
        return (vec![], 0);
    }
    if n == 1 {
        return (vec![vec![]], 0);
    }

    let neighbors: Vec<Vec<u32>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dists: Vec<(f32, u32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (l2_sq(&vectors[i], &vectors[j]), j as u32))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(max_neighbors);
            dists.into_iter().map(|(_, j)| j).collect()
        })
        .collect();

    let ep = medoid(vectors);
    (neighbors, ep)
}

/// Returns the index of the vector closest to the centroid of `vectors`.
fn medoid(vectors: &[Vec<f32>]) -> u32 {
    let n = vectors.len();
    let dim = vectors[0].len();
    let centroid: Vec<f32> = (0..dim)
        .map(|d| vectors.iter().map(|v| v[d]).sum::<f32>() / n as f32)
        .collect();
    (0..n)
        .min_by(|&a, &b| {
            l2_sq(&vectors[a], &centroid)
                .partial_cmp(&l2_sq(&vectors[b], &centroid))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0) as u32
}
