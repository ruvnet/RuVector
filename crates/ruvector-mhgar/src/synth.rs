//! Deterministic synthetic dataset for MHGAR benchmarks.
//!
//! Generates a hub-satellite knowledge graph that isolates the value of
//! multi-hop retrieval:
//!
//! * `num_hubs` hub entities, each with a random D-dimensional unit vector.
//! * Each hub has `satellites_per_hub` satellite entities, whose vectors are
//!   small Gaussian perturbations of the hub vector (controlled by `noise_std`).
//! * Satellites are connected to their hub by undirected edges (weight = 1.0).
//! * Hubs are NOT connected to each other (isolated domains).
//!
//! Query generation:
//! * Queries are generated close to hub vectors (not satellite vectors).
//! * Ground truth for recall: the hub itself PLUS its satellites.
//!
//! This means a pure vector-only search finds the hub but may miss distant
//! satellites, while graph-augmented search recovers them via 1-hop expansion.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, StandardNormal};

use crate::{graph::KnowledgeGraph, vector::FlatIndex};

pub struct SynthDataset {
    pub index: FlatIndex,
    pub graph: KnowledgeGraph,
    /// For each query: the list of hub + satellite ids considered ground truth.
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<u32>>,
    pub num_hubs: usize,
    pub satellites_per_hub: usize,
    pub dim: usize,
}

/// The regime under which satellites are placed relative to their hub.
#[derive(Debug, Clone, Copy)]
pub enum SatelliteMode {
    /// Satellites are small Gaussian perturbations of the hub vector.
    /// Vector search finds them directly. Graph expansion provides minimal gain.
    NearHub { noise_std: f32 },
    /// Satellites are independent random unit vectors, semantically distant from the hub.
    /// Vector search cannot find them; graph traversal is the only path.
    CrossCluster,
}

/// Build a hub-satellite synthetic dataset.
///
/// # Arguments
/// * `mode` – controls satellite vector placement (see [`SatelliteMode`]).
/// * `num_hubs` – number of independent topic clusters.
/// * `satellites_per_hub` – number of satellite entities per hub.
/// * `dim` – embedding dimensionality.
/// * `noise_std` – noise for `NearHub` mode (ignored for `CrossCluster`).
/// * `num_queries` – number of query vectors to generate.
/// * `seed` – RNG seed for full determinism.
pub fn build(
    mode: SatelliteMode,
    num_hubs: usize,
    satellites_per_hub: usize,
    dim: usize,
    noise_std: f32,
    num_queries: usize,
    seed: u64,
) -> SynthDataset {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut index = FlatIndex::new(dim);
    let mut graph = KnowledgeGraph::new();

    // Generate hub entities (ids 0..num_hubs).
    let hub_vecs: Vec<Vec<f32>> = (0..num_hubs)
        .map(|_| random_unit_vector(dim, &mut rng))
        .collect();

    for (hub_id, hub_vec) in hub_vecs.iter().enumerate() {
        index.insert(hub_id as u32, hub_vec).unwrap();
    }

    // Generate satellite entities (ids num_hubs..).
    let mut next_id = num_hubs as u32;
    let mut hub_satellite_map: Vec<Vec<u32>> = vec![Vec::new(); num_hubs];

    match mode {
        SatelliteMode::NearHub { noise_std: nstd } => {
            let normal = Normal::new(0.0f32, nstd).unwrap();
            for hub_id in 0..num_hubs {
                let hub_vec = &hub_vecs[hub_id];
                for _ in 0..satellites_per_hub {
                    let sat_vec: Vec<f32> = hub_vec
                        .iter()
                        .map(|&v| v + normal.sample(&mut rng))
                        .collect();
                    let sat_vec = normalise(sat_vec);
                    let sat_id = next_id;
                    index.insert(sat_id, &sat_vec).unwrap();
                    graph.add_undirected_edge(hub_id as u32, sat_id, 1.0);
                    hub_satellite_map[hub_id].push(sat_id);
                    next_id += 1;
                }
            }
        }
        SatelliteMode::CrossCluster => {
            // Satellites are independently random — semantically distant from their hubs.
            // Graph edges are the ONLY link between hub and satellite.
            for hub_id in 0..num_hubs {
                for _ in 0..satellites_per_hub {
                    let sat_vec = random_unit_vector(dim, &mut rng);
                    let sat_id = next_id;
                    index.insert(sat_id, &sat_vec).unwrap();
                    graph.add_undirected_edge(hub_id as u32, sat_id, 1.0);
                    hub_satellite_map[hub_id].push(sat_id);
                    next_id += 1;
                }
            }
        }
    }

    // Generate queries: perturbations of hub vectors.
    let query_noise = Normal::new(0.0f32, noise_std * 0.3).unwrap();
    let mut queries = Vec::with_capacity(num_queries);
    let mut ground_truth = Vec::with_capacity(num_queries);

    for q in 0..num_queries {
        let hub_id = q % num_hubs;
        let hub_vec = &hub_vecs[hub_id];
        let query: Vec<f32> = hub_vec
            .iter()
            .map(|&v| v + query_noise.sample(&mut rng))
            .collect();
        let query = normalise(query);

        // Ground truth: hub itself + all its satellites.
        let mut gt = vec![hub_id as u32];
        gt.extend_from_slice(&hub_satellite_map[hub_id]);
        ground_truth.push(gt);
        queries.push(query);
    }

    SynthDataset {
        index,
        graph,
        queries,
        ground_truth,
        num_hubs,
        satellites_per_hub,
        dim,
    }
}

fn random_unit_vector(dim: usize, rng: &mut StdRng) -> Vec<f32> {
    let v: Vec<f32> = (0..dim).map(|_| StandardNormal.sample(rng)).collect();
    normalise(v)
}

fn normalise(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}
