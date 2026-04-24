//! Synthetic food web data generation for three ecosystem types.
//!
//! Each ecosystem is represented as a directed energy-flow graph where
//! edge weights encode predation/energy transfer probabilities. The
//! adjacency matrix is row-normalized to produce a TPM suitable for
//! IIT Phi computation.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// A single species in the food web.
#[derive(Clone, Debug)]
pub struct Species {
    pub name: String,
    pub trophic_level: TrophicLevel,
}

/// Trophic classification for coloring and grouping.
#[derive(Clone, Debug, PartialEq)]
pub enum TrophicLevel {
    Producer,
    PrimaryConsumer,
    SecondaryConsumer,
    Decomposer,
    Apex,
}

impl TrophicLevel {
    pub fn color(&self) -> &'static str {
        match self {
            TrophicLevel::Producer => "#27ae60",
            TrophicLevel::PrimaryConsumer => "#f1c40f",
            TrophicLevel::SecondaryConsumer => "#e67e22",
            TrophicLevel::Decomposer => "#8e44ad",
            TrophicLevel::Apex => "#e74c3c",
        }
    }

    #[allow(dead_code)]
    pub fn label(&self) -> &'static str {
        match self {
            TrophicLevel::Producer => "Producer",
            TrophicLevel::PrimaryConsumer => "Primary Consumer",
            TrophicLevel::SecondaryConsumer => "Secondary Consumer",
            TrophicLevel::Decomposer => "Decomposer",
            TrophicLevel::Apex => "Apex",
        }
    }
}

/// A complete ecosystem food web.
pub struct Ecosystem {
    pub name: String,
    pub species: Vec<Species>,
    /// Row-major adjacency matrix (energy flow weights, not yet normalized).
    pub adjacency: Vec<f64>,
    /// Row-normalized TPM for IIT analysis.
    pub tpm: Vec<f64>,
}

impl Ecosystem {
    pub fn n(&self) -> usize {
        self.species.len()
    }

    pub fn connection_count(&self) -> usize {
        self.adjacency.iter().filter(|&&w| w > 1e-10).count()
    }

    /// Build a TPM with one species removed (row/column zeroed, renormalized).
    pub fn tpm_without_species(&self, remove_idx: usize) -> Vec<f64> {
        let n = self.n();
        let mut tpm = self.adjacency.clone();
        // Zero out the removed species' row and column
        for j in 0..n {
            tpm[remove_idx * n + j] = 0.0;
            tpm[j * n + remove_idx] = 0.0;
        }
        // Self-loop for removed species to keep matrix stochastic
        tpm[remove_idx * n + remove_idx] = 1.0;
        // Row-normalize remaining rows
        row_normalize(&mut tpm, n);
        tpm
    }
}

/// Row-normalize a flat n x n matrix in place.
fn row_normalize(data: &mut [f64], n: usize) {
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| data[i * n + j]).sum();
        if row_sum > 1e-30 {
            for j in 0..n {
                data[i * n + j] /= row_sum;
            }
        } else {
            // Uniform distribution for isolated nodes
            for j in 0..n {
                data[i * n + j] = 1.0 / n as f64;
            }
        }
    }
}

/// Generate all three ecosystem food webs.
pub fn generate_all_ecosystems() -> Vec<Ecosystem> {
    vec![
        generate_tropical_rainforest(),
        generate_agricultural_monoculture(),
        generate_coral_reef(),
    ]
}

/// Tropical rainforest: 12 species, high connectivity, many redundant pathways.
/// Expected: HIGH Phi (deeply integrated).
fn generate_tropical_rainforest() -> Ecosystem {
    let species = vec![
        // Producers (0-2)
        Species {
            name: "Canopy Tree".into(),
            trophic_level: TrophicLevel::Producer,
        },
        Species {
            name: "Understory Shrub".into(),
            trophic_level: TrophicLevel::Producer,
        },
        Species {
            name: "Epiphyte".into(),
            trophic_level: TrophicLevel::Producer,
        },
        // Primary consumers (3-5)
        Species {
            name: "Leaf Insect".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        Species {
            name: "Fruit Bird".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        Species {
            name: "Herbivore Mammal".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        // Secondary consumers (6-8)
        Species {
            name: "Snake".into(),
            trophic_level: TrophicLevel::SecondaryConsumer,
        },
        Species {
            name: "Raptor".into(),
            trophic_level: TrophicLevel::SecondaryConsumer,
        },
        Species {
            name: "Wild Cat".into(),
            trophic_level: TrophicLevel::SecondaryConsumer,
        },
        // Decomposers (9-11)
        Species {
            name: "Fungi".into(),
            trophic_level: TrophicLevel::Decomposer,
        },
        Species {
            name: "Bacteria".into(),
            trophic_level: TrophicLevel::Decomposer,
        },
        Species {
            name: "Earthworm".into(),
            trophic_level: TrophicLevel::Decomposer,
        },
    ];
    let n = species.len();
    let mut rng = ChaCha8Rng::seed_from_u64(100);

    // Dense energy flow matrix with many cross-trophic connections
    let mut adj = vec![0.0f64; n * n];

    // Producers support each other through shared soil nutrients (weak)
    set_symmetric(&mut adj, n, 0, 1, 0.15 + rng.gen::<f64>() * 0.05);
    set_symmetric(&mut adj, n, 0, 2, 0.10 + rng.gen::<f64>() * 0.05);
    set_symmetric(&mut adj, n, 1, 2, 0.12 + rng.gen::<f64>() * 0.05);

    // Producers -> Primary consumers (strong, multiple pathways)
    for prod in 0..3 {
        for herb in 3..6 {
            adj[herb * n + prod] = 0.3 + rng.gen::<f64>() * 0.2;
            // Nutrient recycling back
            adj[prod * n + herb] = 0.02 + rng.gen::<f64>() * 0.02;
        }
    }

    // Primary -> Secondary consumers (strong)
    for herb in 3..6 {
        for pred in 6..9 {
            adj[pred * n + herb] = 0.25 + rng.gen::<f64>() * 0.15;
        }
    }

    // Cross-predation among secondary consumers
    set_edge(&mut adj, n, 6, 7, 0.05 + rng.gen::<f64>() * 0.03);
    set_edge(&mut adj, n, 7, 8, 0.04 + rng.gen::<f64>() * 0.03);
    set_edge(&mut adj, n, 8, 6, 0.03 + rng.gen::<f64>() * 0.02);

    // Decomposers receive from all levels (death/waste)
    for src in 0..9 {
        for dec in 9..12 {
            adj[dec * n + src] = 0.08 + rng.gen::<f64>() * 0.06;
        }
    }

    // Decomposers return nutrients to producers
    for dec in 9..12 {
        for prod in 0..3 {
            adj[prod * n + dec] = 0.20 + rng.gen::<f64>() * 0.10;
        }
    }

    // Decomposer cross-interactions
    set_symmetric(&mut adj, n, 9, 10, 0.10 + rng.gen::<f64>() * 0.05);
    set_symmetric(&mut adj, n, 10, 11, 0.08 + rng.gen::<f64>() * 0.04);
    set_symmetric(&mut adj, n, 9, 11, 0.06 + rng.gen::<f64>() * 0.03);

    // Secondary consumers occasionally eat decomposers (omnivory)
    for pred in 6..9 {
        adj[pred * n + 11] = 0.03 + rng.gen::<f64>() * 0.02; // eat earthworms
    }

    let mut tpm = adj.clone();
    row_normalize(&mut tpm, n);

    Ecosystem {
        name: "Tropical Rainforest".to_string(),
        species,
        adjacency: adj,
        tpm,
    }
}

/// Agricultural monoculture: 8 species, sparse linear chains.
/// Expected: LOW Phi (fragile, decomposable).
fn generate_agricultural_monoculture() -> Ecosystem {
    let species = vec![
        // 0: Crop (producer)
        Species {
            name: "Wheat Crop".into(),
            trophic_level: TrophicLevel::Producer,
        },
        // 1: Pest
        Species {
            name: "Aphid Pest".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        // 2: Predator of pest
        Species {
            name: "Ladybug".into(),
            trophic_level: TrophicLevel::SecondaryConsumer,
        },
        // 3: Pollinator
        Species {
            name: "Honeybee".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        // 4-5: Soil microbes
        Species {
            name: "Nitrogen Fixer".into(),
            trophic_level: TrophicLevel::Decomposer,
        },
        Species {
            name: "Mycorrhiza".into(),
            trophic_level: TrophicLevel::Decomposer,
        },
        // 6: Weed
        Species {
            name: "Weed".into(),
            trophic_level: TrophicLevel::Producer,
        },
        // 7: Resistant pest variant
        Species {
            name: "Resistant Aphid".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
    ];
    let n = species.len();
    let mut rng = ChaCha8Rng::seed_from_u64(200);

    let mut adj = vec![0.0f64; n * n];

    // Simple linear chain: crop -> pest -> predator
    adj[1 * n + 0] = 0.7 + rng.gen::<f64>() * 0.1; // pest eats crop
    adj[2 * n + 1] = 0.6 + rng.gen::<f64>() * 0.1; // ladybug eats pest
    adj[2 * n + 7] = 0.3 + rng.gen::<f64>() * 0.1; // ladybug eats resistant pest

    // Pollinator weakly interacts with crop
    adj[3 * n + 0] = 0.2 + rng.gen::<f64>() * 0.05; // bee visits crop
    adj[0 * n + 3] = 0.15 + rng.gen::<f64>() * 0.05; // crop benefits from bee

    // Soil microbes support crop
    adj[0 * n + 4] = 0.25 + rng.gen::<f64>() * 0.05; // nitrogen fixation
    adj[0 * n + 5] = 0.20 + rng.gen::<f64>() * 0.05; // mycorrhizal support

    // Crop waste feeds soil microbes (weak)
    adj[4 * n + 0] = 0.10 + rng.gen::<f64>() * 0.03;
    adj[5 * n + 0] = 0.08 + rng.gen::<f64>() * 0.03;

    // Weed competes with crop (negative interaction modeled as weak link)
    adj[6 * n + 0] = 0.05 + rng.gen::<f64>() * 0.02;
    adj[0 * n + 6] = 0.02 + rng.gen::<f64>() * 0.01;

    // Resistant pest also eats crop
    adj[7 * n + 0] = 0.5 + rng.gen::<f64>() * 0.1;

    // Pest and resistant pest weakly interact
    adj[1 * n + 7] = 0.02;
    adj[7 * n + 1] = 0.02;

    let mut tpm = adj.clone();
    row_normalize(&mut tpm, n);

    Ecosystem {
        name: "Agricultural Monoculture".to_string(),
        species,
        adjacency: adj,
        tpm,
    }
}

/// Coral reef: 10 species, moderate connectivity with keystone species (coral).
/// Expected: MODERATE Phi, but removing coral collapses integration.
fn generate_coral_reef() -> Ecosystem {
    let species = vec![
        // 0: Coral (keystone)
        Species {
            name: "Coral".into(),
            trophic_level: TrophicLevel::Producer,
        },
        // 1: Algae
        Species {
            name: "Algae".into(),
            trophic_level: TrophicLevel::Producer,
        },
        // 2-4: Fish
        Species {
            name: "Clownfish".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        Species {
            name: "Parrotfish".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        Species {
            name: "Grouper".into(),
            trophic_level: TrophicLevel::SecondaryConsumer,
        },
        // 5-6: Invertebrates
        Species {
            name: "Sea Urchin".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        Species {
            name: "Crown-of-Thorns".into(),
            trophic_level: TrophicLevel::PrimaryConsumer,
        },
        // 7: Shark (apex)
        Species {
            name: "Reef Shark".into(),
            trophic_level: TrophicLevel::Apex,
        },
        // 8: Sea turtle
        Species {
            name: "Sea Turtle".into(),
            trophic_level: TrophicLevel::SecondaryConsumer,
        },
        // 9: Plankton
        Species {
            name: "Plankton".into(),
            trophic_level: TrophicLevel::Producer,
        },
    ];
    let n = species.len();
    let mut rng = ChaCha8Rng::seed_from_u64(300);

    let mut adj = vec![0.0f64; n * n];

    // Coral is the structural keystone: many species depend on it
    // Coral shelters clownfish
    adj[2 * n + 0] = 0.5 + rng.gen::<f64>() * 0.1;
    // Parrotfish grazes algae off coral (mutually beneficial)
    adj[3 * n + 1] = 0.4 + rng.gen::<f64>() * 0.1;
    adj[0 * n + 3] = 0.3 + rng.gen::<f64>() * 0.1; // coral benefits from parrotfish
                                                   // Grouper eats smaller fish
    adj[4 * n + 2] = 0.3 + rng.gen::<f64>() * 0.1;
    adj[4 * n + 3] = 0.2 + rng.gen::<f64>() * 0.1;
    // Sea urchin grazes algae
    adj[5 * n + 1] = 0.35 + rng.gen::<f64>() * 0.1;
    // Crown-of-thorns eats coral (destructive)
    adj[6 * n + 0] = 0.4 + rng.gen::<f64>() * 0.1;
    // Shark eats grouper and turtle
    adj[7 * n + 4] = 0.35 + rng.gen::<f64>() * 0.1;
    adj[7 * n + 8] = 0.15 + rng.gen::<f64>() * 0.05;
    // Sea turtle eats algae and invertebrates
    adj[8 * n + 1] = 0.2 + rng.gen::<f64>() * 0.05;
    adj[8 * n + 5] = 0.15 + rng.gen::<f64>() * 0.05;
    adj[8 * n + 6] = 0.10 + rng.gen::<f64>() * 0.05;
    // Plankton feeds coral and clownfish
    adj[0 * n + 9] = 0.3 + rng.gen::<f64>() * 0.1;
    adj[2 * n + 9] = 0.2 + rng.gen::<f64>() * 0.05;
    // Algae and coral compete for space
    adj[1 * n + 0] = 0.1 + rng.gen::<f64>() * 0.05;
    adj[0 * n + 1] = 0.08 + rng.gen::<f64>() * 0.03;

    // Nutrient recycling from consumers back to producers/plankton
    for consumer in [2, 3, 4, 5, 6, 7, 8] {
        adj[9 * n + consumer] = 0.03 + rng.gen::<f64>() * 0.02;
    }

    let mut tpm = adj.clone();
    row_normalize(&mut tpm, n);

    Ecosystem {
        name: "Coral Reef".to_string(),
        species,
        adjacency: adj,
        tpm,
    }
}

/// Set a directed edge weight.
fn set_edge(adj: &mut [f64], n: usize, from: usize, to: usize, weight: f64) {
    adj[from * n + to] = weight;
}

/// Set symmetric (bidirectional) edge weights.
fn set_symmetric(adj: &mut [f64], n: usize, a: usize, b: usize, weight: f64) {
    adj[a * n + b] = weight;
    adj[b * n + a] = weight;
}
