//! # DNA Analysis Pipeline using RuVector
//!
//! A complete bioinformatics pipeline demonstrating HNSW-based vector search
//! on real genomic data from NCBI GenBank.
//!
//! ## Pipeline stages:
//! 1. **K-mer Embedding** - Convert DNA sequences to frequency vectors via k-mer analysis
//! 2. **Species Identification** - HNSW nearest-neighbor search on reference embeddings
//! 3. **Variant Detection** - Detect mutations via embedding distance from reference
//! 4. **Phylogenetic Distance** - Pairwise distance matrix between organisms
//!
//! ## Data sources (all public, hardcoded):
//! - PhiX174 bacteriophage (GenBank: J02482)
//! - SARS-CoV-2 Wuhan-Hu-1 (GenBank: MN908947, first 800bp)
//! - E. coli K-12 (GenBank: U00096, first 800bp)
//! - Human mitochondrial DNA (GenBank: NC_012920, first 800bp)
//!
//! Run with: `cargo run --example dna_analysis_pipeline -p ruvector-core --release`

use ruvector_core::distance::{cosine_distance, euclidean_distance};
use ruvector_core::index::hnsw::HnswIndex;
use ruvector_core::index::VectorIndex;
use ruvector_core::types::{DistanceMetric, HnswConfig, SearchResult};
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Real DNA sequences from NCBI GenBank (public domain)
// ---------------------------------------------------------------------------

/// PhiX174 bacteriophage, complete genome (GenBank J02482), first ~800 bp
const PHIX174: &str = "\
GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAAAAATTATCTT\
GATAAAGCAGGAATTACTACTGCTTGTTTACGAATTAAATCGAAGTGGACTGCTGGCGGAAAATGAGAAA\
ATTCGACCTATCCTTGCGCAGCTCGAGAAGCTCTTACTTTGCGACCTTTCGCCATCAACTAACGATTCTG\
TCAAAAACTGACGCGTTGGATGAGGAGAAGTGGCTTAATATGCTTGGCACGTTCGTCAAGGACTGGTTTA\
GATATGAGTCACATTTTGTTCATGGTAGAGATTCTCTTGTTGACATTTTAAAAGAGCGTGGATTACTATCT\
GAGTCCGATGCTGTTCAACCACTAATAGGTAAGAAATCATGAGTCAAGTTACTGAACAATCCGTACGTTTC\
CAGACCGCTTTGGCCTCTATTAAGCTCATTCAGGCTTCTGCCGTTTTGGATTTAACCGAAGATGATTTCGA\
TTTTCTGACGAGTAACAAAGTTTGGATTGCTACTGACCGCTCTCGTGCTCGTCGCTGCGTTGAGGCTTGC\
GTTTATGGTACGCTGGACTTTGTGGGATACCCTCGCTTTCCTGCTCCTGTTGAGTTTATTGCTGCCGTCA\
TTGCTTATTATGTTCATCCCGTCAACATTCAAACGGCCTGTCTCATCATGGAAGGCGCTGAATTTACGGAA\
AACATTATTAATGGCGTCGAGCGTCCGGTTAAAGCCGCTGAATTGTTCGCGTTTACCTTGCGTGTACGCG\
CAGGAAACACTGACGTTCTTACTGACGCAGAAGAAAACGTGCGTCAAAAATTACGTGCG";

/// SARS-CoV-2 Wuhan-Hu-1 (GenBank MN908947.3), first 800 bp
const SARS_COV2: &str = "\
ATTAAAGGTTTATACCTTCCCAGGTAACAAACCAACCAACTTTCGATCTCTTGTAGATCTGTTCTCTAAACG\
AACTTTAAAATCTGTGTGGCTGTCACTCGGCTGCATGCTTAGTGCACTCACGCAGTATAATTAATAACTAAT\
TACTGTCGTTGACAGGACACGAGTAACTCGTCTATCTTCTGCAGGCTGCTTACGGTTTCGTCCGTGTTGCA\
GCCGATCATCAGCACATCTAGGTTTCGTCCGGGTGTGACCGAAAGGTAAGATGGAGAGCCTTGTCCCTGGTT\
TCAACGAGAAAACACACGTCCAACTCAGTTTGCCTGTTTTACAGGTTCGCGACGTGCTCGTACGTGGCTTTG\
GAGACTCCGTGGAGGAGGTCTTATCAGAGGCACGTCAACATCTTAAAGATGGCACTTGTGGCTTAGTAGAAGT\
TGAAAAAGGCGTTTTGCCTCAACTTGAACAGCCCTATGTGTTCATCAAACGTTCGGATGCTCGAACTGCACC\
TCATGGTCATGTTATGGTTGAGCTGGTAGCAGAACTCGAAGGCATTCAGTACGGTCGTAGTGGTGAGACACTT\
GGTGTCCTTGTCCCTCATGTGGGCGAAATACCAGTGGCTTACCGCAAGGTTCTTCTTCGTAAGAACGGTAATA\
AAGGAGCTGGTGGCCATAGTTACGGCGCCGATCTAAAGTCATTTGACTTAGGCGACGAGCTTGGCACTGATCC\
TTATGAAGATTTTCAAGAAAACTGGAACACTAAACATAGCAGTGGTGTTACCCGTGAACTCATGCGTGAGCTT\
AACGGAGGGGCATACACTCGCTATGTCGATAACAACTTCTGTGGCCCTGATGGC";

/// E. coli K-12 MG1655 (GenBank U00096.3), first 800 bp
const ECOLI: &str = "\
AGCTTTTCATTCTGACTGCAACGGGCAATATGTCTCTGTGTGGATTAAAAAAAGAGTGTCTGATAGCAGCTTC\
TGAACTGGTTACCTGCCGTGAGTAAATTAAAATTTTATTGACTTAGGTCACTAAATACTTTAACCAATATAG\
GCATAGCGCACAGACAGATAAAAATTACAGAGTACACAACATCCATGAAACGCATTAGCACCACCATTACCAC\
CACCATCACCATTACCACAGGTAACGGTGCGGGCTGACGCGTACAGGAAACACAGAAAAAAGCCCGCACCTGA\
CAGTGCGGGCTTTTTTTTTCGACCAAAGGTAACGAGGTAACAACCATGCGAGTGTTGAAGTTCGGCGGTACA\
TCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAAAGCAATGCCAGGCAGGGGCAGGTG\
GCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATGATTGAAAAAACCATTAGCGGCC\
AGGATGCTTTACCCAATATCAGCGATGCCGAACGTATTTTTGCCGAACTTTTGACGGGACTCGCCGCCGCCCA\
GCCGGGGTTCCCGCTGGCGCAATTGAAAACTTTCGTCGATCAGGAATTTGCCCAAATAAAACATGTCCTGCAT\
GGCATTAGTTTGTTGGGGCAGTGCCCGGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAAATGTCG\
ATCGCCATTATGGCCGGCGTATTAGAAGCGCGCGGTCACAACGTTACTGTTATCGATCCGGTCGAAAAACTGC\
TGGCAGTGGGGCATTACCTCGAATCTACCGTCGATATTGCTGAGTCCACCCGCCGTATTGCGGCAAGCCGCAT\
TCCGGCTGATCACATGGTGCTGATGGCAGGTTTCACCGCCGGTAATGAAAAAGGCGAACTGGTGGTG";

/// Human mitochondrial DNA (GenBank NC_012920.1), first 800 bp
const HUMAN_MITO: &str = "\
GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGGT\
ATGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTCGCAGTATCTGTCTTTGATTCCTGC\
CATCATGATTCTTCTCAAACATTTTACTGCTCAAGATCCCCTATACAGTGATAGATAACATTAATCATAAACT\
TTAAATATTTAGCTCTCCTTTTAAATTTACAAACCTAAGTATTTTACTTAAATTTTCAGCTTTTCACTCTCATC\
AGCCATAAATTCAAACTGGCACAAACTAACCCCCCTTTTCAAAAATCAATCTCAAATTTATCTATAAAATCCAG\
GCAAAATTATCTACTATTCAATCAACCATCCCATATTAATCGAATGCCCCCCCCATCCCCCCCACTCCTCTTTT\
TACAGAAAGAGGATCAAACATTTCATCACATTTCAAACAAATTCAGAGTAAAAATTTTTAAAAATTTAAATAAA\
AAACATCCAAGCATACAAATCAAACTTTTTTCCCTAAGCCATAACTAATTAGTATAACATTGTCCTATTTTACT\
CAACATTCAATTCATTCATTCAACCCCCAACAATCATAATTTGACTCCATTTTCAAACTAATCCCCCCAACTCC\
TTTTCTTCCCCACATCAATAATACAACAGCATTCACCCATCTTTTCAATCAATTTAATTCACTCAATCAATCAAC\
ACTCTTAACTAACTAACCTCCTCAAACCCAACATTCAACAAACAATCAAGCTAACCCCACCCCCCAATCTTCAAA\
CCACACTCAACACATCCACTCTTCAAAACTACCAAACACATC";

// ---------------------------------------------------------------------------
// K-mer embedding engine
// ---------------------------------------------------------------------------

/// Compute all canonical k-mers of length `k` from a DNA alphabet {A,C,G,T}.
/// Returns a sorted list so the vector dimension index is deterministic.
fn enumerate_kmers(k: usize) -> Vec<String> {
    let bases = ['A', 'C', 'G', 'T'];
    let total = 4usize.pow(k as u32);
    let mut kmers = Vec::with_capacity(total);

    for i in 0..total {
        let mut kmer = String::with_capacity(k);
        let mut val = i;
        for _ in 0..k {
            kmer.push(bases[val % 4]);
            val /= 4;
        }
        kmers.push(kmer);
    }
    kmers.sort();
    kmers
}

/// Build a k-mer index mapping each k-mer string to its vector position.
fn build_kmer_index(kmers: &[String]) -> HashMap<&str, usize> {
    kmers.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect()
}

/// Convert a DNA sequence to a normalized k-mer frequency vector.
///
/// The vector has `4^k` dimensions (one per distinct k-mer). Each element is
/// the frequency (count / total_kmers) of that k-mer in the sequence.
/// The result is L2-normalized so cosine distance is meaningful.
fn sequence_to_embedding(
    seq: &str,
    k: usize,
    kmer_index: &HashMap<&str, usize>,
    dimensions: usize,
) -> Vec<f32> {
    let seq_upper: String = seq.chars().filter(|c| "ACGTacgt".contains(*c)).collect();
    let seq_bytes = seq_upper.as_bytes();
    let mut counts = vec![0u32; dimensions];
    let mut total = 0u32;

    if seq_bytes.len() >= k {
        for window in seq_bytes.windows(k) {
            let kmer = std::str::from_utf8(window).unwrap().to_uppercase();
            if let Some(&idx) = kmer_index.get(kmer.as_str()) {
                counts[idx] += 1;
                total += 1;
            }
        }
    }

    // Frequency vector
    let mut embedding: Vec<f32> = if total > 0 {
        counts.iter().map(|&c| c as f32 / total as f32).collect()
    } else {
        vec![0.0; dimensions]
    };

    // L2-normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut embedding {
            *v /= norm;
        }
    }

    embedding
}

/// Introduce point mutations at given positions in a DNA sequence.
fn mutate_sequence(seq: &str, positions: &[usize]) -> String {
    let complement = |b: char| -> char {
        match b {
            'A' => 'T',
            'T' => 'A',
            'C' => 'G',
            'G' => 'C',
            c => c,
        }
    };
    let mut chars: Vec<char> = seq.chars().collect();
    for &pos in positions {
        if pos < chars.len() {
            chars[pos] = complement(chars[pos]);
        }
    }
    chars.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Pipeline stages
// ---------------------------------------------------------------------------

/// Stage 1: Build reference embeddings and print summary statistics.
fn stage_embeddings<'a>(
    references: &'a [(&'a str, &'a str)],
    k: usize,
    kmer_index: &HashMap<&str, usize>,
    dimensions: usize,
) -> Vec<(&'a str, Vec<f32>)> {
    println!("========================================================");
    println!("  STAGE 1: K-mer Sequence Embedding (k={})", k);
    println!("========================================================");
    println!("  Embedding dimension: {} (4^{} distinct {}-mers)", dimensions, k, k);
    println!();

    let start = Instant::now();
    let mut embeddings = Vec::new();

    for &(name, seq) in references {
        let clean: String = seq.chars().filter(|c| "ACGTacgt".contains(*c)).collect();
        let emb = sequence_to_embedding(seq, k, kmer_index, dimensions);

        // Count non-zero dimensions
        let nonzero = emb.iter().filter(|&&x| x > 0.0).count();
        let max_val = emb.iter().cloned().fold(0.0f32, f32::max);

        println!(
            "  {:25} | len {:>5} bp | non-zero dims: {:>4}/{} | max component: {:.4}",
            name,
            clean.len(),
            nonzero,
            dimensions,
            max_val,
        );
        embeddings.push((name, emb));
    }

    let elapsed = start.elapsed();
    println!();
    println!("  Embedding time: {:.2?}", elapsed);
    println!();

    embeddings
}

/// Stage 2: Species identification via HNSW nearest-neighbor search.
fn stage_species_identification(
    reference_embeddings: &[(&str, Vec<f32>)],
    k: usize,
    kmer_index: &HashMap<&str, usize>,
    dimensions: usize,
) {
    println!("========================================================");
    println!("  STAGE 2: Species Identification via HNSW Index");
    println!("========================================================");
    println!();

    let start = Instant::now();

    // Build HNSW index from reference embeddings
    let hnsw_config = HnswConfig {
        m: 16,
        ef_construction: 200,
        ef_search: 100,
        max_elements: 100,
    };

    let mut index = HnswIndex::new(dimensions, DistanceMetric::Cosine, hnsw_config)
        .expect("Failed to create HNSW index");

    for (name, emb) in reference_embeddings {
        index
            .add(name.to_string(), emb.clone())
            .expect("Failed to add reference to HNSW index");
    }

    let index_time = start.elapsed();
    println!("  HNSW index built in {:.2?} ({} references)", index_time, reference_embeddings.len());
    println!();

    // Create query fragments (subsequences) from known organisms
    let query_fragments: Vec<(&str, &str, &str)> = vec![
        // (fragment_name, source_organism, subsequence)
        ("PhiX174 fragment (bp 100-400)", "PhiX174", &PHIX174[100..400]),
        ("SARS-CoV-2 fragment (bp 200-500)", "SARS-CoV-2", &SARS_COV2[200..500]),
        ("E. coli fragment (bp 50-350)", "E. coli K-12", &ECOLI[50..350]),
        ("Human mito fragment (bp 300-600)", "Human mito", &HUMAN_MITO[300..600]),
    ];

    println!("  {:40} | {:>15} | {:>15} | {:>8}", "Query Fragment", "True Species", "HNSW Match", "Distance");
    println!("  {}", "-".repeat(90));

    let search_start = Instant::now();
    let mut correct = 0;

    for (frag_name, true_species, subseq) in &query_fragments {
        let query_emb = sequence_to_embedding(subseq, k, kmer_index, dimensions);

        let results: Vec<SearchResult> = index
            .search(&query_emb, 4)
            .expect("HNSW search failed");

        let top_match = &results[0];
        let is_correct = top_match.id == *true_species;
        if is_correct {
            correct += 1;
        }

        println!(
            "  {:40} | {:>15} | {:>15} | {:.6} {}",
            frag_name,
            true_species,
            top_match.id,
            top_match.score,
            if is_correct { "[OK]" } else { "[MISS]" },
        );
    }

    let search_time = search_start.elapsed();
    println!();
    println!(
        "  Accuracy: {}/{} ({:.0}%)",
        correct,
        query_fragments.len(),
        100.0 * correct as f64 / query_fragments.len() as f64,
    );
    println!("  Search time: {:.2?} ({} queries)", search_time, query_fragments.len());

    // Show full ranking for one query
    println!();
    println!("  --- Full ranking for \"SARS-CoV-2 fragment (bp 200-500)\" ---");
    let sars_query = sequence_to_embedding(&SARS_COV2[200..500], k, kmer_index, dimensions);
    let full_results = index.search(&sars_query, 4).expect("search failed");
    for (rank, r) in full_results.iter().enumerate() {
        println!("    #{}: {:15} (distance: {:.6})", rank + 1, r.id, r.score);
    }
    println!();
}

/// Stage 3: Variant detection via embedding distance.
fn stage_variant_detection(
    k: usize,
    kmer_index: &HashMap<&str, usize>,
    dimensions: usize,
) {
    println!("========================================================");
    println!("  STAGE 3: Variant Detection Simulation");
    println!("========================================================");
    println!();

    let start = Instant::now();

    // Reference: PhiX174
    let reference_emb = sequence_to_embedding(PHIX174, k, kmer_index, dimensions);

    // Introduce controlled mutations
    let mutation_sets: Vec<(&str, Vec<usize>)> = vec![
        ("1 SNP (pos 100)", vec![100]),
        ("3 SNPs (pos 100,200,300)", vec![100, 200, 300]),
        ("10 SNPs (scattered)", vec![50, 100, 150, 200, 250, 300, 350, 400, 450, 500]),
        ("25 SNPs (dense cluster)", (100..125).collect()),
        ("50 SNPs (bp 0-49)", (0..50).collect()),
    ];

    println!(
        "  Reference: PhiX174 ({} bp)",
        PHIX174.chars().filter(|c| "ACGTacgt".contains(*c)).count()
    );
    println!();
    println!(
        "  {:35} | {:>12} | {:>12} | {:>10}",
        "Mutation Pattern", "Cosine Dist", "Euclid Dist", "Sensitivity"
    );
    println!("  {}", "-".repeat(80));

    for (label, positions) in &mutation_sets {
        let mutated = mutate_sequence(PHIX174, positions);
        let mutated_emb = sequence_to_embedding(&mutated, k, kmer_index, dimensions);

        let cos_dist = cosine_distance(&reference_emb, &mutated_emb);
        let euc_dist = euclidean_distance(&reference_emb, &mutated_emb);

        // Sensitivity classification
        let sensitivity = if cos_dist < 0.001 {
            "Below threshold"
        } else if cos_dist < 0.01 {
            "Low"
        } else if cos_dist < 0.05 {
            "Medium"
        } else {
            "High"
        };

        println!(
            "  {:35} | {:>12.8} | {:>12.8} | {:>10}",
            label, cos_dist, euc_dist, sensitivity,
        );
    }

    let elapsed = start.elapsed();
    println!();
    println!("  Variant detection time: {:.2?}", elapsed);
    println!();
}

/// Stage 4: Phylogenetic distance matrix.
fn stage_phylogenetic_distance(
    embeddings: &[(&str, Vec<f32>)],
) {
    println!("========================================================");
    println!("  STAGE 4: Phylogenetic Distance Matrix (Cosine)");
    println!("========================================================");
    println!();

    let start = Instant::now();

    let n = embeddings.len();
    let names: Vec<&str> = embeddings.iter().map(|(name, _)| *name).collect();

    // Header row
    print!("  {:>15}", "");
    for name in &names {
        print!(" {:>14}", &name[..name.len().min(14)]);
    }
    println!();
    print!("  {:>15}", "");
    for _ in &names {
        print!(" {:>14}", "-".repeat(14));
    }
    println!();

    // Distance matrix
    let mut matrix = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        print!("  {:>15}", &names[i][..names[i].len().min(15)]);
        for j in 0..n {
            let dist = cosine_distance(&embeddings[i].1, &embeddings[j].1);
            matrix[i][j] = dist;
            print!(" {:>14.6}", dist);
        }
        println!();
    }

    // Find closest and farthest pairs
    let mut min_dist = f32::MAX;
    let mut max_dist = 0.0f32;
    let mut min_pair = (0, 0);
    let mut max_pair = (0, 0);

    for i in 0..n {
        for j in (i + 1)..n {
            if matrix[i][j] < min_dist {
                min_dist = matrix[i][j];
                min_pair = (i, j);
            }
            if matrix[i][j] > max_dist {
                max_dist = matrix[i][j];
                max_pair = (i, j);
            }
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!(
        "  Most similar pair:  {} <-> {} (distance: {:.6})",
        names[min_pair.0], names[min_pair.1], min_dist,
    );
    println!(
        "  Most distant pair:  {} <-> {} (distance: {:.6})",
        names[max_pair.0], names[max_pair.1], max_dist,
    );
    println!();

    // Euclidean distance matrix
    println!("  --- Euclidean Distance Matrix ---");
    println!();
    print!("  {:>15}", "");
    for name in &names {
        print!(" {:>14}", &name[..name.len().min(14)]);
    }
    println!();
    print!("  {:>15}", "");
    for _ in &names {
        print!(" {:>14}", "-".repeat(14));
    }
    println!();

    for i in 0..n {
        print!("  {:>15}", &names[i][..names[i].len().min(15)]);
        for j in 0..n {
            let dist = euclidean_distance(&embeddings[i].1, &embeddings[j].1);
            print!(" {:>14.6}", dist);
        }
        println!();
    }

    println!();
    println!("  Phylogenetic analysis time: {:.2?}", elapsed);
    println!();
}

// ---------------------------------------------------------------------------
// GC-content analysis (a simple but meaningful genomic statistic)
// ---------------------------------------------------------------------------

fn gc_content(seq: &str) -> f64 {
    let upper: String = seq.chars().filter(|c| "ACGTacgt".contains(*c)).collect();
    let gc = upper.chars().filter(|c| *c == 'G' || *c == 'C' || *c == 'g' || *c == 'c').count();
    if upper.is_empty() {
        0.0
    } else {
        gc as f64 / upper.len() as f64
    }
}

fn stage_gc_analysis(references: &[(&str, &str)]) {
    println!("========================================================");
    println!("  BONUS: GC Content Analysis");
    println!("========================================================");
    println!();
    println!("  {:25} | {:>8} | {:>10}", "Organism", "GC %", "Length (bp)");
    println!("  {}", "-".repeat(50));

    for &(name, seq) in references {
        let clean: String = seq.chars().filter(|c| "ACGTacgt".contains(*c)).collect();
        let gc = gc_content(seq);
        println!("  {:25} | {:>7.2}% | {:>10}", name, gc * 100.0, clean.len());
    }
    println!();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let total_start = Instant::now();

    println!();
    println!("############################################################");
    println!("#                                                          #");
    println!("#   RuVector DNA Analysis Pipeline                         #");
    println!("#   Real genomic data from NCBI GenBank                    #");
    println!("#                                                          #");
    println!("############################################################");
    println!();

    // Configuration
    let k = 4; // 4-mer analysis (256-dimensional embeddings)
    let kmers = enumerate_kmers(k);
    let dimensions = kmers.len(); // 4^4 = 256
    let kmer_index = build_kmer_index(&kmers);

    // Reference organisms
    let references: Vec<(&str, &str)> = vec![
        ("PhiX174", PHIX174),
        ("SARS-CoV-2", SARS_COV2),
        ("E. coli K-12", ECOLI),
        ("Human mito", HUMAN_MITO),
    ];

    // Stage 1: Embed
    let embeddings = stage_embeddings(&references, k, &kmer_index, dimensions);

    // Stage 2: Species identification
    stage_species_identification(&embeddings, k, &kmer_index, dimensions);

    // Stage 3: Variant detection
    stage_variant_detection(k, &kmer_index, dimensions);

    // Stage 4: Phylogenetic distance
    stage_phylogenetic_distance(&embeddings);

    // Bonus: GC content
    stage_gc_analysis(&references);

    // Summary
    let total_elapsed = total_start.elapsed();
    println!("========================================================");
    println!("  Pipeline complete in {:.2?}", total_elapsed);
    println!("========================================================");
    println!();
}
