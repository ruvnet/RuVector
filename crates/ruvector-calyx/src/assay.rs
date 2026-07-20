//! Signal density — "how much does this lens actually tell us" (the paper's
//! *Assay*).
//!
//! "Differentiate" — the third of Calyx's four verbs — measures which lenses
//! add information rather than assuming every model is useful. We estimate the
//! mutual information (in bits) between a lens's similarity score and a binary
//! relevance label, then divide by the lens's declared cost to get *signal
//! density*: information gained per unit of compute. The anneal optimizer
//! promotes high-density lenses.

/// One (lens-similarity, was-this-pair-relevant) observation.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub similarity: f32,
    pub relevant: bool,
}

/// Estimated information content of a lens and its economics.
#[derive(Debug, Clone)]
pub struct LensSignal {
    pub lens: String,
    /// Mutual information between similarity and relevance, in bits (>= 0).
    pub bits: f32,
    /// Declared measurement cost in microseconds.
    pub cost_us: f32,
    /// Signal density: bits per microsecond (information per dollar/ms/watt).
    pub density: f32,
}

/// Estimate mutual information (bits) between a lens's similarity score and a
/// binary relevance label, by binning similarity into `bins` buckets.
///
/// Returns `0.0` for an empty/degenerate sample. The estimate is a lower bound
/// proxy for the paper's MI-based contribution measure; it is monotone in how
/// well the lens separates relevant from irrelevant pairs.
pub fn mutual_information_bits(obs: &[Observation], bins: usize) -> f32 {
    let n = obs.len();
    if n == 0 || bins == 0 {
        return 0.0;
    }
    let bins = bins.max(2);
    // Similarity is cosine in [-1, 1]; map to [0, 1] then to a bin index.
    let bin_of = |s: f32| -> usize {
        let t = ((s + 1.0) * 0.5).clamp(0.0, 0.999_999);
        (t * bins as f32) as usize
    };

    let mut joint = vec![[0u32; 2]; bins]; // joint[bin][label]
    let mut p_label = [0u32; 2];
    let mut p_bin = vec![0u32; bins];
    for o in obs {
        let b = bin_of(o.similarity);
        let l = o.relevant as usize;
        joint[b][l] += 1;
        p_label[l] += 1;
        p_bin[b] += 1;
    }

    let nf = n as f32;
    let mut mi = 0.0f32;
    for b in 0..bins {
        for l in 0..2 {
            let c = joint[b][l];
            if c == 0 {
                continue;
            }
            let pxy = c as f32 / nf;
            let px = p_bin[b] as f32 / nf;
            let py = p_label[l] as f32 / nf;
            if px > 0.0 && py > 0.0 {
                mi += pxy * (pxy / (px * py)).log2();
            }
        }
    }
    mi.max(0.0)
}

/// Compute signal density for one lens.
pub fn signal_density(lens: impl Into<String>, obs: &[Observation], cost_us: f32) -> LensSignal {
    let bits = mutual_information_bits(obs, 8);
    let density = if cost_us > 0.0 { bits / cost_us } else { 0.0 };
    LensSignal {
        lens: lens.into(),
        bits,
        cost_us,
        density,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn informative_lens_beats_random_lens() {
        // Perfectly separating lens: relevant pairs always high similarity.
        let mut informative = Vec::new();
        let mut random = Vec::new();
        for i in 0..200 {
            let relevant = i % 2 == 0;
            informative.push(Observation {
                similarity: if relevant { 0.9 } else { -0.9 },
                relevant,
            });
            // Random lens: similarity uncorrelated with label.
            random.push(Observation {
                similarity: if i % 3 == 0 { 0.5 } else { -0.5 },
                relevant,
            });
        }
        let inf = signal_density("good", &informative, 2.0);
        let rnd = signal_density("noise", &random, 2.0);
        assert!(inf.bits > 0.9, "informative bits = {}", inf.bits);
        assert!(rnd.bits < inf.bits);
        assert!(inf.density > rnd.density);
    }
}
