//! Deterministic synthetic agent session: a sequence of topic "plateaus"
//! (long stretches where the context barely moves) separated by short,
//! sharp topic-switch transitions. One memory is written per step, embedded
//! near that step's context.

use emergent_time::entropy::entropy_from_spectrum;
use emergent_time::StateSnapshot;

/// Deterministic xorshift64* PRNG (no external RNG dependency), mirroring the
/// private `Rng` already used in `emergent_time::structural_clock`'s own test
/// scenario generator.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    /// Next value in `[-1, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((v >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    pub fn next_vec(&mut self, dim: usize) -> Vec<f64> {
        (0..dim).map(|_| self.next_f64()).collect()
    }
}

pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

fn normalize(v: &mut [f64]) {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n > 1e-12 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn lerp_vec(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + t * (y - x)).collect()
}

fn softmax(xs: &[f64], temp: f64) -> Vec<f64> {
    let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| ((x - m) / temp).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

#[derive(Clone, Copy, Debug)]
pub struct ScenarioConfig {
    pub dim: usize,
    pub n_topics: usize,
    /// Steps spent on each topic before the next switch.
    pub plateau_len: usize,
    /// Steps over which a topic switch is ramped (linear interpolation).
    pub switch_width: usize,
    pub context_noise: f64,
    pub memory_noise: f64,
    /// Softmax temperature for the entropy channel.
    pub entropy_temp: f64,
    pub seed: u64,
}

impl Default for ScenarioConfig {
    fn default() -> Self {
        ScenarioConfig {
            dim: 32,
            n_topics: 4,
            plateau_len: 60,
            switch_width: 3,
            // Small relative to a topic-switch jump (~sqrt(2) between two
            // near-orthogonal unit centroids at dim=32) so a quiet plateau's
            // accumulated embedding movement stays well below one switch's,
            // even summed over a long (150-step) plateau. This is what makes
            // the structural clock actually behave differently from a
            // reparametrized wall clock — see crate docs.
            context_noise: 0.001,
            memory_noise: 0.08,
            entropy_temp: 0.25,
            seed: 0xC0FFEE,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MemoryItem {
    pub id: usize,
    pub embedding: Vec<f64>,
    pub write_step: usize,
    pub topic: usize,
}

#[derive(Clone, Debug)]
pub struct Session {
    pub contexts: Vec<Vec<f64>>,
    pub topics: Vec<Vec<f64>>,
    pub memories: Vec<MemoryItem>,
    pub snapshots: Vec<StateSnapshot>,
    pub topic_of_step: Vec<usize>,
}

impl Session {
    pub fn total_steps(&self) -> usize {
        self.contexts.len()
    }
}

/// Build one deterministic session from `cfg`. Topic centroids are random
/// unit vectors (no explicit separation construction: at `dim >= 16` random
/// vectors are already near-orthogonal in expectation). Topics are visited in
/// a fixed order `0, 1, ..., n_topics-1`, each held for `plateau_len` steps.
pub fn generate_session(cfg: &ScenarioConfig) -> Session {
    let mut rng = Rng::new(cfg.seed);
    let topics: Vec<Vec<f64>> = (0..cfg.n_topics)
        .map(|_| {
            let mut v = rng.next_vec(cfg.dim);
            normalize(&mut v);
            v
        })
        .collect();

    let total_steps = cfg.n_topics * cfg.plateau_len;
    let mut contexts = Vec::with_capacity(total_steps);
    let mut topic_of_step = Vec::with_capacity(total_steps);

    for i in 0..total_steps {
        let b = (i / cfg.plateau_len).min(cfg.n_topics - 1);
        let lp = i % cfg.plateau_len;
        let target = if b == 0 || lp >= cfg.switch_width {
            topics[b].clone()
        } else {
            let frac = (lp + 1) as f64 / cfg.switch_width as f64;
            lerp_vec(&topics[b - 1], &topics[b], frac)
        };
        let noise = rng.next_vec(cfg.dim);
        let embedding: Vec<f64> = target
            .iter()
            .zip(&noise)
            .map(|(t, n)| t + cfg.context_noise * n)
            .collect();
        contexts.push(embedding);
        topic_of_step.push(b);
    }

    let memories: Vec<MemoryItem> = (0..total_steps)
        .map(|i| {
            let noise = rng.next_vec(cfg.dim);
            let embedding: Vec<f64> = contexts[i]
                .iter()
                .zip(&noise)
                .map(|(c, n)| c + cfg.memory_noise * n)
                .collect();
            MemoryItem {
                id: i,
                embedding,
                write_step: i,
                topic: topic_of_step[i],
            }
        })
        .collect();

    let snapshots = build_snapshots(&contexts, &topics, cfg.entropy_temp);

    Session {
        contexts,
        topics,
        memories,
        snapshots,
        topic_of_step,
    }
}

/// Derive each step's `StateSnapshot`: `embedding` is the raw context vector
/// (`Δv` channel); `entropy` is the Shannon entropy in nats of the softmax
/// over cosine similarities from the context to every topic centroid (`ΔS`
/// channel) — high when the context sits between topics (a switch), low when
/// it sits close to a single topic (mid-plateau). This is a genuine derived
/// quantity from the actual trajectory (analogous to a topic classifier's
/// confidence), not a fabricated curve. `coherence`/`graph`/`pred_error` are
/// left at `0.0`: every clock instantiated in this crate weights those
/// channels at zero, so their value is inert here — no honest signal for them
/// exists in this harness (see crate docs).
pub fn build_snapshots(
    contexts: &[Vec<f64>],
    topics: &[Vec<f64>],
    entropy_temp: f64,
) -> Vec<StateSnapshot> {
    contexts
        .iter()
        .map(|c| {
            let sims: Vec<f64> = topics.iter().map(|t| cosine(c, t)).collect();
            let probs = softmax(&sims, entropy_temp);
            let entropy = entropy_from_spectrum(&probs);
            StateSnapshot::new(c.clone(), entropy, 0.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topics_are_unit_length() {
        let cfg = ScenarioConfig::default();
        let session = generate_session(&cfg);
        for t in &session.topics {
            let n: f64 = t.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-9, "topic norm {n}");
        }
    }

    #[test]
    fn session_has_one_memory_per_step() {
        let cfg = ScenarioConfig::default();
        let session = generate_session(&cfg);
        assert_eq!(session.memories.len(), session.total_steps());
        assert_eq!(session.total_steps(), cfg.n_topics * cfg.plateau_len);
    }

    #[test]
    fn entropy_spikes_at_switch_and_settles_mid_plateau() {
        // At a topic switch, the context sits between two topic centroids so
        // similarity is split -> higher entropy. Mid-plateau it is close to
        // one centroid -> lower entropy. This is the discriminating property
        // the entropy channel is supposed to have; if it didn't hold, using
        // it as a `ΔS` signal would be pointless.
        let cfg = ScenarioConfig {
            plateau_len: 40,
            switch_width: 3,
            ..ScenarioConfig::default()
        };
        let session = generate_session(&cfg);
        let switch_step = cfg.plateau_len + 1; // inside the first switch ramp
        let mid_plateau_step = cfg.plateau_len + 20; // deep into topic 1's plateau
        assert!(
            session.snapshots[switch_step].entropy > session.snapshots[mid_plateau_step].entropy,
            "switch entropy {} should exceed mid-plateau entropy {}",
            session.snapshots[switch_step].entropy,
            session.snapshots[mid_plateau_step].entropy
        );
    }

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-9);
    }
}
