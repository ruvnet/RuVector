//! The delta engine: turns a window of multi-modal readings into a proof-gated
//! [`DeltaWitness`]. The pipeline is `delta → boundary → coherence → proof →
//! action` — it models *state transition*, not a fixed task label.

use crate::coherence::detect_boundary;
use crate::modality::Modality;
use crate::state::{Reading, WorldState};
use crate::witness::{evidence_hash, Action, DeltaWitness, ProofGate};
use std::collections::HashMap;

/// Configuration for [`DeltaEngine`].
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// EWMA smoothing for baselines.
    pub alpha: f32,
    /// |delta| above which a modality is considered to have responded.
    pub active_threshold: f32,
    /// Minimum historical responsiveness for a silent modality to count as a
    /// contradiction ("it usually reacts here, but didn't").
    pub responsive_min: f32,
    /// How many prior changed-zone delta vectors to remember for novelty.
    pub history_cap: usize,
    /// Proof-gate thresholds.
    pub gate: ProofGate,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            active_threshold: 0.4,
            responsive_min: 0.3,
            history_cap: 256,
            gate: ProofGate::default(),
        }
    }
}

/// Stateful physical-perception engine.
pub struct DeltaEngine {
    cfg: EngineConfig,
    state: WorldState,
    history: Vec<Vec<f64>>, // prior changed-zone delta vectors
    prev_hash: Option<String>,
}

impl DeltaEngine {
    /// Create an engine.
    pub fn new(cfg: EngineConfig) -> Self {
        let state = WorldState::new(cfg.alpha, cfg.active_threshold);
        Self {
            cfg,
            state,
            history: Vec::new(),
            prev_hash: None,
        }
    }

    /// Borrow the rolling world state (baselines, responsiveness).
    pub fn state(&self) -> &WorldState {
        &self.state
    }

    /// Observe one time window of readings and emit a proof-gated witness.
    pub fn observe(&mut self, readings: &[Reading], t: u64) -> DeltaWitness {
        // 1. Per-zone delta vectors over a fixed modality order.
        let mut by_zone: HashMap<String, HashMap<Modality, f32>> = HashMap::new();
        for r in readings {
            by_zone
                .entry(r.zone.clone())
                .or_default()
                .insert(r.modality, r.value);
        }
        let mut zones: Vec<String> = by_zone.keys().cloned().collect();
        zones.sort();

        let delta_vec = |zone: &str| -> Vec<f64> {
            Modality::ALL
                .iter()
                .map(|&m| match by_zone[zone].get(&m) {
                    Some(&v) => (v - self.state.baseline(zone, m)).abs() as f64,
                    None => 0.0,
                })
                .collect()
        };
        let deltas: Vec<(String, Vec<f64>)> =
            zones.iter().map(|z| (z.clone(), delta_vec(z))).collect();

        // 2. Boundary via coherence min-cut.
        let boundary = detect_boundary(&deltas);
        let (changed, coherence, changed_vec) = match boundary {
            Some(b) => {
                let v = deltas
                    .iter()
                    .find(|(z, _)| z == &b.zone)
                    .map(|(_, v)| v.clone());
                (b.zone, b.coherence, v.unwrap_or_default())
            }
            None => {
                let w = self.finish(readings, &deltas, t, NullWitness::empty());
                return w;
            }
        };

        // 3. Supporting / contradicting modalities in the changed zone.
        let thr = self.cfg.active_threshold;
        let mut supporting = Vec::new();
        let mut contradicting = Vec::new();
        let mut contradiction = 0.0f32;
        for &m in &Modality::ALL {
            let mag = match by_zone[&changed].get(&m) {
                Some(&v) => (v - self.state.baseline(&changed, m)).abs(),
                None => 0.0,
            };
            if mag >= thr {
                supporting.push(m);
            } else if self.state.seen(&changed, m)
                && self.state.responsiveness(&changed, m) >= self.cfg.responsive_min
            {
                // Usually reacts here, but stayed silent — first-class disagreement.
                contradicting.push(m);
                contradiction = contradiction.max(m.physics().spoof_resistance);
            }
        }

        // 4. Novelty vs prior changed-zone states.
        let novelty = self.novelty(&changed_vec);

        // 5. Proof gate -> bounded authority.
        let action = self.cfg.gate.decide(novelty, coherence, contradiction);

        let w = NullWitness {
            changed_boundary: changed,
            supporting,
            contradicting,
            novelty,
            coherence,
            contradiction,
            action,
            changed_vec: Some(changed_vec),
        };
        self.finish(readings, &deltas, t, w)
    }

    fn novelty(&self, vec: &[f64]) -> f32 {
        if self.history.is_empty() {
            return 1.0;
        }
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let dist = |a: &[f64], b: &[f64]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt()
        };
        let min_d = self
            .history
            .iter()
            .map(|h| dist(h, vec))
            .fold(f64::INFINITY, f64::min);
        (min_d / (norm(vec) + 1e-9)).clamp(0.0, 1.0) as f32
    }

    /// Hash evidence, build the witness, then fold the readings into state and
    /// remember the changed vector for future novelty.
    fn finish(
        &mut self,
        readings: &[Reading],
        deltas: &[(String, Vec<f64>)],
        t: u64,
        w: NullWitness,
    ) -> DeltaWitness {
        // Canonical raw + feature bytes for the evidence chain.
        let mut raw = String::new();
        let mut sorted: Vec<&Reading> = readings.iter().collect();
        sorted.sort_by(|a, b| {
            (a.zone.as_str(), a.modality.name()).cmp(&(b.zone.as_str(), b.modality.name()))
        });
        for r in sorted {
            raw.push_str(&format!("{}:{}:{:.6};", r.zone, r.modality.name(), r.value));
        }
        let mut feat = String::new();
        for (z, v) in deltas {
            feat.push_str(z);
            for x in v {
                feat.push_str(&format!(":{x:.6}"));
            }
            feat.push(';');
        }
        let hash = evidence_hash(
            raw.as_bytes(),
            feat.as_bytes(),
            &w.changed_boundary,
            w.novelty,
            w.coherence,
            w.contradiction,
            w.action,
            self.prev_hash.as_deref(),
        );

        let witness = DeltaWitness {
            t,
            changed_boundary: w.changed_boundary,
            supporting_modalities: w.supporting,
            contradicting_modalities: w.contradicting,
            novelty: w.novelty,
            coherence: w.coherence,
            contradiction: w.contradiction,
            action: w.action,
            evidence_hash: hash.clone(),
            prev_hash: self.prev_hash.take(),
        };
        self.prev_hash = Some(hash);

        // Remember the changed vector (compress: only store meaningful events).
        if let Some(v) = w.changed_vec {
            if v.iter().any(|&x| x as f32 >= self.cfg.active_threshold) {
                self.history.push(v);
                if self.history.len() > self.cfg.history_cap {
                    self.history.remove(0);
                }
            }
        }

        // Fold readings into the rolling baselines.
        for r in readings {
            self.state.update(r);
        }
        witness
    }
}

/// Internal scratch for an in-progress witness.
struct NullWitness {
    changed_boundary: String,
    supporting: Vec<Modality>,
    contradicting: Vec<Modality>,
    novelty: f32,
    coherence: f32,
    contradiction: f32,
    action: Action,
    changed_vec: Option<Vec<f64>>,
}

impl NullWitness {
    fn empty() -> Self {
        Self {
            changed_boundary: String::new(),
            supporting: Vec::new(),
            contradicting: Vec::new(),
            novelty: 0.0,
            coherence: 0.0,
            contradiction: 0.0,
            action: Action::Ignore,
            changed_vec: None,
        }
    }
}
