//! SONA online LoRA adaptation — ADR-183 Tier 3 iter 19.
//!
//! Self-Organising Neural Adapter. Incrementally fine-tunes the per-room
//! LoRA adapter from live CSI vitals, using triplet loss with hard-negative
//! mining over a class-balanced embedding bank.
//!
//! ## Algorithm
//!
//! 1. Classify each incoming `VitalReading` into one of 5 classes
//!    {absent, resting, exercising, sleeping, stressed} using rule-based
//!    heuristics on breathing/HR/motion.
//! 2. Store the raw 8-dim feature vector in a per-class circular buffer
//!    (capacity `BANK_CAP` per class).
//! 3. After accumulating `WARMUP_SAMPLES` total, run a mini-batch triplet-loss
//!    gradient step every `STEP_EVERY` new samples:
//!      - anchor: a random sample from a random class c_a
//!      - positive: another sample from c_a
//!      - hard-negative: sample from the class c_n ≠ c_a whose centroid is
//!        closest to the anchor embedding
//!    Gradient flows through the LoRA `apply()` operation (two small matmuls).
//! 4. Adam update on `loraB` and `loraA` (lr=1e-4, β₁=0.9, β₂=0.999).
//! 5. Persist updated adapter JSON every `SAVE_EVERY` steps.

#[cfg(feature = "csi-embed")]
pub use inner::SonaAdapter;

#[cfg(feature = "csi-embed")]
mod inner {
    use std::path::{Path, PathBuf};
    use ruvector_hailo::{CsiEmbedderCpu, CsiFeatures, CsiLoraAdapter, LORA_RANK, CSI_EMBED_DIM};
    use crate::types::VitalReading;

    const BANK_CAP: usize = 64;
    const WARMUP_SAMPLES: usize = 50;
    const STEP_EVERY: usize = 10;
    const SAVE_EVERY: usize = 100;
    const MARGIN: f32 = 0.2;
    const LR: f32 = 1e-4;
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPS: f32 = 1e-8;

    const N_CLASSES: usize = 5;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Class { Absent = 0, Resting = 1, Sleeping = 2, Exercising = 3, Stressed = 4 }

    impl Class {
        fn from_vitals(r: &VitalReading) -> Self {
            let hr = r.heart_rate.value_bpm as f32;
            let br = r.breathing.value_bpm as f32;
            if hr < 20.0 && br < 4.0 {
                return Class::Absent;
            }
            if hr > 100.0 {
                return Class::Exercising;
            }
            if hr > 90.0 {
                return Class::Stressed;
            }
            if br > 0.0 && br < 14.0 && hr < 65.0 {
                return Class::Sleeping;
            }
            Class::Resting
        }

        fn idx(self) -> usize { self as usize }
    }

    /// Adam optimizer state for one parameter matrix stored as a flat Vec.
    struct AdamState {
        m: Vec<f32>,
        v: Vec<f32>,
        step: u64,
    }

    impl AdamState {
        fn new(size: usize) -> Self {
            Self { m: vec![0.0; size], v: vec![0.0; size], step: 0 }
        }

        fn update(&mut self, params: &mut [f32], grad: &[f32]) {
            self.step += 1;
            let t = self.step as f32;
            let bc1 = 1.0 - BETA1.powf(t);
            let bc2 = 1.0 - BETA2.powf(t);
            for i in 0..params.len() {
                self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * grad[i];
                self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * grad[i] * grad[i];
                let m_hat = self.m[i] / bc1;
                let v_hat = self.v[i] / bc2;
                params[i] -= LR * m_hat / (v_hat.sqrt() + EPS);
            }
        }
    }

    /// SONA online LoRA adapter — owns the adapter weights and updates them.
    pub struct SonaAdapter {
        embedder: CsiEmbedderCpu,
        lora_a: Vec<f32>,
        lora_b: Vec<f32>,
        scaling: f32,
        adam_a: AdamState,
        adam_b: AdamState,
        banks: [Vec<[f32; 8]>; N_CLASSES],
        total_samples: usize,
        samples_since_step: usize,
        steps: usize,
        adapter_path: PathBuf,
    }

    impl SonaAdapter {
        /// Load base model + LoRA adapter from disk.
        pub fn load(model_path: &Path, adapter_path: &Path) -> Result<Self, String> {
            let adapter = CsiLoraAdapter::load(adapter_path)
                .map_err(|e| format!("load LoRA: {e:?}"))?;
            let embedder = CsiEmbedderCpu::open(model_path)
                .map_err(|e| format!("load model: {e:?}"))?;

            let (lora_a, lora_b, scaling) = adapter.into_parts();
            let a_len = lora_a.len();
            let b_len = lora_b.len();

            Ok(Self {
                embedder,
                lora_a,
                lora_b,
                scaling,
                adam_a: AdamState::new(a_len),
                adam_b: AdamState::new(b_len),
                banks: std::array::from_fn(|_| Vec::with_capacity(BANK_CAP)),
                total_samples: 0,
                samples_since_step: 0,
                steps: 0,
                adapter_path: adapter_path.to_path_buf(),
            })
        }

        /// Apply current LoRA to a base embedding.
        fn apply_lora(&self, emb: &[f32; CSI_EMBED_DIM]) -> [f32; CSI_EMBED_DIM] {
            // intermediate = loraB @ emb  →  [LORA_RANK]
            let mut inter = [0f32; LORA_RANK];
            for j in 0..LORA_RANK {
                let off = j * CSI_EMBED_DIM;
                for k in 0..CSI_EMBED_DIM {
                    inter[j] += self.lora_b[off + k] * emb[k];
                }
            }
            // delta = loraA @ inter  →  [CSI_EMBED_DIM]
            let mut out = *emb;
            for i in 0..CSI_EMBED_DIM {
                let off = i * LORA_RANK;
                let mut d = 0f32;
                for j in 0..LORA_RANK {
                    d += self.lora_a[off + j] * inter[j];
                }
                out[i] = emb[i] + self.scaling * d;
            }
            l2_norm(&mut out);
            out
        }

        /// Push a new reading; may trigger a gradient step.
        pub fn push(&mut self, reading: &VitalReading) {
            let class = Class::from_vitals(reading);
            let features = vitals_to_features(reading);
            let bank = &mut self.banks[class.idx()];
            if bank.len() >= BANK_CAP {
                bank.remove(0);
            }
            bank.push(features);
            self.total_samples += 1;
            self.samples_since_step += 1;

            if self.total_samples >= WARMUP_SAMPLES && self.samples_since_step >= STEP_EVERY {
                self.gradient_step();
                self.samples_since_step = 0;
                self.steps += 1;
                if self.steps % SAVE_EVERY == 0 {
                    if let Err(e) = self.save() {
                        tracing::warn!("sona: save adapter failed: {e}");
                    }
                }
            }
        }

        /// Current embedding for a feature vector (base + current LoRA).
        pub fn embed(&self, features: &CsiFeatures) -> [f32; CSI_EMBED_DIM] {
            let base = self.embedder.embed(features);
            self.apply_lora(&base)
        }

        fn gradient_step(&mut self) {
            // Select anchor class (must have ≥ 2 samples)
            let anchor_class = match (0..N_CLASSES).find(|&c| self.banks[c].len() >= 2) {
                Some(c) => c,
                None => return,
            };
            let bank_a = &self.banks[anchor_class];
            let anchor_feat = bank_a[bank_a.len() / 2];
            let pos_feat = bank_a[bank_a.len() - 1];

            // Embed anchor and positive
            let anchor_base = self.embedder.embed(&arr_to_features(anchor_feat));
            let pos_base = self.embedder.embed(&arr_to_features(pos_feat));
            let anchor_emb = self.apply_lora(&anchor_base);
            let pos_emb = self.apply_lora(&pos_base);

            // Hard-negative: class whose centroid is closest to anchor
            let neg_class = (0..N_CLASSES)
                .filter(|&c| c != anchor_class && !self.banks[c].is_empty())
                .min_by(|&a, &b| {
                    let da = centroid_dist(&self.banks[a], &anchor_emb, self);
                    let db = centroid_dist(&self.banks[b], &anchor_emb, self);
                    da.partial_cmp(&db).unwrap()
                });

            let neg_class = match neg_class {
                Some(c) => c,
                None => return,
            };
            let bank_n = &self.banks[neg_class];
            let neg_feat = bank_n[bank_n.len() / 2];
            let neg_base = self.embedder.embed(&arr_to_features(neg_feat));
            let neg_emb = self.apply_lora(&neg_base);

            // Triplet loss: L = max(0, d(a,p) - d(a,n) + margin)
            let d_ap = 1.0 - dot(&anchor_emb, &pos_emb);
            let d_an = 1.0 - dot(&anchor_emb, &neg_emb);
            let loss = (d_ap - d_an + MARGIN).max(0.0);
            if loss == 0.0 { return; }

            // Gradient w.r.t. loraB (via anchor embedding path only for simplicity)
            // dL/d(loraB[j,k]) = -dL/d_an * d(d_an)/d(anchor_emb[i]) * d(anchor_emb)/d(delta[i])
            //                                              * d(delta[i])/d(inter[j]) * d(inter[j])/d(loraB[j,k])
            // Simplified: update loraB using approximate gradient via outer product
            let grad_anchor_from_neg: [f32; CSI_EMBED_DIM] = {
                let mut g = [0f32; CSI_EMBED_DIM];
                for i in 0..CSI_EMBED_DIM {
                    // d(d_an)/d(anchor_emb[i]) = -neg_emb[i] (after L2 norm, approximate)
                    g[i] = neg_emb[i];
                }
                g
            };

            // Backprop through apply_lora for anchor
            let (grad_lora_b, grad_lora_a) =
                self.backprop_lora(&anchor_base, &grad_anchor_from_neg);

            self.adam_b.update(&mut self.lora_b, &grad_lora_b);
            self.adam_a.update(&mut self.lora_a, &grad_lora_a);
        }

        /// Backpropagate a gradient on the output embedding through the LoRA apply op.
        /// Returns (grad_loraB, grad_loraA).
        fn backprop_lora(
            &self,
            emb: &[f32; CSI_EMBED_DIM],
            grad_out: &[f32; CSI_EMBED_DIM],
        ) -> (Vec<f32>, Vec<f32>) {
            // Forward: inter = loraB @ emb; delta = loraA @ inter; out = emb + s*delta
            let mut inter = [0f32; LORA_RANK];
            for j in 0..LORA_RANK {
                let off = j * CSI_EMBED_DIM;
                for k in 0..CSI_EMBED_DIM {
                    inter[j] += self.lora_b[off + k] * emb[k];
                }
            }

            // grad_delta = scaling * grad_out  (delta appears additively before norm)
            let mut grad_delta = [0f32; CSI_EMBED_DIM];
            for i in 0..CSI_EMBED_DIM {
                grad_delta[i] = self.scaling * grad_out[i];
            }

            // grad_loraA[i,j] = grad_delta[i] * inter[j]
            let mut grad_a = vec![0f32; CSI_EMBED_DIM * LORA_RANK];
            for i in 0..CSI_EMBED_DIM {
                let off = i * LORA_RANK;
                for j in 0..LORA_RANK {
                    grad_a[off + j] = grad_delta[i] * inter[j];
                }
            }

            // grad_inter = loraA^T @ grad_delta
            let mut grad_inter = [0f32; LORA_RANK];
            for j in 0..LORA_RANK {
                for i in 0..CSI_EMBED_DIM {
                    grad_inter[j] += self.lora_a[i * LORA_RANK + j] * grad_delta[i];
                }
            }

            // grad_loraB[j,k] = grad_inter[j] * emb[k]
            let mut grad_b = vec![0f32; LORA_RANK * CSI_EMBED_DIM];
            for j in 0..LORA_RANK {
                let off = j * CSI_EMBED_DIM;
                for k in 0..CSI_EMBED_DIM {
                    grad_b[off + k] = grad_inter[j] * emb[k];
                }
            }

            (grad_b, grad_a)
        }

        fn save(&self) -> Result<(), String> {
            use std::io::Write as _;
            let mut out = String::with_capacity(64 * 1024);
            out.push_str("{\"config\":{\"rank\":");
            out.push_str(&LORA_RANK.to_string());
            out.push_str(",\"alpha\":");
            out.push_str(&(LORA_RANK * 2).to_string());
            out.push_str("},\"inputDim\":");
            out.push_str(&CSI_EMBED_DIM.to_string());
            out.push_str(",\"outputDim\":");
            out.push_str(&CSI_EMBED_DIM.to_string());
            out.push_str(",\"sona\":{\"step\":");
            out.push_str(&self.steps.to_string());
            out.push_str(",\"lr\":1e-4,\"beta1\":0.9,\"beta2\":0.999},\"weights\":{\"loraA\":");
            push_matrix_flat(&mut out, &self.lora_a, CSI_EMBED_DIM, LORA_RANK);
            out.push_str(",\"loraB\":");
            push_matrix_flat(&mut out, &self.lora_b, LORA_RANK, CSI_EMBED_DIM);
            out.push_str(",\"scaling\":");
            out.push_str(&format!("{:.1}", self.scaling));
            out.push_str("}}");

            let tmp = self.adapter_path.with_extension("json.tmp");
            let mut f = std::fs::File::create(&tmp)
                .map_err(|e| format!("create tmp: {e}"))?;
            f.write_all(out.as_bytes()).map_err(|e| format!("write: {e}"))?;
            drop(f);
            std::fs::rename(&tmp, &self.adapter_path)
                .map_err(|e| format!("rename: {e}"))?;
            tracing::info!(
                "sona: adapter saved step={} path={}",
                self.steps,
                self.adapter_path.display()
            );
            Ok(())
        }

        pub fn steps(&self) -> usize { self.steps }
        pub fn total_samples(&self) -> usize { self.total_samples }
    }

    fn l2_norm(v: &mut [f32; CSI_EMBED_DIM]) {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in v.iter_mut() { *x /= norm; }
    }

    fn dot(a: &[f32; CSI_EMBED_DIM], b: &[f32; CSI_EMBED_DIM]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn vitals_to_features(r: &VitalReading) -> [f32; 8] {
        [
            (r.breathing.value_bpm as f32 / 30.0).clamp(0.0, 1.0),
            r.breathing.confidence as f32,
            (r.heart_rate.value_bpm as f32 / 120.0).clamp(0.0, 1.0),
            r.heart_rate.confidence as f32,
            0.0_f32, // motion_score not tracked at this tier
            (r.snr_db / 40.0).clamp(0.0, 1.0),
            0.0_f32, // peak_amp_breathing not tracked at this tier
            0.0_f32, // peak_amp_hr not tracked at this tier
        ]
    }

    fn arr_to_features(a: [f32; 8]) -> CsiFeatures {
        CsiFeatures {
            breathing_bpm_norm: a[0],
            breathing_confidence: a[1],
            heart_rate_bpm_norm: a[2],
            heart_rate_confidence: a[3],
            motion_score: a[4],
            log_snr_norm: a[5],
            peak_amp_breathing_norm: a[6],
            peak_amp_hr_norm: a[7],
        }
    }

    fn centroid_dist(bank: &[[f32; 8]], anchor: &[f32; CSI_EMBED_DIM], sona: &SonaAdapter) -> f32 {
        if bank.is_empty() { return f32::MAX; }
        let mut centroid = [0f32; CSI_EMBED_DIM];
        for feat in bank {
            let base = sona.embedder.embed(&arr_to_features(*feat));
            let emb = sona.apply_lora(&base);
            for i in 0..CSI_EMBED_DIM { centroid[i] += emb[i]; }
        }
        let n = bank.len() as f32;
        for v in &mut centroid { *v /= n; }
        1.0 - dot(anchor, &centroid)
    }

    fn push_matrix_flat(out: &mut String, flat: &[f32], rows: usize, cols: usize) {
        out.push('[');
        for r in 0..rows {
            out.push('[');
            for c in 0..cols {
                let v = flat[r * cols + c];
                if v == 0.0 {
                    out.push_str("0.0");
                } else {
                    out.push_str(&format!("{v:.8e}"));
                }
                if c + 1 < cols { out.push(','); }
            }
            out.push(']');
            if r + 1 < rows { out.push(','); }
        }
        out.push(']');
    }
}
