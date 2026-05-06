//! Brain POST shim — fan vital-sign summaries to the cognitum-v0
//! brain at `http://cognitum-v0:9876/memories`.
//!
//! Reuses RuView's `brain_bridge.rs` shape (`{category, content}`)
//! verbatim — no new schema. ADR-183 §"Open questions" #2.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;

use crate::error::Result;
use crate::state::WorkerState;
use crate::types::{VitalReading, VitalStatus};

/// JSON body POSTed to `/memories`.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryPost {
    pub category: String,
    pub content: String,
}

/// Reqwest-backed client. Cheap to clone (`reqwest::Client` is `Arc`-
/// like internally).
#[derive(Debug, Clone)]
pub struct BrainClient {
    http: reqwest::Client,
    base_url: String,
    node_name: String,
}

impl BrainClient {
    pub fn new(base_url: String, node_name: String) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .user_agent(concat!(
                "ruview-vitals-worker/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()?;
        Ok(Self {
            http,
            base_url,
            node_name,
        })
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    #[must_use]
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// POST `{category, content}` to `<base_url>/memories`. 5 s
    /// timeout; surfaces non-2xx responses as [`crate::Error::Http`].
    pub async fn post_memory(&self, category: &str, content: &str) -> Result<()> {
        let payload = MemoryPost {
            category: category.to_string(),
            content: content.to_string(),
        };
        self.http
            .post(format!("{}/memories", self.base_url))
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

/// Build the natural-language summary for one reading. Format mirrors
/// the iter-123 telemetry bridge's pattern so cluster-side cosine
/// search can treat both bridges' outputs uniformly.
#[must_use]
pub fn format_vitals_summary(reading: &VitalReading, node_name: &str) -> String {
    if reading.status == VitalStatus::Unavailable {
        return format!(
            "wifi vitals node {} on {}: pipeline warmup or no signal (status unavailable, snr {:.1} dB)",
            reading.node_id, node_name, reading.snr_db
        );
    }
    format!(
        "wifi vitals node {} on {}: breathing {:.1} bpm (conf {:.0}%) heart rate {:.1} bpm \
         (conf {:.0}%) snr {:.1} dB status {}",
        reading.node_id,
        node_name,
        reading.breathing.value_bpm,
        reading.breathing.confidence * 100.0,
        reading.heart_rate.value_bpm,
        reading.heart_rate.confidence * 100.0,
        reading.snr_db,
        status_label(reading.status),
    )
}

/// Lowercase, hyphen-free label — keeps the embed text stable.
const fn status_label(s: VitalStatus) -> &'static str {
    match s {
        VitalStatus::Valid => "valid",
        VitalStatus::Degraded => "degraded",
        VitalStatus::Unreliable => "unreliable",
        VitalStatus::Unavailable => "unavailable",
    }
}

/// Serialise a 128-dim embedding and POST it as "spatial-csi-embedding".
#[cfg(feature = "csi-embed")]
async fn post_csi_embedding(
    client: &BrainClient,
    state: &Arc<WorkerState>,
    reading: &VitalReading,
    embedding: &[f32; 128],
) {
    let mut buf = String::with_capacity(128 * 12);
    buf.push('[');
    for (i, v) in embedding.iter().enumerate() {
        if i > 0 { buf.push(','); }
        buf.push_str(&format!("{v:.6}"));
    }
    buf.push(']');
    let content = format!(
        "node_id={} node={} embedding={}",
        reading.node_id, state.config.node_name, buf
    );
    match client.post_memory("spatial-csi-embedding", &content).await {
        Ok(()) => {
            state.stats.brain_posts_ok.fetch_add(1, Ordering::Relaxed);
            tracing::info!(node_id = reading.node_id, "POST spatial-csi-embedding ok");
        }
        Err(e) => {
            state.stats.brain_posts_failed.fetch_add(1, Ordering::Relaxed);
            tracing::warn!(error = %e, node_id = reading.node_id, "POST spatial-csi-embedding failed");
        }
    }
}

/// Build the 8-element normalised feature vector from a `VitalReading`.
/// Normalisation constants match those documented in `csi_embedder.rs`.
#[cfg(feature = "csi-embed")]
fn reading_to_csi_features(r: &VitalReading) -> ruvector_hailo::CsiFeatures {
    ruvector_hailo::CsiFeatures {
        breathing_bpm_norm:      (r.breathing.value_bpm  as f32 / 30.0).clamp(0.0, 1.0),
        breathing_confidence:    r.breathing.confidence  as f32,
        heart_rate_bpm_norm:     (r.heart_rate.value_bpm as f32 / 120.0).clamp(0.0, 1.0),
        heart_rate_confidence:   r.heart_rate.confidence as f32,
        motion_score:            0.0_f32, // not tracked at this worker tier
        log_snr_norm:            (r.snr_db as f32 / 40.0).clamp(0.0, 1.0),
        peak_amp_breathing_norm: r.breathing.confidence  as f32,
        peak_amp_hr_norm:        r.heart_rate.confidence as f32,
    }
}

/// Periodic loop: every `interval`, snapshot the latest readings and
/// POST a memory per node. Runs until cancelled (i.e. forever for the
/// worker; used as `tokio::spawn(run_brain_loop(...))`).
///
/// The loop never panics — POST failures are counted in
/// [`crate::state::WorkerStats::brain_posts_failed`] and surfaced via
/// `GetStats`.
pub async fn run_brain_loop(client: BrainClient, state: Arc<WorkerState>, interval: Duration) {
    tracing::info!(
        url = client.base_url(),
        node = client.node_name(),
        interval_secs = interval.as_secs(),
        "brain loop starting"
    );

    // ADR-183 iter 19: SONA online LoRA adapter (preferred when lora_path is set).
    // Falls back to static CsiEmbedderCpu when only model_path is set (no LoRA).
    #[cfg(feature = "csi-embed")]
    let sona: Option<Arc<std::sync::Mutex<crate::sona::SonaAdapter>>> = {
        match (
            state.config.csi_model_path.as_deref(),
            state.config.csi_lora_path.as_deref(),
        ) {
            (Some(mp), Some(lp)) => {
                match crate::sona::SonaAdapter::load(mp, lp) {
                    Ok(s) => {
                        tracing::info!(
                            model = %mp.display(),
                            lora  = %lp.display(),
                            "SONA online LoRA adapter loaded (ADR-183 iter 19)"
                        );
                        Some(Arc::new(std::sync::Mutex::new(s)))
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "SONA load failed — falling back to static embedder");
                        None
                    }
                }
            }
            _ => None,
        }
    };

    // ADR-183 iter 19 fix: subscribe to ALL broadcast readings so SONA gets
    // every reading (~900/min), not just one per 60 s brain tick.
    // The subscriber task owns an Arc clone of the SONA mutex; the brain tick
    // below only calls embed(), which takes &self.
    #[cfg(feature = "csi-embed")]
    if let Some(ref sona_arc) = sona {
        let sona_sub = Arc::clone(sona_arc);
        let state_sub = Arc::clone(&state);
        tokio::spawn(async move {
            let mut rx = state_sub.subscribe();
            let mut recv_count: u64 = 0;
            loop {
                match rx.recv().await {
                    Ok(reading) => {
                        recv_count += 1;
                        if recv_count % 500 == 0 {
                            tracing::info!(recv_count, "sona subscriber: readings received");
                        }
                        // Pass all readings including Unavailable (empty room → Class::Absent).
                        // Class::from_vitals maps (hr=0, br=0) → Absent so SONA learns the
                        // "no person present" embedding. The status filter was blocking all
                        // pushes in empty-room deployment (ADR-183 iter 19 fix).
                        match sona_sub.lock() {
                            Ok(mut guard) => {
                                let steps_before = guard.steps();
                                guard.push(&reading);
                                let steps_after = guard.steps();
                                if steps_after > steps_before && steps_after % 10 == 0 {
                                    tracing::info!(
                                        steps = steps_after,
                                        total = guard.total_samples(),
                                        "sona: gradient step taken"
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(%e, "sona: mutex poisoned — subscriber stopping");
                                break;
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(dropped = n, "sona: broadcast lagged; readings skipped");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    // Static embedder: used when model_path is set but no LoRA (or SONA failed).
    #[cfg(feature = "csi-embed")]
    let csi_embedder: Option<ruvector_hailo::CsiEmbedderCpu> = if sona.is_some() {
        None // SONA takes over when both paths are set
    } else {
        match state.config.csi_model_path.as_deref() {
            Some(mp) => {
                match ruvector_hailo::CsiEmbedderCpu::open_with_lora(mp, None) {
                    Ok(e) => {
                        tracing::info!(path = %mp.display(), "CSI embedder loaded (ADR-183 Tier 3)");
                        Some(e)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, path = %mp.display(), "CSI embedder load failed");
                        None
                    }
                }
            }
            None => None,
        }
    };

    let mut tick = tokio::time::interval(interval);
    // Skip the immediate first tick — let the pipeline collect at
    // least one full window before we POST.
    tick.tick().await;

    loop {
        tick.tick().await;
        let readings = state.latest_snapshot().await;
        tracing::debug!(
            count = readings.len(),
            "brain tick: snapshotting latest readings"
        );
        if readings.is_empty() {
            continue;
        }
        for reading in &readings {
            let summary = format_vitals_summary(reading, &state.config.node_name);
            match client.post_memory("spatial-vitals", &summary).await {
                Ok(()) => {
                    state.stats.brain_posts_ok.fetch_add(1, Ordering::Relaxed);
                    tracing::info!(
                        node_id = reading.node_id,
                        breathing_bpm = reading.breathing.value_bpm,
                        heart_rate_bpm = reading.heart_rate.value_bpm,
                        "POST /memories ok"
                    );
                }
                Err(e) => {
                    state.stats.brain_posts_failed.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(error = %e, node_id = reading.node_id, "POST /memories failed");
                }
            }

            // ADR-183 Tier 3 iter 19: SONA online adaptation + CSI embedding POST.
            // push() is driven by the subscriber task (all readings, ~900/min).
            // Here we only call embed() to get the current adapted embedding.
            #[cfg(feature = "csi-embed")]
            if let Some(ref sona_mutex) = sona {
                if reading.status != crate::types::VitalStatus::Unavailable {
                    let embedding = {
                        let sona = sona_mutex.lock().unwrap();
                        let features = reading_to_csi_features(reading);
                        sona.embed(&features)
                    };
                    post_csi_embedding(&client, &state, reading, &embedding).await;
                }
            } else {
                // Static embedder fallback (no LoRA path set).
                #[cfg(feature = "csi-embed")]
                if let Some(ref embedder) = csi_embedder {
                    if reading.status != crate::types::VitalStatus::Unavailable {
                        let features = reading_to_csi_features(reading);
                        let embedding = embedder.embed(&features);
                        post_csi_embedding(&client, &state, reading, &embedding).await;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::unavailable_reading;
    use crate::types::{VitalEstimate, VitalStatus};

    #[test]
    fn unavailable_reading_summary_mentions_warmup() {
        let r = unavailable_reading(5, 0);
        let s = format_vitals_summary(&r, "cognitum-cluster-1");
        assert!(s.contains("warmup"));
        assert!(s.contains("node 5"));
        assert!(s.contains("cognitum-cluster-1"));
    }

    #[test]
    fn valid_reading_summary_includes_bpm_and_status() {
        let r = VitalReading {
            node_id: 9,
            timestamp_us: 0,
            breathing: VitalEstimate {
                value_bpm: 14.5,
                confidence: 0.85,
                status: VitalStatus::Valid,
            },
            heart_rate: VitalEstimate {
                value_bpm: 72.0,
                confidence: 0.7,
                status: VitalStatus::Valid,
            },
            snr_db: 32.0,
            subcarrier_count: 56,
            window_frames: 900,
            status: VitalStatus::Valid,
        };
        let s = format_vitals_summary(&r, "cognitum-cluster-2");
        assert!(s.contains("breathing 14.5 bpm"));
        assert!(s.contains("heart rate 72.0 bpm"));
        assert!(s.contains("snr 32.0 dB"));
        assert!(s.contains("status valid"));
        assert!(s.contains("conf 85%"));
    }

    #[test]
    fn memory_post_serialises() {
        let p = MemoryPost {
            category: "spatial-vitals".into(),
            content: "test".into(),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"category\":\"spatial-vitals\""));
        assert!(json.contains("\"content\":\"test\""));
    }

    /// Verify the feature-extraction function produces a sensible 8-vector.
    /// Kept out of the `csi-embed` feature gate since `reading_to_csi_features`
    /// is conditionally compiled — this test only runs with the feature.
    #[cfg(feature = "csi-embed")]
    #[test]
    fn reading_to_features_normalises_correctly() {
        use crate::types::{VitalEstimate, VitalStatus};
        let r = VitalReading {
            node_id: 3,
            timestamp_us: 0,
            breathing: VitalEstimate {
                value_bpm: 15.0,
                confidence: 0.9,
                status: VitalStatus::Valid,
            },
            heart_rate: VitalEstimate {
                value_bpm: 60.0,
                confidence: 0.8,
                status: VitalStatus::Valid,
            },
            snr_db: 20.0,
            subcarrier_count: 56,
            window_frames: 900,
            status: VitalStatus::Valid,
        };
        let f = reading_to_csi_features(&r);
        let arr = f.to_array();
        // breathing_bpm_norm = 15/30 = 0.5
        assert!((arr[0] - 0.5).abs() < 1e-5, "breathing norm");
        // heart_rate_bpm_norm = 60/120 = 0.5
        assert!((arr[2] - 0.5).abs() < 1e-5, "hr norm");
        // log_snr_norm = 20/40 = 0.5
        assert!((arr[5] - 0.5).abs() < 1e-5, "snr norm");
        // All values in [0, 1]
        for v in arr {
            assert!(v >= 0.0 && v <= 1.0, "value out of [0,1]: {v}");
        }
    }
}
