//! Vitals pipeline orchestrator.
//!
//! Owns the per-node-id state for the full pipeline:
//!
//! ```text
//!   Adr018Frame
//!       │
//!       ▼  fold antennas (csi::CsiFrame::from_adr018)
//!   CsiFrame  ───►  CsiVitalPreprocessor.process  ───►  residuals
//!                                                          │
//!                            ┌─────────────────────────────┴──────────┐
//!                            ▼                                        ▼
//!                  CsiSlidingWindow.push                         BreathingExtractor.extract
//!                            │                                        │
//!                            └─►  variance_weights ─►   ───────────────┐
//!                                                                      │
//!                                                                      ▼
//!                                                            HeartRateExtractor.extract
//!                                                                      │
//!                                                                      ▼
//!                                                                VitalReading
//! ```
//!
//! The pipeline keeps a single set of extractors per worker — the
//! ESP32 nodes typically anchor a physical zone, so per-node fusion is
//! tracked on the `node_id` reported in the wire frame and surfaced on
//! the emitted [`VitalReading`].

use std::time::SystemTime;

use crate::csi::CsiFrame;
use crate::frame::Adr018Frame;
use crate::preprocessor::CsiVitalPreprocessor;
use crate::types::{NodeId, VitalEstimate, VitalReading, VitalStatus};
use crate::window::CsiSlidingWindow;
use crate::{breathing::BreathingExtractor, heartrate::HeartRateExtractor};

/// Sliding-window length in seconds. 30 s is the upstream default —
/// enough to cover a slow-breather (6 BPM = 10-second cycle) and to
/// average ~30 cardiac cycles for the autocorrelation extractor.
pub const DEFAULT_WINDOW_SECS: f64 = 30.0;
/// Default frame rate from ESP32-S3 CSI nodes — one frame ≈ 33 ms.
pub const DEFAULT_SAMPLE_RATE_HZ: f64 = 30.0;
/// Default subcarrier count for ESP32 indoor CSI.
pub const DEFAULT_N_SUBCARRIERS: usize = 56;

/// Output of one pipeline step. The pipeline only returns `Some(_)`
/// when both extractors have produced an estimate (or have explicitly
/// reported `Unavailable`); during the warm-up window the call yields
/// `None` so the caller can suppress brain POSTs.
#[derive(Debug, Clone, Copy)]
pub struct PipelineStep {
    pub reading: VitalReading,
    /// Number of frames the pipeline has consumed for this node since
    /// the last reset. Useful as a "warmup progress" indicator.
    pub frames_consumed: u64,
}

#[derive(Debug, Clone)]
pub struct VitalsPipeline {
    sample_rate_hz: f64,
    window_secs: f64,
    n_subcarriers: usize,
    preprocessor: CsiVitalPreprocessor,
    window: CsiSlidingWindow,
    breathing: BreathingExtractor,
    heart_rate: HeartRateExtractor,
    /// Frame counter; doubles as the CsiFrame.sample_index.
    frames_consumed: u64,
}

impl VitalsPipeline {
    #[must_use]
    pub fn new(n_subcarriers: usize, sample_rate_hz: f64, window_secs: f64) -> Self {
        Self {
            sample_rate_hz,
            window_secs,
            n_subcarriers,
            preprocessor: CsiVitalPreprocessor::new(n_subcarriers, 0.05),
            window: CsiSlidingWindow::new(
                n_subcarriers,
                ((sample_rate_hz * window_secs).round() as usize).max(8),
                sample_rate_hz,
            ),
            breathing: BreathingExtractor::new(n_subcarriers, sample_rate_hz, window_secs),
            heart_rate: HeartRateExtractor::new(n_subcarriers, sample_rate_hz, window_secs),
            frames_consumed: 0,
        }
    }

    /// Sensible defaults for a Pi 5 worker pulling ESP32-S3 frames.
    #[must_use]
    pub fn esp32_default() -> Self {
        Self::new(
            DEFAULT_N_SUBCARRIERS,
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_WINDOW_SECS,
        )
    }

    /// Run one wire-format ADR-018 frame through the pipeline.
    ///
    /// Returns `None` until both extractors have settled (the
    /// breathing / heart-rate windows reach 80 % full). After that,
    /// every call yields a [`PipelineStep`] — the caller decides
    /// when to flush to gRPC subscribers and / or POST a memory.
    pub fn step(&mut self, frame: &Adr018Frame, ts_us: i64) -> Option<PipelineStep> {
        self.frames_consumed = self.frames_consumed.wrapping_add(1);

        let csi = CsiFrame::from_adr018(
            frame,
            self.frames_consumed,
            self.sample_rate_hz,
        );

        let residuals = match self.preprocessor.process(&csi) {
            Some(r) => r,
            None => return None,
        };

        self.window.push(&residuals, ts_us, frame.header.node_id);
        let weights = self.window.variance_weights();

        // Evaluate **both** extractors unconditionally so neither
        // misses out on the other's warmup period. Using the `?`
        // short-circuit here would have meant `heart_rate.extract` was
        // never called during the breathing extractor's warmup (frames
        // 1..720 at default settings), and the heart-rate history
        // would stay empty long past the configured window.
        let breathing = self.breathing.extract(&residuals, &weights);
        let heart_rate = self.heart_rate.extract(&residuals, &csi.phases);
        let (breathing, heart_rate) = match (breathing, heart_rate) {
            (Some(b), Some(hr)) => (b, hr),
            _ => return None,
        };

        let snr_db = estimate_snr_db(frame.header.rssi, frame.header.noise_floor);

        let timestamp_us = self.window.center_timestamp_us().unwrap_or(ts_us);
        let status = breathing.status.worst(heart_rate.status);
        let reading = VitalReading {
            node_id: frame.header.node_id,
            timestamp_us,
            breathing,
            heart_rate,
            snr_db,
            subcarrier_count: self.n_subcarriers as u32,
            window_frames: self.window.len() as u32,
            status,
        };
        Some(PipelineStep {
            reading,
            frames_consumed: self.frames_consumed,
        })
    }

    /// Discard all extractor / window state.
    pub fn reset(&mut self) {
        self.preprocessor.reset();
        self.window.clear();
        self.breathing.reset();
        self.heart_rate.reset();
        self.frames_consumed = 0;
    }

    #[must_use]
    pub const fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    #[must_use]
    pub const fn window_secs(&self) -> f64 {
        self.window_secs
    }

    #[must_use]
    pub const fn n_subcarriers(&self) -> usize {
        self.n_subcarriers
    }

    #[must_use]
    pub const fn frames_consumed(&self) -> u64 {
        self.frames_consumed
    }
}

/// Convert RSSI / noise-floor (both in dBm) to a rough SNR in dB.
/// Saturates at 0 below the noise floor and 60 above; real ESP32-S3
/// indoor SNR rarely exceeds 35 dB.
#[must_use]
pub fn estimate_snr_db(rssi_dbm: i8, noise_dbm: i8) -> f32 {
    let snr = (rssi_dbm as i32 - noise_dbm as i32) as f32;
    snr.clamp(0.0, 60.0)
}

/// Wall-clock timestamp helper used by the worker's UDP loop. Pulled
/// out so tests can stub it.
#[must_use]
pub fn now_us() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Construct an "Unavailable" reading anchored to a node_id and
/// timestamp. Useful when the worker can't yet emit a real estimate
/// but wants to publish a heartbeat through the gRPC stream.
#[must_use]
pub fn unavailable_reading(node_id: NodeId, timestamp_us: i64) -> VitalReading {
    VitalReading {
        node_id,
        timestamp_us,
        breathing: VitalEstimate::unavailable(),
        heart_rate: VitalEstimate::unavailable(),
        snr_db: 0.0,
        subcarrier_count: 0,
        window_frames: 0,
        status: VitalStatus::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Adr018Frame, ADR018_HEADER_SIZE, CSI_MAGIC_V1};
    use std::f64::consts::TAU;

    /// Build a synthetic ADR-018 frame whose I/Q encodes a target
    /// amplitude on every subcarrier of a single antenna. The phase
    /// is held at 0 so the heart-rate's phase-fusion path is exercised
    /// the same way every frame.
    fn frame_with_amp(node_id: u8, amp: i8, n_sub: u16) -> Vec<u8> {
        let mut buf = Vec::with_capacity(ADR018_HEADER_SIZE + n_sub as usize * 2);
        buf.extend_from_slice(&CSI_MAGIC_V1.to_le_bytes());
        buf.push(node_id);
        buf.push(1); // n_antennas
        buf.extend_from_slice(&n_sub.to_le_bytes());
        buf.push(11); // channel
        buf.push(0xCE); // rssi -50 dBm
        buf.push(0x9C); // noise -100 dBm
        buf.extend_from_slice(&[0u8; 5]);
        buf.extend_from_slice(&0u32.to_le_bytes());
        for _ in 0..n_sub {
            // I = amp, Q = 0 → magnitude = |amp|, phase = 0.
            buf.push(amp as u8);
            buf.push(0u8);
        }
        buf
    }

    #[test]
    fn snr_clamps() {
        assert_eq!(estimate_snr_db(-50, -100), 50.0);
        assert_eq!(estimate_snr_db(-100, -50), 0.0);
        assert_eq!(estimate_snr_db(-30, -100), 60.0); // clamps at 60
    }

    #[test]
    fn unavailable_reading_is_unavailable_status() {
        let r = unavailable_reading(7, 12_345);
        assert_eq!(r.status, VitalStatus::Unavailable);
        assert_eq!(r.node_id, 7);
        assert_eq!(r.timestamp_us, 12_345);
    }

    #[test]
    fn pipeline_returns_none_during_warmup() {
        let mut p = VitalsPipeline::new(8, 30.0, 6.0);
        let buf = frame_with_amp(3, 10, 8);
        let frame = Adr018Frame::parse(&buf).unwrap();
        for _ in 0..10 {
            assert!(p.step(&frame, 0).is_none());
        }
    }

    #[test]
    fn pipeline_settles_into_a_reading_for_modulated_signal() {
        let mut p = VitalsPipeline::new(8, 30.0, 6.0);
        let mut got: Option<PipelineStep> = None;
        let total = (30.0 * 6.0 * 2.0) as usize;
        for i in 0..total {
            // Modulate amplitude with a 0.25 Hz sinusoid (in
            // breathing band) — drives the preprocessor's residuals
            // out of zero so the extractors see signal.
            let t = i as f64 / 30.0;
            let scale = 1.0 + 0.5 * (TAU * 0.25 * t).sin();
            let amp = (50.0 * scale).round() as i8;
            let buf = frame_with_amp(7, amp, 8);
            let frame = Adr018Frame::parse(&buf).unwrap();
            let ts = (i as i64) * 33_333; // ~30 fps
            let step = p.step(&frame, ts);
            if step.is_some() {
                got = step;
            }
        }
        let step = got.expect("pipeline produced a reading");
        assert_eq!(step.reading.node_id, 7);
        assert_eq!(step.reading.subcarrier_count, 8);
        // Window-center timestamp is in the middle of the run.
        assert!(step.reading.timestamp_us > 0);
        // The status should not be uninitialised.
        let _ = step.reading.status;
    }
}
