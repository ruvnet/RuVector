//! Domain CSI frame — antenna-folded amplitude + phase per subcarrier.
//!
//! [`Adr018Frame`](crate::frame::Adr018Frame) is the *wire-format*
//! frame: header + I/Q payload, possibly multiple antennas. The
//! vitals pipeline operates on a single per-subcarrier amplitude /
//! phase vector, so we fold antennas at the boundary and produce a
//! [`CsiFrame`] mirroring upstream RuView's
//! `wifi_densepose_vitals::CsiFrame`. The mirror is exact so that
//! `--features ruview-integration` can swap in the upstream
//! extractors with no glue code.

use crate::frame::Adr018Frame;

/// One CSI frame after antenna folding. Vital-sign extractors consume
/// this — they don't see the wire format.
#[derive(Debug, Clone)]
pub struct CsiFrame {
    /// Per-subcarrier amplitude (linear). Length == `n_subcarriers`.
    pub amplitudes: Vec<f64>,
    /// Per-subcarrier phase in radians. Length == `n_subcarriers`.
    pub phases: Vec<f64>,
    /// Number of subcarriers in this frame.
    pub n_subcarriers: usize,
    /// Monotonically increasing sample index (frame number).
    pub sample_index: u64,
    /// Frame rate in Hz (the *frame* rate, not the OFDM symbol rate).
    pub sample_rate_hz: f64,
}

impl CsiFrame {
    /// Construct a frame, validating that amplitude / phase lengths
    /// match `n_subcarriers`.
    #[must_use]
    pub fn new(
        amplitudes: Vec<f64>,
        phases: Vec<f64>,
        n_subcarriers: usize,
        sample_index: u64,
        sample_rate_hz: f64,
    ) -> Option<Self> {
        if amplitudes.len() != n_subcarriers || phases.len() != n_subcarriers {
            return None;
        }
        Some(Self {
            amplitudes,
            phases,
            n_subcarriers,
            sample_index,
            sample_rate_hz,
        })
    }

    /// Fold an ADR-018 wire frame's antennas into one amplitude /
    /// phase vector.
    ///
    /// - **amplitude** is the *mean* magnitude across antennas
    ///   (`amp[sc] = (1/n_ant) Σ √(I² + Q²)`).
    /// - **phase** is the *circular mean* across antennas
    ///   (`phase[sc] = atan2(Σ sinθ, Σ cosθ)`) — using a plain
    ///   arithmetic mean wraps around at ±π and corrupts the signal,
    ///   while the circular mean handles the discontinuity cleanly.
    #[must_use]
    pub fn from_adr018(frame: &Adr018Frame, sample_index: u64, sample_rate_hz: f64) -> Self {
        let payload = frame.payload();
        let n_sub = payload.n_subcarriers();
        let n_ant = payload.n_antennas().max(1);

        let mut amps = vec![0.0_f64; n_sub];
        let mut sin_sum = vec![0.0_f64; n_sub];
        let mut cos_sum = vec![0.0_f64; n_sub];

        for ant in 0..n_ant {
            for sc in 0..n_sub {
                let (i, q) = payload.sample(ant, sc).unwrap_or((0, 0));
                let i = f64::from(i);
                let q = f64::from(q);
                amps[sc] += (i * i + q * q).sqrt();
                let phase = q.atan2(i);
                sin_sum[sc] += phase.sin();
                cos_sum[sc] += phase.cos();
            }
        }
        let inv_ant = 1.0_f64 / n_ant as f64;
        for a in &mut amps {
            *a *= inv_ant;
        }
        let phases: Vec<f64> = (0..n_sub)
            .map(|sc| sin_sum[sc].atan2(cos_sum[sc]))
            .collect();

        Self {
            amplitudes: amps,
            phases,
            n_subcarriers: n_sub,
            sample_index,
            sample_rate_hz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Adr018Frame, ADR018_HEADER_SIZE, CSI_MAGIC_V1};

    /// Same synthetic builder as `frame::tests` — but local so this
    /// module's tests are self-contained.
    fn synth(n_ant: u8, sub: u16, payload: &[i8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(ADR018_HEADER_SIZE + payload.len());
        buf.extend_from_slice(&CSI_MAGIC_V1.to_le_bytes());
        buf.push(0); // node_id
        buf.push(n_ant);
        buf.extend_from_slice(&sub.to_le_bytes());
        buf.push(1); // channel
        buf.push(0); // rssi
        buf.push(0); // noise floor
        buf.extend_from_slice(&[0u8; 5]); // reserved
        buf.extend_from_slice(&0u32.to_le_bytes()); // ts_us
        buf.extend(payload.iter().map(|&v| v as u8));
        buf
    }

    #[test]
    fn from_adr018_single_antenna() {
        // 1 antenna, 4 subcarriers: (3,4),(5,12),(8,15),(7,24)
        let payload: [i8; 8] = [3, 4, 5, 12, 8, 15, 7, 24];
        let buf = synth(1, 4, &payload);
        let frame = Adr018Frame::parse(&buf).unwrap();
        let csi = CsiFrame::from_adr018(&frame, 7, 30.0);
        assert_eq!(csi.n_subcarriers, 4);
        assert_eq!(csi.sample_index, 7);
        assert!((csi.sample_rate_hz - 30.0).abs() < 1e-9);
        let want_amp = [5.0, 13.0, 17.0, 25.0];
        for (got, want) in csi.amplitudes.iter().zip(want_amp) {
            assert!((got - want).abs() < 1e-9);
        }
        // Phases for (3,4): atan2(4,3) ≈ 0.927
        assert!((csi.phases[0] - 4f64.atan2(3.0)).abs() < 1e-9);
    }

    #[test]
    fn from_adr018_folds_two_antennas_amplitude_mean() {
        // 2 antennas × 2 subcarriers
        // ant 0: (3,4),(0,5)  → amps 5, 5
        // ant 1: (6,8),(0,15) → amps 10, 15
        // mean: (5+10)/2=7.5, (5+15)/2=10
        let payload: [i8; 8] = [3, 4, 0, 5, 6, 8, 0, 15];
        let buf = synth(2, 2, &payload);
        let frame = Adr018Frame::parse(&buf).unwrap();
        let csi = CsiFrame::from_adr018(&frame, 0, 30.0);
        assert!((csi.amplitudes[0] - 7.5).abs() < 1e-9);
        assert!((csi.amplitudes[1] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn circular_mean_handles_phase_wraparound() {
        // 2 antennas × 1 subcarrier, both phases ~ ±π:
        // ant 0: (-127, 1)  → phase ≈ +π (just above)
        // ant 1: (-127, -1) → phase ≈ -π (just below)
        // Arithmetic mean would be ~0 (wrong); circular mean → ±π.
        let payload: [i8; 4] = [-127, 1, -127, -1];
        let buf = synth(2, 1, &payload);
        let frame = Adr018Frame::parse(&buf).unwrap();
        let csi = CsiFrame::from_adr018(&frame, 0, 30.0);
        assert!(
            csi.phases[0].abs() > 3.0,
            "expected near ±π, got {}",
            csi.phases[0]
        );
    }

    #[test]
    fn new_validates_lengths() {
        assert!(CsiFrame::new(vec![1.0], vec![0.0, 1.0], 2, 0, 30.0).is_none());
        assert!(CsiFrame::new(vec![1.0, 2.0], vec![0.0], 2, 0, 30.0).is_none());
        assert!(CsiFrame::new(vec![1.0, 2.0], vec![0.0, 1.0], 2, 0, 30.0).is_some());
    }
}
