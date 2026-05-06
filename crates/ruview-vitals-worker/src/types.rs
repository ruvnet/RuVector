//! Vital-sign domain types — mirrors the public surface of
//! `wifi_densepose_vitals::types` so the optional
//! `--features ruview-integration` swap is mechanical.
//!
//! Owning these types in-crate (rather than re-exporting from the
//! upstream RuView crate) keeps the workspace `cargo check` hermetic
//! when RuView isn't checked out, which is the default path per
//! ADR-183 Open Question 1.

use serde::{Deserialize, Serialize};

/// ADR-018 `node_id` — 1 byte, identifies the source ESP32 sensor.
pub type NodeId = u8;

/// Status of a vital-sign measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VitalStatus {
    /// Valid measurement with clinical-grade confidence.
    Valid,
    /// Measurement present but with reduced confidence.
    Degraded,
    /// Measurement unreliable (e.g. single subcarrier source, low SNR).
    Unreliable,
    /// No measurement possible (e.g. pre-warmup, dead carriers).
    Unavailable,
}

impl VitalStatus {
    /// Combine two statuses, returning the worse of the two.
    /// Order of severity (worst → best): Unavailable → Unreliable →
    /// Degraded → Valid.
    #[must_use]
    pub const fn worst(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unavailable, _) | (_, Self::Unavailable) => Self::Unavailable,
            (Self::Unreliable, _) | (_, Self::Unreliable) => Self::Unreliable,
            (Self::Degraded, _) | (_, Self::Degraded) => Self::Degraded,
            _ => Self::Valid,
        }
    }

    /// Map to the wire-level proto enum.
    #[must_use]
    pub const fn as_proto(self) -> i32 {
        // Matches `proto/vitals.proto` Status enum.
        match self {
            Self::Unavailable => 1,
            Self::Valid => 2,
            Self::Degraded => 3,
            Self::Unreliable => 4,
        }
    }
}

impl Default for VitalStatus {
    fn default() -> Self {
        Self::Unavailable
    }
}

/// A single vital-sign estimate (breathing or heart rate).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VitalEstimate {
    /// Estimated value in BPM (beats / breaths per minute). 0.0 when
    /// `status == Unavailable`.
    pub value_bpm: f64,
    /// Confidence in [0.0, 1.0].
    pub confidence: f64,
    /// Measurement status.
    pub status: VitalStatus,
}

impl VitalEstimate {
    /// Sentinel for "no measurement possible". Matches the upstream
    /// RuView `VitalEstimate::unavailable` shape.
    #[must_use]
    pub const fn unavailable() -> Self {
        Self {
            value_bpm: 0.0,
            confidence: 0.0,
            status: VitalStatus::Unavailable,
        }
    }

    /// True if this estimate is `Valid` or `Degraded` (i.e. usable).
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        matches!(self.status, VitalStatus::Valid | VitalStatus::Degraded)
    }
}

impl Default for VitalEstimate {
    fn default() -> Self {
        Self::unavailable()
    }
}

/// Combined vital-sign reading for one sliding window. This is what the
/// gRPC service streams and what the brain POST shim summarises.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VitalReading {
    /// Source ADR-018 node_id.
    pub node_id: NodeId,
    /// Window-center wall clock (microseconds since UNIX epoch).
    pub timestamp_us: i64,
    /// Respiratory rate estimate (0.1-0.5 Hz band).
    pub breathing: VitalEstimate,
    /// Heart rate estimate (0.8-2.0 Hz band).
    pub heart_rate: VitalEstimate,
    /// Estimated SNR for the window (dB).
    pub snr_db: f32,
    /// Number of subcarriers used.
    pub subcarrier_count: u32,
    /// Frames in the sliding window when this reading was produced.
    pub window_frames: u32,
    /// Worst-case status across both estimates.
    pub status: VitalStatus,
}

impl VitalReading {
    /// Empty reading anchored to `node_id` at `timestamp_us`.
    #[must_use]
    pub const fn unavailable(node_id: NodeId, timestamp_us: i64) -> Self {
        Self {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_picks_most_severe() {
        assert_eq!(
            VitalStatus::Valid.worst(VitalStatus::Degraded),
            VitalStatus::Degraded
        );
        assert_eq!(
            VitalStatus::Degraded.worst(VitalStatus::Unavailable),
            VitalStatus::Unavailable
        );
        assert_eq!(
            VitalStatus::Unreliable.worst(VitalStatus::Valid),
            VitalStatus::Unreliable
        );
        assert_eq!(
            VitalStatus::Valid.worst(VitalStatus::Valid),
            VitalStatus::Valid
        );
    }

    #[test]
    fn unavailable_is_default() {
        assert_eq!(VitalEstimate::default().status, VitalStatus::Unavailable);
        assert!(!VitalEstimate::default().is_usable());
    }

    #[test]
    fn proto_status_ids_are_stable() {
        // These values are part of the wire contract — DO NOT renumber
        // without bumping the proto package version.
        assert_eq!(VitalStatus::Unavailable.as_proto(), 1);
        assert_eq!(VitalStatus::Valid.as_proto(), 2);
        assert_eq!(VitalStatus::Degraded.as_proto(), 3);
        assert_eq!(VitalStatus::Unreliable.as_proto(), 4);
    }
}
