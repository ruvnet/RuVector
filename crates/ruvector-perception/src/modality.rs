//! Physically-typed sensing modalities (substrate-aware: each modality has its
//! own latency, decay, and spoof-resistance — edges in the coherence graph are
//! not generic, they carry physics).

use serde::{Deserialize, Serialize};

/// A physical sensing modality. The graph is *typed*: an RF edge does not behave
/// like a thermal edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Radio (WiFi CSI, BLE RSSI) — fast, multipath-sensitive, easy to spoof statically.
    Rf,
    /// Structural vibration (piezo / accelerometer) — propagation delay, damping.
    Vibration,
    /// Acoustic (mic) — echo paths, directionality.
    Acoustic,
    /// Thermal — slow diffusion, hysteresis; responds to animate heat sources.
    Thermal,
    /// Chemical (gas / QCM / SAW) — very slow, leak/identity cues.
    Chemical,
    /// Optical / light modulation.
    Optical,
}

impl Modality {
    /// All modalities, for iteration.
    pub const ALL: [Modality; 6] = [
        Modality::Rf,
        Modality::Vibration,
        Modality::Acoustic,
        Modality::Thermal,
        Modality::Chemical,
        Modality::Optical,
    ];

    /// Short stable name (used in witnesses and hashing).
    pub fn name(self) -> &'static str {
        match self {
            Modality::Rf => "rf",
            Modality::Vibration => "vibration",
            Modality::Acoustic => "acoustic",
            Modality::Thermal => "thermal",
            Modality::Chemical => "chemical",
            Modality::Optical => "optical",
        }
    }

    /// Typed physics metadata used to weight evidence.
    pub fn physics(self) -> Physics {
        match self {
            Modality::Rf => Physics {
                latency: 0.01,
                decay: 0.2,
                spoof_resistance: 0.3,
            },
            Modality::Vibration => Physics {
                latency: 0.05,
                decay: 0.5,
                spoof_resistance: 0.7,
            },
            Modality::Acoustic => Physics {
                latency: 0.03,
                decay: 0.4,
                spoof_resistance: 0.6,
            },
            Modality::Thermal => Physics {
                latency: 2.0,
                decay: 0.95,
                spoof_resistance: 0.8,
            },
            Modality::Chemical => Physics {
                latency: 5.0,
                decay: 0.98,
                spoof_resistance: 0.9,
            },
            Modality::Optical => Physics {
                latency: 0.005,
                decay: 0.1,
                spoof_resistance: 0.2,
            },
        }
    }
}

/// Physical constants attached to a modality edge.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Physics {
    /// Characteristic response latency (seconds).
    pub latency: f32,
    /// Temporal persistence in `[0, 1]` (how slowly a change fades).
    pub decay: f32,
    /// Resistance to static spoofing / replay in `[0, 1]` (higher = harder to fake).
    pub spoof_resistance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_unique_and_physics_present() {
        let mut seen = std::collections::HashSet::new();
        for m in Modality::ALL {
            assert!(seen.insert(m.name()));
            let p = m.physics();
            assert!(p.spoof_resistance >= 0.0 && p.spoof_resistance <= 1.0);
        }
        // Thermal is slower and harder to spoof than RF — a real physical prior.
        assert!(Modality::Thermal.physics().latency > Modality::Rf.physics().latency);
        assert!(
            Modality::Thermal.physics().spoof_resistance > Modality::Rf.physics().spoof_resistance
        );
    }
}
