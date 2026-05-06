//! ADR-018 binary CSI frame parser.
//!
//! ## Wire format (v1 / v6, little-endian)
//!
//! ```text
//!   bytes 0..4   magic        u32: 0xC5110001 (raw I/Q) | 0xC5110006 (feature state)
//!   byte  4      node_id      u8
//!   byte  5      n_antennas   u8 (treat as max(1))
//!   bytes 6..8   n_subcarriers u16
//!   byte  8      channel      u8
//!   byte  9      rssi         i8 (dBm)
//!   byte 10      noise_floor  i8 (dBm)
//!   bytes 11..16 reserved
//!   bytes 16..20 timestamp_us u32
//!   bytes 20..   I/Q payload  n_subcarriers × 2 × n_antennas signed bytes
//! ```
//!
//! Unlike the iter-123 telemetry bridge (`ruview-csi-bridge`) which
//! dropped the I/Q payload, the vitals worker **keeps** it — that is
//! the entire point of ADR-183 Tier 1. We decode each I/Q pair into a
//! single complex sample and derive amplitude / phase per subcarrier.

use crate::types::NodeId;

/// ADR-018 v1 magic — raw I/Q payload follows the header.
pub const CSI_MAGIC_V1: u32 = 0xC511_0001;
/// ADR-018 v6 magic — feature-state payload (still I/Q-shaped).
pub const CSI_MAGIC_V6: u32 = 0xC511_0006;
/// Header size in bytes.
pub const ADR018_HEADER_SIZE: usize = 20;

/// Decoded ADR-018 header fields. Pure-`Copy` — cheap to clone, fits
/// in two registers on aarch64.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Adr018Header {
    pub magic: u32,
    pub node_id: NodeId,
    pub n_antennas: u8,
    pub n_subcarriers: u16,
    pub channel: u8,
    pub rssi: i8,
    pub noise_floor: i8,
    pub timestamp_us: u32,
}

impl Adr018Header {
    /// Parse an ADR-018 header from the first 20 bytes of a UDP datagram.
    ///
    /// Returns `None` when the buffer is shorter than the header or the
    /// magic is unrecognised. Pure-header parse — does not touch the
    /// I/Q payload.
    #[must_use]
    pub fn parse(buf: &[u8]) -> Option<Self> {
        if buf.len() < ADR018_HEADER_SIZE {
            return None;
        }
        let magic = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        if magic != CSI_MAGIC_V1 && magic != CSI_MAGIC_V6 {
            return None;
        }
        Some(Self {
            magic,
            node_id: buf[4],
            n_antennas: buf[5].max(1),
            n_subcarriers: u16::from_le_bytes([buf[6], buf[7]]),
            channel: buf[8],
            rssi: buf[9] as i8,
            noise_floor: buf[10] as i8,
            timestamp_us: u32::from_le_bytes([buf[16], buf[17], buf[18], buf[19]]),
        })
    }

    /// Expected payload byte count: `n_subcarriers × 2 × n_antennas`.
    /// Saturates at `u32::MAX` to avoid overflow on a malformed frame.
    #[must_use]
    pub const fn expected_payload_bytes(&self) -> u32 {
        let sub = self.n_subcarriers as u32;
        let ant = self.n_antennas as u32;
        sub.saturating_mul(2).saturating_mul(ant)
    }
}

/// Borrowed view over the I/Q payload of an ADR-018 frame. The first
/// antenna's subcarriers come first, then antenna 2, etc. Each
/// subcarrier is two signed bytes (I, Q).
#[derive(Debug, Clone, Copy)]
pub struct CsiPayload<'a> {
    bytes: &'a [u8],
    n_subcarriers: usize,
    n_antennas: usize,
}

impl<'a> CsiPayload<'a> {
    /// Number of subcarriers per antenna.
    #[must_use]
    pub const fn n_subcarriers(&self) -> usize {
        self.n_subcarriers
    }

    /// Number of antennas in this frame.
    #[must_use]
    pub const fn n_antennas(&self) -> usize {
        self.n_antennas
    }

    /// Raw byte view (after the header).
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    /// I/Q sample for `(antenna, subcarrier)`. Returns `None` when the
    /// indices are out of range.
    #[must_use]
    pub fn sample(&self, antenna: usize, subcarrier: usize) -> Option<(i8, i8)> {
        if antenna >= self.n_antennas || subcarrier >= self.n_subcarriers {
            return None;
        }
        let idx = (antenna * self.n_subcarriers + subcarrier) * 2;
        let i = *self.bytes.get(idx)? as i8;
        let q = *self.bytes.get(idx + 1)? as i8;
        Some((i, q))
    }

    /// Decode amplitudes (`sqrt(I² + Q²)`) for one antenna, one f64 per
    /// subcarrier. The result vector has length `n_subcarriers`.
    ///
    /// Vital-sign extraction folds across antennas elsewhere; this is
    /// the per-antenna primitive.
    #[must_use]
    pub fn amplitudes(&self, antenna: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_subcarriers);
        for sc in 0..self.n_subcarriers {
            let (i, q) = self.sample(antenna, sc).unwrap_or((0, 0));
            let i = f64::from(i);
            let q = f64::from(q);
            out.push((i * i + q * q).sqrt());
        }
        out
    }

    /// Decode phases (`atan2(Q, I)` in radians) for one antenna.
    #[must_use]
    pub fn phases(&self, antenna: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n_subcarriers);
        for sc in 0..self.n_subcarriers {
            let (i, q) = self.sample(antenna, sc).unwrap_or((0, 0));
            out.push(f64::from(q).atan2(f64::from(i)));
        }
        out
    }
}

/// Owned ADR-018 frame — header + a copy of the I/Q payload.
///
/// Owning the payload makes the worker's sliding window easy to reason
/// about: it just stores `Frame` values. UDP receive buffers are
/// reused per call, so we copy out.
#[derive(Debug, Clone)]
pub struct Adr018Frame {
    pub header: Adr018Header,
    pub iq: Vec<u8>,
}

impl Adr018Frame {
    /// Parse a full UDP datagram into an owned frame. Returns `None` if
    /// the datagram is too short to contain the declared payload, or
    /// the magic is unrecognised.
    #[must_use]
    pub fn parse(buf: &[u8]) -> Option<Self> {
        let header = Adr018Header::parse(buf)?;
        let want = header.expected_payload_bytes() as usize;
        let have = buf.len().saturating_sub(ADR018_HEADER_SIZE);
        if have < want {
            return None;
        }
        let iq = buf[ADR018_HEADER_SIZE..ADR018_HEADER_SIZE + want].to_vec();
        Some(Self { header, iq })
    }

    /// Borrowed view over the payload.
    #[must_use]
    pub fn payload(&self) -> CsiPayload<'_> {
        CsiPayload {
            bytes: &self.iq,
            n_subcarriers: self.header.n_subcarriers as usize,
            n_antennas: self.header.n_antennas as usize,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic ADR-018 v1 frame for tests: 2 antennas × 4
    /// subcarriers, easy-to-reason I/Q values.
    fn synth_frame() -> Vec<u8> {
        let mut buf = Vec::with_capacity(ADR018_HEADER_SIZE + 16);
        // header
        buf.extend_from_slice(&CSI_MAGIC_V1.to_le_bytes());
        buf.push(7); // node_id
        buf.push(2); // n_antennas
        buf.extend_from_slice(&4u16.to_le_bytes()); // n_subcarriers
        buf.push(11); // channel
        buf.push(0xCE); // rssi = -50 dBm (i8 from u8)
        buf.push(0x9C); // noise_floor = -100 dBm
        buf.extend_from_slice(&[0u8; 5]); // reserved
        buf.extend_from_slice(&123_456u32.to_le_bytes()); // timestamp_us
        // payload: 2 antennas × 4 subcarriers × 2 bytes = 16 bytes
        // antenna 0: (3,4),(5,12),(8,15),(7,24)  → amps 5,13,17,25
        // antenna 1: (0,0),(1,0),(0,1),(2,2)
        let payload: [i8; 16] = [3, 4, 5, 12, 8, 15, 7, 24, 0, 0, 1, 0, 0, 1, 2, 2];
        buf.extend(payload.iter().map(|&v| v as u8));
        buf
    }

    #[test]
    fn header_parses_v1() {
        let buf = synth_frame();
        let h = Adr018Header::parse(&buf).expect("v1 header");
        assert_eq!(h.magic, CSI_MAGIC_V1);
        assert_eq!(h.node_id, 7);
        assert_eq!(h.n_antennas, 2);
        assert_eq!(h.n_subcarriers, 4);
        assert_eq!(h.channel, 11);
        assert_eq!(h.rssi, -50);
        assert_eq!(h.noise_floor, -100);
        assert_eq!(h.timestamp_us, 123_456);
        assert_eq!(h.expected_payload_bytes(), 16);
    }

    #[test]
    fn header_rejects_bad_magic() {
        let mut buf = synth_frame();
        buf[0] = 0xDE;
        buf[1] = 0xAD;
        buf[2] = 0xBE;
        buf[3] = 0xEF;
        assert!(Adr018Header::parse(&buf).is_none());
    }

    #[test]
    fn header_rejects_short_buf() {
        let buf = vec![0u8; ADR018_HEADER_SIZE - 1];
        assert!(Adr018Header::parse(&buf).is_none());
    }

    #[test]
    fn n_antennas_clamps_to_one() {
        let mut buf = synth_frame();
        buf[5] = 0; // n_antennas = 0 — we treat as 1
        // truncate payload to match new "1 antenna × 4 subcarriers × 2 bytes = 8"
        buf.truncate(ADR018_HEADER_SIZE + 8);
        let h = Adr018Header::parse(&buf).expect("header");
        assert_eq!(h.n_antennas, 1);
    }

    #[test]
    fn frame_parses_and_yields_payload() {
        let buf = synth_frame();
        let frame = Adr018Frame::parse(&buf).expect("frame");
        assert_eq!(frame.iq.len(), 16);
        let payload = frame.payload();
        assert_eq!(payload.n_subcarriers(), 4);
        assert_eq!(payload.n_antennas(), 2);
        assert_eq!(payload.sample(0, 0), Some((3, 4)));
        assert_eq!(payload.sample(1, 3), Some((2, 2)));
        assert_eq!(payload.sample(2, 0), None); // out of range
    }

    #[test]
    fn frame_rejects_short_payload() {
        let mut buf = synth_frame();
        buf.truncate(ADR018_HEADER_SIZE + 8); // only half the payload
        assert!(Adr018Frame::parse(&buf).is_none());
    }

    #[test]
    fn amplitudes_are_pythagorean() {
        let buf = synth_frame();
        let frame = Adr018Frame::parse(&buf).expect("frame");
        let amps = frame.payload().amplitudes(0);
        assert_eq!(amps.len(), 4);
        // (3,4) → 5; (5,12) → 13; (8,15) → 17; (7,24) → 25
        let expected = [5.0, 13.0, 17.0, 25.0];
        for (got, want) in amps.iter().zip(expected) {
            assert!((got - want).abs() < 1e-9, "got {got} want {want}");
        }
    }

    #[test]
    fn phases_are_finite() {
        let buf = synth_frame();
        let frame = Adr018Frame::parse(&buf).expect("frame");
        let phases = frame.payload().phases(0);
        assert_eq!(phases.len(), 4);
        assert!(phases.iter().all(|p| p.is_finite()));
    }
}
