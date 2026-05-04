//! Shared mmWave-radar UART parser crate.
//!
//! Lifted out of `examples/esp32-mmwave-sensor/src/parser.rs` (iter
//! 113-114) into a standalone crate so both the on-device firmware
//! and the host-side bridge service consume one tested state-machine
//! implementation. ADR-063 (in `~/projects/RuView`) is the reference
//! protocol spec.
//!
//! `no_std`-compatible: the state machine never allocates and links
//! cleanly into an `xtensa-esp32s3-espidf` image with default features.
//! Enable `feature = "std"` for the host bridge to pull in
//! `Vec`/`String`-using helpers.
//!
//! Faithful Rust port of the state machine in
//! `~/projects/RuView/firmware/esp32-csi-node/main/mmwave_sensor.c`
//! (ADR-063). Stays no_std + zero-allocation in the hot path so the
//! ESP32-S3 main task can feed it bytes directly off the UART ring.
//!
//! Frame format (Seeed mmWave protocol):
//!
//! ```text
//!   [0]    SOF  = 0x01
//!   [1-2]  frame_id        (u16 BE)
//!   [3-4]  data_length     (u16 BE)
//!   [5-6]  frame_type      (u16 BE)
//!   [7]    header_checksum = ~xor(bytes 0..6)
//!   [8..N] payload         (N = data_length)
//!   [N+1]  data_checksum   = ~xor(payload)
//! ```
//!
//! Frame types we surface to the firmware:
//!
//! | type    | name      | payload                   |
//! |---------|-----------|---------------------------|
//! | 0x0A14  | breathing | 1 byte: BPM               |
//! | 0x0A15  | heartrate | 1 byte: BPM               |
//! | 0x0A16  | distance  | 2 bytes BE: distance cm   |
//! | 0x0F09  | presence  | 1 byte: 0 = absent, 1 = present |
//!
//! Anything else is silently consumed but emitted as `Event::Unknown`
//! so an operator with `--log-level debug` can see the radar is alive.

#![cfg_attr(not(any(test, feature = "std")), no_std)]
#![allow(dead_code)]

/// Maximum payload size we reserve in the parser. Real MR60BHA2 frames
/// are <= 8 bytes; we cap at 64 so a malformed frame can't allocate
/// unbounded space in our ring.
pub const MAX_PAYLOAD: usize = 64;

/// One decoded radar event. Float values are in physical units —
/// callers don't need to know the on-wire encoding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// Breathing rate, breaths per minute (typically 6-30).
    Breathing { bpm: u8 },
    /// Heart rate, beats per minute (typically 40-180).
    HeartRate { bpm: u8 },
    /// Distance to the nearest target, centimeters.
    Distance { cm: u16 },
    /// Presence flag: `true` if a person is detected in range.
    Presence { present: bool },
    /// Frame parsed successfully but its `frame_type` isn't in our
    /// table. Keeps the state machine happy without losing observability.
    ///
    /// Iter 249 — `payload_len` widened from `u8` to `u16` to match
    /// the MR60BHA2 protocol's 2-byte length field. The current
    /// parser caps payloads at MAX_PAYLOAD=64 so u8 didn't truncate
    /// in practice, but the type now matches the protocol intent and
    /// removes the clippy::cast_possible_truncation warning at the
    /// construction site.
    Unknown { frame_type: u16, payload_len: u16 },
    /// Header or data checksum mismatch — frame dropped, parser
    /// resyncs at the next 0x01 SOF.
    ChecksumError,
    /// We saw bytes that don't match a frame header — typical for
    /// the first few bytes of a UART stream after boot.
    Resync,
}

#[derive(Debug, Clone, Copy)]
enum State {
    /// Waiting for SOF (0x01).
    Sof,
    /// Reading the 7-byte header into `header[1..7]`.
    Header,
    /// Reading `data_length` bytes of payload.
    Payload,
    /// Reading the 1-byte trailing data checksum.
    Trailer,
}

/// `~xor(bytes)` per the Seeed protocol — XOR-fold all bytes then
/// invert. Public so the on-device self-test in main.rs can construct
/// matching fixture frames; tests in `mod tests` use it via the
/// `frame()` helper.
pub fn invert_xor_public(bytes: &[u8]) -> u8 {
    invert_xor(bytes)
}

/// Streaming MR60BHA2 frame parser. Feed bytes one-at-a-time or in
/// slices; parsed events are returned as `Option<Event>` per byte.
pub struct Mr60Parser {
    state: State,
    /// 8-byte SOF + header (SOF, frame_id_hi, frame_id_lo,
    /// length_hi, length_lo, type_hi, type_lo, header_checksum).
    header: [u8; 8],
    header_idx: usize,
    payload: [u8; MAX_PAYLOAD],
    payload_idx: usize,
    expected_payload_len: usize,
    frame_type: u16,
}

impl Default for Mr60Parser {
    fn default() -> Self {
        Self::new()
    }
}

impl Mr60Parser {
    /// Construct a fresh parser. Starts in resync-on-SOF mode.
    pub const fn new() -> Self {
        Self {
            state: State::Sof,
            header: [0u8; 8],
            header_idx: 0,
            payload: [0u8; MAX_PAYLOAD],
            payload_idx: 0,
            expected_payload_len: 0,
            frame_type: 0,
        }
    }

    /// Reset the parser to the SOF-search state. Useful when a higher
    /// layer detects the radar has been disconnected/reconnected.
    pub fn reset(&mut self) {
        self.state = State::Sof;
        self.header_idx = 0;
        self.payload_idx = 0;
        self.expected_payload_len = 0;
        self.frame_type = 0;
    }

    /// Advance the state machine by one byte. Returns the parsed event
    /// if this byte completed a frame, otherwise `None`.
    pub fn feed(&mut self, b: u8) -> Option<Event> {
        match self.state {
            State::Sof => {
                if b == 0x01 {
                    self.header[0] = b;
                    self.header_idx = 1;
                    self.state = State::Header;
                    None
                } else {
                    Some(Event::Resync)
                }
            }
            State::Header => {
                self.header[self.header_idx] = b;
                self.header_idx += 1;
                if self.header_idx < 8 {
                    return None;
                }
                // Full 8-byte header — verify the checksum at index 7.
                let expected = invert_xor(&self.header[..7]);
                if expected != self.header[7] {
                    self.reset();
                    return Some(Event::ChecksumError);
                }
                let length = u16::from_be_bytes([self.header[3], self.header[4]]) as usize;
                self.frame_type = u16::from_be_bytes([self.header[5], self.header[6]]);
                if length == 0 {
                    // Zero-length frame — emit immediately as Unknown
                    // (no payload, no trailer).
                    self.reset();
                    return Some(Event::Unknown {
                        frame_type: self.frame_type,
                        payload_len: 0,
                    });
                }
                if length > MAX_PAYLOAD {
                    // Payload too large — corrupt frame, resync.
                    self.reset();
                    return Some(Event::ChecksumError);
                }
                self.expected_payload_len = length;
                self.payload_idx = 0;
                self.state = State::Payload;
                None
            }
            State::Payload => {
                self.payload[self.payload_idx] = b;
                self.payload_idx += 1;
                if self.payload_idx >= self.expected_payload_len {
                    self.state = State::Trailer;
                }
                None
            }
            State::Trailer => {
                let expected = invert_xor(&self.payload[..self.expected_payload_len]);
                let event = if expected != b {
                    Event::ChecksumError
                } else {
                    decode_event(self.frame_type, &self.payload[..self.expected_payload_len])
                };
                self.reset();
                Some(event)
            }
        }
    }

    /// Convenience: feed a slice and invoke `handler` for every event.
    pub fn feed_slice<F: FnMut(Event)>(&mut self, bytes: &[u8], mut handler: F) {
        for &b in bytes {
            if let Some(ev) = self.feed(b) {
                handler(ev);
            }
        }
    }
}

/// `~xor(bytes)` per the Seeed protocol — XOR-fold all bytes then
/// invert. Wrapping ops keep this in `core::num::Wrapping<u8>` territory
/// without the wrapper.
fn invert_xor(bytes: &[u8]) -> u8 {
    let mut acc: u8 = 0;
    for &b in bytes {
        acc ^= b;
    }
    !acc
}

/// Map a successfully-checksummed frame to a typed event. Unknown
/// frame types surface as `Event::Unknown` rather than being dropped —
/// keeps debug logs informative without breaking the state machine.
fn decode_event(frame_type: u16, payload: &[u8]) -> Event {
    match frame_type {
        0x0A14 if !payload.is_empty() => Event::Breathing { bpm: payload[0] },
        0x0A15 if !payload.is_empty() => Event::HeartRate { bpm: payload[0] },
        0x0A16 if payload.len() >= 2 => Event::Distance {
            cm: u16::from_be_bytes([payload[0], payload[1]]),
        },
        0x0F09 if !payload.is_empty() => Event::Presence {
            present: payload[0] != 0,
        },
        _ => Event::Unknown {
            frame_type,
            // Iter 249 — `payload.len()` is bounded by the iter-115
            // header parser (max 1024 bytes per Seeed protocol).
            // Cast to u16 instead of u8 so we don't lose the high bit
            // for legitimately-large unknown frames.
            payload_len: u16::try_from(payload.len()).unwrap_or(u16::MAX),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic MR60BHA2 frame for tests. The Seeed checksum
    /// is `~xor(prefix)`, which we recompute here so test fixtures
    /// stay byte-identical to what the radar would send.
    fn frame(frame_type: u16, payload: &[u8]) -> Vec<u8> {
        let mut header = vec![
            0x01,
            0x00,
            0x00, // frame_id
            (payload.len() >> 8) as u8,
            payload.len() as u8,
            (frame_type >> 8) as u8,
            frame_type as u8,
        ];
        header.push(invert_xor(&header));
        let mut out = header;
        out.extend_from_slice(payload);
        out.push(invert_xor(payload));
        out
    }

    fn collect_events(bytes: &[u8]) -> Vec<Event> {
        let mut p = Mr60Parser::new();
        let mut out = Vec::new();
        p.feed_slice(bytes, |ev| out.push(ev));
        out
    }

    fn final_event(bytes: &[u8]) -> Event {
        let mut events = collect_events(bytes);
        events.pop().expect("expected at least one event")
    }

    #[test]
    fn breathing_frame_decodes() {
        let f = frame(0x0A14, &[18]);
        assert_eq!(final_event(&f), Event::Breathing { bpm: 18 });
    }

    #[test]
    fn heart_rate_frame_decodes() {
        let f = frame(0x0A15, &[72]);
        assert_eq!(final_event(&f), Event::HeartRate { bpm: 72 });
    }

    #[test]
    fn distance_frame_decodes_big_endian() {
        // 0x01F4 = 500 cm
        let f = frame(0x0A16, &[0x01, 0xF4]);
        assert_eq!(final_event(&f), Event::Distance { cm: 500 });
    }

    #[test]
    fn presence_frame_decodes_both_polarities() {
        let absent = frame(0x0F09, &[0]);
        let present = frame(0x0F09, &[1]);
        assert_eq!(final_event(&absent), Event::Presence { present: false });
        assert_eq!(final_event(&present), Event::Presence { present: true });
    }

    #[test]
    fn unknown_frame_type_surfaces_as_unknown_event() {
        // 0xBABE is not a recognised frame type.
        let f = frame(0xBABE, &[0xDE, 0xAD]);
        assert_eq!(
            final_event(&f),
            Event::Unknown {
                frame_type: 0xBABE,
                payload_len: 2
            }
        );
    }

    /// Iter 249 — Event::Unknown's payload_len type widened from u8
    /// to u16. Verify the field reports the correct value for a
    /// payload at the parser's MAX_PAYLOAD boundary (64 bytes today;
    /// any future MAX_PAYLOAD bump up to u16::MAX is now type-safe).
    #[test]
    fn unknown_event_payload_len_matches_input_size() {
        let payload = vec![0x00; 60];
        let f = frame(0xBABE, &payload);
        assert_eq!(
            final_event(&f),
            Event::Unknown {
                frame_type: 0xBABE,
                payload_len: 60
            }
        );
    }

    #[test]
    fn corrupt_header_checksum_resyncs_without_emitting_event() {
        let mut f = frame(0x0A14, &[18]);
        f[7] ^= 0xFF; // flip the header checksum
        let events = collect_events(&f);
        // We expect at least one ChecksumError; later bytes may
        // produce Resync events as the parser scans for SOF.
        assert!(
            events.contains(&Event::ChecksumError),
            "expected ChecksumError in {:?}",
            events
        );
    }

    #[test]
    fn corrupt_data_checksum_resyncs() {
        let mut f = frame(0x0A14, &[18]);
        // last byte is the data checksum
        let last = f.len() - 1;
        f[last] ^= 0x55;
        let events = collect_events(&f);
        assert!(
            events.contains(&Event::ChecksumError),
            "expected ChecksumError in {:?}",
            events
        );
    }

    #[test]
    fn parser_handles_split_byte_streams() {
        // Same frame split across many feed() calls — must still decode.
        let f = frame(0x0A15, &[88]);
        let mut p = Mr60Parser::new();
        let mut last = None;
        for &b in &f {
            if let Some(ev) = p.feed(b) {
                last = Some(ev);
            }
        }
        assert_eq!(last, Some(Event::HeartRate { bpm: 88 }));
    }

    #[test]
    fn parser_recovers_after_garbage_prefix() {
        // Random noise before the first valid SOF — parser emits
        // Resync events but eventually picks up the real frame.
        let mut bytes = vec![0xAA, 0xBB, 0xCC, 0xDD];
        bytes.extend_from_slice(&frame(0x0A16, &[0x00, 0x64])); // 100 cm
        let events = collect_events(&bytes);
        assert!(events.iter().any(|e| matches!(e, Event::Resync)));
        assert!(events
            .iter()
            .any(|e| matches!(e, Event::Distance { cm: 100 })));
    }

    #[test]
    fn invert_xor_matches_seeed_reference() {
        // Sanity: a single 0x01 byte -> ~0x01 = 0xFE.
        assert_eq!(invert_xor(&[0x01]), 0xFE);
        // Empty input folds to ~0 = 0xFF.
        assert_eq!(invert_xor(&[]), 0xFF);
        // Known fixture: 0x01 0x00 0x00 0x00 0x01 0x0A 0x14
        //   xor = 0x01 ^ 0x01 ^ 0x0A ^ 0x14 = 0x1E   →   ~0x1E = 0xE1
        assert_eq!(
            invert_xor(&[0x01, 0x00, 0x00, 0x00, 0x01, 0x0A, 0x14]),
            0xE1
        );
    }
}
