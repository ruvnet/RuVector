# ruvector-mmwave

Streaming parser for the Seeed MR60BHA2 60 GHz radar's UART
protocol. Pure-Rust, no_std-compatible, zero-allocation hot path.

> **Status:** library, **11 unit tests + proptest fuzz suite**
> passing. Shared between the host-side `ruvector-mmwave-bridge`
> (parses serial input → emits NL events → cluster embed RPC) and
> the iter-115 firmware self-test that runs on the radar's MCU
> directly.

## Why a separate crate

ADR-178 Gap H: keeping the parser separate from
`ruvector-hailo-cluster` (the bridge's home crate) means a regression
in either side surfaces independently. The parser is byte-for-byte
deterministic against fuzzed inputs; the bridge layers transport,
TLS, fingerprinting on top.

## Wire format (Seeed MR60BHA2 v0.3)

```text
8-byte header  | variable payload | trailing checksum
[0x01]         | <up to 64 bytes>  | invert_xor(payload)
[frame_id_hi]
[frame_id_lo]
[length_hi]    ← 16-bit big-endian payload length
[length_lo]
[type_hi]      ← 16-bit big-endian frame type
[type_lo]
[invert_xor of 7 prior bytes]
```

Frame types currently parsed:

| `frame_type` | meaning | payload shape |
|--------------|---------|---------------|
| `0x0A05` | breathing rate | `[bpm: u8]` |
| `0x0A06` | heart rate | `[bpm: u8]` |
| `0x0A14` | nearest target distance | `[cm: u16 BE]` |
| `0x0F09` | presence flag | `[present: bool]` |
| anything else | `Event::Unknown { frame_type, payload_len }` | (iter 249) `payload_len` is `u16` |

## API surface

```rust
use ruvector_mmwave::{Event, Mr60Parser};

let mut p = Mr60Parser::new();
let frame: &[u8] = /* 60 bytes from /dev/ttyUSB0 */;
p.feed_slice(frame, |ev| match ev {
    Event::Breathing { bpm } => println!("breathing {} bpm", bpm),
    Event::HeartRate { bpm } => println!("heart rate {} bpm", bpm),
    Event::Distance { cm } => println!("distance {} cm", cm),
    Event::Presence { present } => println!("present={}", present),
    Event::Unknown { frame_type, payload_len } => {
        eprintln!("unknown frame 0x{:04x} len={}", frame_type, payload_len);
    }
    Event::ChecksumError => eprintln!("dropped frame, parser resynced"),
    Event::Resync { skipped } => eprintln!("desync, dropped {} bytes", skipped),
});
```

The closure signature is `FnMut(Event)`; the parser invokes it
zero-or-more times per byte fed. State machine resyncs cleanly on
checksum failure or unexpected SOF — no manual reset needed.

## Benchmarks

Run `cargo bench -p ruvector-mmwave` for the full criterion sweep.
Steady-state on cognitum-v0 (Pi 5):

- ~3.2 GB/s feed rate on `feed_slice` with all-recognized frames
  (most expensive event type)
- ~7.1 GB/s on the no-event-emitted path (waiting for SOF)
- Zero allocations per byte fed — the buffer is fixed at 64 bytes,
  see `MAX_PAYLOAD` const.

For typical UART rates (115200 baud → ~14 KB/s), the parser cost
is < 0.001% of one core.

## Property tests

`tests/tokenizer_proptest.rs` (proptest v1) feeds:
- arbitrary-length byte strings to verify the parser never panics
- frames with corrupted checksums to verify clean Resync
- frames with valid headers but truncated payloads to verify the
  state machine waits without emitting

## See also

- `crates/ruvector-hailo-cluster/src/bin/mmwave-bridge.rs` — the
  host-side bridge that consumes this parser and posts NL events
  to the hailo-backend cluster.
- `examples/esp32-mmwave-sensor/` — firmware-side use of this
  parser on an ESP32 paired to the radar over UART.
- ADR-063 — original mmwave integration design.
- ADR-178 Gap H — rationale for the separate crate boundary.
