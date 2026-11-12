//! Adversarial review repro (finding F-03): the runtime's manifest scanner
//! allocates and copies each candidate's declared payload before it can reject
//! it, so a file packed with fake MANIFEST headers makes `open` quadratic.
//! Runs under every policy because the scan precedes any signature check.

use rvf_runtime::RvfOptions;
use rvf_runtime::RvfStore;
use rvf_types::security::SecurityPolicy;
use std::time::Instant;

/// `n` back-to-back 64-byte MANIFEST headers; header i claims every byte
/// after it as payload. The payload (the next header) fails the manifest
/// parse because its bytes 14..18 give a huge seg_count.
fn hostile_file(n: usize) -> Vec<u8> {
    let tmpl = rvf_wire::writer::write_segment(
        rvf_types::SegmentType::Manifest as u8,
        &[],
        rvf_types::SegmentFlags::empty(),
        u64::MAX,
    );
    let mut out = Vec::with_capacity(n * 64);
    for i in 0..n {
        let mut h = tmpl[..64].to_vec();
        h[0x08..0x10].copy_from_slice(&u64::MAX.to_le_bytes());
        let rem = ((n - i - 1) * 64) as u64;
        let base = rem & !0xFFFF;
        let mut pl = base | 0x8000;
        if pl > rem {
            pl = if base >= 0x10000 {
                (base - 0x10000) | 0x8000
            } else {
                rem
            };
        }
        h[0x10..0x18].copy_from_slice(&pl.to_le_bytes());
        h[0x28..0x38].fill(0); // zero hash: payload hash is not checked by the scan anyway
        out.extend_from_slice(&h);
    }
    out
}

/// F-03 regression: the scan's work is bounded by counters, not by wall
/// time. A file of N fake MANIFEST headers used to cost O(N * file) bytes
/// copied; now at most MAX_PARSE_ATTEMPTS payloads are parsed, in place, and
/// nothing is copied for a rejected candidate.
#[test]
fn f03_fake_manifest_headers_do_not_make_open_quadratic() {
    let mut prev: Option<rvf_runtime::read_path::ScanStats> = None;
    for mib in [1usize, 2, 4, 8] {
        let bytes = hostile_file(mib * 1024 * 1024 / 64);
        let t = Instant::now();
        let (found, st) = rvf_runtime::read_path::locate_manifest_with_stats(&bytes);
        eprintln!(
            "{mib} MiB hostile file: {found:?} {st:?} in {:?}",
            t.elapsed()
        );
        assert!(found.is_none(), "no real manifest in the hostile file");
        assert!(st.parse_attempts <= 256, "{st:?}");
        assert_eq!(
            st.bytes_copied, 0,
            "rejected candidates are never copied: {st:?}"
        );
        assert!(st.max_buffer <= (8 << 20) + 64, "bounded window: {st:?}");
        if let Some(p) = prev {
            // Doubling the file must not grow the work that is attacker-controlled.
            assert!(st.parse_attempts <= p.parse_attempts.max(256));
            assert!(st.payload_bytes_parsed <= 2 * p.payload_bytes_parsed.max(1 << 20));
        }
        prev = Some(st);
    }
}

/// Through the real open path under every policy: terminates, fails closed.
#[test]
fn f03_hostile_file_open_fails_fast_under_every_policy() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("h.rvf");
    std::fs::write(&p, hostile_file(2 * 1024 * 1024 / 64)).unwrap();
    for pol in [
        SecurityPolicy::Permissive,
        SecurityPolicy::WarnOnly,
        SecurityPolicy::Strict,
        SecurityPolicy::Paranoid,
    ] {
        let t = Instant::now();
        let r = RvfStore::open_with_policy(
            &p,
            RvfOptions {
                dimension: 4,
                ..Default::default()
            },
            pol,
        );
        assert!(r.is_err(), "{pol:?}");
        assert!(
            t.elapsed().as_secs_f64() < 5.0,
            "{pol:?} took {:?}",
            t.elapsed()
        );
    }
}
