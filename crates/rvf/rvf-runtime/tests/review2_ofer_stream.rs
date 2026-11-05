//! Round 2 review: StreamIngest / StreamClient / layer decoders.
#[path = "support/stream_fixture.rs"]
mod fx;
use fx::*;
use rvf_runtime::progressive_stream::client::{
    recover, ClientConfig, MemTransport, RunEnd, StreamClient,
};
use rvf_runtime::progressive_stream::layers::{decode_a, decode_b};
use rvf_runtime::progressive_stream::{StreamIngest, StreamWitness};
use rvf_wire::stream::{StreamPolicy, StreamState};

/// F2-03: the witness list is capped at 4096 and then silently drops records,
/// including the terminal StreamPoisoned / Complete record the RVM audit chain
/// is meant to consume. A flapping transport (2 records per reconnect) fills it.
#[test]
fn f2_03_witness_cap_drops_the_terminal_record() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    let d_len = 5 + f.bundle.descriptor.len();
    ing.feed(&full[..d_len]);
    for _ in 0..2100 {
        ing.end_of_input();
        assert!(ing.resume());
    }
    assert!(ing.witnesses().len() >= 4096);
    let out = ing.feed(&[0xEE]);
    assert!(out.refused.is_some());
    assert!(matches!(ing.state(), StreamState::Poisoned(_)));
    let has = ing
        .witnesses()
        .iter()
        .any(|w| matches!(w, StreamWitness::StreamPoisoned { .. }));
    assert!(
        has,
        "terminal StreamPoisoned witness was dropped (cap 4096)"
    );
}

/// F2-04: `with_log` opens the log without append/seek-to-end. After
/// `recover()` + `with_ingest` + `with_log`, new frames are written at file
/// offset 0, overwriting the head of the log; the next recover() cannot replay.
#[test]
fn f2_04_log_is_corrupted_when_a_recovered_client_resumes() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("frames.log");
    let half = frames(&f)[..frames(&f).len() / 2].concat();
    {
        let mut c = StreamClient::new(
            StreamPolicy::new(&keys),
            MemTransport::new(half.clone()),
            ClientConfig {
                max_stalls: 1,
                frame_retries: 2,
            },
        )
        .with_log(&log)
        .unwrap();
        assert_eq!(c.run(&mut |_| {}), RunEnd::Suspended);
    }
    assert_eq!(std::fs::metadata(&log).unwrap().len(), half.len() as u64);
    let (ing, replayed) = recover(StreamPolicy::new(&keys), &log).unwrap();
    assert_eq!(replayed, half.len() as u64);
    let mut c = StreamClient::with_ingest(
        ing,
        MemTransport::new(full.clone()),
        ClientConfig::default(),
    )
    .with_log(&log)
    .unwrap();
    let end = c.run(&mut |_| {});
    assert!(end.all_verified(), "{end:?}");
    let on_disk = std::fs::read(&log).unwrap();
    assert_eq!(
        on_disk.len(),
        full.len(),
        "log should hold the whole stream"
    );
    let (ing2, _) = recover(StreamPolicy::new(&keys), &log).unwrap();
    assert!(matches!(
        ing2.state(),
        StreamState::Complete | StreamState::CompleteUnverified
    ));
}

/// Truncated / hostile logs never publish unverified data and never panic.
#[test]
fn f2_oracle_hostile_logs() {
    let f = fixture();
    let keys = [f.public];
    let full = frames(&f).concat();
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("l.log");
    let mut x = 77u64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for i in 0..60 {
        let mut b = full.clone();
        match i % 4 {
            0 => b.truncate((rnd() as usize) % b.len()),
            1 => {
                let at = (rnd() as usize) % b.len();
                b[at] ^= 1 << (rnd() % 8);
            }
            2 => b.extend((0..(rnd() % 500)).map(|_| rnd() as u8)),
            _ => {
                let at = (rnd() as usize) % b.len();
                b.drain(at..(at + 50).min(b.len()));
            }
        }
        std::fs::write(&p, &b).unwrap();
        let (ing, _) = recover(StreamPolicy::new(&keys), &p).unwrap();
        assert_authentic(&ing, &f.ds);
    }
    assert!(recover(StreamPolicy::new(&keys), dir.path()).is_err());
}

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/statm")
        .unwrap()
        .split_whitespace()
        .nth(1)
        .unwrap()
        .parse::<u64>()
        .unwrap()
        * 4
}

/// F2-01: allocation sized from signer-controlled counts. The count checks
/// bound items by remaining payload bytes, but each 4-byte item expands to a
/// 24+ byte struct. Measures resident growth for a Layer A payload that is
/// 40 MB of empty top layers, and a Layer B of 12-byte empty adjacency rows.
#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "reads /proc/self/statm (Linux only)")]
fn f2_01_layer_decode_memory_amplification() {
    let n = 10_000_000u32;
    let mut a = Vec::new();
    a.extend_from_slice(b"RVPA");
    a.extend_from_slice(&1u32.to_le_bytes());
    a.extend_from_slice(&0u32.to_le_bytes());
    a.extend_from_slice(&0u32.to_le_bytes());
    a.extend_from_slice(&n.to_le_bytes());
    a.extend(std::iter::repeat(0u8).take(n as usize * 4));
    a.extend_from_slice(&[0u8; 12]);
    let before = rss_kb();
    // F2-01 fixed: the decode budget (2x payload) rejects the amplifying
    // payload before allocating it.
    let d = decode_a(&a);
    assert!(d.is_none(), "an amplifying Layer A is rejected");
    let grown_a = rss_kb().saturating_sub(before);
    eprintln!(
        "Layer A: payload {} KB -> resident +{} KB ({}x)",
        a.len() / 1024,
        grown_a,
        grown_a * 1024 / a.len() as u64
    );
    drop(d);

    let m = 3_000_000u32;
    let mut b = Vec::new();
    b.extend_from_slice(b"RVPB");
    b.extend_from_slice(&m.to_le_bytes());
    for i in 0..m as u64 {
        b.extend_from_slice(&i.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes());
    let before = rss_kb();
    let d = decode_b(&b);
    assert!(d.is_none(), "an amplifying Layer B is rejected");
    let grown_b = rss_kb().saturating_sub(before);
    eprintln!(
        "Layer B: payload {} KB -> resident +{} KB ({}x)",
        b.len() / 1024,
        grown_b,
        grown_b * 1024 / b.len() as u64
    );
    drop(d);
    assert!(
        grown_a * 1024 < 3 * a.len() as u64 && grown_b * 1024 < 3 * b.len() as u64,
        "decode grew resident memory by more than 3x the payload"
    );
}

/// F2-11 (32-bit): `StreamIngest::feed` computes `hdr + declared as usize`
/// without `try_from`; with a policy admitting a 0xFFFF_FFF0-byte segment the
/// sum overflows a 32-bit usize (debug: panic; release: wraps to a runt frame
/// that poisons). Run: cargo test --target armv7-unknown-linux-musleabihf.
#[cfg(target_pointer_width = "32")]
#[test]
fn f2_11_ingest_total_overflows_on_32bit() {
    use ed25519_dalek::Signer;
    use rvf_types::stream::*;
    let sk = ed25519_dalek::SigningKey::from_bytes(&[0xA7; 32]);
    let pk = sk.verifying_key().to_bytes();
    let big = 0xFFFF_FFF0u64;
    let e = StreamEntry {
        seg_id: 1,
        seg_type: 0x01,
        stage: 0,
        flags: 0,
        requires_mask: 0,
        body_offset: 0,
        stored_len: big,
        uncompressed_len: 0,
        chunk_table_hash: [7; 32],
    };
    let h = StreamHeader {
        flags: 0,
        chunk_log2: STREAM_CHUNK_LOG2_SEGMENT,
        file_id: [1; 16],
        epoch: 1,
        body_len: big,
        rvf_identity: [0; 32],
        policy_hash: [0; 32],
        seg_count: 1,
        table_len: 72,
        expires_ns: 0,
        signer_key_id: rvf_types::sha256::sha256(&pk)[..8].try_into().unwrap(),
    };
    let mut d = h.to_bytes().to_vec();
    d.extend_from_slice(&e.to_bytes());
    d.extend_from_slice(&64u16.to_le_bytes());
    let msg = [STREAM_SIG_DOMAIN, &d[..]].concat();
    let sig = sk.sign(&msg).to_bytes();
    d.extend_from_slice(&sig);
    let trusted = [pk];
    let mut p = StreamPolicy::new(&trusted);
    p.max_segment_bytes = u32::MAX as u64;
    p.max_body_len = u32::MAX as u64;
    p.max_class_bytes = [u32::MAX as u64; 4];
    let mut ing = StreamIngest::new(p);
    let mut f = vec![0u8];
    f.extend_from_slice(&(d.len() as u32).to_le_bytes());
    f.extend_from_slice(&d);
    assert!(ing.feed(&f).refused.is_none());
    let mut hdr = vec![0x01u8];
    hdr.extend_from_slice(&1u64.to_le_bytes());
    hdr.extend_from_slice(&0u32.to_le_bytes());
    hdr.extend_from_slice(&(big as u32).to_le_bytes());
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ing.feed(&hdr)));
    eprintln!("32-bit ingest feed: {:?}", r.as_ref().map(|o| o.refused));
    // F2-11 fixed: checked arithmetic; the oversized frame is refused cleanly.
    let out = r.expect("no overflow panic");
    assert_eq!(out.refused, Some(rvf_types::ErrorCode::StreamFrameSizeLie));
}

/// F2-02: a (signed) Layer A whose partition map covers `0..u64::MAX` makes
/// `StreamIngest::query` iterate the whole id range per probed centroid
/// (`rvf-index/src/progressive.rs:117`): a hostile or buggy publisher stalls
/// every client that queries before Layer C is published.
#[test]
fn f2_02_hostile_partition_range_hangs_query() {
    use ed25519_dalek::SigningKey;
    use rvf_index::{LayerA, PartitionEntry};
    use rvf_runtime::progressive_stream::layers::encode_a;
    use rvf_types::SegmentFlags;
    use rvf_wire::stream::StreamDescriptorBuilder;
    use rvf_wire::writer::write_segment_with_algo;

    let a = LayerA {
        entry_points: vec![],
        top_layers: vec![],
        top_layer_start: 0,
        centroids: vec![vec![0.0; 4]],
        partition_map: vec![PartitionEntry {
            centroid_id: 0,
            vector_id_start: 0,
            vector_id_end: u64::MAX,
            segment_ref: 0,
            block_ref: 0,
        }],
    };
    let idx_payload = encode_a(&a, 4, &[]);
    let mut vp = Vec::new();
    vp.extend_from_slice(&4u16.to_le_bytes());
    vp.extend_from_slice(&1u32.to_le_bytes());
    vp.extend_from_slice(&7u64.to_le_bytes());
    for x in [1.0f32, 2.0, 3.0, 4.0] {
        vp.extend_from_slice(&x.to_le_bytes());
    }
    let mut body = write_segment_with_algo(0x02, &idx_payload, SegmentFlags::empty(), 1, 2);
    body.extend(write_segment_with_algo(
        0x01,
        &vp,
        SegmentFlags::empty(),
        2,
        2,
    ));
    let key = SigningKey::from_bytes(&[0xA7; 32]);
    let pk = key.verifying_key().to_bytes();
    let b = StreamDescriptorBuilder::new([3; 16], key)
        .chunk_log2(12)
        .stages(&[1, 2])
        .requires_lower_stages(true)
        .build(&body)
        .unwrap();
    let keys = [pk];
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    for fr in b.frames_stage_order(&body) {
        assert!(ing.feed(&fr).refused.is_none());
    }
    assert!(ing.layers().layer_a);
    // StreamIngest is not Send (policy holds a dyn Fn), so a watchdog thread
    // reports the hang and exits the process (the test is a defect repro).
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d2 = done.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(5));
        if !d2.load(std::sync::atomic::Ordering::SeqCst) {
            eprintln!("F2-02 HANG: query did not return within 5 s (iterating 2^64 ids)");
            std::process::exit(101);
        }
    });
    let r = ing.query(&[1.0, 2.0, 3.0, 4.0], 1, 16);
    done.store(true, std::sync::atomic::Ordering::SeqCst);
    assert_eq!(r.envelope.results.len(), 1);
    // F2-02 fixed: work is counted in lookups, not time. The hostile range
    // costs at most one step per visible vector per probed centroid.
    let steps = ing.vectors().range_steps();
    assert!(steps <= 4 * ing.vectors().len() as u64 + 4, "steps {steps}");
}

/// F2-08: `stage_plan` (server startup path for any served body) does
/// unchecked `usize` arithmetic on a hostile `payload_length`.
/// Debug: overflow panic. Release: see eprintln (wrap).
#[test]
fn f2_08_stage_plan_hostile_payload_length() {
    let mut body =
        rvf_wire::writer::write_segment(0x01, &[0u8; 16], rvf_types::SegmentFlags::empty(), 1);
    body[0x10..0x18].copy_from_slice(&u64::MAX.to_le_bytes());
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let d2 = done.clone();
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(5));
        if !d2.load(std::sync::atomic::Ordering::SeqCst) {
            eprintln!("F2-08 HANG");
            std::process::exit(102);
        }
    });
    let r = std::panic::catch_unwind(|| rvf_runtime::progressive_stream::build::stage_plan(&body));
    done.store(true, std::sync::atomic::Ordering::SeqCst);
    eprintln!(
        "stage_plan on hostile length -> {:?}",
        r.as_ref().map(|v| v.len())
    );
    assert!(r.is_ok(), "stage_plan panicked on a hostile payload_length");
}
