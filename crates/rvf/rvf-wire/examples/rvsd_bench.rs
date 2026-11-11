//! Plain timing harness for RVSD (ADR-353). Run in release mode:
//!   cargo run --release -p rvf-wire --example rvsd_bench -- all
//! Sub-commands: all | throughput | latency | gen <dir> | stream <dir> | buffered <dir>
//! (`all` runs the memory scenarios in child processes so VmHWM is not shared).

use ed25519_dalek::SigningKey;
use rvf_crypto::stream_sha256 as sha256;
use rvf_types::SegmentFlags;
use rvf_wire::stream::*;
use rvf_wire::writer::write_segment;
use std::io::{Read, Write};
use std::time::Instant;

const MIB: usize = 1 << 20;
const FILE_ID: [u8; 16] = *b"rvsd-bench-file1";

fn key() -> SigningKey {
    SigningKey::from_bytes(&[0x2A; 32])
}

fn body(segments: usize, seg_bytes: usize) -> Vec<u8> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut b = Vec::with_capacity(segments * (seg_bytes + 128));
    for i in 0..segments {
        let p: Vec<u8> = (0..seg_bytes)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect();
        b.extend(write_segment(0x01, &p, SegmentFlags::empty(), i as u64 + 1));
    }
    b
}

/// `field` from /proc/self/status in KiB, or "n/a" where /proc is unavailable.
fn status_kb(field: &str) -> String {
    let Ok(s) = std::fs::read_to_string("/proc/self/status") else {
        return "n/a".to_string();
    };
    s.lines()
        .find(|l| l.starts_with(field))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .map_or_else(|| "n/a".to_string(), |v| v.to_string())
}

fn loadavg() -> String {
    match std::fs::read_to_string("/proc/loadavg") {
        Ok(s) => s.trim().to_string(),
        Err(_) => "n/a".to_string(),
    }
}

fn gbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

fn throughput() {
    let b = body(64, 4 * MIB);
    println!(
        "body {} MiB, cores visible {}",
        b.len() / MIB,
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    let t = Instant::now();
    let h = sha256(&b);
    let hs = t.elapsed().as_secs_f64();
    println!(
        "stream hash backend (fast-sha256) over the body: {:.3} GB/s ({:02x}{:02x}..)",
        gbps(b.len(), hs),
        h[0],
        h[1]
    );
    {
        let t = Instant::now();
        let d = rvf_types::sha256(&b);
        println!(
            "portable rvf_types::sha256 over the body: {:.3} GB/s ({:02x}{:02x}..)",
            gbps(b.len(), t.elapsed().as_secs_f64()),
            d[0],
            d[1]
        );
    }
    {
        use sha2::Digest;
        let t = Instant::now();
        let d = sha2::Sha256::digest(&b);
        println!("reference sha2 crate (dev-dependency, hardware path if available): {:.3} GB/s ({:02x}{:02x}..)", gbps(b.len(), t.elapsed().as_secs_f64()), d[0], d[1]);
    }
    let keys = [key().verifying_key().to_bytes()];
    for (label, log2) in [
        ("chunked 64 KiB", 16u8),
        ("chunked 256 KiB", 18),
        ("chunked 1 MiB", 20),
        ("segment-granular (4 MiB)", 0xFF),
    ] {
        let bundle = StreamDescriptorBuilder::new(FILE_ID, key())
            .chunk_log2(log2)
            .build(&b)
            .unwrap();
        let frames: Vec<Vec<u8>> = bundle.frames_file_order(&b).collect();
        let wire: usize = frames.iter().map(|f| f.len()).sum();
        let tables: usize = frames.iter().filter(|f| f[0] == 2).map(|f| f.len()).sum();
        let mut best = f64::MAX;
        for _ in 0..3 {
            let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
            let t = Instant::now();
            for f in &frames {
                v.push(f).unwrap();
            }
            best = best.min(t.elapsed().as_secs_f64());
            assert_eq!(v.state(), StreamState::Complete);
        }
        println!(
            "{label:26} verifier {:.3} GB/s (best of 3) | descriptor {} B ({:.4}% of body) + tables {} B ({:.4}%) = {:.4}% | wire/body {:.5}",
            gbps(b.len(), best),
            bundle.descriptor.len(),
            100.0 * bundle.descriptor.len() as f64 / b.len() as f64,
            tables,
            100.0 * tables as f64 / b.len() as f64,
            100.0 * (bundle.descriptor.len() + tables) as f64 / b.len() as f64,
            wire as f64 / b.len() as f64
        );
    }
}

fn latency() {
    let b = body(4096, 0);
    let bundle = StreamDescriptorBuilder::new(FILE_ID, key())
        .chunk_log2(16)
        .build(&b)
        .unwrap();
    let keys = [key().verifying_key().to_bytes()];
    let pol = StreamPolicy::new(&keys);
    let mut times: Vec<f64> = (0..50)
        .map(|_| {
            let t = Instant::now();
            decode_descriptor(&bundle.descriptor, &pol).unwrap();
            t.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("decode_descriptor, 4096 segments, {} B: median {:.1} us, min {:.1} us, max {:.1} us (50 runs)", bundle.descriptor.len(), times[25], times[0], times[49]);
}

fn gen(dir: &str) {
    let b = body(64, 4 * MIB);
    let bundle = StreamDescriptorBuilder::new(FILE_ID, key())
        .chunk_log2(16)
        .build(&b)
        .unwrap();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(format!("{dir}/body.bin"), &b).unwrap();
    let mut f =
        std::io::BufWriter::new(std::fs::File::create(format!("{dir}/stream.bin")).unwrap());
    for fr in bundle.frames_file_order(&b) {
        f.write_all(&fr).unwrap();
    }
    f.flush().unwrap();
    std::fs::write(format!("{dir}/desc.bin"), &bundle.descriptor).unwrap();
}

fn stream(dir: &str) {
    let keys = [key().verifying_key().to_bytes()];
    let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
    let mut rd = FrameReader::new();
    let mut f = std::fs::File::open(format!("{dir}/stream.bin")).unwrap();
    let mut buf = vec![0u8; 64 * 1024];
    let (mut n, mut total) = (0u64, 0u64);
    let t = Instant::now();
    loop {
        let k = f.read(&mut buf).unwrap();
        if k == 0 {
            break;
        }
        total += k as u64;
        rd.feed(&mut v, &buf[..k], &mut |r| {
            if let Ok(Event::Chunk { chunk, .. }) = r {
                n += chunk.bytes.len() as u64;
            }
        })
        .unwrap();
    }
    let s = t.elapsed().as_secs_f64();
    assert_eq!(v.state(), StreamState::Complete);
    println!("streamed: {} MiB verified from a {} MiB frame file in {:.2}s ({:.3} GB/s incl. read); VmHWM {} KiB", n / MIB as u64, total / MIB as u64, s, gbps(n as usize, s), status_kb("VmHWM"));
}

fn buffered(dir: &str) {
    let keys = [key().verifying_key().to_bytes()];
    let desc = std::fs::read(format!("{dir}/desc.bin")).unwrap();
    let d = decode_descriptor(&desc, &StreamPolicy::new(&keys)).unwrap();
    let t = Instant::now();
    let body = std::fs::read(format!("{dir}/body.bin")).unwrap();
    assert_eq!(sha256(&body), d.header.rvf_identity);
    for e in &d.entries {
        let seg = &body[e.body_offset as usize..(e.body_offset + e.stored_len) as usize];
        let leaves: Vec<[u8; 32]> = seg
            .chunks(1 << d.header.chunk_log2)
            .enumerate()
            .map(|(i, c)| rvf_crypto::leaf_hash(&d.header.file_id, e.seg_id, i as u32, c))
            .collect();
        assert_eq!(
            rvf_crypto::chunk_list_hash(&d.header.file_id, e.seg_id, &leaves),
            e.chunk_table_hash
        );
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "buffered: whole {} MiB body read and verified in {:.2}s ({:.3} GB/s); VmHWM {} KiB",
        body.len() / MIB,
        s,
        gbps(body.len(), s),
        status_kb("VmHWM")
    );
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("throughput") => throughput(),
        Some("latency") => latency(),
        Some("gen") => gen(&a[2]),
        Some("stream") => stream(&a[2]),
        Some("buffered") => buffered(&a[2]),
        _ => {
            println!("load average at start: {}", loadavg());
            latency();
            throughput();
            let exe = std::env::current_exe().unwrap();
            let dir = std::env::var("RVSD_BENCH_DIR").unwrap_or_else(|_| "/tmp/rvsd-bench".into());
            for sub in ["gen", "stream", "buffered"] {
                let o = std::process::Command::new(&exe)
                    .args([sub, &dir])
                    .output()
                    .unwrap();
                print!("{}", String::from_utf8_lossy(&o.stdout));
                assert!(
                    o.status.success(),
                    "{sub}: {}",
                    String::from_utf8_lossy(&o.stderr)
                );
            }
            println!("load average at end: {}", loadavg());
        }
    }
}
