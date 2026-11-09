//! F2-01 / F2-02 regressions with counters instead of wall clocks: allocator
//! peak while decoding layers, the ingest buffer budget, and the number of
//! lookups a hostile partition range can cause.

use rvf_index::{HnswLayer, LayerA, LayerB, PartitionEntry};
use rvf_runtime::progressive_stream::layers::{decode_a, decode_b, encode_a, encode_b};
use rvf_runtime::progressive_stream::StreamIngest;
use rvf_wire::stream::StreamPolicy;
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

struct Counting;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards to the system allocator, only adding counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = LIVE.fetch_add(l.size(), SeqCst) + l.size();
            PEAK.fetch_max(now, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size(), SeqCst);
        System.dealloc(p, l)
    }
}
#[global_allocator]
static A: Counting = Counting;
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Peak extra bytes allocated while running `f`.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = LIVE.load(SeqCst);
    PEAK.store(base, SeqCst);
    let r = f();
    (r, PEAK.load(SeqCst).saturating_sub(base))
}

#[test]
fn hostile_layer_payloads_are_rejected_within_twice_their_size() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // 4M empty top layers: 16 MB on the wire, would be ~130 MB decoded.
    let n = 4_000_000u32;
    let mut a = Vec::new();
    a.extend_from_slice(b"RVPA");
    a.extend_from_slice(&1u32.to_le_bytes());
    a.extend_from_slice(&0u32.to_le_bytes());
    a.extend_from_slice(&0u32.to_le_bytes());
    a.extend_from_slice(&n.to_le_bytes());
    a.extend(std::iter::repeat_n(0u8, n as usize * 4));
    a.extend_from_slice(&[0u8; 12]);
    let (r, peak) = peak_during(|| decode_a(&a));
    assert!(r.is_none());
    assert!(peak <= 2 * a.len(), "peak {peak} for payload {}", a.len());

    // 3M empty adjacency rows in Layer B.
    let m = 3_000_000u32;
    let mut b = Vec::new();
    b.extend_from_slice(b"RVPB");
    b.extend_from_slice(&m.to_le_bytes());
    for i in 0..m as u64 {
        b.extend_from_slice(&i.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    b.extend_from_slice(&0u32.to_le_bytes());
    let (r, peak) = peak_during(|| decode_b(&b));
    assert!(r.is_none());
    assert!(peak <= 2 * b.len(), "peak {peak} for payload {}", b.len());
}

#[test]
fn legitimate_layers_decode_within_the_budget() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut l = HnswLayer::default();
    for id in 0..5000u64 {
        l.adjacency
            .insert(id, (0..16).map(|k| (id + k) % 5000).collect());
    }
    let a = LayerA {
        entry_points: vec![(1, 3)],
        top_layers: vec![l],
        top_layer_start: 3,
        centroids: vec![vec![0.5; 32]; 64],
        partition_map: (0..64)
            .map(|c| PartitionEntry {
                centroid_id: c,
                vector_id_start: c as u64 * 10,
                vector_id_end: c as u64 * 10 + 10,
                segment_ref: 0,
                block_ref: 0,
            })
            .collect(),
    };
    let enc = encode_a(&a, 32, &[]);
    let (d, peak) = peak_during(|| decode_a(&enc));
    assert!(d.is_some(), "a real Layer A fits the 2x budget");
    assert!(
        peak <= 2 * enc.len() + 4096,
        "peak {peak} for payload {}",
        enc.len()
    );

    let mut pa = BTreeMap::new();
    for id in 0..5000u64 {
        pa.insert(id, (0..32).map(|k| (id + k) % 5000).collect());
    }
    let enc = encode_b(&LayerB {
        partial_adjacency: pa,
        covered_ranges: vec![(0, 5000)],
    });
    let (d, peak) = peak_during(|| decode_b(&enc));
    assert!(d.is_some());
    assert!(
        peak <= 2 * enc.len() + 4096,
        "peak {peak} for payload {}",
        enc.len()
    );
}

/// F4-01: a signed descriptor that DECLARES a 512 MiB VEC segment while only a
/// few hundred KB are ever delivered must not cost memory proportional to the
/// claim. Allocated peak (not address space) stays within a small multiple of
/// the delivered bytes.
#[test]
fn f4_01_declared_segment_size_is_not_prepaid() {
    use ed25519_dalek::Signer;
    use rvf_types::sha256::Sha256;
    use rvf_types::stream::*;
    use rvf_wire::stream::{chunk_frame, descriptor_frame, table_frame};
    let claimed: u64 = 512 << 20;
    let chunk: u64 = 1 << 16;
    let n = claimed / chunk;
    let dim = 1usize;
    let rec = 8 + 4 * dim;
    let payload_len = claimed - 64;
    let count = ((payload_len - 6) / rec as u64) as u32;
    let mut c0 = vec![0u8; 64];
    c0[..4].copy_from_slice(&rvf_types::SEGMENT_MAGIC.to_le_bytes());
    c0[4] = rvf_types::SEGMENT_VERSION;
    c0[5] = 0x01;
    c0[0x10..0x18].copy_from_slice(&payload_len.to_le_bytes());
    c0.extend_from_slice(&(dim as u16).to_le_bytes());
    c0.extend_from_slice(&count.to_le_bytes());
    while (c0.len() as u64) + rec as u64 <= chunk {
        c0.extend_from_slice(&(c0.len() as u64).to_le_bytes());
        c0.extend_from_slice(&1.0f32.to_le_bytes());
    }
    c0.resize(chunk as usize, 0);
    let file_id = [9u8; 16];
    let seg_id = 1u64;
    let leaf = |idx: u32, bytes: &[u8]| {
        let mut h = Sha256::new();
        h.update(&[0x00]);
        h.update(&file_id);
        h.update(&seg_id.to_le_bytes());
        h.update(&idx.to_le_bytes());
        h.update(&(bytes.len() as u64).to_le_bytes());
        h.update(bytes);
        h.finalize()
    };
    let mut leaves = vec![[0xEE; 32]; n as usize];
    leaves[0] = leaf(0, &c0);
    let mut th = Sha256::new();
    th.update(&[0x01]);
    th.update(&file_id);
    th.update(&seg_id.to_le_bytes());
    th.update(&(n as u32).to_le_bytes());
    for l in &leaves {
        th.update(l);
    }
    let sk = ed25519_dalek::SigningKey::from_bytes(&[0xA7; 32]);
    let pk = sk.verifying_key().to_bytes();
    let e = StreamEntry {
        seg_id,
        seg_type: 0x01,
        stage: 0,
        flags: 0,
        requires_mask: 0,
        body_offset: 0,
        stored_len: claimed,
        uncompressed_len: 0,
        chunk_table_hash: th.finalize(),
    };
    let h = StreamHeader {
        flags: STREAM_FLAG_CHUNKED,
        chunk_log2: 16,
        file_id,
        epoch: 1,
        body_len: claimed,
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
    let sig = sk.sign(&[STREAM_SIG_DOMAIN, &d[..]].concat()).to_bytes();
    d.extend_from_slice(&sig);
    let keys = [pk];
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    let mut wire = 0usize;
    let (_, peak) = peak_during(|| {
        for f in [
            descriptor_frame(&d),
            table_frame(seg_id, &leaves),
            chunk_frame(seg_id, 0, &c0),
        ] {
            wire += f.len();
            assert!(ing.feed(&f).refused.is_none(), "frame refused");
        }
    });
    eprintln!(
        "wire {} KB; claimed {} MB; peak allocated {} KB",
        wire / 1024,
        claimed >> 20,
        peak / 1024
    );
    assert!(ing.vectors().len() < 10_000);
    assert!(
        peak < 8 * wire + (1 << 20),
        "peak {peak} B allocated for {wire} B on the wire (claim {claimed} B)"
    );
}
