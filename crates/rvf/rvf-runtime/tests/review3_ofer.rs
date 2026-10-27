//! Round 3 review.
use ed25519_dalek::SigningKey;
use rvf_index::{LayerA, PartitionEntry};
use rvf_runtime::progressive_stream::layers::encode_a;
use rvf_runtime::progressive_stream::StreamIngest;
use rvf_types::SegmentFlags;
use rvf_wire::stream::{StreamDescriptorBuilder, StreamPolicy};
use rvf_wire::writer::write_segment_with_algo;
use std::time::Instant;
/// Serialises tests that measure process-wide RSS or wall time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[path = "support/signed.rs"]
#[allow(dead_code)]
mod signed;

fn stream_with(parts: Vec<PartitionEntry>, nvec: usize) -> (Vec<Vec<u8>>, [u8; 32]) {
    let a = LayerA {
        entry_points: vec![],
        top_layers: vec![],
        top_layer_start: 0,
        centroids: vec![vec![0.0; 4]],
        partition_map: parts,
    };
    let idx = encode_a(&a, 4, &[]);
    let mut vp = Vec::new();
    vp.extend_from_slice(&4u16.to_le_bytes());
    vp.extend_from_slice(&(nvec as u32).to_le_bytes());
    for i in 0..nvec as u64 {
        vp.extend_from_slice(&i.to_le_bytes());
        for x in [i as f32, 1.0, 2.0, 3.0] {
            vp.extend_from_slice(&x.to_le_bytes());
        }
    }
    let mut body = write_segment_with_algo(0x02, &idx, SegmentFlags::empty(), 1, 2);
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
        .chunk_log2(16)
        .stages(&[1, 2])
        .requires_lower_stages(true)
        .build(&body)
        .unwrap();
    (b.frames_stage_order(&body).collect(), pk)
}

fn part(c: u32, s: u64, e: u64) -> PartitionEntry {
    PartitionEntry {
        centroid_id: c,
        vector_id_start: s,
        vector_id_end: e,
        segment_ref: 0,
        block_ref: 0,
    }
}

/// F3-01 (fixed, part 1): identical or overlapping partitions are rejected when
/// Layer A is decoded (partitions must be sorted and disjoint), so they cannot
/// multiply query cost. The stream is poisoned at Layer A.
#[test]
fn f3_01_overlapping_partitions_are_rejected_at_decode() {
    for parts in [
        (0..250).map(|_| part(0, 0, u64::MAX)).collect::<Vec<_>>(),
        vec![part(0, 0, 100), part(0, 50, 200)],
        vec![part(0, 100, 200), part(1, 0, 100)], // unsorted
        vec![part(0, 10, 5)],                     // inverted
    ] {
        let (frames, pk) = stream_with(parts, 100);
        let keys = [pk];
        let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
        let mut refused = false;
        for f in &frames {
            refused |= ing.feed(f).refused.is_some();
        }
        assert!(refused && !ing.layers().layer_a);
        assert!(matches!(
            ing.state(),
            rvf_wire::stream::StreamState::Poisoned(rvf_types::ErrorCode::StreamTableInvalid)
        ));
    }
}

/// F3-01 (fixed, part 2): even a valid (disjoint) but hostile map is bounded by
/// a deterministic per-query work budget, counted in steps. 60,000 disjoint
/// wide partitions over 20,000 vectors would cost 1.2e9 steps; the search stops
/// at the budget and says so.
#[test]
fn f3_01_work_budget_bounds_a_hostile_but_valid_partition_map() {
    use rvf_runtime::progressive_stream::{QueryMode, QueryPath};
    let nvec = 20_000;
    let parts: Vec<PartitionEntry> = (0..60_000u64)
        .map(|i| part(0, i << 40, (i + 1) << 40))
        .collect();
    let (frames, pk) = stream_with(parts, nvec);
    let keys = [pk];
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    for f in &frames {
        assert!(ing.feed(f).refused.is_none());
    }
    let budget = 1_000_000 + 16 * nvec as u64;
    let r = ing.query_with(&[1.0, 1.0, 2.0, 3.0], 5, 16, QueryMode::LayersOnly);
    eprintln!(
        "steps {} (budget {budget}), truncated {}",
        r.work_steps, r.work_truncated
    );
    assert_eq!(r.path, QueryPath::Layers);
    assert!(r.work_truncated, "budget exhausted is reported");
    assert!(r.work_steps <= budget, "{} > {budget}", r.work_steps);
    assert_eq!(
        r.envelope.quality,
        rvf_types::quality::ResponseQuality::Degraded
    );
    assert!(r.envelope.degradation.is_some());
    // An explicit smaller budget is honoured exactly.
    ing.set_query_work_budget(Some(50_000));
    let r = ing.query_with(&[1.0, 1.0, 2.0, 3.0], 5, 16, QueryMode::LayersOnly);
    assert!(r.work_truncated && r.work_steps <= 50_000);
}

mod rollback {
    #![cfg(feature = "ed25519")]
    use super::signed::*;
    use rvf_runtime::RvfStore;
    use rvf_types::security::SecurityPolicy;

    /// F3-02: under a signed page the scan no longer looks at newer manifests,
    /// so replacing only the 4 KiB trailing page with an OLDER valid page rolls
    /// a Strict reader back to the old state while the newer segments and
    /// manifest remain in the file. (Rollback by truncation was always possible;
    /// this needs no truncation and leaves the newer data physically present.)
    #[test]
    fn f3_02_page_swap_rolls_back_without_truncation() {
        let d = tempfile::tempdir().unwrap();
        let p = signed_store(d.path()); // 12 vectors, page P1
        let old = std::fs::read(&p).unwrap();
        let old_page = page_of(&old);
        {
            let mut s = RvfStore::open_signed(
                &p,
                opts(&[public(&SECRET)]),
                SecurityPolicy::Strict,
                &SECRET,
            )
            .unwrap();
            let v = rows(6, 900.0);
            let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
            s.ingest_batch(&refs, &(100..106).collect::<Vec<u64>>(), None)
                .unwrap();
            s.close().unwrap();
        }
        let mut now = std::fs::read(&p).unwrap();
        let fresh =
            RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict)
                .unwrap();
        assert_eq!(fresh.status().total_vectors, 18);
        drop(fresh);
        set_page(&mut now, &old_page);
        std::fs::write(&p, &now).unwrap();
        let r = RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict);
        let n = r.map(|s| s.status().total_vectors);
        eprintln!("after swapping in the old page: {n:?} (file still holds the newer manifest)");
        assert!(n.is_err(), "Strict served the rolled-back state");
    }
}

/// F3-03: two sessions that download the SAME descriptor concurrently: the
/// second `commit_pin` fails with a conflict although the pin is identical
/// (the CAS compares against the pin seen at admission, not the identical
/// stored value that documentation says is a no-op).
#[test]
fn f3_03_identical_concurrent_session_gets_spurious_conflict() {
    use rvf_wire::stream::{MemoryPinStore, StreamVerifier};
    let pins = MemoryPinStore::new();
    let key = SigningKey::from_bytes(&[0xA7; 32]);
    let pk = key.verifying_key().to_bytes();
    let body = write_segment_with_algo(0x01, &[1u8; 200], SegmentFlags::empty(), 1, 2);
    let b = StreamDescriptorBuilder::new([4; 16], key)
        .chunk_log2(12)
        .build(&body)
        .unwrap();
    let keys = [pk];
    let mk = || {
        let mut p = StreamPolicy::new(&keys);
        p.pins = Some(&pins);
        StreamVerifier::new(p)
    };
    let (mut s1, mut s2) = (mk(), mk());
    let frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    for f in &frames {
        s1.push(f).unwrap();
        s2.push(f).unwrap();
    }
    s1.commit_pin(&pins).expect("first commit");
    let r = s2.commit_pin(&pins);
    assert!(
        r.is_ok(),
        "identical pin re-recorded must be a no-op success, got {r:?}"
    );
}

/// F3-04: the `deleted` tombstone set is not charged to any budget; journal
/// segments are small on the wire (12 B per entry) and each is released after
/// parsing, so the set grows with every journal segment in the stream.
#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "reads /proc/self/statm (Linux only)")]
fn f3_04_tombstone_set_growth_is_unbudgeted() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut body = Vec::new();
    let nseg = 40u64;
    let per = 100_000u64;
    let mut stages = Vec::new();
    for s in 0..nseg {
        let mut p = Vec::new();
        p.extend_from_slice(&(per as u32).to_le_bytes());
        p.extend_from_slice(&[0u8; 12]);
        for i in 0..per {
            p.push(0x01);
            p.extend_from_slice(&[0u8; 3]);
            p.extend_from_slice(&(s * per + i).to_le_bytes());
        }
        body.extend(write_segment_with_algo(
            0x04,
            &p,
            SegmentFlags::empty(),
            s + 1,
            2,
        ));
        stages.push(0u8);
    }
    let key = SigningKey::from_bytes(&[0xA7; 32]);
    let pk = key.verifying_key().to_bytes();
    let b = StreamDescriptorBuilder::new([5; 16], key)
        .chunk_log2(16)
        .stages(&stages)
        .build(&body)
        .unwrap();
    let keys = [pk];
    let rss = || {
        std::fs::read_to_string("/proc/self/statm")
            .unwrap()
            .split_whitespace()
            .nth(1)
            .unwrap()
            .parse::<u64>()
            .unwrap()
            * 4
    };
    let before = rss();
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    let mut wire = 0u64;
    for f in b.frames_file_order(&body) {
        wire += f.len() as u64;
        assert!(ing.feed(&f).refused.is_none());
    }
    let grown = (rss() - before) * 1024;
    eprintln!(
        "journal wire {} KB -> resident +{} KB ({:.1}x)",
        wire / 1024,
        grown / 1024,
        grown as f64 / wire as f64
    );
    assert!(
        grown < 2 * wire,
        "tombstone set retains {:.1}x the wire bytes with no budget",
        grown as f64 / wire as f64
    );
}

fn stream_with_centroids(ncent: usize, nvec: usize) -> (Vec<Vec<u8>>, [u8; 32]) {
    let a = LayerA {
        entry_points: vec![],
        top_layers: vec![],
        top_layer_start: 0,
        centroids: (0..ncent)
            .map(|i| vec![((i as u64 * 2654435761) % 1000003) as f32; 8])
            .collect(),
        partition_map: vec![],
    };
    let idx = encode_a(&a, 8, &[]);
    let mut vp = Vec::new();
    vp.extend_from_slice(&8u16.to_le_bytes());
    vp.extend_from_slice(&(nvec as u32).to_le_bytes());
    for i in 0..nvec as u64 {
        vp.extend_from_slice(&i.to_le_bytes());
        for _ in 0..8 {
            vp.extend_from_slice(&(i as f32).to_le_bytes());
        }
    }
    let mut body = write_segment_with_algo(0x02, &idx, SegmentFlags::empty(), 1, 2);
    body.extend(write_segment_with_algo(
        0x01,
        &vp,
        SegmentFlags::empty(),
        2,
        2,
    ));
    let key = SigningKey::from_bytes(&[0xA7; 32]);
    let pk = key.verifying_key().to_bytes();
    let b = StreamDescriptorBuilder::new([6; 16], key)
        .chunk_log2(16)
        .stages(&[1, 2])
        .requires_lower_stages(true)
        .build(&body)
        .unwrap();
    (b.frames_stage_order(&body).collect(), pk)
}

/// F4-02: the layer-search work budget counts vector-store lookups only. The
/// centroid scan and sort in `ProgressiveIndex::search_layer_a_only` are
/// proportional to the number of centroids in the (signed) Layer A and are
/// not charged: cost per query grows with the payload, whatever the budget.
#[test]
fn f4_02_centroid_scan_is_outside_the_work_budget() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use rvf_runtime::progressive_stream::QueryMode;
    let mut t = Vec::new();
    for c in [250_000usize, 500_000, 1_000_000] {
        let (frames, pk) = stream_with_centroids(c, 100);
        let keys = [pk];
        let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
        for f in &frames {
            assert!(ing.feed(f).refused.is_none(), "centroids {c}");
        }
        ing.set_query_work_budget(Some(1_000));
        let s = Instant::now();
        let r = ing.query_with(&[0.5; 8], 3, 16, QueryMode::LayersOnly);
        let dt = s.elapsed();
        eprintln!(
            "{c} centroids: {dt:?}, work_steps {}, truncated {}",
            r.work_steps, r.work_truncated
        );
        t.push(dt.as_secs_f64());
    }
    assert!(
        t[2] < 0.05,
        "layer query time grows with centroid count despite a 1000-step budget: {t:?}"
    );
}

/// F4-03: with no Layer A published the Auto policy scans every visible vector
/// (`!can_layers ||` short-circuits the latency bound), so per-query cost is
/// unbounded in the number of arrived vectors.
#[test]
fn f4_03_scan_without_layer_a_ignores_the_latency_bound() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    use rvf_runtime::progressive_stream::QueryMode;
    let nvec = 4_000_000usize;
    let mut vp = Vec::new();
    vp.extend_from_slice(&1u16.to_le_bytes());
    vp.extend_from_slice(&(nvec as u32).to_le_bytes());
    for i in 0..nvec as u64 {
        vp.extend_from_slice(&i.to_le_bytes());
        vp.extend_from_slice(&(i as f32).to_le_bytes());
    }
    let body = write_segment_with_algo(0x01, &vp, SegmentFlags::empty(), 1, 2);
    let key = SigningKey::from_bytes(&[0xA7; 32]);
    let pk = key.verifying_key().to_bytes();
    let b = StreamDescriptorBuilder::new([7; 16], key)
        .chunk_log2(16)
        .build(&body)
        .unwrap();
    let keys = [pk];
    let mut ing = StreamIngest::new(StreamPolicy::new(&keys));
    let mut frames = b.frames_file_order(&body);
    // stop before the last chunk so the stream stays incomplete and layerless
    let all: Vec<Vec<u8>> = frames.by_ref().collect();
    for f in &all[..all.len() - 1] {
        assert!(ing.feed(f).refused.is_none());
    }
    let s = Instant::now();
    let r = ing.query_with(&[3.0], 3, 16, QueryMode::Auto);
    let dt = s.elapsed();
    eprintln!(
        "{} visible vectors, no Layer A: Auto query took {dt:?}, path {:?}, bound 25 ms",
        r.visible_vectors, r.path
    );
    assert!(
        dt.as_millis() < 100,
        "Auto scanned {} vectors in {dt:?}",
        r.visible_vectors
    );
}

mod rollback4 {
    #![cfg(feature = "ed25519")]
    use super::signed::*;
    use rvf_runtime::RvfStore;
    use rvf_types::security::SecurityPolicy;

    fn prep() -> (tempfile::TempDir, std::path::PathBuf, Vec<u8>, usize) {
        let d = tempfile::tempdir().unwrap();
        let p = signed_store(d.path());
        let old_page = page_of(&std::fs::read(&p).unwrap()).to_vec();
        {
            let mut s = RvfStore::open_signed(
                &p,
                opts(&[public(&SECRET)]),
                SecurityPolicy::Strict,
                &SECRET,
            )
            .unwrap();
            let v = rows(6, 900.0);
            let refs: Vec<&[f32]> = v.iter().map(Vec::as_slice).collect();
            s.ingest_batch(&refs, &(100..106).collect::<Vec<u64>>(), None)
                .unwrap();
            s.close().unwrap();
        }
        let now = std::fs::read(&p).unwrap();
        let (body, _) = rvf_wire::tail_scan::split_root_page(&now);
        let (moff, _) = rvf_wire::tail_scan::find_latest_manifest(body).unwrap();
        (d, p, old_page, moff)
    }

    /// F4-04: freshness is a required choice; no default of "none".
    #[test]
    fn f4_04_freshness_is_required() {
        let (_d, p, _old, _moff) = prep();
        let mut o = opts(&[public(&SECRET)]);
        o.freshness = None;
        match RvfStore::open_with_policy(&p, o, SecurityPolicy::Strict) {
            Err(rvf_types::RvfError::Code(rvf_types::ErrorCode::FreshnessRequired)) => {}
            other => panic!("expected FreshnessRequired, got {:?}", other.map(|_| ())),
        }
    }

    /// F4-04: ANY bytes after the signed manifest count as an unsigned tail,
    /// parseable or not; `serve_signed_prefix` serves the
    /// signed state and reports the tail.
    #[test]
    fn f4_04_any_bytes_after_the_signed_manifest_are_an_unsigned_tail() {
        let (_d, p, _old, _moff) = prep();
        let b = std::fs::read(&p).unwrap();
        let (body, _) = rvf_wire::tail_scan::split_root_page(&b);
        let mut t = body.to_vec();
        let mut shaped = vec![0u8; 64];
        shaped[..4].copy_from_slice(&rvf_types::SEGMENT_MAGIC.to_le_bytes());
        shaped[5] = rvf_types::SegmentType::Manifest as u8;
        t.extend_from_slice(&shaped); // damaged header, still manifest-shaped
        t.extend_from_slice(&b[b.len() - 4096..]);
        std::fs::write(&p, &t).unwrap();
        assert!(
            RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict)
                .is_err()
        );
        let mut o = opts(&[public(&SECRET)]);
        o.serve_signed_prefix = true;
        let s = RvfStore::open_with_policy(&p, o, SecurityPolicy::Strict).unwrap();
        assert!(s.unsigned_tail_present());
    }

    /// F4-04: the F3-02 default refusal depends on FINDING the newer manifest.
    /// Rollback by page swap plus one corrupted magic byte of the newer
    /// manifest (or >256 junk candidates after it) is served silently under
    /// default options; only `min_epoch` stops it.
    #[test]
    fn f4_04_page_swap_plus_hidden_newer_manifest_defeats_the_default() {
        let (_d, p, old_page, moff) = prep();
        let mut b = std::fs::read(&p).unwrap();
        // 1. plain swap: refused by default (the fix works)
        let mut swapped = b.clone();
        set_page(&mut swapped, old_page.as_slice().try_into().unwrap());
        std::fs::write(&p, &swapped).unwrap();
        assert!(
            RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict)
                .is_err()
        );
        // 2. swap + damage the newer manifest header
        b[moff] ^= 0xFF;
        set_page(&mut b, old_page.as_slice().try_into().unwrap());
        std::fs::write(&p, &b).unwrap();
        let r = RvfStore::open_with_policy(&p, opts(&[public(&SECRET)]), SecurityPolicy::Strict);
        let n = r.map(|s| s.status().total_vectors);
        eprintln!("swap + damaged newer manifest header, default options: {n:?}");
        assert!(
            n.is_err(),
            "rolled-back state served silently under default options"
        );
    }
}

/// PinStore::advance matrix after the idempotence change (expected to pass).
#[test]
fn f4_oracle_pin_advance_matrix() {
    use rvf_wire::stream::{EpochPin, MemoryPinStore, PinStore};
    let s = MemoryPinStore::new();
    let (sg, fid) = ([1u8; 8], [2u8; 16]);
    let p = |e, h: u8| EpochPin {
        epoch: e,
        descriptor_hash: [h; 32],
    };
    assert!(s.advance(&sg, &fid, None, p(5, 1)).is_ok());
    assert!(
        s.advance(&sg, &fid, None, p(5, 1)).is_ok(),
        "identical, stale expectation"
    );
    assert!(
        s.advance(&sg, &fid, Some(p(5, 1)), p(5, 2)).is_err(),
        "different hash, same epoch"
    );
    assert!(
        s.advance(&sg, &fid, None, p(5, 2)).is_err(),
        "different hash, stale expectation"
    );
    assert!(
        s.advance(&sg, &fid, Some(p(5, 1)), p(4, 1)).is_err(),
        "lower epoch"
    );
    assert!(
        s.advance(&sg, &fid, None, p(9, 3)).is_err(),
        "higher epoch with stale expectation"
    );
    assert!(s.advance(&sg, &fid, Some(p(5, 1)), p(9, 3)).is_ok());
    assert!(
        s.advance(&sg, &fid, Some(p(9, 3)), p(5, 1)).is_err(),
        "never backwards"
    );
    assert_eq!(s.get(&sg, &fid), Some(p(9, 3)));
    assert!(
        s.advance(&sg, &[3u8; 16], None, p(1, 1)).is_ok(),
        "other lineage independent"
    );
}

/// F4-05: once a multi-chunk segment's table is freed, a late "duplicate" chunk
/// of the right length but arbitrary bytes is accepted as `Duplicate`
/// (unverified; the doc on the event still says "authentic"). Nothing is
/// applied, and it is charged to the duplicate cap. A late table frame and a
/// wrong-length chunk are still refused.
#[test]
fn f4_05_late_garbage_duplicate_is_accepted_after_table_freed() {
    use rvf_wire::stream::{chunk_frame, table_frame, Event, StreamVerifier};
    let key = SigningKey::from_bytes(&[0xA7; 32]);
    let pk = key.verifying_key().to_bytes();
    let body = write_segment_with_algo(
        0x01,
        &vec![7u8; 3 * 4096 + 100],
        SegmentFlags::empty(),
        1,
        2,
    );
    let b = StreamDescriptorBuilder::new([8; 16], key)
        .chunk_log2(12)
        .build(&body)
        .unwrap();
    let keys = [pk];
    let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
    let frames: Vec<Vec<u8>> = b.frames_file_order(&body).collect();
    let mut last_chunk = Vec::new();
    for f in &frames {
        v.push(f).unwrap();
        if f[0] == 0x01 {
            last_chunk = f.clone();
        }
    }
    assert!(matches!(
        v.state(),
        rvf_wire::stream::StreamState::Complete | rvf_wire::stream::StreamState::CompleteUnverified
    ));
    // stream complete: duplicates are BadState. Use a two-segment body instead.
    let _ = last_chunk;
    let mut body2 = write_segment_with_algo(
        0x01,
        &vec![7u8; 3 * 4096 + 100],
        SegmentFlags::empty(),
        1,
        2,
    );
    body2.extend(write_segment_with_algo(
        0x01,
        &vec![9u8; 4096 + 50],
        SegmentFlags::empty(),
        2,
        2,
    ));
    let key = SigningKey::from_bytes(&[0xA7; 32]);
    let b2 = StreamDescriptorBuilder::new([8; 16], key)
        .chunk_log2(12)
        .build(&body2)
        .unwrap();
    let mut v = StreamVerifier::new(StreamPolicy::new(&keys));
    let fr: Vec<Vec<u8>> = b2.frames_file_order(&body2).collect();
    // descriptor + all frames of segment 1 (table + chunks)
    let seg1 = b2.entries[0];
    let n1 = b2.header.chunk_count(seg1.stored_len).unwrap() as usize;
    for f in &fr[..1 + 1 + n1] {
        v.push(f).unwrap();
    }
    // garbage duplicate of chunk 1 of the finished segment 1, correct length
    let g = chunk_frame(seg1.seg_id, 1, &vec![0xAB; 4096]);
    let r = v.push(&g);
    eprintln!(
        "garbage duplicate after table freed -> {:?}",
        r.as_ref().map(|e| matches!(e, Event::Duplicate { .. }))
    );
    assert!(
        matches!(r, Ok(Event::Unverified { .. })),
        "late unverifiable bytes must not be reported as an authentic duplicate"
    );
    // late table frame refused; wrong-length chunk poisons
    let t = table_frame(seg1.seg_id, &vec![[0u8; 32]; n1]);
    assert!(v.push(&t).is_err());
}
