use std::path::Path;
use std::process::{Command, Output};

fn rvf(args: &[&str], cwd: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rvf"))
        .args(args)
        .current_dir(cwd)
        .output()
        .unwrap()
}
fn text(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}
fn write_vectors(dir: &Path, base: usize, n: usize) {
    let recs: Vec<String> = (0..n)
        .map(|i| format!("{{\"id\":{},\"vector\":[{},1,2,3]}}", base + i + 1, i))
        .collect();
    std::fs::write(dir.join("v.json"), format!("[{}]", recs.join(","))).unwrap();
}
fn store() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    assert!(rvf(&["create", "-d", "4", "s.rvf"], d.path())
        .status
        .success());
    for round in 0..3 {
        write_vectors(d.path(), round * 10, 5);
        assert!(rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path())
            .status
            .success());
    }
    d
}
fn witness_payload_offsets(bytes: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + 64 <= bytes.len() && bytes[pos..pos + 4] == 0x5256_4653u32.to_le_bytes() {
        let plen = u64::from_le_bytes(bytes[pos + 0x10..pos + 0x18].try_into().unwrap()) as usize;
        if bytes[pos + 5] == 0x0A {
            out.push((pos + 64, plen));
        }
        pos = (pos + 64 + plen + 63) & !63;
    }
    out
}
fn att_payload(extra: usize) -> Vec<u8> {
    // count=1 attestation-shaped payload, padded to total len `extra`+...
    let mut p = Vec::new();
    p.extend_from_slice(&1u32.to_le_bytes());
    p.extend_from_slice(&0u64.to_le_bytes());
    p.extend_from_slice(&[0u8; 73]);
    p.resize(85 + extra, 0xAB);
    p
}

/// ATTACK: replace the NEWEST witness segment (an ingest record) with a payload
/// shaped like an attestation witness; repair the content hash. Should be flagged.
#[test]
fn disguise_newest_witness_as_attestation() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let offs = witness_payload_offsets(&bytes);
    let (start, _) = *offs.last().unwrap();
    let hdr = start - 64;
    let payload = att_payload(43); // 128 bytes -> same 192-byte padded slot
    let seg = rvf_wire::writer::write_segment(0x0A, &payload, rvf_types::SegmentFlags::empty(), 0);
    assert_eq!(seg.len(), 192);
    let id = u64::from_le_bytes(bytes[hdr + 8..hdr + 16].try_into().unwrap());
    let seg = rvf_wire::writer::write_segment(0x0A, &payload, rvf_types::SegmentFlags::empty(), id);
    bytes[hdr..hdr + 192].copy_from_slice(&seg);
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    eprintln!("{}", text(&o));
    assert_ne!(
        o.status.code(),
        Some(0),
        "disguised witness accepted: {}",
        text(&o)
    );
}

/// FALSE REJECT: a legit attestation witness whose payload length is a multiple of 73.
#[test]
fn legit_attestation_len_multiple_of_73_false_fail() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    use rvf_crypto::attestation::*;
    let _ = &mut bytes;
    // build payload manually via rvf_crypto
    let hdr = rvf_types::AttestationHeader {
        platform: 1,
        attestation_type: 0x03,
        quote_length: 0,
        reserved_0: 0,
        measurement: [0; 32],
        signer_id: [0; 32],
        timestamp_ns: 1,
        nonce: [0; 16],
        svn: 1,
        sig_algo: 1,
        flags: 0,
        reserved_1: [0; 3],
        report_data_len: 22,
    };
    let rec = encode_attestation_record(&hdr, &[7u8; 22], &[]);
    let payload = build_attestation_witness_payload(
        &[rec],
        &[1],
        &[rvf_types::AttestationWitnessType::PlatformAttestation],
    )
    .unwrap();
    assert_eq!(payload.len() % 73, 0, "len {}", payload.len());
    let seg =
        rvf_wire::writer::write_segment(0x0A, &payload, rvf_types::SegmentFlags::empty(), 9999);
    bytes.extend_from_slice(&seg);
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    eprintln!("{}", text(&o));
    assert!(
        o.status.success(),
        "valid attestation reported as failure: {}",
        text(&o)
    );
}

/// Truncate the newest witness off (documented) -- and also replace by zero witnesses.
#[test]
fn strip_all_witnesses_reports_ok() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    for (s, _) in witness_payload_offsets(&bytes) {
        bytes[s - 64 + 5] = 0x7F;
    }
    std::fs::write(&p, &bytes).unwrap();
    // Default: exit 0 but an explicit warning; --require-witness fails.
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    assert!(
        text(&o).contains("WARNING: nothing was verified"),
        "{}",
        text(&o)
    );
    let o = rvf(&["verify-witness", "--require-witness", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
}

/// Reorder: swap segment 0 and 1 payloads (hash repaired by swapping whole segs).
#[test]
fn swap_two_witness_segments() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let offs = witness_payload_offsets(&bytes);
    let a = offs[0].0 - 64;
    let b = offs[1].0 - 64;
    let sa = bytes[a..a + 137].to_vec();
    let sb = bytes[b..b + 137].to_vec();
    // swap payloads only (keep ids/hashes consistent by swapping full segment bytes)
    bytes[a..a + 137].copy_from_slice(&sb);
    bytes[b..b + 137].copy_from_slice(&sa);
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
}

/// Torn tail then writable reopen+ingest: does verify-witness / strict open still work?
#[test]
fn torn_tail_then_write_then_verify_and_strict() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    // torn witness append: valid header claiming 73 bytes payload but only 10 present
    let seg =
        rvf_wire::writer::write_segment(0x0A, &[1u8; 73], rvf_types::SegmentFlags::empty(), 777);
    bytes.extend_from_slice(&seg[..64 + 10]);
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["status", "s.rvf"], d.path());
    eprintln!("strict status on torn: code={:?}", o.status.code());
    assert!(!o.status.success());
    write_vectors(d.path(), 100, 3);
    let o = rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path());
    eprintln!("ingest after torn: {:?} {}", o.status.code(), text(&o));
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    eprintln!(
        "verify after torn+write: {:?} {}",
        o.status.code(),
        text(&o)
    );
    let o2 = rvf(&["status", "s.rvf"], d.path());
    eprintln!("status after: {:?} {}", o2.status.code(), text(&o2));
    assert!(o.status.success(), "{}", text(&o));
    assert!(o2.status.success(), "{}", text(&o2));
}

// ---------------- batch 2: runtime-driven scenarios ----------------
use rvf_runtime::{QueryOptions, RvfOptions, RvfStore};

fn check(path: &Path, what: &str) {
    let dir = path.parent().unwrap();
    let f = path.file_name().unwrap().to_str().unwrap();
    let v = rvf(&["verify-witness", f], dir);
    assert!(
        v.status.success(),
        "[{what}] verify-witness failed: {}",
        text(&v)
    );
    for c in ["status", "inspect"] {
        let s = rvf(&[c, f], dir);
        assert!(
            s.status.success(),
            "[{what}] strict {c} failed: {}",
            text(&s)
        );
    }
}
fn v4(x: f32) -> Vec<f32> {
    vec![x, 1.0, 2.0, 3.0]
}

#[test]
fn audited_query_then_compact_then_ingest() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("a.rvf");
    let mut o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    o.witness.audit_queries = true;
    let mut s = RvfStore::create(&path, o).unwrap();
    for i in 1..=5u64 {
        s.ingest_batch(&[&v4(i as f32)[..]], &[i], None).unwrap();
    }
    s.query_audited(&v4(1.0), 2, &QueryOptions::default())
        .unwrap();
    s.delete(&[1, 2]).unwrap();
    s.query_audited(&v4(1.0), 2, &QueryOptions::default())
        .unwrap();
    s.compact().unwrap();
    check(&path, "after compact (same session)");
    s.query_audited(&v4(1.0), 2, &QueryOptions::default())
        .unwrap(); // unlisted tail witness
    s.ingest_batch(&[&v4(9.0)[..]], &[9], None).unwrap();
    s.close().unwrap();
    check(&path, "after audited+ingest");
    let mut s = RvfStore::open(&path).unwrap();
    s.query_audited(&v4(1.0), 2, &QueryOptions::default())
        .unwrap(); // open() default audit=false
    s.compact().unwrap(); // unlisted? compacting with trailing witness
    s.close().unwrap();
    check(&path, "reopen+compact");
}

#[test]
fn audited_unlisted_tail_then_compact_then_reopen_ingest() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("b.rvf");
    let mut o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    o.witness.audit_queries = true;
    let mut s = RvfStore::create(&path, o).unwrap();
    for i in 1..=3u64 {
        s.ingest_batch(&[&v4(i as f32)[..]], &[i], None).unwrap();
    }
    s.delete(&[1]).unwrap();
    s.query_audited(&v4(1.0), 2, &QueryOptions::default())
        .unwrap(); // unlisted W after manifest
    s.compact().unwrap();
    s.close().unwrap();
    check(&path, "unlisted-then-compact");
    let mut s = RvfStore::open(&path).unwrap();
    s.ingest_batch(&[&v4(7.0)[..]], &[7], None).unwrap();
    s.close().unwrap();
    check(&path, "then reopen+ingest");
}

#[test]
fn derive_embed_kernel_noop_delete() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("p.rvf");
    let o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    let mut s = RvfStore::create(&path, o).unwrap();
    for i in 1..=3u64 {
        s.ingest_batch(&[&v4(i as f32)[..]], &[i], None).unwrap();
    }
    let e0 = s.epoch();
    let r = s.delete(&[1000]).unwrap();
    eprintln!(
        "noop delete: deleted={} epoch {}->{}",
        r.deleted, e0, r.epoch
    );
    s.embed_kernel(1, 0, 0, &[0x90u8; 64], 0, None).unwrap();
    s.ingest_batch(&[&v4(11.0)[..]], &[11], None).unwrap();
    let child = d.path().join("c.rvf");
    let mut c = s
        .derive(&child, rvf_types::DerivationType::Clone, None)
        .unwrap();
    c.ingest_batch(&[&v4(21.0)[..]], &[21], None).unwrap();
    c.delete(&[21]).unwrap();
    c.compact().unwrap();
    c.close().unwrap();
    s.close().unwrap();
    check(&path, "parent");
    check(&child, "child");
}

#[test]
fn partially_rejected_and_fully_rejected_batches_epoch() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("e.rvf");
    let o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    let mut s = RvfStore::create(&path, o).unwrap();
    let bad = vec![1.0f32; 3];
    let good = v4(1.0);
    let e0 = s.epoch();
    let r = s
        .ingest_batch(&[&bad[..], &bad[..]], &[1, 2], None)
        .unwrap();
    assert_eq!((r.accepted, r.rejected, s.epoch()), (0, 2, e0));
    let r = s
        .ingest_batch(&[&bad[..], &good[..]], &[1, 2], None)
        .unwrap();
    assert_eq!((r.accepted, r.rejected), (1, 1));
    assert_eq!(s.epoch(), e0 + 1);
    // empty batch
    let r = s.ingest_batch(&[], &[], None).unwrap();
    eprintln!(
        "empty batch: {:?} epoch {}",
        (r.accepted, r.rejected),
        s.epoch()
    );
    assert_eq!(s.epoch(), e0 + 1);
    // duplicate ids where one is dim-rejected
    let r = s.ingest_batch(&[&bad[..], &good[..]], &[5, 5], None);
    eprintln!(
        "dup id with one rejected: {:?}",
        r.map(|r| (r.accepted, r.rejected))
    );
    let e_before = s.epoch();
    // dup ids both valid
    let r = s.ingest_batch(&[&good[..], &good[..]], &[6, 6], None);
    assert!(r.is_err());
    assert_eq!(s.epoch(), e_before, "failed ingest must not move epoch");
    s.close().unwrap();
    check(&path, "epoch");
}

#[test]
fn nonfinite_and_extreme_queries() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("n.rvf");
    let o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    let mut s = RvfStore::create(&path, o).unwrap();
    for i in 1..=5u64 {
        s.ingest_batch(&[&v4(i as f32)[..]], &[i], None).unwrap();
    }
    // NaN vector ingest accepted?
    let nanv = vec![f32::NAN, 0.0, 0.0, 0.0];
    let r = s.ingest_batch(&[&nanv[..]], &[100], None);
    eprintln!(
        "ingest NaN vector -> {:?}",
        r.as_ref()
            .map(|r| (r.accepted, r.rejected))
            .map_err(|e| format!("{e:?}"))
    );
    let infv = vec![f32::INFINITY, 0.0, 0.0, 0.0];
    let r = s.ingest_batch(&[&infv[..]], &[101], None);
    eprintln!(
        "ingest inf vector -> {:?}",
        r.as_ref()
            .map(|r| (r.accepted, r.rejected))
            .map_err(|e| format!("{e:?}"))
    );
    let q = QueryOptions::default();
    for (name, qv) in [
        ("neg zero", vec![-0.0f32, 1.0, 2.0, 3.0]),
        ("subnormal", vec![f32::MIN_POSITIVE / 4.0, 0.0, 0.0, 0.0]),
        ("f32::MAX", vec![f32::MAX, f32::MAX, f32::MAX, f32::MAX]),
        ("-f32::MAX", vec![-f32::MAX, 0.0, 0.0, 0.0]),
    ] {
        let r = s.query(&qv, 5, &q);
        eprintln!(
            "{name}: {:?}",
            r.map(|v| v.iter().map(|x| (x.id, x.distance)).collect::<Vec<_>>())
                .map_err(|e| format!("{e:?}"))
        );
    }
    for (name, qv) in [
        ("nan", vec![f32::NAN; 4]),
        ("inf", vec![f32::INFINITY, 0., 0., 0.]),
        ("-inf", vec![f32::NEG_INFINITY, 0., 0., 0.]),
    ] {
        let mut qo = QueryOptions::default();
        assert!(s.query(&qv, 5, &qo).is_err(), "{name} plain");
        qo.force_exact = true;
        assert!(s.query(&qv, 5, &qo).is_err(), "{name} exact");
        qo.rabitq = true;
        assert!(s.query(&qv, 5, &qo).is_err(), "{name} rabitq");
        assert!(
            s.query_with_envelope(&qv, 5, &QueryOptions::default())
                .is_err(),
            "{name} envelope"
        );
        let mut a = RvfStore::open(&path); // locked by s -> skip
        let _ = &mut a;
    }
    // k = 0 and wrong-dim ordering
    eprintln!("k=0: {:?}", s.query(&v4(1.0), 0, &q).map(|v| v.len()));
    eprintln!(
        "nan wrong dim: {:?}",
        s.query(&[f32::NAN], 1, &q)
            .map(|v| v.len())
            .map_err(|e| format!("{e:?}"))
    );
    s.close().unwrap();
}

#[test]
fn attestation_then_runtime_write_chain() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    // a REAL attestation payload; an invalid lookalike must fail (see disguise_* tests)
    let seg = rvf_wire::writer::write_segment(
        0x0A,
        &real_attestation(22),
        rvf_types::SegmentFlags::empty(),
        5555,
    );
    bytes.extend_from_slice(&seg);
    std::fs::write(&p, &bytes).unwrap();
    write_vectors(d.path(), 200, 2);
    let o = rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path());
    assert!(o.status.success(), "{}", text(&o));
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    eprintln!("{}", text(&o));
    assert!(o.status.success(), "{}", text(&o));
}

#[test]
fn strict_open_every_truncation_point() {
    let d = store();
    let p = d.path().join("s.rvf");
    let full = std::fs::read(&p).unwrap();
    // segment boundaries (unpadded end and padded end)
    let mut ok_cuts = std::collections::BTreeSet::new();
    let mut inside: Vec<(usize, usize)> = vec![];
    let mut pos = 0usize;
    while pos + 64 <= full.len() && full[pos..pos + 4] == 0x5256_4653u32.to_le_bytes() {
        let plen = u64::from_le_bytes(full[pos + 0x10..pos + 0x18].try_into().unwrap()) as usize;
        ok_cuts.insert(pos);
        ok_cuts.insert(pos + 64 + plen);
        inside.push((pos, pos + 64 + plen));
        pos = (pos + 64 + plen + 63) & !63;
        ok_cuts.insert(pos);
    }
    eprintln!("len {} boundaries {:?}", full.len(), ok_cuts);
    let mut false_accept = vec![];
    let mut false_reject = vec![];
    for cut in 1..=full.len() {
        std::fs::write(&p, &full[..cut]).unwrap();
        let o = rvf(&["status", "--json", "s.rvf"], d.path());
        let acc = o.status.success();
        let boundary = !inside.iter().any(|&(s, e)| s < cut && cut < e);
        let in_padding = ok_cuts.iter().any(|_| false);
        let _ = in_padding;
        if acc && !boundary {
            false_accept.push((
                cut,
                text(&o)
                    .replace('\n', " ")
                    .chars()
                    .take(120)
                    .collect::<String>(),
            ));
        }
        if !acc && cut == full.len() {
            false_reject.push(cut);
        }
    }
    eprintln!("accepted at NON-boundary cuts: {}", false_accept.len());
    for (c, t) in false_accept.iter().take(60) {
        eprintln!("  cut {c}: {t}");
    }
    assert!(false_reject.is_empty());
    assert!(
        false_accept.is_empty(),
        "strict status accepted {} cut(s) that end inside a segment: {:?}",
        false_accept.len(),
        false_accept.iter().map(|x| x.0).collect::<Vec<_>>()
    );
    // trailing zero padding / junk on a valid file
    for extra in [
        vec![0u8; 1],
        vec![0u8; 63],
        vec![0u8; 4096],
        vec![0xFFu8; 100],
        b"RVFS".to_vec(),
        b"RVFS\x01".to_vec(),
        vec![0u8; 64],
    ] {
        let mut b = full.clone();
        b.extend_from_slice(&extra);
        std::fs::write(&p, &b).unwrap();
        let o = rvf(&["status", "s.rvf"], d.path());
        eprintln!(
            "trailing {} bytes (first {:02x?}): strict status ok={}",
            extra.len(),
            &extra[..extra.len().min(4)],
            o.status.success()
        );
    }
}

fn legacy_payload(action: &str) -> Vec<u8> {
    let mut p = vec![0x02u8];
    p.extend_from_slice(&1_700_000_000_000_000_000u64.to_le_bytes());
    p.extend_from_slice(&(action.len() as u32).to_le_bytes());
    p.extend_from_slice(action.as_bytes());
    p.extend_from_slice(&[0u8; 32]);
    p
}

/// A genuine rvf-runtime<=0.4.0 witness record whose total length is 73 bytes
/// (e.g. "ingest:count=10000,epoch=100" is 28 bytes) is reported as a FAILED chain;
/// --allow-legacy does not help.
#[test]
fn genuine_legacy_record_of_73_bytes_is_hard_failure() {
    let d = tempfile::tempdir().unwrap();
    assert!(rvf(&["create", "-d", "4", "s.rvf"], d.path())
        .status
        .success());
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let action = "ingest:count=10000,epoch=100";
    assert_eq!(action.len(), 28);
    let pl = legacy_payload(action);
    assert_eq!(pl.len(), 73);
    bytes.extend_from_slice(&rvf_wire::writer::write_segment(
        0x0A,
        &pl,
        rvf_types::SegmentFlags::empty(),
        77,
    ));
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["verify-witness", "--allow-legacy", "s.rvf"], d.path());
    eprintln!("code={:?} {}", o.status.code(), text(&o));
    assert!(
        o.status.success(),
        "genuine legacy record -> hard failure even with --allow-legacy"
    );
}

#[test]
fn legacy_short_record_exit2_and_allow_legacy() {
    let d = tempfile::tempdir().unwrap();
    assert!(rvf(&["create", "-d", "4", "s.rvf"], d.path())
        .status
        .success());
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    bytes.extend_from_slice(&rvf_wire::writer::write_segment(
        0x0A,
        &legacy_payload("ingest:count=1,epoch=1"),
        rvf_types::SegmentFlags::empty(),
        77,
    ));
    std::fs::write(&p, &bytes).unwrap();
    assert_eq!(
        rvf(&["verify-witness", "s.rvf"], d.path()).status.code(),
        Some(2)
    );
    assert_eq!(
        rvf(&["verify-witness", "--allow-legacy", "s.rvf"], d.path())
            .status
            .code(),
        Some(0)
    );
    // legacy then runtime write then verify
    write_vectors(d.path(), 0, 2);
    assert!(rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path())
        .status
        .success());
    let o = rvf(&["verify-witness", "--allow-legacy", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(0), "{}", text(&o));
    assert_eq!(
        rvf(&["verify-witness", "s.rvf"], d.path()).status.code(),
        Some(2)
    );
}

#[test]
fn duplicate_last_witness_segment_detected() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let offs = witness_payload_offsets(&bytes);
    let (s, l) = *offs.last().unwrap();
    let seg = bytes[s - 64..s + l].to_vec();
    bytes.extend_from_slice(&[0u8; 55]);
    bytes.extend_from_slice(&seg);
    std::fs::write(&p, &bytes).unwrap();
    assert_eq!(
        rvf(&["verify-witness", "s.rvf"], d.path()).status.code(),
        Some(1)
    );
}

#[test]
fn nan_vector_ingested_then_ann_and_exact_queries() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("h.rvf");
    let o = RvfOptions {
        dimension: 4,
        ..Default::default()
    };
    let mut s = RvfStore::create(&path, o).unwrap();
    let vs: Vec<Vec<f32>> = (0..3000u64)
        .map(|i| {
            vec![
                (i % 97) as f32,
                (i % 13) as f32,
                (i % 7) as f32,
                i as f32 / 100.0,
            ]
        })
        .collect();
    let refs: Vec<&[f32]> = vs.iter().map(|v| &v[..]).collect();
    let ids: Vec<u64> = (1..=3000).collect();
    s.ingest_batch(&refs, &ids, None).unwrap();
    let nanv = vec![f32::NAN, 0.0, 0.0, 0.0];
    let r = s.ingest_batch(&[&nanv[..]], &[9999], None).unwrap();
    eprintln!("nan ingest accepted={}", r.accepted);
    let q = vec![1.0f32, 1.0, 1.0, 1.0];
    for force in [false, true] {
        let mut qo = QueryOptions::default();
        qo.force_exact = force;
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.query(&q, 10, &qo)));
        match res {
            Ok(Ok(v)) => eprintln!(
                "force_exact={force}: {} results, any nan dist: {}, has 9999: {}",
                v.len(),
                v.iter().any(|x| x.distance.is_nan()),
                v.iter().any(|x| x.id == 9999)
            ),
            Ok(Err(e)) => eprintln!("force_exact={force}: err {e:?}"),
            Err(_) => eprintln!("force_exact={force}: PANIC"),
        }
    }
    // k larger than live count: NaN rows surface?
    let v = s
        .query(
            &q,
            5000,
            &QueryOptions {
                force_exact: true,
                ..Default::default()
            },
        )
        .unwrap();
    eprintln!(
        "k=5000 exact: {} results; nan dist rows: {}",
        v.len(),
        v.iter().filter(|x| x.distance.is_nan()).count()
    );
    s.close().unwrap();
    let s2 = RvfStore::open_readonly(&path).unwrap();
    eprintln!("reopen ok: {}", s2.status().total_vectors);
}

fn real_attestation(report_len: usize) -> Vec<u8> {
    use rvf_crypto::attestation::*;
    let hdr = rvf_types::AttestationHeader {
        platform: 1,
        attestation_type: 0x03,
        quote_length: 0,
        reserved_0: 0,
        measurement: [0; 32],
        signer_id: [0; 32],
        timestamp_ns: 1,
        nonce: [0; 16],
        svn: 1,
        sig_algo: 1,
        flags: 0,
        reserved_1: [0; 3],
        report_data_len: report_len as _,
    };
    let rec = encode_attestation_record(&hdr, &vec![7u8; report_len], &[]);
    build_attestation_witness_payload(
        &[rec],
        &[1],
        &[rvf_types::AttestationWitnessType::PlatformAttestation],
    )
    .unwrap()
}

/// A forged (hash-repaired) lookalike in the MIDDLE of the chain is also caught.
#[test]
fn lookalike_attestation_in_middle_fails() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let offs = witness_payload_offsets(&bytes);
    let (start, _) = offs[1];
    let hdr = start - 64;
    let id = u64::from_le_bytes(bytes[hdr + 8..hdr + 16].try_into().unwrap());
    let seg = rvf_wire::writer::write_segment(
        0x0A,
        &att_payload(43),
        rvf_types::SegmentFlags::empty(),
        id,
    );
    bytes[hdr..hdr + 192].copy_from_slice(&seg);
    std::fs::write(&p, &bytes).unwrap();
    assert_eq!(
        rvf(&["verify-witness", "s.rvf"], d.path()).status.code(),
        Some(1)
    );
}

/// A forged payload whose length is a multiple of 73 must never verify, and a
/// 73-byte forgery that is NOT a legacy record is a hard failure.
#[test]
fn forged_73_multiple_payload_never_valid() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let offs = witness_payload_offsets(&bytes);
    let (start, _) = *offs.last().unwrap();
    let hdr = start - 64;
    let id = u64::from_le_bytes(bytes[hdr + 8..hdr + 16].try_into().unwrap());
    let seg =
        rvf_wire::writer::write_segment(0x0A, &[0xEEu8; 73], rvf_types::SegmentFlags::empty(), id);
    bytes[hdr..hdr + 192].copy_from_slice(&seg);
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["verify-witness", "--allow-legacy", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(1), "{}", text(&o));
}

/// A genuine 73-byte legacy record is reported as legacy, never valid.
#[test]
fn genuine_73_byte_legacy_is_legacy_not_valid() {
    let d = tempfile::tempdir().unwrap();
    assert!(rvf(&["create", "-d", "4", "s.rvf"], d.path())
        .status
        .success());
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let pl = legacy_payload("ingest:count=10000,epoch=100");
    bytes.extend_from_slice(&rvf_wire::writer::write_segment(
        0x0A,
        &pl,
        rvf_types::SegmentFlags::empty(),
        77,
    ));
    std::fs::write(&p, &bytes).unwrap();
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(2), "{}", text(&o));
    assert!(!text(&o).contains("verified successfully"));
    // runtime continues from zero after it, and the result verifies as legacy+chain
    write_vectors(d.path(), 0, 2);
    assert!(rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path())
        .status
        .success());
    let o = rvf(&["verify-witness", "--allow-legacy", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(0), "{}", text(&o));
}

/// Two attestation segments with chain entries around them: the chain
/// continues across them (attestation does not move the head).
#[test]
fn attestation_between_runtime_entries_keeps_chain() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    for (i, n) in [22usize, 30].iter().enumerate() {
        bytes.extend_from_slice(&rvf_wire::writer::write_segment(
            0x0A,
            &real_attestation(*n),
            rvf_types::SegmentFlags::empty(),
            6000 + i as u64,
        ));
    }
    std::fs::write(&p, &bytes).unwrap();
    write_vectors(d.path(), 300, 2);
    assert!(rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path())
        .status
        .success());
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    assert_eq!(o.status.code(), Some(0), "{}", text(&o));
}

/// Torn tail -> writable open -> the torn bytes are gone and the file is clean.
#[test]
fn torn_tail_is_repaired_by_writable_open() {
    let d = store();
    let p = d.path().join("s.rvf");
    let mut bytes = std::fs::read(&p).unwrap();
    let before = bytes.len();
    let seg =
        rvf_wire::writer::write_segment(0x0A, &[1u8; 73], rvf_types::SegmentFlags::empty(), 777);
    bytes.extend_from_slice(&seg[..64 + 10]);
    std::fs::write(&p, &bytes).unwrap();
    write_vectors(d.path(), 100, 3);
    assert!(rvf(&["ingest", "-i", "v.json", "s.rvf"], d.path())
        .status
        .success());
    let after = std::fs::read(&p).unwrap();
    assert!(after.len() > before);
    assert_eq!(
        &after[..before],
        &bytes[..before],
        "committed prefix untouched"
    );
    assert!(rvf(&["status", "s.rvf"], d.path()).status.success());
    assert!(rvf(&["verify-witness", "s.rvf"], d.path()).status.success());
}

/// Existing damaged files (torn header whose length covers later segments)
/// are handled by the verifier without following the stale length.
#[test]
fn verifier_ignores_stale_torn_header_swallowing_later_segments() {
    let d = store();
    let p = d.path().join("s.rvf");
    let bytes = std::fs::read(&p).unwrap();
    let mut b = bytes.clone();
    // Stale torn header (declares a 4 KiB witness payload) right after the
    // last segment, then a legitimate later append inside its claimed extent.
    let mut torn =
        rvf_wire::writer::write_segment(0x0A, &[1u8; 73], rvf_types::SegmentFlags::empty(), 777);
    torn.truncate(64);
    torn[0x10..0x18].copy_from_slice(&4096u64.to_le_bytes());
    b.extend_from_slice(&torn);
    b.resize((b.len() + 63) & !63, 0);
    std::fs::write(&p, &b).unwrap();
    // Append a real runtime entry the safe way: with a repaired copy, then
    // graft its segment after the torn header.
    let copy = d.path().join("copy.rvf");
    std::fs::write(&copy, &bytes).unwrap();
    write_vectors(d.path(), 400, 2);
    assert!(rvf(&["ingest", "-i", "v.json", "copy.rvf"], d.path())
        .status
        .success());
    let grown = std::fs::read(&copy).unwrap();
    b.extend_from_slice(&grown[bytes.len()..]);
    std::fs::write(&p, &b).unwrap();
    let o = rvf(&["verify-witness", "s.rvf"], d.path());
    // The later chain entries are still visited (chain may or may not link
    // across the grafted manifest bytes), but the run must not skip them.
    assert!(text(&o).contains("Witness segments:"), "{}", text(&o));
    let n: usize = text(&o)
        .lines()
        .find(|l| l.contains("Witness segments:"))
        .and_then(|l| l.split_whitespace().last())
        .and_then(|v| v.parse().ok())
        .unwrap();
    assert!(
        n >= 4,
        "later segments swallowed by torn header: {n}\n{}",
        text(&o)
    );
}
