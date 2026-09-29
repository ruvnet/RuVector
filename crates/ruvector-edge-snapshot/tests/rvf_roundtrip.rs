//! RVF export → import round trip, limits, and foreign (runtime-shaped) files.

mod common;
use common::*;
use ruvector_edge_snapshot::rvf_format::{legacy_content_hash, DirEntry, RvfManifest};
use ruvector_edge_snapshot::*;

fn export(data: &[Row], dim: u16, metric: Metric, per_seg: u32) -> Vec<u8> {
    let mut file = Vec::new();
    let mut sink = |b: &[u8]| file.extend_from_slice(b);
    let mut x = RvfExporter::new(dim, metric, per_seg).unwrap();
    for r in data {
        x.push(r, &mut sink).unwrap();
    }
    let s = x.finish(&mut sink);
    assert_eq!(s.rows, data.len() as u64);
    assert_eq!(s.bytes, file.len() as u64);
    file
}

fn spec(dim: u16, metric: Metric) -> ImportSpec {
    ImportSpec {
        dim,
        metric,
        limits: ImportLimits {
            max_batch_rows: 64,
            ..ImportLimits::default()
        },
    }
}

/// Import `file` in `piece`-byte pieces; returns rows and batches. The
/// summary defaults to `inspect_tail` of `file` itself.
fn import(
    file: &[u8],
    spec: ImportSpec,
    summary: Option<RvfSummary>,
    piece: usize,
) -> Result<(Vec<Row>, Vec<ImportBatch>), ImportError> {
    let summary = match summary {
        Some(s) => s,
        None => inspect_tail(file, file.len() as u64, &spec.limits)?,
    };
    let mut imp = RvfImporter::new(spec, summary, ImportCursor::default())?;
    let mut batches = Vec::new();
    for p in file.chunks(piece) {
        imp.feed(p)?;
        while let Some(b) = imp.next_batch()? {
            batches.push(b);
        }
    }
    let t = imp.finish()?;
    let rows: Vec<Row> = batches.iter().flat_map(|b| b.rows.clone()).collect();
    assert_eq!(t.rows, rows.len() as u64);
    assert_eq!(t.batches, batches.len() as u64);
    Ok((rows, batches))
}

#[test]
fn export_import_round_trip_with_and_without_summary() {
    let dim = 12;
    let data = rows(1_000, dim, 11);
    let file = export(&data, dim, Metric::Cosine, 100);
    let lim = ImportLimits::default();
    let tail = &file[file.len().saturating_sub(4096)..];
    let summary = inspect_tail(tail, file.len() as u64, &lim).unwrap();
    assert_eq!(
        (summary.dim, summary.metric, summary.total_vectors),
        (dim, Metric::Cosine, 1_000)
    );
    assert_eq!(summary.vec_segs.len(), 10);
    assert_eq!(summary.planned_rows, 1_000);

    for piece in [1, 7, 4096, file.len()] {
        let (back, batches) = import(
            &file,
            spec(dim, Metric::Cosine),
            Some(summary.clone()),
            piece,
        )
        .unwrap();
        assert_rows_identical(&data, &back);
        assert!(batches.iter().all(|b| b.rows.len() <= 64));
        assert!(batches.windows(2).all(|w| w[1].seq == w[0].seq + 1));
    }
    let (back, _) = import(&file, spec(dim, Metric::Cosine), None, 999).unwrap();
    assert_rows_identical(&data, &back);
}

#[test]
fn export_segments_parse_with_rvf_wire_and_legacy_hash() {
    let dim = 4;
    let file = export(&rows(10, dim, 2), dim, Metric::L2, 4);
    let mut pos = 0;
    let mut types = Vec::new();
    while pos < file.len() {
        let (h, payload) = rvf_wire::read_segment(&file[pos..]).unwrap();
        assert_eq!(h.checksum_algo, 0);
        assert_eq!(h.content_hash, legacy_content_hash(payload));
        types.push(h.seg_type);
        pos += 64 + payload.len();
    }
    assert_eq!(types, [0x0B, 0x01, 0x0B, 0x01, 0x0B, 0x01, 0x05]);
}

#[test]
fn dimension_and_metric_mismatches_are_refused() {
    let file = export(&rows(50, 8, 3), 8, Metric::Dot, 16);
    let lim = ImportLimits::default();
    let summary = inspect_tail(&file, file.len() as u64, &lim).unwrap();
    // Both refusals happen in `new`, before a single batch can exist (the
    // summary is mandatory, so there is no mode that checks the metric late).
    assert_eq!(
        RvfImporter::new(
            spec(16, Metric::Dot),
            summary.clone(),
            ImportCursor::default()
        )
        .unwrap_err(),
        ImportError::DimensionMismatch {
            expected: 16,
            got: 8
        }
    );
    assert_eq!(
        RvfImporter::new(spec(8, Metric::L2), summary, ImportCursor::default()).unwrap_err(),
        ImportError::MetricMismatch
    );
}

#[test]
fn size_quota_corruption_and_truncation_are_refused() {
    let dim = 8;
    let file = export(&rows(300, dim, 4), dim, Metric::Cosine, 100);
    let sum = || Some(inspect_tail(&file, file.len() as u64, &ImportLimits::default()).unwrap());
    let mut s = spec(dim, Metric::Cosine);
    s.limits.max_segment_payload = 1_000;
    assert!(matches!(
        import(&file, s, sum(), 4096),
        Err(ImportError::SegmentTooLarge(_))
    ));
    // The same limit refuses the directory entries while summarising.
    assert!(inspect_tail(&file, file.len() as u64, &s.limits).is_err());
    let mut s = spec(dim, Metric::Cosine);
    s.limits.max_file_bytes = file.len() as u64 - 1;
    assert_eq!(
        import(&file, s, sum(), 4096).unwrap_err(),
        ImportError::FileTooLarge
    );
    // Over quota is refused up front from the summary's planned rows.
    let mut s = spec(dim, Metric::Cosine);
    s.limits.max_rows = 299;
    assert_eq!(
        RvfImporter::new(s, sum().unwrap(), ImportCursor::default()).unwrap_err(),
        ImportError::QuotaExceeded { limit: 299 }
    );

    let mut bad = file.clone();
    bad[64 + 20] ^= 1; // inside the first sidecar payload
    assert_eq!(
        import(&bad, spec(dim, Metric::Cosine), sum(), 4096).unwrap_err(),
        ImportError::Checksum(0)
    );
    let cut = &file[..file.len() - 10];
    assert_eq!(
        import(cut, spec(dim, Metric::Cosine), sum(), 4096).unwrap_err(),
        ImportError::Truncated
    );
    assert!(inspect_tail(cut, cut.len() as u64, &ImportLimits::default()).is_err());
    assert_eq!(
        import(&file[..0], spec(dim, Metric::Cosine), None, 1).unwrap_err(),
        ImportError::NoManifest
    );
}

/// A file shaped like rvf-runtime output: no sidecars, several manifests,
/// a deletion and a superseded segment.
fn runtime_file(dim: u16) -> (Vec<u8>, Vec<(u64, Vec<f32>)>) {
    let mut file = Vec::new();
    let mut dir = Vec::new();
    let seg = |file: &mut Vec<u8>, t: u8, id: u64, payload: &[u8]| {
        let off = file.len() as u64;
        let mut h = [0u8; 64];
        h[0..4].copy_from_slice(&0x5256_4653u32.to_le_bytes());
        h[4] = 1;
        h[5] = t;
        h[8..16].copy_from_slice(&id.to_le_bytes());
        h[16..24].copy_from_slice(&(payload.len() as u64).to_le_bytes());
        h[0x28..0x38].copy_from_slice(&legacy_content_hash(payload));
        file.extend_from_slice(&h);
        file.extend_from_slice(payload);
        DirEntry {
            seg_id: id,
            offset: off,
            payload_len: payload.len() as u64,
            seg_type: t,
        }
    };
    let vecs: Vec<(u64, Vec<f32>)> = (0..6u64)
        .map(|i| (100 + i, vec![i as f32; usize::from(dim)]))
        .collect();
    let vec_payload = |rows: &[(u64, Vec<f32>)]| {
        let mut p = dim.to_le_bytes().to_vec();
        p.extend_from_slice(&(rows.len() as u32).to_le_bytes());
        for (id, v) in rows {
            p.extend_from_slice(&id.to_le_bytes());
            v.iter().for_each(|x| p.extend_from_slice(&x.to_le_bytes()));
        }
        p
    };
    let m = |dir: &[DirEntry], deleted: Vec<u64>| {
        RvfManifest {
            epoch: 1,
            dim,
            total_vectors: 6,
            profile: 0,
            metric_id: 2,
            dir: dir.to_vec(),
            deleted,
        }
        .encode()
    };
    // Superseded segment: present in the stream, absent from the final directory.
    let _orphan = seg(
        &mut file,
        0x01,
        1,
        &vec_payload(&[(999, vec![9.0; usize::from(dim)])]),
    );
    dir.push(seg(&mut file, 0x01, 2, &vec_payload(&vecs[..3])));
    let m1 = m(&dir, vec![]);
    seg(&mut file, 0x05, 3, &m1);
    dir.push(seg(&mut file, 0x01, 4, &vec_payload(&vecs[3..])));
    let m2 = m(&dir, vec![101]);
    seg(&mut file, 0x05, 5, &m2);
    (file, vecs)
}

#[test]
fn runtime_shaped_file_honours_the_final_manifest() {
    let dim = 3;
    let (file, vecs) = runtime_file(dim);
    let summary = inspect_tail(&file, file.len() as u64, &ImportLimits::default()).unwrap();
    assert_eq!(summary.deleted, vec![101]);
    // The superseded segment and the deleted id are not upserts.
    assert_eq!((summary.planned_rows, summary.estimated_upserts()), (6, 5));
    let (back, _) = import(&file, spec(dim, Metric::Cosine), Some(summary), 64).unwrap();
    let want: Vec<String> = vecs
        .iter()
        .map(|v| v.0)
        .filter(|&i| i != 101)
        .map(|i| i.to_string())
        .collect();
    let got: Vec<String> = back.iter().map(|r| r.id.clone()).collect();
    assert_eq!(got, want);
    assert!(back
        .iter()
        .all(|r| r.metadata.is_none() && r.values.len() == 3));
}

#[test]
fn inspect_tail_accepts_a_partial_tail_and_refuses_a_short_one() {
    let dim = 8;
    let file = export(&rows(200, dim, 6), dim, Metric::L2, 50);
    let lim = ImportLimits::default();
    let full = inspect_tail(&file, file.len() as u64, &lim).unwrap();
    let part = inspect_tail(&file[file.len() - 300..], file.len() as u64, &lim).unwrap();
    assert_eq!(full, part);
    assert_eq!(
        inspect_tail(&file[file.len() - 40..], file.len() as u64, &lim).unwrap_err(),
        ImportError::NoManifest
    );
}

#[test]
fn hostile_mutations_never_panic() {
    let dim = 4;
    let file = export(&rows(60, dim, 8), dim, Metric::Cosine, 16);
    let snap_obj = {
        let chain = WitnessChain::new(TENANT_A).unwrap();
        write_snapshot(
            TENANT_A,
            &rows(30, dim, 9),
            dim,
            1,
            &chain,
            SnapshotLimits::default(),
            None,
        )
        .object
    };
    let orig = inspect_tail(&file, file.len() as u64, &ImportLimits::default()).unwrap();
    let mut rng = Rng(0x5eed);
    for _ in 0..3_000 {
        let mut f = file.clone();
        for _ in 0..1 + rng.next() % 4 {
            let i = (rng.next() as usize) % f.len();
            f[i] = rng.next() as u8;
        }
        if rng.next().is_multiple_of(4) {
            f.truncate((rng.next() as usize) % f.len());
        }
        let piece = 1 + (rng.next() as usize) % 512;
        // Mutated bytes against the original file's summary (the
        // interesting mismatch case) and against their own tail.
        let _ = import(&f, spec(dim, Metric::Cosine), Some(orig.clone()), piece);
        let _ = import(&f, spec(dim, Metric::Cosine), None, piece);
        let _ = inspect_tail(&f, f.len() as u64, &ImportLimits::default());
        let mut m = snap_obj.clone();
        let i = (rng.next() as usize) % m.len();
        m[i] = rng.next() as u8;
        let _ = SealedManifest::from_object(&m);
    }
}

#[test]
fn large_dim_export_round_trips_with_default_limits() {
    // 4096 dims × 1024 rows would be a 16.8 MB VEC_SEG; the exporter splits by bytes.
    let dim = 4096;
    let data = rows(1_100, dim, 12);
    let file = export(&data, dim, Metric::Dot, 1024);
    let spec = ImportSpec {
        dim,
        metric: Metric::Dot,
        limits: ImportLimits::default(),
    };
    let (back, _) = import(&file, spec, None, 1 << 20).unwrap();
    assert_rows_identical(&data, &back);
}

#[test]
fn inspect_tail_honours_alignment_padding_and_refuses_oversized_manifests_early() {
    let dim = 4;
    let data = rows(300, dim, 13);
    let plain = export(&data, dim, Metric::L2, 100);
    let lim = ImportLimits::default();
    let s0 = inspect_tail(&plain, plain.len() as u64, &lim).unwrap();

    // rvf-wire style: the final manifest carries alignment_pad = 7 plus 7
    // zero bytes (the pad is outside the content hash).
    let mut padded = plain.clone();
    let m = s0.manifest_offset as usize;
    padded[m + 0x3C..m + 0x40].copy_from_slice(&7u32.to_le_bytes());
    padded.extend_from_slice(&[0; 7]);
    let s1 = inspect_tail(&padded, padded.len() as u64, &lim).unwrap();
    assert_eq!(
        (s1.manifest_offset, s1.file_len),
        (s0.manifest_offset, s0.file_len + 7)
    );
    let (back, _) = import(&padded, spec(dim, Metric::L2), Some(s1), 333).unwrap();
    assert_rows_identical(&data, &back);

    // The manifest header's declared length is refused before hashing or
    // decoding its directory.
    let small = ImportLimits {
        max_segment_payload: 100,
        ..lim
    };
    assert!(matches!(
        inspect_tail(&plain, plain.len() as u64, &small),
        Err(ImportError::SegmentTooLarge(n)) if n > 100
    ));
    assert_eq!(
        ImportLimits::default().max_manifest_entries,
        (8 << 20) / 25,
        "consistent with one 8 MiB manifest"
    );
}

#[test]
fn directory_length_disagreeing_with_a_segment_is_refused() {
    let dim = 4;
    let file = export(&rows(40, dim, 2), dim, Metric::L2, 10);
    let mut s = inspect_tail(&file, file.len() as u64, &ImportLimits::default()).unwrap();
    // A summary whose directory claims a different payload length for the
    // first live VEC_SEG (planned_rows would no longer be enforced).
    s.vec_segs[0].1 += 12;
    assert_eq!(
        import(&file, spec(dim, Metric::L2), Some(s), 4096).unwrap_err(),
        ImportError::ManifestMismatch("directory length")
    );
}
