//! Acceptance test for the disk-backed rerank path landed by this PR.
//!
//! Closes the deferral from PR #384: previously `with_originals_in_memory(false)`
//! was rejected at `build()` time. After this PR, the originals are written to
//! a sidecar (`<storage_path>/originals.bin`), the in-memory `FlatVectors` is
//! dropped, and the rerank pass reads originals back via mmap. Net DRAM drops
//! to (codes + graph) only — the 17.5× compression target the research
//! roadmap projected.
//!
//! What this test verifies:
//!
//!   1. `with_originals_in_memory(false)` is honored — the index reports
//!      `originals_on_disk() == true` and `originals_memory_bytes() == 0`.
//!   2. Recall@10 is still ≥ 0.85 on the same dataset shape PR #384 used.
//!   3. Codes-vs-originals memory ratio reflects the codes savings: at
//!      D=128 the in-memory variant should give >16× compression of codes
//!      relative to originals, and the disk-backed variant should give
//!      *infinite* compression (originals heap = 0).
//!   4. Save → drop → load round-trip preserves search results
//!      bit-identically (ADR-154 determinism).
//!   5. Without `storage_path` set, `build()` returns `InvalidConfig`
//!      when `with_originals_in_memory(false)` is requested.
//!
//! `rabitq` feature gate matches the existing integration test in
//! `quantizer_search_uses_codes.rs` — the disk-backed path itself is
//! feature-agnostic, but the test uses RaBitQ for code-side compression
//! parity with PR #384's measurements.
#![cfg(feature = "rabitq")]

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_diskann::{DiskAnnConfig, DiskAnnError, DiskAnnIndex, QuantizerKind};
use tempfile::tempdir;

fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn brute_force_topk(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum();
            (i, d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

/// Helper: build a DiskAnnIndex with the given originals mode and a fresh
/// storage_path inside the supplied tempdir. Returns the storage path so
/// callers can drop+reload without re-deriving it.
fn build_index(
    dir: &std::path::Path,
    vectors: &[Vec<f32>],
    keep_in_memory: bool,
    rerank_factor: usize,
) -> (DiskAnnIndex, std::path::PathBuf) {
    let dim = vectors[0].len();
    let storage = dir.join("idx");
    let config = DiskAnnConfig {
        dim,
        max_degree: 64,
        build_beam: 256,
        search_beam: 512,
        alpha: 1.2,
        storage_path: Some(storage.clone()),
        ..Default::default()
    }
    .with_rabitq_seed(0xBEEF)
    .with_quantizer_kind(QuantizerKind::Rabitq)
    .with_rerank_factor(rerank_factor)
    .with_originals_in_memory(keep_in_memory);

    let mut index = DiskAnnIndex::new(config);
    let entries: Vec<(String, Vec<f32>)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("v{i}"), v.clone()))
        .collect();
    index.insert_batch(entries).unwrap();
    index.build().unwrap();
    (index, storage)
}

#[test]
fn disk_backed_originals_drop_dram_to_zero() {
    // Smaller dataset (D=128, n=2000) keeps wall-time under control while
    // still hitting the 17.5× target from PR #384's measurements.
    let dim = 128;
    let n = 2_000;
    let vectors = random_unit_vectors(n, dim, 7);

    let dir = tempdir().unwrap();

    // ---- In-memory variant (baseline) ---------------------------------
    let (mem_index, _) = build_index(&dir.path().join("mem"), &vectors, true, 8);
    let mem_codes = mem_index.codes_memory_bytes();
    let mem_orig = mem_index.originals_memory_bytes();
    assert!(
        !mem_index.originals_on_disk(),
        "in-memory mode flagged on-disk"
    );
    assert!(
        mem_orig > 0,
        "in-memory variant should hold originals in DRAM"
    );

    // ---- Disk-backed variant ------------------------------------------
    let (disk_index, _) = build_index(&dir.path().join("disk"), &vectors, false, 8);
    let disk_codes = disk_index.codes_memory_bytes();
    let disk_orig = disk_index.originals_memory_bytes();

    // The disk-backed variant must report 0 DRAM bytes for originals — the
    // mmap pages are kernel-owned and only paged in on demand.
    assert_eq!(
        disk_orig, 0,
        "disk-backed originals should not occupy DRAM (got {disk_orig} bytes)"
    );
    assert!(
        disk_index.originals_on_disk(),
        "disk-backed index should report originals_on_disk() == true"
    );
    // Codes slab should be unchanged between the two variants — the
    // originals knob doesn't touch the quantizer.
    assert_eq!(
        disk_codes, mem_codes,
        "codes slab differs between in-memory and disk-backed variants \
         (disk={disk_codes} mem={mem_codes})"
    );

    // ---- Compression ratio --------------------------------------------
    // f32 originals at D=128 occupy 512B/vec. Codes are 16+4=20B/vec for
    // RaBitQ at this dim. In-memory variant compresses by ~25×; disk-backed
    // variant goes to "0 DRAM cost" because originals leave DRAM entirely.
    let in_mem_ratio = mem_orig as f32 / mem_codes as f32;
    eprintln!(
        "[disk-backed mem] codes={disk_codes}B originals_dram={disk_orig}B \
         (vs in-mem variant: codes={mem_codes}B originals={mem_orig}B, ratio={in_mem_ratio:.2}×)"
    );
    assert!(
        in_mem_ratio >= 16.0,
        "in-memory codes-vs-originals ratio {in_mem_ratio:.2}× < 16× target \
         (codes={mem_codes}B, originals={mem_orig}B)"
    );
}

#[test]
fn disk_backed_search_meets_recall_floor() {
    // Same recall floor (≥ 0.85) PR #384 set for the in-memory variant;
    // the disk-backed path produces byte-identical f32 reads, so recall
    // must match.
    let dim = 128;
    let n = 1_000;
    let k = 10;
    let vectors = random_unit_vectors(n, dim, 0xC0DE_C0DE);

    let dir = tempdir().unwrap();
    let (index, _) = build_index(&dir.path(), &vectors, false, 40);

    let queries = random_unit_vectors(30, dim, 0xACE);
    let mut recall_sum = 0.0f32;
    for query in &queries {
        let gt: std::collections::HashSet<usize> =
            brute_force_topk(&vectors, query, k).into_iter().collect();
        let results = index.search(query, k).unwrap();
        let found: std::collections::HashSet<usize> = results
            .iter()
            .map(|r| {
                r.id.trim_start_matches('v')
                    .parse::<usize>()
                    .expect("v-prefixed id")
            })
            .collect();
        let recall = gt.intersection(&found).count() as f32 / k as f32;
        recall_sum += recall;
    }
    let avg_recall = recall_sum / queries.len() as f32;
    eprintln!("[disk-backed recall] recall@{k} = {avg_recall:.3}");
    assert!(
        avg_recall >= 0.85,
        "disk-backed RaBitQ recall@{k} = {avg_recall:.3} < 0.85 floor"
    );
}

#[test]
fn disk_backed_in_memory_results_match_bitwise() {
    // Determinism guard: the f32 bytes the disk-backed rerank reads from
    // mmap must be bit-identical to the bytes the in-memory rerank reads
    // from a Vec — both come from the same `FlatVectors` write path. We
    // verify by:
    //   1. building a disk-backed index (originals → sidecar via mmap)
    //   2. dropping it and reloading via `DiskAnnIndex::load`
    //   3. patching the saved config to flip `keep_originals_in_memory` to
    //      true and removing the sidecar, then loading again as in-memory
    //      (which reads vectors.bin into a Vec)
    //   4. the codes file (PQ; chosen because RaBitQ codes don't persist
    //      yet — see `index.rs::save()` notes) is identical, so traversal
    //      is identical, so the candidate set is identical, so any
    //      difference must come from rerank arithmetic.
    //
    // Both load paths read from the same on-disk f32 bytes, so the rerank
    // distances must agree bit-for-bit. If this test ever fires, mmap is
    // being read with the wrong endianness/alignment.
    let dim = 64;
    let n = 500;
    let k = 10;
    let vectors = random_unit_vectors(n, dim, 0xDEAD);

    let dir = tempdir().unwrap();
    let storage = dir.path().join("idx");

    // PQ-backed disk-backed build. PQ codes persist via save(), so a load
    // round-trip preserves the traversal hot path.
    let config = DiskAnnConfig {
        dim,
        max_degree: 32,
        build_beam: 96,
        search_beam: 96,
        alpha: 1.2,
        pq_subspaces: 8,
        pq_iterations: 5,
        storage_path: Some(storage.clone()),
        ..Default::default()
    }
    .with_rerank_factor(8)
    .with_originals_in_memory(false);

    let mut idx = DiskAnnIndex::new(config);
    let entries: Vec<(String, Vec<f32>)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("v{i}"), v.clone()))
        .collect();
    idx.insert_batch(entries).unwrap();
    idx.build().unwrap();
    assert!(idx.originals_on_disk());

    // Snapshot disk-backed search results, then drop the index.
    let queries = random_unit_vectors(10, dim, 0x1234);
    let disk_results: Vec<Vec<(String, u32)>> = queries
        .iter()
        .map(|q| {
            idx.search(q, k)
                .unwrap()
                .into_iter()
                .map(|r| (r.id, r.distance.to_bits()))
                .collect()
        })
        .collect();
    drop(idx);

    // Convert the saved index to the in-memory layout by removing the
    // sidecar and flipping the saved config flag. The graph + PQ codes
    // are unchanged, so traversal results match the disk-backed run.
    std::fs::remove_file(storage.join("originals.bin")).unwrap();
    let cfg_path = storage.join("config.json");
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    cfg["keep_originals_in_memory"] = serde_json::json!(true);
    std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg).unwrap()).unwrap();

    let mem_index = DiskAnnIndex::load(&storage).unwrap();
    assert!(
        !mem_index.originals_on_disk(),
        "expected in-memory after sidecar removal"
    );

    for (q, want) in queries.iter().zip(disk_results.iter()) {
        let got: Vec<(String, u32)> = mem_index
            .search(q, k)
            .unwrap()
            .into_iter()
            .map(|r| (r.id, r.distance.to_bits()))
            .collect();
        assert_eq!(
            got, *want,
            "rerank arithmetic differs between disk-backed and in-memory paths"
        );
    }
}

#[test]
fn disk_backed_save_load_round_trip_preserves_results() {
    // Build with disk-backed mode → save → drop → load → search must
    // match the pre-drop search results exactly. This catches bugs where
    // the load path silently re-reads `vectors.bin` as in-memory, which
    // would still give correct results but break the compression target.
    //
    // Uses PQ rather than RaBitQ because PQ codes persist via save() —
    // RaBitQ persistence is a future PR (see save() comments in
    // index.rs). Without that, a loaded RaBitQ-built index runs the
    // legacy f32 path and can't be compared to the pre-drop run.
    let dim = 64;
    let n = 300;
    let k = 5;
    let vectors = random_unit_vectors(n, dim, 0xABCD);

    let dir = tempdir().unwrap();
    let storage = dir.path().join("idx");

    let config = DiskAnnConfig {
        dim,
        max_degree: 32,
        build_beam: 64,
        search_beam: 64,
        alpha: 1.2,
        pq_subspaces: 8,
        pq_iterations: 5,
        storage_path: Some(storage.clone()),
        ..Default::default()
    }
    .with_rerank_factor(4)
    .with_originals_in_memory(false);

    // Snapshot search results before drop.
    let queries = random_unit_vectors(8, dim, 0x9999);
    let pre_results: Vec<Vec<(String, u32)>> = {
        let mut idx = DiskAnnIndex::new(config);
        let entries: Vec<(String, Vec<f32>)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();
        idx.insert_batch(entries).unwrap();
        idx.build().unwrap();
        assert!(idx.originals_on_disk());
        queries
            .iter()
            .map(|q| {
                idx.search(q, k)
                    .unwrap()
                    .into_iter()
                    .map(|r| (r.id, r.distance.to_bits()))
                    .collect()
            })
            .collect()
        // idx dropped here
    };

    // Reload from disk — the originals.bin sidecar should be picked up
    // automatically because the saved config has keep_originals_in_memory =
    // false.
    let loaded = DiskAnnIndex::load(&storage).unwrap();
    assert!(
        loaded.originals_on_disk(),
        "loaded index should detect originals.bin sidecar and stay disk-backed"
    );
    for (q, want) in queries.iter().zip(pre_results.iter()) {
        let got: Vec<(String, u32)> = loaded
            .search(q, k)
            .unwrap()
            .into_iter()
            .map(|r| (r.id, r.distance.to_bits()))
            .collect();
        assert_eq!(
            got, *want,
            "search results changed across save → load round-trip"
        );
    }
}

#[test]
fn disk_backed_save_load_round_trip_preserves_results_rabitq() {
    // RaBitQ-flavored sibling of the PQ round-trip test above. Closes the
    // limitation flagged in PR #385: previously a reloaded RaBitQ-built
    // index dropped its rotation matrix + binary codes and silently fell
    // back to the f32 traversal path, so a save → drop → load round-trip
    // changed search results. After this PR the rabitq.bin sidecar
    // persists `(dim, seed)` plus the codes; load() replays the rotation
    // deterministically (ADR-154) and search results must be bit-identical.
    let dim = 64;
    let n = 300;
    let k = 5;
    let vectors = random_unit_vectors(n, dim, 0xABCD);

    let dir = tempdir().unwrap();
    let storage = dir.path().join("idx");

    let config = DiskAnnConfig {
        dim,
        max_degree: 32,
        build_beam: 64,
        search_beam: 64,
        alpha: 1.2,
        storage_path: Some(storage.clone()),
        ..Default::default()
    }
    .with_rabitq_seed(0xFEED_FACE)
    .with_quantizer_kind(QuantizerKind::Rabitq)
    .with_rerank_factor(4)
    .with_originals_in_memory(false);

    let queries = random_unit_vectors(8, dim, 0x9999);
    let pre_results: Vec<Vec<(String, u32)>> = {
        let mut idx = DiskAnnIndex::new(config);
        let entries: Vec<(String, Vec<f32>)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();
        idx.insert_batch(entries).unwrap();
        idx.build().unwrap();
        assert_eq!(idx.quantizer_kind(), QuantizerKind::Rabitq);
        assert!(idx.codes_memory_bytes() > 0, "RaBitQ codes slab is empty");
        assert!(idx.originals_on_disk());
        queries
            .iter()
            .map(|q| {
                idx.search(q, k)
                    .unwrap()
                    .into_iter()
                    .map(|r| (r.id, r.distance.to_bits()))
                    .collect()
            })
            .collect()
        // idx dropped here — proves load() is restoring state from disk
        // rather than reusing in-memory residuals.
    };

    // Reload. The rabitq.bin sidecar must be picked up so the loaded
    // index keeps the codes-driven traversal path (not the f32 fallback
    // that PR #383/#384/#385 silently reverted to).
    let loaded = DiskAnnIndex::load(&storage).unwrap();
    assert_eq!(
        loaded.quantizer_kind(),
        QuantizerKind::Rabitq,
        "loaded index should detect rabitq.bin sidecar"
    );
    assert!(loaded.originals_on_disk());
    assert!(
        loaded.codes_memory_bytes() > 0,
        "RaBitQ codes slab not restored on load — search will fall back to f32"
    );
    for (q, want) in queries.iter().zip(pre_results.iter()) {
        let got: Vec<(String, u32)> = loaded
            .search(q, k)
            .unwrap()
            .into_iter()
            .map(|r| (r.id, r.distance.to_bits()))
            .collect();
        assert_eq!(
            got, *want,
            "RaBitQ search results changed across save → load round-trip — \
             rotation matrix + codes are not being persisted"
        );
    }
}

#[test]
fn v1_pq_index_without_quantizer_kind_tag_still_loads() {
    // Back-compat guard: v1 indexes (saved by PR #383/#384/#385) don't
    // have the new `quantizer_kind` JSON tag. The load path must fall
    // back to the file-presence heuristic ("pq.bin exists → PQ") so
    // existing on-disk PQ indexes keep loading byte-identically.
    let dim = 32;
    let n = 200;
    let k = 5;
    let vectors = random_unit_vectors(n, dim, 0xF00D);

    let dir = tempdir().unwrap();
    let storage = dir.path().join("idx");

    let config = DiskAnnConfig {
        dim,
        max_degree: 16,
        build_beam: 32,
        search_beam: 32,
        alpha: 1.2,
        pq_subspaces: 4,
        pq_iterations: 5,
        storage_path: Some(storage.clone()),
        ..Default::default()
    };

    let queries = random_unit_vectors(5, dim, 0x1357);
    let pre_results: Vec<Vec<(String, u32)>> = {
        let mut idx = DiskAnnIndex::new(config);
        let entries: Vec<(String, Vec<f32>)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("v{i}"), v.clone()))
            .collect();
        idx.insert_batch(entries).unwrap();
        idx.build().unwrap();
        queries
            .iter()
            .map(|q| {
                idx.search(q, k)
                    .unwrap()
                    .into_iter()
                    .map(|r| (r.id, r.distance.to_bits()))
                    .collect()
            })
            .collect()
    };

    // Strip `quantizer_kind` from config.json to simulate a v1 save.
    let cfg_path = storage.join("config.json");
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    cfg.as_object_mut().unwrap().remove("quantizer_kind");
    std::fs::write(&cfg_path, serde_json::to_string_pretty(&cfg).unwrap()).unwrap();

    // Sanity-check the tag is gone.
    let cfg_check: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    assert!(cfg_check.get("quantizer_kind").is_none());

    // Load and search — must still go through the PQ path because pq.bin
    // exists, and results must be byte-identical.
    let loaded = DiskAnnIndex::load(&storage).unwrap();
    assert_eq!(loaded.quantizer_kind(), QuantizerKind::Pq);
    for (q, want) in queries.iter().zip(pre_results.iter()) {
        let got: Vec<(String, u32)> = loaded
            .search(q, k)
            .unwrap()
            .into_iter()
            .map(|r| (r.id, r.distance.to_bits()))
            .collect();
        assert_eq!(got, *want, "v1 PQ index search results changed on load");
    }
}

#[test]
fn rabitq_load_rejects_missing_sidecar() {
    // If the saved config tags `quantizer_kind = rabitq` but the sidecar
    // is missing (e.g. a v1 index from before this PR, or a corrupted
    // index where someone deleted rabitq.bin), load() must surface a
    // clear `InvalidConfig` error rather than silently falling back to
    // the f32 path. The brief: "If the saved magic is RaBitQ but the
    // section is missing, return a clear error (don't silently fall back)."
    let dim = 32;
    let n = 100;
    let vectors = random_unit_vectors(n, dim, 0xFADE);

    let dir = tempdir().unwrap();
    let storage = dir.path().join("idx");

    let config = DiskAnnConfig {
        dim,
        max_degree: 16,
        build_beam: 32,
        search_beam: 32,
        alpha: 1.2,
        storage_path: Some(storage.clone()),
        ..Default::default()
    }
    .with_rabitq_seed(0xBEEF)
    .with_quantizer_kind(QuantizerKind::Rabitq)
    .with_rerank_factor(2);

    let mut idx = DiskAnnIndex::new(config);
    let entries: Vec<(String, Vec<f32>)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("v{i}"), v.clone()))
        .collect();
    idx.insert_batch(entries).unwrap();
    idx.build().unwrap();
    drop(idx);

    // Simulate corruption: delete the rabitq sidecar but leave the
    // config tag claiming RaBitQ.
    std::fs::remove_file(storage.join("rabitq.bin")).unwrap();

    match DiskAnnIndex::load(&storage) {
        Ok(_) => panic!("expected load to fail when rabitq sidecar is missing"),
        Err(DiskAnnError::InvalidConfig(msg)) => {
            assert!(
                msg.contains("rabitq"),
                "expected error mentioning rabitq, got: {msg}"
            );
        }
        Err(other) => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn disk_backed_without_storage_path_rejected() {
    // The whole point of storage_path is "where do I spill the originals".
    // Without it, disk-backed mode has nowhere to write — surface an
    // InvalidConfig at build() time rather than silently degrading.
    let dim = 32;
    let n = 100;
    let vectors = random_unit_vectors(n, dim, 0xFADE);

    let config = DiskAnnConfig {
        dim,
        max_degree: 16,
        build_beam: 32,
        search_beam: 32,
        alpha: 1.2,
        // storage_path intentionally None
        ..Default::default()
    }
    .with_rabitq_seed(1)
    .with_quantizer_kind(QuantizerKind::Rabitq)
    .with_originals_in_memory(false);

    let mut index = DiskAnnIndex::new(config);
    let entries: Vec<(String, Vec<f32>)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (format!("v{i}"), v.clone()))
        .collect();
    index.insert_batch(entries).unwrap();
    let err = index.build().unwrap_err();
    match err {
        DiskAnnError::InvalidConfig(msg) => {
            assert!(
                msg.contains("storage_path"),
                "expected error mentioning storage_path, got: {msg}"
            );
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}
