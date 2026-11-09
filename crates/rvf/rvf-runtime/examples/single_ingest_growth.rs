//! File growth under many single-vector ingests (#1382).
//!
//! Calls `ingest_batch` once per vector and reports the file size, the
//! segment-directory size by type, and the live vector bytes at checkpoints,
//! then reopens the store and checks every vector reads back.
//!
//! Run: cargo run --release --example single_ingest_growth -p rvf-runtime -- [n] [dim] [meta]

use std::collections::BTreeMap;
use std::time::Instant;

use rvf_runtime::options::{
    DistanceMetric, MetadataEntry, MetadataValue, QueryOptions, RvfOptions,
};
use rvf_runtime::RvfStore;

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn vector(i: u64, dim: usize) -> Vec<f32> {
    (0..dim as u64)
        .map(|j| {
            (splitmix(i.wrapping_mul(1_000_003).wrapping_add(j)) >> 40) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn type_name(t: u8) -> &'static str {
    match t {
        0x01 => "vec",
        0x02 => "index",
        0x05 => "manifest",
        0x07 => "meta",
        0x0A => "witness",
        _ => "other",
    }
}

fn dir_summary(store: &RvfStore) -> String {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for &(_, _, _, t) in store.segment_dir() {
        *counts.entry(type_name(t)).or_default() += 1;
    }
    counts
        .iter()
        .map(|(k, v)| format!("{k}={v}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let nums: Vec<usize> = args.iter().filter_map(|a| a.parse().ok()).collect();
    let n = nums.first().copied().unwrap_or(3000);
    let dim = nums.get(1).copied().unwrap_or(128);
    let with_meta = args.iter().any(|a| a == "meta");

    let dir = std::env::temp_dir().join(format!("rvf-growth-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("growth.rvf");
    let mut store = RvfStore::create(
        &path,
        RvfOptions {
            dimension: dim as u16,
            metric: DistanceMetric::L2,
            ..Default::default()
        },
    )
    .unwrap();

    println!("n={n} dim={dim} meta={with_meta}");
    println!("| single ingests | file size | live vector bytes | dir entries | ingest wall |");
    println!("|---|---|---|---|---|");
    let step = (n / 6).max(1);
    let t = Instant::now();
    for i in 0..n as u64 {
        let v = vector(i, dim);
        let meta = [MetadataEntry {
            field_id: 0,
            value: MetadataValue::U64(i % 7),
        }];
        store
            .ingest_batch(&[v.as_slice()], &[i], with_meta.then_some(&meta[..]))
            .unwrap();
        let done = i as usize + 1;
        if done.is_multiple_of(step) || done == n {
            let size = std::fs::metadata(&path).unwrap().len();
            println!(
                "| {done} | {:.2} MB | {:.2} MB | {} | {:.1} s |",
                size as f64 / 1e6,
                (done * dim * 4) as f64 / 1e6,
                dir_summary(&store),
                t.elapsed().as_secs_f64()
            );
        }
    }
    store.close().unwrap();

    let t = Instant::now();
    let store = RvfStore::open(&path).unwrap();
    let open_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(
        store.status().total_vectors as usize,
        n,
        "vector count after reopen"
    );
    let opts = QueryOptions {
        force_exact: true,
        ..Default::default()
    };
    for i in (0..n as u64).step_by((n / 50).max(1)) {
        let hit = store.query(&vector(i, dim), 1, &opts).unwrap();
        assert_eq!(hit[0].id, i, "nearest neighbour of vector {i} after reopen");
        assert!(hit[0].distance <= 1e-6);
        if with_meta {
            assert!(
                store.get_metadata(i).is_some(),
                "metadata for {i} after reopen"
            );
        }
    }
    println!(
        "reopen ok in {open_ms:.1} ms: {} vectors, every sampled vector reads back",
        n
    );
    drop(store);
    let _ = std::fs::remove_dir_all(&dir);
}
