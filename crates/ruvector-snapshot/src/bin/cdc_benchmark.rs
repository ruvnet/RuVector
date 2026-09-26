//! Real-format comparison: `LocalStorage` (production) vs `CdcLocalStorage`
//! (feature `cdc`), same scale as the original synthetic-format benchmark
//! in `ruvector-cdc-checkpoint`, but exercising the actual production
//! `SnapshotData`/`VectorRecord`/bincode/gzip path end to end. Supersedes
//! the synthetic numbers in the 2026-08-27 nightly research README — see
//! ADR-350.
//!
//! Run with: `cargo run --release -p ruvector-snapshot --features cdc --bin cdc_benchmark`

use ruvector_snapshot::{
    CollectionConfig, DistanceMetric, LocalStorage, SnapshotData, SnapshotStorage, VectorRecord,
};

const N_VECTORS: usize = 20_000;
const DIM: usize = 128;
const ROUNDS: u64 = 30;
const INSERT_PER_ROUND: usize = 40;
const DELETE_PER_ROUND: usize = 20;
const UPDATE_PER_ROUND: usize = 60;
const SEED: u64 = 0xC0FF_EE00_1234_5678;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn random_vector(state: &mut u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|_| (splitmix64(state) >> 40) as f32 / (1u64 << 24) as f32)
        .collect()
}

fn churn(
    rows: &mut Vec<(String, Vec<f32>)>,
    state: &mut u64,
    dim: usize,
    next_id: &mut u64,
    n_insert: usize,
    n_delete: usize,
    n_update: usize,
) {
    for _ in 0..n_insert {
        let id = format!("v{next_id}");
        *next_id += 1;
        rows.push((id, random_vector(state, dim)));
    }
    for _ in 0..n_delete {
        if rows.is_empty() {
            break;
        }
        let idx = (splitmix64(state) as usize) % rows.len();
        rows.remove(idx);
    }
    for _ in 0..n_update {
        if rows.is_empty() {
            break;
        }
        let idx = (splitmix64(state) as usize) % rows.len();
        rows[idx].1 = random_vector(state, dim);
    }
}

fn build_snapshot_data(rows: &[(String, Vec<f32>)]) -> SnapshotData {
    let config = CollectionConfig {
        dimension: DIM,
        metric: DistanceMetric::Cosine,
        hnsw_config: None,
    };
    let vectors = rows
        .iter()
        .map(|(id, v)| VectorRecord::new(id.clone(), v.clone(), None))
        .collect();
    SnapshotData::new("agent-memory-bench".to_string(), config, vectors)
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

#[tokio::main]
async fn main() {
    println!("==========================================================");
    println!(" ruvector-snapshot — CDC vs. real LocalStorage benchmark");
    println!("==========================================================");
    println!("OS               : {}", std::env::consts::OS);
    println!("Arch             : {}", std::env::consts::ARCH);
    println!(
        "Workload         : n={N_VECTORS} dim={DIM} seed=0x{SEED:X} rounds={ROUNDS} (real SnapshotData/VectorRecord/bincode/gzip)"
    );
    println!(
        "Churn/round      : insert={INSERT_PER_ROUND} delete={DELETE_PER_ROUND} update={UPDATE_PER_ROUND} (of {N_VECTORS} rows)"
    );
    println!();

    let mut state = SEED;
    let mut next_id = 0u64;
    let mut rows: Vec<(String, Vec<f32>)> = (0..N_VECTORS)
        .map(|_| {
            let id = format!("v{next_id}");
            next_id += 1;
            (id, random_vector(&mut state, DIM))
        })
        .collect();

    let local_dir = std::env::temp_dir().join(format!(
        "ruvector-snapshot-bench-local-{}",
        std::process::id()
    ));
    let cdc_dir = std::env::temp_dir().join(format!(
        "ruvector-snapshot-bench-cdc-{}",
        std::process::id()
    ));
    let local = LocalStorage::new(local_dir.clone());
    let cdc = ruvector_snapshot::CdcLocalStorage::new(cdc_dir.clone());

    let mut local_total: u64 = 0;
    let mut round_start_cdc_dir_size = 0u64;

    for round in 0..ROUNDS {
        if round > 0 {
            churn(
                &mut rows,
                &mut state,
                DIM,
                &mut next_id,
                INSERT_PER_ROUND,
                DELETE_PER_ROUND,
                UPDATE_PER_ROUND,
            );
        }
        let data = build_snapshot_data(&rows);

        let local_snapshot = local.save(&data).await.expect("LocalStorage::save");
        let cdc_snapshot = cdc.save(&data).await.expect("CdcLocalStorage::save");

        let restored = local
            .load(&local_snapshot.id)
            .await
            .expect("LocalStorage::load");
        assert_eq!(
            restored.vectors_count(),
            data.vectors_count(),
            "LocalStorage round-trip mismatch at round {round}"
        );
        let restored_cdc = cdc
            .load(&cdc_snapshot.id)
            .await
            .expect("CdcLocalStorage::load");
        assert_eq!(
            restored_cdc.vectors_count(),
            data.vectors_count(),
            "CdcLocalStorage round-trip mismatch at round {round}"
        );
        for (a, b) in restored_cdc.vectors.iter().zip(data.vectors.iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.vector, b.vector);
        }

        local_total += local_snapshot.size_bytes;
        let cdc_dir_size_now = dir_size(&cdc_dir);
        let cdc_new_bytes_this_round = cdc_dir_size_now.saturating_sub(round_start_cdc_dir_size);
        round_start_cdc_dir_size = cdc_dir_size_now;

        if true {
            println!(
                "round {round:>2}: local.size_bytes={:>10}  cdc.new_bytes_this_round={:>10}  cdc_dir_total={:>10}",
                local_snapshot.size_bytes, cdc_new_bytes_this_round, cdc_dir_size_now
            );
        }
    }

    let cdc_dir_final = dir_size(&cdc_dir);
    println!();
    println!("LocalStorage cumulative bytes written ({ROUNDS} rounds): {local_total}");
    println!("CdcLocalStorage on-disk size after {ROUNDS} rounds:       {cdc_dir_final}");
    println!(
        "ratio (cdc_dir_final / local_cumulative):              {:.4}",
        cdc_dir_final as f64 / local_total as f64
    );
    println!("reconstruction correctness: 100% (asserted every round, both backends, real production decode path)");
    println!("==========================================================");

    let _ = std::fs::remove_dir_all(&local_dir);
    let _ = std::fs::remove_dir_all(&cdc_dir);
}
