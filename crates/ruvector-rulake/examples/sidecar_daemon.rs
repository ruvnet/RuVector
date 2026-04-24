//! Minimal cache-sidecar daemon demonstrating the bundle protocol
//! (ADR-155 §Consequences).
//!
//! Watches a publish directory for updates to `table.rulake.json`
//! and calls `RuLake::refresh_from_bundle_dir` to keep the reader
//! cache coherent with whatever the publisher has signed off.
//!
//! Run with:
//!   cargo run --release -p ruvector-rulake --example sidecar_daemon
//!
//! The example simulates a full deployment in one process:
//!   - A "publisher" backend with a collection of vectors
//!   - A "reader" RuLake serving queries against the same collection
//!   - The publisher updates the backend and emits a new bundle
//!   - The reader's daemon polls, detects the witness change,
//!     invalidates, and the next query re-primes automatically
//!
//! A real deployment replaces the polling loop with an inotify
//! watcher or GCS object-change notification; the shape is identical.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ruvector_rulake::{cache::Consistency, LocalBackend, RefreshResult, RuLake};

fn main() {
    println!("=== ruLake sidecar daemon demo ===\n");

    // Shared publish directory. In production this would be a GCS
    // mount, an S3FS prefix, or an NFS share.
    let publish_dir =
        std::env::temp_dir().join(format!("rulake-sidecar-demo-{}", std::process::id()));
    std::fs::create_dir_all(&publish_dir).unwrap();
    println!("Publish directory: {}", publish_dir.display());

    // --- Publisher side ---
    let backend = Arc::new(LocalBackend::new("publisher"));
    backend
        .put_collection(
            "memories",
            8,
            (0..100).collect(),
            (0..100)
                .map(|i| (0..8).map(|j| (i + j) as f32 * 0.1).collect())
                .collect(),
        )
        .unwrap();

    let publisher_lake = RuLake::new(20, 42);
    publisher_lake.register_backend(backend.clone()).unwrap();
    let key = ("publisher".to_string(), "memories".to_string());

    publisher_lake.publish_bundle(&key, &publish_dir).unwrap();
    println!("[publisher] emitted initial table.rulake.json");

    // --- Reader side ---
    let reader_lake =
        Arc::new(RuLake::new(20, 42).with_consistency(Consistency::Eventual { ttl_ms: 60_000 }));
    reader_lake.register_backend(backend.clone()).unwrap();

    // --- Daemon goroutine (polling-based) ---
    let daemon_lake = Arc::clone(&reader_lake);
    let daemon_dir = publish_dir.clone();
    let daemon_key = key.clone();
    let daemon_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_handle = Arc::clone(&daemon_stop);
    let daemon = thread::spawn(move || {
        let poll_every = Duration::from_millis(100);
        let deadline = Instant::now() + Duration::from_secs(5);
        while !stop_handle.load(std::sync::atomic::Ordering::Relaxed) && Instant::now() < deadline {
            match daemon_lake.refresh_from_bundle_dir(&daemon_key, &daemon_dir) {
                Ok(RefreshResult::Invalidated) => {
                    println!("[daemon]    bundle rotated — cache invalidated");
                }
                Ok(RefreshResult::UpToDate) => {} // quiet
                Ok(RefreshResult::BundleMissing) => {
                    eprintln!("[daemon]    sidecar missing at publish dir");
                }
                Err(e) => eprintln!("[daemon]    refresh error: {e}"),
            }
            thread::sleep(poll_every);
        }
    });

    // --- Workload: read a bunch, then publisher mutates, then read more ---
    let q = vec![0.5f32; 8];

    for _ in 0..10 {
        let _ = reader_lake
            .search_one("publisher", "memories", &q, 3)
            .unwrap();
    }
    let stats1 = reader_lake.cache_stats();
    println!(
        "[reader]    after 10 warm queries: hit_rate={:.3} primes={}",
        stats1.hit_rate().unwrap_or(0.0),
        stats1.primes
    );

    // Publisher side-effect: the backend data changed. Re-publish.
    backend.append("memories", 999, vec![9.0; 8]).unwrap();
    publisher_lake.publish_bundle(&key, &publish_dir).unwrap();
    println!("[publisher] mutated backend + re-published bundle");

    // Give the daemon a few poll cycles to see it.
    thread::sleep(Duration::from_millis(300));

    // Next queries: first one re-primes (daemon dropped the pointer),
    // the rest are warm again.
    for _ in 0..10 {
        let _ = reader_lake
            .search_one("publisher", "memories", &q, 3)
            .unwrap();
    }
    let stats2 = reader_lake.cache_stats();
    println!(
        "[reader]    after mutation: hit_rate={:.3} primes={} invalidations={}",
        stats2.hit_rate().unwrap_or(0.0),
        stats2.primes,
        stats2.invalidations
    );

    assert!(
        stats2.primes > stats1.primes,
        "daemon should have triggered a re-prime"
    );

    daemon_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    daemon.join().unwrap();

    let _ = std::fs::remove_dir_all(&publish_dir);
    println!("\n✓ daemon + reader stayed coherent through the bundle rotation");
}
