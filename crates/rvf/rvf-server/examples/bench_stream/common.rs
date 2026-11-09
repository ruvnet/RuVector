//! Shared helpers for the stream benchmarks.

use ed25519_dalek::SigningKey;
use rvf_runtime::progressive_stream::build::{build_stream_store, BuildParams, StreamDataset};
use rvf_server::stream::{self, StreamServer, StreamServerConfig};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const KEY: [u8; 32] = [0xA7; 32];

pub fn loadavg() -> String {
    match std::fs::read_to_string("/proc/loadavg") {
        Ok(s) => s.trim().to_string(),
        Err(_) => "n/a".to_string(),
    }
}

pub fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter()
        .position(|x| x == name)
        .and_then(|i| a.get(i + 1))
        .cloned()
        .unwrap_or(default.into())
}

pub fn pubkey() -> [u8; 32] {
    SigningKey::from_bytes(&KEY).verifying_key().to_bytes()
}

pub fn exact_topk(ds: &StreamDataset, queries: &[Vec<f32>], k: usize) -> Vec<Vec<u64>> {
    let mut out = vec![Vec::new(); queries.len()];
    std::thread::scope(|s| {
        for (qs, os) in queries.chunks(8).zip(out.chunks_mut(8)) {
            s.spawn(move || {
                for (q, o) in qs.iter().zip(os.iter_mut()) {
                    let mut d: Vec<(f32, u64)> = (0..ds.len())
                        .map(|i| {
                            (
                                ds.vector(i)
                                    .iter()
                                    .zip(q)
                                    .map(|(a, b)| (a - b) * (a - b))
                                    .sum(),
                                i as u64,
                            )
                        })
                        .collect();
                    d.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
                    d.truncate(k);
                    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    *o = d.into_iter().map(|x| x.1).collect();
                }
            });
        }
    });
    out
}

pub fn recall(got: &[u64], truth: &[u64]) -> f64 {
    got.iter().filter(|x| truth.contains(x)).count() as f64 / truth.len() as f64
}

pub struct Fix {
    pub path: PathBuf,
    pub ds: StreamDataset,
    pub file_id: [u8; 16],
}

pub fn fixture(n: usize, dim: usize, clusters: usize, cache: &Path) -> Fix {
    let ds = StreamDataset::clustered(n, dim, clusters, 0.4, 7);
    std::fs::create_dir_all(cache).unwrap();
    let path = cache.join(format!("{n}x{dim}.rvf"));
    if !path.exists() {
        let t = Instant::now();
        let tmp = path.with_extension("tmp");
        let _ = std::fs::remove_file(&tmp);
        build_stream_store(&tmp, &ds, &BuildParams::default(), &[9u8; 32]).unwrap();
        std::fs::rename(&tmp, &path).unwrap();
        println!(
            "(built fixture {n}x{dim} in {:.1}s, incl. the store's own HNSW build)",
            t.elapsed().as_secs_f64()
        );
    }
    let b = std::fs::read(&path).unwrap();
    let file_id: [u8; 16] = b[b.len() - 4096 + 0xF00..][..16].try_into().unwrap();
    Fix { path, ds, file_id }
}

pub fn queries(ds: &StreamDataset, nq: usize) -> Vec<Vec<f32>> {
    (0..nq)
        .map(|i| {
            ds.vector(i * 7919 % ds.len())
                .iter()
                .map(|x| x + 0.01)
                .collect()
        })
        .collect()
}

pub fn start_server(
    f: &Fix,
    chunk_log2: u8,
) -> (
    std::net::SocketAddr,
    std::sync::Arc<StreamServer>,
    tokio::runtime::Runtime,
) {
    let kf = f.path.with_extension("key");
    std::fs::write(
        &kf,
        KEY.iter().map(|b| format!("{b:02x}")).collect::<String>(),
    )
    .unwrap();
    let mut cfg = StreamServerConfig::new(f.path.clone(), kf);
    cfg.chunk_log2 = chunk_log2;
    let srv = StreamServer::load(cfg).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let l = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let addr = l.local_addr().unwrap();
    let app = stream::router(srv.clone());
    rt.spawn(async move {
        axum::serve(
            l,
            app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .await
        .unwrap()
    });
    (addr, srv, rt)
}
