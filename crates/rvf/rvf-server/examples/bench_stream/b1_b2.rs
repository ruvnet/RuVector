//! B1 (time to first query) and B2 (recall vs bytes).

use crate::common::*;
use ed25519_dalek::SigningKey;
use rvf_runtime::progressive_stream::build::descriptor_for;
use rvf_runtime::progressive_stream::build::BuildParams;
use rvf_runtime::progressive_stream::client::{
    ClientConfig, MemTransport, RunEnd, StreamClient, Throttled, Transport,
};
use rvf_runtime::{QueryOptions, RvfStore};
use rvf_server::stream_client::HttpTransport;
use rvf_wire::stream::StreamPolicy;
use std::time::Instant;

// ---------------------------------------------------------------- B1
pub fn b1(f: &Fix, n: usize, dim: usize) {
    println!("\n== B1 time-to-first-query (index source held constant: the persisted Layer C) ==");
    let (addr, srv, _rt) = start_server(f, 16);
    let keys = [pubkey()];
    let q = queries(&f.ds, 1).remove(0);
    // Buffered open + first query on the identical file (persisted index loaded).
    let t = Instant::now();
    let store = RvfStore::open_readonly(&f.path).unwrap();
    let open_s = t.elapsed().as_secs_f64();
    assert!(store.index_ready(), "persisted INDEX_SEG must be loaded");
    let t = Instant::now();
    let _ = store.query(&q, 10, &QueryOptions::default()).unwrap();
    let first_q = t.elapsed().as_secs_f64();
    println!(
        "buffered, local disk: open_readonly {open_s:.3}s + first query {:.1}ms (index_ready=true)",
        first_q * 1e3
    );
    println!(
        "stream: {} bytes, file {} bytes",
        srv.stream_len(),
        std::fs::metadata(&f.path).unwrap().len()
    );
    println!(
        "{:>8} | {:>10} {:>10} {:>10} | {:>10} | {:>12} | {:>8}",
        "Mbps",
        "stream:A",
        "stream:B",
        "stream:C=done",
        "download",
        "buffered=dl+open+q",
        "A speedup"
    );
    for mbps in [10.0, 100.0, 1000.0] {
        // stream
        let t0 = Instant::now();
        let mut first = [None::<f64>; 3];
        let mut first_hit = None;
        let mut c = StreamClient::new(
            StreamPolicy::new(&keys),
            Throttled {
                inner: HttpTransport::new(addr.to_string(), "/v1/stream/body"),
                mbps,
            },
            ClientConfig::default(),
        );
        let end = c.run(&mut |ing| {
            let l = ing.layers();
            for (i, on) in [l.layer_a, l.layer_b, l.layer_c].into_iter().enumerate() {
                if on && first[i].is_none() {
                    first[i] = Some(t0.elapsed().as_secs_f64());
                    if i == 0 {
                        let a = ing.query(&q, 10, 256);
                        if !a.envelope.results.is_empty() {
                            first_hit = Some(t0.elapsed().as_secs_f64());
                        }
                    }
                }
            }
        });
        assert_eq!(end, RunEnd::CompleteUnverified);
        // plain download of the same frame stream, no verification
        let t1 = Instant::now();
        let mut dl = Throttled {
            inner: HttpTransport::new(addr.to_string(), "/v1/stream/body"),
            mbps,
        };
        dl.fetch(0, &mut |_| true).unwrap();
        let dl_s = t1.elapsed().as_secs_f64();
        let buffered = dl_s + open_s + first_q;
        let fh = first_hit.unwrap_or(f64::NAN);
        println!(
            "{mbps:>8} | {:>9.2}s {:>9.2}s {:>13.2}s | {:>9.2}s | {:>17.2}s | {:>7.1}x  (first non-empty answer at {fh:.2}s)",
            first[0].unwrap(), first[1].unwrap(), first[2].unwrap(), dl_s, buffered, buffered / fh
        );
    }
    // Why the dossier's 198.6 s: an ingest-only store persists no index.
    println!("-- lazy index build (the dossier's 198.6 s first query) --");
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path().join("ingest_only.rvf");
    let mut s = RvfStore::create(
        &p,
        rvf_runtime::RvfOptions {
            dimension: dim as u16,
            ..Default::default()
        },
    )
    .unwrap();
    for ch in (0..n).collect::<Vec<_>>().chunks(10_000) {
        let refs: Vec<&[f32]> = ch.iter().map(|&i| f.ds.vector(i)).collect();
        let ids: Vec<u64> = ch.iter().map(|&i| i as u64).collect();
        s.ingest_batch(&refs, &ids, None).unwrap();
    }
    s.close().unwrap();
    let r = RvfStore::open_readonly(&p).unwrap();
    println!(
        "ingest-only store, closed without any query: INDEX_SEGs in the file = {}, index_ready after open_readonly = {}",
        r.segment_dir().iter().filter(|e| e.3 == 0x02).count(),
        r.index_ready()
    );
    let t = Instant::now();
    let _ = r.query(&q, 10, &QueryOptions::default()).unwrap();
    let first = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let _ = r.query(&q, 10, &QueryOptions::default()).unwrap();
    println!(
        "first query {first:.1}s (builds the HNSW in memory), second {:.2}ms",
        t.elapsed().as_secs_f64() * 1e3
    );
}

// ---------------------------------------------------------------- B2
pub fn b2(f: &Fix) {
    use rvf_runtime::progressive_stream::{QueryMode, QueryPath};
    println!(
        "\n== B2 recall@10 vs bytes received (ef=256; exact ground truth over all {} vectors) ==",
        f.ds.len()
    );
    println!("layers = old behaviour (ProgressiveIndex only); auto = scan-or-layers policy; p95 is per-query latency of the auto answer");
    let (body, bundle) = descriptor_for(
        &f.path,
        f.file_id,
        SigningKey::from_bytes(&KEY),
        &BuildParams::default(),
    )
    .unwrap();
    let stream: Vec<u8> = bundle.frames_stage_order(&body).flatten().collect();
    let qs = queries(&f.ds, 200);
    let truth = exact_topk(&f.ds, &qs, 10);
    let keys = [pubkey()];
    let mut c = StreamClient::new(
        StreamPolicy::new(&keys),
        MemTransport::new(stream.clone()),
        ClientConfig::default(),
    );
    println!(
        "{:<26} {:>11} {:>6} | {:>8} {:>8} | {:>9} {:>9} {:>6} {:>9} {:>9}",
        "checkpoint",
        "bytes",
        "%body",
        "visible",
        "cover",
        "layers",
        "auto",
        "path",
        "auto p95",
        "scan-only"
    );
    let total = stream.len() as f64;
    let n = f.ds.len();
    let mut seen = [false; 7];
    c.run(&mut |ing| {
        let l = ing.layers();
        let frac = ing.offset() as f64 / total;
        let all = ing.vectors().len() == n;
        let cond: [(&str, bool); 7] = [
            ("layer A (+hot vectors)", l.layer_a),
            ("A+B", l.layer_b),
            ("40% of bytes", frac >= 0.40),
            ("60% of bytes", frac >= 0.60),
            ("80% of bytes", frac >= 0.80),
            ("A+B, all vectors arrived", l.layer_b && all),
            ("A+B+C (complete)", l.layer_c),
        ];
        for (i, (name, hit)) in cond.iter().enumerate() {
            if !*hit || seen[i] {
                continue;
            }
            seen[i] = true;
            let (mut r_layers, mut r_auto, mut r_scan) = (0.0, 0.0, 0.0);
            let mut lat = Vec::new();
            let mut path = QueryPath::Empty;
            let mut cover = 0.0;
            for (q, t) in qs.iter().zip(&truth) {
                let ids = |a: &rvf_runtime::progressive_stream::StreamAnswer| -> Vec<u64> {
                    a.envelope.results.iter().map(|x| x.id).collect()
                };
                r_layers += recall(&ids(&ing.query_with(q, 10, 256, QueryMode::LayersOnly)), t);
                let t0 = Instant::now();
                let a = ing.query(q, 10, 256);
                lat.push(t0.elapsed().as_secs_f64() * 1e3);
                path = a.path;
                cover = a.coverage_vectors;
                r_auto += recall(&ids(&a), t);
                r_scan += recall(&ids(&ing.query_with(q, 10, 256, QueryMode::ScanOnly)), t);
            }
            lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let k = qs.len() as f64;
            println!(
                "{:<26} {:>11} {:>5.1}% | {:>8} {:>7.0}% | {:>9.3} {:>9.3} {:>6} {:>7.2}ms {:>9.3}",
                name,
                ing.offset(),
                100.0 * frac,
                ing.vectors().len(),
                100.0 * cover,
                r_layers / k,
                r_auto / k,
                format!("{path:?}").chars().take(6).collect::<String>(),
                lat[lat.len() * 95 / 100],
                r_scan / k
            );
        }
    });
    println!(
        "(rvf_index::available_recall's constants are NOT used; the figures above are measured)"
    );
}
