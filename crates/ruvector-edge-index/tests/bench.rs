//! Timings (printed, not asserted). `cargo test -p ruvector-edge-index
//! --test bench -- --nocapture`. Native timings only; wasm32 in a Worker
//! isolate will be slower.

mod common;

use std::time::{Duration, Instant};

use common::*;
use ruvector_edge_index::{
    HnswIndex, HnswParams, Metric, QuantFlatIndex, QuantParams, SliceFetch, MAX_CHUNK_BYTES,
};

fn pct(mut v: Vec<Duration>, p: f64) -> Duration {
    v.sort();
    v[((v.len() - 1) as f64 * p) as usize]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

#[test]
fn bench_build_query_persist() {
    let (base, queries) = data(N, DIM, QUERIES);
    let metric = Metric::Cosine;
    let t = Instant::now();
    let quant = QuantParams::train(metric, DIM, &base[..1000 * DIM], 1).unwrap();
    eprintln!("train quantizer (1k x {DIM}): {:.2} ms", ms(t.elapsed()));

    let t = Instant::now();
    let mut flat = QuantFlatIndex::with_capacity(quant.clone(), N as u32, N).unwrap();
    for (i, v) in base.chunks_exact(DIM).enumerate() {
        flat.upsert(i as u32, v).unwrap();
    }
    eprintln!("flat encode {N}: {:.1} ms", ms(t.elapsed()));
    let mut fetch = SliceFetch::new(&base, DIM);
    let lat: Vec<Duration> = queries
        .chunks_exact(DIM)
        .map(|q| {
            let t = Instant::now();
            flat.search_rerank(q, K, 4 * K, &mut fetch).unwrap();
            t.elapsed()
        })
        .collect();
    eprintln!(
        "flat query+rerank(40): p50 {:.3} ms  p95 {:.3} ms",
        ms(pct(lat.clone(), 0.5)),
        ms(pct(lat, 0.95))
    );

    let t = Instant::now();
    let mut h = HnswIndex::with_capacity(HnswParams::default(), quant, N).unwrap();
    for (i, v) in base.chunks_exact(DIM).enumerate() {
        h.insert(i as u32, v, &mut op_rng(i as u64)).unwrap();
    }
    let build = t.elapsed();
    eprintln!(
        "hnsw build {N} (m 16, m0 32, efc 128): {:.0} ms ({:.3} ms/insert)",
        ms(build),
        ms(build) / N as f64
    );
    for ef in [32, 64, 128] {
        let lat: Vec<Duration> = queries
            .chunks_exact(DIM)
            .map(|q| {
                let t = Instant::now();
                h.search_rerank(q, K, ef, &mut fetch).unwrap();
                t.elapsed()
            })
            .collect();
        eprintln!(
            "hnsw query+rerank ef={ef}: p50 {:.3} ms  p95 {:.3} ms",
            ms(pct(lat.clone(), 0.5)),
            ms(pct(lat, 0.95))
        );
    }

    let t = Instant::now();
    let enc = h.to_chunks(1, MAX_CHUNK_BYTES).unwrap();
    let encode = t.elapsed();
    let max_chunk = enc.chunks.iter().map(|c| c.bytes.len()).max().unwrap_or(0);
    eprintln!(
        "hnsw encode: {:.1} ms, {} chunks (max {} B), payload {} B, sha256 {}",
        ms(encode),
        enc.chunks.len(),
        max_chunk,
        enc.payload_len,
        &enc.sha256_hex()[..12]
    );
    // Lazy load: stream rows (a cursor) into a decode, then replay ≤ 200
    // ops (ADR §15 M2b: p95 < 1 s end to end).
    let extra = World::new(DIM).points(3, 200);
    let mib = enc.payload_len as f64 / f64::from(1u32 << 20);
    let (mut dec, mut total) = (Vec::new(), Vec::new());
    for _ in 0..20 {
        let rows = enc.chunks.clone();
        let t = Instant::now();
        let mut loaded =
            HnswIndex::from_chunk_iter(rows, Some(&enc.sha256), enc.payload_len).unwrap();
        dec.push(t.elapsed());
        loaded.reserve(200);
        for (j, v) in extra.chunks_exact(DIM).enumerate() {
            let iid = (N + j) as u32;
            loaded.insert(iid, v, &mut op_rng(iid as u64)).unwrap();
        }
        total.push(t.elapsed());
    }
    eprintln!(
        "hnsw lazy load {mib:.2} MiB: decode+verify p50 {:.1} ms p95 {:.1} ms ({:.2} ms/MiB); + 200-op replay p95 {:.1} ms",
        ms(pct(dec.clone(), 0.5)),
        ms(pct(dec.clone(), 0.95)),
        ms(pct(dec, 0.5)) / mib,
        ms(pct(total, 0.95))
    );
    eprintln!(
        "resident: hnsw {} B ({:.1} B/vector), flat {} B ({:.1} B/vector)",
        h.memory_bytes(),
        h.memory_bytes() as f64 / N as f64,
        flat.memory_bytes(),
        flat.memory_bytes() as f64 / N as f64
    );

    // Unstructured control (iid Gaussian, not asserted): the hard case.
    let mut g = Gen::new(99);
    let rand: Vec<f32> = (0..N * DIM).map(|_| g.gauss()).collect();
    let rq: Vec<f32> = (0..50 * DIM).map(|_| g.gauss()).collect();
    let q = QuantParams::train(Metric::L2, DIM, &rand[..1000 * DIM], 1).unwrap();
    let mut r = HnswIndex::with_capacity(HnswParams::default(), q, N).unwrap();
    for (i, v) in rand.chunks_exact(DIM).enumerate() {
        r.insert(i as u32, v, &mut op_rng(i as u64)).unwrap();
    }
    let mut rf = SliceFetch::new(&rand, DIM);
    let tr = truth(Metric::L2, &rand, &rq, DIM, K);
    for ef in [64, 256] {
        let rec: f64 = rq
            .chunks_exact(DIM)
            .zip(&tr)
            .map(|(qv, t)| recall(t, &r.search_rerank(qv, K, ef, &mut rf).unwrap()))
            .sum::<f64>()
            / tr.len() as f64;
        eprintln!("control: iid gaussian 10k x 384 l2 hnsw ef={ef}: recall@10 {rec:.4}");
    }
    let mut rflat = QuantFlatIndex::with_capacity(r.quant().clone(), N as u32, N).unwrap();
    for (i, v) in rand.chunks_exact(DIM).enumerate() {
        rflat.upsert(i as u32, v).unwrap();
    }
    for over in [4, 10] {
        let rec: f64 = rq
            .chunks_exact(DIM)
            .zip(&tr)
            .map(|(qv, t)| recall(t, &rflat.search_rerank(qv, K, over * K, &mut rf).unwrap()))
            .sum::<f64>()
            / tr.len() as f64;
        eprintln!(
            "control: iid gaussian 10k x 384 l2 flat rerank({}): recall@10 {rec:.4}",
            over * K
        );
    }
}
