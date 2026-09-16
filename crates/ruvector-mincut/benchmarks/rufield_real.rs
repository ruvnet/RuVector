use ruvector_mincut::routing::rufield::{AwarenessPolicy, RuFieldRouter};
use ruvector_mincut::routing::{Arc, RoadRouter};
use std::{
    fs::File,
    io::{BufReader, Read},
    time::Instant,
};
fn u32r(r: &mut impl Read) -> u32 {
    let mut b = [0; 4];
    r.read_exact(&mut b).unwrap();
    u32::from_le_bytes(b)
}
fn f64r(r: &mut impl Read) -> f64 {
    let mut b = [0; 8];
    r.read_exact(&mut b).unwrap();
    f64::from_le_bytes(b)
}
fn percentile(v: &mut [u128], p: f64) -> u128 {
    v.sort_unstable();
    v[((v.len() as f64 * p).ceil() as usize).saturating_sub(1)]
}
fn main() {
    let path = std::env::args().nth(1).expect("normalized .roads path");
    let mut input = BufReader::new(File::open(path).unwrap());
    let (n, m, q) = (
        u32r(&mut input) as usize,
        u32r(&mut input) as usize,
        u32r(&mut input) as usize,
    );
    let arcs: Vec<_> = (0..m)
        .map(|_| Arc {
            source: u32r(&mut input),
            target: u32r(&mut input),
            cost: u32r(&mut input),
        })
        .collect();
    for _ in 0..n * 2 {
        f64r(&mut input);
    }
    let k = u32r(&mut input);
    for _ in 0..k {
        u32r(&mut input);
    }
    let queries: Vec<_> = (0..q)
        .map(|_| {
            let s = u32r(&mut input);
            let t = u32r(&mut input);
            let mut b = [0; 8];
            input.read_exact(&mut b).unwrap();
            (s, t, i64::from_le_bytes(b))
        })
        .collect();
    let traces = u32r(&mut input);
    for _ in 0..traces {
        u32r(&mut input);
        u32r(&mut input);
        u32r(&mut input);
        let mut b = [0; 16];
        input.read_exact(&mut b).unwrap();
    }
    let policy = AwarenessPolicy {
        max_penalty: 60_000,
        close_at_millionths: 1_000_001,
        ttl_ns: 1_000_000,
        max_lateness_ns: 10,
    };
    let mut aware = RuFieldRouter::new(RoadRouter::new(n, arcs, vec![]).unwrap(), policy).unwrap();
    let nodes: Vec<u32> = (0..4096).map(|i| ((i * 7919) % n) as u32).collect();
    for (i, &node) in nodes.iter().enumerate() {
        aware.bind_zone(format!("zone-{i}"), node).unwrap();
    }
    let baseline: Vec<_> = queries
        .iter()
        .take(32)
        .map(|&(s, t, _)| {
            aware
                .route(s, t, false, 20_000_000, || false)
                .unwrap()
                .map(|r| r.cost)
        })
        .collect();
    let mut samples = Vec::with_capacity(10_000);
    let mut changed = 0usize;
    for i in 0..10_000usize {
        let timestamp = 100 + i as u64;
        let json = format!(
            r#"{{"event_id":"real-{i}","timestamp_ns":{timestamp},"observation":{{"zone_id":"zone-{}","space_cell":null,"confidence":0.8,"features":{{"presence":1.0,"motion_energy":0.5,"transient":0.1}},"privacy_class":"P2"}},"provenance":{{"synthetic":false}}}}"#,
            i % nodes.len()
        );
        let start = Instant::now();
        changed += aware
            .ingest_json(json.as_bytes(), true, timestamp)
            .unwrap()
            .changed_arcs;
        samples.push(start.elapsed().as_nanos());
    }
    let p50 = percentile(&mut samples.clone(), 0.50);
    let p95 = percentile(&mut samples, 0.95);
    let start = Instant::now();
    let expired = aware.expire(1_020_000).unwrap();
    let expiry = start.elapsed().as_nanos();
    for (i, &(s, t, _)) in queries.iter().take(32).enumerate() {
        assert_eq!(
            aware
                .route(s, t, false, 20_000_000, || false)
                .unwrap()
                .map(|r| r.cost),
            baseline[i]
        );
    }
    println!(
        r#"{{"nodes":{n},"arcs":{m},"events":10000,"bindings":{},"ingest_p50_ns":{p50},"ingest_p95_ns":{p95},"changed_arcs":{changed},"expired_arcs":{expired},"expiry_ns":{expiry}}}"#,
        nodes.len()
    );
}
