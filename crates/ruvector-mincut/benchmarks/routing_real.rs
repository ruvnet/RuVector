#[allow(dead_code)]
#[path = "../src/routing/mod.rs"]
mod routing;
use routing::{Arc, RoadRouter};
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
fn main() {
    let path = std::env::args().nth(1).expect("normalized .roads path");
    let mut input = BufReader::new(File::open(path).unwrap());
    let (n, m, q) = (
        u32r(&mut input) as usize,
        u32r(&mut input) as usize,
        u32r(&mut input) as usize,
    );
    assert!(n <= 1_000_000 && m <= 4_000_000 && q <= 10000);
    let arcs: Vec<_> = (0..m)
        .map(|_| Arc {
            source: u32r(&mut input),
            target: u32r(&mut input),
            cost: u32r(&mut input),
        })
        .collect();
    let coords: Vec<_> = (0..n * 2).map(|_| f64r(&mut input)).collect();
    let k = u32r(&mut input);
    assert!(k <= 16);
    let landmarks: Vec<_> = (0..k).map(|_| u32r(&mut input)).collect();
    let queries: Vec<_> = (0..q)
        .map(|_| {
            let s = u32r(&mut input);
            let t = u32r(&mut input);
            let mut b = [0; 8];
            input.read_exact(&mut b).unwrap();
            (s, t, i64::from_le_bytes(b))
        })
        .collect();
    let trace_count = u32r(&mut input);
    assert!(trace_count <= 128);
    let traces: Vec<_> = (0..trace_count)
        .map(|_| {
            let (s, t, id) = (u32r(&mut input), u32r(&mut input), u32r(&mut input));
            let mut b = [0; 8];
            input.read_exact(&mut b).unwrap();
            let closed = i64::from_le_bytes(b);
            input.read_exact(&mut b).unwrap();
            (s, t, id, closed, i64::from_le_bytes(b))
        })
        .collect();
    let start = Instant::now();
    let mut router = RoadRouter::new(n, arcs.clone(), vec![]).unwrap();
    let build = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    router.set_coordinates(&coords).unwrap();
    let map_ms = start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    router.prepare(&landmarks, 100_000_000, || false).unwrap();
    let prepare_ms = start.elapsed().as_secs_f64() * 1000.;
    println!("{{\"type\":\"build\",\"nodes\":{n},\"arcs\":{m},\"build_ms\":{build},\"map_ms\":{map_ms},\"prepare_ms\":{prepare_ms}}}");
    for (i, &(s, t, expected)) in queries.iter().enumerate() {
        // Alternate execution order. Warm-up precedes three measured repetitions.
        for repeat in 0..4 {
            for alt in if i % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let result = router.route(s, t, alt, 20_000_000, || false).unwrap();
                let ns = start.elapsed().as_nanos();
                assert_eq!(
                    result.as_ref().map(|r| r.cost as i64).unwrap_or(-1),
                    expected
                );
                if let Some(ref route) = result {
                    assert_eq!(route.nodes.first(), Some(&s));
                    assert_eq!(route.nodes.last(), Some(&t));
                    assert_eq!(
                        route
                            .arcs
                            .iter()
                            .map(|&id| arcs[id as usize].cost as u64)
                            .sum::<u64>(),
                        route.cost
                    );
                    for (idx, &id) in route.arcs.iter().enumerate() {
                        assert_eq!(arcs[id as usize].source, route.nodes[idx]);
                        assert_eq!(arcs[id as usize].target, route.nodes[idx + 1]);
                    }
                }
                if repeat > 0 {
                    println!("{{\"type\":\"query\",\"query\":{i},\"repeat\":{repeat},\"alt\":{alt},\"ns\":{ns},\"settled\":{},\"cost\":{expected}}}",result.as_ref().map(|r|r.settled).unwrap_or(0));
                }
            }
        }
    }
    for &(s, t, id, closed, original) in &traces {
        router.prepare(&landmarks, 100_000_000, || false).unwrap();
        router.update(&[(id, None)]).unwrap();
        for alt in [true, false] {
            let route = router.route(s, t, alt, 20_000_000, || false).unwrap();
            assert_eq!(route.as_ref().map(|r| r.cost as i64).unwrap_or(-1), closed);
            if let Some(route) = route {
                assert!(!route.arcs.contains(&id));
            }
        }
        router
            .update(&[(id, Some(arcs[id as usize].cost))])
            .unwrap();
        assert_eq!(
            router
                .route(s, t, true, 20_000_000, || false)
                .unwrap()
                .unwrap()
                .cost as i64,
            original
        );
    }
    println!("{{\"type\":\"closures\",\"pairs\":{}}}", traces.len());
    // Fair scan baseline: precompute the same unit-sphere representation once.
    let projected: Vec<[f64; 3]> = coords
        .chunks_exact(2)
        .map(|c| {
            let a = c[0].to_radians();
            let b = c[1].to_radians();
            [a.cos() * b.cos(), a.cos() * b.sin(), a.sin()]
        })
        .collect();
    let mut indexed = 0u128;
    let mut linear = 0u128;
    for i in 0..256 {
        let id = i * 977 % n;
        let (lat, lon) = (coords[id * 2] + 0.00001, coords[id * 2 + 1] - 0.00001);
        let start = Instant::now();
        let snap = router.nearest(lat, lon, 21_000_000.).unwrap().unwrap();
        indexed += start.elapsed().as_nanos();
        let start = Instant::now();
        let a = lat.to_radians();
        let b = lon.to_radians();
        let xyz = [a.cos() * b.cos(), a.cos() * b.sin(), a.sin()];
        let expected = projected
            .iter()
            .enumerate()
            .map(|(id, p)| {
                let d = (0..3).map(|i| (xyz[i] - p[i]).powi(2)).sum::<f64>();
                (id, d)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
            .unwrap();
        linear += start.elapsed().as_nanos();
        assert_eq!(snap.node as usize, expected.0);
    }
    println!(
        "{{\"type\":\"snap\",\"queries\":256,\"indexed_ns\":{indexed},\"linear_ns\":{linear}}}"
    );
}
