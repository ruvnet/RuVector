//! Standalone real-data kernel benchmark. Compile with rustc -O, outside Cargo.
#[path = "../src/algorithm/exact.rs"]
mod exact;
use std::collections::HashSet;
use std::time::Instant;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let expected: f64 = std::env::args().nth(2).unwrap().parse().unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines = text.lines();
    let n: usize = lines
        .next()
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse()
        .unwrap();
    let edges: Vec<_> = lines
        .map(|line| {
            let p: Vec<_> = line.split_whitespace().collect();
            (
                p[0].parse::<usize>().unwrap(),
                p[1].parse::<usize>().unwrap(),
                p[2].parse::<f64>().unwrap(),
            )
        })
        .collect();
    for iteration in 0..3 {
        let start = Instant::now();
        let (value, side) = exact::minimum_cut(n, &edges);
        let micros = start.elapsed().as_secs_f64() * 1e6;
        let side: HashSet<_> = side.into_iter().collect();
        assert!(!side.is_empty() && side.len() < n);
        let witness: f64 = edges
            .iter()
            .filter(|(u, v, _)| side.contains(u) != side.contains(v))
            .map(|e| e.2)
            .sum();
        assert_eq!(value, expected);
        assert_eq!(witness, expected);
        println!("{{\"iteration\":{iteration},\"micros\":{micros},\"value\":{value}}}");
    }
}
