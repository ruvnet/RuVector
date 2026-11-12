//! Benchmark: cold full-recompile vs warm cached assembly (PIR WP26).
//!
//! Simulates the ReCache workload shape at the context-assembly layer:
//! N tool schemas (~1–4 KB each), R requests, each request presenting the
//! full set in a fresh shuffled order. The cold path compiles the ordered
//! set from scratch every request; the warm path assembles cached blocks.
//! Outputs both timings and the speedup ratio, and verifies byte-identity
//! of the two paths on every request before timing anything.
//!
//! Deterministic: fixed LCG seed, no external dependencies.

use mcp_gate::schema_cache::{compile_fresh, ResourceId, SchemaCacheConfig, SchemaResourceCache};
use std::time::Instant;

const N_SCHEMAS: usize = 100;
const N_REQUESTS: usize = 200;

/// Deterministic LCG (numerical-recipes constants) for shuffling.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
}

fn shuffled(order: &mut [usize], rng: &mut Lcg) {
    for i in (1..order.len()).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// Build a realistic tool schema of roughly 1–4 KB canonical size.
fn make_schema(i: usize) -> serde_json::Value {
    let n_props = 8 + (i % 24); // varies canonical size ~1–4 KB
    let mut properties = serde_json::Map::new();
    for p in 0..n_props {
        properties.insert(
            format!("param_{p:02}"),
            serde_json::json!({
                "type": if p % 3 == 0 { "string" } else { "object" },
                "description": format!(
                    "Parameter {p} of tool {i}: controls behaviour of the \
                     simulated gate action, including nested targets and \
                     constraint metadata used for permission evaluation."
                ),
                "properties": {
                    "value": { "type": "string" },
                    "constraints": { "type": "array", "items": { "type": "string" } }
                }
            }),
        );
    }
    serde_json::json!({
        "name": format!("tool_{i:03}"),
        "description": format!("Simulated MCP tool #{i} for the schema-cache benchmark."),
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": ["param_00"]
        }
    })
}

fn main() {
    let schemas: Vec<serde_json::Value> = (0..N_SCHEMAS).map(make_schema).collect();
    let total_bytes: usize = schemas
        .iter()
        .map(|s| mcp_gate::schema_cache::canonicalize(s).unwrap().len())
        .sum();

    let config = SchemaCacheConfig::default();
    let mut cache = SchemaResourceCache::with_config(config.clone());
    let ids: Vec<ResourceId> = schemas
        .iter()
        .map(|s| cache.insert(s).expect("insert"))
        .collect();

    // Pre-generate the request orderings so both paths see identical work.
    let mut rng = Lcg(0x5eed_5eed_5eed_5eed);
    let mut orderings: Vec<Vec<usize>> = Vec::with_capacity(N_REQUESTS);
    let mut order: Vec<usize> = (0..N_SCHEMAS).collect();
    for _ in 0..N_REQUESTS {
        shuffled(&mut order, &mut rng);
        orderings.push(order.clone());
    }

    // Correctness first: every request's warm assembly must be byte-identical
    // to the cold compile of the same ordering.
    for ord in &orderings {
        let refs: Vec<&serde_json::Value> = ord.iter().map(|&i| &schemas[i]).collect();
        let req_ids: Vec<ResourceId> = ord.iter().map(|&i| ids[i]).collect();
        assert_eq!(
            cache.assemble(&req_ids).unwrap(),
            compile_fresh(&refs, &config).unwrap(),
            "warm assembly diverged from cold compile"
        );
    }

    // Cold path: from-scratch compile per request.
    let start = Instant::now();
    let mut cold_bytes = 0usize;
    for ord in &orderings {
        let refs: Vec<&serde_json::Value> = ord.iter().map(|&i| &schemas[i]).collect();
        cold_bytes += compile_fresh(&refs, &config).unwrap().len();
    }
    let cold = start.elapsed();

    // Warm path: cached assembly per request.
    let start = Instant::now();
    let mut warm_bytes = 0usize;
    for ord in &orderings {
        let req_ids: Vec<ResourceId> = ord.iter().map(|&i| ids[i]).collect();
        warm_bytes += cache.assemble(&req_ids).unwrap().len();
    }
    let warm = start.elapsed();

    assert_eq!(cold_bytes, warm_bytes);

    let ratio = cold.as_secs_f64() / warm.as_secs_f64();
    println!("schema-cache benchmark (PIR WP26)");
    println!(
        "  {} schemas, {:.1} KB canonical total, {} shuffled-order requests",
        N_SCHEMAS,
        total_bytes as f64 / 1024.0,
        N_REQUESTS
    );
    println!(
        "  cold (full recompile per request): {:>10.3} ms total, {:>8.1} µs/request",
        cold.as_secs_f64() * 1e3,
        cold.as_secs_f64() * 1e6 / N_REQUESTS as f64
    );
    println!(
        "  warm (cached assembly per request): {:>9.3} ms total, {:>8.1} µs/request",
        warm.as_secs_f64() * 1e3,
        warm.as_secs_f64() * 1e6 / N_REQUESTS as f64
    );
    println!("  assembly speedup: {ratio:.2}x");
}
