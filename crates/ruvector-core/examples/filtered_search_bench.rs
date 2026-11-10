//! Filtered search: old path (unfiltered top-k, then drop non-matching) vs the filter-aware planner.
//!
//! Seeded clustered data, one boolean metadata attribute per selectivity (assigned independently of the clusters),
//! plus one attribute that holds for 10 of the 100 clusters (most queries have no passing neighbour nearby), and
//! brute-force filtered ground truth. Reports recall@10, latency (mean/p50/p99) and the share of queries that return
//! fewer than min(k, #matching) results.
//!
//! ```text
//! cargo run --release -p ruvector-core --example filtered_search_bench -- [n] [dim] [queries]
//! ```
//! Defaults: n = 100000, dim = 128, queries = 200.

use ruvector_core::types::{DbOptions, HnswConfig};
use ruvector_core::vector_db::FilterPlan;
use ruvector_core::{DistanceMetric, SearchQuery, SearchResult, VectorDB, VectorEntry};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Instant;

const K: usize = 10;
const EF: usize = 100;
/// (metadata key, selectivity) of the cluster-independent attributes.
const SELECTIVITIES: [(&str, f64); 6] = [
    ("s50", 0.5),
    ("s10", 0.1),
    ("s5", 0.05),
    ("s2", 0.02),
    ("s1", 0.01),
    ("s01", 0.001),
];

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f32 {
        let (u, v) = (self.unit().max(1e-12), self.unit());
        ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()) as f32
    }
}

/// Points around the given Gaussian centroids, with the centroid each was drawn from.
fn clustered(n: usize, rng: &mut Rng, centroids: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<usize>) {
    (0..n)
        .map(|_| {
            let c = (rng.next_u64() % centroids.len() as u64) as usize;
            let v = centroids[c]
                .iter()
                .map(|x| x + 0.35 * rng.gauss())
                .collect();
            (v, c)
        })
        .unzip()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

struct Stats {
    recall: f64,
    mean: f64,
    p50: f64,
    p99: f64,
    short: f64,
}

fn summarize(mut lat_us: Vec<f64>, recall_sum: f64, short: usize) -> Stats {
    let q = lat_us.len();
    lat_us.sort_by(|a, b| a.total_cmp(b));
    Stats {
        recall: recall_sum / q as f64,
        mean: lat_us.iter().sum::<f64>() / q as f64,
        p50: lat_us[q / 2],
        p99: lat_us[((q as f64 * 0.99) as usize).min(q - 1)],
        short: 100.0 * short as f64 / q as f64,
    }
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("numeric argument"))
        .collect();
    let n = args.first().copied().unwrap_or(100_000);
    let dim = args.get(1).copied().unwrap_or(128);
    let nq = args.get(2).copied().unwrap_or(200);

    let mut rng = Rng(0x5EED);
    let centroids: Vec<Vec<f32>> = (0..100)
        .map(|_| (0..dim).map(|_| rng.gauss()).collect())
        .collect();
    let (base, cluster) = clustered(n, &mut rng, &centroids);
    let (queries, _) = clustered(nq, &mut rng, &centroids);
    // (metadata key, label, membership)
    let mut attrs: Vec<(String, String, Vec<bool>)> = SELECTIVITIES
        .iter()
        .map(|&(key, s)| {
            let member: Vec<bool> = (0..n).map(|_| rng.unit() < s).collect();
            (key.to_string(), format!("{:.1}%", s * 100.0), member)
        })
        .collect();
    attrs.push((
        "c10".to_string(),
        "10.0% (10 of 100 clusters)".to_string(),
        cluster.iter().map(|&c| c < 10).collect(),
    ));

    let db = VectorDB::new(DbOptions {
        dimensions: dim,
        distance_metric: DistanceMetric::Euclidean,
        storage_path: "memory://".to_string(),
        hnsw_config: Some(HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: EF,
            max_elements: n,
        }),
        quantization: None,
    })
    .expect("db");
    let t = Instant::now();
    let entries: Vec<VectorEntry> = base
        .iter()
        .enumerate()
        .map(|(i, v)| VectorEntry {
            id: Some(i.to_string()),
            vector: v.clone(),
            metadata: Some(
                attrs
                    .iter()
                    .map(|(key, _, member)| (key.clone(), json!(member[i])))
                    .collect(),
            ),
        })
        .collect();
    db.insert_batch(entries).expect("insert");
    println!(
        "n={n} dim={dim} queries={nq} k={K} ef={EF} m=16 ef_construction=200  (build {:.1}s)",
        t.elapsed().as_secs_f64()
    );

    // The alternative to the in-memory metadata view: scan stored metadata on every filtered query.
    let t = Instant::now();
    let mut matched = 0usize;
    for i in 0..n {
        let e = db.get(&i.to_string()).unwrap().unwrap();
        matched += e.metadata.unwrap()["s1"].as_bool().unwrap() as usize;
    }
    println!(
        "one storage metadata scan (the per-query cost without the view): {:.1} ms ({matched} matches)\n",
        t.elapsed().as_secs_f64() * 1e3
    );

    println!(
        "| selectivity | path | recall@10 | mean µs | p50 µs | p99 µs | % queries < min(k, matches) |"
    );
    println!("|---|---|---|---|---|---|---|");
    for (key, label, member) in &attrs {
        let truths: Vec<Vec<usize>> = queries
            .iter()
            .map(|q| {
                let mut d: Vec<(f32, usize)> = (0..n)
                    .filter(|&i| member[i])
                    .map(|i| (l2(q, &base[i]), i))
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                d.into_iter().take(K).map(|x| x.1).collect()
            })
            .collect();
        let filter: HashMap<String, Value> = HashMap::from([(key.clone(), json!(true))]);
        let run = |name: &str, search: &dyn Fn(&[f32]) -> Vec<SearchResult>| {
            let mut lat = Vec::with_capacity(nq);
            let (mut recall, mut short) = (0.0, 0usize);
            for (q, truth) in queries.iter().zip(&truths) {
                let t = Instant::now();
                let got = search(q);
                lat.push(t.elapsed().as_secs_f64() * 1e6);
                let want = truth.len().min(K);
                if got.len() < want {
                    short += 1;
                }
                let hit = got
                    .iter()
                    .filter(|r| truth.contains(&r.id.parse().unwrap()))
                    .count();
                recall += if want == 0 {
                    1.0
                } else {
                    hit as f64 / want as f64
                };
            }
            let st = summarize(lat, recall, short);
            println!(
                "| {label} | {name} | {:.3} | {:.0} | {:.0} | {:.0} | {:.1}% |",
                st.recall, st.mean, st.p50, st.p99, st.short
            );
        };
        // OLD: exactly the previous `VectorDB::search` — unfiltered top-k with ef, then retain matching metadata.
        run("old (search then retain)", &|q| {
            let mut r = db
                .search(SearchQuery {
                    vector: q.to_vec(),
                    k: K,
                    filter: None,
                    ef_search: Some(EF),
                })
                .unwrap();
            r.retain(|x| {
                x.metadata
                    .as_ref()
                    .is_some_and(|m| filter.iter().all(|(k, v)| m.get(k) == Some(v)))
            });
            r
        });
        for (name, plan) in [
            ("new: planner (Auto)", FilterPlan::Auto),
            ("new: ACORN walk only", FilterPlan::Graph),
            ("new: exact scan only", FilterPlan::Exact),
        ] {
            run(name, &|q| {
                db.search_with_plan(
                    SearchQuery {
                        vector: q.to_vec(),
                        k: K,
                        filter: Some(filter.clone()),
                        ef_search: Some(EF),
                    },
                    plan,
                )
                .unwrap()
            });
        }
    }
}
