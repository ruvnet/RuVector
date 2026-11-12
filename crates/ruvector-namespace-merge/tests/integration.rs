use ruvector_namespace_merge::{
    dataset::{Dataset, DatasetConfig, Namespace},
    recall_at_k,
    router::{AllSearch, CentroidFilter, MinCutRoute, NamespaceRouter},
};

const K: usize = 10;
const SEED: u64 = 0xABCD_1234;

fn make_dataset() -> Dataset {
    Dataset::generate(&DatasetConfig {
        per_ns: 200,
        dims: 32,
        seed: SEED,
        noise: 0.20,
    })
}

fn manual_dataset(vectors: &[&[f32]]) -> Dataset {
    let dims = vectors.first().map_or(0, |vector| vector.len());
    let namespaces = vectors
        .iter()
        .enumerate()
        .map(|(id, vector)| Namespace::new(id, format!("ns-{id}"), vector.to_vec(), 1, dims))
        .collect();

    Dataset {
        namespaces,
        dims,
        per_ns: 1,
    }
}

#[test]
fn mincut_single_namespace_remains_searchable() {
    let ds = manual_dataset(&[&[1.0, 0.0]]);
    let result = MinCutRoute::new(&ds).search(&ds, &[1.0, 0.0], 1);

    assert_eq!(result.ns_searched, 1);
    assert_eq!(result.dist_ops, 1);
    assert_eq!(result.hits.len(), 1);
}

#[test]
fn mincut_equal_similarities_searches_all_namespaces() {
    let ds = manual_dataset(&[&[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0]]);
    let result = MinCutRoute::new(&ds).search(&ds, &[0.0, 1.0], 3);

    assert_eq!(result.ns_searched, 3);
    assert_eq!(result.dist_ops, 3);
    assert_eq!(result.hits.len(), 3);
}

#[test]
fn mincut_empty_dataset_returns_empty_result() {
    let ds = Dataset {
        namespaces: Vec::new(),
        dims: 2,
        per_ns: 0,
    };
    let result = MinCutRoute::new(&ds).search(&ds, &[1.0, 0.0], 10);

    assert!(result.hits.is_empty());
    assert_eq!(result.ns_searched, 0);
    assert_eq!(result.dist_ops, 0);
}

#[test]
fn all_search_recall_one() {
    let ds = make_dataset();
    let queries = ds.group_a_queries(50, SEED ^ 1);
    let router = AllSearch;

    // AllSearch is the ground truth — its recall vs itself must be 1.0
    let gt: Vec<_> = queries
        .iter()
        .map(|q| router.search(&ds, q, K).hits)
        .collect();

    for (q, truth) in queries.iter().zip(gt.iter()) {
        let res = router.search(&ds, q, K);
        let r = recall_at_k(&res.hits, truth, K);
        assert!(
            (r - 1.0).abs() < 1e-6,
            "AllSearch recall vs self must be 1.0, got {r}"
        );
    }
}

#[test]
fn centroid_filter_high_recall() {
    let ds = make_dataset();
    let queries = ds.group_a_queries(50, SEED ^ 2);
    let all = AllSearch;
    let cf = CentroidFilter::new(0.20);

    let gt: Vec<_> = queries.iter().map(|q| all.search(&ds, q, K).hits).collect();

    let avg_recall: f32 = queries
        .iter()
        .zip(gt.iter())
        .map(|(q, truth)| {
            let res = cf.search(&ds, q, K);
            recall_at_k(&res.hits, truth, K)
        })
        .sum::<f32>()
        / queries.len() as f32;

    assert!(
        avg_recall >= 0.75,
        "CentroidFilter recall@{K} = {avg_recall:.4}, expected ≥ 0.75"
    );
}

#[test]
fn mincut_route_searches_fewer_ns_than_all() {
    let ds = make_dataset();
    let queries = ds.group_a_queries(50, SEED ^ 3);
    let all = AllSearch;
    let mc = MinCutRoute::new(&ds);

    let gt: Vec<_> = queries.iter().map(|q| all.search(&ds, q, K).hits).collect();

    let mut mc_ns_sum = 0usize;
    let mut all_ns_sum = 0usize;
    let mut avg_recall = 0f32;

    for (q, truth) in queries.iter().zip(gt.iter()) {
        let res_all = all.search(&ds, q, K);
        let res_mc = mc.search(&ds, q, K);
        mc_ns_sum += res_mc.ns_searched;
        all_ns_sum += res_all.ns_searched;
        avg_recall += recall_at_k(&res_mc.hits, truth, K);
    }

    let avg_recall = avg_recall / queries.len() as f32;
    let mc_avg_ns = mc_ns_sum as f64 / queries.len() as f64;
    let all_avg_ns = all_ns_sum as f64 / queries.len() as f64;

    println!("MinCutRoute: avg_ns={mc_avg_ns:.2}, avg_recall={avg_recall:.4}");
    println!("AllSearch:   avg_ns={all_avg_ns:.2}");

    assert!(
        mc_avg_ns < all_avg_ns,
        "MinCutRoute should search fewer namespaces: mc={mc_avg_ns:.2} vs all={all_avg_ns:.2}"
    );
    assert!(
        avg_recall >= 0.70,
        "MinCutRoute recall@{K} = {avg_recall:.4}, expected ≥ 0.70"
    );
}

#[test]
fn flow_unit_two_cluster_query() {
    // Simple regression: A-group query should keep both A namespaces on S-side
    use ruvector_namespace_merge::flow::FlowGraph;

    // Simulate: 2 A namespaces (high q_sim), 1 C namespace (low q_sim)
    // q_sim = [0.60, 0.55, 0.02], inter_sim(A0,A1) = 0.95, others near 0
    let scale = 10_000i64;
    let n = 3; // namespaces
    let s = 3; // source
    let t = 4; // sink

    let mut g = FlowGraph::new(5);

    let q_sim = [0.60f32, 0.55f32, 0.02f32];
    for (i, query_sim) in q_sim.iter().copied().enumerate().take(n) {
        let qs = query_sim.clamp(0.0, 1.0);
        g.add_edge(s, i, (qs * scale as f32).round() as i64);
        g.add_edge(i, t, ((1.0 - qs) * scale as f32).round() as i64);
    }
    // inter-sim: A0↔A1 = 0.95, others ≈ 0
    g.add_undirected(0, 1, (0.95f32 * scale as f32).round() as i64);
    g.add_undirected(0, 2, (0.01f32 * scale as f32).round() as i64);
    g.add_undirected(1, 2, (0.01f32 * scale as f32).round() as i64);

    g.max_flow(s, t);
    let side = g.source_side(s);

    println!("Unit test side: {:?}", &side[..3]);
    // Both A namespaces must be on S-side (searched)
    assert!(side[0], "A0 must be on S-side");
    assert!(side[1], "A1 must be on S-side");
    // C namespace must be on T-side (skipped)
    assert!(!side[2], "C must be on T-side");
}
