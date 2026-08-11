//! Diagnostic: trace PQ behaviour on a trivially separable dataset.
use ruvector_streaming_qng::{
    sq_l2, nearest_centroid,
    pq::{Codebook, M, K},
    full_precision::FullPrecision,
    static_pq::StaticPq,
    AnnVariant, recall_at_k,
};

fn main() {
    let dims = 32;
    let n_per_cluster = 200;
    let n_clusters = 4;
    let n = n_per_cluster * n_clusters;
    let std = 0.08_f32;

    // Build indexed vectors
    let mut vecs: Vec<Vec<f32>> = Vec::new();
    for c in 0..n_clusters {
        for i in 0..n_per_cluster {
            let seed_val = (c * 1000 + i) as f32 * 0.001;
            let mut v = vec![c as f32 * 4.0];  // dim 0 separates clusters
            for d in 1..dims {
                let centroid_d = ((c * 7 + d * 3) % 4) as f32 * 0.3;
                v.push(centroid_d + (seed_val * 17.0 + d as f32 * 3.14).sin() * std);
            }
            vecs.push(v);
        }
    }

    // Build indexed vectors sorted by cluster (0,0,...,1,1,...,2,2,...,3,3,...)
    // vecs[0..200]: cluster 0, vecs[200..400]: cluster 1, etc.

    // Query from cluster 0
    let query: Vec<f32> = {
        let mut v = vec![0.0_f32];
        for d in 1..dims {
            let centroid_d = (0 * 7 + d * 3) % 4;
            v.push(centroid_d as f32 * 0.3 + 0.01);
        }
        v
    };

    // Ground truth: should be 10 vectors from cluster 0 (indices 0..200)
    let mut fp = FullPrecision::new();
    fp.build(&vecs);
    let gt = fp.search(&query, 10);
    println!("Ground truth top-5 ids: {:?}",
             gt.iter().take(5).map(|h| h.id).collect::<Vec<_>>());
    println!("All top-10 from cluster 0? {}",
             gt.iter().take(10).all(|h| h.id < 200));

    // PQ search
    let mut spq = StaticPq::new();
    spq.build(&vecs);
    let pq_hits = spq.search(&query, 10);
    println!("StaticPQ top-5 ids: {:?}",
             pq_hits.iter().take(5).map(|h| h.id).collect::<Vec<_>>());
    let recall = recall_at_k(&pq_hits, &gt, 10);
    println!("StaticPQ recall@10: {recall:.4}");

    // Check codebook internals
    // Train codebook manually on subspace 0 (dims 0..8 with M=4, ds=8)
    let sub0_samples: Vec<Vec<f32>> = vecs.iter().map(|v| v[0..8].to_vec()).collect();
    let cb = Codebook::train(&vecs, M, K, 1);

    // Distance from query-sub0 to each centroid in subspace 0
    let q_sub0 = &query[0..8];
    println!("\nSubspace-0 centroids (first 8-dim each), distances from cluster-0 query:");
    for (k_idx, centroid) in cb.centroids[0].iter().enumerate() {
        let d = sq_l2(q_sub0, centroid);
        let first_dim = centroid[0];
        println!("  centroid[{k_idx}]: dim0={first_dim:.3}, dist={d:.4}");
    }

    // Encode cluster-0 and cluster-1 vectors
    let code0 = cb.encode(&vecs[0]);    // cluster 0
    let code1 = cb.encode(&vecs[200]);  // cluster 1
    println!("\nCode for cluster-0 vec[0]:   {:?}", code0);
    println!("Code for cluster-1 vec[200]: {:?}", code1);

    // ADC table for query
    let table = cb.adc_table(&query);
    let adc_dist0 = Codebook::adc_dist(&table, &code0);
    let adc_dist1 = Codebook::adc_dist(&table, &code1);
    println!("\nADC dist to cluster-0 vector: {adc_dist0:.4}");
    println!("ADC dist to cluster-1 vector: {adc_dist1:.4}");
    println!("True sq_l2 to cluster-0 vector: {:.4}", sq_l2(&query, &vecs[0]));
    println!("True sq_l2 to cluster-1 vector: {:.4}", sq_l2(&query, &vecs[200]));

    // Compare all subspace-0 centroid dim0 values
    println!("\nAll subspace-0 centroid dim0 values:");
    let mut dim0_vals: Vec<(usize, f32)> = cb.centroids[0].iter().enumerate()
        .map(|(i, c)| (i, c[0])).collect();
    dim0_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (idx, val) in &dim0_vals {
        println!("  centroid[{idx}].dim0 = {val:.4}");
    }

    // Check: code0[0] should refer to a centroid with dim0 ≈ 0
    // code1[0] should refer to a centroid with dim0 ≈ 4
    println!("\nCluster-0 subspace-0 centroid: centroid[{}].dim0 = {:.4}",
             code0[0], cb.centroids[0][code0[0] as usize][0]);
    println!("Cluster-1 subspace-0 centroid: centroid[{}].dim0 = {:.4}",
             code1[0], cb.centroids[0][code1[0] as usize][0]);
}
