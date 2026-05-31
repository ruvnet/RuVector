use ruvector_symphonyqg::graph::SymphonyGraph;
use ruvector_symphonyqg::vamana::VamanaConfig;
use ruvector_symphonyqg::Config;
fn main() {
    let n = 50_000;
    let dim = 128;
    let mut s = 42u32 | 1;
    let vecs: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    (s as f32 / u32::MAX as f32) - 0.5
                })
                .collect()
        })
        .collect();
    let cfg = Config {
        dim,
        m_base: 16,
        ef_construction: 200,
        ..Config::default()
    };
    let g_before = ruvector_symphonyqg::build::build(&vecs, &cfg);
    let g_after =
        ruvector_symphonyqg::vamana::refine(g_before.clone(), &cfg, &VamanaConfig::default());
    let dist = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>();
    let ep_vec = g_before.vector_of(0);
    let probe = |g: &SymphonyGraph, label: &str| {
        let nbrs = g.neighbors_of(0);
        let real_nbrs: Vec<u32> = nbrs
            .iter()
            .copied()
            .filter(|&n| (n as usize) < g.n)
            .collect();
        let mut dists: Vec<f32> = real_nbrs
            .iter()
            .map(|&n| dist(ep_vec, g.vector_of(n as usize)))
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{}: real_nbrs={}, dist range [{:.3}, {:.3}], median {:.3}",
            label,
            real_nbrs.len(),
            dists.first().unwrap_or(&0.0),
            dists.last().unwrap_or(&0.0),
            dists.get(real_nbrs.len() / 2).unwrap_or(&0.0)
        );
    };
    probe(&g_before, "BEFORE refine");
    probe(&g_after, "AFTER  refine");
}
