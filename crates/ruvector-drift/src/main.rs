//! ruvector-drift demo: shows all three detectors on stable vs drifted data.
use rand::{Rng, SeedableRng};
use ruvector_drift::{CentroidDrift, CoherenceDrift, DriftConfig, DriftDetector, PsiDrift};

fn gaussian(n: usize, d: usize, mean: f32, sigma: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| mean + rng.gen_range(-sigma..sigma))
                .collect()
        })
        .collect()
}

fn run_pair(name: &str, det: &mut dyn DriftDetector, stable: bool) {
    let report = det.detect();
    let status = if report.is_drifted { "DRIFT" } else { "STABLE" };
    let expected = if stable { "STABLE" } else { "DRIFT" };
    let ok = if stable {
        !report.is_drifted
    } else {
        report.is_drifted
    };
    println!(
        "  [{name}] score={:.4}  status={status}  expected={expected}  {}",
        report.drift_score,
        if ok { "PASS" } else { "FAIL" }
    );
}

fn main() {
    let d = 64usize;
    let n = 500usize;
    let cfg = DriftConfig::default();

    println!("\n=== ruvector-drift demo ===\n");

    // Scenario 1: Stable — same distribution
    println!("Scenario 1: Stable (same Gaussian cluster)");
    {
        let base = gaussian(n, d, 0.0, 0.3, 1);
        let next = gaussian(n, d, 0.0, 0.3, 2);

        let mut c = CentroidDrift::new(cfg.clone());
        c.add_window(0, &base);
        c.add_window(1, &next);

        let mut p = PsiDrift::new(cfg.clone());
        p.add_window(0, &base);
        p.add_window(1, &next);

        let mut co = CoherenceDrift::new(cfg.clone());
        co.add_window(0, &base);
        co.add_window(1, &next);

        run_pair("CentroidDrift", &mut c, true);
        run_pair("PsiDrift    ", &mut p, true);
        run_pair("CoherenceDrift", &mut co, true);
    }

    // Scenario 2: Large centroid shift
    println!("\nScenario 2: Drift — mean shifts by 10.0");
    {
        let base = gaussian(n, d, 0.0, 0.3, 3);
        let next = gaussian(n, d, 10.0, 0.3, 4);

        let mut c = CentroidDrift::new(cfg.clone());
        c.add_window(0, &base);
        c.add_window(1, &next);

        let mut p = PsiDrift::new(cfg.clone());
        p.add_window(0, &base);
        p.add_window(1, &next);

        let mut co = CoherenceDrift::new(cfg.clone());
        co.add_window(0, &base);
        co.add_window(1, &next);

        run_pair("CentroidDrift", &mut c, false);
        run_pair("PsiDrift    ", &mut p, false);
        run_pair("CoherenceDrift", &mut co, false);
    }

    // Scenario 3: Distribution shape change — centroid stays put but spread widens
    println!("\nScenario 3: Distribution shape drift — spread widens (σ 0.1→2.0)");
    {
        let base = gaussian(n, d, 0.0, 0.1, 5);
        let next = gaussian(n, d, 0.0, 2.0, 6);

        let mut c = CentroidDrift::new(cfg.clone());
        c.add_window(0, &base);
        c.add_window(1, &next);

        let mut p = PsiDrift::new(cfg.clone());
        p.add_window(0, &base);
        p.add_window(1, &next);

        let mut co = CoherenceDrift::new(cfg.clone());
        co.add_window(0, &base);
        co.add_window(1, &next);

        run_pair("CentroidDrift", &mut c, false);
        run_pair("PsiDrift    ", &mut p, false);
        run_pair("CoherenceDrift", &mut co, false);
    }

    // Scenario 4: Fragmentation — tight cluster splits into two opposing clusters
    println!("\nScenario 4: Fragmentation — cluster splits into two opposing poles");
    {
        let base: Vec<Vec<f32>> = (0..n).map(|_| vec![1.0f32; d]).collect();
        let fragmented: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                if i < n / 2 {
                    vec![1.0f32; d]
                } else {
                    vec![-1.0f32; d]
                }
            })
            .collect();

        let mut c = CentroidDrift::new(cfg.clone());
        c.add_window(0, &base);
        c.add_window(1, &fragmented);

        let mut p = PsiDrift::new(cfg.clone());
        p.add_window(0, &base);
        p.add_window(1, &fragmented);

        let mut co = CoherenceDrift::new(cfg.clone());
        co.add_window(0, &base);
        co.add_window(1, &fragmented);

        run_pair("CentroidDrift", &mut c, false);
        run_pair("PsiDrift    ", &mut p, false);
        run_pair("CoherenceDrift", &mut co, false);
    }

    println!("\n=== Demo complete ===\n");
}
