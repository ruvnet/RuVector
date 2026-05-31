//! Quick demo: feed 300 stable + 100 drifted vectors and print drift signals.
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use ruvector_drift::{DriftDetector, EwaDriftDetector, GraphCoherenceDriftDetector};

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let dim = 64usize;

    let mut ewa = EwaDriftDetector::new(dim, 0.05, 0.12, 50);
    let mut gc = GraphCoherenceDriftDetector::new(64, 5, 50, 0.08);

    println!("Epoch  | EWA score | EWA alert | GC score | GC alert | Phase");
    println!("{}", "-".repeat(70));

    for epoch in 0..400 {
        let phase = if epoch < 300 { "stable" } else { "drift" };
        let mean = if epoch < 300 { 0.0f32 } else { 2.0 };

        let v: Vec<f32> = (0..dim)
            .map(|_| Normal::new(mean, 1.0).unwrap().sample(&mut rng))
            .collect();
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v: Vec<f32> = v.into_iter().map(|x| x / n.max(1e-8)).collect();

        let se = ewa.observe(&v);
        let sg = gc.observe(&v);

        if epoch % 50 == 0 || se.alert || sg.alert {
            println!(
                "{:5}  | {:9.4} | {:9} | {:8.4} | {:8} | {}",
                epoch,
                se.score,
                se.alert,
                sg.score,
                sg.alert,
                phase
            );
        }
    }

    println!();
    let s = ewa.summary();
    println!("EWA:  {} observations, {} alerts", s.n_observed, s.n_alerts);
    let s = gc.summary();
    println!("GC:   {} observations, {} alerts", s.n_observed, s.n_alerts);
}
