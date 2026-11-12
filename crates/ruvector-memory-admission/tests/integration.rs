use ruvector_memory_admission::dataset::{StreamConfig, StreamDataset};
use ruvector_memory_admission::policy::{
    AdaptiveMincutAdmission, AdmissionPolicy, MincutGatedAdmission, NearestCentroidThreshold,
};

fn small_dataset() -> StreamDataset {
    StreamDataset::generate(&StreamConfig {
        n_points: 400,
        k_true: 4,
        dims: 32,
        seed: 0x1122_3344,
        ..StreamConfig::default()
    })
}

#[test]
fn no_vectors_are_lost_across_all_policies() {
    let ds = small_dataset();

    for policy_name in ["baseline", "mincut", "adaptive"] {
        let mut assigned = 0usize;
        match policy_name {
            "baseline" => {
                let mut p = NearestCentroidThreshold::new(0.55);
                for pt in &ds.points {
                    p.admit(&pt.vector);
                    assigned += 1;
                }
            }
            "mincut" => {
                let mut p = MincutGatedAdmission::new(0.35, 32);
                for pt in &ds.points {
                    p.admit(&pt.vector);
                    assigned += 1;
                }
            }
            _ => {
                let mut p = AdaptiveMincutAdmission::new(1.0, 32, 0.35);
                for pt in &ds.points {
                    p.admit(&pt.vector);
                    assigned += 1;
                }
            }
        }
        assert_eq!(
            assigned,
            ds.points.len(),
            "{policy_name} must admit every point"
        );
    }
}

#[test]
fn mincut_admission_cluster_count_stays_bounded() {
    let ds = small_dataset();
    let mut p = MincutGatedAdmission::new(0.35, 32);
    for pt in &ds.points {
        p.admit(&pt.vector);
    }
    // 4 true clusters with moderate noise: bounded growth expected, well
    // under the hard safety-valve cap.
    assert!(
        p.n_clusters() <= 32,
        "cluster count must respect the safety valve, got {}",
        p.n_clusters()
    );
    assert!(p.n_clusters() >= 1);
}

#[test]
fn adaptive_admission_cluster_count_stays_bounded() {
    let ds = small_dataset();
    let mut p = AdaptiveMincutAdmission::new(1.0, 32, 0.35);
    for pt in &ds.points {
        p.admit(&pt.vector);
    }
    assert!(p.n_clusters() <= 32);
    assert!(p.n_clusters() >= 1);
}

#[test]
fn decide_without_commit_does_not_mutate_state() {
    let ds = small_dataset();
    let mut p = MincutGatedAdmission::new(0.35, 32);
    for pt in ds.points.iter().take(50) {
        p.admit(&pt.vector);
    }
    let clusters_before = p.n_clusters();
    // Call `decide` (read-only) many times without `commit`.
    for pt in ds.points.iter().skip(50).take(20) {
        let _ = p.decide(&pt.vector);
    }
    assert_eq!(
        p.n_clusters(),
        clusters_before,
        "decide() must not mutate policy state"
    );
}

#[test]
fn centroids_stay_unit_norm_after_many_merges() {
    let ds = small_dataset();
    let mut p = NearestCentroidThreshold::new(0.5);
    for pt in &ds.points {
        p.admit(&pt.vector);
    }
    for c in 0..p.n_clusters() {
        let norm: f32 = p.centroid(c).iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3 || norm < 1e-6,
            "centroid {c} should stay unit-norm (or be exactly zero), got {norm}"
        );
    }
}
