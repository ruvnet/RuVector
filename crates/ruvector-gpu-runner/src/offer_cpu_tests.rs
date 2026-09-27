//! CPU-mode offer filtering and $/effective-core-hour ranking.
//! Fixture offers 4xxxxxxx/5xxxxxxx with `cpu_*` fields are real vast.ai
//! offers (read-only `POST /bundles/`, 2026-09-27); 9000000x are synthetic.

use super::tests::{fixture, CM};
use super::*;

fn cpu_filter() -> OfferFilter {
    OfferFilter {
        gpu_names: vec![],
        num_gpus: 0,
        min_gpu_ram_gb: 0.0,
        min_cuda: 0.0,
        max_dph: 1.0,
        min_reliability: RELIABILITY_FLOOR,
        require_datacenter: false,
        cpu_mode: true,
        min_cpu_cores: 32.0,
        min_ram_gb: 0.0,
    }
}

fn ids(v: &[Offer]) -> Vec<u64> {
    v.iter().map(|o| o.id).collect()
}

#[test]
fn ranks_by_usd_per_effective_core_hour() {
    let (ok, _) = rank(&fixture(), &cpu_filter(), &CM);
    assert!(ok.len() >= 5, "{:?}", ids(&ok));
    let per: Vec<f64> = ok
        .iter()
        .map(|o| o.usd_per_core_hour(&CM).unwrap())
        .collect();
    assert!(per.windows(2).all(|w| w[0] <= w[1]), "{per:?}");
    // 44455690 (64 effective cores, $0.155/h) beats the cheapest-per-hour
    // offer 49260634 (36 cores, $0.089/h): $/h order != $/core-hour order.
    assert_eq!(ok[0].id, 44455690);
    let cheapest_dph = ok
        .iter()
        .min_by(|a, b| a.dph_total.total_cmp(&b.dph_total))
        .unwrap();
    assert_eq!(cheapest_dph.id, 49260634);
    assert_ne!(ok[0].id, cheapest_dph.id);
}

#[test]
fn uses_effective_not_host_cores() {
    // 51941431: host has 256 threads but only 32 are allocated to the offer.
    let o = fixture().into_iter().find(|o| o.id == 51941431).unwrap();
    assert_eq!(o.cpu_cores, Some(256.0));
    let per = o.usd_per_core_hour(&CM).unwrap();
    assert!((per - estimate(&o, &CM, 1.0).hourly_usd / 32.0).abs() < 1e-12);
    let mut f = cpu_filter();
    f.min_cpu_cores = 64.0;
    assert!(f
        .reject_reason(&o, &CM)
        .unwrap()
        .contains("cpu_cores_effective"));
}

#[test]
fn min_ram_rejects_small_allocations() {
    let mut f = cpu_filter();
    f.min_ram_gb = 32.0;
    let (ok, rej) = rank(&fixture(), &f, &CM);
    // 44455690 has ~16 GB, 49260634 has 31985 MB (< 32 GB).
    for id in [44455690, 49260634] {
        assert!(!ids(&ok).contains(&id));
        assert!(rej
            .iter()
            .any(|(r, why)| *r == id && why.contains("cpu_ram")));
    }
    assert_eq!(ok[0].id, 47674541);
    assert!(ok.iter().all(|o| o.cpu_ram.unwrap() >= 32_000.0));
}

#[test]
fn safety_filters_still_apply_in_cpu_mode() {
    let (ok, rej) = rank(&fixture(), &cpu_filter(), &CM);
    let why = |id: u64| rej.iter().find(|(r, _)| *r == id).map(|(_, w)| w.clone());
    // Cheapest per core, but unverified / below the reliability floor.
    assert!(why(90000002).unwrap().contains("not verified"));
    assert!(why(90000003).unwrap().contains("reliability"));
    for o in &ok {
        assert!(o.is_verified());
        assert!(o.reliability2.unwrap() >= RELIABILITY_FLOOR);
        assert!(estimate(o, &CM, 1.0).hourly_usd <= 1.0);
    }
    // Floor cannot be lowered in CPU mode either.
    let mut f = cpu_filter();
    f.min_reliability = 0.5;
    assert!(why(90000003).is_some());
    assert!(f
        .reject_reason(
            &fixture().into_iter().find(|o| o.id == 90000003).unwrap(),
            &CM
        )
        .unwrap()
        .contains("reliability"));
    // Budget-relevant: max_dph still enforced on the planning rate.
    f = cpu_filter();
    f.max_dph = 0.01;
    assert!(rank(&fixture(), &f, &CM).0.is_empty());
}

#[test]
fn gpu_less_and_any_gpu_offers_allowed() {
    let (ok, _) = rank(&fixture(), &cpu_filter(), &CM);
    let gpu_less = ok
        .iter()
        .find(|o| o.id == 90000001)
        .expect("GPU-less offer accepted");
    assert_eq!(gpu_less.num_gpus, 0);
    assert!(gpu_less.gpu_name.is_empty());
    let models: std::collections::BTreeSet<&str> = ok.iter().map(|o| o.gpu_name.as_str()).collect();
    assert!(models.len() >= 4, "{models:?}");
    // Same offer is rejected in GPU mode.
    let mut f = cpu_filter();
    f.cpu_mode = false;
    f.gpu_names = vec!["RTX 4090".into()];
    f.num_gpus = 1;
    assert!(f.reject_reason(gpu_less, &CM).is_some());
}

#[test]
fn offers_without_cpu_fields_rejected_in_cpu_mode() {
    let mut f = cpu_filter();
    f.min_cpu_cores = 0.0;
    let (_, rej) = rank(&fixture(), &f, &CM);
    // The original 10 GPU fixture offers carry no cpu_* fields.
    for id in [43681500u64, 47256370, 48644508, 48951871] {
        let (_, w) = rej.iter().find(|(r, _)| *r == id).unwrap();
        assert!(w.contains("cpu_cores_effective unknown"), "{id}: {w}");
    }
}

#[test]
fn require_datacenter_and_tiebreak() {
    let mut f = cpu_filter();
    f.require_datacenter = true;
    let (ok, _) = rank(&fixture(), &f, &CM);
    assert_eq!(ids(&ok), vec![90000001]);
}

#[test]
fn query_omits_gpu_keys_in_cpu_mode() {
    let mut f = cpu_filter();
    f.min_ram_gb = 64.0;
    let q = f.query_json(60.0, 256);
    for k in ["gpu_name", "num_gpus", "gpu_ram", "cuda_max_good"] {
        assert!(q.get(k).is_none(), "{k} present: {q}");
    }
    assert_eq!(q["cpu_cores_effective"]["gte"], 32.0);
    assert_eq!(q["cpu_ram"]["gte"], 64000.0);
    assert_eq!(q["verified"]["eq"], true);
    assert_eq!(q["reliability2"]["gte"], RELIABILITY_FLOOR);
    // GPU mode keeps them.
    let g = OfferFilter {
        gpu_names: vec!["RTX 4090".into()],
        num_gpus: 1,
        min_gpu_ram_gb: 24.0,
        min_cuda: 12.6,
        cpu_mode: false,
        min_cpu_cores: 0.0,
        ..f
    };
    let q = g.query_json(60.0, 64);
    for k in ["gpu_name", "num_gpus", "gpu_ram", "cuda_max_good"] {
        assert!(q.get(k).is_some(), "{k} missing");
    }
}

#[test]
fn gpu_mode_cannot_relax_gpu_constraints() {
    let mut f = cpu_filter();
    assert!(f.problems().is_empty());
    f.cpu_mode = false;
    let p = f.problems();
    assert!(p.iter().any(|m| m.contains("--gpu")));
    assert!(p.iter().any(|m| m.contains("--num-gpus 0")));
}
