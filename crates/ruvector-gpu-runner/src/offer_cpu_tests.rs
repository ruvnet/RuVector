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
        min_cpu_speed: crate::cpu_perf::DEFAULT_MIN_CPU_SPEED,
    }
}

fn ids(v: &[Offer]) -> Vec<u64> {
    v.iter().map(|o| o.id).collect()
}

#[test]
fn ranks_by_usd_per_effective_throughput_hour() {
    let (ok, _) = rank(&fixture(), &cpu_filter(), &CM);
    assert!(ok.len() >= 5, "{:?}", ids(&ok));
    let per: Vec<f64> = ok
        .iter()
        .map(|o| o.usd_per_throughput_hour(&CM).unwrap())
        .collect();
    assert!(per.windows(2).all(|w| w[0] <= w[1]), "{per:?}");
    // 44455690 (64 effective Zen 2 threads, $0.155/h) beats the cheapest-per-hour
    // offer 49260634 (36 Broadwell threads, $0.089/h).
    assert_eq!(ok[0].id, 44455690);
    let cheapest_dph = ok
        .iter()
        .min_by(|a, b| a.dph_total.total_cmp(&b.dph_total))
        .unwrap();
    assert_eq!(cheapest_dph.id, 49260634);
    assert_ne!(ok[0].id, cheapest_dph.id);
    // The family factor reorders offers vs plain $/thread: the Threadripper PRO
    // 5995WX (Zen 3) costs more per thread than the E5-2697A v4 (Broadwell)
    // but less per unit of throughput.
    let pos = |id: u64| ok.iter().position(|o| o.id == id).unwrap();
    let (tr, bdw) = (&ok[pos(51736668)], &ok[pos(46084498)]);
    assert!(tr.usd_per_core_hour(&CM) > bdw.usd_per_core_hour(&CM));
    assert!(pos(51736668) < pos(46084498));
    // Unknown family ranks last despite a mid-range $/thread.
    assert_eq!(ok.last().unwrap().id, 90000001);
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
    // Next best per throughput: the 96-thread EPYC 7K62 (Zen 2), ahead of the
    // cheaper-per-thread E5-2673 v3 (Haswell).
    assert_eq!(ok[0].id, 31639257);
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

#[test]
fn xeon_phi_is_rejected_not_merely_ranked_lower() {
    // 31231107 is the Xeon Phi 7210 offer the pre-fix $/thread ranking chose
    // (audit dry-run 2026-09-27): cheapest per thread of every safe real offer
    // (9000000x are synthetic).
    let all = fixture();
    let phi = all.iter().find(|o| o.id == 31231107).unwrap();
    let f = cpu_filter();
    let safe_per_core = all
        .iter()
        .filter(|o| o.id < 90_000_000)
        .filter(|o| o.is_verified() && o.reliability2.unwrap_or(0.0) >= RELIABILITY_FLOOR)
        .filter_map(|o| o.usd_per_core_hour(&CM).map(|p| (o.id, p)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    assert_eq!(safe_per_core.0, 31231107);
    let (ok, rej) = rank(&all, &f, &CM);
    assert!(!ids(&ok).contains(&31231107));
    let why = &rej.iter().find(|(r, _)| *r == 31231107).unwrap().1;
    assert!(
        why.contains("deny-listed") && why.contains("Xeon Phi"),
        "{why}"
    );
    // No floor setting lets it back in.
    let mut f0 = cpu_filter();
    f0.min_cpu_speed = 0.0;
    assert!(f0.reject_reason(phi, &CM).unwrap().contains("Xeon Phi"));
    assert_eq!(phi.usd_per_throughput_hour(&CM), None);
}

#[test]
fn slow_family_floor_and_override() {
    // E7-4860 v2 (Ivy Bridge, no AVX2/FMA), live offer 51765869.
    let (ok, rej) = rank(&fixture(), &cpu_filter(), &CM);
    assert!(!ids(&ok).contains(&51765869));
    let why = &rej.iter().find(|(r, _)| *r == 51765869).unwrap().1;
    assert!(why.contains("--min-cpu-speed"), "{why}");
    // Lowering the floor admits it, ranked by throughput (0.45 x 96 threads).
    let mut f = cpu_filter();
    f.min_cpu_speed = 0.0;
    let (ok, _) = rank(&fixture(), &f, &CM);
    let e7 = ok.iter().find(|o| o.id == 51765869).unwrap();
    assert!((e7.cpu_throughput().unwrap() - 0.45 * 96.0).abs() < 1e-9);
    assert_ne!(ok[0].id, 51765869);
    let mut bad = cpu_filter();
    bad.min_cpu_speed = -1.0;
    assert!(bad.problems().iter().any(|p| p.contains("--min-cpu-speed")));
}

#[test]
fn non_amd64_rejected_in_cpu_mode_only() {
    let o = fixture().into_iter().find(|o| o.id == 90000004).unwrap();
    let mut f = cpu_filter();
    f.min_cpu_speed = 0.0;
    assert!(f.reject_reason(&o, &CM).unwrap().contains("cpu_arch arm64"));
}

#[test]
fn gpu_mode_ignores_cpu_family() {
    // GPU mode never consults the CPU table: the Phi host is judged on its GPU.
    let phi = fixture().into_iter().find(|o| o.id == 31231107).unwrap();
    let f = OfferFilter {
        gpu_names: vec!["RTX 5060 Ti".into()],
        num_gpus: 1,
        min_gpu_ram_gb: 16.0,
        min_cuda: 12.6,
        cpu_mode: false,
        min_cpu_cores: 0.0,
        ..cpu_filter()
    };
    assert_eq!(f.reject_reason(&phi, &CM), None);
}
