//! Offer model, selection filters and cost estimation.
//!
//! Server-side filters sent to vast.ai are a pre-filter only: every constraint
//! that gates spend is re-checked client-side in [`OfferFilter::reject_reason`],
//! because the bundles endpoint has been observed to silently ignore some keys
//! (e.g. `verification` returned `deverified` hosts).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Hard floor for host reliability. The CLI refuses anything lower.
pub const RELIABILITY_FLOOR: f64 = 0.98;
/// vast.ai `hosting_type` value used by datacenter hosts (verified against
/// a `datacenter: {eq: true}` query: every result carried `hosting_type = 1`).
pub const HOSTING_TYPE_DATACENTER: i64 = 1;
/// Hours per month used by vast.ai to convert `storage_cost` ($/GB/month).
const HOURS_PER_MONTH: f64 = 720.0;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Offer {
    pub id: u64,
    pub gpu_name: String,
    pub num_gpus: u32,
    /// MiB.
    pub gpu_ram: f64,
    pub dph_total: f64,
    #[serde(default)]
    pub dph_base: Option<f64>,
    #[serde(default)]
    pub cuda_max_good: Option<f64>,
    /// $/GB/month.
    #[serde(default)]
    pub storage_cost: Option<f64>,
    /// $/GB.
    #[serde(default)]
    pub inet_up_cost: Option<f64>,
    /// $/GB.
    #[serde(default)]
    pub inet_down_cost: Option<f64>,
    #[serde(default)]
    pub reliability2: Option<f64>,
    #[serde(default)]
    pub verification: Option<String>,
    #[serde(default)]
    pub hosting_type: Option<i64>,
    #[serde(default)]
    pub geolocation: Option<String>,
    #[serde(default)]
    pub disk_space: Option<f64>,
    #[serde(default)]
    pub machine_id: Option<u64>,
    #[serde(default)]
    pub rentable: Option<bool>,
    #[serde(default)]
    pub rented: Option<bool>,
}

impl Offer {
    pub fn is_datacenter(&self) -> bool {
        self.hosting_type == Some(HOSTING_TYPE_DATACENTER)
    }
    pub fn is_verified(&self) -> bool {
        self.verification.as_deref() == Some("verified")
    }
}

/// What the job needs from the host, beyond GPU compute.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct CostModel {
    pub disk_gb: f64,
    pub upload_gb: f64,
    pub download_gb: f64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Estimate {
    /// Planning rate: max(dph_total, dph_base + disk storage).
    pub hourly_usd: f64,
    pub compute_hourly_usd: f64,
    pub storage_hourly_usd: f64,
    /// One-off transfer cost for expected up/download volume.
    pub bandwidth_usd: f64,
    pub max_hours: f64,
    /// hourly * max_hours + bandwidth: worst case if the job runs to the cap.
    pub worst_case_usd: f64,
}

pub fn estimate(o: &Offer, cm: &CostModel, max_hours: f64) -> Estimate {
    let base = o.dph_base.unwrap_or(o.dph_total);
    let storage = cm.disk_gb * o.storage_cost.unwrap_or(0.0) / HOURS_PER_MONTH;
    let hourly = o.dph_total.max(base + storage);
    let bandwidth = cm.upload_gb * o.inet_up_cost.unwrap_or(0.0)
        + cm.download_gb * o.inet_down_cost.unwrap_or(0.0);
    Estimate {
        hourly_usd: hourly,
        compute_hourly_usd: base,
        storage_hourly_usd: storage,
        bandwidth_usd: bandwidth,
        max_hours,
        worst_case_usd: hourly * max_hours + bandwidth,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct OfferFilter {
    pub gpu_names: Vec<String>,
    pub num_gpus: u32,
    pub min_gpu_ram_gb: f64,
    pub min_cuda: f64,
    pub max_dph: f64,
    pub min_reliability: f64,
    pub require_datacenter: bool,
}

impl OfferFilter {
    /// Server-side pre-filter for `POST /bundles/`.
    pub fn query_json(&self, disk_gb: f64, limit: u32) -> Value {
        let mut q = json!({
            "verified": {"eq": true},
            "rentable": {"eq": true},
            "rented": {"eq": false},
            "reliability2": {"gte": self.min_reliability},
            "num_gpus": {"eq": self.num_gpus},
            "gpu_name": {"in": self.gpu_names},
            "gpu_ram": {"gte": self.min_gpu_ram_gb * 1000.0},
            "cuda_max_good": {"gte": self.min_cuda},
            "dph_total": {"lte": self.max_dph},
            "disk_space": {"gte": disk_gb},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
            "limit": limit,
        });
        if self.require_datacenter {
            q["datacenter"] = json!({"eq": true});
        }
        q
    }

    /// Client-side enforcement. `None` means the offer is acceptable.
    pub fn reject_reason(&self, o: &Offer, cm: &CostModel) -> Option<String> {
        let rel = o.reliability2.unwrap_or(0.0);
        let cuda = o.cuda_max_good.unwrap_or(0.0);
        let hourly = estimate(o, cm, 1.0).hourly_usd;
        if !o.is_verified() {
            return Some(format!("host not verified ({:?})", o.verification));
        }
        if rel < self.min_reliability.max(RELIABILITY_FLOOR) {
            return Some(format!(
                "reliability {rel:.4} < {:.2}",
                self.min_reliability
            ));
        }
        if !self.gpu_names.iter().any(|g| g == &o.gpu_name) {
            return Some(format!("gpu {} not in allow-list", o.gpu_name));
        }
        if o.num_gpus != self.num_gpus {
            return Some(format!("num_gpus {} != {}", o.num_gpus, self.num_gpus));
        }
        if o.gpu_ram < self.min_gpu_ram_gb * 1000.0 {
            return Some(format!(
                "gpu_ram {} MiB < {} GB",
                o.gpu_ram, self.min_gpu_ram_gb
            ));
        }
        if cuda < self.min_cuda {
            return Some(format!("cuda {cuda} < {}", self.min_cuda));
        }
        if o.disk_space.unwrap_or(0.0) < cm.disk_gb {
            return Some(format!("disk {:?} GB < {} GB", o.disk_space, cm.disk_gb));
        }
        if hourly > self.max_dph {
            return Some(format!("rate ${hourly:.3}/h > max ${:.3}/h", self.max_dph));
        }
        if o.rented == Some(true) || o.rentable == Some(false) {
            return Some("not rentable".into());
        }
        if self.require_datacenter && !o.is_datacenter() {
            return Some("not a datacenter host".into());
        }
        None
    }
}

/// Filter then rank: datacenter hosts first, then lowest planning rate,
/// then highest reliability. Returns (ranked acceptable offers, rejections).
pub fn rank(offers: &[Offer], f: &OfferFilter, cm: &CostModel) -> (Vec<Offer>, Vec<(u64, String)>) {
    let mut ok = Vec::new();
    let mut rejected = Vec::new();
    for o in offers {
        match f.reject_reason(o, cm) {
            None => ok.push(o.clone()),
            Some(r) => rejected.push((o.id, r)),
        }
    }
    ok.sort_by(|a, b| {
        let ra = estimate(a, cm, 1.0).hourly_usd;
        let rb = estimate(b, cm, 1.0).hourly_usd;
        b.is_datacenter()
            .cmp(&a.is_datacenter())
            .then(ra.total_cmp(&rb))
            .then(
                b.reliability2
                    .unwrap_or(0.0)
                    .total_cmp(&a.reliability2.unwrap_or(0.0)),
            )
    });
    (ok, rejected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<Offer> {
        let v: Value = serde_json::from_str(include_str!("../tests/fixtures/offers.json")).unwrap();
        serde_json::from_value(v["offers"].clone()).unwrap()
    }
    fn filter() -> OfferFilter {
        OfferFilter {
            gpu_names: vec!["RTX 4090".into(), "RTX A6000".into()],
            num_gpus: 1,
            min_gpu_ram_gb: 24.0,
            min_cuda: 12.4,
            max_dph: 1.0,
            min_reliability: 0.98,
            require_datacenter: false,
        }
    }
    const CM: CostModel = CostModel {
        disk_gb: 60.0,
        upload_gb: 2.0,
        download_gb: 10.0,
    };

    #[test]
    fn rejects_unverified_and_unreliable() {
        let (ok, rej) = rank(&fixture(), &filter(), &CM);
        assert!(!ok.is_empty());
        for o in &ok {
            assert!(o.is_verified());
            assert!(o.reliability2.unwrap() >= RELIABILITY_FLOOR);
        }
        assert!(rej.iter().any(|(_, r)| r.contains("not verified")));
        let low: Vec<u64> = fixture()
            .iter()
            .filter(|o| o.reliability2.unwrap_or(0.0) < 0.98)
            .map(|o| o.id)
            .collect();
        assert!(!low.is_empty());
        assert!(ok.iter().all(|o| !low.contains(&o.id)));
    }

    #[test]
    fn datacenter_ranked_first() {
        let (ok, _) = rank(&fixture(), &filter(), &CM);
        let first_non_dc = ok
            .iter()
            .position(|o| !o.is_datacenter())
            .unwrap_or(ok.len());
        assert!(ok[first_non_dc..].iter().all(|o| !o.is_datacenter()));
        assert!(ok[0].is_datacenter());
    }

    #[test]
    fn estimate_includes_storage_and_bandwidth() {
        let o = &fixture()[0];
        let e = estimate(o, &CM, 3.0);
        assert!(e.hourly_usd >= o.dph_total);
        assert!(e.storage_hourly_usd > 0.0);
        let expect = e.hourly_usd * 3.0 + e.bandwidth_usd;
        assert!((e.worst_case_usd - expect).abs() < 1e-9);
    }

    #[test]
    fn reliability_floor_cannot_be_lowered() {
        let mut f = filter();
        f.min_reliability = 0.5;
        let mut o = fixture()[0].clone();
        o.reliability2 = Some(0.97);
        assert!(f.reject_reason(&o, &CM).unwrap().contains("reliability"));
    }

    #[test]
    fn max_dph_enforced_on_planning_rate() {
        let mut f = filter();
        f.max_dph = 0.01;
        let (ok, _) = rank(&fixture(), &f, &CM);
        assert!(ok.is_empty());
    }
}
