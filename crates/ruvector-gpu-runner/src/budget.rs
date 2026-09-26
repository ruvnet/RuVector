//! Budget gate (pre-launch) and watchdog verdicts (post-launch). Pure logic.

use crate::offer::Estimate;

#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub max_usd: f64,
    pub max_hours: f64,
    /// Destroy when projected spend reaches this fraction of the cap, so the
    /// poll interval cannot push actual spend past `max_usd`.
    pub watchdog_fraction: f64,
}

/// Pre-launch gate. Returns every reason to refuse (empty = OK).
pub fn plan_blockers(b: &Budget, est: &Estimate, credit_usd: Option<f64>) -> Vec<String> {
    let mut out = Vec::new();
    if !(b.max_usd > 0.0 && b.max_hours > 0.0) {
        out.push("max-usd and max-hours must be > 0".into());
    }
    if !(0.5..=1.0).contains(&b.watchdog_fraction) {
        out.push("watchdog fraction must be within [0.5, 1.0]".into());
    }
    if est.worst_case_usd > b.max_usd {
        out.push(format!(
            "estimate ${:.2} (${:.4}/h x {}h + ${:.2} transfer) exceeds --max-usd ${:.2}",
            est.worst_case_usd, est.hourly_usd, b.max_hours, est.bandwidth_usd, b.max_usd
        ));
    }
    match credit_usd {
        Some(c) if c < est.worst_case_usd => out.push(format!(
            "account credit ${c:.2} < worst-case estimate ${:.2}",
            est.worst_case_usd
        )),
        None => out.push("could not read account credit".into()),
        _ => {}
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
pub enum Watch {
    Ok { spent_usd: f64 },
    BudgetCap { spent_usd: f64 },
    Timeout { spent_usd: f64 },
}

/// Evaluate the watchdog. `elapsed_h` counts from the create call (billing
/// starts at creation, not at `running`).
pub fn watchdog(b: &Budget, hourly_usd: f64, bandwidth_usd: f64, elapsed_h: f64) -> Watch {
    let spent = elapsed_h * hourly_usd + bandwidth_usd;
    if spent >= b.watchdog_fraction * b.max_usd {
        Watch::BudgetCap { spent_usd: spent }
    } else if elapsed_h >= b.watchdog_fraction * b.max_hours {
        Watch::Timeout { spent_usd: spent }
    } else {
        Watch::Ok { spent_usd: spent }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn est(hourly: f64, hours: f64) -> Estimate {
        Estimate {
            hourly_usd: hourly,
            compute_hourly_usd: hourly,
            storage_hourly_usd: 0.0,
            bandwidth_usd: 0.1,
            max_hours: hours,
            worst_case_usd: hourly * hours + 0.1,
        }
    }
    const B: Budget = Budget {
        max_usd: 5.0,
        max_hours: 3.0,
        watchdog_fraction: 0.95,
    };

    #[test]
    fn refuses_over_budget() {
        assert!(plan_blockers(&B, &est(0.5, 3.0), Some(100.0)).is_empty());
        let r = plan_blockers(&B, &est(2.0, 3.0), Some(100.0));
        assert!(r.iter().any(|m| m.contains("exceeds --max-usd")));
    }

    #[test]
    fn refuses_low_credit_or_unknown_credit() {
        assert!(!plan_blockers(&B, &est(0.5, 3.0), Some(1.0)).is_empty());
        assert!(!plan_blockers(&B, &est(0.5, 3.0), None).is_empty());
    }

    #[test]
    fn watchdog_trips_on_budget_before_cap() {
        let b = Budget {
            max_usd: 1.0,
            max_hours: 100.0,
            watchdog_fraction: 0.9,
        };
        assert!(matches!(watchdog(&b, 1.0, 0.0, 0.5), Watch::Ok { .. }));
        assert!(matches!(
            watchdog(&b, 1.0, 0.0, 0.9),
            Watch::BudgetCap { .. }
        ));
    }

    #[test]
    fn watchdog_trips_on_time() {
        assert!(matches!(watchdog(&B, 0.1, 0.0, 2.9), Watch::Timeout { .. }));
        assert!(matches!(watchdog(&B, 0.1, 0.0, 1.0), Watch::Ok { .. }));
    }
}
