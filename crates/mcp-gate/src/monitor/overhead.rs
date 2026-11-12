//! Monitoring overhead accounting (PIR WP33, ADR-337).
//!
//! The design target for this ladder is **under 5% average monitoring
//! overhead** while every mandatory class is inspected at 100%. A target
//! nobody measures is a slogan, so the ladder records what it spends and
//! exposes the ratio.
//!
//! # What the denominator is
//!
//! [`OverheadAccount::fraction`] is `monitoring_time / workload_time`, where
//! workload time is what the caller reports for the operations themselves.
//! Both are caller-attributed wall-clock microseconds. This is deliberately
//! *not* modelled on OpenAI's published figure (their 2026-08-18 update
//! estimates monitoring at roughly 20% of the inference compute **being
//! monitored** — a subset denominator over a deliberately narrow monitoring
//! scope, which is not comparable to an average across all operations).
//!
//! # Why the counters are separate
//!
//! `inspections` and `mandatory_inspections` are tracked independently so the
//! overhead figure cannot be improved by quietly narrowing the mandatory set:
//! declassifying a class shows up immediately as `mandatory_inspections`
//! falling while throughput is unchanged. The ratio alone would hide that.
//!
//! A count with no denominator cannot support a coverage claim, though.
//! `mandatory_inspections = 0` is what a ladder reports when it saw no
//! mandatory operations *and* what it reports when it stopped classifying
//! them — indistinguishable without knowing how many it saw. So
//! `mandatory_operations_seen` is recorded at classification time, before any
//! decision can divert the operation, and
//! [`OverheadAccount::mandatory_coverage`] is the ratio that actually means
//! "100% of mandatory operations were inspected".
//!
//! Note the two are legitimately unequal when an operation halts *before*
//! reaching an investigator — a critical detector score, or a failure on any
//! fail-closed path. Those operations were stopped, not skipped, which is the
//! stronger outcome; coverage below 1.0 therefore wants reading alongside
//! [`OverheadAccount::halts`] rather than alone.

/// Running totals for one ladder's monitoring activity.
///
/// Counters saturate rather than wrap: an overflowed counter that silently
/// restarted at zero would understate exactly the quantity this type exists
/// to make checkable.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OverheadAccount {
    workload_us: u128,
    monitoring_us: u128,
    operations: u64,
    inspections: u64,
    mandatory_operations_seen: u64,
    mandatory_inspections: u64,
    rungs_purchased: u64,
    halts: u64,
}

impl OverheadAccount {
    /// A fresh, empty account.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the cost of the operation itself.
    pub(crate) fn record_workload(&mut self, micros: u128) {
        self.workload_us = self.workload_us.saturating_add(micros);
        self.operations = self.operations.saturating_add(1);
    }

    /// Record time spent deciding and investigating.
    pub(crate) fn record_monitoring(&mut self, micros: u128) {
        self.monitoring_us = self.monitoring_us.saturating_add(micros);
    }

    /// Record that an operation was classified into at least one mandatory
    /// class. Called immediately after classification, before any decision
    /// can divert the operation, so it is the honest denominator for
    /// [`Self::mandatory_coverage`].
    pub(crate) fn record_mandatory_seen(&mut self) {
        self.mandatory_operations_seen = self.mandatory_operations_seen.saturating_add(1);
    }

    /// Record that an operation was inspected, and whether it was inspected
    /// because a mandatory class required it.
    pub(crate) fn record_inspection(&mut self, mandatory: bool) {
        self.inspections = self.inspections.saturating_add(1);
        if mandatory {
            self.mandatory_inspections = self.mandatory_inspections.saturating_add(1);
        }
    }

    /// Record one purchased investigator rung.
    pub(crate) fn record_rung(&mut self) {
        self.rungs_purchased = self.rungs_purchased.saturating_add(1);
    }

    /// Record a terminal halt.
    pub(crate) fn record_halt(&mut self) {
        self.halts = self.halts.saturating_add(1);
    }

    /// Monitoring time as a fraction of workload time.
    ///
    /// `None` when no workload has been recorded. Returning `None` rather
    /// than `0.0` matters: a zero denominator would otherwise report a
    /// flattering `0.0` overhead for a ladder that has never run, which is
    /// precisely the number someone would quote.
    pub fn fraction(&self) -> Option<f64> {
        if self.workload_us == 0 {
            return None;
        }
        Some(self.monitoring_us as f64 / self.workload_us as f64)
    }

    /// Total microseconds attributed to the monitored operations.
    pub fn workload_us(&self) -> u128 {
        self.workload_us
    }

    /// Total microseconds attributed to monitoring.
    pub fn monitoring_us(&self) -> u128 {
        self.monitoring_us
    }

    /// Operations seen.
    pub fn operations(&self) -> u64 {
        self.operations
    }

    /// Operations inspected beyond the tiny detector.
    pub fn inspections(&self) -> u64 {
        self.inspections
    }

    /// Inspections that a mandatory class compelled.
    ///
    /// Read alongside [`Self::mandatory_operations_seen`]: this count alone
    /// cannot distinguish "no mandatory operations arrived" from "mandatory
    /// operations stopped being recognised".
    pub fn mandatory_inspections(&self) -> u64 {
        self.mandatory_inspections
    }

    /// Operations that classification placed in at least one mandatory class.
    pub fn mandatory_operations_seen(&self) -> u64 {
        self.mandatory_operations_seen
    }

    /// Fraction of mandatory operations that reached an investigator.
    ///
    /// `None` when none were seen — not `1.0`, which would report perfect
    /// coverage for a ladder that has never classified anything.
    ///
    /// Below 1.0 does not necessarily mean an operation slipped through: an
    /// operation halted before reaching a rung (critical detector score, or
    /// any fail-closed path) counts in the denominator but not the numerator.
    /// Read with [`Self::halts`].
    pub fn mandatory_coverage(&self) -> Option<f64> {
        if self.mandatory_operations_seen == 0 {
            return None;
        }
        Some(self.mandatory_inspections as f64 / self.mandatory_operations_seen as f64)
    }

    /// Investigator rungs purchased across all operations.
    pub fn rungs_purchased(&self) -> u64 {
        self.rungs_purchased
    }

    /// Operations that ended in a terminal halt.
    pub fn halts(&self) -> u64 {
        self.halts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_account_reports_no_fraction_rather_than_zero() {
        assert_eq!(OverheadAccount::new().fraction(), None);
    }

    #[test]
    fn fraction_is_monitoring_over_workload() {
        let mut a = OverheadAccount::new();
        a.record_workload(1000);
        a.record_monitoring(30);
        let f = a.fraction().expect("workload recorded");
        assert!((f - 0.03).abs() < 1e-12, "got {f}");
    }

    #[test]
    fn counters_saturate_rather_than_wrap() {
        let mut a = OverheadAccount::new();
        a.record_workload(u128::MAX);
        a.record_workload(10);
        assert_eq!(a.workload_us(), u128::MAX);
        a.record_monitoring(u128::MAX);
        a.record_monitoring(10);
        assert_eq!(a.monitoring_us(), u128::MAX);
    }

    #[test]
    fn mandatory_inspections_are_counted_separately() {
        let mut a = OverheadAccount::new();
        a.record_inspection(true);
        a.record_inspection(false);
        a.record_inspection(false);
        assert_eq!(a.inspections(), 3);
        assert_eq!(a.mandatory_inspections(), 1);
    }

    #[test]
    fn coverage_needs_a_denominator_and_reports_none_without_one() {
        let mut a = OverheadAccount::new();
        // Zero inspections and zero seen is not 100% coverage.
        assert_eq!(a.mandatory_coverage(), None);
        a.record_mandatory_seen();
        a.record_mandatory_seen();
        a.record_inspection(true);
        assert_eq!(a.mandatory_coverage(), Some(0.5));
        a.record_inspection(true);
        assert_eq!(a.mandatory_coverage(), Some(1.0));
    }

    #[test]
    fn a_ladder_that_stopped_classifying_is_distinguishable_from_a_quiet_one() {
        // The failure MEDIUM-5 identified: both of these report
        // mandatory_inspections == 0, and only the denominator tells them
        // apart.
        let quiet = OverheadAccount::new();
        let mut stopped_classifying = OverheadAccount::new();
        stopped_classifying.record_mandatory_seen();
        assert_eq!(quiet.mandatory_inspections(), 0);
        assert_eq!(stopped_classifying.mandatory_inspections(), 0);
        assert_eq!(quiet.mandatory_coverage(), None);
        assert_eq!(stopped_classifying.mandatory_coverage(), Some(0.0));
    }
}
