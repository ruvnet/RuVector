use super::*;

/// The promotion gate. Holds the sequential-test parameters, the transfer
/// tolerance and the (stateful) day budget.
#[derive(Debug, Clone)]
pub struct Gate {
    pub alpha: f32,
    pub lambda: f32,
    pub transfer: TransferHoldout,
    pub budget: Budget,
    /// Accuracy non-inferiority margin for the calibration criterion (ADR-004
    /// gate 2b): a calibration-only promotion is allowed only if the champion's
    /// validation accuracy is within this of the baseline's.
    pub accuracy_tolerance: f32,
}

impl Gate {
    #[must_use]
    pub fn new(budget: Budget) -> Self {
        Self {
            alpha: 0.05,
            lambda: 0.5,
            transfer: TransferHoldout::default(),
            budget,
            accuracy_tolerance: 0.02,
        }
    }

    /// Judge one proposal. Takes `&mut self` because the budget is stateful
    /// (the brief's `&self` cannot spend a budget without interior mutability,
    /// which would be worse); `day_key` charges the right day.
    pub fn evaluate(
        &mut self,
        proposal: Proposal,
        evidence: &Evidence,
        day_key: &str,
    ) -> GateOutcome {
        let val = evidence.val_metrics();
        let transfer = evidence.transfer_metrics();

        // Budget first: exhaustion pauses, never lowers the bar.
        if !self.budget.try_consume(day_key) {
            let stat = PairedSequentialTest::new(self.alpha, self.lambda).statistic();
            return self.outcome(
                proposal,
                evidence,
                val,
                transfer,
                stat,
                None,
                None,
                GateDecision::Paused("daily evaluation budget exhausted".into()),
            );
        }

        // Criterion 1 (gate 2): paired accuracy test.
        let mut acc_test = PairedSequentialTest::new(self.alpha, self.lambda);
        acc_test.update_all(&evidence.paired);
        let acc_stat = acc_test.statistic();

        // Criterion 2b: paired NLL (calibration) test, when NLL is provided.
        let cal_pairs = evidence.calibration_pairs();
        let cal_stat = if cal_pairs.is_empty() {
            None
        } else {
            let mut cal_test = PairedSequentialTest::new(self.alpha, self.lambda);
            cal_test.update_all(&cal_pairs);
            Some((cal_test.rejected(), cal_test.statistic()))
        };

        let transfer_ok = self.transfer.passes(
            evidence.baseline_transfer_acc,
            evidence.champion_transfer_acc,
        );
        let accuracy_non_inferior = evidence.val_accuracy_delta() >= -self.accuracy_tolerance;
        let calibration_promotes = cal_stat
            .as_ref()
            .map(|(rejected, _)| *rejected && accuracy_non_inferior)
            .unwrap_or(false);

        // Promote on EITHER criterion; transfer holdout gates both (ADR-004).
        let (decision, promoted_by) = if !transfer_ok {
            (
                GateDecision::Reject("transfer split regressed beyond tolerance".into()),
                None,
            )
        } else if acc_test.rejected() {
            (GateDecision::Promote, Some(PromotionCriterion::Accuracy))
        } else if calibration_promotes {
            (GateDecision::Promote, Some(PromotionCriterion::Calibration))
        } else {
            (
                GateDecision::Reject(
                    "neither the accuracy nor the calibration test rejected no-improvement".into(),
                ),
                None,
            )
        };

        let cal_statistic = cal_stat.map(|(_, s)| s);
        self.outcome(
            proposal,
            evidence,
            val,
            transfer,
            acc_stat,
            cal_statistic,
            promoted_by,
            decision,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn outcome(
        &self,
        proposal: Proposal,
        evidence: &Evidence,
        val: Metrics,
        transfer: Metrics,
        statistic: TestStatistic,
        calibration_statistic: Option<TestStatistic>,
        promoted_by: Option<PromotionCriterion>,
        decision: GateDecision,
    ) -> GateOutcome {
        let receipt = Receipt {
            seq: 0,
            proposal,
            parent: proposal.parent,
            kind: proposal.kind,
            val,
            transfer,
            test: None,
            statistic,
            calibration_statistic,
            promoted_by,
            decision: decision.clone(),
            model_id: evidence.model_id.clone(),
            head: evidence.head,
            temperature: evidence.temperature,
            budget_consumed: self.budget.consumed,
            created_seq: evidence.created_seq,
            created: evidence.created.clone(),
            prev_hash: String::new(),
            hash: String::new(),
            signature: None,
        };
        GateOutcome { decision, receipt }
    }

    /// Score the frozen test split. Requires a [`TestToken`], which only a
    /// [`Campaign`] can mint (twice) — so this is unreachable from inside the
    /// loop. Consumes the token by value.
    #[must_use]
    pub fn score_test(
        &self,
        _token: TestToken,
        baseline_acc: f32,
        champion_acc: f32,
        n: u32,
    ) -> Metrics {
        Metrics {
            baseline_accuracy: baseline_acc,
            champion_accuracy: champion_acc,
            n,
            ece: None,
            brier: None,
        }
    }
}
