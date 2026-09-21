use super::*;

impl<E: Embedder> crate::engine::Engine<E> {
    /// Run a campaign over [`EngineOptions`] for this engine's embedder.
    /// Returns the baseline → champion comparison, the chained receipts, and
    /// the (twice-scored) test metrics. Does not mutate the engine's own bank or
    /// caches — pass the champion options to [`with_options`](Self::with_options)
    /// to adopt them.
    pub fn optimize(&self, spec: &CampaignSpec) -> Result<CampaignReport> {
        let dims = self.embedder().dims();
        let model_id = self.embedder().id().to_string();
        let allow_cal = !model_id.ends_with("@test-double");

        // Compile the scorer once (embed criteria / legend / predicate).
        let qtexts = question_texts(&spec.question_def);
        let qrefs: Vec<&str> = qtexts.iter().map(String::as_str).collect();
        let qembs = if qrefs.is_empty() {
            Vec::new()
        } else {
            self.embedder().embed(&qrefs)?
        };
        let compiled = build_compiled(&spec.question_def, &qembs);

        // Embed every row's state once.
        let row_refs: Vec<&str> = spec.rows.iter().map(|r| r.text.as_str()).collect();
        let row_embs = if row_refs.is_empty() {
            Vec::new()
        } else {
            self.embedder().embed(&row_refs)?
        };
        if row_embs.len() != spec.rows.len() {
            return Err(TypesafeError::Embedder(
                "row embedding count mismatch".into(),
            ));
        }

        let index = class_index(&compiled);
        let resolved: Vec<Resolved> = spec
            .rows
            .iter()
            .zip(row_embs)
            .map(|(row, emb)| Resolved {
                emb,
                class: index
                    .as_ref()
                    .and_then(|m| m.get(row.label.as_str()).copied()),
                y: crate::engine::support::parse_noul_label(&row.label),
                split: row.split,
            })
            .collect();

        // Training / calibration splits (shared by every arm).
        let class_train = class_pairs(&resolved, Split::Train);
        let class_calib = class_pairs(&resolved, Split::Calibration);
        let noul_train = noul_pairs(&resolved, Split::Train);
        let noul_calib = noul_pairs(&resolved, Split::Calibration);

        let fit_opts = |opts: &EngineOptions| -> Artifact {
            match &compiled {
                Compiled::Class(cp) => {
                    fit::fit_class_artifact(opts, cp, &class_train, &class_calib, dims, allow_cal)
                }
                Compiled::Noul { .. } => {
                    fit::fit_noul_artifact(opts, &noul_train, &noul_calib, dims, allow_cal)
                }
            }
        };

        // Incumbent (baseline) — the champion of the campaign starts here and
        // is replaced only on a promotion (successive).
        let mut champ_opts = spec.base_options.clone();
        let mut champ_art = fit_opts(&champ_opts);
        let baseline_art = fit_opts(&spec.base_options);

        let proposals = if spec.proposals.is_empty() {
            default_grid(&spec.base_options)
        } else {
            spec.proposals.clone()
        };

        let mut gate = Gate {
            alpha: spec.alpha,
            lambda: spec.lambda,
            transfer: crate::loop_gate::TransferHoldout {
                tolerance: spec.transfer_tolerance,
            },
            budget: Budget::new(spec.budget_per_day, spec.day_key.clone()),
            accuracy_tolerance: spec.accuracy_tolerance,
        };
        let mut log = ReceiptLog::new();
        let mut arms = Vec::new();
        let mut promotions = 0usize;

        for (i, opts) in proposals.iter().enumerate() {
            let cand_art = fit_opts(opts);
            // Paired validation outcomes (accuracy + NLL): incumbent vs candidate.
            let (paired, paired_nll) = self.paired_validation(
                &compiled,
                &index,
                &resolved,
                &champ_opts,
                &champ_art,
                opts,
                &cand_art,
                &model_id,
            );
            let base_xfer = self.split_accuracy(
                &compiled,
                &index,
                &resolved,
                &champ_opts,
                &champ_art,
                Split::Transfer,
                &model_id,
            );
            let cand_xfer = self.split_accuracy(
                &compiled,
                &index,
                &resolved,
                opts,
                &cand_art,
                Split::Transfer,
                &model_id,
            );
            let evidence = Evidence {
                paired,
                paired_nll: Some(paired_nll),
                baseline_transfer_acc: base_xfer.0,
                champion_transfer_acc: cand_xfer.0,
                transfer_n: cand_xfer.1,
                model_id: model_id.clone(),
                head: cand_art.head(),
                temperature: cand_art.temperature(),
                created_seq: spec.created_seq_base + i as u64,
                created: spec.created.clone(),
            };
            let proposal = Proposal {
                id: spec.created_seq_base + i as u64 + 1,
                parent: Some(spec.created_seq_base),
                kind: ProposalKind::ModelArm,
                description_hash: options_hash(opts),
            };
            let outcome = gate.evaluate(proposal, &evidence, &spec.day_key);
            let promoted = matches!(outcome.decision, GateDecision::Promote);
            log.push(outcome.receipt.clone());
            arms.push(ArmResult {
                proposal_id: proposal.id,
                options: opts.clone(),
                decision: outcome.decision.clone(),
                val: outcome.receipt.val.clone(),
                transfer: outcome.receipt.transfer.clone(),
                statistic: outcome.receipt.statistic.clone(),
                calibration_statistic: outcome.receipt.calibration_statistic.clone(),
                promoted_by: outcome.receipt.promoted_by,
                promoted,
            });
            if promoted {
                champ_opts = opts.clone();
                champ_art = cand_art;
                promotions += 1;
            }
        }

        // Final per-split metrics for baseline and champion.
        let (baseline_val, champion_val) = self.pair_metrics(
            &compiled,
            &index,
            &resolved,
            &spec.base_options,
            &baseline_art,
            &champ_opts,
            &champ_art,
            Split::Validation,
            &model_id,
        );
        let (baseline_transfer, champion_transfer) = self.pair_metrics(
            &compiled,
            &index,
            &resolved,
            &spec.base_options,
            &baseline_art,
            &champ_opts,
            &champ_art,
            Split::Transfer,
            &model_id,
        );

        // Test — reachable ONLY through `score_test_split`, which consumes a
        // single-use `TestToken`. A `Campaign` mints exactly two (baseline,
        // champion), so the test split is physically unscoreable a third time
        // (ADR-004 gate 1). `split_scored` refuses `Split::Test` outright.
        let mut campaign = Campaign::new();
        let mut test_scorings = 0u8;
        let baseline_test = match campaign.test_token() {
            Some(token) => {
                test_scorings += 1;
                let (acc, n, e) = self.score_test_split(
                    token,
                    &compiled,
                    &index,
                    &resolved,
                    &spec.base_options,
                    &baseline_art,
                    &model_id,
                );
                Metrics {
                    baseline_accuracy: acc,
                    champion_accuracy: acc,
                    n,
                    ece: Some(e),
                    brier: None,
                }
            }
            None => empty_metrics(),
        };
        let champion_test = match campaign.test_token() {
            Some(token) => {
                test_scorings += 1;
                let (acc, n, e) = self.score_test_split(
                    token,
                    &compiled,
                    &index,
                    &resolved,
                    &champ_opts,
                    &champ_art,
                    &model_id,
                );
                Metrics {
                    baseline_accuracy: baseline_test.baseline_accuracy,
                    champion_accuracy: acc,
                    n,
                    ece: Some(e),
                    brier: None,
                }
            }
            None => empty_metrics(),
        };
        // A third scoring is impossible: the campaign is out of tokens.
        debug_assert!(campaign.test_token().is_none());

        // Final champion receipt — carries the test metrics into the hash chain
        // (ADR-004 "every promotion writes a receipt … with the test statistic").
        let champion_receipt = Receipt {
            seq: 0,
            proposal: Proposal {
                id: spec.created_seq_base,
                parent: None,
                kind: ProposalKind::ModelArm,
                description_hash: options_hash(&champ_opts),
            },
            parent: None,
            kind: ProposalKind::ModelArm,
            val: champion_val.clone(),
            transfer: champion_transfer.clone(),
            test: Some(champion_test.clone()),
            statistic: PairedSequentialTest::new(spec.alpha, spec.lambda).statistic(),
            calibration_statistic: None,
            promoted_by: None,
            decision: if promotions > 0 {
                GateDecision::Promote
            } else {
                GateDecision::Reject("no proposal reached significance".into())
            },
            model_id: model_id.clone(),
            head: champ_art.head(),
            temperature: champ_art.temperature(),
            budget_consumed: gate.budget.consumed,
            created_seq: spec.created_seq_base,
            created: spec.created.clone(),
            prev_hash: String::new(),
            hash: String::new(),
            signature: None,
        };
        log.push(champion_receipt);

        let budget_consumed = gate.budget.consumed;
        Ok(CampaignReport {
            question: spec.question.clone(),
            embedder_id: model_id,
            baseline_options: spec.base_options.clone(),
            champion_options: champ_opts,
            baseline_val,
            champion_val,
            baseline_transfer,
            champion_transfer,
            baseline_test,
            champion_test,
            arms,
            promotions,
            budget_consumed,
            test_scorings,
            receipts: log,
        })
    }
}
