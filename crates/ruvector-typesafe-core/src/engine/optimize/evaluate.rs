use super::*;

impl<E: Embedder> crate::engine::Engine<E> {
    /// Predict the class index and confidence for one state under `opts`.
    fn class_predict(
        &self,
        cp: &ClassProtos,
        art: &Artifact,
        opts: &EngineOptions,
        emb: &[f32],
        model: &str,
        index: &BTreeMap<String, usize>,
    ) -> (usize, f32) {
        match fit::class_answer(opts, cp, art, emb, model) {
            Answer::Choice { choice, meta, .. } => {
                (*index.get(&choice).unwrap_or(&usize::MAX), meta.confidence)
            }
            Answer::Score { score, meta, .. } => (score, meta.confidence),
            Answer::Noul { .. } => (usize::MAX, 0.0),
        }
    }

    /// `(correct, NLL of the true class)` for one class item.
    #[allow(clippy::too_many_arguments)]
    fn class_predict_nll(
        &self,
        cp: &ClassProtos,
        art: &Artifact,
        opts: &EngineOptions,
        emb: &[f32],
        model: &str,
        index: &BTreeMap<String, usize>,
        truth: usize,
    ) -> (bool, f32) {
        let (pred, _) = self.class_predict(cp, art, opts, emb, model, index);
        let p_true = match fit::class_answer(opts, cp, art, emb, model) {
            Answer::Choice { probabilities, .. } => cp
                .keys
                .get(truth)
                .and_then(|k| probabilities.get(k).copied())
                .unwrap_or(0.0),
            Answer::Score { probabilities, .. } => probabilities.get(truth).copied().unwrap_or(0.0),
            Answer::Noul { .. } => 0.0,
        };
        (pred == truth, nll(p_true))
    }

    /// Paired validation outcomes: `(accuracy pairs, NLL pairs)`. The NLL pairs
    /// (baseline, champion) drive the calibration criterion (ADR-004 gate 2b).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub(super) fn paired_validation(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        base_opts: &EngineOptions,
        base_art: &Artifact,
        cand_opts: &EngineOptions,
        cand_art: &Artifact,
        model: &str,
    ) -> (Vec<(bool, bool)>, Vec<(f32, f32)>) {
        let mut acc = Vec::new();
        let mut nlls = Vec::new();
        for r in resolved.iter().filter(|r| r.split == Split::Validation) {
            match (compiled, index) {
                (Compiled::Class(cp), Some(idx)) => {
                    let Some(truth) = r.class else { continue };
                    let (bc, bn) =
                        self.class_predict_nll(cp, base_art, base_opts, &r.emb, model, idx, truth);
                    let (cc, cn) =
                        self.class_predict_nll(cp, cand_art, cand_opts, &r.emb, model, idx, truth);
                    acc.push((bc, cc));
                    nlls.push((bn, cn));
                }
                (Compiled::Noul { predicate }, _) => {
                    let Some(y) = r.y else { continue };
                    let (bc, bn) = noul_correct_nll(base_art, predicate, &r.emb, model, y);
                    let (cc, cn) = noul_correct_nll(cand_art, predicate, &r.emb, model, y);
                    acc.push((bc, cc));
                    nlls.push((bn, cn));
                }
                _ => {}
            }
        }
        (acc, nlls)
    }

    /// `(accuracy, n)` on one split under `opts`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn split_accuracy(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        opts: &EngineOptions,
        art: &Artifact,
        split: Split,
        model: &str,
    ) -> (f32, u32) {
        let (acc, n, _) = self.split_scored(compiled, index, resolved, opts, art, split, model);
        (acc, n)
    }

    /// Score the frozen test split — reachable ONLY with a [`TestToken`], which
    /// only a [`Campaign`] can mint (twice). Consumes the token by value, so the
    /// number of test scorings is bounded by the tokens the campaign hands out
    /// (ADR-004 gate 1: test scored only for baseline and final champion).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn score_test_split(
        &self,
        _token: TestToken,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        opts: &EngineOptions,
        art: &Artifact,
        model: &str,
    ) -> (f32, u32, f32) {
        self.split_scored_any(compiled, index, resolved, opts, art, Split::Test, model)
    }

    /// `(accuracy, n, ece)` on a NON-test split under `opts`. Refuses
    /// `Split::Test` so the only path to test scoring is [`score_test_split`].
    #[allow(clippy::too_many_arguments)]
    fn split_scored(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        opts: &EngineOptions,
        art: &Artifact,
        split: Split,
        model: &str,
    ) -> (f32, u32, f32) {
        assert!(
            split != Split::Test,
            "the test split is scored only through a TestToken (score_test_split)",
        );
        self.split_scored_any(compiled, index, resolved, opts, art, split, model)
    }

    /// The shared scorer for any split (token-agnostic). Private, called only by
    /// the two guarded wrappers above.
    #[allow(clippy::too_many_arguments)]
    fn split_scored_any(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        opts: &EngineOptions,
        art: &Artifact,
        split: Split,
        model: &str,
    ) -> (f32, u32, f32) {
        let mut correct = Vec::new();
        let mut conf = Vec::new();
        for r in resolved.iter().filter(|r| r.split == split) {
            match (compiled, index) {
                (Compiled::Class(cp), Some(idx)) => {
                    let Some(truth) = r.class else { continue };
                    let (pred, c) = self.class_predict(cp, art, opts, &r.emb, model, idx);
                    correct.push(pred == truth);
                    conf.push(c);
                }
                (Compiled::Noul { predicate }, _) => {
                    let Some(y) = r.y else { continue };
                    let ok = noul_correct(art, predicate, &r.emb, model, y);
                    correct.push(ok);
                    // Confidence is max(p, 1-p) from the noul answer.
                    if let Answer::Noul { noul, .. } =
                        fit::noul_answer(art, predicate, &r.emb, model)
                    {
                        conf.push(noul.max(1.0 - noul));
                    }
                }
                _ => {}
            }
        }
        let n = correct.len() as u32;
        let hits = correct.iter().filter(|&&c| c).count() as u32;
        (Metrics::ratio(hits, n), n, ece(&conf, &correct))
    }

    /// Baseline and champion metrics on one split, packaged as receipt Metrics.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pair_metrics(
        &self,
        compiled: &Compiled,
        index: &Option<BTreeMap<String, usize>>,
        resolved: &[Resolved],
        base_opts: &EngineOptions,
        base_art: &Artifact,
        champ_opts: &EngineOptions,
        champ_art: &Artifact,
        split: Split,
        model: &str,
    ) -> (Metrics, Metrics) {
        let (ba, bn, be) =
            self.split_scored(compiled, index, resolved, base_opts, base_art, split, model);
        let (ca, cn, ce) = self.split_scored(
            compiled, index, resolved, champ_opts, champ_art, split, model,
        );
        (
            Metrics {
                baseline_accuracy: ba,
                champion_accuracy: ba,
                n: bn,
                ece: Some(be),
                brier: None,
            },
            Metrics {
                baseline_accuracy: ba,
                champion_accuracy: ca,
                n: cn.max(bn),
                ece: Some(ce),
                brier: None,
            },
        )
    }
}
