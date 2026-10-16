use super::{bounded, graph::Graph, invalid, Meter, MAX_BYTES};
use crate::{error::Result, types::*};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
};

type Fact = (String, String, bool);
#[derive(Default)]
pub(super) struct EvidenceSet {
    domain: BTreeSet<(String, String)>,
    facts: BTreeMap<Fact, Vec<Evidence>>,
    witnesses: usize,
    references: [usize; 3],
    bytes: usize,
}
fn order(a: &Evidence, b: &Evidence) -> Ordering {
    let kind = |e: &Evidence| {
        if e.kind == EvidenceKind::Observed {
            0
        } else {
            1
        }
    };
    kind(a)
        .cmp(&kind(b))
        .then(a.observation_ids.cmp(&b.observation_ids))
        .then(a.sources.cmp(&b.sources))
        .then(a.rule_ids.cmp(&b.rule_ids))
        .then(a.expires_at.cmp(&b.expires_at))
}
fn size(e: &Evidence) -> usize {
    e.subject.len()
        + e.predicate.len()
        + 128
        + e.observation_ids
            .iter()
            .chain(&e.sources)
            .chain(&e.rule_ids)
            .map(|s| s.len() + 32)
            .sum::<usize>()
}
impl EvidenceSet {
    pub fn retained_bytes(&self) -> usize {
        self.bytes
    }
    pub fn add_domain(
        &mut self,
        subject: &str,
        predicate: &str,
        limit: usize,
        meter: &mut Meter,
    ) -> Result<()> {
        meter.charge(1)?;
        let key = (subject.to_string(), predicate.to_string());
        if !self.domain.contains(&key) {
            bounded(self.domain.len() + 1, limit)?;
            let bytes = subject.len() + predicate.len() + 64;
            bounded(self.bytes + bytes, MAX_BYTES)?;
            self.bytes += bytes;
            self.domain.insert(key);
        }
        Ok(())
    }
    pub fn insert(&mut self, witness: Evidence, limit: usize, meter: &mut Meter) -> Result<()> {
        let key = (
            witness.subject.clone(),
            witness.predicate.clone(),
            witness.value,
        );
        let refs = [
            witness.observation_ids.len(),
            witness.sources.len(),
            witness.rule_ids.len(),
        ];
        meter.charge(1 + refs.iter().sum::<usize>())?;
        let index = self
            .facts
            .get(&key)
            .map(|v| v.binary_search_by(|e| order(e, &witness)));
        if matches!(index, Some(Ok(_))) {
            return Ok(());
        }
        bounded(self.witnesses + 1, limit)?;
        for (total, extra) in self.references.iter().zip(refs) {
            bounded(total + extra, limit)?;
        }
        bounded(self.bytes + size(&witness), MAX_BYTES)?;
        self.bytes += size(&witness);
        self.witnesses += 1;
        for (total, extra) in self.references.iter_mut().zip(refs) {
            *total += extra;
        }
        let entry = self.facts.entry(key).or_default();
        entry.insert(
            match index {
                Some(Err(i)) => i,
                _ => 0,
            },
            witness,
        );
        Ok(())
    }
    pub fn assess(&self, limit: usize, meter: &mut Meter) -> Result<Vec<Decision>> {
        bounded(self.domain.len(), limit)?;
        let mut decisions = Vec::new();
        for (subject, predicate) in &self.domain {
            meter.charge(1)?;
            let positive = self.facts.get(&(subject.clone(), predicate.clone(), true));
            let negative = self.facts.get(&(subject.clone(), predicate.clone(), false));
            let status = match (positive.is_some(), negative.is_some()) {
                (true, true) => Status::Conflicting,
                (true, false) => Status::Supported,
                (false, true) => Status::Refuted,
                (false, false) => Status::Unknown,
            };
            let mut evidence = Vec::new();
            for witnesses in [negative, positive].into_iter().flatten() {
                for witness in witnesses {
                    meter.charge(
                        1 + witness.observation_ids.len()
                            + witness.sources.len()
                            + witness.rule_ids.len(),
                    )?;
                    evidence.push(witness.clone());
                }
            }
            decisions.push(Decision {
                subject: subject.clone(),
                predicate: predicate.clone(),
                status,
                evidence,
            });
        }
        Ok(decisions)
    }
}

/// Monotone finite signed Horn closure with one canonical witness per newly derived fact.
/// False premises require explicit negative evidence; absence never satisfies a premise.
pub(super) fn infer(
    inputs: &[&EvidenceSet],
    rule_ids: &[String],
    graph: &Graph<'_>,
    context: &Context,
    meter: &mut Meter,
) -> Result<EvidenceSet> {
    let mut result = EvidenceSet::default();
    let mut subjects = BTreeSet::new();
    for input in inputs {
        for (subject, predicate) in &input.domain {
            subjects.insert(subject.clone());
            result.add_domain(subject, predicate, context.max_results, meter)?;
        }
        for witnesses in input.facts.values() {
            for witness in witnesses {
                result.insert(witness.clone(), context.max_results, meter)?;
            }
        }
    }
    let mut rules = BTreeMap::new();
    for id in rule_ids {
        meter.charge(1)?;
        let rule = graph
            .rules
            .get(id.as_str())
            .ok_or_else(|| invalid(format!("unknown rule {id}")))?;
        rules.insert(id, *rule);
    }
    for subject in &subjects {
        for rule in rules.values() {
            result.add_domain(
                subject,
                &rule.conclusion.predicate,
                context.max_results,
                meter,
            )?;
        }
    }
    loop {
        let mut changed = false;
        for subject in &subjects {
            for rule in rules.values() {
                meter.charge(1)?;
                let fact = (
                    subject.clone(),
                    rule.conclusion.predicate.clone(),
                    rule.conclusion.value,
                );
                if result.facts.contains_key(&fact) {
                    continue;
                }
                let mut premises = Vec::new();
                meter.charge(
                    rule.requires
                        .len()
                        .saturating_mul(super::graph::log_work(rule.requires.len())),
                )?;
                let mut conditions = rule.requires.iter().collect::<Vec<_>>();
                conditions.sort();
                for condition in conditions {
                    meter.charge(1)?;
                    match result.facts.get(&(
                        subject.clone(),
                        condition.predicate.clone(),
                        condition.value,
                    )) {
                        Some(witnesses) => premises.push(&witnesses[0]),
                        None => break,
                    }
                }
                if premises.len() != rule.requires.len() {
                    continue;
                }
                let mut witness = Evidence {
                    subject: subject.clone(),
                    predicate: rule.conclusion.predicate.clone(),
                    value: rule.conclusion.value,
                    kind: EvidenceKind::Derived,
                    observation_ids: BTreeSet::new(),
                    sources: BTreeSet::new(),
                    rule_ids: BTreeSet::from([rule.id.clone()]),
                    expires_at: u64::MAX,
                };
                for premise in premises {
                    meter.charge(1)?;
                    union(
                        &mut witness.observation_ids,
                        &premise.observation_ids,
                        context.max_results,
                        meter,
                    )?;
                    union(
                        &mut witness.sources,
                        &premise.sources,
                        context.max_results,
                        meter,
                    )?;
                    union(
                        &mut witness.rule_ids,
                        &premise.rule_ids,
                        context.max_results,
                        meter,
                    )?;
                    witness.expires_at = witness.expires_at.min(premise.expires_at);
                    bounded(size(&witness), MAX_BYTES)?;
                }
                result.insert(witness, context.max_results, meter)?;
                changed = true;
            }
        }
        if !changed {
            return Ok(result);
        }
    }
}
fn union(
    target: &mut BTreeSet<String>,
    source: &BTreeSet<String>,
    limit: usize,
    meter: &mut Meter,
) -> Result<()> {
    for value in source {
        meter.charge(1)?;
        if !target.contains(value) {
            bounded(target.len() + 1, limit)?;
            target.insert(value.clone());
        }
    }
    Ok(())
}
