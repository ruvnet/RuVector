use super::{bounded, evidence::EvidenceSet, invalid, Meter};
use crate::{error::Result, types::*};
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Graph<'a> {
    pub nodes: BTreeMap<&'a str, &'a Node>,
    edges: BTreeMap<(&'a str, &'a str), Vec<&'a str>>,
    observations: BTreeMap<(&'a str, &'a str), Vec<&'a Observation>>,
    pub rules: BTreeMap<&'a str, &'a Rule>,
}
impl<'a> Graph<'a> {
    pub fn new(snapshot: &'a Snapshot, meter: &mut Meter) -> Result<Self> {
        let mut graph = Self {
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            observations: BTreeMap::new(),
            rules: BTreeMap::new(),
        };
        for node in &snapshot.nodes {
            meter.charge(1)?;
            graph.nodes.insert(node.id.as_str(), node);
        }
        for edge in &snapshot.edges {
            meter.charge(1)?;
            graph
                .edges
                .entry((&edge.source, &edge.relation))
                .or_default()
                .push(&edge.target);
        }
        for targets in graph.edges.values_mut() {
            meter.charge(targets.len().saturating_mul(log_work(targets.len())))?;
            targets.sort_unstable();
            targets.dedup();
        }
        for observation in &snapshot.observations {
            meter.charge(1)?;
            graph
                .observations
                .entry((&observation.subject, &observation.predicate))
                .or_default()
                .push(observation);
        }
        for observations in graph.observations.values_mut() {
            meter.charge(
                observations
                    .len()
                    .saturating_mul(log_work(observations.len())),
            )?;
            observations.sort_by(|a, b| a.id.cmp(&b.id));
        }
        for rule in &snapshot.rules {
            meter.charge(1)?;
            graph.rules.insert(&rule.id, rule);
        }
        Ok(graph)
    }
    pub fn nodes(&self, label: &str, limit: usize, meter: &mut Meter) -> Result<Vec<&'a str>> {
        let mut result = Vec::new();
        for node in self.nodes.values() {
            meter.charge(1)?;
            if node.label == label {
                bounded(result.len() + 1, limit)?;
                result.push(node.id.as_str());
            }
        }
        Ok(result)
    }
    pub fn select(&self, ids: &[String], limit: usize, meter: &mut Meter) -> Result<Vec<&'a str>> {
        bounded(ids.len(), limit)?;
        let mut result = BTreeSet::new();
        for id in ids {
            meter.charge(1)?;
            let node = self
                .nodes
                .get(id.as_str())
                .ok_or_else(|| invalid(format!("unknown node {id}")))?;
            result.insert(node.id.as_str());
        }
        Ok(result.into_iter().collect())
    }
    /// Outgoing transitive neighborhood, including seed nodes (distance zero).
    pub fn traverse(
        &self,
        seeds: &[&'a str],
        relation: &str,
        depth: usize,
        limit: usize,
        meter: &mut Meter,
    ) -> Result<Vec<&'a str>> {
        bounded(seeds.len(), limit)?;
        meter.charge(seeds.len())?;
        let mut seen: BTreeSet<&str> = seeds.iter().copied().collect();
        let mut frontier = seen.clone();
        for _ in 0..depth {
            if frontier.is_empty() {
                break;
            }
            let mut next = BTreeSet::new();
            for source in frontier {
                meter.charge(1)?;
                if let Some(targets) = self.edges.get(&(source, relation)) {
                    for target in targets {
                        meter.charge(1)?;
                        if !seen.contains(*target) {
                            bounded(seen.len() + 1, limit)?;
                            seen.insert(*target);
                            next.insert(*target);
                        }
                    }
                }
            }
            frontier = next;
        }
        Ok(seen.into_iter().collect())
    }
    /// Exact cosine ranking; absent or zero norm vectors are not candidates.
    pub fn similar(
        &self,
        input: &[&'a str],
        vector: &[f32],
        k: usize,
        limit: usize,
        meter: &mut Meter,
    ) -> Result<Vec<&'a str>> {
        bounded(input.len(), limit)?;
        bounded(k, limit)?;
        meter.charge(vector.len())?;
        let norm: f64 = vector
            .iter()
            .map(|x| f64::from(*x).powi(2))
            .sum::<f64>()
            .sqrt();
        if norm == 0.0 {
            return Err(invalid("query vector must have nonzero norm"));
        }
        let mut scored = Vec::new();
        for id in input {
            meter.charge(1)?;
            let node = self.nodes.get(*id).ok_or_else(|| invalid("unknown node"))?;
            if let Some(candidate) = &node.vector {
                if candidate.len() != vector.len() {
                    return Err(invalid("vector dimension mismatch"));
                }
                meter.charge(candidate.len().saturating_mul(3))?;
                let mut dot = 0.0_f64;
                let mut length = 0.0_f64;
                for (left, right) in candidate.iter().zip(vector) {
                    dot += f64::from(*left) * f64::from(*right);
                    length += f64::from(*left).powi(2);
                }
                if length > 0.0 {
                    scored.push((dot / (length.sqrt() * norm), *id));
                }
            }
        }
        meter.charge(scored.len().saturating_mul(log_work(scored.len())))?;
        // Keep the historical logical sort charge so receipt hashes and budget
        // failures do not depend on this physical selection optimization.
        let compare = |a: &(f64, &str), b: &(f64, &str)| b.0.total_cmp(&a.0).then(a.1.cmp(b.1));
        if k < scored.len() {
            scored.select_nth_unstable_by(k, compare);
            scored.truncate(k);
        }
        scored.sort_unstable_by(compare);
        Ok(scored.into_iter().take(k).map(|(_, id)| id).collect())
    }
    pub fn observe(
        &self,
        input: &[&'a str],
        predicate: &str,
        within: u64,
        context: &Context,
        meter: &mut Meter,
    ) -> Result<EvidenceSet> {
        let mut result = EvidenceSet::default();
        for subject in input {
            meter.charge(1)?;
            result.add_domain(subject, predicate, context.max_results, meter)?;
            if let Some(observations) = self.observations.get(&(*subject, predicate)) {
                for observation in observations {
                    meter.charge(1)?;
                    if observation.observed_at <= context.now
                        && context.now < observation.expires_at
                        && context.now - observation.observed_at <= within
                    {
                        let witness = Evidence {
                            subject: (*subject).to_string(),
                            predicate: predicate.into(),
                            value: observation.value,
                            kind: EvidenceKind::Observed,
                            observation_ids: BTreeSet::from([observation.id.clone()]),
                            sources: BTreeSet::from([observation.source.clone()]),
                            rule_ids: BTreeSet::new(),
                            expires_at: observation.expires_at.min(
                                observation
                                    .observed_at
                                    .saturating_add(within)
                                    .saturating_add(1),
                            ),
                        };
                        result.insert(witness, context.max_results, meter)?;
                    }
                }
            }
        }
        Ok(result)
    }
}
pub(super) fn log_work(size: usize) -> usize {
    if size < 2 {
        1
    } else {
        usize::BITS as usize - (size - 1).leading_zeros() as usize
    }
}
