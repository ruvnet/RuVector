use super::{invalid, Meter, MAX_BUDGET, MAX_BYTES, MAX_RECORDS, MAX_VECTOR};
use crate::{
    error::{Error, Result},
    types::*,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, Write},
};

pub(super) fn context(context: &Context) -> Result<()> {
    if context.max_steps == 0 || context.max_steps > MAX_BUDGET {
        return Err(invalid("max_steps must be in 1..=1000000000"));
    }
    if context.max_results == 0 || context.max_results > MAX_RECORDS {
        return Err(invalid("max_results must be in 1..=100000"));
    }
    Ok(())
}
fn text(value: &str, field: &str, meter: &mut Meter) -> Result<()> {
    if value.trim().is_empty() || value.len() > 4096 {
        return Err(invalid(format!(
            "{field} must contain 1..=4096 UTF-8 bytes"
        )));
    }
    meter.charge(value.len() / 64 + 1)
}
fn unique<'a>(
    seen: &mut BTreeSet<&'a str>,
    id: &'a str,
    field: &str,
    meter: &mut Meter,
) -> Result<()> {
    text(id, field, meter)?;
    if !seen.insert(id) {
        return Err(invalid(format!("duplicate {field}: {id}")));
    }
    Ok(())
}
pub(super) fn snapshot(snapshot: &Snapshot, meter: &mut Meter) -> Result<()> {
    if snapshot.version != 1 {
        return Err(invalid("unsupported snapshot version"));
    }
    let count = snapshot
        .nodes
        .len()
        .saturating_add(snapshot.edges.len())
        .saturating_add(snapshot.observations.len())
        .saturating_add(snapshot.rules.len());
    if count > MAX_RECORDS {
        return Err(invalid("snapshot exceeds 100000 records"));
    }
    text(&snapshot.revision, "snapshot revision", meter)?;
    let mut nodes = BTreeSet::new();
    let mut property_items = 0;
    for node in &snapshot.nodes {
        unique(&mut nodes, &node.id, "node id", meter)?;
        text(&node.label, "node label", meter)?;
        if let Some(vector) = &node.vector {
            if vector.is_empty()
                || vector.len() > MAX_VECTOR
                || vector.iter().any(|x| !x.is_finite())
            {
                return Err(invalid("node vectors must have 1..=4096 finite components"));
            }
            meter.charge(vector.len())?;
        }
        if node.properties.len() > MAX_RECORDS {
            return Err(invalid("too many properties"));
        }
        for (key, value) in &node.properties {
            text(key, "property key", meter)?;
            property(value, 0, &mut property_items, meter)?;
        }
    }
    let mut edges = BTreeSet::new();
    for edge in &snapshot.edges {
        unique(&mut edges, &edge.id, "edge id", meter)?;
        text(&edge.relation, "edge relation", meter)?;
        if !nodes.contains(edge.source.as_str()) || !nodes.contains(edge.target.as_str()) {
            return Err(invalid("edge references an unknown node"));
        }
    }
    let mut observations = BTreeSet::new();
    for observation in &snapshot.observations {
        unique(&mut observations, &observation.id, "observation id", meter)?;
        text(&observation.predicate, "observation predicate", meter)?;
        text(&observation.source, "observation source", meter)?;
        if !nodes.contains(observation.subject.as_str()) {
            return Err(invalid("observation references an unknown subject"));
        }
        if observation.expires_at <= observation.observed_at {
            return Err(invalid("observation expires_at must exceed observed_at"));
        }
    }
    let mut rules = BTreeSet::new();
    let mut condition_count = 0_usize;
    for rule in &snapshot.rules {
        unique(&mut rules, &rule.id, "rule id", meter)?;
        text(&rule.conclusion.predicate, "conclusion predicate", meter)?;
        if rule.requires.is_empty() || rule.requires.len() > 1024 {
            return Err(invalid("rules require 1..=1024 explicit premises"));
        }
        condition_count = condition_count.saturating_add(rule.requires.len());
        if condition_count > MAX_RECORDS {
            return Err(invalid("snapshot exceeds 100000 rule premises"));
        }
        let mut conditions = BTreeSet::new();
        for condition in &rule.requires {
            text(&condition.predicate, "premise predicate", meter)?;
            if !conditions.insert(condition) {
                return Err(invalid("duplicate rule premise"));
            }
        }
    }
    meter.charge(serialized_size(snapshot)? / 64 + 1)?;
    Ok(())
}
fn property(
    value: &serde_json::Value,
    depth: usize,
    count: &mut usize,
    meter: &mut Meter,
) -> Result<()> {
    *count += 1;
    if depth > 32 || *count > MAX_RECORDS {
        return Err(invalid("property nesting or item limit exceeded"));
    }
    meter.charge(1)?;
    match value {
        serde_json::Value::Array(items) => {
            if items.len() > MAX_RECORDS {
                return Err(invalid("property array too large"));
            }
            for item in items {
                property(item, depth + 1, count, meter)?;
            }
        }
        serde_json::Value::Object(items) => {
            if items.len() > MAX_RECORDS {
                return Err(invalid("property map too large"));
            }
            // Canonical receipt hashing sorts nested keys independently of the
            // host's serde_json map backend. Charge that work before hashing.
            meter.charge(
                items
                    .len()
                    .saturating_mul(super::graph::log_work(items.len())),
            )?;
            for (key, value) in items {
                text(key, "property key", meter)?;
                property(value, depth + 1, count, meter)?;
            }
        }
        serde_json::Value::String(value) => {
            if value.len() > 1024 * 1024 {
                return Err(invalid("property string too large"));
            }
            meter.charge(value.len() / 64 + 1)?;
        }
        _ => {}
    }
    Ok(())
}
pub(super) fn prediction(
    batch: &PredictionBatch,
    request: &PredictionRequest,
    limit: usize,
) -> Result<()> {
    let bad = |message| Error::new("PROVIDER_ERROR", message);
    if batch.model != request.model
        || batch.revision.trim().is_empty()
        || batch.revision.len() > 4096
        || batch.model_hash.len() != 64
        || !batch.model_hash.bytes().all(|c| c.is_ascii_hexdigit())
    {
        return Err(bad("model identity or hash is invalid"));
    }
    if batch.work > request.max_work || batch.work < batch.candidates.len() as u64 {
        return Err(bad("provider reported invalid work accounting"));
    }
    if batch.candidates.len() > limit {
        return Err(bad("provider exceeded result limit"));
    }
    let subjects: BTreeSet<_> = request.subjects.iter().collect();
    let mut counts = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for candidate in &batch.candidates {
        if !subjects.contains(&candidate.subject)
            || candidate.relation != request.relation
            || candidate.object.trim().is_empty()
            || candidate.object.len() > 4096
            || !candidate.score.is_finite()
        {
            return Err(bad("provider returned invalid prediction"));
        }
        if !seen.insert((&candidate.subject, &candidate.object)) {
            return Err(bad("provider returned duplicate prediction"));
        }
        let count = counts.entry(&candidate.subject).or_insert(0);
        *count += 1;
        if *count > request.k {
            return Err(bad("provider exceeded per-subject top count"));
        }
    }
    serialized_size(batch)?;
    Ok(())
}
struct SizeWriter(usize);
impl Write for SizeWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0 = self.0.saturating_add(buffer.len());
        if self.0 > MAX_BYTES {
            return Err(io::Error::other("serialized input exceeds 16 MiB"));
        }
        Ok(buffer.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(super) fn serialized_size(value: &impl Serialize) -> Result<usize> {
    let mut writer = SizeWriter(0);
    serde_json::to_writer(&mut writer, value).map_err(|e| invalid(e.to_string()))?;
    Ok(writer.0)
}
