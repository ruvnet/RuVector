use super::{invalid, validation};
use crate::{
    error::{Error, Result},
    types::*,
};
use serde::{
    ser::{SerializeMap, SerializeSeq},
    Serialize, Serializer,
};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

struct HashWriter(Sha256);
impl Write for HashWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.update(buffer);
        Ok(buffer.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(super) fn hash(value: &impl Serialize) -> Result<String> {
    let mut writer = HashWriter(Sha256::new());
    serde_json::to_writer(&mut writer, value).map_err(|e| invalid(e.to_string()))?;
    Ok(format!("{:x}", writer.0.finalize()))
}
#[derive(Serialize)]
struct HashableReceipt<'a> {
    version: u32,
    program_hash: &'a str,
    snapshot_hash: &'a str,
    snapshot_revision: &'a str,
    context_hash: &'a str,
    now: u64,
    work: u64,
    output: &'a Output,
}
/// SHA256 over the canonical receipt fields excluding `receipt_hash` itself.
/// This checksum supplies integrity only. It is not a signature or proof of truth.
pub fn hash_receipt(receipt: &Receipt) -> Result<String> {
    validation::serialized_size(receipt)?;
    hash(&HashableReceipt {
        version: receipt.version,
        program_hash: &receipt.program_hash,
        snapshot_hash: &receipt.snapshot_hash,
        snapshot_revision: &receipt.snapshot_revision,
        context_hash: &receipt.context_hash,
        now: receipt.now,
        work: receipt.work,
        output: &receipt.output,
    })
}
pub fn verify_receipt(receipt: &Receipt) -> Result<()> {
    if receipt.version != 1 || receipt.receipt_hash != hash_receipt(receipt)? {
        return Err(Error::new(
            "INTEGRITY_ERROR",
            "receipt checksum mismatch or unsupported version",
        ));
    }
    Ok(())
}
#[derive(Serialize)]
struct CanonicalRule<'a> {
    id: &'a str,
    requires: Vec<&'a Condition>,
    conclusion: &'a Condition,
}
#[derive(Serialize)]
struct CanonicalSnapshot<'a> {
    version: u32,
    revision: &'a str,
    nodes: Vec<CanonicalNode<'a>>,
    edges: Vec<&'a Edge>,
    observations: Vec<&'a Observation>,
    rules: Vec<CanonicalRule<'a>>,
}

#[derive(Serialize)]
struct CanonicalNode<'a> {
    id: &'a str,
    label: &'a str,
    properties: CanonicalProperties<'a>,
    vector: &'a Option<Vec<f32>>,
}

struct CanonicalProperties<'a>(&'a std::collections::BTreeMap<String, serde_json::Value>);
impl Serialize for CanonicalProperties<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (key, value) in self.0 {
            map.serialize_entry(key, &CanonicalValue(value))?;
        }
        map.end()
    }
}

/// Preserve array order while canonicalizing objects even when a host enables
/// serde_json/preserve_order. Borrow values rather than cloning the snapshot.
struct CanonicalValue<'a>(&'a serde_json::Value);
impl Serialize for CanonicalValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self.0 {
            serde_json::Value::Object(values) => {
                let mut entries: Vec<_> = values.iter().collect();
                entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, &CanonicalValue(value))?;
                }
                map.end()
            }
            serde_json::Value::Array(values) => {
                let mut sequence = serializer.serialize_seq(Some(values.len()))?;
                for value in values {
                    sequence.serialize_element(&CanonicalValue(value))?;
                }
                sequence.end()
            }
            value => value.serialize(serializer),
        }
    }
}

pub(super) fn snapshot_hash(snapshot: &Snapshot) -> Result<String> {
    let mut canonical = CanonicalSnapshot {
        version: snapshot.version,
        revision: &snapshot.revision,
        nodes: snapshot
            .nodes
            .iter()
            .map(|node| CanonicalNode {
                id: &node.id,
                label: &node.label,
                properties: CanonicalProperties(&node.properties),
                vector: &node.vector,
            })
            .collect(),
        edges: snapshot.edges.iter().collect(),
        observations: snapshot.observations.iter().collect(),
        rules: snapshot
            .rules
            .iter()
            .map(|r| {
                let mut requires = r.requires.iter().collect::<Vec<_>>();
                requires.sort();
                CanonicalRule {
                    id: &r.id,
                    requires,
                    conclusion: &r.conclusion,
                }
            })
            .collect(),
    };
    canonical.nodes.sort_by(|a, b| a.id.cmp(b.id));
    canonical.edges.sort_by(|a, b| a.id.cmp(&b.id));
    canonical.observations.sort_by(|a, b| a.id.cmp(&b.id));
    canonical.rules.sort_by(|a, b| a.id.cmp(b.id));
    hash(&canonical)
}
