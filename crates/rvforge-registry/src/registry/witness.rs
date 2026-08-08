//! Witness receipts and the per-subject hash chain.
//!
//! Every registry operation that changes what a reader would conclude —
//! publish, revoke, verify — appends a receipt to its subject's chain. A chain
//! is verifiable on its own: receipt *n* names receipt *n-1*, the first names
//! nothing, and each receipt's id is recomputable from its own content, so an
//! edited or removed link cannot be hidden.

use super::{Registry, WITNESS_DIR};
use crate::error::{RegistryError, Result};
use crate::objects::{
    compute_id, digest_hex, Actor, Outcome, Signature, SignatureRole, WitnessEvent,
    WitnessEvidence, WitnessReceipt, DIGEST_PREFIX, SCHEMA_VERSION,
};
use rvf_forge_core::canonical::hex_lower;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// What to record in a witness receipt.
#[derive(Clone, Debug)]
pub struct ReceiptRequest<'a> {
    /// The object the receipt is about. Need not be a content address — a
    /// `keyId` is a legitimate subject for a key revocation.
    pub subject: &'a str,
    /// What happened.
    pub event: WitnessEvent,
    /// How it came out.
    pub outcome: Outcome,
    /// Who performed it.
    pub actor: Actor,
    /// Detail for the witness viewer.
    pub details: String,
    /// RFC 3339 instant, supplied by the caller.
    pub timestamp: &'a str,
}

impl Registry {
    /// Record a verification against a subject, extending its witness chain.
    ///
    /// # Errors
    ///
    /// As [`Registry::emit_receipt`].
    pub fn record_verification(
        &self,
        subject: &str,
        outcome: Outcome,
        actor: Actor,
        details: impl Into<String>,
        at: &str,
    ) -> Result<WitnessReceipt> {
        self.emit_receipt(&ReceiptRequest {
            subject,
            event: WitnessEvent::Verify,
            outcome,
            actor,
            details: details.into(),
            timestamp: at,
        })
    }

    /// Build, sign if a signer is configured, store, and chain a receipt.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] on write failure, otherwise as
    /// [`crate::store::ObjectStore::put`].
    pub fn emit_receipt(&self, request: &ReceiptRequest<'_>) -> Result<WitnessReceipt> {
        let previous = self.last_receipt_id(request.subject)?;
        let mut receipt = WitnessReceipt {
            schema_version: SCHEMA_VERSION,
            object_type: crate::objects::TYPE_WITNESS_RECEIPT.into(),
            receipt_id: String::new(),
            subject: request.subject.to_string(),
            event: request.event,
            outcome: request.outcome,
            actor: request.actor.clone(),
            evidence: WitnessEvidence {
                details: request.details.clone(),
            },
            timestamp: request.timestamp.to_string(),
            prev_receipt: previous,
            signatures: Vec::new(),
        };

        // The id excludes signatures, so it can be computed before signing and
        // is unchanged by the signature that is then attached.
        let receipt_id = compute_id(&receipt)?;
        if let Some(signer) = &self.receipt_signer {
            receipt.signatures.push(Signature {
                key_id: signer.key_id(),
                role: SignatureRole::Registry,
                sig: crate::crypto::b64_encode(&signer.sign_bytes(receipt_id.as_bytes())),
            });
        }

        let stored = self.store().put(receipt)?;
        self.append_witness_index(request.subject, &stored.receipt_id)?;
        Ok(stored)
    }

    /// Every receipt for a subject, oldest first.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Io`] on read failure, otherwise as
    /// [`crate::store::ObjectStore::get`].
    pub fn receipts_for(&self, subject: &str) -> Result<Vec<WitnessReceipt>> {
        self.receipt_ids(subject)?
            .iter()
            .map(|id| self.store().get::<WitnessReceipt>(id))
            .collect()
    }

    /// Verify a subject's witness chain: recomputable ids, a `null` head, and
    /// each link naming its immediate predecessor.
    ///
    /// # Errors
    ///
    /// [`RegistryError::Log`] naming the first broken link, otherwise as
    /// [`Registry::receipts_for`].
    pub fn verify_witness_chain(&self, subject: &str) -> Result<()> {
        let receipts = self.receipts_for(subject)?;
        let mut expected_prev: Option<String> = None;
        for (i, receipt) in receipts.iter().enumerate() {
            let recomputed = compute_id(receipt)?;
            if recomputed != receipt.receipt_id {
                return Err(RegistryError::Log {
                    detail: format!(
                        "receipt {i} for {subject} hashes to {recomputed} but claims {}",
                        receipt.receipt_id
                    ),
                });
            }
            if receipt.prev_receipt != expected_prev {
                return Err(RegistryError::Log {
                    detail: format!(
                        "receipt {i} for {subject} names prevReceipt {:?}, expected {:?}",
                        receipt.prev_receipt, expected_prev
                    ),
                });
            }
            expected_prev = Some(receipt.receipt_id.clone());
        }
        Ok(())
    }

    fn witness_index_path(&self, subject: &str) -> PathBuf {
        // A subject may be a `sha256:` id or a `keyId`, so it is hashed rather
        // than used directly: one naming rule, no escaping, no collisions with
        // path separators.
        let key = hex_lower(&crate::merkle::sha256(subject.as_bytes()));
        self.root().join(WITNESS_DIR).join(format!("{key}.jsonl"))
    }

    fn receipt_ids(&self, subject: &str) -> Result<Vec<String>> {
        let path = self.witness_index_path(subject);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let text = fs::read_to_string(&path).map_err(|e| RegistryError::io(&path, &e))?;
        let mut ids = Vec::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            digest_hex("receipt id", line)?;
            ids.push(line.to_string());
        }
        Ok(ids)
    }

    fn last_receipt_id(&self, subject: &str) -> Result<Option<String>> {
        Ok(self.receipt_ids(subject)?.pop())
    }

    fn append_witness_index(&self, subject: &str, receipt_id: &str) -> Result<()> {
        debug_assert!(receipt_id.starts_with(DIGEST_PREFIX));
        let path = self.witness_index_path(subject);
        let parent = path
            .parent()
            .ok_or_else(|| RegistryError::invalid("witness index path has no parent"))?;
        fs::create_dir_all(parent).map_err(|e| RegistryError::io(parent, &e))?;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| RegistryError::io(&path, &e))?;
        writeln!(file, "{receipt_id}").map_err(|e| RegistryError::io(&path, &e))
    }
}
