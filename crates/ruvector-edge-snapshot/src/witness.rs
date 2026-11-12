//! Per-tenant witness chain over snapshot manifests, and the signing ports.
//!
//! Entry hash:
//! `sha256("rvedge-witness-v1\0" ‖ tenant_key ‖ seq ‖ prev ‖ manifest_root ‖
//! collection_uid ‖ shard ‖ epoch ‖ audit_head)`; the genesis `prev` is
//! `sha256("rvedge-witness-genesis-v1\0" ‖ tenant_key)`. The tenant key is in
//! every hash, so a chain cannot be replayed under another tenant. The chain
//! is only tamper-evident against a **trusted head** held by the tenant's
//! ledger; a chain that merely self-verifies proves nothing if the attacker
//! supplies all of it, so [`verify_chain`] always takes that head.

use crate::bytes::put_str16;
use crate::error::SnapshotError;
use crate::manifest::{signing_message, SealedManifest};
use crate::types::{sha256, validate_tenant_key};

const ENTRY_TAG: &[u8] = b"rvedge-witness-v1\0";
const GENESIS_TAG: &[u8] = b"rvedge-witness-genesis-v1\0";

/// One chain entry (stored in the tenant ledger, shipped with audit).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessEntry {
    /// Position (0-based, contiguous).
    pub seq: u64,
    /// Hash of the previous entry (or genesis).
    pub prev: [u8; 32],
    /// Root of the snapshot manifest this entry witnesses.
    pub manifest_root: [u8; 32],
    /// Collection uid of that snapshot.
    pub collection_uid: String,
    /// Shard of that snapshot.
    pub shard: u16,
    /// Epoch of that snapshot.
    pub epoch: u64,
    /// Audit-chain head recorded in that snapshot's manifest (with the
    /// caller's service it locates the R2 objects, see [`WitnessEntry::key_prefix`]).
    pub audit_head: [u8; 32],
    /// This entry's hash.
    pub hash: [u8; 32],
}

impl WitnessEntry {
    /// R2 key prefix of the witnessed snapshot. The restoring Worker holds
    /// the entry (ledger) and the caller's [`crate::ShardRef`] (tenant,
    /// service), so it can locate `manifest.rvf` without listing R2. Pass
    /// only validated identifiers (a `ShardRef`'s); the entry's own fields
    /// come from a validated manifest.
    pub fn key_prefix(&self, tenant_key: &str, service: &str) -> String {
        crate::manifest::key_prefix(
            tenant_key,
            service,
            &self.collection_uid,
            self.shard,
            self.epoch,
            &self.audit_head,
        )
    }

    /// R2 key of the witnessed snapshot's manifest object.
    pub fn manifest_key(&self, tenant_key: &str, service: &str) -> String {
        format!("{}/manifest.rvf", self.key_prefix(tenant_key, service))
    }
}

/// Genesis `prev` for a tenant.
pub fn genesis(tenant_key: &str) -> [u8; 32] {
    sha256(&[GENESIS_TAG, tenant_key.as_bytes()])
}

fn entry_hash(tenant_key: &str, e: &WitnessEntry) -> [u8; 32] {
    let mut buf = Vec::with_capacity(160);
    put_str16(&mut buf, tenant_key);
    buf.extend_from_slice(&e.seq.to_le_bytes());
    buf.extend_from_slice(&e.prev);
    buf.extend_from_slice(&e.manifest_root);
    put_str16(&mut buf, &e.collection_uid);
    buf.extend_from_slice(&e.shard.to_le_bytes());
    buf.extend_from_slice(&e.epoch.to_le_bytes());
    buf.extend_from_slice(&e.audit_head);
    sha256(&[ENTRY_TAG, &buf])
}

/// Append-only chain state for one tenant (head + next seq).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessChain {
    tenant_key: String,
    head: [u8; 32],
    next_seq: u64,
}

impl WitnessChain {
    /// A fresh chain at genesis.
    pub fn new(tenant_key: &str) -> Result<Self, SnapshotError> {
        validate_tenant_key(tenant_key)?;
        Ok(WitnessChain {
            tenant_key: tenant_key.to_string(),
            head: genesis(tenant_key),
            next_seq: 0,
        })
    }

    /// Resume from persisted `(head, next_seq)`.
    pub fn resume(tenant_key: &str, head: [u8; 32], next_seq: u64) -> Result<Self, SnapshotError> {
        validate_tenant_key(tenant_key)?;
        Ok(WitnessChain {
            tenant_key: tenant_key.to_string(),
            head,
            next_seq,
        })
    }

    /// Current head (persist it in the tenant ledger after every append).
    pub fn head(&self) -> [u8; 32] {
        self.head
    }

    /// Next sequence number.
    pub fn next_seq(&self) -> u64 {
        self.next_seq
    }

    /// Witness a sealed manifest at the current head. Refuses a manifest of
    /// another tenant or one whose root does not verify. A manifest does not
    /// name the head it was started against, so concurrent snapshots of one
    /// tenant (several shards / collections, hourly alarms) each append in
    /// whatever order they finish; the entry's `prev` fixes its position.
    pub fn append(&mut self, sealed: &SealedManifest) -> Result<WitnessEntry, SnapshotError> {
        let m = &sealed.manifest;
        if m.tenant_key != self.tenant_key {
            return Err(SnapshotError::TenantMismatch);
        }
        if !sealed.root_matches() {
            return Err(SnapshotError::ManifestChecksum);
        }
        let mut e = WitnessEntry {
            seq: self.next_seq,
            prev: self.head,
            manifest_root: sealed.root,
            collection_uid: m.collection_uid.clone(),
            shard: m.shard,
            epoch: m.epoch,
            audit_head: m.audit_head,
            hash: [0; 32],
        };
        e.hash = entry_hash(&self.tenant_key, &e);
        self.head = e.hash;
        self.next_seq += 1;
        Ok(e)
    }
}

/// Verify `entries` from genesis to `trusted_head`.
pub fn verify_chain(
    tenant_key: &str,
    entries: &[WitnessEntry],
    trusted_head: &[u8; 32],
) -> Result<(), SnapshotError> {
    verify_chain_from(tenant_key, 0, &genesis(tenant_key), entries, trusted_head)
}

/// Verify a chain suffix starting at a trusted checkpoint `(start_seq, start_prev)`.
pub fn verify_chain_from(
    tenant_key: &str,
    start_seq: u64,
    start_prev: &[u8; 32],
    entries: &[WitnessEntry],
    trusted_head: &[u8; 32],
) -> Result<(), SnapshotError> {
    validate_tenant_key(tenant_key)?;
    let mut prev = *start_prev;
    for (i, e) in entries.iter().enumerate() {
        let seq = start_seq + i as u64;
        if e.seq != seq || e.prev != prev || entry_hash(tenant_key, e) != e.hash {
            return Err(SnapshotError::ChainBreak { seq });
        }
        prev = e.hash;
    }
    if &prev != trusted_head {
        return Err(SnapshotError::ChainHeadMismatch);
    }
    Ok(())
}

/// Signing port. The Worker implements it over a secret binding; no key
/// material lives in this crate.
pub trait Signer {
    /// Key identifier recorded next to the signature.
    fn key_id(&self) -> &str;
    /// Sign `msg` (Ed25519, 64-byte signature).
    fn sign(&self, msg: &[u8]) -> Result<[u8; 64], SnapshotError>;
}

/// Verification port. `tenant_key` is the tenant the signed manifest (or
/// export witness) names. `key_id` is attacker-chosen, so an implementation
/// must accept only keys authorised for that tenant or service-wide keys.
pub trait SignatureVerifier {
    /// `true` iff `sig` is a valid signature of `msg` under `key_id` **and**
    /// that key may sign for `tenant_key`.
    fn verify(&self, tenant_key: &str, key_id: &str, msg: &[u8], sig: &[u8; 64]) -> bool;
}

/// A registered verification key: `(tenant scope, key_id, key)`.
type ScopedKey = (Option<String>, String, ed25519_dalek::VerifyingKey);

/// Ed25519 verifier over a set of public keys (verify-only; no RNG, wasm clean).
#[derive(Debug, Default, Clone)]
pub struct Ed25519Verifier {
    keys: Vec<ScopedKey>,
}

impl Ed25519Verifier {
    /// Empty key set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a **service-wide** key (the service's own snapshot signing
    /// key), valid for manifests of every tenant.
    pub fn add_key(&mut self, key_id: &str, public: [u8; 32]) -> Result<(), SnapshotError> {
        self.insert(None, key_id, public)
    }

    /// Register a key that is valid only for manifests naming `tenant_key`.
    pub fn add_tenant_key(
        &mut self,
        tenant_key: &str,
        key_id: &str,
        public: [u8; 32],
    ) -> Result<(), SnapshotError> {
        validate_tenant_key(tenant_key)?;
        self.insert(Some(tenant_key.to_string()), key_id, public)
    }

    fn insert(
        &mut self,
        scope: Option<String>,
        key_id: &str,
        public: [u8; 32],
    ) -> Result<(), SnapshotError> {
        let vk = ed25519_dalek::VerifyingKey::from_bytes(&public)
            .map_err(|_| SnapshotError::InvalidIdentifier("public key"))?;
        self.keys.push((scope, key_id.to_string(), vk));
        Ok(())
    }
}

impl SignatureVerifier for Ed25519Verifier {
    fn verify(&self, tenant_key: &str, key_id: &str, msg: &[u8], sig: &[u8; 64]) -> bool {
        let sig = ed25519_dalek::Signature::from_bytes(sig);
        self.keys
            .iter()
            .filter(|(scope, k, _)| k == key_id && scope.as_deref().is_none_or(|t| t == tenant_key))
            .any(|(_, _, vk)| vk.verify_strict(msg, &sig).is_ok())
    }
}

/// Check the signature on a sealed manifest, under a key authorised for the
/// tenant the manifest names.
pub fn verify_signature(
    sealed: &SealedManifest,
    verifier: &dyn SignatureVerifier,
) -> Result<(), SnapshotError> {
    let s = sealed
        .signature
        .as_ref()
        .ok_or(SnapshotError::SignatureMissing)?;
    let msg = signing_message(&sealed.root);
    if verifier.verify(&sealed.manifest.tenant_key, &s.key_id, &msg, &s.signature) {
        Ok(())
    } else {
        Err(SnapshotError::SignatureInvalid)
    }
}
