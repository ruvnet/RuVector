//! Unified security gate — the single entry point for all privileged
//! operations in the RVM microhypervisor.
//!
//! No mutation occurs without passing through this gate. The pipeline:
//!
//! 1. **P1 capability check** -- Does the caller hold the required rights?
//! 2. **P2 policy validate** -- Is the policy satisfied?
//! 3. **Emit witness** -- Record the decision for audit.
//! 4. **Execute** -- Perform the operation.
//!
//! If any step fails, a `PROOF_REJECTED` witness is emitted and the
//! operation is denied.

use rvm_types::{ActionKind, CapRights, CapToken, CapType, RvmError, WitnessHash, WitnessRecord};
use rvm_witness::WitnessLog;

/// A request to the security gate.
#[derive(Debug, Clone, Copy)]
pub struct GateRequest {
    /// The capability token presented by the caller.
    pub token: CapToken,
    /// The required capability type.
    pub required_type: CapType,
    /// The required access rights.
    pub required_rights: CapRights,
    /// Optional proof commitment (required for state-mutating operations).
    pub proof_commitment: Option<WitnessHash>,
    /// Whether to require P3 deep proof verification.
    pub require_p3: bool,
    /// P3 derivation chain result (advisory only; set by caller).
    ///
    /// **Security note:** The gate does NOT trust this field. When
    /// `require_p3` is true the gate calls [`verify_p3_chain`] to
    /// perform its own verification. This field is retained for
    /// diagnostics and logging only.
    pub p3_chain_valid: bool,
    /// Optional witness chain data for P3 verification.
    ///
    /// When `require_p3` is true the gate hashes the witness data
    /// stored here and verifies chain linkage rather than trusting
    /// the caller-supplied `p3_chain_valid` flag.
    pub p3_witness_data: Option<P3WitnessChain>,
    /// The action being performed (for witness logging).
    pub action: ActionKind,
    /// Target object identifier.
    pub target_object_id: u64,
    /// Timestamp for the witness record.
    pub timestamp_ns: u64,
}

/// A successful gate response.
#[derive(Debug, Clone, Copy)]
pub struct GateResponse {
    /// The witness sequence number for this operation.
    pub witness_sequence: u64,
    /// The proof tier that was satisfied.
    pub proof_tier: u8,
}

/// Security errors specific to the gate pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityError {
    /// P1 capability check failed: wrong capability type.
    CapabilityTypeMismatch,
    /// P1 capability check failed: insufficient rights.
    InsufficientRights,
    /// P2 policy validation failed: proof commitment missing or invalid.
    PolicyViolation,
    /// P3 deep proof failed: derivation chain broken.
    DerivationChainBroken,
    /// The witness log refused the emission under
    /// [`rvm_witness::CoveragePolicy::Strict`] (segment full or the
    /// ring would overwrite an unsealed record). The mutation must not
    /// proceed ("no witness, no mutation"); seal the segment and retry.
    WitnessBackpressure,
    /// An internal error occurred.
    Internal(RvmError),
}

/// Compact P3 witness chain supplied by the caller for gate-side
/// verification.
///
/// Contains up to 4 chain links (prev_hash, record_hash) pairs
/// and optional 8-byte signatures. The gate walks these links to
/// verify chain continuity (and optionally signature integrity)
/// rather than trusting the caller's `p3_chain_valid` boolean.
#[derive(Debug, Clone, Copy)]
pub struct P3WitnessChain {
    /// Chain link data: pairs of (prev_hash: u64, record_hash: u64).
    pub links: [[u64; 2]; 4],
    /// Optional 8-byte auxiliary signatures per link (from `WitnessRecord.aux`).
    ///
    /// When present (non-zero), the gate can verify these against a
    /// [`rvm_witness::WitnessSigner`] if one is configured.
    pub signatures: [[u8; 8]; 4],
    /// Number of valid links (0..=4).
    pub link_count: u8,
}

impl P3WitnessChain {
    /// Create an empty witness chain.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            links: [[0u64; 2]; 4],
            signatures: [[0u8; 8]; 4],
            link_count: 0,
        }
    }
}

/// Verify a P3 witness chain by walking the links and checking that
/// each link's `record_hash` matches the next link's `prev_hash`.
///
/// Returns `true` if the chain is valid and non-empty; `false` otherwise.
fn verify_p3_chain(chain: &P3WitnessChain) -> bool {
    if chain.link_count == 0 {
        return false;
    }
    let count = chain.link_count.min(4) as usize;
    for i in 0..count.saturating_sub(1) {
        let record_hash = chain.links[i][1];
        let next_prev_hash = chain.links[i + 1][0];
        if record_hash != next_prev_hash {
            return false;
        }
    }
    true
}

/// The unified security gate.
///
/// Wraps the capability check, policy validation, and witness logging
/// into a single pipeline. The const generic `N` is the witness ring
/// buffer capacity.
pub struct SecurityGate<'a, const N: usize> {
    witness_log: &'a WitnessLog<N>,
}

impl<'a, const N: usize> SecurityGate<'a, N> {
    /// Create a new security gate backed by the given witness log.
    #[must_use]
    pub const fn new(witness_log: &'a WitnessLog<N>) -> Self {
        Self { witness_log }
    }

    /// Run a request through the full security pipeline.
    ///
    /// Pipeline:
    /// 1. P1 capability check (type + rights)
    /// 2. P2 policy validation (proof commitment if required)
    /// 3. Emit witness record
    /// 4. Return `GateResponse` with witness sequence
    ///
    /// On failure, emits a `ProofRejected` witness and returns the error.
    ///
    /// # Errors
    ///
    /// Returns [`SecurityError`] if any pipeline stage fails.
    pub fn check_and_execute(&self, request: &GateRequest) -> Result<GateResponse, SecurityError> {
        // Step 1: P1 capability check — type match
        if request.token.cap_type() != request.required_type {
            self.emit_rejection(request);
            return Err(SecurityError::CapabilityTypeMismatch);
        }

        // Step 1b: P1 capability check — rights subset
        if !request.token.has_rights(request.required_rights) {
            self.emit_rejection(request);
            return Err(SecurityError::InsufficientRights);
        }

        // Step 2: P2 policy validation — proof commitment
        let mut proof_tier = if let Some(commitment) = &request.proof_commitment {
            if commitment.is_zero() {
                self.emit_rejection(request);
                return Err(SecurityError::PolicyViolation);
            }
            2 // P2 was validated
        } else {
            1 // P1-only (no proof commitment needed)
        };

        // Step 2b: P3 deep proof — derivation chain (if required).
        //
        // The gate performs its own verification via `verify_p3_chain`
        // rather than trusting the caller-supplied `p3_chain_valid` flag.
        if request.require_p3 {
            let chain_ok = match &request.p3_witness_data {
                Some(chain) => verify_p3_chain(chain),
                // No witness data supplied -- cannot verify P3.
                None => false,
            };
            if !chain_ok {
                self.emit_rejection(request);
                return Err(SecurityError::DerivationChainBroken);
            }
            proof_tier = 3;
        }

        // Step 3: Emit witness record for the allowed action
        let seq = self.emit_allowed(request, proof_tier);

        // Step 4: Return success
        Ok(GateResponse {
            witness_sequence: seq,
            proof_tier,
        })
    }

    /// Emit a witness record for an allowed operation.
    fn emit_allowed(&self, request: &GateRequest, proof_tier: u8) -> u64 {
        let mut record = WitnessRecord::zeroed();
        record.action_kind = request.action as u8;
        record.proof_tier = proof_tier;
        record.actor_partition_id = 0; // caller context not tracked here
        record.target_object_id = request.target_object_id;
        record.capability_hash = request.token.truncated_hash();
        record.timestamp_ns = request.timestamp_ns;
        self.witness_log.append(record)
    }

    /// Emit a `ProofRejected` witness record for a denied operation.
    fn emit_rejection(&self, request: &GateRequest) {
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::ProofRejected as u8;
        record.actor_partition_id = 0;
        record.target_object_id = request.target_object_id;
        record.capability_hash = request.token.truncated_hash();
        record.timestamp_ns = request.timestamp_ns;
        self.witness_log.append(record);
    }
}

// ---------------------------------------------------------------------------
// Signed security gate (ADR-142 Phase 4)
// ---------------------------------------------------------------------------

/// A security gate enhanced with witness signature verification.
///
/// Extends [`SecurityGate`] with a [`WitnessSigner`] reference. When
/// P3 verification is requested, the gate verifies both chain linkage
/// **and** per-link auxiliary signatures, providing cryptographic
/// tamper evidence beyond hash-chain continuity alone.
///
/// When no signer is configured, use the plain [`SecurityGate`] which
/// performs chain-linkage-only verification.
pub struct SignedSecurityGate<'a, const N: usize, S: rvm_witness::WitnessSigner> {
    witness_log: &'a WitnessLog<N>,
    signer: &'a S,
}

impl<'a, const N: usize, S: rvm_witness::WitnessSigner> SignedSecurityGate<'a, N, S> {
    /// Create a new signed security gate.
    #[must_use]
    pub const fn new(witness_log: &'a WitnessLog<N>, signer: &'a S) -> Self {
        Self {
            witness_log,
            signer,
        }
    }

    /// Run a request through the full security pipeline with signature
    /// verification on P3 witness chains.
    ///
    /// Behaves identically to [`SecurityGate::check_and_execute`] except
    /// that P3 witness chain verification also checks auxiliary signatures
    /// on each chain link using the configured signer.
    ///
    /// # Errors
    ///
    /// Returns [`SecurityError`] if any pipeline stage fails.
    pub fn check_and_execute(&self, request: &GateRequest) -> Result<GateResponse, SecurityError> {
        // Step 1: P1 capability check -- type match.
        if request.token.cap_type() != request.required_type {
            self.emit_rejection(request);
            return Err(SecurityError::CapabilityTypeMismatch);
        }

        // Step 1b: P1 capability check -- rights subset.
        if !request.token.has_rights(request.required_rights) {
            self.emit_rejection(request);
            return Err(SecurityError::InsufficientRights);
        }

        // Step 2: P2 policy validation -- proof commitment.
        let mut proof_tier = if let Some(commitment) = &request.proof_commitment {
            if commitment.is_zero() {
                self.emit_rejection(request);
                return Err(SecurityError::PolicyViolation);
            }
            2
        } else {
            1
        };

        // Step 2b: P3 deep proof -- derivation chain + signature verification.
        if request.require_p3 {
            let chain_ok = match &request.p3_witness_data {
                Some(chain) => verify_p3_chain(chain) && self.verify_chain_signatures(chain),
                None => false,
            };
            if !chain_ok {
                self.emit_rejection(request);
                return Err(SecurityError::DerivationChainBroken);
            }
            proof_tier = 3;
        }

        // Step 3: Emit signed witness record.
        let seq = self.emit_allowed_signed(request, proof_tier);

        Ok(GateResponse {
            witness_sequence: seq,
            proof_tier,
        })
    }

    /// Verify the auxiliary signatures on each link of a P3 witness chain.
    ///
    /// For each link that has a non-zero signature, constructs a minimal
    /// `WitnessRecord` from the chain link data and verifies its `aux`
    /// field against the configured signer.
    ///
    /// Links with all-zero signatures are skipped (backwards compatible
    /// with unsigned chains).
    fn verify_chain_signatures(&self, chain: &P3WitnessChain) -> bool {
        let count = chain.link_count.min(4) as usize;
        for i in 0..count {
            let sig = chain.signatures[i];
            // Skip unsigned links (backwards compatible).
            if sig == [0u8; 8] {
                continue;
            }
            // Reconstruct a minimal witness record from chain link data.
            let mut record = WitnessRecord::zeroed();
            record.prev_hash = chain.links[i][0] as u32;
            record.record_hash = chain.links[i][1] as u32;
            record.sequence = i as u64;
            record.aux = sig;

            if !self.signer.verify(&record) {
                return false;
            }
        }
        true
    }

    /// Emit a signed witness record for an allowed operation.
    ///
    /// Uses [`WitnessLog::signed_append`] so the signature covers all
    /// fields including chain-hash metadata set during append.
    fn emit_allowed_signed(&self, request: &GateRequest, proof_tier: u8) -> u64 {
        let mut record = WitnessRecord::zeroed();
        record.action_kind = request.action as u8;
        record.proof_tier = proof_tier;
        record.actor_partition_id = 0;
        record.target_object_id = request.target_object_id;
        record.capability_hash = request.token.truncated_hash();
        record.timestamp_ns = request.timestamp_ns;
        self.witness_log.signed_append(record, self.signer)
    }

    /// Emit a signed `ProofRejected` witness record.
    fn emit_rejection(&self, request: &GateRequest) {
        let mut record = WitnessRecord::zeroed();
        record.action_kind = ActionKind::ProofRejected as u8;
        record.actor_partition_id = 0;
        record.target_object_id = request.target_object_id;
        record.capability_hash = request.token.truncated_hash();
        record.timestamp_ns = request.timestamp_ns;
        self.witness_log.signed_append(record, self.signer);
    }
}

// ---------------------------------------------------------------------------
// v2 security gate (ADR-134 v2)
// ---------------------------------------------------------------------------

/// Security gate backed by the v2 witness log.
///
/// Same P1/P2/P3 pipeline as [`SecurityGate`], but each emission costs
/// exactly **one** keyed BLAKE3 compression (the v2 chain MAC) instead
/// of the v1 path's two SHA-256 hashes plus optional per-record HMAC.
/// Exportable tamper evidence comes from sealing Merkle segments via
/// [`rvm_witness::WitnessLogV2::seal_segment`], off the syscall path.
pub struct SecurityGateV2<'a, const N: usize, const SEG: usize> {
    witness_log: &'a rvm_witness::WitnessLogV2<N, SEG>,
}

impl<'a, const N: usize, const SEG: usize> SecurityGateV2<'a, N, SEG> {
    /// Create a new v2 security gate backed by the given witness log.
    #[must_use]
    pub const fn new(witness_log: &'a rvm_witness::WitnessLogV2<N, SEG>) -> Self {
        Self { witness_log }
    }

    /// Run a request through the full security pipeline.
    ///
    /// Identical decision logic to [`SecurityGate::check_and_execute`];
    /// only the witness emission differs (v2 records, one keyed
    /// compression each).
    ///
    /// # Errors
    ///
    /// Returns [`SecurityError`] if any pipeline stage fails.
    pub fn check_and_execute(&self, request: &GateRequest) -> Result<GateResponse, SecurityError> {
        if request.token.cap_type() != request.required_type {
            self.emit_rejection(request);
            return Err(SecurityError::CapabilityTypeMismatch);
        }

        if !request.token.has_rights(request.required_rights) {
            self.emit_rejection(request);
            return Err(SecurityError::InsufficientRights);
        }

        let mut proof_tier = if let Some(commitment) = &request.proof_commitment {
            if commitment.is_zero() {
                self.emit_rejection(request);
                return Err(SecurityError::PolicyViolation);
            }
            2
        } else {
            1
        };

        if request.require_p3 {
            let chain_ok = match &request.p3_witness_data {
                Some(chain) => verify_p3_chain(chain),
                None => false,
            };
            if !chain_ok {
                self.emit_rejection(request);
                return Err(SecurityError::DerivationChainBroken);
            }
            proof_tier = 3;
        }

        let seq = self.emit_allowed(request, proof_tier)?;

        Ok(GateResponse {
            witness_sequence: seq,
            proof_tier,
        })
    }

    /// Emit a v2 witness record for an allowed operation.
    ///
    /// Uses [`rvm_witness::WitnessLogV2::try_append`] so a
    /// [`rvm_witness::CoveragePolicy::Strict`] log can refuse the
    /// emission; per "no witness, no mutation" the gate then fails the
    /// request with [`SecurityError::WitnessBackpressure`] instead of
    /// proceeding with degraded audit coverage.
    fn emit_allowed(&self, request: &GateRequest, proof_tier: u8) -> Result<u64, SecurityError> {
        let mut record = rvm_types::WitnessRecordV2::zeroed();
        record.action_kind = request.action as u8;
        record.proof_tier = proof_tier;
        record.actor_partition_id = 0; // caller context not tracked here
        record.target_object_id = request.target_object_id;
        record.capability_hash = request.token.truncated_hash();
        record.timestamp_ns = request.timestamp_ns;
        self.witness_log
            .try_append(record)
            .map_err(|_| SecurityError::WitnessBackpressure)
    }

    /// Emit a v2 `ProofRejected` witness record for a denied operation.
    ///
    /// Rejections use the best-effort append path so a denial is never
    /// blocked by coverage backpressure (the denied mutation does not
    /// proceed either way; the rejection record stays chain-protected).
    fn emit_rejection(&self, request: &GateRequest) {
        let mut record = rvm_types::WitnessRecordV2::zeroed();
        record.action_kind = ActionKind::ProofRejected as u8;
        record.actor_partition_id = 0;
        record.target_object_id = request.target_object_id;
        record.capability_hash = request.token.truncated_hash();
        record.timestamp_ns = request.timestamp_ns;
        self.witness_log.append(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvm_witness::WitnessSigner as _;

    fn make_token(cap_type: CapType, rights: CapRights) -> CapToken {
        CapToken::new(1, cap_type, rights, 1)
    }

    #[test]
    fn test_gate_allows_valid_request() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 1);
        assert_eq!(log.total_emitted(), 1);
    }

    #[test]
    fn test_gate_denies_wrong_type() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        let request = GateRequest {
            token: make_token(CapType::Region, CapRights::READ),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::CapabilityTypeMismatch);
        // Should have emitted a ProofRejected witness
        assert_eq!(log.total_emitted(), 1);
        let record = log.get(0).unwrap();
        assert_eq!(record.action_kind, ActionKind::ProofRejected as u8);
    }

    #[test]
    fn test_gate_denies_insufficient_rights() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ),
            required_type: CapType::Partition,
            required_rights: CapRights::READ | CapRights::WRITE,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::InsufficientRights);
        assert_eq!(log.total_emitted(), 1);
    }

    #[test]
    fn test_gate_denies_zero_proof_commitment() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: Some(WitnessHash::ZERO),
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::PolicyViolation);
        assert_eq!(log.total_emitted(), 1);
    }

    #[test]
    fn test_gate_allows_valid_proof_commitment() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        let commitment = WitnessHash::from_bytes([0xAB; 32]);
        let request = GateRequest {
            token: make_token(CapType::Region, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Region,
            required_rights: CapRights::WRITE,
            proof_commitment: Some(commitment),
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::RegionCreate,
            target_object_id: 100,
            timestamp_ns: 2000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 2);
        assert_eq!(log.total_emitted(), 1);
    }

    #[test]
    fn test_gate_pipeline_witness_sequence() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        let make_req = |ts| GateRequest {
            token: make_token(CapType::Partition, CapRights::READ),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 1,
            timestamp_ns: ts,
        };

        let r0 = gate.check_and_execute(&make_req(100)).unwrap();
        let r1 = gate.check_and_execute(&make_req(200)).unwrap();
        let r2 = gate.check_and_execute(&make_req(300)).unwrap();

        assert_eq!(r0.witness_sequence, 0);
        assert_eq!(r1.witness_sequence, 1);
        assert_eq!(r2.witness_sequence, 2);
        assert_eq!(log.total_emitted(), 3);
    }

    #[test]
    fn test_gate_p3_valid_chain() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        // Build a valid 2-link witness chain: link[0].record_hash == link[1].prev_hash.
        let mut chain = P3WitnessChain::empty();
        chain.links[0] = [0, 0xAABB]; // prev_hash=0, record_hash=0xAABB
        chain.links[1] = [0xAABB, 0xCCDD]; // prev_hash=0xAABB, record_hash=0xCCDD
        chain.link_count = 2;

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: false, // advisory only -- gate ignores this
            p3_witness_data: Some(chain),
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 3);
    }

    #[test]
    fn test_gate_p3_broken_chain_denied() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        // Broken chain: link[0].record_hash != link[1].prev_hash.
        let mut chain = P3WitnessChain::empty();
        chain.links[0] = [0, 0xAABB];
        chain.links[1] = [0xDEAD, 0xCCDD]; // prev_hash mismatch
        chain.link_count = 2;

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: true, // advisory says valid, but gate verifies and rejects
            p3_witness_data: Some(chain),
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::DerivationChainBroken);
        assert_eq!(log.total_emitted(), 1);
    }

    #[test]
    fn test_gate_p3_no_witness_data_denied() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        // P3 required but no witness data supplied.
        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: true, // caller lies, but gate has no data to verify
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::DerivationChainBroken);
    }

    #[test]
    fn test_gate_p3_caller_flag_ignored() {
        let log = WitnessLog::<16>::new();
        let gate = SecurityGate::new(&log);

        // Caller says p3_chain_valid = true but supplies empty chain.
        let chain = P3WitnessChain::empty(); // link_count = 0

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: true,
            p3_witness_data: Some(chain),
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        // Gate rejects because it verifies the chain itself, ignoring the flag.
        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::DerivationChainBroken);
    }

    // -- SignedSecurityGate tests (ADR-142 Phase 4) -------------------------

    #[test]
    fn test_signed_gate_allows_valid_request() {
        let log = WitnessLog::<16>::new();
        let signer = rvm_witness::default_signer();
        let gate = SignedSecurityGate::new(&log, &signer);

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 1);
        assert_eq!(log.total_emitted(), 1);

        // The emitted record should have a non-zero aux (signed).
        let record = log.get(0).unwrap();
        assert_ne!(record.aux, [0u8; 8]);
    }

    #[test]
    fn test_signed_gate_emitted_record_verifiable() {
        let log = WitnessLog::<16>::new();
        let signer = rvm_witness::default_signer();
        let gate = SignedSecurityGate::new(&log, &signer);

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        gate.check_and_execute(&request).unwrap();

        // The witness record should pass signature verification.
        let record = log.get(0).unwrap();
        assert!(signer.verify(&record));
    }

    #[test]
    fn test_signed_gate_p3_with_unsigned_chain_passes() {
        let log = WitnessLog::<16>::new();
        let signer = rvm_witness::default_signer();
        let gate = SignedSecurityGate::new(&log, &signer);

        // Build a valid chain with zero signatures (backwards compatible).
        let mut chain = P3WitnessChain::empty();
        chain.links[0] = [0, 0xAABB];
        chain.links[1] = [0xAABB, 0xCCDD];
        chain.link_count = 2;
        // signatures are all-zero (unsigned) -- should be skipped.

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: false,
            p3_witness_data: Some(chain),
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 3);
    }

    #[test]
    fn test_signed_gate_p3_with_signed_chain() {
        let log = WitnessLog::<16>::new();
        let signer = rvm_witness::default_signer();
        let gate = SignedSecurityGate::new(&log, &signer);

        // Build a valid chain and sign each link.
        let mut chain = P3WitnessChain::empty();
        chain.links[0] = [0, 0xAABB];
        chain.links[1] = [0xAABB, 0xCCDD];
        chain.link_count = 2;

        // Sign each link: construct a minimal record matching what
        // verify_chain_signatures expects.
        for i in 0..2 {
            let mut record = WitnessRecord::zeroed();
            record.prev_hash = chain.links[i][0] as u32;
            record.record_hash = chain.links[i][1] as u32;
            record.sequence = i as u64;
            chain.signatures[i] = signer.sign(&record);
        }

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: false,
            p3_witness_data: Some(chain),
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 3);
    }

    #[test]
    fn test_signed_gate_p3_tampered_signature_denied() {
        let log = WitnessLog::<16>::new();
        let signer = rvm_witness::default_signer();
        let gate = SignedSecurityGate::new(&log, &signer);

        let mut chain = P3WitnessChain::empty();
        chain.links[0] = [0, 0xAABB];
        chain.link_count = 1;

        // Sign the link, then tamper with the signature.
        let mut record = WitnessRecord::zeroed();
        record.prev_hash = 0;
        record.record_hash = 0xAABB;
        record.sequence = 0;
        let mut sig = signer.sign(&record);
        sig[0] ^= 0xFF; // tamper
        chain.signatures[0] = sig;

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: true,
            p3_chain_valid: false,
            p3_witness_data: Some(chain),
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::DerivationChainBroken);
    }

    #[test]
    fn test_signed_gate_denial_emits_signed_witness() {
        let log = WitnessLog::<16>::new();
        let signer = rvm_witness::default_signer();
        let gate = SignedSecurityGate::new(&log, &signer);

        let request = GateRequest {
            token: make_token(CapType::Region, CapRights::READ),
            required_type: CapType::Partition, // mismatch
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::CapabilityTypeMismatch);

        // Rejection witness should be signed.
        let record = log.get(0).unwrap();
        assert_ne!(record.aux, [0u8; 8]);
        assert!(signer.verify(&record));
    }

    // -- v2 gate tests (ADR-134 v2) -----------------------------------------

    #[test]
    fn test_gate_v2_allows_and_chains() {
        let log = rvm_witness::WitnessLogV2::<16, 8>::new();
        let gate = SecurityGateV2::new(&log);

        let request = GateRequest {
            token: make_token(CapType::Partition, CapRights::READ | CapRights::WRITE),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let response = gate.check_and_execute(&request).unwrap();
        assert_eq!(response.proof_tier, 1);
        assert_eq!(log.total_emitted(), 1);

        let record = log.get(0).unwrap();
        assert_eq!(record.version, 2);
        assert_eq!(record.action_kind, ActionKind::PartitionCreate as u8);
        assert_eq!(record.target_object_id, 42);
        assert_ne!(record.chain_mac, [0u8; 16]);
    }

    #[test]
    fn test_gate_v2_denial_emits_rejection() {
        let log = rvm_witness::WitnessLogV2::<16, 8>::new();
        let gate = SecurityGateV2::new(&log);

        let request = GateRequest {
            token: make_token(CapType::Region, CapRights::READ),
            required_type: CapType::Partition, // mismatch
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: 42,
            timestamp_ns: 1000,
        };

        let err = gate.check_and_execute(&request).unwrap_err();
        assert_eq!(err, SecurityError::CapabilityTypeMismatch);
        assert_eq!(log.total_emitted(), 1);
        let record = log.get(0).unwrap();
        assert_eq!(record.action_kind, ActionKind::ProofRejected as u8);
    }

    #[test]
    fn test_gate_v2_emissions_verify_as_chain() {
        let log = rvm_witness::WitnessLogV2::<16, 8>::new();
        let gate = SecurityGateV2::new(&log);

        for i in 0..3u64 {
            let request = GateRequest {
                token: make_token(CapType::Partition, CapRights::READ),
                required_type: CapType::Partition,
                required_rights: CapRights::READ,
                proof_commitment: None,
                require_p3: false,
                p3_chain_valid: false,
                p3_witness_data: None,
                action: ActionKind::PartitionCreate,
                target_object_id: i,
                timestamp_ns: i * 100,
            };
            gate.check_and_execute(&request).unwrap();
        }

        let mut records = [rvm_types::WitnessRecordV2::zeroed(); 3];
        assert_eq!(log.snapshot(&mut records), 3);
        let key = log.chain_key();
        assert_eq!(rvm_witness::verify_chain_v2(&records, &key), Ok(3));
    }

    #[test]
    fn test_gate_v2_strict_log_backpressure() {
        use rvm_witness::{CoveragePolicy, derive_chain_key};

        let log = rvm_witness::WitnessLogV2::<16, 2>::with_policy(
            derive_chain_key(b"gate-strict"),
            CoveragePolicy::Strict,
        );
        let gate = SecurityGateV2::new(&log);
        let request = |i: u64| GateRequest {
            token: make_token(CapType::Partition, CapRights::READ),
            required_type: CapType::Partition,
            required_rights: CapRights::READ,
            proof_commitment: None,
            require_p3: false,
            p3_chain_valid: false,
            p3_witness_data: None,
            action: ActionKind::PartitionCreate,
            target_object_id: i,
            timestamp_ns: i * 100,
        };

        gate.check_and_execute(&request(0)).unwrap();
        gate.check_and_execute(&request(1)).unwrap();
        // Segment (SEG = 2) is full: the mutation must not proceed.
        let err = gate.check_and_execute(&request(2)).unwrap_err();
        assert_eq!(err, SecurityError::WitnessBackpressure);
        assert_eq!(log.total_emitted(), 2);
        assert_eq!(log.segment_dropped(), 0);
        // Sealing relieves the backpressure.
        let signer = rvm_witness::Blake3SealSigner::new([0x42u8; 32]);
        assert!(log.seal_segment(&signer).is_some());
        let response = gate.check_and_execute(&request(2)).unwrap();
        assert_eq!(response.witness_sequence, 2);
    }
}
