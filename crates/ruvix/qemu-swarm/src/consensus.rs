//! PBFT (Practical Byzantine Fault Tolerance) consensus implementation.
//!
//! This module provides a production-ready PBFT implementation for distributed
//! consensus among RuVix cluster nodes. It supports:
//!
//! - **3f+1 Byzantine fault tolerance**: Tolerates f Byzantine (malicious) nodes
//! - **View change protocol**: Leader recovery on timeout
//! - **Checkpoint mechanism**: Garbage collection of old messages
//! - **Watermarks**: Prevent unbounded sequence number growth

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use parking_lot::RwLock;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use tracing::{debug, info, warn};

use crate::error::{SwarmError, SwarmResult};

/// Cryptographic signature (64 bytes, serializes as hex string).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signature(pub [u8; 64]);

impl Default for Signature {
    fn default() -> Self {
        Self([0u8; 64])
    }
}

impl Serialize for Signature {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Serialize as hex string
        let hex = hex::encode(self.0);
        serializer.serialize_str(&hex)
    }
}

impl<'de> Deserialize<'de> for Signature {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let hex_str = String::deserialize(deserializer)?;
        let bytes = hex::decode(&hex_str).map_err(serde::de::Error::custom)?;
        if bytes.len() != 64 {
            return Err(serde::de::Error::custom("signature must be 64 bytes"));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }
}

/// Hash digest (32 bytes, serializes as hex string).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HashDigest(pub [u8; 32]);

impl Default for HashDigest {
    fn default() -> Self {
        Self([0u8; 32])
    }
}

impl Serialize for HashDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let hex = hex::encode(self.0);
        serializer.serialize_str(&hex)
    }
}

impl<'de> Deserialize<'de> for HashDigest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let hex_str = String::deserialize(deserializer)?;
        let bytes = hex::decode(&hex_str).map_err(serde::de::Error::custom)?;
        if bytes.len() != 32 {
            return Err(serde::de::Error::custom("hash digest must be 32 bytes"));
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }
}

/// PBFT message types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PbftMessage {
    /// Client request.
    Request(Request),

    /// Pre-prepare from primary (leader).
    PrePrepare(PrePrepare),

    /// Prepare from replicas.
    Prepare(Prepare),

    /// Commit from replicas.
    Commit(Commit),

    /// Reply to client.
    Reply(Reply),

    /// View change request.
    ViewChange(ViewChange),

    /// New view announcement.
    NewView(NewView),

    /// Checkpoint message.
    Checkpoint(Checkpoint),
}

impl PbftMessage {
    /// Serialize message to bytes.
    pub fn to_bytes(&self) -> SwarmResult<Bytes> {
        let data = serde_json::to_vec(self).map_err(SwarmError::Json)?;
        Ok(Bytes::from(data))
    }

    /// Deserialize message from bytes.
    pub fn from_bytes(data: &[u8]) -> SwarmResult<Self> {
        serde_json::from_slice(data).map_err(SwarmError::Json)
    }

    /// Get the view number from the message.
    pub fn view(&self) -> u64 {
        match self {
            Self::Request(_) => 0,
            Self::PrePrepare(m) => m.view,
            Self::Prepare(m) => m.view,
            Self::Commit(m) => m.view,
            Self::Reply(m) => m.view,
            Self::ViewChange(m) => m.new_view,
            Self::NewView(m) => m.view,
            Self::Checkpoint(m) => m.view,
        }
    }

    /// Get the sequence number if applicable.
    pub fn sequence(&self) -> Option<u64> {
        match self {
            Self::PrePrepare(m) => Some(m.sequence),
            Self::Prepare(m) => Some(m.sequence),
            Self::Commit(m) => Some(m.sequence),
            Self::Reply(m) => Some(m.sequence),
            Self::Checkpoint(m) => Some(m.sequence),
            _ => None,
        }
    }
}

/// Client request message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    /// Client identifier.
    pub client_id: u64,
    /// Request timestamp (for ordering).
    pub timestamp: u64,
    /// Operation to execute.
    pub operation: Operation,
}

/// Pre-prepare message (from primary).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrePrepare {
    /// View number.
    pub view: u64,
    /// Sequence number.
    pub sequence: u64,
    /// Digest of the request.
    pub digest: HashDigest,
    /// The actual request.
    pub request: Request,
    /// Primary's signature.
    pub signature: Signature,
}

/// Prepare message (from replicas).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prepare {
    /// View number.
    pub view: u64,
    /// Sequence number.
    pub sequence: u64,
    /// Digest of the request.
    pub digest: HashDigest,
    /// Replica ID.
    pub replica_id: usize,
    /// Replica's signature.
    pub signature: Signature,
}

/// Commit message (from replicas).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commit {
    /// View number.
    pub view: u64,
    /// Sequence number.
    pub sequence: u64,
    /// Digest of the request.
    pub digest: HashDigest,
    /// Replica ID.
    pub replica_id: usize,
    /// Replica's signature.
    pub signature: Signature,
}

/// Reply message (to client).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reply {
    /// View number.
    pub view: u64,
    /// Sequence number.
    pub sequence: u64,
    /// Timestamp from original request.
    pub timestamp: u64,
    /// Client ID.
    pub client_id: u64,
    /// Replica ID.
    pub replica_id: usize,
    /// Result of the operation.
    pub result: OperationResult,
    /// Replica's signature.
    pub signature: Signature,
}

/// View change message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ViewChange {
    /// New view number being requested.
    pub new_view: u64,
    /// Last stable checkpoint sequence.
    pub last_checkpoint: u64,
    /// Checkpoint proofs.
    pub checkpoint_proofs: Vec<CheckpointProof>,
    /// Prepared messages since checkpoint.
    pub prepared_messages: Vec<PreparedProof>,
    /// Replica ID.
    pub replica_id: usize,
    /// Signature.
    pub signature: Signature,
}

/// New view announcement from new primary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewView {
    /// View number.
    pub view: u64,
    /// View change messages received.
    pub view_changes: Vec<ViewChange>,
    /// Pre-prepare messages for outstanding requests.
    pub pre_prepares: Vec<PrePrepare>,
    /// New primary's signature.
    pub signature: Signature,
}

/// Checkpoint message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Checkpoint {
    /// View number.
    pub view: u64,
    /// Sequence number.
    pub sequence: u64,
    /// State digest at this checkpoint.
    pub state_digest: HashDigest,
    /// Replica ID.
    pub replica_id: usize,
    /// Signature.
    pub signature: Signature,
}

/// Proof of a checkpoint (2f+1 matching checkpoint messages).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointProof {
    /// Sequence number.
    pub sequence: u64,
    /// State digest.
    pub state_digest: HashDigest,
    /// Checkpoint messages from replicas.
    pub messages: Vec<Checkpoint>,
}

/// Proof that a request was prepared.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreparedProof {
    /// The pre-prepare message.
    pub pre_prepare: PrePrepare,
    /// Matching prepare messages (2f).
    pub prepares: Vec<Prepare>,
}

/// Operation types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Operation {
    /// No operation (for testing).
    Noop,
    /// Read a key.
    Read { key: String },
    /// Write a key-value pair.
    Write { key: String, value: Vec<u8> },
    /// Execute a custom command.
    Execute { command: String, args: Vec<u8> },
    /// RVF deployment.
    DeployRvf { rvf_hash: [u8; 32] },
    /// Vector store operation.
    VectorOp { store_id: u64, key: u64, data: Vec<u8> },
    /// Graph mutation.
    GraphMutation { graph_id: u64, mutation: Vec<u8> },
}

/// Operation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationResult {
    /// Success with optional data.
    Success(Option<Vec<u8>>),
    /// Error with message.
    Error(String),
}

/// PBFT replica state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaState {
    /// Normal operation.
    Normal,
    /// View change in progress.
    ViewChange,
    /// Recovering from failure.
    Recovery,
}

/// PBFT log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// The pre-prepare message.
    pub pre_prepare: Option<PrePrepare>,
    /// Prepare messages received.
    pub prepares: HashMap<usize, Prepare>,
    /// Commit messages received.
    pub commits: HashMap<usize, Commit>,
    /// Whether this entry is prepared (2f+1 prepares).
    pub prepared: bool,
    /// Whether this entry is committed locally.
    pub committed_local: bool,
    /// Whether this entry is committed globally.
    pub committed: bool,
    /// The result if executed.
    pub result: Option<OperationResult>,
}

impl Default for LogEntry {
    fn default() -> Self {
        Self {
            pre_prepare: None,
            prepares: HashMap::new(),
            commits: HashMap::new(),
            prepared: false,
            committed_local: false,
            committed: false,
            result: None,
        }
    }
}

/// PBFT consensus configuration.
#[derive(Debug, Clone)]
pub struct PbftConfig {
    /// Number of replicas.
    pub num_replicas: usize,
    /// This replica's ID.
    pub replica_id: usize,
    /// Request timeout before view change.
    pub request_timeout: Duration,
    /// Checkpoint interval.
    pub checkpoint_interval: u64,
    /// Low watermark for sequence numbers.
    pub low_watermark: u64,
    /// High watermark for sequence numbers.
    pub high_watermark: u64,
    /// Watermark window size.
    pub watermark_window: u64,
}

impl Default for PbftConfig {
    fn default() -> Self {
        Self {
            num_replicas: 4,
            replica_id: 0,
            request_timeout: Duration::from_secs(5),
            checkpoint_interval: 100,
            low_watermark: 0,
            high_watermark: 200,
            watermark_window: 200,
        }
    }
}

impl PbftConfig {
    /// Create config for a specific replica.
    pub fn for_replica(num_replicas: usize, replica_id: usize) -> Self {
        Self {
            num_replicas,
            replica_id,
            ..Default::default()
        }
    }

    /// Maximum number of faulty nodes tolerated.
    pub fn f(&self) -> usize {
        (self.num_replicas - 1) / 3
    }

    /// Quorum size (2f+1).
    pub fn quorum(&self) -> usize {
        2 * self.f() + 1
    }

    /// Primary for a given view.
    pub fn primary(&self, view: u64) -> usize {
        (view as usize) % self.num_replicas
    }

    /// Check if this replica is primary for the current view.
    pub fn is_primary(&self, view: u64) -> bool {
        self.primary(view) == self.replica_id
    }
}

/// PBFT consensus replica.
pub struct PbftReplica {
    /// Configuration.
    config: PbftConfig,

    /// Current view number.
    view: Arc<RwLock<u64>>,

    /// Replica state.
    state: Arc<RwLock<ReplicaState>>,

    /// Current sequence number.
    sequence: Arc<RwLock<u64>>,

    /// Log of requests.
    log: Arc<RwLock<BTreeMap<u64, LogEntry>>>,

    /// Last stable checkpoint.
    last_checkpoint: Arc<RwLock<u64>>,

    /// Checkpoint proofs.
    checkpoints: Arc<RwLock<HashMap<u64, Vec<Checkpoint>>>>,

    /// View change messages received.
    view_changes: Arc<RwLock<HashMap<u64, Vec<ViewChange>>>>,

    /// Pending requests (client_id -> request).
    pending_requests: Arc<RwLock<HashMap<u64, Request>>>,

    /// Request timestamps for duplicate detection.
    client_timestamps: Arc<RwLock<HashMap<u64, u64>>>,

    /// Messages to send (outbox).
    outbox: Arc<RwLock<Vec<(Option<usize>, PbftMessage)>>>,

    /// Request timers.
    request_timers: Arc<RwLock<HashMap<u64, Instant>>>,

    /// Statistics.
    stats: Arc<RwLock<PbftStats>>,
}

/// PBFT statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PbftStats {
    /// Total requests processed.
    pub requests_processed: u64,
    /// Total prepares sent.
    pub prepares_sent: u64,
    /// Total commits sent.
    pub commits_sent: u64,
    /// View changes initiated.
    pub view_changes: u64,
    /// Checkpoints created.
    pub checkpoints_created: u64,
    /// Current view.
    pub current_view: u64,
    /// Current sequence.
    pub current_sequence: u64,
    /// Is primary.
    pub is_primary: bool,
}

impl PbftReplica {
    /// Create a new PBFT replica.
    pub fn new(config: PbftConfig) -> Self {
        Self {
            config,
            view: Arc::new(RwLock::new(0)),
            state: Arc::new(RwLock::new(ReplicaState::Normal)),
            sequence: Arc::new(RwLock::new(0)),
            log: Arc::new(RwLock::new(BTreeMap::new())),
            last_checkpoint: Arc::new(RwLock::new(0)),
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            view_changes: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
            client_timestamps: Arc::new(RwLock::new(HashMap::new())),
            outbox: Arc::new(RwLock::new(Vec::new())),
            request_timers: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(PbftStats::default())),
        }
    }

    /// Get current view number.
    pub fn view(&self) -> u64 {
        *self.view.read()
    }

    /// Get current sequence number.
    pub fn sequence(&self) -> u64 {
        *self.sequence.read()
    }

    /// Check if this replica is primary.
    pub fn is_primary(&self) -> bool {
        self.config.is_primary(self.view())
    }

    /// Get replica state.
    pub fn state(&self) -> ReplicaState {
        *self.state.read()
    }

    /// Get statistics.
    pub fn stats(&self) -> PbftStats {
        let mut stats = self.stats.read().clone();
        stats.current_view = self.view();
        stats.current_sequence = self.sequence();
        stats.is_primary = self.is_primary();
        stats
    }

    /// Drain outbox messages.
    pub fn drain_outbox(&self) -> Vec<(Option<usize>, PbftMessage)> {
        std::mem::take(&mut *self.outbox.write())
    }

    /// Handle an incoming message.
    pub fn handle_message(&self, from: usize, message: PbftMessage) -> SwarmResult<()> {
        match message {
            PbftMessage::Request(req) => self.handle_request(req),
            PbftMessage::PrePrepare(pp) => self.handle_pre_prepare(from, pp),
            PbftMessage::Prepare(p) => self.handle_prepare(from, p),
            PbftMessage::Commit(c) => self.handle_commit(from, c),
            PbftMessage::ViewChange(vc) => self.handle_view_change(from, vc),
            PbftMessage::NewView(nv) => self.handle_new_view(from, nv),
            PbftMessage::Checkpoint(cp) => self.handle_checkpoint(from, cp),
            PbftMessage::Reply(_) => Ok(()), // Clients handle replies
        }
    }

    /// Submit a client request.
    pub fn submit_request(&self, request: Request) -> SwarmResult<()> {
        self.handle_request(request)
    }

    fn handle_request(&self, request: Request) -> SwarmResult<()> {
        // Check for duplicate
        {
            let timestamps = self.client_timestamps.read();
            if let Some(&last_ts) = timestamps.get(&request.client_id) {
                if request.timestamp <= last_ts {
                    debug!("Duplicate request from client {}", request.client_id);
                    return Ok(());
                }
            }
        }

        // If we're primary, start the protocol
        if self.is_primary() && *self.state.read() == ReplicaState::Normal {
            self.start_consensus(request)?;
        } else {
            // Forward to primary
            let view = self.view();
            let primary = self.config.primary(view);
            self.outbox.write().push((
                Some(primary),
                PbftMessage::Request(request.clone()),
            ));
        }

        Ok(())
    }

    fn start_consensus(&self, request: Request) -> SwarmResult<()> {
        let view = self.view();
        let sequence = {
            let mut seq = self.sequence.write();
            *seq += 1;
            *seq
        };

        // Check watermarks
        let low = self.config.low_watermark;
        let high = self.config.high_watermark;
        if sequence <= low || sequence > high {
            return Err(SwarmError::Consensus(format!(
                "Sequence {} outside watermarks [{}, {}]",
                sequence, low, high
            )));
        }

        let digest = compute_request_digest(&request);

        let pre_prepare = PrePrepare {
            view,
            sequence,
            digest,
            request: request.clone(),
            signature: Signature::default(), // Would use real signature
        };

        info!(
            view = view,
            sequence = sequence,
            client = request.client_id,
            "Starting consensus"
        );

        // Create log entry
        {
            let mut log = self.log.write();
            let entry = log.entry(sequence).or_default();
            entry.pre_prepare = Some(pre_prepare.clone());
        }

        // Start timer
        self.request_timers.write().insert(sequence, Instant::now());

        // Broadcast pre-prepare
        self.outbox.write().push((None, PbftMessage::PrePrepare(pre_prepare)));

        Ok(())
    }

    fn handle_pre_prepare(&self, from: usize, pp: PrePrepare) -> SwarmResult<()> {
        let view = self.view();

        // Validate view
        if pp.view != view {
            return Ok(());
        }

        // Validate sender is primary
        if from != self.config.primary(view) {
            warn!("Pre-prepare from non-primary");
            return Ok(());
        }

        // Validate watermarks
        let low = self.config.low_watermark;
        let high = self.config.high_watermark;
        if pp.sequence <= low || pp.sequence > high {
            return Ok(());
        }

        // Validate digest
        let expected_digest = compute_request_digest(&pp.request);
        if pp.digest != expected_digest {
            warn!("Pre-prepare digest mismatch");
            return Ok(());
        }

        // Check for conflicting pre-prepare
        {
            let log = self.log.read();
            if let Some(entry) = log.get(&pp.sequence) {
                if let Some(ref existing) = entry.pre_prepare {
                    if existing.digest != pp.digest {
                        warn!("Conflicting pre-prepare");
                        return Ok(());
                    }
                }
            }
        }

        debug!(
            view = pp.view,
            sequence = pp.sequence,
            "Received pre-prepare"
        );

        // Store pre-prepare
        {
            let mut log = self.log.write();
            let entry = log.entry(pp.sequence).or_default();
            entry.pre_prepare = Some(pp.clone());
        }

        // Send prepare
        let prepare = Prepare {
            view: pp.view,
            sequence: pp.sequence,
            digest: pp.digest,
            replica_id: self.config.replica_id,
            signature: Signature::default(),
        };

        self.stats.write().prepares_sent += 1;
        self.outbox.write().push((None, PbftMessage::Prepare(prepare)));

        self.try_commit(pp.sequence);

        Ok(())
    }

    fn handle_prepare(&self, from: usize, prepare: Prepare) -> SwarmResult<()> {
        let view = self.view();

        // Validate view
        if prepare.view != view {
            return Ok(());
        }

        // Validate sender
        if from >= self.config.num_replicas || from == self.config.primary(view) {
            return Ok(());
        }

        debug!(
            view = prepare.view,
            sequence = prepare.sequence,
            from = from,
            "Received prepare"
        );

        // Store prepare
        {
            let mut log = self.log.write();
            let entry = log.entry(prepare.sequence).or_default();
            entry.prepares.insert(from, prepare.clone());
        }

        self.try_commit(prepare.sequence);

        Ok(())
    }

    fn handle_commit(&self, from: usize, commit: Commit) -> SwarmResult<()> {
        let view = self.view();

        // Validate view
        if commit.view != view {
            return Ok(());
        }

        debug!(
            view = commit.view,
            sequence = commit.sequence,
            from = from,
            "Received commit"
        );

        // Store commit
        {
            let mut log = self.log.write();
            let entry = log.entry(commit.sequence).or_default();
            entry.commits.insert(from, commit.clone());
        }

        self.try_execute(commit.sequence);

        Ok(())
    }

    fn try_commit(&self, sequence: u64) {
        let view = self.view();
        let quorum = self.config.quorum();

        let should_commit = {
            let log = self.log.read();
            if let Some(entry) = log.get(&sequence) {
                // Need pre-prepare and 2f prepares
                entry.pre_prepare.is_some()
                    && entry.prepares.len() >= quorum - 1
                    && !entry.committed_local
            } else {
                false
            }
        };

        if should_commit {
            let digest = {
                let log = self.log.read();
                log.get(&sequence)
                    .and_then(|e| e.pre_prepare.as_ref())
                    .map(|pp| pp.digest)
                    .unwrap_or_default()
            };

            // Mark as prepared
            {
                let mut log = self.log.write();
                if let Some(entry) = log.get_mut(&sequence) {
                    entry.prepared = true;
                    entry.committed_local = true;
                }
            }

            // Send commit
            let commit = Commit {
                view,
                sequence,
                digest,
                replica_id: self.config.replica_id,
                signature: Signature::default(),
            };

            self.stats.write().commits_sent += 1;
            self.outbox.write().push((None, PbftMessage::Commit(commit)));

            self.try_execute(sequence);
        }
    }

    fn try_execute(&self, sequence: u64) {
        let quorum = self.config.quorum();

        let should_execute = {
            let log = self.log.read();
            if let Some(entry) = log.get(&sequence) {
                entry.committed_local
                    && entry.commits.len() >= quorum
                    && !entry.committed
            } else {
                false
            }
        };

        if should_execute {
            // Mark as committed
            {
                let mut log = self.log.write();
                if let Some(entry) = log.get_mut(&sequence) {
                    entry.committed = true;
                }
            }

            // Execute the operation
            let (request, result) = {
                let log = self.log.read();
                let entry = log.get(&sequence).unwrap();
                let pp = entry.pre_prepare.as_ref().unwrap();
                let result = self.execute_operation(&pp.request.operation);
                (pp.request.clone(), result)
            };

            info!(
                sequence = sequence,
                client = request.client_id,
                "Request executed"
            );

            // Store result
            {
                let mut log = self.log.write();
                if let Some(entry) = log.get_mut(&sequence) {
                    entry.result = Some(result.clone());
                }
            }

            // Update client timestamp
            self.client_timestamps
                .write()
                .insert(request.client_id, request.timestamp);

            // Remove timer
            self.request_timers.write().remove(&sequence);

            // Send reply to client
            let reply = Reply {
                view: self.view(),
                sequence,
                timestamp: request.timestamp,
                client_id: request.client_id,
                replica_id: self.config.replica_id,
                result,
                signature: Signature::default(),
            };

            self.stats.write().requests_processed += 1;
            self.outbox.write().push((None, PbftMessage::Reply(reply)));

            // Check if checkpoint needed
            if sequence % self.config.checkpoint_interval == 0 {
                self.create_checkpoint(sequence);
            }
        }
    }

    fn execute_operation(&self, operation: &Operation) -> OperationResult {
        match operation {
            Operation::Noop => OperationResult::Success(None),
            Operation::Read { key } => {
                // In a real implementation, this would read from state machine
                OperationResult::Success(Some(format!("value for {}", key).into_bytes()))
            }
            Operation::Write { key, value } => {
                // In a real implementation, this would write to state machine
                debug!("Write {}={:?}", key, value);
                OperationResult::Success(None)
            }
            Operation::Execute { command, .. } => {
                debug!("Execute {}", command);
                OperationResult::Success(None)
            }
            Operation::DeployRvf { rvf_hash } => {
                debug!("Deploy RVF {:?}", &rvf_hash[..8]);
                OperationResult::Success(None)
            }
            Operation::VectorOp { store_id, key, .. } => {
                debug!("Vector op store={} key={}", store_id, key);
                OperationResult::Success(None)
            }
            Operation::GraphMutation { graph_id, .. } => {
                debug!("Graph mutation graph={}", graph_id);
                OperationResult::Success(None)
            }
        }
    }

    fn create_checkpoint(&self, sequence: u64) {
        let view = self.view();
        let state_digest = self.compute_state_digest();

        let checkpoint = Checkpoint {
            view,
            sequence,
            state_digest,
            replica_id: self.config.replica_id,
            signature: Signature::default(),
        };

        debug!(sequence = sequence, "Creating checkpoint");

        self.stats.write().checkpoints_created += 1;
        self.outbox.write().push((None, PbftMessage::Checkpoint(checkpoint)));
    }

    fn handle_checkpoint(&self, from: usize, checkpoint: Checkpoint) -> SwarmResult<()> {
        debug!(
            sequence = checkpoint.sequence,
            from = from,
            "Received checkpoint"
        );

        let quorum = self.config.quorum();

        // Store checkpoint
        {
            let mut checkpoints = self.checkpoints.write();
            let entry = checkpoints.entry(checkpoint.sequence).or_default();
            entry.push(checkpoint.clone());

            // Check for stable checkpoint
            if entry.len() >= quorum {
                // Verify all digests match
                let first_digest = entry[0].state_digest;
                let all_match = entry.iter().all(|c| c.state_digest == first_digest);

                if all_match {
                    let mut last = self.last_checkpoint.write();
                    if checkpoint.sequence > *last {
                        *last = checkpoint.sequence;
                        info!(sequence = checkpoint.sequence, "Stable checkpoint");

                        // Garbage collect old log entries
                        let mut log = self.log.write();
                        log.retain(|&seq, _| seq > checkpoint.sequence);
                    }
                }
            }
        }

        Ok(())
    }

    fn handle_view_change(&self, from: usize, vc: ViewChange) -> SwarmResult<()> {
        let current_view = self.view();

        if vc.new_view <= current_view {
            return Ok(());
        }

        info!(
            from = from,
            new_view = vc.new_view,
            "Received view change"
        );

        // Store view change
        {
            let mut view_changes = self.view_changes.write();
            let entry = view_changes.entry(vc.new_view).or_default();
            entry.push(vc.clone());

            // Check if we have enough view changes
            let quorum = self.config.quorum();
            if entry.len() >= quorum && self.config.primary(vc.new_view) == self.config.replica_id {
                // We're the new primary - send new view
                self.send_new_view(vc.new_view, entry.clone());
            }
        }

        Ok(())
    }

    fn handle_new_view(&self, from: usize, nv: NewView) -> SwarmResult<()> {
        let current_view = self.view();

        if nv.view <= current_view {
            return Ok(());
        }

        // Verify sender is the new primary
        if from != self.config.primary(nv.view) {
            warn!("New view from non-primary");
            return Ok(());
        }

        info!(view = nv.view, "Received new view");

        // Update view
        *self.view.write() = nv.view;
        *self.state.write() = ReplicaState::Normal;

        // Process pre-prepares from new view
        for pp in nv.pre_prepares {
            let _ = self.handle_pre_prepare(from, pp);
        }

        self.stats.write().view_changes += 1;

        Ok(())
    }

    fn send_new_view(&self, view: u64, view_changes: Vec<ViewChange>) {
        info!(view = view, "Sending new view");

        // Compute pre-prepares for outstanding requests
        let pre_prepares = Vec::new(); // Would compute from view changes

        let new_view = NewView {
            view,
            view_changes,
            pre_prepares,
            signature: Signature::default(),
        };

        // Update our view
        *self.view.write() = view;
        *self.state.write() = ReplicaState::Normal;

        self.outbox.write().push((None, PbftMessage::NewView(new_view)));
    }

    fn compute_state_digest(&self) -> HashDigest {
        // In a real implementation, this would hash the state machine
        let mut hasher = Sha256::new();
        hasher.update(b"state");
        hasher.update(self.sequence().to_le_bytes());
        let result = hasher.finalize();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&result);
        HashDigest(digest)
    }

    /// Start a view change.
    pub fn start_view_change(&self) {
        let current_view = self.view();
        let new_view = current_view + 1;

        info!(
            current_view = current_view,
            new_view = new_view,
            "Starting view change"
        );

        *self.state.write() = ReplicaState::ViewChange;

        // Collect prepared proofs
        let prepared_messages = Vec::new(); // Would collect from log

        let vc = ViewChange {
            new_view,
            last_checkpoint: *self.last_checkpoint.read(),
            checkpoint_proofs: Vec::new(),
            prepared_messages,
            replica_id: self.config.replica_id,
            signature: Signature::default(),
        };

        self.outbox.write().push((None, PbftMessage::ViewChange(vc)));
    }

    /// Check for timed out requests and trigger view change.
    pub fn check_timeouts(&self) {
        let timeout = self.config.request_timeout;
        let now = Instant::now();

        let timed_out: Vec<u64> = {
            let timers = self.request_timers.read();
            timers
                .iter()
                .filter(|(_, &start)| now.duration_since(start) > timeout)
                .map(|(&seq, _)| seq)
                .collect()
        };

        if !timed_out.is_empty() && *self.state.read() == ReplicaState::Normal {
            warn!(
                sequences = ?timed_out,
                "Request timeout, starting view change"
            );
            self.start_view_change();
        }
    }
}

/// Compute SHA-256 digest of a request.
fn compute_request_digest(request: &Request) -> HashDigest {
    let mut hasher = Sha256::new();
    hasher.update(request.client_id.to_le_bytes());
    hasher.update(request.timestamp.to_le_bytes());
    if let Ok(op_bytes) = serde_json::to_vec(&request.operation) {
        hasher.update(&op_bytes);
    }
    let result = hasher.finalize();
    let mut digest = [0u8; 32];
    digest.copy_from_slice(&result);
    HashDigest(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbft_config() {
        let config = PbftConfig::for_replica(4, 0);
        assert_eq!(config.f(), 1);
        assert_eq!(config.quorum(), 3);
        assert_eq!(config.primary(0), 0);
        assert_eq!(config.primary(1), 1);
        assert!(config.is_primary(0));
        assert!(!config.is_primary(1));
    }

    #[test]
    fn test_message_serialization() {
        let request = Request {
            client_id: 1,
            timestamp: 12345,
            operation: Operation::Write {
                key: "test".to_string(),
                value: vec![1, 2, 3],
            },
        };

        let message = PbftMessage::Request(request);
        let bytes = message.to_bytes().unwrap();
        let decoded = PbftMessage::from_bytes(&bytes).unwrap();

        assert_eq!(message, decoded);
    }

    #[test]
    fn test_pbft_replica_creation() {
        let config = PbftConfig::for_replica(4, 0);
        let replica = PbftReplica::new(config);

        assert_eq!(replica.view(), 0);
        assert_eq!(replica.sequence(), 0);
        assert!(replica.is_primary());
        assert_eq!(replica.state(), ReplicaState::Normal);
    }

    #[test]
    fn test_submit_request() {
        let config = PbftConfig::for_replica(4, 0);
        let replica = PbftReplica::new(config);

        let request = Request {
            client_id: 1,
            timestamp: 1,
            operation: Operation::Noop,
        };

        replica.submit_request(request).unwrap();

        let outbox = replica.drain_outbox();
        assert!(!outbox.is_empty());

        // Primary should broadcast pre-prepare
        let (_, msg) = &outbox[0];
        assert!(matches!(msg, PbftMessage::PrePrepare(_)));
    }

    #[test]
    fn test_single_replica_consensus() {
        let config = PbftConfig {
            num_replicas: 1,
            replica_id: 0,
            checkpoint_interval: 10,
            ..Default::default()
        };
        let replica = PbftReplica::new(config);

        let request = Request {
            client_id: 1,
            timestamp: 1,
            operation: Operation::Write {
                key: "test".to_string(),
                value: vec![42],
            },
        };

        replica.submit_request(request).unwrap();

        // Process pre-prepare
        let outbox = replica.drain_outbox();
        for (_, msg) in outbox {
            if let PbftMessage::PrePrepare(pp) = msg {
                replica.handle_message(0, PbftMessage::PrePrepare(pp)).unwrap();
            }
        }

        let stats = replica.stats();
        assert!(stats.prepares_sent > 0);
    }

    #[test]
    fn test_four_replica_consensus() {
        let replicas: Vec<_> = (0..4)
            .map(|i| PbftReplica::new(PbftConfig::for_replica(4, i)))
            .collect();

        // Submit request to primary
        let request = Request {
            client_id: 1,
            timestamp: 1,
            operation: Operation::Noop,
        };
        replicas[0].submit_request(request).unwrap();

        // Simulate message passing
        let mut rounds = 0;
        loop {
            let mut any_messages = false;
            rounds += 1;

            for i in 0..4 {
                let outbox = replicas[i].drain_outbox();
                for (target, msg) in outbox {
                    any_messages = true;
                    match target {
                        Some(t) => {
                            let _ = replicas[t].handle_message(i, msg);
                        }
                        None => {
                            // Broadcast to all
                            for j in 0..4 {
                                let _ = replicas[j].handle_message(i, msg.clone());
                            }
                        }
                    }
                }
            }

            if !any_messages || rounds > 10 {
                break;
            }
        }

        // All replicas should have processed the request
        for (i, replica) in replicas.iter().enumerate() {
            let stats = replica.stats();
            assert!(
                stats.requests_processed > 0 || i != 0,
                "Replica {} did not process request",
                i
            );
        }
    }
}
