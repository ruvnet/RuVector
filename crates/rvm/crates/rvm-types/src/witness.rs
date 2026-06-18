//! Witness record types for the audit subsystem.
//!
//! Every privileged action in RVM emits a compact, immutable audit record.
//! This is a core invariant (INV-3): **no witness, no mutation**.
//!
//! The witness record is exactly 64 bytes, cache-line aligned, with FNV-1a
//! hash chaining for tamper evidence. See ADR-134 for the full specification.

/// A single witness record. Exactly 64 bytes, cache-line aligned.
///
/// All fields are little-endian. The record is `#[repr(C, align(64))]` to
/// guarantee layout and alignment on all target architectures (`AArch64`,
/// RISC-V, x86-64).
///
/// # Layout
///
/// | Offset | Size | Field                | Description |
/// |--------|------|----------------------|-------------|
/// | 0      | 8    | `sequence`           | Monotonic sequence number |
/// | 8      | 8    | `timestamp_ns`       | Nanosecond timestamp |
/// | 16     | 1    | `action_kind`        | Privileged action discriminant |
/// | 17     | 1    | `proof_tier`         | Proof tier (1, 2, or 3) |
/// | 18     | 1    | `flags`              | Action-specific flags |
/// | 19     | 1    | `_reserved`          | Reserved (must be zero) |
/// | 20     | 4    | `actor_partition_id` | Actor partition |
/// | 24     | 8    | `target_object_id`   | Target object |
/// | 32     | 4    | `capability_hash`    | Truncated cap hash |
/// | 36     | 8    | `payload`            | Action-specific data |
/// | 44     | 4    | `prev_hash`          | FNV-1a chain link |
/// | 48     | 4    | `record_hash`        | FNV-1a self-integrity |
/// | 52     | 8    | `aux`                | Secondary payload / TEE sig |
/// | 60     | 4    | `_pad`               | Padding to 64 bytes |
#[derive(Debug, Clone, Copy)]
#[repr(C, align(64))]
pub struct WitnessRecord {
    /// Monotonic sequence number. Provides global ordering of all privileged actions.
    pub sequence: u64,
    /// Nanosecond timestamp from the system timer (`CNTVCT_EL0` / `rdtsc`).
    pub timestamp_ns: u64,
    /// Which privileged action was performed (see [`ActionKind`]).
    pub action_kind: u8,
    /// Which proof tier authorized this action (1 = P1, 2 = P2, 3 = P3).
    pub proof_tier: u8,
    /// Action-specific flags (interpretation varies by `action_kind`).
    pub flags: u8,
    /// Reserved for future use. Must be zero.
    reserved: u8,
    /// Partition that performed the action.
    pub actor_partition_id: u32,
    /// Object acted upon: partition, region, capability, etc.
    pub target_object_id: u64,
    /// Truncated FNV-1a hash of the capability used (not the full token).
    pub capability_hash: u32,
    /// Action-specific data, packed by kind.
    ///
    /// Examples:
    /// - `PartitionSplit`: `new_id_a` in bytes \[0..4\], `new_id_b` in bytes \[4..8\].
    /// - `RegionTransfer`: `from_partition` in bytes \[0..4\], `to_partition` in bytes \[4..8\].
    pub payload: [u8; 8],
    /// Folded hash of the previous record's chain value (chain link for
    /// tamper evidence). The chain value binds the previous record's
    /// content hash, sequence, and its own predecessor.
    pub prev_hash: u32,
    /// Folded hash of bytes \[0..44\] of this record (self-integrity).
    /// Covers all content fields: `sequence`, `timestamp_ns`,
    /// `action_kind`, `proof_tier`, `flags`, `actor_partition_id`,
    /// `target_object_id`, `capability_hash`, and `payload`.
    pub record_hash: u32,
    /// Secondary payload or TEE signature fragment.
    pub aux: [u8; 8],
    /// Padding to guarantee 64-byte total size.
    pad: [u8; 4],
}

// Compile-time size assertion: the record MUST be exactly 64 bytes.
const _: () = {
    assert!(core::mem::size_of::<WitnessRecord>() == 64);
};

impl WitnessRecord {
    /// Number of leading bytes of the serialized record that constitute
    /// the record *content* (everything before `prev_hash`): `sequence`,
    /// `timestamp_ns`, `action_kind`, `proof_tier`, `flags`, reserved,
    /// `actor_partition_id`, `target_object_id`, `capability_hash`,
    /// and `payload`. The self-integrity `record_hash` is computed over
    /// exactly these bytes.
    pub const CONTENT_LEN: usize = 44;

    /// Number of leading bytes of the serialized record covered by a
    /// signature (everything before `aux` and padding): the content
    /// bytes plus `prev_hash` and `record_hash`.
    pub const SIGNED_LEN: usize = 52;

    /// Serialize the record's fields to a 64-byte little-endian array
    /// in layout order.
    ///
    /// This is the canonical serialization used for both the
    /// self-integrity `record_hash` (over `[..CONTENT_LEN]`) and witness
    /// signatures (over `[..SIGNED_LEN]`). Fields are serialized
    /// manually rather than via `repr(C)` transmutation to avoid
    /// depending on padding semantics across platforms.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut buf = [0u8; 64];
        buf[0..8].copy_from_slice(&self.sequence.to_le_bytes());
        buf[8..16].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[16] = self.action_kind;
        buf[17] = self.proof_tier;
        buf[18] = self.flags;
        buf[19] = self.reserved;
        buf[20..24].copy_from_slice(&self.actor_partition_id.to_le_bytes());
        buf[24..32].copy_from_slice(&self.target_object_id.to_le_bytes());
        buf[32..36].copy_from_slice(&self.capability_hash.to_le_bytes());
        buf[36..44].copy_from_slice(&self.payload);
        buf[44..48].copy_from_slice(&self.prev_hash.to_le_bytes());
        buf[48..52].copy_from_slice(&self.record_hash.to_le_bytes());
        buf[52..60].copy_from_slice(&self.aux);
        // buf[60..64] is pad, stays zero.
        buf
    }

    /// Create a zeroed witness record (genesis / placeholder).
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            sequence: 0,
            timestamp_ns: 0,
            action_kind: 0,
            proof_tier: 0,
            flags: 0,
            reserved: 0,
            actor_partition_id: 0,
            target_object_id: 0,
            capability_hash: 0,
            payload: [0; 8],
            prev_hash: 0,
            record_hash: 0,
            aux: [0; 8],
            pad: [0; 4],
        }
    }

    /// Deserialize a record from its canonical 64-byte little-endian
    /// wire form (inverse of [`Self::to_bytes`]).
    ///
    /// Used by versioned log readers that dispatch on the wire version
    /// byte (offset [`WIRE_VERSION_OFFSET`]): v1 records carry `0`
    /// there (the reserved byte), v2 records carry `2`.
    #[must_use]
    pub fn from_bytes(buf: &[u8; 64]) -> Self {
        let mut u64_buf = [0u8; 8];
        let mut u32_buf = [0u8; 4];
        u64_buf.copy_from_slice(&buf[0..8]);
        let sequence = u64::from_le_bytes(u64_buf);
        u64_buf.copy_from_slice(&buf[8..16]);
        let timestamp_ns = u64::from_le_bytes(u64_buf);
        u32_buf.copy_from_slice(&buf[20..24]);
        let actor_partition_id = u32::from_le_bytes(u32_buf);
        u64_buf.copy_from_slice(&buf[24..32]);
        let target_object_id = u64::from_le_bytes(u64_buf);
        u32_buf.copy_from_slice(&buf[32..36]);
        let capability_hash = u32::from_le_bytes(u32_buf);
        let mut payload = [0u8; 8];
        payload.copy_from_slice(&buf[36..44]);
        u32_buf.copy_from_slice(&buf[44..48]);
        let prev_hash = u32::from_le_bytes(u32_buf);
        u32_buf.copy_from_slice(&buf[48..52]);
        let record_hash = u32::from_le_bytes(u32_buf);
        let mut aux = [0u8; 8];
        aux.copy_from_slice(&buf[52..60]);
        Self {
            sequence,
            timestamp_ns,
            action_kind: buf[16],
            proof_tier: buf[17],
            flags: buf[18],
            reserved: buf[19],
            actor_partition_id,
            target_object_id,
            capability_hash,
            payload,
            prev_hash,
            record_hash,
            aux,
            pad: [0; 4],
        }
    }
}

/// Byte offset of the wire format version discriminator, shared by all
/// witness record versions.
///
/// In v1 records this is the `reserved` byte (always `0`); in v2 records
/// it is the explicit `version` field (always `2`). A reader can
/// therefore dispatch on `bytes[offset + WIRE_VERSION_OFFSET]` to decide
/// whether the next record is 64 bytes (v1) or 96 bytes (v2).
pub const WIRE_VERSION_OFFSET: usize = 19;

/// A version-2 witness record. Exactly 96 bytes (ADR-134 v2).
///
/// The v2 format widens the tamper-evidence chain from the v1 32-bit
/// folded links to full 128-bit keyed-BLAKE3 MACs, at the cost of
/// growing the record from 64 to 96 bytes (+50%). The first 44 bytes
/// (the *content* region) keep the exact v1 field order so emitters and
/// audit queries port over unchanged; byte 19 is the version
/// discriminator (`2` here, `0` in v1, see [`WIRE_VERSION_OFFSET`]).
///
/// # Layout (all little-endian)
///
/// | Offset | Size | Field                | Description |
/// |--------|------|----------------------|-------------|
/// | 0      | 8    | `sequence`           | Monotonic sequence number |
/// | 8      | 8    | `timestamp_ns`       | Nanosecond timestamp |
/// | 16     | 1    | `action_kind`        | Privileged action discriminant |
/// | 17     | 1    | `proof_tier`         | Proof tier (1, 2, or 3) |
/// | 18     | 1    | `flags`              | Action-specific flags |
/// | 19     | 1    | `version`            | Format version, always `2` |
/// | 20     | 4    | `actor_partition_id` | Actor partition |
/// | 24     | 8    | `target_object_id`   | Target object |
/// | 32     | 4    | `capability_hash`    | Truncated cap hash |
/// | 36     | 8    | `payload`            | Action-specific data |
/// | 44     | 8    | `aux`                | Secondary payload (not chained) |
/// | 52     | 4    | `reserved`           | Reserved (must be zero) |
/// | 56     | 16   | `prev_mac`           | Predecessor's `chain_mac` |
/// | 72     | 16   | `chain_mac`          | Keyed-BLAKE3 chain MAC |
/// | 88     | 8    | `pad`                | Padding to 96 bytes |
///
/// # Chain construction
///
/// `chain_mac = trunc128(BLAKE3_keyed(key, bytes[0..44] || prev_mac))`.
/// The MAC input is 60 bytes, i.e. exactly one keyed BLAKE3 compression,
/// and it binds **both** the record content (including `sequence`) and
/// the full-width link to the predecessor. A single MAC therefore
/// provides self-integrity *and* the chain link; v1 needed two hashes
/// per record for a strictly weaker (32-bit folded) guarantee.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
pub struct WitnessRecordV2 {
    /// Monotonic sequence number. Provides global ordering of all privileged actions.
    pub sequence: u64,
    /// Nanosecond timestamp from the system timer (`CNTVCT_EL0` / `rdtsc`).
    pub timestamp_ns: u64,
    /// Which privileged action was performed (see [`ActionKind`]).
    pub action_kind: u8,
    /// Which proof tier authorized this action (1 = P1, 2 = P2, 3 = P3).
    pub proof_tier: u8,
    /// Action-specific flags (interpretation varies by `action_kind`).
    pub flags: u8,
    /// Wire format version. Always [`Self::VERSION`] (`2`) for this type.
    pub version: u8,
    /// Partition that performed the action.
    pub actor_partition_id: u32,
    /// Object acted upon: partition, region, capability, etc.
    pub target_object_id: u64,
    /// Truncated FNV-1a hash of the capability used (not the full token).
    pub capability_hash: u32,
    /// Action-specific data, packed by kind.
    pub payload: [u8; 8],
    /// Secondary payload. **Not** covered by the chain MAC, mirroring
    /// v1 where `aux` held an after-the-fact signature.
    pub aux: [u8; 8],
    /// Reserved for future use. Must be zero.
    reserved: [u8; 4],
    /// Full-width (128-bit) chain link: the predecessor's `chain_mac`,
    /// or the log's genesis value for the first record.
    pub prev_mac: [u8; 16],
    /// Keyed-BLAKE3 MAC over `bytes[0..44] || prev_mac`, truncated to
    /// 128 bits. Serves as both self-integrity hash and chain value.
    pub chain_mac: [u8; 16],
    /// Padding to guarantee 96-byte total size.
    pad: [u8; 8],
}

// Compile-time size assertion: the v2 record MUST be exactly 96 bytes.
const _: () = {
    assert!(core::mem::size_of::<WitnessRecordV2>() == 96);
};

impl WitnessRecordV2 {
    /// Wire format version discriminator stored at byte 19.
    pub const VERSION: u8 = 2;

    /// Serialized size in bytes.
    pub const SIZE: usize = 96;

    /// Number of leading bytes that constitute the record *content*
    /// (same field order as v1): `sequence` through `payload`.
    pub const CONTENT_LEN: usize = 44;

    /// Length of the chain MAC input: `CONTENT_LEN` content bytes
    /// followed by the 16-byte `prev_mac`. At 60 bytes this fits in a
    /// single 64-byte BLAKE3 block, so each append costs exactly one
    /// keyed compression.
    pub const MAC_INPUT_LEN: usize = Self::CONTENT_LEN + 16;

    /// Create a zeroed v2 record (version byte set, everything else zero).
    #[must_use]
    pub const fn zeroed() -> Self {
        Self {
            sequence: 0,
            timestamp_ns: 0,
            action_kind: 0,
            proof_tier: 0,
            flags: 0,
            version: Self::VERSION,
            actor_partition_id: 0,
            target_object_id: 0,
            capability_hash: 0,
            payload: [0; 8],
            aux: [0; 8],
            reserved: [0; 4],
            prev_mac: [0; 16],
            chain_mac: [0; 16],
            pad: [0; 8],
        }
    }

    /// Build a v2 record from the content fields of a v1
    /// [`WitnessRecord`] (the common emitter currency).
    ///
    /// Copies `timestamp_ns`, `action_kind`, `proof_tier`, `flags`,
    /// `actor_partition_id`, `target_object_id`, `capability_hash`,
    /// `payload`, and `aux`. The v1 chain fields (`sequence`,
    /// `prev_hash`, `record_hash`) are **ignored**: the v2 log assigns
    /// its own sequence and MACs on append.
    #[must_use]
    pub fn from_v1_content(record: &WitnessRecord) -> Self {
        let mut out = Self::zeroed();
        out.timestamp_ns = record.timestamp_ns;
        out.action_kind = record.action_kind;
        out.proof_tier = record.proof_tier;
        out.flags = record.flags;
        out.actor_partition_id = record.actor_partition_id;
        out.target_object_id = record.target_object_id;
        out.capability_hash = record.capability_hash;
        out.payload = record.payload;
        out.aux = record.aux;
        out
    }

    /// Serialize to the canonical 96-byte little-endian wire form.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; 96] {
        let mut buf = [0u8; 96];
        buf[0..8].copy_from_slice(&self.sequence.to_le_bytes());
        buf[8..16].copy_from_slice(&self.timestamp_ns.to_le_bytes());
        buf[16] = self.action_kind;
        buf[17] = self.proof_tier;
        buf[18] = self.flags;
        buf[19] = self.version;
        buf[20..24].copy_from_slice(&self.actor_partition_id.to_le_bytes());
        buf[24..32].copy_from_slice(&self.target_object_id.to_le_bytes());
        buf[32..36].copy_from_slice(&self.capability_hash.to_le_bytes());
        buf[36..44].copy_from_slice(&self.payload);
        buf[44..52].copy_from_slice(&self.aux);
        buf[52..56].copy_from_slice(&self.reserved);
        buf[56..72].copy_from_slice(&self.prev_mac);
        buf[72..88].copy_from_slice(&self.chain_mac);
        // buf[88..96] is pad, stays zero.
        buf
    }

    /// Deserialize from the canonical 96-byte wire form (inverse of
    /// [`Self::to_bytes`]). Does not validate the version byte; callers
    /// performing verification must check `version == Self::VERSION`.
    #[must_use]
    pub fn from_bytes(buf: &[u8; 96]) -> Self {
        let mut u64_buf = [0u8; 8];
        let mut u32_buf = [0u8; 4];
        u64_buf.copy_from_slice(&buf[0..8]);
        let sequence = u64::from_le_bytes(u64_buf);
        u64_buf.copy_from_slice(&buf[8..16]);
        let timestamp_ns = u64::from_le_bytes(u64_buf);
        u32_buf.copy_from_slice(&buf[20..24]);
        let actor_partition_id = u32::from_le_bytes(u32_buf);
        u64_buf.copy_from_slice(&buf[24..32]);
        let target_object_id = u64::from_le_bytes(u64_buf);
        u32_buf.copy_from_slice(&buf[32..36]);
        let capability_hash = u32::from_le_bytes(u32_buf);
        let mut payload = [0u8; 8];
        payload.copy_from_slice(&buf[36..44]);
        let mut aux = [0u8; 8];
        aux.copy_from_slice(&buf[44..52]);
        let mut reserved = [0u8; 4];
        reserved.copy_from_slice(&buf[52..56]);
        let mut prev_mac = [0u8; 16];
        prev_mac.copy_from_slice(&buf[56..72]);
        let mut chain_mac = [0u8; 16];
        chain_mac.copy_from_slice(&buf[72..88]);
        Self {
            sequence,
            timestamp_ns,
            action_kind: buf[16],
            proof_tier: buf[17],
            flags: buf[18],
            version: buf[19],
            actor_partition_id,
            target_object_id,
            capability_hash,
            payload,
            aux,
            reserved,
            prev_mac,
            chain_mac,
            pad: [0; 8],
        }
    }
}

/// A 256-bit witness commitment hash.
///
/// Used to anchor state transitions in the RVM witness trail. This is
/// a fixed-size value type suitable for embedding in `no_std` contexts
/// without heap allocation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct WitnessHash {
    bytes: [u8; 32],
}

impl WitnessHash {
    /// The zero hash, used as a sentinel for the genesis state.
    pub const ZERO: Self = Self { bytes: [0u8; 32] };

    /// Create a witness hash from raw bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { bytes }
    }

    /// Return the raw byte representation.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }

    /// Check whether this is the zero (genesis) hash.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        let mut i = 0;
        while i < 32 {
            if self.bytes[i] != 0 {
                return false;
            }
            i += 1;
        }
        true
    }
}

impl core::fmt::Debug for WitnessHash {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "WitnessHash(")?;
        for byte in &self.bytes[..4] {
            write!(f, "{byte:02x}")?;
        }
        write!(f, "..)")
    }
}

impl core::fmt::Display for WitnessHash {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for byte in &self.bytes {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Privileged actions that produce witness records (ADR-134, Section 2).
///
/// Organized by subsystem. Hex values allow easy filtering by prefix in
/// audit queries (0x0_ = partition, 0x1_ = capability, 0x2_ = memory, etc.).
///
/// If a privileged action exists without a corresponding kind, the system
/// has an audit gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ActionKind {
    // --- Partition lifecycle (0x01-0x0F) ---
    /// A new partition was created.
    PartitionCreate      = 0x01,
    /// A partition was destroyed and its resources freed.
    PartitionDestroy     = 0x02,
    /// A partition was suspended (tasks paused).
    PartitionSuspend     = 0x03,
    /// A suspended partition was resumed.
    PartitionResume      = 0x04,
    /// A partition was split along a mincut boundary.
    PartitionSplit       = 0x05,
    /// Two partitions were merged into one.
    PartitionMerge       = 0x06,
    /// A partition was hibernated to dormant/cold storage.
    PartitionHibernate   = 0x07,
    /// A hibernated partition was reconstructed from its receipt.
    PartitionReconstruct = 0x08,
    /// A partition was migrated to another node.
    PartitionMigrate     = 0x09,

    // --- Capability operations (0x10-0x1F) ---
    /// A capability was granted (copied) to another partition.
    CapabilityGrant      = 0x10,
    /// A capability was revoked.
    CapabilityRevoke     = 0x11,
    /// A capability was delegated (with depth decrement).
    CapabilityDelegate   = 0x12,
    /// Delegation depth was increased (escalation).
    CapabilityEscalate   = 0x13,
    /// Capability was attenuated during a partition split (DC-8).
    CapabilityAttenuated = 0x14,

    // --- Memory operations (0x20-0x2F) ---
    /// A memory region was created.
    RegionCreate         = 0x20,
    /// A memory region was destroyed.
    RegionDestroy        = 0x21,
    /// A memory region was transferred to another partition.
    RegionTransfer       = 0x22,
    /// A memory region was shared (read-only) with another partition.
    RegionShare          = 0x23,
    /// A shared memory region was unshared.
    RegionUnshare        = 0x24,
    /// A memory region was promoted to a warmer tier.
    RegionPromote        = 0x25,
    /// A memory region was demoted to a colder tier.
    RegionDemote         = 0x26,
    /// A stage-2 mapping was added for a memory region.
    RegionMap            = 0x27,
    /// A stage-2 mapping was removed for a memory region.
    RegionUnmap          = 0x28,

    // --- Communication (0x30-0x3F) ---
    /// A communication edge was created between two partitions.
    CommEdgeCreate       = 0x30,
    /// A communication edge was destroyed.
    CommEdgeDestroy      = 0x31,
    /// An IPC message was sent.
    IpcSend              = 0x32,
    /// An IPC message was received.
    IpcReceive           = 0x33,
    /// A zero-copy memory share was established.
    ZeroCopyShare        = 0x34,
    /// A notification signal was sent.
    NotificationSignal   = 0x35,

    // --- Device operations (0x40-0x4F) ---
    /// A device lease was granted.
    DeviceLeaseGrant     = 0x40,
    /// A device lease was revoked.
    DeviceLeaseRevoke    = 0x41,
    /// A device lease expired (time-bounded).
    DeviceLeaseExpire    = 0x42,
    /// A device lease was renewed.
    DeviceLeaseRenew     = 0x43,

    // --- Proof verification (0x50-0x5F) ---
    /// A P1 capability check passed.
    ProofVerifiedP1      = 0x50,
    /// A P2 policy validation passed.
    ProofVerifiedP2      = 0x51,
    /// A P3 deep proof passed.
    ProofVerifiedP3      = 0x52,
    /// A proof was rejected.
    ProofRejected        = 0x53,
    /// A proof was escalated to a higher tier.
    ProofEscalated       = 0x54,

    // --- Scheduler decisions (0x60-0x6F) ---
    /// Scheduler epoch boundary (bulk switch summary per DC-10).
    SchedulerEpoch       = 0x60,
    /// Scheduler mode switched (Reflex / Flow / Recovery).
    SchedulerModeSwitch  = 0x61,
    /// A task was spawned within a partition.
    TaskSpawn            = 0x62,
    /// A task was terminated.
    TaskTerminate        = 0x63,
    /// Scheduler triggered a structural split.
    StructuralSplit      = 0x64,
    /// Scheduler triggered a structural merge.
    StructuralMerge      = 0x65,

    // --- Recovery actions (0x70-0x7F) ---
    /// System entered recovery mode.
    RecoveryEnter        = 0x70,
    /// System exited recovery mode.
    RecoveryExit         = 0x71,
    /// A recovery checkpoint was created.
    CheckpointCreated    = 0x72,
    /// A recovery checkpoint was restored.
    CheckpointRestored   = 0x73,
    /// Mincut budget was exceeded, stale cut used (DC-2 fallback).
    MinCutBudgetExceeded = 0x74,
    /// System entered degraded mode (DC-6).
    DegradedModeEntered  = 0x75,
    /// System exited degraded mode.
    DegradedModeExited   = 0x76,

    // --- Boot and attestation (0x80-0x8F) ---
    /// Boot attestation record (genesis witness).
    BootAttestation      = 0x80,
    /// Boot sequence completed successfully.
    BootComplete         = 0x81,
    /// TEE-backed attestation record.
    TeeAttestation       = 0x82,

    // --- Vector/Graph mutations (0x90-0x9F) ---
    /// A vector was inserted into the coherence graph.
    VectorPut            = 0x90,
    /// A vector was deleted from the coherence graph.
    VectorDelete         = 0x91,
    /// A graph mutation occurred.
    GraphMutation        = 0x92,
    /// Coherence scores were recomputed.
    CoherenceRecomputed  = 0x93,

    // --- VMID management (0xA0-0xAF) ---
    /// A physical VMID was reclaimed from a hibernated partition (DC-12).
    VmidReclaim          = 0xA0,
    /// Migration timed out and was aborted (DC-7).
    MigrationTimeout     = 0xA1,
}

impl ActionKind {
    /// Return the subsystem prefix for this action kind.
    ///
    /// Useful for filtering audit queries by subsystem:
    /// 0 = partition, 1 = capability, 2 = memory, 3 = communication,
    /// 4 = device, 5 = proof, 6 = scheduler, 7 = recovery,
    /// 8 = boot, 9 = graph, 0xA = VMID management.
    #[must_use]
    pub const fn subsystem(self) -> u8 {
        (self as u8) >> 4
    }
}

/// FNV-1a hash over a byte slice.
///
/// Chosen for speed (< 50 ns for 64 bytes), not cryptographic strength.
/// For tamper resistance against a capable adversary, use the optional
/// TEE-backed `WitnessSigner` (ADR-134, Section 9).
///
/// Unrolls the per-byte loop by 8 for inputs >= 8 bytes while preserving
/// standard FNV-1a byte-order sensitivity. The remainder is handled
/// one byte at a time.
#[inline]
#[must_use]
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

    let mut hash: u64 = FNV_OFFSET;
    let len = data.len();
    let mut i = 0;

    // Process 8 bytes at a time (unrolled), preserving standard FNV-1a
    // per-byte XOR-then-multiply semantics for hash compatibility.
    while i + 8 <= len {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= data[i + 1] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= data[i + 2] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= data[i + 3] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= data[i + 4] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= data[i + 5] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= data[i + 6] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        hash ^= data[i + 7] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 8;
    }

    // Handle remaining bytes one at a time.
    while i < len {
        hash ^= data[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }

    hash
}

/// FNV-1a hash truncated to 32 bits.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn fnv1a_32(data: &[u8]) -> u32 {
    // Intentional truncation: 64-bit hash folded to 32 bits.
    fnv1a_64(data) as u32
}

/// Default witness ring buffer capacity in records.
///
/// 16 MiB / 64 bytes = 262,144 records.
/// At 100,000 privileged actions per second this gives approximately 2.6
/// seconds of hot storage before overflow drain is needed.
pub const WITNESS_RING_CAPACITY: usize = 262_144;

/// Witness record size in bytes.
pub const WITNESS_RECORD_SIZE: usize = 64;
