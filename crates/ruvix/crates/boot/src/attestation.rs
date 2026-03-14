//! Boot attestation types and utilities.
//!
//! Boot attestation records the initial system state:
//! - RVF package hash
//! - Capability table hash
//! - Region layout hash
//! - Boot timestamp

use sha2::{Sha256, Digest};

/// Boot attestation entry recorded as the first witness log entry.
///
/// Contains cryptographic hashes of the initial system state
/// for later verification and audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct BootAttestation {
    /// SHA-256 hash of the RVF package that was booted.
    pub rvf_hash: [u8; 32],

    /// SHA-256 hash of the initial capability table.
    pub capability_table_hash: [u8; 32],

    /// SHA-256 hash of the region layout.
    pub region_layout_hash: [u8; 32],

    /// Boot timestamp in nanoseconds since UNIX epoch.
    pub boot_timestamp_ns: u64,

    /// Boot sequence number (for multi-boot detection).
    pub boot_sequence: u64,

    /// Platform identifier.
    pub platform_id: u64,

    /// Reserved for future use.
    pub reserved: [u8; 16],
}

impl BootAttestation {
    /// Size of boot attestation in bytes.
    pub const SIZE: usize = 32 + 32 + 32 + 8 + 8 + 8 + 16; // 136 bytes

    /// Creates a new boot attestation.
    #[must_use]
    pub fn new(
        rvf_hash: [u8; 32],
        capability_table_hash: [u8; 32],
        region_layout_hash: [u8; 32],
        boot_timestamp_ns: u64,
    ) -> Self {
        Self {
            rvf_hash,
            capability_table_hash,
            region_layout_hash,
            boot_timestamp_ns,
            boot_sequence: 0,
            platform_id: 0,
            reserved: [0u8; 16],
        }
    }

    /// Creates a boot attestation with full metadata.
    #[must_use]
    pub fn with_metadata(
        rvf_hash: [u8; 32],
        capability_table_hash: [u8; 32],
        region_layout_hash: [u8; 32],
        boot_timestamp_ns: u64,
        boot_sequence: u64,
        platform_id: u64,
    ) -> Self {
        Self {
            rvf_hash,
            capability_table_hash,
            region_layout_hash,
            boot_timestamp_ns,
            boot_sequence,
            platform_id,
            reserved: [0u8; 16],
        }
    }

    /// Computes the hash of this attestation.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.rvf_hash);
        hasher.update(&self.capability_table_hash);
        hasher.update(&self.region_layout_hash);
        hasher.update(&self.boot_timestamp_ns.to_le_bytes());
        hasher.update(&self.boot_sequence.to_le_bytes());
        hasher.update(&self.platform_id.to_le_bytes());
        hasher.update(&self.reserved);

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Serializes the attestation to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut bytes = [0u8; Self::SIZE];

        bytes[0..32].copy_from_slice(&self.rvf_hash);
        bytes[32..64].copy_from_slice(&self.capability_table_hash);
        bytes[64..96].copy_from_slice(&self.region_layout_hash);
        bytes[96..104].copy_from_slice(&self.boot_timestamp_ns.to_le_bytes());
        bytes[104..112].copy_from_slice(&self.boot_sequence.to_le_bytes());
        bytes[112..120].copy_from_slice(&self.platform_id.to_le_bytes());
        bytes[120..136].copy_from_slice(&self.reserved);

        bytes
    }

    /// Deserializes an attestation from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::SIZE {
            return None;
        }

        let mut rvf_hash = [0u8; 32];
        rvf_hash.copy_from_slice(&bytes[0..32]);

        let mut capability_table_hash = [0u8; 32];
        capability_table_hash.copy_from_slice(&bytes[32..64]);

        let mut region_layout_hash = [0u8; 32];
        region_layout_hash.copy_from_slice(&bytes[64..96]);

        let boot_timestamp_ns = u64::from_le_bytes([
            bytes[96], bytes[97], bytes[98], bytes[99],
            bytes[100], bytes[101], bytes[102], bytes[103],
        ]);

        let boot_sequence = u64::from_le_bytes([
            bytes[104], bytes[105], bytes[106], bytes[107],
            bytes[108], bytes[109], bytes[110], bytes[111],
        ]);

        let platform_id = u64::from_le_bytes([
            bytes[112], bytes[113], bytes[114], bytes[115],
            bytes[116], bytes[117], bytes[118], bytes[119],
        ]);

        let mut reserved = [0u8; 16];
        reserved.copy_from_slice(&bytes[120..136]);

        Some(Self {
            rvf_hash,
            capability_table_hash,
            region_layout_hash,
            boot_timestamp_ns,
            boot_sequence,
            platform_id,
            reserved,
        })
    }

    /// Verifies that this attestation matches expected values.
    #[must_use]
    pub fn verify(&self, expected_rvf_hash: &[u8; 32]) -> bool {
        self.rvf_hash == *expected_rvf_hash
    }
}

impl Default for BootAttestation {
    fn default() -> Self {
        Self {
            rvf_hash: [0u8; 32],
            capability_table_hash: [0u8; 32],
            region_layout_hash: [0u8; 32],
            boot_timestamp_ns: 0,
            boot_sequence: 0,
            platform_id: 0,
            reserved: [0u8; 16],
        }
    }
}

/// Witness log entry for attestation.
///
/// Used for entries other than boot attestation that record
/// proof-gated mutations and other security events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct AttestationEntry {
    /// Entry type identifier.
    pub entry_type: AttestationEntryType,

    /// Hash of the attested data.
    pub data_hash: [u8; 32],

    /// Timestamp in nanoseconds since UNIX epoch.
    pub timestamp_ns: u64,

    /// Task that generated this attestation.
    pub task_id: u32,

    /// Component that generated this attestation.
    pub component_id: u32,

    /// Additional flags.
    pub flags: AttestationFlags,

    /// Reserved for future use.
    pub reserved: [u8; 8],
}

/// Attestation entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AttestationEntryType {
    /// Proof-gated vector mutation.
    VectorMutation = 0,

    /// Proof-gated graph mutation.
    GraphMutation = 1,

    /// Capability delegation.
    CapabilityDelegate = 2,

    /// Capability revocation.
    CapabilityRevoke = 3,

    /// Component state checkpoint.
    Checkpoint = 4,

    /// Component state rollback.
    Rollback = 5,

    /// Custom application-defined.
    Custom = 255,
}

/// Attestation entry flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct AttestationFlags(pub u32);

impl AttestationFlags {
    /// No flags.
    pub const NONE: Self = Self(0);

    /// Entry contains high-priority mutation.
    pub const HIGH_PRIORITY: Self = Self(1 << 0);

    /// Entry was generated by a deadline-driven task.
    pub const DEADLINE_DRIVEN: Self = Self(1 << 1);

    /// Entry required Deep proof tier.
    pub const DEEP_PROOF: Self = Self(1 << 2);

    /// Entry was generated during rollback.
    pub const DURING_ROLLBACK: Self = Self(1 << 3);

    /// Checks if a flag is set.
    #[inline]
    #[must_use]
    pub const fn contains(&self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    /// Returns the union of two flag sets.
    #[inline]
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl AttestationEntry {
    /// Size of attestation entry in bytes.
    pub const SIZE: usize = 1 + 32 + 8 + 4 + 4 + 4 + 8; // 61 bytes

    /// Creates a new attestation entry.
    #[must_use]
    pub fn new(
        entry_type: AttestationEntryType,
        data_hash: [u8; 32],
        timestamp_ns: u64,
        task_id: u32,
        component_id: u32,
    ) -> Self {
        Self {
            entry_type,
            data_hash,
            timestamp_ns,
            task_id,
            component_id,
            flags: AttestationFlags::NONE,
            reserved: [0u8; 8],
        }
    }

    /// Creates an attestation entry with flags.
    #[must_use]
    pub fn with_flags(
        entry_type: AttestationEntryType,
        data_hash: [u8; 32],
        timestamp_ns: u64,
        task_id: u32,
        component_id: u32,
        flags: AttestationFlags,
    ) -> Self {
        Self {
            entry_type,
            data_hash,
            timestamp_ns,
            task_id,
            component_id,
            flags,
            reserved: [0u8; 8],
        }
    }

    /// Computes the hash of this entry.
    #[must_use]
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&[self.entry_type as u8]);
        hasher.update(&self.data_hash);
        hasher.update(&self.timestamp_ns.to_le_bytes());
        hasher.update(&self.task_id.to_le_bytes());
        hasher.update(&self.component_id.to_le_bytes());
        hasher.update(&self.flags.0.to_le_bytes());
        hasher.update(&self.reserved);

        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

impl Default for AttestationEntry {
    fn default() -> Self {
        Self {
            entry_type: AttestationEntryType::Custom,
            data_hash: [0u8; 32],
            timestamp_ns: 0,
            task_id: 0,
            component_id: 0,
            flags: AttestationFlags::NONE,
            reserved: [0u8; 8],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boot_attestation_creation() {
        let att = BootAttestation::new(
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            1234567890,
        );

        assert_eq!(att.rvf_hash, [1u8; 32]);
        assert_eq!(att.capability_table_hash, [2u8; 32]);
        assert_eq!(att.region_layout_hash, [3u8; 32]);
        assert_eq!(att.boot_timestamp_ns, 1234567890);
    }

    #[test]
    fn test_boot_attestation_serialization() {
        let att = BootAttestation::new(
            [0xAA; 32],
            [0xBB; 32],
            [0xCC; 32],
            999999999,
        );

        let bytes = att.to_bytes();
        let recovered = BootAttestation::from_bytes(&bytes).unwrap();

        assert_eq!(att, recovered);
    }

    #[test]
    fn test_boot_attestation_hash() {
        let att1 = BootAttestation::new([1u8; 32], [2u8; 32], [3u8; 32], 100);
        let att2 = BootAttestation::new([1u8; 32], [2u8; 32], [3u8; 32], 100);
        let att3 = BootAttestation::new([1u8; 32], [2u8; 32], [4u8; 32], 100);

        // Same inputs = same hash
        assert_eq!(att1.hash(), att2.hash());

        // Different inputs = different hash
        assert_ne!(att1.hash(), att3.hash());
    }

    #[test]
    fn test_boot_attestation_verify() {
        let expected_hash = [0xDEu8; 32];
        let att = BootAttestation::new(expected_hash, [0u8; 32], [0u8; 32], 0);

        assert!(att.verify(&expected_hash));
        assert!(!att.verify(&[0u8; 32]));
    }

    #[test]
    fn test_attestation_entry_creation() {
        let entry = AttestationEntry::new(
            AttestationEntryType::VectorMutation,
            [0xAB; 32],
            1234567890,
            1,
            2,
        );

        assert_eq!(entry.entry_type, AttestationEntryType::VectorMutation);
        assert_eq!(entry.task_id, 1);
        assert_eq!(entry.component_id, 2);
    }

    #[test]
    fn test_attestation_flags() {
        let flags = AttestationFlags::HIGH_PRIORITY.union(AttestationFlags::DEEP_PROOF);

        assert!(flags.contains(AttestationFlags::HIGH_PRIORITY));
        assert!(flags.contains(AttestationFlags::DEEP_PROOF));
        assert!(!flags.contains(AttestationFlags::DEADLINE_DRIVEN));
    }

    #[test]
    fn test_boot_attestation_size() {
        assert_eq!(BootAttestation::SIZE, 136);
    }
}
