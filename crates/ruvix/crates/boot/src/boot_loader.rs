//! Main boot loader implementation.
//!
//! Orchestrates the five-stage boot sequence per ADR-087 Section 9.1.

use crate::manifest::RvfManifest;
use crate::signature::ML_DSA_65_PUBLIC_KEY_SIZE;
use crate::stages::{Stage0Hardware, Stage1Verify, Stage2Create, Stage3Mount, Stage4Attest};
use crate::witness_log::{WitnessLog, WitnessLogConfig};
use crate::attestation::BootAttestation;
use crate::capability_distribution::CapabilityDistribution;
use ruvix_types::KernelError;
use ruvix_cap::BootCapabilitySet;

/// Boot stage enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum BootStage {
    /// Stage 0: Hardware initialization.
    HardwareInit = 0,

    /// Stage 1: RVF verification.
    RvfVerify = 1,

    /// Stage 2: Kernel object creation.
    ObjectCreate = 2,

    /// Stage 3: Component mount.
    ComponentMount = 3,

    /// Stage 4: First attestation.
    FirstAttestation = 4,

    /// Boot complete.
    Complete = 5,
}

impl BootStage {
    /// Returns the stage name.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::HardwareInit => "Hardware Init",
            Self::RvfVerify => "RVF Verify",
            Self::ObjectCreate => "Object Create",
            Self::ComponentMount => "Component Mount",
            Self::FirstAttestation => "First Attestation",
            Self::Complete => "Complete",
        }
    }
}

/// Boot loader configuration.
#[derive(Debug, Clone)]
pub struct BootConfig {
    /// Boot public key for signature verification.
    pub public_key: [u8; ML_DSA_65_PUBLIC_KEY_SIZE],

    /// Witness log configuration.
    pub witness_log_config: WitnessLogConfig,

    /// Whether to enable verbose boot logging.
    pub verbose: bool,

    /// Platform identifier override (0 = auto-detect).
    pub platform_id_override: u64,
}

impl Default for BootConfig {
    fn default() -> Self {
        Self {
            public_key: [0u8; ML_DSA_65_PUBLIC_KEY_SIZE],
            witness_log_config: WitnessLogConfig::default(),
            verbose: false,
            platform_id_override: 0,
        }
    }
}

/// Boot result containing all boot artifacts.
#[derive(Debug)]
pub struct BootResult {
    /// Current boot stage.
    pub stage: BootStage,

    /// Hardware info from Stage 0.
    pub hardware: Stage0Hardware,

    /// Verified manifest from Stage 1.
    pub manifest: Option<RvfManifest>,

    /// Boot capabilities from Stage 2.
    pub boot_capabilities: Option<BootCapabilitySet>,

    /// Witness log from Stage 2.
    pub witness_log: Option<WitnessLog>,

    /// Capability distribution from Stage 3.
    pub capability_distribution: Option<CapabilityDistribution>,

    /// Boot attestation from Stage 4.
    pub boot_attestation: Option<BootAttestation>,

    /// Whether SEC-001 capability drop was performed.
    pub sec001_capability_drop: bool,
}

impl BootResult {
    /// Creates a new boot result.
    #[must_use]
    fn new() -> Self {
        Self {
            stage: BootStage::HardwareInit,
            hardware: Stage0Hardware::new(),
            manifest: None,
            boot_capabilities: None,
            witness_log: None,
            capability_distribution: None,
            boot_attestation: None,
            sec001_capability_drop: false,
        }
    }
}

/// Main boot loader.
///
/// Orchestrates the five-stage boot sequence:
/// - Stage 0: Hardware initialization
/// - Stage 1: RVF manifest verification
/// - Stage 2: Kernel object creation
/// - Stage 3: Component mount + capability distribution
/// - Stage 4: First attestation
pub struct BootLoader {
    /// Configuration.
    config: BootConfig,

    /// Boot result accumulator.
    result: BootResult,

    /// Stage handlers.
    stage0: Stage0Hardware,
    stage1: Stage1Verify,
    stage2: Stage2Create,
    stage3: Stage3Mount,
    stage4: Stage4Attest,
}

impl BootLoader {
    /// Creates a new boot loader with the given configuration.
    #[must_use]
    pub fn new(config: BootConfig) -> Self {
        let mut stage1 = Stage1Verify::new();
        stage1.set_public_key(&config.public_key);

        Self {
            config,
            result: BootResult::new(),
            stage0: Stage0Hardware::new(),
            stage1,
            stage2: Stage2Create::new(),
            stage3: Stage3Mount::new(),
            stage4: Stage4Attest::new(),
        }
    }

    /// Creates a boot loader for testing (test keys, verbose).
    #[cfg(test)]
    #[must_use]
    pub fn test_loader() -> Self {
        let config = BootConfig {
            public_key: [0u8; ML_DSA_65_PUBLIC_KEY_SIZE],
            witness_log_config: WitnessLogConfig::default(),
            verbose: true,
            platform_id_override: 0,
        };

        Self::new(config)
    }

    /// Executes the full boot sequence.
    ///
    /// # Panics
    ///
    /// Panics if signature verification fails (SEC-001).
    ///
    /// # Errors
    ///
    /// Returns an error if any boot stage fails (except signature verification,
    /// which panics per SEC-001).
    pub fn boot(&mut self, manifest_bytes: &[u8], signature: &[u8]) -> Result<&BootResult, KernelError> {
        // Stage 0: Hardware Init
        self.execute_stage0()?;

        // Stage 1: RVF Verify (panics on signature failure)
        self.execute_stage1(manifest_bytes, signature)?;

        // Stage 2: Object Create
        self.execute_stage2()?;

        // Stage 3: Component Mount
        self.execute_stage3()?;

        // Stage 4: First Attestation
        self.execute_stage4()?;

        self.result.stage = BootStage::Complete;

        if self.config.verbose {
            eprintln!("Boot complete");
        }

        Ok(&self.result)
    }

    /// Executes Stage 0: Hardware initialization.
    fn execute_stage0(&mut self) -> Result<(), KernelError> {
        if self.config.verbose {
            eprintln!("Executing Stage 0: Hardware Init");
        }

        self.stage0.execute()?;
        self.result.hardware = self.stage0.clone();
        self.result.stage = BootStage::RvfVerify;

        Ok(())
    }

    /// Executes Stage 1: RVF verification.
    ///
    /// # Panics
    ///
    /// Panics if signature verification fails (SEC-001).
    fn execute_stage1(&mut self, manifest_bytes: &[u8], signature: &[u8]) -> Result<(), KernelError> {
        if self.config.verbose {
            eprintln!("Executing Stage 1: RVF Verify");
        }

        // This will panic on signature failure (SEC-001)
        self.stage1.execute(manifest_bytes, signature)?;

        self.result.manifest = self.stage1.manifest.clone();
        self.result.stage = BootStage::ObjectCreate;

        Ok(())
    }

    /// Executes Stage 2: Kernel object creation.
    fn execute_stage2(&mut self) -> Result<(), KernelError> {
        if self.config.verbose {
            eprintln!("Executing Stage 2: Object Create");
        }

        let manifest = self.result.manifest.as_ref()
            .ok_or(KernelError::InternalError)?;

        let physical_memory = self.result.hardware.physical_memory_bytes;

        self.stage2.execute(manifest, physical_memory)?;

        self.result.boot_capabilities = self.stage2.boot_capabilities.clone();
        self.result.witness_log = self.stage2.witness_log.take();
        self.result.stage = BootStage::ComponentMount;

        Ok(())
    }

    /// Executes Stage 3: Component mount + capability distribution.
    fn execute_stage3(&mut self) -> Result<(), KernelError> {
        if self.config.verbose {
            eprintln!("Executing Stage 3: Component Mount");
        }

        let manifest = self.result.manifest.as_ref()
            .ok_or(KernelError::InternalError)?;

        let boot_caps = self.result.boot_capabilities.as_ref()
            .ok_or(KernelError::InternalError)?;

        self.stage3.execute(manifest, boot_caps)?;

        self.result.capability_distribution = self.stage3.capability_distribution.clone();
        self.result.sec001_capability_drop = self.stage3.capability_distribution
            .as_ref()
            .map(|d| d.root_dropped_to_minimum)
            .unwrap_or(false);
        self.result.stage = BootStage::FirstAttestation;

        Ok(())
    }

    /// Executes Stage 4: First attestation.
    fn execute_stage4(&mut self) -> Result<(), KernelError> {
        if self.config.verbose {
            eprintln!("Executing Stage 4: First Attestation");
        }

        let manifest = self.result.manifest.as_ref()
            .ok_or(KernelError::InternalError)?;

        let boot_caps = self.result.boot_capabilities.as_ref()
            .ok_or(KernelError::InternalError)?;

        let witness_log = self.result.witness_log.as_mut()
            .ok_or(KernelError::InternalError)?;

        self.stage4.execute(manifest, witness_log, boot_caps)?;

        self.result.boot_attestation = self.stage4.attestation.clone();

        Ok(())
    }

    /// Returns the current boot stage.
    #[inline]
    #[must_use]
    pub fn current_stage(&self) -> BootStage {
        self.result.stage
    }

    /// Returns the boot result (may be incomplete if boot failed).
    #[inline]
    #[must_use]
    pub fn result(&self) -> &BootResult {
        &self.result
    }

    /// Checks if SEC-001 capability drop was performed.
    #[inline]
    #[must_use]
    pub fn sec001_compliant(&self) -> bool {
        self.result.sec001_capability_drop
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::SIGNATURE_SIZE;

    fn create_test_manifest() -> Vec<u8> {
        let mut manifest = vec![0u8; 100];
        manifest[0..4].copy_from_slice(b"RVF1");
        manifest[4..6].copy_from_slice(&1u16.to_le_bytes());
        manifest[6..8].copy_from_slice(&0u16.to_le_bytes());
        manifest
    }

    fn create_test_signature(manifest: &[u8]) -> Vec<u8> {
        use sha2::{Sha256, Digest};

        let mut sig = vec![0u8; SIGNATURE_SIZE];
        sig[0..4].copy_from_slice(b"TEST");

        let mut hasher = Sha256::new();
        hasher.update(manifest);
        let hash = hasher.finalize();
        sig[4..36].copy_from_slice(&hash);

        sig
    }

    #[test]
    fn test_boot_stage_ordering() {
        assert!(BootStage::HardwareInit < BootStage::RvfVerify);
        assert!(BootStage::RvfVerify < BootStage::ObjectCreate);
        assert!(BootStage::ObjectCreate < BootStage::ComponentMount);
        assert!(BootStage::ComponentMount < BootStage::FirstAttestation);
        assert!(BootStage::FirstAttestation < BootStage::Complete);
    }

    #[test]
    fn test_boot_config_default() {
        let config = BootConfig::default();
        assert_eq!(config.public_key, [0u8; ML_DSA_65_PUBLIC_KEY_SIZE]);
        assert!(!config.verbose);
    }

    #[test]
    fn test_full_boot_sequence() {
        let mut loader = BootLoader::test_loader();

        let manifest = create_test_manifest();
        let signature = create_test_signature(&manifest);

        let result = loader.boot(&manifest, &signature).unwrap();

        assert_eq!(result.stage, BootStage::Complete);
        assert!(result.manifest.is_some());
        assert!(result.boot_capabilities.is_some());
        assert!(result.witness_log.is_some());
        assert!(result.boot_attestation.is_some());
        assert!(result.sec001_capability_drop);
    }

    #[test]
    fn test_boot_sec001_compliant() {
        let mut loader = BootLoader::test_loader();

        let manifest = create_test_manifest();
        let signature = create_test_signature(&manifest);

        loader.boot(&manifest, &signature).unwrap();

        assert!(loader.sec001_compliant());
    }

    #[test]
    fn test_boot_current_stage() {
        let loader = BootLoader::test_loader();

        // Initial stage should be HardwareInit
        assert_eq!(loader.current_stage(), BootStage::HardwareInit);
    }

    #[test]
    #[should_panic(expected = "Boot signature verification failed")]
    fn test_boot_invalid_signature_panics() {
        let mut loader = BootLoader::test_loader();

        let manifest = create_test_manifest();
        let wrong_manifest = b"wrong manifest";
        let signature = create_test_signature(wrong_manifest);

        // This should panic due to SEC-001
        let _ = loader.boot(&manifest, &signature);
    }

    #[test]
    fn test_boot_invalid_manifest() {
        let mut loader = BootLoader::test_loader();

        // Invalid manifest (wrong magic)
        let manifest = b"XXXX";
        let signature = vec![0u8; SIGNATURE_SIZE]; // All zeros is valid for test key

        let result = loader.boot(manifest, &signature);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), KernelError::InvalidManifest);
    }
}
