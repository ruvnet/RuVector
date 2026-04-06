//! RVF package mounting (rvf_mount syscall implementation).
//!
//! The rvf_mount operation:
//! 1. Verifies package signature
//! 2. Parses manifest
//! 3. Creates regions per memory schema
//! 4. Mounts WASM components
//! 5. Distributes capabilities per manifest
//! 6. Connects queues per wiring
//! 7. Spawns initial tasks per WIT entry points

use crate::manifest::RvfManifest;
use crate::signature::SignatureVerifier;
use ruvix_types::{
    KernelError, RvfMountHandle, RvfVerifyStatus, RegionHandle,
    TaskHandle, TaskPriority,
};

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Configuration for RVF mount operations.
#[derive(Debug, Clone)]
pub struct MountConfig {
    /// Maximum components per package.
    pub max_components: usize,

    /// Maximum regions per package.
    pub max_regions: usize,

    /// Maximum queues per package.
    pub max_queues: usize,

    /// Default task priority for spawned tasks.
    pub default_task_priority: TaskPriority,

    /// Whether to verify signatures (should always be true in production).
    pub verify_signatures: bool,
}

impl Default for MountConfig {
    fn default() -> Self {
        Self {
            max_components: 256,
            max_regions: 256,
            max_queues: 1024,
            default_task_priority: TaskPriority::Normal,
            verify_signatures: true,
        }
    }
}

/// Result of an RVF mount operation.
#[derive(Debug, Clone)]
pub struct MountResult {
    /// Handle to the mounted RVF package.
    pub mount_handle: RvfMountHandle,

    /// Verification status.
    pub verify_status: RvfVerifyStatus,

    /// Number of components mounted.
    pub components_mounted: usize,

    /// Number of regions created.
    pub regions_created: usize,

    /// Number of queues connected.
    pub queues_connected: usize,

    /// Number of tasks spawned.
    pub tasks_spawned: usize,

    /// Created region handles.
    pub region_handles: [Option<RegionHandle>; 256],
    /// Number of region handles in the array.
    pub region_handle_count: usize,

    /// Spawned task handles.
    pub task_handles: [Option<TaskHandle>; 64],
    /// Number of task handles in the array.
    pub task_handle_count: usize,
}

impl MountResult {
    /// Creates a successful mount result.
    #[must_use]
    pub fn success(mount_handle: RvfMountHandle) -> Self {
        Self {
            mount_handle,
            verify_status: RvfVerifyStatus::SignatureValid,
            components_mounted: 0,
            regions_created: 0,
            queues_connected: 0,
            tasks_spawned: 0,
            region_handles: [None; 256],
            region_handle_count: 0,
            task_handles: [None; 64],
            task_handle_count: 0,
        }
    }

    /// Creates a failed mount result.
    #[must_use]
    pub fn failure(status: RvfVerifyStatus) -> Self {
        Self {
            mount_handle: RvfMountHandle::null(),
            verify_status: status,
            components_mounted: 0,
            regions_created: 0,
            queues_connected: 0,
            tasks_spawned: 0,
            region_handles: [None; 256],
            region_handle_count: 0,
            task_handles: [None; 64],
            task_handle_count: 0,
        }
    }

    /// Adds a created region handle.
    pub fn add_region(&mut self, handle: RegionHandle) -> Result<(), KernelError> {
        if self.region_handle_count >= 256 {
            return Err(KernelError::LimitExceeded);
        }
        self.region_handles[self.region_handle_count] = Some(handle);
        self.region_handle_count += 1;
        self.regions_created += 1;
        Ok(())
    }

    /// Adds a spawned task handle.
    pub fn add_task(&mut self, handle: TaskHandle) -> Result<(), KernelError> {
        if self.task_handle_count >= 64 {
            return Err(KernelError::LimitExceeded);
        }
        self.task_handles[self.task_handle_count] = Some(handle);
        self.task_handle_count += 1;
        self.tasks_spawned += 1;
        Ok(())
    }
}

/// RVF mount operation handler.
///
/// Implements the rvf_mount syscall per ADR-087.
pub struct RvfMount {
    /// Configuration.
    config: MountConfig,

    /// Signature verifier.
    verifier: Option<SignatureVerifier>,

    /// Next mount handle ID.
    next_mount_id: u32,

    /// Next region handle ID.
    next_region_id: u32,

    /// Next task handle ID.
    next_task_id: u32,
}

impl RvfMount {
    /// Creates a new RVF mount handler.
    #[must_use]
    pub fn new(config: MountConfig) -> Self {
        Self {
            config,
            verifier: None,
            next_mount_id: 1,
            next_region_id: 0x1000,
            next_task_id: 0x100,
        }
    }

    /// Sets the signature verification public key.
    ///
    /// # Panics
    ///
    /// Panics if the public key has wrong length. Prefer using
    /// [`SignatureVerifier::try_new`] directly for non-panicking construction.
    pub fn set_public_key(&mut self, public_key: &[u8]) {
        self.verifier = Some(SignatureVerifier::new(public_key));
    }

    /// Mounts an RVF package.
    ///
    /// # Steps
    ///
    /// 1. Verify package signature
    /// 2. Parse manifest
    /// 3. Create regions per memory schema
    /// 4. Mount WASM components
    /// 5. Distribute capabilities per manifest
    /// 6. Connect queues per wiring
    /// 7. Spawn initial tasks per WIT entry points
    ///
    /// # Errors
    ///
    /// - `InvalidSignature` if signature verification fails
    /// - `InvalidManifest` if manifest parsing fails
    /// - `OutOfMemory` if region creation fails
    /// - `LimitExceeded` if max components/regions/queues exceeded
    pub fn mount(
        &mut self,
        manifest_bytes: &[u8],
        signature: &[u8],
        _package_bytes: &[u8], // WASM components - not used in Phase A
    ) -> Result<MountResult, KernelError> {
        // Step 1: Verify signature
        if self.config.verify_signatures {
            self.verify_signature(manifest_bytes, signature)?;
        }

        // Step 2: Parse manifest
        let manifest = RvfManifest::parse(manifest_bytes)?;

        // Validate manifest
        if !manifest.validate() {
            return Ok(MountResult::failure(RvfVerifyStatus::ManifestInvalid));
        }

        // Check limits
        if manifest.component_graph.component_count() > self.config.max_components {
            return Err(KernelError::LimitExceeded);
        }
        if manifest.memory_schema.region_count() > self.config.max_regions {
            return Err(KernelError::LimitExceeded);
        }
        if manifest.component_graph.wiring_count() > self.config.max_queues {
            return Err(KernelError::LimitExceeded);
        }

        // Allocate mount handle
        let mount_handle = RvfMountHandle::new(self.next_mount_id, 0);
        self.next_mount_id += 1;

        let mut result = MountResult::success(mount_handle);

        // Step 3: Create regions
        self.create_regions(&manifest, &mut result)?;

        // Step 4: Mount components (Phase A: mock)
        result.components_mounted = manifest.component_graph.component_count();

        // Step 5: Distribute capabilities (Phase A: mock)
        // In production, this creates actual capability entries

        // Step 6: Connect queues (Phase A: mock)
        result.queues_connected = manifest.component_graph.wiring_count();

        // Step 7: Spawn tasks
        self.spawn_initial_tasks(&manifest, &mut result)?;

        Ok(result)
    }

    fn verify_signature(&self, manifest: &[u8], signature: &[u8]) -> Result<(), KernelError> {
        match &self.verifier {
            Some(verifier) => {
                let result = verifier.verify(manifest, signature);
                if result.is_valid() {
                    Ok(())
                } else {
                    Err(KernelError::InvalidSignature)
                }
            }
            None => {
                // No verifier configured - require signature verification
                Err(KernelError::InternalError)
            }
        }
    }

    fn create_regions(
        &mut self,
        manifest: &RvfManifest,
        result: &mut MountResult,
    ) -> Result<(), KernelError> {
        let region_count = manifest.memory_schema.region_count();

        for _ in 0..region_count {
            let handle = RegionHandle::new(self.next_region_id, 0);
            self.next_region_id += 1;
            result.add_region(handle)?;
        }

        Ok(())
    }

    fn spawn_initial_tasks(
        &mut self,
        manifest: &RvfManifest,
        result: &mut MountResult,
    ) -> Result<(), KernelError> {
        let component_count = manifest.component_graph.component_count();

        // In production, we would check each component's entry point
        // and spawn a task if it has one. For Phase A, spawn one task
        // per component with an entry point.

        // Phase A: Spawn one task for the root component
        if component_count > 0 {
            let handle = TaskHandle::new(self.next_task_id, 0);
            self.next_task_id += 1;
            result.add_task(handle)?;
        }

        Ok(())
    }

    /// Unmounts an RVF package.
    ///
    /// # Steps
    ///
    /// 1. Stop all tasks
    /// 2. Disconnect queues
    /// 3. Revoke capabilities
    /// 4. Destroy regions
    /// 5. Free mount handle
    pub fn unmount(&mut self, _handle: RvfMountHandle) -> Result<(), KernelError> {
        // Phase A: Mock implementation
        // In production, this would perform the full unmount sequence
        Ok(())
    }
}

impl Default for RvfMount {
    fn default() -> Self {
        Self::new(MountConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signature::SIGNATURE_SIZE;

    fn create_test_manifest() -> Vec<u8> {
        let mut manifest = vec![0u8; 100];
        manifest[0..4].copy_from_slice(b"RVF1");
        manifest[4..6].copy_from_slice(&1u16.to_le_bytes()); // major
        manifest[6..8].copy_from_slice(&0u16.to_le_bytes()); // minor
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
    fn test_mount_config_default() {
        let config = MountConfig::default();
        assert_eq!(config.max_components, 256);
        assert_eq!(config.max_regions, 256);
        assert!(config.verify_signatures);
    }

    #[test]
    fn test_mount_result_success() {
        let handle = RvfMountHandle::new(1, 0);
        let result = MountResult::success(handle);

        assert_eq!(result.verify_status, RvfVerifyStatus::SignatureValid);
        assert!(!result.mount_handle.is_null());
    }

    #[test]
    fn test_mount_result_add_region() {
        let mut result = MountResult::success(RvfMountHandle::new(1, 0));

        result.add_region(RegionHandle::new(1, 0)).unwrap();
        result.add_region(RegionHandle::new(2, 0)).unwrap();

        assert_eq!(result.regions_created, 2);
        assert_eq!(result.region_handle_count, 2);
    }

    #[test]
    fn test_mount_result_add_task() {
        let mut result = MountResult::success(RvfMountHandle::new(1, 0));

        result.add_task(TaskHandle::new(1, 0)).unwrap();

        assert_eq!(result.tasks_spawned, 1);
        assert_eq!(result.task_handle_count, 1);
    }

    #[test]
    fn test_rvf_mount_basic() {
        let mut mount = RvfMount::new(MountConfig::default());
        mount.set_public_key(&[0u8; 1952]); // Test key

        let manifest = create_test_manifest();
        let signature = create_test_signature(&manifest);

        let result = mount.mount(&manifest, &signature, &[]).unwrap();

        assert_eq!(result.verify_status, RvfVerifyStatus::SignatureValid);
        assert!(!result.mount_handle.is_null());
    }

    #[test]
    fn test_rvf_mount_invalid_signature() {
        let mut mount = RvfMount::new(MountConfig::default());
        mount.set_public_key(&[0u8; 1952]);

        let manifest = create_test_manifest();
        let wrong_manifest = b"wrong manifest";
        let signature = create_test_signature(wrong_manifest);

        let result = mount.mount(&manifest, &signature, &[]);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), KernelError::InvalidSignature);
    }

    #[test]
    fn test_rvf_mount_skip_verification() {
        let mut config = MountConfig::default();
        config.verify_signatures = false;

        let mut mount = RvfMount::new(config);

        let manifest = create_test_manifest();
        let bad_signature = [0u8; 100]; // Wrong size, but we skip verification

        let result = mount.mount(&manifest, &bad_signature, &[]).unwrap();

        assert_eq!(result.verify_status, RvfVerifyStatus::SignatureValid);
    }

    #[test]
    fn test_rvf_unmount() {
        let mut mount = RvfMount::default();
        let handle = RvfMountHandle::new(1, 0);

        // Should not fail
        mount.unmount(handle).unwrap();
    }
}
