//! Initial capability set creation at boot.
//!
//! At kernel boot, the root task is created with a set of initial
//! capabilities granting access to all physical resources. These
//! capabilities form the basis of all subsequent derivation.
//!
//! # Initial Capabilities (from ADR-087 Section 6.3)
//!
//! 1. **Physical memory regions** - Full access to all physical RAM
//! 2. **Boot RVF package** - The verified firmware/kernel image
//! 3. **Kernel witness log** - For proof attestation
//! 4. **Root interrupt queue** - For interrupt handling

use ruvix_types::{CapRights, ObjectType};

/// Initial capability descriptor for boot-time creation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitialCapability {
    /// Object identifier.
    pub object_id: u64,

    /// Object type.
    pub object_type: ObjectType,

    /// Rights granted.
    pub rights: CapRights,

    /// Badge for identification.
    pub badge: u64,

    /// Description for debugging.
    pub description: &'static str,
}

impl InitialCapability {
    /// Creates a new initial capability descriptor.
    #[inline]
    #[must_use]
    pub const fn new(
        object_id: u64,
        object_type: ObjectType,
        rights: CapRights,
        badge: u64,
        description: &'static str,
    ) -> Self {
        Self {
            object_id,
            object_type,
            rights,
            badge,
            description,
        }
    }

    /// Creates a physical memory capability.
    #[inline]
    #[must_use]
    pub const fn memory(object_id: u64, start_addr: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Region,
            rights: CapRights::ALL,
            badge: start_addr,
            description: "Physical memory region",
        }
    }

    /// Creates a read-only physical memory capability (e.g., ROM).
    #[inline]
    #[must_use]
    pub const fn memory_readonly(object_id: u64, start_addr: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Region,
            // Read + Grant (can share) but no Write, Execute, or Revoke
            rights: CapRights::READ.union(CapRights::GRANT),
            badge: start_addr,
            description: "Read-only memory region (ROM)",
        }
    }

    /// Creates a boot RVF package capability.
    #[inline]
    #[must_use]
    pub const fn rvf_package(object_id: u64, package_hash_lo: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::RvfMount,
            rights: CapRights::READ.union(CapRights::EXECUTE).union(CapRights::PROVE),
            badge: package_hash_lo,
            description: "Boot RVF package",
        }
    }

    /// Creates a kernel witness log capability.
    #[inline]
    #[must_use]
    pub const fn witness_log(object_id: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::WitnessLog,
            rights: CapRights::ALL,
            badge: 0,
            description: "Kernel witness log",
        }
    }

    /// Creates a root interrupt queue capability.
    #[inline]
    #[must_use]
    pub const fn interrupt_queue(object_id: u64, irq_mask: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Queue,
            rights: CapRights::ALL,
            badge: irq_mask,
            description: "Root interrupt queue",
        }
    }

    /// Creates a root timer capability.
    #[inline]
    #[must_use]
    pub const fn timer(object_id: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Timer,
            rights: CapRights::ALL,
            badge: 0,
            description: "System timer",
        }
    }

    /// Creates a root task capability (self-reference).
    #[inline]
    #[must_use]
    pub const fn root_task(object_id: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::Task,
            rights: CapRights::ALL,
            badge: 0,
            description: "Root task (self)",
        }
    }

    /// Creates a vector store capability for neural memory.
    #[inline]
    #[must_use]
    pub const fn vector_store(object_id: u64, dimension: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::VectorStore,
            rights: CapRights::ALL,
            badge: dimension,
            description: "Neural vector store",
        }
    }

    /// Creates a proof graph capability (uses GraphStore for proof-related data).
    #[inline]
    #[must_use]
    pub const fn proof_graph(object_id: u64) -> Self {
        Self {
            object_id,
            object_type: ObjectType::GraphStore,
            rights: CapRights::ALL,
            badge: 0,
            description: "Proof verification graph",
        }
    }
}

/// The complete set of initial capabilities for the root task.
///
/// This struct describes all capabilities that should be created
/// during kernel boot and granted to the root task.
#[derive(Debug, Clone)]
pub struct BootCapabilitySet {
    /// Memory region capabilities.
    memory_regions: [Option<InitialCapability>; 16],
    memory_count: usize,

    /// RVF package capability.
    rvf_package: Option<InitialCapability>,

    /// Witness log capability.
    witness_log: Option<InitialCapability>,

    /// Interrupt queue capability.
    interrupt_queue: Option<InitialCapability>,

    /// Timer capability.
    timer: Option<InitialCapability>,

    /// Root task self-reference.
    root_task: Option<InitialCapability>,

    /// Vector store capability.
    vector_store: Option<InitialCapability>,

    /// Proof graph capability.
    proof_graph: Option<InitialCapability>,
}

impl BootCapabilitySet {
    /// Creates an empty boot capability set.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            memory_regions: [None; 16],
            memory_count: 0,
            rvf_package: None,
            witness_log: None,
            interrupt_queue: None,
            timer: None,
            root_task: None,
            vector_store: None,
            proof_graph: None,
        }
    }

    /// Creates a minimal boot set with essential capabilities.
    pub fn minimal(root_task_id: u64) -> Self {
        let mut set = Self::new();
        set.root_task = Some(InitialCapability::root_task(root_task_id));
        set
    }

    /// Creates a full boot set with all standard capabilities.
    pub fn full(
        root_task_id: u64,
        memory_start: u64,
        memory_size: u64,
        rvf_object_id: u64,
        rvf_hash: u64,
    ) -> Self {
        let mut set = Self::new();

        // Root task self-reference
        set.root_task = Some(InitialCapability::root_task(root_task_id));

        // Single memory region covering all physical RAM
        set.memory_regions[0] = Some(InitialCapability::memory(memory_start, memory_size));
        set.memory_count = 1;

        // Boot RVF package
        set.rvf_package = Some(InitialCapability::rvf_package(rvf_object_id, rvf_hash));

        // Witness log
        set.witness_log = Some(InitialCapability::witness_log(root_task_id + 1));

        // Interrupt queue (all IRQs enabled)
        set.interrupt_queue = Some(InitialCapability::interrupt_queue(
            root_task_id + 2,
            u64::MAX,
        ));

        // Timer
        set.timer = Some(InitialCapability::timer(root_task_id + 3));

        set
    }

    /// Adds a memory region capability.
    pub fn add_memory_region(&mut self, object_id: u64, start_addr: u64) -> bool {
        if self.memory_count >= 16 {
            return false;
        }

        self.memory_regions[self.memory_count] = Some(InitialCapability::memory(object_id, start_addr));
        self.memory_count += 1;
        true
    }

    /// Adds a read-only memory region (e.g., ROM).
    pub fn add_readonly_region(&mut self, object_id: u64, start_addr: u64) -> bool {
        if self.memory_count >= 16 {
            return false;
        }

        self.memory_regions[self.memory_count] = Some(InitialCapability::memory_readonly(object_id, start_addr));
        self.memory_count += 1;
        true
    }

    /// Sets the RVF package capability.
    #[inline]
    pub fn set_rvf_package(&mut self, object_id: u64, package_hash: u64) {
        self.rvf_package = Some(InitialCapability::rvf_package(object_id, package_hash));
    }

    /// Sets the witness log capability.
    #[inline]
    pub fn set_witness_log(&mut self, object_id: u64) {
        self.witness_log = Some(InitialCapability::witness_log(object_id));
    }

    /// Sets the interrupt queue capability.
    #[inline]
    pub fn set_interrupt_queue(&mut self, object_id: u64, irq_mask: u64) {
        self.interrupt_queue = Some(InitialCapability::interrupt_queue(object_id, irq_mask));
    }

    /// Sets the timer capability.
    #[inline]
    pub fn set_timer(&mut self, object_id: u64) {
        self.timer = Some(InitialCapability::timer(object_id));
    }

    /// Sets the root task capability.
    #[inline]
    pub fn set_root_task(&mut self, object_id: u64) {
        self.root_task = Some(InitialCapability::root_task(object_id));
    }

    /// Sets the vector store capability.
    #[inline]
    pub fn set_vector_store(&mut self, object_id: u64, dimension: u64) {
        self.vector_store = Some(InitialCapability::vector_store(object_id, dimension));
    }

    /// Sets the proof graph capability.
    #[inline]
    pub fn set_proof_graph(&mut self, object_id: u64) {
        self.proof_graph = Some(InitialCapability::proof_graph(object_id));
    }

    /// Returns the number of memory regions.
    #[inline]
    #[must_use]
    pub const fn memory_region_count(&self) -> usize {
        self.memory_count
    }

    /// Returns the total number of capabilities in the set.
    #[inline]
    #[must_use]
    pub fn total_count(&self) -> usize {
        let mut count = self.memory_count;

        if self.rvf_package.is_some() { count += 1; }
        if self.witness_log.is_some() { count += 1; }
        if self.interrupt_queue.is_some() { count += 1; }
        if self.timer.is_some() { count += 1; }
        if self.root_task.is_some() { count += 1; }
        if self.vector_store.is_some() { count += 1; }
        if self.proof_graph.is_some() { count += 1; }

        count
    }

    /// Returns an iterator over all capabilities in the set.
    pub fn iter(&self) -> impl Iterator<Item = &InitialCapability> {
        self.memory_regions[..self.memory_count]
            .iter()
            .filter_map(|c| c.as_ref())
            .chain(self.rvf_package.iter())
            .chain(self.witness_log.iter())
            .chain(self.interrupt_queue.iter())
            .chain(self.timer.iter())
            .chain(self.root_task.iter())
            .chain(self.vector_store.iter())
            .chain(self.proof_graph.iter())
    }
}

impl Default for BootCapabilitySet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_capability_memory() {
        let cap = InitialCapability::memory(0x1000, 0x8000_0000);

        assert_eq!(cap.object_id, 0x1000);
        assert_eq!(cap.object_type, ObjectType::Region);
        assert_eq!(cap.rights, CapRights::ALL);
        assert_eq!(cap.badge, 0x8000_0000);
    }

    #[test]
    fn test_initial_capability_rvf() {
        let cap = InitialCapability::rvf_package(0x2000, 0xDEADBEEF);

        assert_eq!(cap.object_type, ObjectType::RvfMount);
        assert!(cap.rights.contains(CapRights::READ));
        assert!(cap.rights.contains(CapRights::EXECUTE));
        assert!(cap.rights.contains(CapRights::PROVE));
        assert!(!cap.rights.contains(CapRights::WRITE));
    }

    #[test]
    fn test_boot_capability_set_minimal() {
        let set = BootCapabilitySet::minimal(1);

        assert_eq!(set.total_count(), 1);
        assert!(set.root_task.is_some());
    }

    #[test]
    fn test_boot_capability_set_full() {
        let set = BootCapabilitySet::full(
            1,          // root_task_id
            0x1000,     // memory_start
            0x10000,    // memory_size
            0x2000,     // rvf_object_id
            0xCAFE,     // rvf_hash
        );

        assert_eq!(set.memory_region_count(), 1);
        assert!(set.rvf_package.is_some());
        assert!(set.witness_log.is_some());
        assert!(set.interrupt_queue.is_some());
        assert!(set.timer.is_some());
        assert!(set.root_task.is_some());

        // Should have 6 capabilities total
        assert_eq!(set.total_count(), 6);
    }

    #[test]
    fn test_boot_capability_set_add_memory() {
        let mut set = BootCapabilitySet::new();

        assert!(set.add_memory_region(0x1000, 0x8000_0000));
        assert!(set.add_memory_region(0x1001, 0x9000_0000));

        assert_eq!(set.memory_region_count(), 2);
    }

    #[test]
    fn test_boot_capability_set_memory_limit() {
        let mut set = BootCapabilitySet::new();

        // Fill up all 16 slots
        for i in 0..16 {
            assert!(set.add_memory_region(i as u64, i as u64 * 0x1000_0000));
        }

        // 17th should fail
        assert!(!set.add_memory_region(16, 0));
    }

    #[test]
    fn test_boot_capability_set_iter() {
        let set = BootCapabilitySet::full(1, 0x1000, 0x10000, 0x2000, 0xCAFE);

        let mut count = 0;
        let mut first_type = None;
        for cap in set.iter() {
            if first_type.is_none() {
                first_type = Some(cap.object_type);
            }
            count += 1;
        }

        assert_eq!(count, 6);
        // First should be memory
        assert_eq!(first_type, Some(ObjectType::Region));
    }

    #[test]
    fn test_readonly_memory_region() {
        let cap = InitialCapability::memory_readonly(0x1000, 0x0000_0000);

        assert!(cap.rights.contains(CapRights::READ));
        assert!(cap.rights.contains(CapRights::GRANT));
        assert!(!cap.rights.contains(CapRights::WRITE));
        assert!(!cap.rights.contains(CapRights::EXECUTE));
    }
}
