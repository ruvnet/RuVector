//! Kernel-resident graph store implementation.
//!
//! Graph stores are kernel objects containing nodes and edges.
//! All mutations are proof-gated via the `graph_apply_proved` syscall.
//!
//! # Design (from ADR-087 Section 4.3)
//!
//! ```rust,ignore
//! pub struct KernelGraphStore {
//!     node_region: RegionHandle,       // slab region for graph nodes
//!     edge_region: RegionHandle,       // slab region for adjacency lists
//!     witness_region: RegionHandle,    // append-only mutation witness log
//!     partition_meta: PartitionMeta,   // MinCut partition metadata
//!     proof_policy: ProofPolicy,
//! }
//! ```
//!
//! # Supported Mutations
//!
//! - AddNode: Add a new node to the graph
//! - RemoveNode: Remove a node and all its edges
//! - AddEdge: Add a new edge between two nodes
//! - RemoveEdge: Remove an edge between two nodes
//! - UpdateEdgeWeight: Update the weight of an existing edge

use crate::coherence::{CoherenceConfig, CoherenceTracker};
use crate::proof_policy::{ProofPolicy, ProofVerifier};
use crate::witness::WitnessLog;
use crate::Result;

use ruvix_region::backing::MemoryBacking;
use ruvix_region::slab::{SlabAllocator, SlotHandle};
use ruvix_types::{
    CapRights, Capability, GraphHandle, GraphMutation, GraphMutationKind, KernelError,
    ProofAttestation, ProofToken, RegionHandle,
};

/// Maximum edges per node (for slab sizing).
const MAX_EDGES_PER_NODE: usize = 64;

/// Graph node stored in the node slab.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GraphNode {
    /// Node ID.
    pub id: u64,

    /// Number of outgoing edges.
    pub edge_count: u32,

    /// Partition ID for MinCut coherence.
    pub partition_id: u32,

    /// Handle to the edge list slot.
    pub edge_list_slot: SlotHandle,

    /// Coherence score for this node.
    pub coherence_score: u16,

    /// Mutation epoch.
    pub mutation_epoch: u64,

    /// Padding for alignment.
    _padding: [u8; 6],
}

impl GraphNode {
    /// Size of a graph node in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Creates a new graph node.
    #[inline]
    #[must_use]
    pub const fn new(id: u64, partition_id: u32) -> Self {
        Self {
            id,
            edge_count: 0,
            partition_id,
            edge_list_slot: SlotHandle::invalid(),
            coherence_score: 10000,
            mutation_epoch: 0,
            _padding: [0; 6],
        }
    }
}

/// Edge entry in an adjacency list.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct EdgeEntry {
    /// Target node ID.
    pub target_id: u64,

    /// Edge weight (fixed-point: weight * 10000).
    pub weight_fp: i32,

    /// Padding for alignment.
    _padding: [u8; 4],
}

impl EdgeEntry {
    /// Size of an edge entry in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Creates a new edge entry.
    #[inline]
    #[must_use]
    pub const fn new(target_id: u64, weight_fp: i32) -> Self {
        Self {
            target_id,
            weight_fp,
            _padding: [0; 4],
        }
    }

    /// Returns the weight as a float.
    #[inline]
    #[must_use]
    pub fn weight(&self) -> f32 {
        self.weight_fp as f32 / 10000.0
    }
}

/// Metadata for MinCut graph partitioning.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PartitionMeta {
    /// Number of partitions.
    pub partition_count: u32,

    /// Edges crossing partition boundaries (cut size).
    pub cut_size: u32,

    /// Average coherence across partitions.
    pub average_coherence: u16,

    /// Last partition update epoch.
    pub last_update_epoch: u64,
}

impl Default for PartitionMeta {
    fn default() -> Self {
        Self {
            partition_count: 1,
            cut_size: 0,
            average_coherence: 10000,
            last_update_epoch: 0,
        }
    }
}

impl PartitionMeta {
    /// Creates new partition metadata.
    #[inline]
    #[must_use]
    pub const fn new(partition_count: u32) -> Self {
        Self {
            partition_count,
            cut_size: 0,
            average_coherence: 10000,
            last_update_epoch: 0,
        }
    }
}

/// Result of a graph mutation.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphMutationResult {
    /// The attestation for this mutation.
    pub attestation: ProofAttestation,

    /// Whether the partition structure changed.
    pub partition_changed: bool,

    /// The affected partition IDs.
    pub affected_partitions: [u32; 2],
}

/// Node-to-slot mapping for graph lookup.
struct NodeMap {
    entries: [(u64, SlotHandle); 256],
    count: usize,
}

impl NodeMap {
    fn new() -> Self {
        Self {
            entries: [(0, SlotHandle::invalid()); 256],
            count: 0,
        }
    }

    fn get(&self, node_id: u64) -> Option<SlotHandle> {
        for i in 0..self.count {
            if self.entries[i].0 == node_id {
                return Some(self.entries[i].1);
            }
        }
        None
    }

    fn insert(&mut self, node_id: u64, handle: SlotHandle) -> Result<()> {
        // Check for existing
        for i in 0..self.count {
            if self.entries[i].0 == node_id {
                self.entries[i].1 = handle;
                return Ok(());
            }
        }

        if self.count >= 256 {
            return Err(KernelError::LimitExceeded);
        }

        self.entries[self.count] = (node_id, handle);
        self.count += 1;
        Ok(())
    }

    fn remove(&mut self, node_id: u64) -> Option<SlotHandle> {
        for i in 0..self.count {
            if self.entries[i].0 == node_id {
                let handle = self.entries[i].1;
                self.entries[i] = self.entries[self.count - 1];
                self.entries[self.count - 1] = (0, SlotHandle::invalid());
                self.count -= 1;
                return Some(handle);
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.count
    }
}

/// Builder for creating kernel graph stores.
pub struct GraphStoreBuilder {
    capacity: u32,
    coherence_config: CoherenceConfig,
    proof_policy: ProofPolicy,
    partition_count: u32,
}

impl GraphStoreBuilder {
    /// Creates a new graph store builder.
    #[inline]
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            coherence_config: CoherenceConfig::default(),
            proof_policy: ProofPolicy::standard(),
            partition_count: 1,
        }
    }

    /// Sets the coherence configuration.
    #[inline]
    #[must_use]
    pub fn with_coherence_config(mut self, config: CoherenceConfig) -> Self {
        self.coherence_config = config;
        self
    }

    /// Sets the proof policy.
    #[inline]
    #[must_use]
    pub fn with_proof_policy(mut self, policy: ProofPolicy) -> Self {
        self.proof_policy = policy;
        self
    }

    /// Sets the initial partition count.
    #[inline]
    #[must_use]
    pub fn with_partitions(mut self, count: u32) -> Self {
        self.partition_count = count;
        self
    }

    /// Builds the kernel graph store.
    pub fn build<B: MemoryBacking>(
        self,
        node_backing: B,
        edge_backing: B,
        witness_backing: B,
        node_handle: RegionHandle,
        edge_handle: RegionHandle,
        witness_handle: RegionHandle,
        store_id: u32,
    ) -> Result<KernelGraphStore<B>> {
        KernelGraphStore::new(
            node_backing,
            edge_backing,
            witness_backing,
            node_handle,
            edge_handle,
            witness_handle,
            self.capacity,
            self.coherence_config,
            self.proof_policy,
            self.partition_count,
            store_id,
        )
    }
}

/// Kernel-resident graph store.
///
/// Implements the `graph_apply_proved` syscall interface.
pub struct KernelGraphStore<B: MemoryBacking> {
    /// Slab region for graph nodes.
    node_slab: SlabAllocator<B>,

    /// Slab region for edge lists.
    edge_slab: SlabAllocator<B>,

    /// Append-only witness log.
    witness_log: WitnessLog<B>,

    /// Region handles for capability checking.
    node_handle: RegionHandle,
    edge_handle: RegionHandle,

    /// Node ID to slot mapping.
    node_map: NodeMap,

    /// Coherence tracker.
    coherence_tracker: CoherenceTracker,

    /// Proof verifier.
    proof_verifier: ProofVerifier,

    /// Partition metadata.
    partition_meta: PartitionMeta,

    /// Maximum capacity.
    capacity: u32,

    /// Store identifier.
    store_id: u32,

    /// Store handle.
    handle: GraphHandle,

    /// Total edge count.
    edge_count: u32,
}

impl<B: MemoryBacking> KernelGraphStore<B> {
    /// Creates a new kernel graph store.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        node_backing: B,
        edge_backing: B,
        witness_backing: B,
        node_handle: RegionHandle,
        edge_handle: RegionHandle,
        witness_handle: RegionHandle,
        capacity: u32,
        coherence_config: CoherenceConfig,
        proof_policy: ProofPolicy,
        partition_count: u32,
        store_id: u32,
    ) -> Result<Self> {
        let node_slab = SlabAllocator::new(node_backing, GraphNode::SIZE, capacity as usize)?;

        // Edge list slot size: header (4 bytes count) + MAX_EDGES_PER_NODE edges
        let edge_slot_size = 4 + MAX_EDGES_PER_NODE * EdgeEntry::SIZE;
        let edge_slab = SlabAllocator::new(edge_backing, edge_slot_size, capacity as usize)?;

        let witness_log = WitnessLog::new(
            witness_backing,
            capacity as usize * 4, // Allow multiple mutations per node
            witness_handle,
            store_id,
        )?;

        Ok(Self {
            node_slab,
            edge_slab,
            witness_log,
            node_handle,
            edge_handle,
            node_map: NodeMap::new(),
            coherence_tracker: CoherenceTracker::new(coherence_config),
            proof_verifier: ProofVerifier::new(proof_policy),
            partition_meta: PartitionMeta::new(partition_count),
            capacity,
            store_id,
            handle: GraphHandle::new(store_id, 0),
            edge_count: 0,
        })
    }

    /// Returns the store handle.
    #[inline]
    #[must_use]
    pub const fn handle(&self) -> GraphHandle {
        self.handle
    }

    /// Returns the number of nodes.
    #[inline]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_map.len()
    }

    /// Returns the number of edges.
    #[inline]
    #[must_use]
    pub const fn edge_count(&self) -> u32 {
        self.edge_count
    }

    /// Returns the capacity.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Returns the partition metadata.
    #[inline]
    #[must_use]
    pub const fn partition_meta(&self) -> &PartitionMeta {
        &self.partition_meta
    }

    /// Implements the `graph_apply_proved` syscall.
    ///
    /// Applies a graph mutation with proof verification.
    /// Requires PROVE right on the capability.
    ///
    /// # Arguments
    ///
    /// * `mutation` - The graph mutation to apply
    /// * `proof` - The proof token authorizing the mutation
    /// * `capability` - Capability authorizing the mutation
    /// * `current_time_ns` - Current time for proof verification
    ///
    /// # Returns
    ///
    /// A mutation result including the attestation.
    pub fn graph_apply_proved(
        &mut self,
        mutation: &GraphMutation,
        proof: &ProofToken,
        capability: &Capability,
        current_time_ns: u64,
    ) -> Result<GraphMutationResult> {
        // Compute mutation hash
        let mutation_hash = compute_graph_mutation_hash(mutation);

        // Verify proof
        let attestation =
            self.proof_verifier
                .verify(proof, &mutation_hash, current_time_ns, capability)?;

        // Apply mutation based on kind
        let (partition_changed, affected_partitions) = match mutation.kind {
            GraphMutationKind::AddNode => self.apply_add_node(mutation.node_a)?,
            GraphMutationKind::RemoveNode => self.apply_remove_node(mutation.node_a)?,
            GraphMutationKind::AddEdge => {
                self.apply_add_edge(mutation.node_a, mutation.node_b, mutation.weight_fp)?
            }
            GraphMutationKind::RemoveEdge => {
                self.apply_remove_edge(mutation.node_a, mutation.node_b)?
            }
            GraphMutationKind::UpdateEdgeWeight => {
                self.apply_update_edge_weight(mutation.node_a, mutation.node_b, mutation.weight_fp)?
            }
            GraphMutationKind::UpdateNodeMeta => {
                // Update node metadata (e.g., partition hint)
                self.apply_update_node_meta(mutation.node_a, mutation.partition_hint)?
            }
        };

        // Record in witness log
        self.witness_log
            .record_graph_mutation(mutation, attestation, current_time_ns)?;

        Ok(GraphMutationResult {
            attestation,
            partition_changed,
            affected_partitions,
        })
    }

    /// Applies an AddNode mutation.
    fn apply_add_node(&mut self, node_id: u64) -> Result<(bool, [u32; 2])> {
        // Check if node already exists
        if self.node_map.get(node_id).is_some() {
            return Err(KernelError::AlreadyExists);
        }

        // Allocate node slot
        let node_slot = self.node_slab.alloc()?;

        // Allocate edge list slot
        let edge_slot = self.edge_slab.alloc()?;

        // Create node
        let mut node = GraphNode::new(node_id, 0);
        node.edge_list_slot = edge_slot;
        node.mutation_epoch = self.coherence_tracker.advance_epoch();

        // Write node
        self.write_node(node_slot, &node)?;

        // Initialize edge list (empty)
        self.write_edge_count(edge_slot, 0)?;

        // Update map
        self.node_map.insert(node_id, node_slot)?;

        // Track coherence
        self.coherence_tracker.on_entry_added(node.coherence_score);

        Ok((false, [0, 0]))
    }

    /// Applies a RemoveNode mutation.
    fn apply_remove_node(&mut self, node_id: u64) -> Result<(bool, [u32; 2])> {
        // Find node
        let node_slot = self.node_map.get(node_id).ok_or(KernelError::NotFound)?;

        // Read node
        let node = self.read_node(node_slot)?;

        // Free edge list
        if !node.edge_list_slot.is_invalid() {
            // Count edges being removed
            let edge_count = self.read_edge_count(node.edge_list_slot)?;
            self.edge_count = self.edge_count.saturating_sub(edge_count);

            self.edge_slab.free(node.edge_list_slot)?;
        }

        // Free node
        self.node_slab.free(node_slot)?;

        // Update map
        self.node_map.remove(node_id);

        // Track coherence
        self.coherence_tracker
            .on_entry_removed(node.coherence_score);

        Ok((true, [node.partition_id, 0]))
    }

    /// Applies an AddEdge mutation.
    fn apply_add_edge(
        &mut self,
        from_id: u64,
        to_id: u64,
        weight_fp: i32,
    ) -> Result<(bool, [u32; 2])> {
        // Find source node
        let from_slot = self.node_map.get(from_id).ok_or(KernelError::NotFound)?;

        // Verify target exists
        if self.node_map.get(to_id).is_none() {
            return Err(KernelError::NotFound);
        }

        // Read source node
        let mut node = self.read_node(from_slot)?;

        // Read edge list
        let edge_count = self.read_edge_count(node.edge_list_slot)?;

        if edge_count as usize >= MAX_EDGES_PER_NODE {
            return Err(KernelError::LimitExceeded);
        }

        // Check for duplicate edge
        for i in 0..edge_count {
            let edge = self.read_edge(node.edge_list_slot, i)?;
            if edge.target_id == to_id {
                return Err(KernelError::AlreadyExists);
            }
        }

        // Add edge
        let edge = EdgeEntry::new(to_id, weight_fp);
        self.write_edge(node.edge_list_slot, edge_count, &edge)?;
        self.write_edge_count(node.edge_list_slot, edge_count + 1)?;

        // Update node
        node.edge_count = edge_count + 1;
        node.mutation_epoch = self.coherence_tracker.advance_epoch();
        self.write_node(from_slot, &node)?;

        self.edge_count += 1;

        // Check if edge crosses partitions
        let to_slot = self.node_map.get(to_id).unwrap();
        let to_node = self.read_node(to_slot)?;

        let partition_changed = node.partition_id != to_node.partition_id;
        if partition_changed {
            self.partition_meta.cut_size += 1;
        }

        Ok((partition_changed, [node.partition_id, to_node.partition_id]))
    }

    /// Applies a RemoveEdge mutation.
    fn apply_remove_edge(&mut self, from_id: u64, to_id: u64) -> Result<(bool, [u32; 2])> {
        // Find source node
        let from_slot = self.node_map.get(from_id).ok_or(KernelError::NotFound)?;

        // Read source node
        let mut node = self.read_node(from_slot)?;

        // Find and remove edge
        let edge_count = self.read_edge_count(node.edge_list_slot)?;
        let mut found_idx = None;

        for i in 0..edge_count {
            let edge = self.read_edge(node.edge_list_slot, i)?;
            if edge.target_id == to_id {
                found_idx = Some(i);
                break;
            }
        }

        let idx = found_idx.ok_or(KernelError::NotFound)?;

        // Swap with last and decrement count
        if idx < edge_count - 1 {
            let last_edge = self.read_edge(node.edge_list_slot, edge_count - 1)?;
            self.write_edge(node.edge_list_slot, idx, &last_edge)?;
        }
        self.write_edge_count(node.edge_list_slot, edge_count - 1)?;

        // Update node
        node.edge_count = edge_count - 1;
        node.mutation_epoch = self.coherence_tracker.advance_epoch();
        self.write_node(from_slot, &node)?;

        self.edge_count = self.edge_count.saturating_sub(1);

        Ok((false, [node.partition_id, 0]))
    }

    /// Applies an UpdateEdgeWeight mutation.
    fn apply_update_edge_weight(
        &mut self,
        from_id: u64,
        to_id: u64,
        weight_fp: i32,
    ) -> Result<(bool, [u32; 2])> {
        // Find source node
        let from_slot = self.node_map.get(from_id).ok_or(KernelError::NotFound)?;

        // Read source node
        let mut node = self.read_node(from_slot)?;

        // Find edge
        let edge_count = self.read_edge_count(node.edge_list_slot)?;

        for i in 0..edge_count {
            let mut edge = self.read_edge(node.edge_list_slot, i)?;
            if edge.target_id == to_id {
                edge.weight_fp = weight_fp;
                self.write_edge(node.edge_list_slot, i, &edge)?;

                // Update node epoch
                node.mutation_epoch = self.coherence_tracker.advance_epoch();
                self.write_node(from_slot, &node)?;

                return Ok((false, [node.partition_id, 0]));
            }
        }

        Err(KernelError::NotFound)
    }

    /// Applies an UpdateNodeMeta mutation.
    fn apply_update_node_meta(
        &mut self,
        node_id: u64,
        new_partition: u32,
    ) -> Result<(bool, [u32; 2])> {
        // Find node
        let node_slot = self.node_map.get(node_id).ok_or(KernelError::NotFound)?;

        // Read node
        let mut node = self.read_node(node_slot)?;
        let old_partition = node.partition_id;

        // Update partition
        node.partition_id = new_partition;
        node.mutation_epoch = self.coherence_tracker.advance_epoch();

        self.write_node(node_slot, &node)?;

        let partition_changed = old_partition != new_partition;
        Ok((partition_changed, [old_partition, new_partition]))
    }

    /// Reads a node from the slab.
    fn read_node(&self, slot: SlotHandle) -> Result<GraphNode> {
        let mut bytes = [0u8; GraphNode::SIZE];
        self.node_slab.read(slot, &mut bytes)?;
        Ok(unsafe { core::ptr::read(bytes.as_ptr() as *const GraphNode) })
    }

    /// Writes a node to the slab.
    fn write_node(&mut self, slot: SlotHandle, node: &GraphNode) -> Result<()> {
        let bytes =
            unsafe { core::slice::from_raw_parts(node as *const _ as *const u8, GraphNode::SIZE) };
        self.node_slab.write(slot, bytes)?;
        Ok(())
    }

    /// Reads the edge count from an edge list slot.
    fn read_edge_count(&self, slot: SlotHandle) -> Result<u32> {
        let mut bytes = [0u8; 4];
        let ptr = self.edge_slab.slot_ptr(slot)?;
        unsafe {
            core::ptr::copy_nonoverlapping(ptr, bytes.as_mut_ptr(), 4);
        }
        Ok(u32::from_le_bytes(bytes))
    }

    /// Writes the edge count to an edge list slot.
    fn write_edge_count(&mut self, slot: SlotHandle, count: u32) -> Result<()> {
        let bytes = count.to_le_bytes();
        let ptr = self.edge_slab.slot_ptr(slot)?;
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, 4);
        }
        Ok(())
    }

    /// Reads an edge from an edge list slot.
    fn read_edge(&self, slot: SlotHandle, index: u32) -> Result<EdgeEntry> {
        let offset = 4 + (index as usize) * EdgeEntry::SIZE;
        let ptr = self.edge_slab.slot_ptr(slot)?;
        let mut bytes = [0u8; EdgeEntry::SIZE];
        unsafe {
            core::ptr::copy_nonoverlapping(ptr.add(offset), bytes.as_mut_ptr(), EdgeEntry::SIZE);
        }
        Ok(unsafe { core::ptr::read(bytes.as_ptr() as *const EdgeEntry) })
    }

    /// Writes an edge to an edge list slot.
    fn write_edge(&mut self, slot: SlotHandle, index: u32, edge: &EdgeEntry) -> Result<()> {
        let offset = 4 + (index as usize) * EdgeEntry::SIZE;
        let ptr = self.edge_slab.slot_ptr(slot)?;
        let bytes = unsafe {
            core::slice::from_raw_parts(edge as *const _ as *const u8, EdgeEntry::SIZE)
        };
        unsafe {
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), EdgeEntry::SIZE);
        }
        Ok(())
    }

    /// Returns the proof policy.
    #[inline]
    #[must_use]
    pub const fn proof_policy(&self) -> &ProofPolicy {
        self.proof_verifier.policy()
    }

    /// Returns witness log statistics.
    #[inline]
    #[must_use]
    pub fn witness_entry_count(&self) -> usize {
        self.witness_log.entry_count()
    }
}

/// Computes a hash of a graph mutation for proof verification.
fn compute_graph_mutation_hash(mutation: &GraphMutation) -> [u8; 32] {
    let mut hash = [0u8; 32];

    hash[0] = mutation.kind as u8;
    hash[1..9].copy_from_slice(&mutation.node_a.to_le_bytes());
    hash[9..17].copy_from_slice(&mutation.node_b.to_le_bytes());
    hash[17..21].copy_from_slice(&mutation.weight_fp.to_le_bytes());
    hash[21..25].copy_from_slice(&mutation.partition_hint.to_le_bytes());

    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_region::backing::StaticBacking;
    use ruvix_types::{ObjectType, ProofPayload, ProofTier};

    fn create_test_capability() -> Capability {
        Capability::new(
            1,
            ObjectType::GraphStore,
            CapRights::READ | CapRights::WRITE | CapRights::PROVE,
            0,
            1,
        )
    }

    fn create_test_proof(mutation: &GraphMutation, valid_until_ns: u64, nonce: u64) -> ProofToken {
        let mutation_hash = compute_graph_mutation_hash(mutation);
        ProofToken::new(
            mutation_hash,
            ProofTier::Standard,
            ProofPayload::Hash { hash: mutation_hash },
            valid_until_ns,
            nonce,
        )
    }

    fn create_test_store() -> KernelGraphStore<StaticBacking<16384>> {
        let node_backing = StaticBacking::<16384>::new();
        let edge_backing = StaticBacking::<16384>::new();
        let witness_backing = StaticBacking::<16384>::new();

        GraphStoreBuilder::new(10) // Small capacity for tests
            .with_proof_policy(ProofPolicy::reflex())
            .build(
                node_backing,
                edge_backing,
                witness_backing,
                RegionHandle::new(1, 0),
                RegionHandle::new(2, 0),
                RegionHandle::new(3, 0),
                1,
            )
            .unwrap()
    }

    #[test]
    fn test_graph_store_creation() {
        let store = create_test_store();
        assert_eq!(store.node_count(), 0);
        assert_eq!(store.edge_count(), 0);
        assert_eq!(store.capacity(), 10);
    }

    #[test]
    fn test_add_node() {
        let mut store = create_test_store();
        let cap = create_test_capability();

        let mutation = GraphMutation::add_node(1);
        let proof = create_test_proof(&mutation, 1_000_000_000, 1);

        let result = store
            .graph_apply_proved(&mutation, &proof, &cap, 500_000_000)
            .unwrap();

        assert!(!result.partition_changed);
        assert_eq!(store.node_count(), 1);
    }

    #[test]
    fn test_add_duplicate_node() {
        let mut store = create_test_store();
        let cap = create_test_capability();

        // Add first node
        let mutation1 = GraphMutation::add_node(1);
        let proof1 = create_test_proof(&mutation1, 1_000_000_000, 1);
        store
            .graph_apply_proved(&mutation1, &proof1, &cap, 500_000_000)
            .unwrap();

        // Try to add duplicate
        let mutation2 = GraphMutation::add_node(1);
        let proof2 = create_test_proof(&mutation2, 1_000_000_000, 2);
        let result = store.graph_apply_proved(&mutation2, &proof2, &cap, 500_000_001);

        assert_eq!(result, Err(KernelError::AlreadyExists));
    }

    #[test]
    fn test_add_edge() {
        let mut store = create_test_store();
        let cap = create_test_capability();

        // Add nodes
        let add1 = GraphMutation::add_node(1);
        let proof1 = create_test_proof(&add1, 1_000_000_000, 1);
        store
            .graph_apply_proved(&add1, &proof1, &cap, 500_000_000)
            .unwrap();

        let add2 = GraphMutation::add_node(2);
        let proof2 = create_test_proof(&add2, 1_000_000_000, 2);
        store
            .graph_apply_proved(&add2, &proof2, &cap, 500_000_001)
            .unwrap();

        // Add edge
        let edge = GraphMutation::add_edge(1, 2, 0.75);
        let proof3 = create_test_proof(&edge, 1_000_000_000, 3);
        store
            .graph_apply_proved(&edge, &proof3, &cap, 500_000_002)
            .unwrap();

        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_remove_edge() {
        let mut store = create_test_store();
        let cap = create_test_capability();

        // Setup: add nodes and edge
        let add1 = GraphMutation::add_node(1);
        let add2 = GraphMutation::add_node(2);
        let edge = GraphMutation::add_edge(1, 2, 0.75);

        store
            .graph_apply_proved(
                &add1,
                &create_test_proof(&add1, 1_000_000_000, 1),
                &cap,
                500_000_000,
            )
            .unwrap();
        store
            .graph_apply_proved(
                &add2,
                &create_test_proof(&add2, 1_000_000_000, 2),
                &cap,
                500_000_001,
            )
            .unwrap();
        store
            .graph_apply_proved(
                &edge,
                &create_test_proof(&edge, 1_000_000_000, 3),
                &cap,
                500_000_002,
            )
            .unwrap();

        assert_eq!(store.edge_count(), 1);

        // Remove edge
        let remove = GraphMutation::remove_edge(1, 2);
        let proof = create_test_proof(&remove, 1_000_000_000, 4);
        store
            .graph_apply_proved(&remove, &proof, &cap, 500_000_003)
            .unwrap();

        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn test_update_edge_weight() {
        let mut store = create_test_store();
        let cap = create_test_capability();

        // Setup
        let add1 = GraphMutation::add_node(1);
        let add2 = GraphMutation::add_node(2);
        let edge = GraphMutation::add_edge(1, 2, 0.5);

        store
            .graph_apply_proved(
                &add1,
                &create_test_proof(&add1, 1_000_000_000, 1),
                &cap,
                500_000_000,
            )
            .unwrap();
        store
            .graph_apply_proved(
                &add2,
                &create_test_proof(&add2, 1_000_000_000, 2),
                &cap,
                500_000_001,
            )
            .unwrap();
        store
            .graph_apply_proved(
                &edge,
                &create_test_proof(&edge, 1_000_000_000, 3),
                &cap,
                500_000_002,
            )
            .unwrap();

        // Update weight
        let update = GraphMutation::update_edge_weight(1, 2, 0.9);
        let proof = create_test_proof(&update, 1_000_000_000, 4);
        store
            .graph_apply_proved(&update, &proof, &cap, 500_000_003)
            .unwrap();

        // Edge count unchanged
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_remove_node() {
        let mut store = create_test_store();
        let cap = create_test_capability();

        // Add node
        let add = GraphMutation::add_node(1);
        let proof1 = create_test_proof(&add, 1_000_000_000, 1);
        store
            .graph_apply_proved(&add, &proof1, &cap, 500_000_000)
            .unwrap();

        assert_eq!(store.node_count(), 1);

        // Remove node
        let remove = GraphMutation::remove_node(1);
        let proof2 = create_test_proof(&remove, 1_000_000_000, 2);
        let result = store
            .graph_apply_proved(&remove, &proof2, &cap, 500_000_001)
            .unwrap();

        assert!(result.partition_changed);
        assert_eq!(store.node_count(), 0);
    }

    #[test]
    fn test_proof_rejected() {
        let mut store = create_test_store();
        let cap = create_test_capability();

        let mutation = GraphMutation::add_node(1);

        // Wrong proof hash
        let wrong_proof = ProofToken::new(
            [0u8; 32],
            ProofTier::Standard,
            ProofPayload::Hash { hash: [0u8; 32] },
            1_000_000_000,
            1,
        );

        let result = store.graph_apply_proved(&mutation, &wrong_proof, &cap, 500_000_000);

        assert_eq!(result, Err(KernelError::ProofRejected));
        assert_eq!(store.node_count(), 0);
    }

    #[test]
    fn test_witness_log_recording() {
        let mut store = create_test_store();
        let cap = create_test_capability();

        // Perform multiple mutations
        for i in 0..5 {
            let mutation = GraphMutation::add_node(i);
            let proof = create_test_proof(&mutation, 1_000_000_000, i);
            store
                .graph_apply_proved(&mutation, &proof, &cap, 500_000_000 + i)
                .unwrap();
        }

        assert_eq!(store.witness_entry_count(), 5);
    }
}
