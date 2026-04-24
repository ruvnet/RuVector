//! HNSW (Hierarchical Navigable Small World) index for kernel vector stores.
//!
//! This module provides a slab-based HNSW implementation suitable for
//! kernel-resident vector stores. Unlike the userspace ruvector-core HNSW,
//! this uses fixed-size slab allocations for O(1) node management.
//!
//! # Design Principles
//!
//! - HNSW nodes are slab-allocated (fixed-size slots)
//! - No heap allocation during search
//! - Generation counters for use-after-free prevention
//! - Compatible with proof-gated mutations

use crate::Result;
use ruvix_region::backing::MemoryBacking;
use ruvix_region::slab::{SlabAllocator, SlotHandle};

/// Configuration for HNSW index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct HnswConfig {
    /// Maximum number of bidirectional links per node (M parameter).
    pub m: u16,

    /// Maximum links at layer 0 (typically 2*M).
    pub m0: u16,

    /// ef_construction parameter for index building.
    pub ef_construction: u16,

    /// ef_search parameter for query time.
    pub ef_search: u16,

    /// Maximum number of layers (derived from capacity).
    pub max_layers: u8,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            ef_search: 50,
            max_layers: 16,
        }
    }
}

impl HnswConfig {
    /// Creates a new HNSW configuration.
    #[inline]
    #[must_use]
    pub const fn new(m: u16, ef_construction: u16) -> Self {
        Self {
            m,
            m0: (m * 2) as u16,
            ef_construction,
            ef_search: 50,
            max_layers: 16,
        }
    }

    /// Sets the ef_search parameter.
    #[inline]
    #[must_use]
    pub const fn with_ef_search(mut self, ef_search: u16) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Calculates the slot size needed for HNSW nodes with this config.
    #[inline]
    #[must_use]
    pub const fn node_slot_size(&self) -> usize {
        // Base header + links array
        // Header: layer (1) + padding (3) + link_count (4) + vector_slot (8) = 16 bytes
        // Links: m0 * 8 bytes (slot handles) for layer 0, m * 8 for other layers
        // We allocate for worst case (layer 0)
        16 + (self.m0 as usize) * 8
    }
}

/// An HNSW graph node stored in a slab slot.
///
/// Each node contains:
/// - The layer this node exists on
/// - Links to other nodes at this layer
/// - Reference to the vector data slot
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct HnswNode {
    /// The layer this node is on (0 = bottom layer).
    pub layer: u8,

    /// Padding for alignment.
    _padding: [u8; 3],

    /// Number of links currently stored.
    pub link_count: u32,

    /// Handle to the vector data slot.
    pub vector_slot: SlotHandle,

    /// Links to other nodes (indices into the node slab).
    /// Only first `link_count` entries are valid.
    /// Size depends on layer (m0 for layer 0, m for others).
    pub links: [SlotHandle; 64], // Fixed max for simplicity
}

impl HnswNode {
    /// Creates a new HNSW node.
    #[inline]
    #[must_use]
    pub const fn new(layer: u8, vector_slot: SlotHandle) -> Self {
        Self {
            layer,
            _padding: [0; 3],
            link_count: 0,
            vector_slot,
            links: [SlotHandle::invalid(); 64],
        }
    }

    /// Returns the maximum number of links for this node.
    #[inline]
    #[must_use]
    pub const fn max_links(&self, config: &HnswConfig) -> u16 {
        if self.layer == 0 {
            config.m0
        } else {
            config.m
        }
    }

    /// Adds a link to another node.
    ///
    /// Returns `false` if the link array is full.
    pub fn add_link(&mut self, target: SlotHandle, config: &HnswConfig) -> bool {
        let max_links = self.max_links(config) as usize;
        if self.link_count as usize >= max_links {
            return false;
        }

        // Check for duplicates
        for i in 0..self.link_count as usize {
            if self.links[i] == target {
                return true; // Already linked
            }
        }

        self.links[self.link_count as usize] = target;
        self.link_count += 1;
        true
    }

    /// Removes a link to another node.
    ///
    /// Returns `true` if the link was found and removed.
    pub fn remove_link(&mut self, target: SlotHandle) -> bool {
        for i in 0..self.link_count as usize {
            if self.links[i] == target {
                // Swap with last and decrement count
                self.links[i] = self.links[self.link_count as usize - 1];
                self.links[self.link_count as usize - 1] = SlotHandle::invalid();
                self.link_count -= 1;
                return true;
            }
        }
        false
    }

    /// Returns the links as a slice.
    #[inline]
    #[must_use]
    pub fn links_slice(&self) -> &[SlotHandle] {
        &self.links[..self.link_count as usize]
    }
}

/// HNSW index stored in slab regions.
pub struct HnswRegion<B: MemoryBacking> {
    /// Slab allocator for HNSW nodes.
    node_slab: SlabAllocator<B>,

    /// Configuration for the HNSW index.
    config: HnswConfig,

    /// Entry point (handle to the top-level node, if any).
    entry_point: Option<SlotHandle>,

    /// Current maximum layer in the graph.
    current_max_layer: u8,

    /// Number of nodes in the index.
    node_count: u32,
}

impl<B: MemoryBacking> HnswRegion<B> {
    /// Creates a new HNSW region.
    ///
    /// # Arguments
    ///
    /// * `backing` - Memory backing for the slab
    /// * `config` - HNSW configuration
    /// * `capacity` - Maximum number of nodes
    pub fn new(backing: B, config: HnswConfig, capacity: usize) -> Result<Self> {
        let slot_size = core::mem::size_of::<HnswNode>();
        let node_slab = SlabAllocator::new(backing, slot_size, capacity)?;

        Ok(Self {
            node_slab,
            config,
            entry_point: None,
            current_max_layer: 0,
            node_count: 0,
        })
    }

    /// Allocates a new HNSW node.
    pub fn alloc_node(&mut self, layer: u8, vector_slot: SlotHandle) -> Result<SlotHandle> {
        let handle = self.node_slab.alloc()?;

        let node = HnswNode::new(layer, vector_slot);
        let node_bytes = unsafe {
            core::slice::from_raw_parts(
                &node as *const _ as *const u8,
                core::mem::size_of::<HnswNode>(),
            )
        };

        self.node_slab.write(handle, node_bytes)?;
        self.node_count += 1;

        // Update entry point if this is the first node or higher layer
        if self.entry_point.is_none() || layer > self.current_max_layer {
            self.entry_point = Some(handle);
            self.current_max_layer = layer;
        }

        Ok(handle)
    }

    /// Frees an HNSW node.
    pub fn free_node(&mut self, handle: SlotHandle) -> Result<()> {
        self.node_slab.free(handle)?;
        self.node_count = self.node_count.saturating_sub(1);

        // If we freed the entry point, we need to find a new one
        if self.entry_point == Some(handle) {
            self.entry_point = None;
            self.current_max_layer = 0;
        }

        Ok(())
    }

    /// Reads an HNSW node.
    pub fn read_node(&self, handle: SlotHandle) -> Result<HnswNode> {
        let mut bytes = [0u8; core::mem::size_of::<HnswNode>()];
        self.node_slab.read(handle, &mut bytes)?;

        // SAFETY: HnswNode is repr(C) and we've read the full size
        Ok(unsafe { core::ptr::read(bytes.as_ptr() as *const HnswNode) })
    }

    /// Writes an HNSW node.
    pub fn write_node(&mut self, handle: SlotHandle, node: &HnswNode) -> Result<()> {
        let node_bytes = unsafe {
            core::slice::from_raw_parts(
                node as *const _ as *const u8,
                core::mem::size_of::<HnswNode>(),
            )
        };
        self.node_slab.write(handle, node_bytes)?;
        Ok(())
    }

    /// Adds a link between two nodes.
    pub fn add_link(&mut self, from: SlotHandle, to: SlotHandle) -> Result<bool> {
        let mut node = self.read_node(from)?;
        let result = node.add_link(to, &self.config);
        self.write_node(from, &node)?;
        Ok(result)
    }

    /// Removes a link between two nodes.
    pub fn remove_link(&mut self, from: SlotHandle, to: SlotHandle) -> Result<bool> {
        let mut node = self.read_node(from)?;
        let result = node.remove_link(to);
        self.write_node(from, &node)?;
        Ok(result)
    }

    /// Returns the entry point handle.
    #[inline]
    #[must_use]
    pub const fn entry_point(&self) -> Option<SlotHandle> {
        self.entry_point
    }

    /// Returns the current maximum layer.
    #[inline]
    #[must_use]
    pub const fn current_max_layer(&self) -> u8 {
        self.current_max_layer
    }

    /// Returns the number of nodes in the index.
    #[inline]
    #[must_use]
    pub const fn node_count(&self) -> u32 {
        self.node_count
    }

    /// Returns the configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Checks if the index is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.node_count == 0
    }

    /// Returns the capacity.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.node_slab.slot_count()
    }

    /// Returns the number of free slots.
    #[inline]
    #[must_use]
    pub fn free_slots(&self) -> usize {
        self.node_slab.free_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_region::backing::StaticBacking;

    #[test]
    fn test_hnsw_config_defaults() {
        let config = HnswConfig::default();
        assert_eq!(config.m, 16);
        assert_eq!(config.m0, 32);
        assert_eq!(config.ef_construction, 200);
    }

    #[test]
    fn test_hnsw_node_links() {
        let config = HnswConfig::new(4, 100);
        let mut node = HnswNode::new(1, SlotHandle::new(0, 0));

        // Layer 1 should allow M links
        assert_eq!(node.max_links(&config), 4);

        // Add links
        assert!(node.add_link(SlotHandle::new(1, 0), &config));
        assert!(node.add_link(SlotHandle::new(2, 0), &config));
        assert!(node.add_link(SlotHandle::new(3, 0), &config));
        assert!(node.add_link(SlotHandle::new(4, 0), &config));

        // Should be full now
        assert!(!node.add_link(SlotHandle::new(5, 0), &config));

        // Remove a link
        assert!(node.remove_link(SlotHandle::new(2, 0)));
        assert_eq!(node.link_count, 3);

        // Can add again
        assert!(node.add_link(SlotHandle::new(5, 0), &config));
    }

    #[test]
    fn test_hnsw_region_alloc_free() {
        let backing = StaticBacking::<16384>::new();
        let config = HnswConfig::default();
        let mut region = HnswRegion::new(backing, config, 10).unwrap();

        assert!(region.is_empty());
        assert_eq!(region.node_count(), 0);

        // Allocate a node
        let vector_slot = SlotHandle::new(0, 0);
        let node_handle = region.alloc_node(0, vector_slot).unwrap();

        assert_eq!(region.node_count(), 1);
        assert_eq!(region.entry_point(), Some(node_handle));

        // Read the node back
        let node = region.read_node(node_handle).unwrap();
        assert_eq!(node.layer, 0);
        assert_eq!(node.vector_slot, vector_slot);

        // Free the node
        region.free_node(node_handle).unwrap();
        assert_eq!(region.node_count(), 0);
        assert!(region.entry_point().is_none());
    }

    #[test]
    fn test_hnsw_region_links() {
        let backing = StaticBacking::<16384>::new();
        let config = HnswConfig::default();
        let mut region = HnswRegion::new(backing, config, 10).unwrap();

        let vector_slot = SlotHandle::new(0, 0);
        let node1 = region.alloc_node(0, vector_slot).unwrap();
        let node2 = region.alloc_node(0, SlotHandle::new(1, 0)).unwrap();
        let node3 = region.alloc_node(0, SlotHandle::new(2, 0)).unwrap();

        // Add bidirectional links
        assert!(region.add_link(node1, node2).unwrap());
        assert!(region.add_link(node2, node1).unwrap());
        assert!(region.add_link(node1, node3).unwrap());
        assert!(region.add_link(node3, node1).unwrap());

        // Verify links
        let n1 = region.read_node(node1).unwrap();
        assert_eq!(n1.link_count, 2);
        assert!(n1.links_slice().contains(&node2));
        assert!(n1.links_slice().contains(&node3));
    }

    #[test]
    fn test_hnsw_region_entry_point_update() {
        let backing = StaticBacking::<16384>::new();
        let config = HnswConfig::default();
        let mut region = HnswRegion::new(backing, config, 10).unwrap();

        let vector_slot = SlotHandle::new(0, 0);

        // First node becomes entry point
        let node1 = region.alloc_node(0, vector_slot).unwrap();
        assert_eq!(region.entry_point(), Some(node1));
        assert_eq!(region.current_max_layer(), 0);

        // Higher layer node becomes new entry point
        let node2 = region.alloc_node(2, SlotHandle::new(1, 0)).unwrap();
        assert_eq!(region.entry_point(), Some(node2));
        assert_eq!(region.current_max_layer(), 2);

        // Lower layer node doesn't change entry point
        let _node3 = region.alloc_node(1, SlotHandle::new(2, 0)).unwrap();
        assert_eq!(region.entry_point(), Some(node2));
    }
}
