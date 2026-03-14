//! Derivation tree for capability revocation propagation.
//!
//! When a capability is derived (via cap_grant), a parent-child relationship
//! is established. Revoking a parent capability invalidates all derived
//! capabilities (children, grandchildren, etc.).

use crate::error::{CapError, CapResult};
use crate::DEFAULT_CAP_TABLE_CAPACITY;
use ruvix_types::CapHandle;

/// A node in the derivation tree.
///
/// Each node tracks its children for revocation propagation.
/// The tree structure allows efficient revocation of entire subtrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DerivationNode {
    /// The capability handle this node represents.
    pub handle: CapHandle,

    /// First child handle (head of child list).
    pub first_child: CapHandle,

    /// Next sibling handle (for child list traversal).
    pub next_sibling: CapHandle,

    /// Whether this node is valid (not revoked).
    pub is_valid: bool,

    /// Depth in the derivation tree (0 = root).
    pub depth: u8,
}

impl DerivationNode {
    /// Creates a new root node.
    #[inline]
    #[must_use]
    pub const fn new_root(handle: CapHandle) -> Self {
        Self {
            handle,
            first_child: CapHandle::null(),
            next_sibling: CapHandle::null(),
            is_valid: true,
            depth: 0,
        }
    }

    /// Creates a new child node.
    #[inline]
    #[must_use]
    pub const fn new_child(handle: CapHandle, depth: u8) -> Self {
        Self {
            handle,
            first_child: CapHandle::null(),
            next_sibling: CapHandle::null(),
            is_valid: true,
            depth,
        }
    }

    /// Returns true if this node has children.
    #[inline]
    #[must_use]
    pub const fn has_children(&self) -> bool {
        !self.first_child.is_null()
    }

    /// Returns true if this node has a sibling.
    #[inline]
    #[must_use]
    pub const fn has_sibling(&self) -> bool {
        !self.next_sibling.is_null()
    }
}

impl DerivationNode {
    /// Returns a const-compatible empty node.
    #[inline]
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            handle: CapHandle::null(),
            first_child: CapHandle::null(),
            next_sibling: CapHandle::null(),
            is_valid: false,
            depth: 0,
        }
    }
}

impl Default for DerivationNode {
    fn default() -> Self {
        Self::empty()
    }
}

/// Derivation tree manager.
///
/// Maintains the parent-child relationships between capabilities
/// for revocation propagation.
pub struct DerivationTree<const N: usize = DEFAULT_CAP_TABLE_CAPACITY> {
    /// Nodes indexed by capability handle ID.
    nodes: [DerivationNode; N],

    /// Number of active nodes.
    count: usize,
}

impl<const N: usize> DerivationTree<N> {
    /// Creates a new empty derivation tree.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: [DerivationNode::empty(); N],
            count: 0,
        }
    }

    /// Returns the number of active nodes.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the tree is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Registers a root capability in the tree.
    pub fn add_root(&mut self, handle: CapHandle) -> CapResult<()> {
        let index = handle.raw().id as usize;
        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        self.nodes[index] = DerivationNode::new_root(handle);
        self.count += 1;
        Ok(())
    }

    /// Registers a derived capability in the tree.
    ///
    /// Links the child to the parent's child list.
    pub fn add_child(
        &mut self,
        parent_handle: CapHandle,
        child_handle: CapHandle,
        depth: u8,
    ) -> CapResult<()> {
        let parent_index = parent_handle.raw().id as usize;
        let child_index = child_handle.raw().id as usize;

        if parent_index >= N || child_index >= N {
            return Err(CapError::InvalidHandle);
        }

        if !self.nodes[parent_index].is_valid {
            return Err(CapError::Revoked);
        }

        // Create the child node
        let mut child_node = DerivationNode::new_child(child_handle, depth);

        // Link to parent's child list
        child_node.next_sibling = self.nodes[parent_index].first_child;
        self.nodes[parent_index].first_child = child_handle;

        self.nodes[child_index] = child_node;
        self.count += 1;

        Ok(())
    }

    /// Revokes a capability and all its descendants.
    ///
    /// Returns the number of capabilities revoked.
    pub fn revoke(&mut self, handle: CapHandle) -> CapResult<usize> {
        let index = handle.raw().id as usize;
        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        if !self.nodes[index].is_valid {
            return Err(CapError::Revoked);
        }

        let count = self.revoke_subtree(handle);
        Ok(count)
    }

    /// Recursively revokes a subtree.
    fn revoke_subtree(&mut self, handle: CapHandle) -> usize {
        let index = handle.raw().id as usize;
        if index >= N || !self.nodes[index].is_valid {
            return 0;
        }

        let mut count = 1;
        self.nodes[index].is_valid = false;
        self.count = self.count.saturating_sub(1);

        // Revoke all children
        let mut child = self.nodes[index].first_child;
        while !child.is_null() {
            let child_index = child.raw().id as usize;
            if child_index < N {
                let next = self.nodes[child_index].next_sibling;
                count += self.revoke_subtree(child);
                child = next;
            } else {
                break;
            }
        }

        count
    }

    /// Looks up a node by handle.
    pub fn lookup(&self, handle: CapHandle) -> CapResult<&DerivationNode> {
        let index = handle.raw().id as usize;
        if index >= N {
            return Err(CapError::InvalidHandle);
        }

        let node = &self.nodes[index];
        if !node.is_valid {
            return Err(CapError::Revoked);
        }

        Ok(node)
    }

    /// Checks if a capability is still valid (not revoked).
    pub fn is_valid(&self, handle: CapHandle) -> bool {
        let index = handle.raw().id as usize;
        index < N && self.nodes[index].is_valid
    }

    /// Returns the depth of a capability in the derivation tree.
    pub fn depth(&self, handle: CapHandle) -> CapResult<u8> {
        self.lookup(handle).map(|n| n.depth)
    }

    /// Collects all handles that would be revoked if the given handle is revoked.
    ///
    /// This is useful for auditing and preview before actual revocation.
    #[cfg(feature = "alloc")]
    pub fn collect_descendants(&self, handle: CapHandle) -> alloc::vec::Vec<CapHandle> {
        let mut result = alloc::vec::Vec::new();
        self.collect_descendants_recursive(handle, &mut result);
        result
    }

    #[cfg(feature = "alloc")]
    fn collect_descendants_recursive(
        &self,
        handle: CapHandle,
        result: &mut alloc::vec::Vec<CapHandle>,
    ) {
        let index = handle.raw().id as usize;
        if index >= N || !self.nodes[index].is_valid {
            return;
        }

        result.push(handle);

        let mut child = self.nodes[index].first_child;
        while !child.is_null() {
            let child_index = child.raw().id as usize;
            if child_index < N {
                let next = self.nodes[child_index].next_sibling;
                self.collect_descendants_recursive(child, result);
                child = next;
            } else {
                break;
            }
        }
    }
}

impl<const N: usize> Default for DerivationTree<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivation_tree_root() {
        let mut tree = DerivationTree::<64>::new();
        let handle = CapHandle::new(0, 0);

        tree.add_root(handle).unwrap();
        assert_eq!(tree.len(), 1);
        assert!(tree.is_valid(handle));

        let node = tree.lookup(handle).unwrap();
        assert_eq!(node.depth, 0);
        assert!(!node.has_children());
    }

    #[test]
    fn test_derivation_tree_child() {
        let mut tree = DerivationTree::<64>::new();
        let parent = CapHandle::new(0, 0);
        let child = CapHandle::new(1, 0);

        tree.add_root(parent).unwrap();
        tree.add_child(parent, child, 1).unwrap();

        assert_eq!(tree.len(), 2);
        assert!(tree.is_valid(child));
        assert_eq!(tree.depth(child).unwrap(), 1);

        let parent_node = tree.lookup(parent).unwrap();
        assert!(parent_node.has_children());
    }

    #[test]
    fn test_derivation_tree_revoke() {
        let mut tree = DerivationTree::<64>::new();
        let root = CapHandle::new(0, 0);
        let child1 = CapHandle::new(1, 0);
        let child2 = CapHandle::new(2, 0);
        let grandchild = CapHandle::new(3, 0);

        tree.add_root(root).unwrap();
        tree.add_child(root, child1, 1).unwrap();
        tree.add_child(root, child2, 1).unwrap();
        tree.add_child(child1, grandchild, 2).unwrap();

        assert_eq!(tree.len(), 4);

        // Revoke the root - should revoke all descendants
        let revoked = tree.revoke(root).unwrap();
        assert_eq!(revoked, 4);
        assert_eq!(tree.len(), 0);

        assert!(!tree.is_valid(root));
        assert!(!tree.is_valid(child1));
        assert!(!tree.is_valid(child2));
        assert!(!tree.is_valid(grandchild));
    }

    #[test]
    fn test_derivation_tree_partial_revoke() {
        let mut tree = DerivationTree::<64>::new();
        let root = CapHandle::new(0, 0);
        let child1 = CapHandle::new(1, 0);
        let child2 = CapHandle::new(2, 0);
        let grandchild = CapHandle::new(3, 0);

        tree.add_root(root).unwrap();
        tree.add_child(root, child1, 1).unwrap();
        tree.add_child(root, child2, 1).unwrap();
        tree.add_child(child1, grandchild, 2).unwrap();

        // Revoke child1 - should revoke child1 and grandchild
        let revoked = tree.revoke(child1).unwrap();
        assert_eq!(revoked, 2);

        assert!(tree.is_valid(root));
        assert!(!tree.is_valid(child1));
        assert!(tree.is_valid(child2));
        assert!(!tree.is_valid(grandchild));
    }
}
