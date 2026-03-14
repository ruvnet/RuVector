//! FDT node representation.

use crate::{property::RegIter, DtbError, DtbResult, RegEntry, MAX_PATH_LEN};

/// A node in the device tree.
///
/// This is a lightweight reference to a node's location in the structure block.
#[derive(Debug, Clone, Copy)]
pub struct Node<'a> {
    /// Node name
    name: &'a str,
    /// Offset into the structure block where this node starts (after BeginNode token and name)
    content_offset: usize,
    /// Depth in the tree (0 = root)
    depth: usize,
    /// Parent node offset (0 for root)
    parent_offset: usize,
}

impl<'a> Node<'a> {
    /// Create a new node reference.
    #[must_use]
    pub const fn new(
        name: &'a str,
        content_offset: usize,
        depth: usize,
        parent_offset: usize,
    ) -> Self {
        Self {
            name,
            content_offset,
            depth,
            parent_offset,
        }
    }

    /// Get the node name.
    #[must_use]
    pub const fn name(&self) -> &str {
        self.name
    }

    /// Get the unit name (name without unit address).
    ///
    /// For "uart@10000", returns "uart".
    #[must_use]
    pub fn unit_name(&self) -> &str {
        self.name
            .find('@')
            .map_or(self.name, |idx| &self.name[..idx])
    }

    /// Get the unit address (address part of name).
    ///
    /// For "uart@10000", returns Some("10000").
    /// For "memory", returns None.
    #[must_use]
    pub fn unit_address(&self) -> Option<&str> {
        self.name.find('@').map(|idx| &self.name[idx + 1..])
    }

    /// Get the content offset.
    #[must_use]
    pub const fn content_offset(&self) -> usize {
        self.content_offset
    }

    /// Get the depth in the tree.
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.depth
    }

    /// Check if this is the root node.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.depth == 0
    }

    /// Get the parent offset.
    #[must_use]
    pub const fn parent_offset(&self) -> usize {
        self.parent_offset
    }
}

/// Builder for constructing node paths.
#[derive(Debug, Clone)]
pub struct PathBuilder {
    /// Path buffer
    path: [u8; MAX_PATH_LEN],
    /// Current length
    len: usize,
}

impl PathBuilder {
    /// Create a new path builder starting at root.
    #[must_use]
    pub const fn new() -> Self {
        let mut path = [0u8; MAX_PATH_LEN];
        path[0] = b'/';
        Self { path, len: 1 }
    }

    /// Create a path builder from a path string.
    pub fn from_str(s: &str) -> DtbResult<Self> {
        if s.len() >= MAX_PATH_LEN {
            return Err(DtbError::PathTooLong { length: s.len() });
        }

        let mut builder = Self::new();
        builder.len = s.len();
        builder.path[..s.len()].copy_from_slice(s.as_bytes());
        Ok(builder)
    }

    /// Append a path component.
    pub fn push(&mut self, component: &str) -> DtbResult<()> {
        // Calculate new length (add 1 for separator if not at root)
        let sep_len = if self.len == 1 { 0 } else { 1 };
        let new_len = self.len + sep_len + component.len();

        if new_len >= MAX_PATH_LEN {
            return Err(DtbError::PathTooLong { length: new_len });
        }

        // Add separator
        if self.len > 1 {
            self.path[self.len] = b'/';
            self.len += 1;
        }

        // Add component
        self.path[self.len..self.len + component.len()].copy_from_slice(component.as_bytes());
        self.len += component.len();

        Ok(())
    }

    /// Pop the last path component.
    pub fn pop(&mut self) {
        if self.len <= 1 {
            return;
        }

        // Find last separator
        if let Some(pos) = self.path[..self.len].iter().rposition(|&b| b == b'/') {
            self.len = if pos == 0 { 1 } else { pos };
        }
    }

    /// Get the current path as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        // Safety: we only ever write valid UTF-8 (ASCII)
        core::str::from_utf8(&self.path[..self.len]).unwrap_or("/")
    }

    /// Get the current length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the path is empty (just root).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len <= 1
    }
}

impl Default for PathBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over child nodes.
#[derive(Debug, Clone)]
pub struct NodeIter<'a> {
    /// Nodes to iterate
    nodes: &'a [Node<'a>],
    /// Current index
    index: usize,
    /// Filter depth (only return nodes at this depth)
    filter_depth: Option<usize>,
    /// Filter parent offset
    filter_parent: Option<usize>,
}

impl<'a> NodeIter<'a> {
    /// Create a new node iterator.
    #[must_use]
    pub const fn new(nodes: &'a [Node<'a>]) -> Self {
        Self {
            nodes,
            index: 0,
            filter_depth: None,
            filter_parent: None,
        }
    }

    /// Filter to only return direct children of a node.
    #[must_use]
    pub const fn children_of(mut self, parent: &Node<'_>) -> Self {
        self.filter_depth = Some(parent.depth + 1);
        self.filter_parent = Some(parent.content_offset);
        self
    }

    /// Filter to only return nodes at a specific depth.
    #[must_use]
    pub const fn at_depth(mut self, depth: usize) -> Self {
        self.filter_depth = Some(depth);
        self
    }
}

impl<'a> Iterator for NodeIter<'a> {
    type Item = &'a Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.nodes.len() {
            let node = &self.nodes[self.index];
            self.index += 1;

            // Apply depth filter
            if let Some(depth) = self.filter_depth {
                if node.depth != depth {
                    continue;
                }
            }

            // Apply parent filter
            if let Some(parent_offset) = self.filter_parent {
                if node.parent_offset != parent_offset {
                    continue;
                }
            }

            return Some(node);
        }
        None
    }
}

/// A parsed reg property with address/size cell information.
#[derive(Debug)]
pub struct ParsedReg<'a> {
    /// Raw property data
    data: &'a [u8],
    /// Number of address cells
    address_cells: u32,
    /// Number of size cells
    size_cells: u32,
}

impl<'a> ParsedReg<'a> {
    /// Create a new parsed reg.
    #[must_use]
    pub const fn new(data: &'a [u8], address_cells: u32, size_cells: u32) -> Self {
        Self {
            data,
            address_cells,
            size_cells,
        }
    }

    /// Get the number of entries.
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        let entry_size = ((self.address_cells + self.size_cells) * 4) as usize;
        if entry_size == 0 {
            0
        } else {
            self.data.len() / entry_size
        }
    }

    /// Iterate over reg entries.
    #[must_use]
    pub const fn iter(&self) -> RegIter<'a> {
        RegIter::new(self.data, self.address_cells, self.size_cells)
    }

    /// Get the first entry.
    #[must_use]
    pub fn first(&self) -> Option<RegEntry> {
        self.iter().next()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec::Vec;

    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new("uart@10000", 100, 2, 50);

        assert_eq!(node.name(), "uart@10000");
        assert_eq!(node.unit_name(), "uart");
        assert_eq!(node.unit_address(), Some("10000"));
        assert_eq!(node.content_offset(), 100);
        assert_eq!(node.depth(), 2);
        assert!(!node.is_root());
    }

    #[test]
    fn test_node_root() {
        let node = Node::new("", 0, 0, 0);
        assert!(node.is_root());
    }

    #[test]
    fn test_node_no_unit_address() {
        let node = Node::new("memory", 100, 1, 0);

        assert_eq!(node.unit_name(), "memory");
        assert_eq!(node.unit_address(), None);
    }

    #[test]
    fn test_path_builder() {
        let mut pb = PathBuilder::new();

        assert_eq!(pb.as_str(), "/");

        pb.push("soc").unwrap();
        assert_eq!(pb.as_str(), "/soc");

        pb.push("uart@10000").unwrap();
        assert_eq!(pb.as_str(), "/soc/uart@10000");

        pb.pop();
        assert_eq!(pb.as_str(), "/soc");

        pb.pop();
        assert_eq!(pb.as_str(), "/");
    }

    #[test]
    fn test_path_builder_overflow() {
        let mut pb = PathBuilder::new();

        // Create a very long path component
        let long_name = "a".repeat(MAX_PATH_LEN);
        let result = pb.push(&long_name);

        assert!(matches!(result, Err(DtbError::PathTooLong { .. })));
    }

    #[test]
    fn test_path_builder_from_str() {
        let pb = PathBuilder::from_str("/soc/uart@10000").unwrap();
        assert_eq!(pb.as_str(), "/soc/uart@10000");
    }

    #[test]
    fn test_parsed_reg() {
        let data = [
            0x00, 0x00, 0x10, 0x00, // addr
            0x00, 0x00, 0x01, 0x00, // size
        ];

        let reg = ParsedReg::new(&data, 1, 1);

        assert_eq!(reg.entry_count(), 1);

        let entry = reg.first().unwrap();
        assert_eq!(entry.address, 0x1000);
        assert_eq!(entry.size, 0x100);
    }

    #[test]
    fn test_node_iter() {
        let nodes = [
            Node::new("", 0, 0, 0),           // root
            Node::new("soc", 10, 1, 0),       // child of root
            Node::new("uart", 20, 2, 10),     // child of soc
            Node::new("timer", 30, 2, 10),    // child of soc
            Node::new("memory", 40, 1, 0),    // child of root
        ];

        // All nodes
        let all: Vec<_> = NodeIter::new(&nodes).collect();
        assert_eq!(all.len(), 5);

        // Nodes at depth 1
        let depth1: Vec<_> = NodeIter::new(&nodes).at_depth(1).collect();
        assert_eq!(depth1.len(), 2);
        assert_eq!(depth1[0].name(), "soc");
        assert_eq!(depth1[1].name(), "memory");

        // Children of soc
        let soc = &nodes[1];
        let soc_children: Vec<_> = NodeIter::new(&nodes).children_of(soc).collect();
        assert_eq!(soc_children.len(), 2);
    }
}
