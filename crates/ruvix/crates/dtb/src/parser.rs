//! Device tree parser.

use crate::{
    align4, node::ParsedReg, read_be_u32, strlen, DtbError, DtbResult, FdtHeader, FdtToken, Node,
    Property, MAX_TREE_DEPTH,
};

/// Maximum number of nodes we can store during parsing.
const MAX_NODES: usize = 512;

/// Maximum path components for find_node.
const MAX_PATH_COMPONENTS: usize = 16;

/// Parsed device tree.
#[derive(Debug)]
pub struct DeviceTree<'a> {
    /// Reference to the DTB blob
    blob: &'a [u8],
    /// Parsed header
    header: FdtHeader,
    /// Parsed nodes (stored inline to avoid allocation)
    nodes: [Option<Node<'a>>; MAX_NODES],
    /// Number of nodes parsed
    node_count: usize,
}

impl<'a> DeviceTree<'a> {
    /// Parse a device tree from a DTB blob.
    ///
    /// # Errors
    ///
    /// Returns `DtbError` if:
    /// - The blob is invalid or corrupt
    /// - The FDT version is not supported
    /// - The structure is malformed
    pub fn parse(blob: &'a [u8]) -> DtbResult<Self> {
        let header = FdtHeader::parse(blob)?;

        let mut dt = Self {
            blob,
            header,
            nodes: [const { None }; MAX_NODES],
            node_count: 0,
        };

        dt.parse_structure_block()?;

        Ok(dt)
    }

    /// Get the FDT header.
    #[must_use]
    pub const fn header(&self) -> &FdtHeader {
        &self.header
    }

    /// Get the total size of the DTB.
    #[must_use]
    pub const fn total_size(&self) -> u32 {
        self.header.total_size
    }

    /// Get the boot CPU physical ID.
    #[must_use]
    pub const fn boot_cpuid(&self) -> u32 {
        self.header.boot_cpuid_phys
    }

    /// Get the root node.
    #[must_use]
    pub fn root(&self) -> Option<&Node<'a>> {
        self.nodes[0].as_ref()
    }

    /// Get the number of nodes.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Find a node by path.
    ///
    /// # Arguments
    ///
    /// * `path` - Absolute path like "/soc/uart@10000"
    #[must_use]
    pub fn find_node(&self, path: &str) -> Option<&Node<'a>> {
        if path == "/" {
            return self.root();
        }

        // Split path into components using a fixed-size buffer
        let path = path.strip_prefix('/').unwrap_or(path);

        // Collect components into a fixed-size array
        let mut components: [&str; MAX_PATH_COMPONENTS] = [""; MAX_PATH_COMPONENTS];
        let mut component_count = 0;

        for component in path.split('/') {
            if component_count >= MAX_PATH_COMPONENTS {
                return None; // Path too long
            }
            components[component_count] = component;
            component_count += 1;
        }

        let mut current_depth = 0;

        for node_opt in &self.nodes[..self.node_count] {
            let node = node_opt.as_ref()?;

            // Check if this node matches the current path component
            if node.depth() == current_depth + 1 {
                let target = components.get(current_depth)?;
                if node.name() == *target {
                    current_depth += 1;

                    // If we've matched all components, this is our node
                    if current_depth == component_count {
                        return Some(node);
                    }
                }
            }
        }

        None
    }

    /// Find all nodes with a given compatible string.
    ///
    /// Returns an iterator over matching nodes.
    pub fn find_compatible<'b>(&'b self, compatible: &'b str) -> CompatibleIter<'a, 'b> {
        CompatibleIter {
            dt: self,
            compatible,
            index: 0,
        }
    }

    /// Find all nodes with a given node name.
    ///
    /// Returns an iterator over matching nodes.
    pub fn find_by_name<'b>(&'b self, name: &'b str) -> NameIter<'a, 'b> {
        NameIter {
            dt: self,
            name,
            index: 0,
        }
    }

    /// Get a property from a node.
    ///
    /// # Errors
    ///
    /// Returns `DtbError::PropertyNotFound` if the property doesn't exist.
    pub fn get_property(&self, node: &Node<'a>, name: &str) -> DtbResult<Property<'a>> {
        let structure = self.header.structure_block(self.blob);
        let mut offset = node.content_offset();

        while offset < structure.len() {
            let token_val = read_be_u32(structure, offset).ok_or(DtbError::UnexpectedEnd)?;
            offset += 4;

            match FdtToken::from_u32(token_val) {
                Some(FdtToken::Prop) => {
                    let len = read_be_u32(structure, offset).ok_or(DtbError::UnexpectedEnd)?;
                    let nameoff =
                        read_be_u32(structure, offset + 4).ok_or(DtbError::UnexpectedEnd)?;
                    offset += 8;

                    // CVE-003 FIX: Check for integer overflow before array indexing
                    let len_usize = len as usize;
                    let value_end = offset
                        .checked_add(len_usize)
                        .ok_or(DtbError::IntegerOverflow)?;
                    if value_end > structure.len() {
                        return Err(DtbError::UnexpectedEnd);
                    }

                    let prop_name = self.header.get_string(self.blob, nameoff)?;
                    let value = &structure[offset..value_end];
                    offset = align4(value_end);

                    if prop_name == name {
                        return Ok(Property::new(prop_name, value));
                    }
                }
                Some(FdtToken::BeginNode) => {
                    // Skip child node
                    break;
                }
                Some(FdtToken::EndNode) | Some(FdtToken::End) => {
                    break;
                }
                Some(FdtToken::Nop) => {
                    continue;
                }
                None => {
                    return Err(DtbError::InvalidToken { value: token_val });
                }
            }
        }

        Err(DtbError::PropertyNotFound)
    }

    /// Get a u32 property value.
    pub fn get_u32(&self, node: &Node<'a>, name: &str) -> DtbResult<u32> {
        self.get_property(node, name)?.as_u32()
    }

    /// Get a u64 property value.
    pub fn get_u64(&self, node: &Node<'a>, name: &str) -> DtbResult<u64> {
        self.get_property(node, name)?.as_u64()
    }

    /// Get a string property value.
    ///
    /// Note: The returned string has the same lifetime as the DTB blob.
    pub fn get_string(&self, node: &Node<'a>, name: &str) -> DtbResult<&'a str> {
        self.get_property(node, name)?.as_str()
    }

    /// Get the reg property parsed with proper cell sizes.
    pub fn get_reg(&self, node: &Node<'a>) -> DtbResult<ParsedReg<'a>> {
        // Get #address-cells and #size-cells from parent
        // Default values per spec: #address-cells = 2, #size-cells = 1
        let address_cells = self
            .get_parent_property_u32(node, "#address-cells")
            .unwrap_or(2);
        let size_cells = self
            .get_parent_property_u32(node, "#size-cells")
            .unwrap_or(1);

        // Get the raw reg data directly from the structure block
        let reg_data = self.get_property_value(node, "reg")?;
        Ok(ParsedReg::new(reg_data, address_cells, size_cells))
    }

    /// Get raw property value bytes directly from the structure block.
    fn get_property_value(&self, node: &Node<'a>, name: &str) -> DtbResult<&'a [u8]> {
        let structure = self.header.structure_block(self.blob);
        let mut offset = node.content_offset();

        while offset < structure.len() {
            let token_val = read_be_u32(structure, offset).ok_or(DtbError::UnexpectedEnd)?;
            offset += 4;

            match FdtToken::from_u32(token_val) {
                Some(FdtToken::Prop) => {
                    let len = read_be_u32(structure, offset).ok_or(DtbError::UnexpectedEnd)?;
                    let nameoff =
                        read_be_u32(structure, offset + 4).ok_or(DtbError::UnexpectedEnd)?;
                    offset += 8;

                    // CVE-003 FIX: Check for integer overflow before array indexing
                    let len_usize = len as usize;
                    let value_end = offset
                        .checked_add(len_usize)
                        .ok_or(DtbError::IntegerOverflow)?;
                    if value_end > structure.len() {
                        return Err(DtbError::UnexpectedEnd);
                    }

                    let prop_name = self.header.get_string(self.blob, nameoff)?;
                    let value = &structure[offset..value_end];
                    offset = align4(value_end);

                    if prop_name == name {
                        return Ok(value);
                    }
                }
                Some(FdtToken::BeginNode) => break,
                Some(FdtToken::EndNode) | Some(FdtToken::End) => break,
                Some(FdtToken::Nop) => continue,
                None => return Err(DtbError::InvalidToken { value: token_val }),
            }
        }

        Err(DtbError::PropertyNotFound)
    }

    /// Get a property from the parent node.
    fn get_parent_property_u32(&self, node: &Node<'a>, name: &str) -> Option<u32> {
        // Find parent node
        let parent = self.nodes[..self.node_count]
            .iter()
            .filter_map(|n| n.as_ref())
            .find(|n| n.content_offset() == node.parent_offset())?;

        self.get_property(parent, name).ok()?.as_u32().ok()
    }

    /// Iterate over all nodes.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &Node<'a>> {
        self.nodes[..self.node_count]
            .iter()
            .filter_map(|n| n.as_ref())
    }

    /// Iterate over children of a node.
    pub fn children(&self, parent: &Node<'a>) -> impl Iterator<Item = &Node<'a>> {
        let parent_offset = parent.content_offset();
        let parent_depth = parent.depth();

        self.nodes[..self.node_count]
            .iter()
            .filter_map(|n| n.as_ref())
            .filter(move |n| n.parent_offset() == parent_offset && n.depth() == parent_depth + 1)
    }

    /// Parse the structure block.
    fn parse_structure_block(&mut self) -> DtbResult<()> {
        let structure = self.header.structure_block(self.blob);
        let mut offset = 0;
        let mut depth = 0;
        let mut parent_stack: [usize; MAX_TREE_DEPTH] = [0; MAX_TREE_DEPTH];

        while offset < structure.len() {
            let token_val = read_be_u32(structure, offset).ok_or(DtbError::UnexpectedEnd)?;
            offset += 4;

            match FdtToken::from_u32(token_val) {
                Some(FdtToken::BeginNode) => {
                    // Get node name
                    let name_len = strlen(structure, offset).ok_or(DtbError::InvalidNodeName)?;
                    let name = core::str::from_utf8(&structure[offset..offset + name_len])
                        .map_err(|_| DtbError::InvalidNodeName)?;

                    offset = align4(offset + name_len + 1);

                    // Get parent offset
                    let parent_offset = if depth == 0 {
                        0
                    } else {
                        parent_stack[depth - 1]
                    };

                    // Store node
                    if self.node_count >= MAX_NODES {
                        // Too many nodes, stop parsing
                        break;
                    }

                    let content_offset = offset;
                    self.nodes[self.node_count] =
                        Some(Node::new(name, content_offset, depth, parent_offset));

                    // Update parent stack
                    if depth < MAX_TREE_DEPTH {
                        parent_stack[depth] = content_offset;
                    }

                    self.node_count += 1;
                    depth += 1;

                    if depth >= MAX_TREE_DEPTH {
                        return Err(DtbError::TreeTooDeep { depth });
                    }
                }
                Some(FdtToken::EndNode) => {
                    if depth == 0 {
                        return Err(DtbError::UnexpectedToken {
                            expected: FdtToken::BeginNode as u32,
                            found: FdtToken::EndNode as u32,
                        });
                    }
                    depth -= 1;
                }
                Some(FdtToken::Prop) => {
                    let len = read_be_u32(structure, offset).ok_or(DtbError::UnexpectedEnd)?;
                    offset += 8; // Skip len and nameoff
                    // CVE-003 FIX: Check for integer overflow
                    let value_end = offset
                        .checked_add(len as usize)
                        .ok_or(DtbError::IntegerOverflow)?;
                    offset = align4(value_end);
                }
                Some(FdtToken::Nop) => {
                    continue;
                }
                Some(FdtToken::End) => {
                    break;
                }
                None => {
                    return Err(DtbError::InvalidToken { value: token_val });
                }
            }
        }

        Ok(())
    }
}

/// Iterator over nodes matching a compatible string.
pub struct CompatibleIter<'a, 'b> {
    dt: &'b DeviceTree<'a>,
    compatible: &'b str,
    index: usize,
}

impl<'a, 'b> Iterator for CompatibleIter<'a, 'b> {
    type Item = &'b Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.dt.node_count {
            let node_opt = &self.dt.nodes[self.index];
            self.index += 1;

            if let Some(node) = node_opt.as_ref() {
                if let Ok(prop) = self.dt.get_property(node, "compatible") {
                    if prop.as_stringlist().any(|s| s == self.compatible) {
                        return Some(node);
                    }
                }
            }
        }
        None
    }
}

/// Iterator over nodes matching a name.
pub struct NameIter<'a, 'b> {
    dt: &'b DeviceTree<'a>,
    name: &'b str,
    index: usize,
}

impl<'a, 'b> Iterator for NameIter<'a, 'b> {
    type Item = &'b Node<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.dt.node_count {
            let node_opt = &self.dt.nodes[self.index];
            self.index += 1;

            if let Some(node) = node_opt.as_ref() {
                if node.unit_name() == self.name {
                    return Some(node);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Create a minimal valid DTB for testing
    fn create_minimal_dtb() -> [u8; 96] {
        let mut dtb = [0u8; 96];

        // Header
        dtb[0..4].copy_from_slice(&0xd00d_feed_u32.to_be_bytes()); // magic
        dtb[4..8].copy_from_slice(&96_u32.to_be_bytes()); // total_size
        dtb[8..12].copy_from_slice(&48_u32.to_be_bytes()); // off_dt_struct
        dtb[12..16].copy_from_slice(&88_u32.to_be_bytes()); // off_dt_strings
        dtb[16..20].copy_from_slice(&40_u32.to_be_bytes()); // off_mem_rsvmap
        dtb[20..24].copy_from_slice(&17_u32.to_be_bytes()); // version
        dtb[24..28].copy_from_slice(&16_u32.to_be_bytes()); // last_comp_version
        dtb[28..32].copy_from_slice(&0_u32.to_be_bytes()); // boot_cpuid
        dtb[32..36].copy_from_slice(&8_u32.to_be_bytes()); // size_dt_strings
        dtb[36..40].copy_from_slice(&40_u32.to_be_bytes()); // size_dt_struct

        // Memory reservation (empty - just the terminator)
        dtb[40..48].copy_from_slice(&[0; 8]);

        // Structure block
        let mut offset = 48;

        // FDT_BEGIN_NODE
        dtb[offset..offset + 4].copy_from_slice(&1_u32.to_be_bytes());
        offset += 4;
        // Root node name (empty string)
        dtb[offset] = 0;
        offset = align4(offset + 1);

        // FDT_PROP for model
        dtb[offset..offset + 4].copy_from_slice(&3_u32.to_be_bytes());
        offset += 4;
        dtb[offset..offset + 4].copy_from_slice(&5_u32.to_be_bytes()); // len
        offset += 4;
        dtb[offset..offset + 4].copy_from_slice(&0_u32.to_be_bytes()); // nameoff
        offset += 4;
        dtb[offset..offset + 5].copy_from_slice(b"test\0");
        offset = align4(offset + 5);

        // FDT_END_NODE
        dtb[offset..offset + 4].copy_from_slice(&2_u32.to_be_bytes());
        offset += 4;

        // FDT_END
        dtb[offset..offset + 4].copy_from_slice(&9_u32.to_be_bytes());

        // Strings block
        dtb[88..94].copy_from_slice(b"model\0");

        dtb
    }

    #[test]
    fn test_parse_minimal_dtb() {
        let dtb = create_minimal_dtb();
        let dt = DeviceTree::parse(&dtb).unwrap();

        assert_eq!(dt.node_count(), 1);
        assert!(dt.root().is_some());
    }

    #[test]
    fn test_find_root() {
        let dtb = create_minimal_dtb();
        let dt = DeviceTree::parse(&dtb).unwrap();

        let root = dt.find_node("/").unwrap();
        assert!(root.is_root());
    }

    #[test]
    fn test_get_property() {
        let dtb = create_minimal_dtb();
        let dt = DeviceTree::parse(&dtb).unwrap();

        let root = dt.root().unwrap();
        let prop = dt.get_property(root, "model").unwrap();

        assert_eq!(prop.name(), "model");
        assert_eq!(prop.as_str().unwrap(), "test");
    }

    #[test]
    fn test_property_not_found() {
        let dtb = create_minimal_dtb();
        let dt = DeviceTree::parse(&dtb).unwrap();

        let root = dt.root().unwrap();
        let result = dt.get_property(root, "nonexistent");

        assert!(matches!(result, Err(DtbError::PropertyNotFound)));
    }

    #[test]
    fn test_invalid_magic() {
        let mut dtb = create_minimal_dtb();
        dtb[0..4].copy_from_slice(&0xDEAD_BEEF_u32.to_be_bytes());

        let result = DeviceTree::parse(&dtb);
        assert!(matches!(result, Err(DtbError::InvalidMagic { .. })));
    }
}
