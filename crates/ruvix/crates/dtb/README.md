# ruvix-dtb

Device Tree Blob (FDT) parser for the RuVix Cognition Kernel.

## Overview

This crate provides a zero-copy parser for Flattened Device Tree (FDT) blobs,
enabling hardware discovery at boot time. It is designed as part of ADR-087
for the RuVix Cognition Kernel.

## Features

- **No-std compatible**: Works in bare-metal environments
- **Zero-copy parsing**: References directly into the FDT blob
- **No heap allocation**: Fixed-size internal buffers
- **Safe parsing**: Validates all data before access
- **FDT version 17**: Compatible with standard device trees

## Core Types

### DeviceTree

Main parser and accessor for device tree data:

```rust
use ruvix_dtb::DeviceTree;

fn discover_hardware(blob: &[u8]) -> Result<(), ruvix_dtb::DtbError> {
    let dt = DeviceTree::parse(blob)?;

    // Get basic info
    println!("Total size: {} bytes", dt.total_size());
    println!("Boot CPU: {}", dt.boot_cpuid());
    println!("Node count: {}", dt.node_count());

    Ok(())
}
```

### Node

Reference to a node in the tree:

```rust
use ruvix_dtb::{DeviceTree, Node};

fn inspect_node(dt: &DeviceTree, node: &Node) {
    println!("Name: {}", node.name());
    println!("Unit name: {}", node.unit_name());

    if let Some(addr) = node.unit_address() {
        println!("Unit address: {}", addr);
    }

    println!("Depth: {}", node.depth());
    println!("Is root: {}", node.is_root());
}
```

### Property

Access to node properties:

```rust
use ruvix_dtb::{DeviceTree, Property, PropertyValue};

fn read_properties(dt: &DeviceTree) -> Result<(), ruvix_dtb::DtbError> {
    let root = dt.root().ok_or(ruvix_dtb::DtbError::NodeNotFound)?;

    // Get string property
    let model = dt.get_string(&root, "model")?;
    println!("Model: {}", model);

    // Get numeric property
    if let Ok(cells) = dt.get_u32(&root, "#address-cells") {
        println!("Address cells: {}", cells);
    }

    // Parse property dynamically
    let prop = dt.get_property(&root, "compatible")?;
    match prop.parse() {
        PropertyValue::String(s) => println!("Compatible: {}", s),
        PropertyValue::U32(v) => println!("Value: 0x{:x}", v),
        _ => println!("Other type"),
    }

    Ok(())
}
```

### RegEntry

Parsed memory region entry:

```rust
use ruvix_dtb::RegEntry;

let entry = RegEntry::new(0x8000_0000, 0x1000_0000);
assert_eq!(entry.address, 0x8000_0000);
assert_eq!(entry.size, 0x1000_0000);
assert_eq!(entry.end(), 0x9000_0000);
assert!(entry.contains(0x8000_0000));
```

## Usage Examples

### Finding Memory Regions

```rust
use ruvix_dtb::DeviceTree;

fn get_memory(blob: &[u8]) -> Result<Vec<(u64, u64)>, ruvix_dtb::DtbError> {
    let dt = DeviceTree::parse(blob)?;
    let mut regions = Vec::new();

    if let Some(memory) = dt.find_node("/memory") {
        let reg = dt.get_reg(&memory)?;
        for entry in reg.iter() {
            regions.push((entry.address, entry.size));
        }
    }

    Ok(regions)
}
```

### Finding Compatible Devices

```rust
use ruvix_dtb::DeviceTree;

fn find_uarts(blob: &[u8]) -> Result<(), ruvix_dtb::DtbError> {
    let dt = DeviceTree::parse(blob)?;

    for node in dt.find_compatible("ns16550a") {
        println!("Found UART: {}", node.name());

        if let Ok(reg) = dt.get_reg(&node) {
            if let Some(entry) = reg.first() {
                println!("  Address: 0x{:x}", entry.address);
            }
        }
    }

    Ok(())
}
```

### Iterating Children

```rust
use ruvix_dtb::DeviceTree;

fn list_soc_devices(blob: &[u8]) -> Result<(), ruvix_dtb::DtbError> {
    let dt = DeviceTree::parse(blob)?;

    if let Some(soc) = dt.find_node("/soc") {
        for child in dt.children(&soc) {
            println!("Device: {}", child.name());
        }
    }

    Ok(())
}
```

## FDT Structure

A Flattened Device Tree consists of:

1. **Header** (40 bytes)
   - Magic number: `0xd00dfeed`
   - Total size, version, offsets

2. **Memory Reservation Block**
   - Reserved memory regions

3. **Structure Block**
   - Tree of nodes and properties
   - Tokens: `FDT_BEGIN_NODE`, `FDT_END_NODE`, `FDT_PROP`, `FDT_NOP`, `FDT_END`

4. **Strings Block**
   - Property name strings

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `FDT_MAGIC` | `0xd00dfeed` | FDT magic number |
| `FDT_VERSION` | 17 | Supported version |
| `MAX_TREE_DEPTH` | 32 | Maximum tree depth |
| `MAX_PATH_LEN` | 256 | Maximum path length |
| `MAX_PROPERTY_NAME_LEN` | 64 | Maximum property name |

## Error Handling

All parsing errors are returned as `DtbError`:

```rust
use ruvix_dtb::{DeviceTree, DtbError};

fn handle_errors(blob: &[u8]) {
    match DeviceTree::parse(blob) {
        Ok(dt) => println!("Parsed {} nodes", dt.node_count()),
        Err(DtbError::InvalidMagic { found }) => {
            println!("Not a DTB (magic: 0x{:08x})", found);
        }
        Err(DtbError::UnsupportedVersion { version }) => {
            println!("Unsupported version: {}", version);
        }
        Err(e) => println!("Parse error: {}", e),
    }
}
```

## License

MIT OR Apache-2.0
