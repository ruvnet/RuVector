# ruvix-net

Minimal networking stack for the RuVix Cognition Kernel (ADR-087 Phase E).

## Overview

This crate provides a complete `no_std` networking stack suitable for embedded systems and OS kernels. It implements:

- **Ethernet II** frame handling
- **ARP** address resolution with cache management
- **IPv4** routing and header processing
- **UDP** datagram transport
- **ICMP** echo (ping) support

## Architecture

```text
+------------------+
|   Application    |  UDP send/receive
+------------------+
         |
+------------------+
|   NetworkStack   |  Coordinates all layers
+------------------+
         |
+--------+---------+
|        |         |
v        v         v
+------+ +-------+ +------+
| UDP  | | ICMP  | | ARP  |
+------+ +-------+ +------+
    |        |         |
    v        v         v
+------------------+
|      IPv4        |  Routing, TTL, checksum
+------------------+
         |
+------------------+
|    Ethernet      |  Frame encapsulation
+------------------+
         |
+------------------+
|  NetworkDevice   |  Hardware abstraction
+------------------+
```

## Features

- `std` - Enable standard library support
- `alloc` - Enable heap allocation via the alloc crate

By default, the crate is `#![no_std]` compatible with zero heap allocations.

## Usage

```rust
use ruvix_net::{
    NetworkStack, StackConfig, NetworkDevice, LoopbackDevice,
    Ipv4Addr, MacAddress,
};

// Create a network device (or use your own hardware driver)
let device = LoopbackDevice::new();

// Configure the stack
let config = StackConfig {
    ip_addr: Ipv4Addr::new(192, 168, 1, 100),
    subnet_mask: Ipv4Addr::new(255, 255, 255, 0),
    gateway: Ipv4Addr::new(192, 168, 1, 1),
    ..Default::default()
};

// Create the network stack
let mut stack = NetworkStack::new(device, config);

// Send a UDP datagram
let result = stack.send_udp(
    12345,                          // source port
    Ipv4Addr::new(192, 168, 1, 2),  // destination IP
    80,                             // destination port
    b"Hello, network!",             // payload
);

// Receive packets
let mut buf = [0u8; 1536];
if let Ok(Some(packet)) = stack.receive(&mut buf) {
    let payload = &buf[packet.payload_offset..][..packet.payload_len];
    // Process received data
}
```

## Layer Details

### Ethernet (`ethernet.rs`)

- `MacAddress` - 6-byte hardware address with broadcast/multicast detection
- `EthernetFrame` - Parsing and serialization of Ethernet II frames
- `EtherType` - Protocol identifiers (IPv4, ARP, IPv6)

### ARP (`arp.rs`)

- `ArpPacket` - ARP request/reply packet handling
- `ArpCache` - Fixed-size cache with timeout-based eviction
- Supports Ethernet/IPv4 address resolution

### IPv4 (`ipv4.rs`)

- `Ipv4Addr` - 4-byte IP address with classification methods
- `Ipv4Header` - Full header parsing with checksum verification
- `Protocol` - Transport protocol identifiers (UDP, TCP, ICMP)

### UDP (`udp.rs`)

- `UdpHeader` - Datagram header with pseudo-header checksum
- `UdpSocket` - Minimal socket abstraction for binding and sending

### ICMP (`icmp.rs`)

- `IcmpHeader` - Generic ICMP message handling
- `IcmpEcho` - Echo request/reply (ping) support
- `IcmpDestUnreachable`, `IcmpTimeExceeded` - Error messages

### Device (`device.rs`)

- `NetworkDevice` trait - Hardware abstraction for send/receive
- `LoopbackDevice` - Testing device that echoes frames
- `NullDevice` - Discard device for testing

### Stack (`stack.rs`)

- `NetworkStack` - Integrated network stack combining all layers
- `StackConfig` - IP address, gateway, and behavior settings
- Automatic ARP resolution and ICMP echo handling

## Network Device Trait

To integrate with actual hardware, implement the `NetworkDevice` trait:

```rust
use ruvix_net::{NetworkDevice, MacAddress, DeviceCapabilities, LinkStatus, DeviceStats, NetResult};

struct MyEthernetDriver {
    // Your hardware state
}

impl NetworkDevice for MyEthernetDriver {
    fn mac_address(&self) -> MacAddress {
        MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::DEFAULT
    }

    fn link_status(&self) -> LinkStatus {
        LinkStatus::Up(LinkSpeed::Speed1G)
    }

    fn stats(&self) -> DeviceStats {
        DeviceStats::new()
    }

    fn send(&mut self, frame: &[u8]) -> NetResult<()> {
        // Transmit frame to hardware
        Ok(())
    }

    fn receive(&mut self, buf: &mut [u8]) -> NetResult<Option<usize>> {
        // Receive frame from hardware
        Ok(None)
    }

    fn set_promiscuous(&mut self, enabled: bool) -> NetResult<()> {
        Ok(())
    }

    fn set_enabled(&mut self, enabled: bool) -> NetResult<()> {
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        true
    }

    fn reset(&mut self) -> NetResult<()> {
        Ok(())
    }
}
```

## Memory Usage

The stack is designed for constrained environments:

- **ARP Cache**: 64 entries x ~24 bytes = ~1.5 KB
- **No heap allocation** in default configuration
- **Fixed buffers** for packet processing

## Error Handling

All operations return `NetResult<T>` which is `Result<T, NetError>`. Error types include:

- `PacketTooShort` - Truncated packet
- `InvalidEthernetFrame` - Malformed Ethernet header
- `Ipv4ChecksumError` - IPv4 checksum mismatch
- `ArpNotFound` - Destination MAC not in cache
- `UdpChecksumError` - UDP checksum mismatch
- And more...

## Testing

```bash
cd crates/ruvix/crates/net
cargo test
```

## License

MIT OR Apache-2.0
