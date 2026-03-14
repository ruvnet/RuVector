//! # RuVix Network Stack
//!
//! This crate provides a minimal networking stack for the RuVix Cognition Kernel
//! as specified in ADR-087 Phase E. It is designed to be `no_std` compatible with
//! optional alloc support for dynamic allocation.
//!
//! ## Network Layers
//!
//! | Layer | Module | Purpose |
//! |-------|--------|---------|
//! | **Link** | `ethernet` | Ethernet II frame handling |
//! | **Network** | `arp`, `ipv4`, `icmp` | Address resolution and IP routing |
//! | **Transport** | `udp` | Connectionless datagram transport |
//!
//! ## Architecture
//!
//! ```text
//! +------------------+
//! |   Application    |
//! +------------------+
//!          |
//! +------------------+
//! |       UDP        |
//! +------------------+
//!          |
//! +------------------+
//! |    IPv4 + ICMP   |
//! +------------------+
//!          |
//! +------------------+
//! |  ARP Resolution  |
//! +------------------+
//!          |
//! +------------------+
//! |    Ethernet      |
//! +------------------+
//!          |
//! +------------------+
//! |  NetworkDevice   |  <-- Hardware abstraction
//! +------------------+
//! ```
//!
//! ## Features
//!
//! - `std`: Enable standard library support
//! - `alloc`: Enable alloc crate support for heap allocation
//!
//! ## Example
//!
//! ```no_run
//! use ruvix_net::{MacAddress, Ipv4Addr, UdpSocket, NetworkStack};
//!
//! // Create a network stack with device
//! // let stack = NetworkStack::new(device, mac, ip);
//!
//! // Send UDP datagram
//! // let socket = stack.udp_bind(8080).unwrap();
//! // socket.send_to(&data, dest_addr, dest_port);
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod arp;
pub mod device;
pub mod error;
pub mod ethernet;
pub mod icmp;
pub mod ipv4;
pub mod stack;
pub mod udp;

// Re-exports for convenience
pub use arp::{ArpCache, ArpOperation, ArpPacket};
pub use device::NetworkDevice;
pub use error::NetError;
pub use ethernet::{EtherType, EthernetFrame, MacAddress};
pub use icmp::{IcmpHeader, IcmpType};
pub use ipv4::{Ipv4Addr, Ipv4Header, Protocol};
pub use stack::NetworkStack;
pub use udp::{UdpHeader, UdpSocket};

/// Maximum Transmission Unit (standard Ethernet).
pub const MTU: usize = 1500;

/// Ethernet header size in bytes.
pub const ETHERNET_HEADER_SIZE: usize = 14;

/// IPv4 header minimum size in bytes.
pub const IPV4_HEADER_MIN_SIZE: usize = 20;

/// UDP header size in bytes.
pub const UDP_HEADER_SIZE: usize = 8;

/// ICMP header size in bytes.
pub const ICMP_HEADER_SIZE: usize = 8;

/// ARP packet size in bytes (for Ethernet/IPv4).
pub const ARP_PACKET_SIZE: usize = 28;

/// Maximum UDP payload size for standard Ethernet.
pub const MAX_UDP_PAYLOAD: usize = MTU - IPV4_HEADER_MIN_SIZE - UDP_HEADER_SIZE;
