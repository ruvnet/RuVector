//! Ethernet II frame handling.
//!
//! This module provides types and functions for parsing and serializing
//! Ethernet II frames as per IEEE 802.3.
//!
//! ## Frame Format
//!
//! ```text
//! +--------------------+--------------------+----------+---------+
//! | Destination MAC    | Source MAC         | EtherType| Payload |
//! | (6 bytes)          | (6 bytes)          | (2 bytes)| (46-1500)|
//! +--------------------+--------------------+----------+---------+
//! ```

use crate::error::{NetError, NetResult};
use crate::ETHERNET_HEADER_SIZE;

/// MAC (Media Access Control) address.
///
/// A 48-bit hardware address uniquely identifying a network interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct MacAddress(pub [u8; 6]);

impl MacAddress {
    /// Broadcast MAC address (ff:ff:ff:ff:ff:ff).
    pub const BROADCAST: Self = Self([0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);

    /// Zero/unspecified MAC address.
    pub const ZERO: Self = Self([0, 0, 0, 0, 0, 0]);

    /// Creates a new MAC address from bytes.
    #[inline]
    #[must_use]
    pub const fn new(bytes: [u8; 6]) -> Self {
        Self(bytes)
    }

    /// Returns the MAC address as a byte slice.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 6] {
        &self.0
    }

    /// Checks if this is the broadcast address.
    #[inline]
    #[must_use]
    pub const fn is_broadcast(&self) -> bool {
        self.0[0] == 0xff
            && self.0[1] == 0xff
            && self.0[2] == 0xff
            && self.0[3] == 0xff
            && self.0[4] == 0xff
            && self.0[5] == 0xff
    }

    /// Checks if this is a multicast address (LSB of first byte is 1).
    #[inline]
    #[must_use]
    pub const fn is_multicast(&self) -> bool {
        (self.0[0] & 0x01) != 0
    }

    /// Checks if this is a unicast address.
    #[inline]
    #[must_use]
    pub const fn is_unicast(&self) -> bool {
        !self.is_multicast()
    }

    /// Checks if this is the zero address.
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.0[0] == 0
            && self.0[1] == 0
            && self.0[2] == 0
            && self.0[3] == 0
            && self.0[4] == 0
            && self.0[5] == 0
    }

    /// Checks if this is a locally administered address (bit 1 of first byte).
    #[inline]
    #[must_use]
    pub const fn is_local(&self) -> bool {
        (self.0[0] & 0x02) != 0
    }

    /// Parses a MAC address from a byte slice.
    #[inline]
    pub fn parse(bytes: &[u8]) -> NetResult<Self> {
        if bytes.len() < 6 {
            return Err(NetError::PacketTooShort);
        }
        let mut addr = [0u8; 6];
        addr.copy_from_slice(&bytes[..6]);
        Ok(Self(addr))
    }
}

/// Ethernet frame type identifiers.
///
/// EtherType indicates which protocol is encapsulated in the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum EtherType {
    /// Internet Protocol version 4 (0x0800).
    Ipv4 = 0x0800,
    /// Address Resolution Protocol (0x0806).
    Arp = 0x0806,
    /// Internet Protocol version 6 (0x86DD).
    Ipv6 = 0x86DD,
    /// VLAN-tagged frame (0x8100).
    Vlan = 0x8100,
    /// Unknown or unsupported type.
    Unknown(u16) = 0xFFFF,
}

impl EtherType {
    /// Converts from a raw u16 value.
    #[inline]
    #[must_use]
    pub const fn from_u16(value: u16) -> Self {
        match value {
            0x0800 => Self::Ipv4,
            0x0806 => Self::Arp,
            0x86DD => Self::Ipv6,
            0x8100 => Self::Vlan,
            other => Self::Unknown(other),
        }
    }

    /// Converts to a raw u16 value.
    #[inline]
    #[must_use]
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::Ipv4 => 0x0800,
            Self::Arp => 0x0806,
            Self::Ipv6 => 0x86DD,
            Self::Vlan => 0x8100,
            Self::Unknown(v) => v,
        }
    }
}

/// Ethernet II frame.
///
/// Represents a parsed Ethernet frame with header fields and payload reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EthernetFrame<'a> {
    /// Destination MAC address.
    pub dest_mac: MacAddress,
    /// Source MAC address.
    pub src_mac: MacAddress,
    /// EtherType indicating payload protocol.
    pub ether_type: EtherType,
    /// Frame payload (protocol data unit).
    pub payload: &'a [u8],
}

impl<'a> EthernetFrame<'a> {
    /// Minimum frame size (header only, no payload).
    pub const MIN_SIZE: usize = ETHERNET_HEADER_SIZE;

    /// Maximum payload size for standard Ethernet.
    pub const MAX_PAYLOAD: usize = 1500;

    /// Parses an Ethernet frame from a byte buffer.
    ///
    /// # Errors
    ///
    /// Returns `NetError::PacketTooShort` if the buffer is smaller than 14 bytes.
    #[inline]
    pub fn parse(bytes: &'a [u8]) -> NetResult<Self> {
        if bytes.len() < Self::MIN_SIZE {
            return Err(NetError::PacketTooShort);
        }

        let dest_mac = MacAddress::parse(&bytes[0..6])?;
        let src_mac = MacAddress::parse(&bytes[6..12])?;
        let ether_type = EtherType::from_u16(u16::from_be_bytes([bytes[12], bytes[13]]));
        let payload = &bytes[14..];

        Ok(Self {
            dest_mac,
            src_mac,
            ether_type,
            payload,
        })
    }

    /// Serializes the Ethernet frame into a buffer.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns `NetError::BufferTooSmall` if the buffer cannot hold the frame.
    /// Returns `NetError::PacketTooLarge` if payload exceeds MTU.
    #[inline]
    pub fn serialize(&self, buf: &mut [u8]) -> NetResult<usize> {
        let total_size = Self::MIN_SIZE + self.payload.len();

        if buf.len() < total_size {
            return Err(NetError::BufferTooSmall);
        }

        if self.payload.len() > Self::MAX_PAYLOAD {
            return Err(NetError::PacketTooLarge);
        }

        buf[0..6].copy_from_slice(&self.dest_mac.0);
        buf[6..12].copy_from_slice(&self.src_mac.0);
        buf[12..14].copy_from_slice(&self.ether_type.to_u16().to_be_bytes());
        buf[14..total_size].copy_from_slice(self.payload);

        Ok(total_size)
    }

    /// Creates a new Ethernet frame for serialization.
    #[inline]
    #[must_use]
    pub const fn new(
        dest_mac: MacAddress,
        src_mac: MacAddress,
        ether_type: EtherType,
        payload: &'a [u8],
    ) -> Self {
        Self {
            dest_mac,
            src_mac,
            ether_type,
            payload,
        }
    }

    /// Returns the total frame size including header.
    #[inline]
    #[must_use]
    pub const fn total_size(&self) -> usize {
        Self::MIN_SIZE + self.payload.len()
    }
}

/// Builder for constructing Ethernet frames.
pub struct EthernetFrameBuilder {
    dest_mac: MacAddress,
    src_mac: MacAddress,
    ether_type: EtherType,
}

impl EthernetFrameBuilder {
    /// Creates a new frame builder with default values.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            dest_mac: MacAddress::ZERO,
            src_mac: MacAddress::ZERO,
            ether_type: EtherType::Ipv4,
        }
    }

    /// Sets the destination MAC address.
    #[inline]
    #[must_use]
    pub const fn dest(mut self, mac: MacAddress) -> Self {
        self.dest_mac = mac;
        self
    }

    /// Sets the source MAC address.
    #[inline]
    #[must_use]
    pub const fn src(mut self, mac: MacAddress) -> Self {
        self.src_mac = mac;
        self
    }

    /// Sets the EtherType.
    #[inline]
    #[must_use]
    pub const fn ether_type(mut self, etype: EtherType) -> Self {
        self.ether_type = etype;
        self
    }

    /// Builds the frame with the given payload.
    #[inline]
    #[must_use]
    pub const fn build<'a>(self, payload: &'a [u8]) -> EthernetFrame<'a> {
        EthernetFrame {
            dest_mac: self.dest_mac,
            src_mac: self.src_mac,
            ether_type: self.ether_type,
            payload,
        }
    }

    /// Serializes the frame header into a buffer (without payload).
    ///
    /// Returns the number of bytes written (always 14 on success).
    #[inline]
    pub fn serialize_header(&self, buf: &mut [u8]) -> NetResult<usize> {
        if buf.len() < ETHERNET_HEADER_SIZE {
            return Err(NetError::BufferTooSmall);
        }

        buf[0..6].copy_from_slice(&self.dest_mac.0);
        buf[6..12].copy_from_slice(&self.src_mac.0);
        buf[12..14].copy_from_slice(&self.ether_type.to_u16().to_be_bytes());

        Ok(ETHERNET_HEADER_SIZE)
    }
}

impl Default for EthernetFrameBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_address_broadcast() {
        assert!(MacAddress::BROADCAST.is_broadcast());
        assert!(MacAddress::BROADCAST.is_multicast());
        assert!(!MacAddress::BROADCAST.is_unicast());
    }

    #[test]
    fn test_mac_address_zero() {
        assert!(MacAddress::ZERO.is_zero());
        assert!(MacAddress::ZERO.is_unicast());
        assert!(!MacAddress::ZERO.is_multicast());
    }

    #[test]
    fn test_mac_address_unicast() {
        let mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert!(mac.is_unicast());
        assert!(!mac.is_multicast());
        assert!(!mac.is_broadcast());
    }

    #[test]
    fn test_mac_address_multicast() {
        let mac = MacAddress::new([0x01, 0x00, 0x5e, 0x00, 0x00, 0x01]);
        assert!(mac.is_multicast());
        assert!(!mac.is_unicast());
    }

    #[test]
    fn test_mac_address_local() {
        let local = MacAddress::new([0x02, 0x00, 0x00, 0x00, 0x00, 0x01]);
        assert!(local.is_local());

        let global = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert!(!global.is_local());
    }

    #[test]
    fn test_ether_type_conversion() {
        assert_eq!(EtherType::from_u16(0x0800), EtherType::Ipv4);
        assert_eq!(EtherType::from_u16(0x0806), EtherType::Arp);
        assert_eq!(EtherType::Ipv4.to_u16(), 0x0800);
        assert_eq!(EtherType::Arp.to_u16(), 0x0806);

        let unknown = EtherType::from_u16(0x1234);
        assert!(matches!(unknown, EtherType::Unknown(0x1234)));
    }

    #[test]
    fn test_ethernet_frame_parse() {
        let frame_bytes = [
            0xff, 0xff, 0xff, 0xff, 0xff, 0xff, // dest MAC (broadcast)
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, // src MAC
            0x08, 0x00, // EtherType (IPv4)
            0xDE, 0xAD, 0xBE, 0xEF, // payload
        ];

        let frame = EthernetFrame::parse(&frame_bytes).unwrap();
        assert!(frame.dest_mac.is_broadcast());
        assert_eq!(frame.src_mac.0, [0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert_eq!(frame.ether_type, EtherType::Ipv4);
        assert_eq!(frame.payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_ethernet_frame_parse_too_short() {
        let short_bytes = [0x00, 0x11, 0x22]; // Only 3 bytes
        let result = EthernetFrame::parse(&short_bytes);
        assert_eq!(result, Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_ethernet_frame_serialize() {
        let payload = [0xDE, 0xAD, 0xBE, 0xEF];
        let frame = EthernetFrame::new(
            MacAddress::BROADCAST,
            MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]),
            EtherType::Ipv4,
            &payload,
        );

        let mut buf = [0u8; 64];
        let len = frame.serialize(&mut buf).unwrap();

        assert_eq!(len, 18); // 14 header + 4 payload
        assert_eq!(&buf[0..6], &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        assert_eq!(&buf[6..12], &[0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        assert_eq!(&buf[12..14], &[0x08, 0x00]);
        assert_eq!(&buf[14..18], &payload);
    }

    #[test]
    fn test_ethernet_frame_roundtrip() {
        let payload = [0x01, 0x02, 0x03, 0x04, 0x05];
        let original = EthernetFrame::new(
            MacAddress::new([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]),
            MacAddress::new([0x11, 0x22, 0x33, 0x44, 0x55, 0x66]),
            EtherType::Arp,
            &payload,
        );

        let mut buf = [0u8; 64];
        let len = original.serialize(&mut buf).unwrap();

        let parsed = EthernetFrame::parse(&buf[..len]).unwrap();
        assert_eq!(original.dest_mac, parsed.dest_mac);
        assert_eq!(original.src_mac, parsed.src_mac);
        assert_eq!(original.ether_type, parsed.ether_type);
        assert_eq!(original.payload, parsed.payload);
    }

    #[test]
    fn test_ethernet_frame_builder() {
        let frame = EthernetFrameBuilder::new()
            .dest(MacAddress::BROADCAST)
            .src(MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]))
            .ether_type(EtherType::Arp)
            .build(&[0x01, 0x02, 0x03]);

        assert!(frame.dest_mac.is_broadcast());
        assert_eq!(frame.ether_type, EtherType::Arp);
        assert_eq!(frame.payload.len(), 3);
    }

    #[test]
    fn test_buffer_too_small() {
        let payload = [0u8; 100];
        let frame = EthernetFrame::new(
            MacAddress::BROADCAST,
            MacAddress::ZERO,
            EtherType::Ipv4,
            &payload,
        );

        let mut small_buf = [0u8; 50]; // Too small for 14 + 100
        let result = frame.serialize(&mut small_buf);
        assert_eq!(result, Err(NetError::BufferTooSmall));
    }
}
