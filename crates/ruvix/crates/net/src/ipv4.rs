//! IPv4 header handling.
//!
//! This module provides types and functions for parsing and serializing
//! IPv4 headers as per RFC 791.
//!
//! ## Header Format
//!
//! ```text
//! 0                   1                   2                   3
//! 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |Version|  IHL  |Type of Service|          Total Length         |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |         Identification        |Flags|      Fragment Offset    |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |  Time to Live |    Protocol   |         Header Checksum       |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                       Source Address                          |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                    Destination Address                        |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```

use crate::error::{NetError, NetResult};
use crate::IPV4_HEADER_MIN_SIZE;

/// IPv4 address.
///
/// A 32-bit network address in network byte order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct Ipv4Addr(pub [u8; 4]);

impl Ipv4Addr {
    /// Broadcast address (255.255.255.255).
    pub const BROADCAST: Self = Self([255, 255, 255, 255]);

    /// Unspecified address (0.0.0.0).
    pub const UNSPECIFIED: Self = Self([0, 0, 0, 0]);

    /// Localhost address (127.0.0.1).
    pub const LOCALHOST: Self = Self([127, 0, 0, 1]);

    /// Creates a new IPv4 address from octets.
    #[inline]
    #[must_use]
    pub const fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
        Self([a, b, c, d])
    }

    /// Creates an IPv4 address from a byte array.
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 4]) -> Self {
        Self(bytes)
    }

    /// Returns the address as a byte slice.
    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 4] {
        &self.0
    }

    /// Returns the address as a u32 in network byte order.
    #[inline]
    #[must_use]
    pub const fn to_u32(&self) -> u32 {
        u32::from_be_bytes(self.0)
    }

    /// Creates an address from a u32 in network byte order.
    #[inline]
    #[must_use]
    pub const fn from_u32(addr: u32) -> Self {
        Self(addr.to_be_bytes())
    }

    /// Checks if this is the broadcast address.
    #[inline]
    #[must_use]
    pub const fn is_broadcast(&self) -> bool {
        self.0[0] == 255 && self.0[1] == 255 && self.0[2] == 255 && self.0[3] == 255
    }

    /// Checks if this is the unspecified address.
    #[inline]
    #[must_use]
    pub const fn is_unspecified(&self) -> bool {
        self.0[0] == 0 && self.0[1] == 0 && self.0[2] == 0 && self.0[3] == 0
    }

    /// Checks if this is a loopback address (127.0.0.0/8).
    #[inline]
    #[must_use]
    pub const fn is_loopback(&self) -> bool {
        self.0[0] == 127
    }

    /// Checks if this is a multicast address (224.0.0.0/4).
    #[inline]
    #[must_use]
    pub const fn is_multicast(&self) -> bool {
        self.0[0] >= 224 && self.0[0] <= 239
    }

    /// Checks if this is a link-local address (169.254.0.0/16).
    #[inline]
    #[must_use]
    pub const fn is_link_local(&self) -> bool {
        self.0[0] == 169 && self.0[1] == 254
    }

    /// Checks if this is a private address (RFC 1918).
    #[inline]
    #[must_use]
    pub const fn is_private(&self) -> bool {
        // 10.0.0.0/8
        self.0[0] == 10
        // 172.16.0.0/12
        || (self.0[0] == 172 && self.0[1] >= 16 && self.0[1] <= 31)
        // 192.168.0.0/16
        || (self.0[0] == 192 && self.0[1] == 168)
    }

    /// Parses an IPv4 address from a byte slice.
    #[inline]
    pub fn parse(bytes: &[u8]) -> NetResult<Self> {
        if bytes.len() < 4 {
            return Err(NetError::PacketTooShort);
        }
        let mut addr = [0u8; 4];
        addr.copy_from_slice(&bytes[..4]);
        Ok(Self(addr))
    }

    /// Checks if this address is in the same subnet as another.
    #[inline]
    #[must_use]
    pub const fn is_same_subnet(&self, other: &Self, mask: &Self) -> bool {
        (self.0[0] & mask.0[0]) == (other.0[0] & mask.0[0])
            && (self.0[1] & mask.0[1]) == (other.0[1] & mask.0[1])
            && (self.0[2] & mask.0[2]) == (other.0[2] & mask.0[2])
            && (self.0[3] & mask.0[3]) == (other.0[3] & mask.0[3])
    }
}

/// IP protocol numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Protocol {
    /// Internet Control Message Protocol.
    Icmp = 1,
    /// Transmission Control Protocol.
    Tcp = 6,
    /// User Datagram Protocol.
    Udp = 17,
    /// Unknown protocol.
    Unknown(u8) = 255,
}

impl Protocol {
    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Icmp,
            6 => Self::Tcp,
            17 => Self::Udp,
            other => Self::Unknown(other),
        }
    }

    /// Converts to a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Icmp => 1,
            Self::Tcp => 6,
            Self::Udp => 17,
            Self::Unknown(v) => v,
        }
    }
}

/// IPv4 header flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ipv4Flags {
    /// Don't Fragment flag.
    pub dont_fragment: bool,
    /// More Fragments flag.
    pub more_fragments: bool,
}

impl Ipv4Flags {
    /// Parses flags from the flags/fragment offset word.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: u16) -> Self {
        Self {
            dont_fragment: (raw & 0x4000) != 0,
            more_fragments: (raw & 0x2000) != 0,
        }
    }

    /// Converts to the raw flags bits (upper 3 bits of the word).
    #[inline]
    #[must_use]
    pub const fn to_raw(self) -> u16 {
        let mut flags = 0u16;
        if self.dont_fragment {
            flags |= 0x4000;
        }
        if self.more_fragments {
            flags |= 0x2000;
        }
        flags
    }
}

/// IPv4 header.
///
/// Represents a parsed IPv4 header without options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ipv4Header {
    /// IP version (always 4).
    pub version: u8,
    /// Internet Header Length in 32-bit words (minimum 5).
    pub ihl: u8,
    /// Differentiated Services Code Point.
    pub dscp: u8,
    /// Explicit Congestion Notification.
    pub ecn: u8,
    /// Total length including header and payload.
    pub total_length: u16,
    /// Identification for fragment reassembly.
    pub identification: u16,
    /// Flags (DF, MF).
    pub flags: Ipv4Flags,
    /// Fragment offset in 8-byte units.
    pub fragment_offset: u16,
    /// Time to live (hop limit).
    pub ttl: u8,
    /// Protocol of the encapsulated data.
    pub protocol: Protocol,
    /// Header checksum.
    pub checksum: u16,
    /// Source IP address.
    pub src_addr: Ipv4Addr,
    /// Destination IP address.
    pub dst_addr: Ipv4Addr,
}

impl Ipv4Header {
    /// Default TTL value for outgoing packets.
    pub const DEFAULT_TTL: u8 = 64;

    /// Parses an IPv4 header from a byte buffer.
    ///
    /// # Errors
    ///
    /// - `PacketTooShort` if buffer is smaller than 20 bytes
    /// - `InvalidIpVersion` if version is not 4
    /// - `InvalidIpHeaderLength` if IHL is less than 5
    #[inline]
    pub fn parse(bytes: &[u8]) -> NetResult<(Self, &[u8])> {
        if bytes.len() < IPV4_HEADER_MIN_SIZE {
            return Err(NetError::PacketTooShort);
        }

        let version_ihl = bytes[0];
        let version = version_ihl >> 4;
        let ihl = version_ihl & 0x0F;

        if version != 4 {
            return Err(NetError::InvalidIpVersion);
        }

        if ihl < 5 {
            return Err(NetError::InvalidIpHeaderLength);
        }

        let header_len = (ihl as usize) * 4;
        if bytes.len() < header_len {
            return Err(NetError::PacketTooShort);
        }

        // CVE-005 FIX: Validate total_length >= header_len
        let total_length = u16::from_be_bytes([bytes[2], bytes[3]]);
        if (total_length as usize) < header_len {
            return Err(NetError::InvalidIpHeaderLength);
        }

        let tos = bytes[1];
        let dscp = tos >> 2;
        let ecn = tos & 0x03;

        // total_length already read and validated above
        let identification = u16::from_be_bytes([bytes[4], bytes[5]]);

        let flags_frag = u16::from_be_bytes([bytes[6], bytes[7]]);
        let flags = Ipv4Flags::from_raw(flags_frag);
        let fragment_offset = flags_frag & 0x1FFF;

        let ttl = bytes[8];
        let protocol = Protocol::from_u8(bytes[9]);
        let checksum = u16::from_be_bytes([bytes[10], bytes[11]]);

        let src_addr = Ipv4Addr::parse(&bytes[12..16])?;
        let dst_addr = Ipv4Addr::parse(&bytes[16..20])?;

        let header = Self {
            version,
            ihl,
            dscp,
            ecn,
            total_length,
            identification,
            flags,
            fragment_offset,
            ttl,
            protocol,
            checksum,
            src_addr,
            dst_addr,
        };

        // Return payload (after header, including any options)
        let payload_start = header_len;
        let payload_end = (total_length as usize).min(bytes.len());
        let payload = &bytes[payload_start..payload_end];

        Ok((header, payload))
    }

    /// Serializes the IPv4 header into a buffer.
    ///
    /// The checksum is calculated and filled in automatically.
    /// Returns the number of bytes written (always 20 for basic header).
    ///
    /// # Errors
    ///
    /// Returns `NetError::BufferTooSmall` if buffer is smaller than 20 bytes.
    #[inline]
    pub fn serialize(&self, buf: &mut [u8]) -> NetResult<usize> {
        if buf.len() < IPV4_HEADER_MIN_SIZE {
            return Err(NetError::BufferTooSmall);
        }

        // Version and IHL
        buf[0] = (self.version << 4) | self.ihl;

        // DSCP and ECN
        buf[1] = (self.dscp << 2) | self.ecn;

        // Total length
        buf[2..4].copy_from_slice(&self.total_length.to_be_bytes());

        // Identification
        buf[4..6].copy_from_slice(&self.identification.to_be_bytes());

        // Flags and fragment offset
        let flags_frag = self.flags.to_raw() | self.fragment_offset;
        buf[6..8].copy_from_slice(&flags_frag.to_be_bytes());

        // TTL
        buf[8] = self.ttl;

        // Protocol
        buf[9] = self.protocol.to_u8();

        // Checksum (set to 0 for calculation)
        buf[10] = 0;
        buf[11] = 0;

        // Source address
        buf[12..16].copy_from_slice(&self.src_addr.0);

        // Destination address
        buf[16..20].copy_from_slice(&self.dst_addr.0);

        // Calculate and set checksum
        let checksum = Self::compute_checksum(&buf[..IPV4_HEADER_MIN_SIZE]);
        buf[10..12].copy_from_slice(&checksum.to_be_bytes());

        Ok(IPV4_HEADER_MIN_SIZE)
    }

    /// Computes the IPv4 header checksum.
    ///
    /// The checksum field should be set to 0 before calling this.
    #[inline]
    #[must_use]
    pub fn compute_checksum(header: &[u8]) -> u16 {
        let mut sum: u32 = 0;

        // Sum all 16-bit words
        for chunk in header.chunks(2) {
            let word = if chunk.len() == 2 {
                u16::from_be_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_be_bytes([chunk[0], 0])
            };
            sum += u32::from(word);
        }

        // Fold 32-bit sum to 16 bits
        while sum > 0xFFFF {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        // One's complement
        !sum as u16
    }

    /// Verifies the header checksum.
    #[inline]
    #[must_use]
    pub fn verify_checksum(header: &[u8]) -> bool {
        Self::compute_checksum(header) == 0
    }

    /// Returns the header length in bytes.
    #[inline]
    #[must_use]
    pub const fn header_len(&self) -> usize {
        (self.ihl as usize) * 4
    }

    /// Returns the payload length.
    #[inline]
    #[must_use]
    pub const fn payload_len(&self) -> usize {
        (self.total_length as usize).saturating_sub(self.header_len())
    }

    /// Creates a new IPv4 header for outgoing packets.
    #[inline]
    #[must_use]
    pub const fn new(
        src_addr: Ipv4Addr,
        dst_addr: Ipv4Addr,
        protocol: Protocol,
        payload_len: u16,
    ) -> Self {
        Self {
            version: 4,
            ihl: 5,
            dscp: 0,
            ecn: 0,
            total_length: IPV4_HEADER_MIN_SIZE as u16 + payload_len,
            identification: 0,
            flags: Ipv4Flags {
                dont_fragment: true,
                more_fragments: false,
            },
            fragment_offset: 0,
            ttl: Self::DEFAULT_TTL,
            protocol,
            checksum: 0, // Will be computed during serialization
            src_addr,
            dst_addr,
        }
    }

    /// Decrements TTL and returns whether the packet should be forwarded.
    #[inline]
    pub fn decrement_ttl(&mut self) -> NetResult<()> {
        if self.ttl == 0 {
            return Err(NetError::TtlExpired);
        }
        self.ttl -= 1;
        if self.ttl == 0 {
            return Err(NetError::TtlExpired);
        }
        Ok(())
    }
}

/// Builder for constructing IPv4 headers.
pub struct Ipv4HeaderBuilder {
    header: Ipv4Header,
}

impl Ipv4HeaderBuilder {
    /// Creates a new header builder with default values.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            header: Ipv4Header {
                version: 4,
                ihl: 5,
                dscp: 0,
                ecn: 0,
                total_length: IPV4_HEADER_MIN_SIZE as u16,
                identification: 0,
                flags: Ipv4Flags {
                    dont_fragment: true,
                    more_fragments: false,
                },
                fragment_offset: 0,
                ttl: Ipv4Header::DEFAULT_TTL,
                protocol: Protocol::Udp,
                checksum: 0,
                src_addr: Ipv4Addr::UNSPECIFIED,
                dst_addr: Ipv4Addr::UNSPECIFIED,
            },
        }
    }

    /// Sets the source address.
    #[inline]
    #[must_use]
    pub const fn src(mut self, addr: Ipv4Addr) -> Self {
        self.header.src_addr = addr;
        self
    }

    /// Sets the destination address.
    #[inline]
    #[must_use]
    pub const fn dst(mut self, addr: Ipv4Addr) -> Self {
        self.header.dst_addr = addr;
        self
    }

    /// Sets the protocol.
    #[inline]
    #[must_use]
    pub const fn protocol(mut self, proto: Protocol) -> Self {
        self.header.protocol = proto;
        self
    }

    /// Sets the TTL.
    #[inline]
    #[must_use]
    pub const fn ttl(mut self, ttl: u8) -> Self {
        self.header.ttl = ttl;
        self
    }

    /// Sets the identification field.
    #[inline]
    #[must_use]
    pub const fn identification(mut self, id: u16) -> Self {
        self.header.identification = id;
        self
    }

    /// Sets the Don't Fragment flag.
    #[inline]
    #[must_use]
    pub const fn dont_fragment(mut self, df: bool) -> Self {
        self.header.flags.dont_fragment = df;
        self
    }

    /// Sets the DSCP value.
    #[inline]
    #[must_use]
    pub const fn dscp(mut self, dscp: u8) -> Self {
        self.header.dscp = dscp & 0x3F;
        self
    }

    /// Builds the header with the given payload length.
    #[inline]
    #[must_use]
    pub const fn build(mut self, payload_len: u16) -> Ipv4Header {
        self.header.total_length = IPV4_HEADER_MIN_SIZE as u16 + payload_len;
        self.header
    }
}

impl Default for Ipv4HeaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipv4_addr_new() {
        let addr = Ipv4Addr::new(192, 168, 1, 1);
        assert_eq!(addr.0, [192, 168, 1, 1]);
    }

    #[test]
    fn test_ipv4_addr_special() {
        assert!(Ipv4Addr::BROADCAST.is_broadcast());
        assert!(Ipv4Addr::UNSPECIFIED.is_unspecified());
        assert!(Ipv4Addr::LOCALHOST.is_loopback());
    }

    #[test]
    fn test_ipv4_addr_classification() {
        // Private addresses
        assert!(Ipv4Addr::new(10, 0, 0, 1).is_private());
        assert!(Ipv4Addr::new(172, 16, 0, 1).is_private());
        assert!(Ipv4Addr::new(192, 168, 1, 1).is_private());
        assert!(!Ipv4Addr::new(8, 8, 8, 8).is_private());

        // Multicast
        assert!(Ipv4Addr::new(224, 0, 0, 1).is_multicast());
        assert!(Ipv4Addr::new(239, 255, 255, 255).is_multicast());
        assert!(!Ipv4Addr::new(192, 168, 1, 1).is_multicast());

        // Link-local
        assert!(Ipv4Addr::new(169, 254, 1, 1).is_link_local());
        assert!(!Ipv4Addr::new(192, 168, 1, 1).is_link_local());
    }

    #[test]
    fn test_ipv4_addr_same_subnet() {
        let addr1 = Ipv4Addr::new(192, 168, 1, 10);
        let addr2 = Ipv4Addr::new(192, 168, 1, 20);
        let addr3 = Ipv4Addr::new(192, 168, 2, 10);
        let mask = Ipv4Addr::new(255, 255, 255, 0);

        assert!(addr1.is_same_subnet(&addr2, &mask));
        assert!(!addr1.is_same_subnet(&addr3, &mask));
    }

    #[test]
    fn test_ipv4_addr_u32_conversion() {
        let addr = Ipv4Addr::new(192, 168, 1, 1);
        let as_u32 = addr.to_u32();
        let back = Ipv4Addr::from_u32(as_u32);
        assert_eq!(addr, back);
    }

    #[test]
    fn test_protocol_conversion() {
        assert_eq!(Protocol::from_u8(1), Protocol::Icmp);
        assert_eq!(Protocol::from_u8(6), Protocol::Tcp);
        assert_eq!(Protocol::from_u8(17), Protocol::Udp);
        assert!(matches!(Protocol::from_u8(99), Protocol::Unknown(99)));

        assert_eq!(Protocol::Icmp.to_u8(), 1);
        assert_eq!(Protocol::Udp.to_u8(), 17);
    }

    #[test]
    fn test_ipv4_header_parse() {
        // Valid IPv4 header with UDP
        #[rustfmt::skip]
        let packet = [
            0x45, 0x00, // Version=4, IHL=5, DSCP=0, ECN=0
            0x00, 0x1C, // Total length = 28
            0x00, 0x01, // Identification
            0x40, 0x00, // Flags=DF, Fragment offset=0
            0x40, 0x11, // TTL=64, Protocol=UDP
            0x00, 0x00, // Checksum (will be wrong, but we're testing parse)
            0xC0, 0xA8, 0x01, 0x01, // Src: 192.168.1.1
            0xC0, 0xA8, 0x01, 0x02, // Dst: 192.168.1.2
            0xDE, 0xAD, 0xBE, 0xEF, // Payload
        ];

        let (header, payload) = Ipv4Header::parse(&packet).unwrap();

        assert_eq!(header.version, 4);
        assert_eq!(header.ihl, 5);
        assert_eq!(header.total_length, 28);
        assert_eq!(header.ttl, 64);
        assert_eq!(header.protocol, Protocol::Udp);
        assert!(header.flags.dont_fragment);
        assert!(!header.flags.more_fragments);
        assert_eq!(header.src_addr, Ipv4Addr::new(192, 168, 1, 1));
        assert_eq!(header.dst_addr, Ipv4Addr::new(192, 168, 1, 2));
        assert_eq!(payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_ipv4_header_serialize() {
        let header = Ipv4Header::new(
            Ipv4Addr::new(192, 168, 1, 1),
            Ipv4Addr::new(192, 168, 1, 2),
            Protocol::Udp,
            8, // 8-byte payload
        );

        let mut buf = [0u8; 64];
        let len = header.serialize(&mut buf).unwrap();

        assert_eq!(len, 20);

        // Verify version and IHL
        assert_eq!(buf[0], 0x45);

        // Verify addresses
        assert_eq!(&buf[12..16], &[192, 168, 1, 1]);
        assert_eq!(&buf[16..20], &[192, 168, 1, 2]);

        // Verify checksum is valid
        assert!(Ipv4Header::verify_checksum(&buf[..20]));
    }

    #[test]
    fn test_ipv4_header_roundtrip() {
        let original = Ipv4HeaderBuilder::new()
            .src(Ipv4Addr::new(10, 0, 0, 1))
            .dst(Ipv4Addr::new(10, 0, 0, 2))
            .protocol(Protocol::Icmp)
            .ttl(128)
            .identification(0x1234)
            .build(100);

        let mut buf = [0u8; 256];
        original.serialize(&mut buf).unwrap();

        let (parsed, _) = Ipv4Header::parse(&buf).unwrap();

        assert_eq!(original.src_addr, parsed.src_addr);
        assert_eq!(original.dst_addr, parsed.dst_addr);
        assert_eq!(original.protocol, parsed.protocol);
        assert_eq!(original.ttl, parsed.ttl);
        assert_eq!(original.identification, parsed.identification);
    }

    #[test]
    fn test_ipv4_header_checksum() {
        // Example from RFC 1071
        #[rustfmt::skip]
        let header = [
            0x45, 0x00, 0x00, 0x73,
            0x00, 0x00, 0x40, 0x00,
            0x40, 0x11, 0x00, 0x00, // Checksum = 0 for calculation
            0xc0, 0xa8, 0x00, 0x01,
            0xc0, 0xa8, 0x00, 0xc7,
        ];

        let checksum = Ipv4Header::compute_checksum(&header);
        assert_ne!(checksum, 0); // Should compute a non-zero checksum
    }

    #[test]
    fn test_ipv4_header_too_short() {
        let short = [0x45, 0x00, 0x00]; // Only 3 bytes
        assert_eq!(Ipv4Header::parse(&short), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_ipv4_header_wrong_version() {
        let mut packet = [0u8; 20];
        packet[0] = 0x65; // Version 6, IHL 5
        assert_eq!(Ipv4Header::parse(&packet), Err(NetError::InvalidIpVersion));
    }

    #[test]
    fn test_ipv4_header_invalid_ihl() {
        let mut packet = [0u8; 20];
        packet[0] = 0x43; // Version 4, IHL 3 (invalid, minimum is 5)
        assert_eq!(
            Ipv4Header::parse(&packet),
            Err(NetError::InvalidIpHeaderLength)
        );
    }

    #[test]
    fn test_ipv4_flags() {
        let df_only = Ipv4Flags::from_raw(0x4000);
        assert!(df_only.dont_fragment);
        assert!(!df_only.more_fragments);

        let mf_only = Ipv4Flags::from_raw(0x2000);
        assert!(!mf_only.dont_fragment);
        assert!(mf_only.more_fragments);

        let both = Ipv4Flags::from_raw(0x6000);
        assert!(both.dont_fragment);
        assert!(both.more_fragments);

        // Roundtrip
        assert_eq!(df_only.to_raw(), 0x4000);
        assert_eq!(mf_only.to_raw(), 0x2000);
    }

    #[test]
    fn test_decrement_ttl() {
        let mut header = Ipv4Header::new(
            Ipv4Addr::LOCALHOST,
            Ipv4Addr::LOCALHOST,
            Protocol::Icmp,
            0,
        );

        header.ttl = 2;
        assert!(header.decrement_ttl().is_ok());
        assert_eq!(header.ttl, 1);

        assert!(header.decrement_ttl().is_err()); // TTL becomes 0

        header.ttl = 0;
        assert_eq!(header.decrement_ttl(), Err(NetError::TtlExpired));
    }
}
