//! Network error types.
//!
//! All network operations return `Result<T, NetError>` to indicate success or failure.
//! Errors are designed to be informative for debugging network issues.

use ruvix_types::KernelError;

/// Network error codes.
///
/// These errors cover all network stack operations from Ethernet to UDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum NetError {
    /// The packet is too short to contain the expected header.
    PacketTooShort = 1,

    /// The packet exceeds the maximum allowed size.
    PacketTooLarge = 2,

    /// Invalid Ethernet frame format.
    InvalidEthernetFrame = 3,

    /// Unknown or unsupported EtherType.
    UnsupportedEtherType = 4,

    /// Invalid IPv4 header format or checksum.
    InvalidIpv4Header = 5,

    /// IPv4 header checksum mismatch.
    Ipv4ChecksumError = 6,

    /// IPv4 version is not 4.
    InvalidIpVersion = 7,

    /// IPv4 header length is invalid.
    InvalidIpHeaderLength = 8,

    /// IPv4 Time To Live expired.
    TtlExpired = 9,

    /// Invalid UDP header format or checksum.
    InvalidUdpHeader = 10,

    /// UDP checksum mismatch.
    UdpChecksumError = 11,

    /// Invalid ICMP header format.
    InvalidIcmpHeader = 12,

    /// Invalid ARP packet format.
    InvalidArpPacket = 13,

    /// No ARP entry found for the target IP.
    ArpNotFound = 14,

    /// ARP cache is full.
    ArpCacheFull = 15,

    /// The socket is not bound to a port.
    SocketNotBound = 16,

    /// The port is already in use.
    PortInUse = 17,

    /// No available ports for ephemeral allocation.
    NoPortsAvailable = 18,

    /// The destination is unreachable.
    DestinationUnreachable = 19,

    /// Network device error.
    DeviceError = 20,

    /// The provided buffer is too small.
    BufferTooSmall = 21,

    /// Operation would block (non-blocking mode).
    WouldBlock = 22,

    /// Invalid port number (0 for source).
    InvalidPort = 23,

    /// Fragment reassembly not supported.
    FragmentationNotSupported = 24,

    /// Kernel error passthrough.
    KernelError(u16) = 255,
}

impl NetError {
    /// Returns a human-readable description of the error.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::PacketTooShort => "Packet too short for header",
            Self::PacketTooLarge => "Packet exceeds maximum size",
            Self::InvalidEthernetFrame => "Invalid Ethernet frame format",
            Self::UnsupportedEtherType => "Unsupported EtherType",
            Self::InvalidIpv4Header => "Invalid IPv4 header",
            Self::Ipv4ChecksumError => "IPv4 checksum mismatch",
            Self::InvalidIpVersion => "Invalid IP version (expected 4)",
            Self::InvalidIpHeaderLength => "Invalid IP header length",
            Self::TtlExpired => "TTL expired",
            Self::InvalidUdpHeader => "Invalid UDP header",
            Self::UdpChecksumError => "UDP checksum mismatch",
            Self::InvalidIcmpHeader => "Invalid ICMP header",
            Self::InvalidArpPacket => "Invalid ARP packet",
            Self::ArpNotFound => "ARP entry not found",
            Self::ArpCacheFull => "ARP cache full",
            Self::SocketNotBound => "Socket not bound",
            Self::PortInUse => "Port already in use",
            Self::NoPortsAvailable => "No ephemeral ports available",
            Self::DestinationUnreachable => "Destination unreachable",
            Self::DeviceError => "Network device error",
            Self::BufferTooSmall => "Buffer too small",
            Self::WouldBlock => "Operation would block",
            Self::InvalidPort => "Invalid port number",
            Self::FragmentationNotSupported => "IP fragmentation not supported",
            Self::KernelError(_) => "Kernel error",
        }
    }

    /// Returns the error code as a u16 (for unit variants).
    #[inline]
    #[must_use]
    pub const fn code(&self) -> u16 {
        match self {
            Self::PacketTooShort => 1,
            Self::PacketTooLarge => 2,
            Self::InvalidEthernetFrame => 3,
            Self::UnsupportedEtherType => 4,
            Self::InvalidIpv4Header => 5,
            Self::Ipv4ChecksumError => 6,
            Self::InvalidIpVersion => 7,
            Self::InvalidIpHeaderLength => 8,
            Self::TtlExpired => 9,
            Self::InvalidUdpHeader => 10,
            Self::UdpChecksumError => 11,
            Self::InvalidIcmpHeader => 12,
            Self::InvalidArpPacket => 13,
            Self::ArpNotFound => 14,
            Self::ArpCacheFull => 15,
            Self::SocketNotBound => 16,
            Self::PortInUse => 17,
            Self::NoPortsAvailable => 18,
            Self::DestinationUnreachable => 19,
            Self::DeviceError => 20,
            Self::BufferTooSmall => 21,
            Self::WouldBlock => 22,
            Self::InvalidPort => 23,
            Self::FragmentationNotSupported => 24,
            Self::KernelError(code) => *code,
        }
    }

    /// Converts from a raw u16 error code.
    #[inline]
    #[must_use]
    pub const fn from_u16(code: u16) -> Option<Self> {
        match code {
            1 => Some(Self::PacketTooShort),
            2 => Some(Self::PacketTooLarge),
            3 => Some(Self::InvalidEthernetFrame),
            4 => Some(Self::UnsupportedEtherType),
            5 => Some(Self::InvalidIpv4Header),
            6 => Some(Self::Ipv4ChecksumError),
            7 => Some(Self::InvalidIpVersion),
            8 => Some(Self::InvalidIpHeaderLength),
            9 => Some(Self::TtlExpired),
            10 => Some(Self::InvalidUdpHeader),
            11 => Some(Self::UdpChecksumError),
            12 => Some(Self::InvalidIcmpHeader),
            13 => Some(Self::InvalidArpPacket),
            14 => Some(Self::ArpNotFound),
            15 => Some(Self::ArpCacheFull),
            16 => Some(Self::SocketNotBound),
            17 => Some(Self::PortInUse),
            18 => Some(Self::NoPortsAvailable),
            19 => Some(Self::DestinationUnreachable),
            20 => Some(Self::DeviceError),
            21 => Some(Self::BufferTooSmall),
            22 => Some(Self::WouldBlock),
            23 => Some(Self::InvalidPort),
            24 => Some(Self::FragmentationNotSupported),
            _ => None,
        }
    }
}

impl From<KernelError> for NetError {
    fn from(err: KernelError) -> Self {
        Self::KernelError(err as u16)
    }
}

/// Result type for network operations.
pub type NetResult<T> = Result<T, NetError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_roundtrip() {
        let errors = [
            NetError::PacketTooShort,
            NetError::InvalidEthernetFrame,
            NetError::ArpNotFound,
            NetError::WouldBlock,
        ];

        for err in errors {
            let code = err.code();
            let parsed = NetError::from_u16(code).unwrap();
            assert_eq!(err, parsed);
        }
    }

    #[test]
    fn test_error_as_str() {
        assert_eq!(NetError::PacketTooShort.as_str(), "Packet too short for header");
        assert_eq!(NetError::ArpNotFound.as_str(), "ARP entry not found");
    }

    #[test]
    fn test_kernel_error_conversion() {
        let kernel_err = KernelError::OutOfMemory;
        let net_err: NetError = kernel_err.into();
        assert!(matches!(net_err, NetError::KernelError(5)));
    }
}
