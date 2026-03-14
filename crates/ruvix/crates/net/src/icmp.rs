//! Internet Control Message Protocol (ICMP) handling.
//!
//! This module provides types and functions for parsing and serializing
//! ICMP messages as per RFC 792.
//!
//! ## Header Format
//!
//! ```text
//! 0                   1                   2                   3
//! 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |     Type      |     Code      |          Checksum             |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                             Data                              |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```

use crate::error::{NetError, NetResult};
use crate::ICMP_HEADER_SIZE;

/// ICMP message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IcmpType {
    /// Echo Reply (pong).
    EchoReply = 0,
    /// Destination Unreachable.
    DestinationUnreachable = 3,
    /// Source Quench (deprecated).
    SourceQuench = 4,
    /// Redirect.
    Redirect = 5,
    /// Echo Request (ping).
    EchoRequest = 8,
    /// Time Exceeded.
    TimeExceeded = 11,
    /// Parameter Problem.
    ParameterProblem = 12,
    /// Timestamp Request.
    TimestampRequest = 13,
    /// Timestamp Reply.
    TimestampReply = 14,
    /// Unknown type.
    Unknown(u8) = 255,
}

impl IcmpType {
    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::EchoReply,
            3 => Self::DestinationUnreachable,
            4 => Self::SourceQuench,
            5 => Self::Redirect,
            8 => Self::EchoRequest,
            11 => Self::TimeExceeded,
            12 => Self::ParameterProblem,
            13 => Self::TimestampRequest,
            14 => Self::TimestampReply,
            other => Self::Unknown(other),
        }
    }

    /// Converts to a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::EchoReply => 0,
            Self::DestinationUnreachable => 3,
            Self::SourceQuench => 4,
            Self::Redirect => 5,
            Self::EchoRequest => 8,
            Self::TimeExceeded => 11,
            Self::ParameterProblem => 12,
            Self::TimestampRequest => 13,
            Self::TimestampReply => 14,
            Self::Unknown(v) => v,
        }
    }
}

/// Destination Unreachable codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DestUnreachableCode {
    /// Network unreachable.
    NetworkUnreachable = 0,
    /// Host unreachable.
    HostUnreachable = 1,
    /// Protocol unreachable.
    ProtocolUnreachable = 2,
    /// Port unreachable.
    PortUnreachable = 3,
    /// Fragmentation needed but DF set.
    FragmentationNeeded = 4,
    /// Source route failed.
    SourceRouteFailed = 5,
    /// Destination network unknown.
    DestNetworkUnknown = 6,
    /// Destination host unknown.
    DestHostUnknown = 7,
    /// Unknown code.
    Unknown(u8) = 255,
}

impl DestUnreachableCode {
    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::NetworkUnreachable,
            1 => Self::HostUnreachable,
            2 => Self::ProtocolUnreachable,
            3 => Self::PortUnreachable,
            4 => Self::FragmentationNeeded,
            5 => Self::SourceRouteFailed,
            6 => Self::DestNetworkUnknown,
            7 => Self::DestHostUnknown,
            other => Self::Unknown(other),
        }
    }

    /// Converts to a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::NetworkUnreachable => 0,
            Self::HostUnreachable => 1,
            Self::ProtocolUnreachable => 2,
            Self::PortUnreachable => 3,
            Self::FragmentationNeeded => 4,
            Self::SourceRouteFailed => 5,
            Self::DestNetworkUnknown => 6,
            Self::DestHostUnknown => 7,
            Self::Unknown(v) => v,
        }
    }
}

/// Time Exceeded codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TimeExceededCode {
    /// TTL exceeded in transit.
    TtlExceeded = 0,
    /// Fragment reassembly time exceeded.
    FragmentReassemblyExceeded = 1,
    /// Unknown code.
    Unknown(u8) = 255,
}

impl TimeExceededCode {
    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::TtlExceeded,
            1 => Self::FragmentReassemblyExceeded,
            other => Self::Unknown(other),
        }
    }

    /// Converts to a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::TtlExceeded => 0,
            Self::FragmentReassemblyExceeded => 1,
            Self::Unknown(v) => v,
        }
    }
}

/// ICMP header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcmpHeader {
    /// ICMP message type.
    pub icmp_type: IcmpType,
    /// Type-specific code.
    pub code: u8,
    /// Checksum.
    pub checksum: u16,
    /// Rest of header (type-specific, 4 bytes).
    pub rest_of_header: [u8; 4],
}

impl IcmpHeader {
    /// Parses an ICMP header from a byte buffer.
    ///
    /// # Errors
    ///
    /// Returns `NetError::PacketTooShort` if buffer is smaller than 8 bytes.
    #[inline]
    pub fn parse(bytes: &[u8]) -> NetResult<(Self, &[u8])> {
        if bytes.len() < ICMP_HEADER_SIZE {
            return Err(NetError::PacketTooShort);
        }

        let icmp_type = IcmpType::from_u8(bytes[0]);
        let code = bytes[1];
        let checksum = u16::from_be_bytes([bytes[2], bytes[3]]);
        let mut rest_of_header = [0u8; 4];
        rest_of_header.copy_from_slice(&bytes[4..8]);

        let header = Self {
            icmp_type,
            code,
            checksum,
            rest_of_header,
        };

        let payload = &bytes[ICMP_HEADER_SIZE..];

        Ok((header, payload))
    }

    /// Serializes the ICMP header into a buffer.
    ///
    /// Returns the number of bytes written (always 8 on success).
    ///
    /// # Errors
    ///
    /// Returns `NetError::BufferTooSmall` if buffer is smaller than 8 bytes.
    #[inline]
    pub fn serialize(&self, buf: &mut [u8]) -> NetResult<usize> {
        if buf.len() < ICMP_HEADER_SIZE {
            return Err(NetError::BufferTooSmall);
        }

        buf[0] = self.icmp_type.to_u8();
        buf[1] = self.code;
        buf[2..4].copy_from_slice(&self.checksum.to_be_bytes());
        buf[4..8].copy_from_slice(&self.rest_of_header);

        Ok(ICMP_HEADER_SIZE)
    }

    /// Computes the ICMP checksum.
    ///
    /// The checksum field should be set to 0 before calling this.
    #[must_use]
    pub fn compute_checksum(icmp_message: &[u8]) -> u16 {
        let mut sum: u32 = 0;

        for chunk in icmp_message.chunks(2) {
            let word = if chunk.len() == 2 {
                u16::from_be_bytes([chunk[0], chunk[1]])
            } else {
                u16::from_be_bytes([chunk[0], 0])
            };
            sum += u32::from(word);
        }

        // Fold
        while sum > 0xFFFF {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }

        !sum as u16
    }

    /// Verifies the ICMP checksum.
    #[must_use]
    pub fn verify_checksum(icmp_message: &[u8]) -> bool {
        Self::compute_checksum(icmp_message) == 0
    }
}

/// Echo Request/Reply message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcmpEcho<'a> {
    /// Identifier (used to match requests/replies).
    pub identifier: u16,
    /// Sequence number.
    pub sequence: u16,
    /// Echo data payload.
    pub data: &'a [u8],
}

impl<'a> IcmpEcho<'a> {
    /// Creates a new Echo Request.
    #[inline]
    #[must_use]
    pub const fn request(identifier: u16, sequence: u16, data: &'a [u8]) -> (IcmpHeader, Self) {
        let header = IcmpHeader {
            icmp_type: IcmpType::EchoRequest,
            code: 0,
            checksum: 0,
            rest_of_header: [
                (identifier >> 8) as u8,
                identifier as u8,
                (sequence >> 8) as u8,
                sequence as u8,
            ],
        };
        let echo = Self {
            identifier,
            sequence,
            data,
        };
        (header, echo)
    }

    /// Creates a new Echo Reply.
    #[inline]
    #[must_use]
    pub const fn reply(identifier: u16, sequence: u16, data: &'a [u8]) -> (IcmpHeader, Self) {
        let header = IcmpHeader {
            icmp_type: IcmpType::EchoReply,
            code: 0,
            checksum: 0,
            rest_of_header: [
                (identifier >> 8) as u8,
                identifier as u8,
                (sequence >> 8) as u8,
                sequence as u8,
            ],
        };
        let echo = Self {
            identifier,
            sequence,
            data,
        };
        (header, echo)
    }

    /// Parses an Echo message from header and payload.
    #[inline]
    pub fn parse(header: &IcmpHeader, payload: &'a [u8]) -> NetResult<Self> {
        if !matches!(
            header.icmp_type,
            IcmpType::EchoRequest | IcmpType::EchoReply
        ) {
            return Err(NetError::InvalidIcmpHeader);
        }

        let identifier =
            u16::from_be_bytes([header.rest_of_header[0], header.rest_of_header[1]]);
        let sequence = u16::from_be_bytes([header.rest_of_header[2], header.rest_of_header[3]]);

        Ok(Self {
            identifier,
            sequence,
            data: payload,
        })
    }

    /// Serializes the Echo message (header + data) into a buffer.
    ///
    /// Computes and fills in the checksum automatically.
    pub fn serialize(&self, is_request: bool, buf: &mut [u8]) -> NetResult<usize> {
        let total_len = ICMP_HEADER_SIZE + self.data.len();
        if buf.len() < total_len {
            return Err(NetError::BufferTooSmall);
        }

        // Build header
        let (header, _) = if is_request {
            Self::request(self.identifier, self.sequence, self.data)
        } else {
            Self::reply(self.identifier, self.sequence, self.data)
        };

        // Serialize header with checksum = 0
        header.serialize(buf)?;

        // Copy data
        buf[ICMP_HEADER_SIZE..total_len].copy_from_slice(self.data);

        // Compute and fill checksum
        let checksum = IcmpHeader::compute_checksum(&buf[..total_len]);
        buf[2..4].copy_from_slice(&checksum.to_be_bytes());

        Ok(total_len)
    }

    /// Creates a reply from a request.
    #[inline]
    #[must_use]
    pub const fn to_reply(&self) -> (IcmpHeader, Self) {
        Self::reply(self.identifier, self.sequence, self.data)
    }
}

/// Destination Unreachable message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcmpDestUnreachable<'a> {
    /// Unreachable code.
    pub code: DestUnreachableCode,
    /// Next-hop MTU (for FragmentationNeeded).
    pub next_hop_mtu: u16,
    /// Original IP header + first 8 bytes of original datagram.
    pub original_data: &'a [u8],
}

impl<'a> IcmpDestUnreachable<'a> {
    /// Creates a new Destination Unreachable message.
    #[inline]
    #[must_use]
    pub const fn new(
        code: DestUnreachableCode,
        next_hop_mtu: u16,
        original_data: &'a [u8],
    ) -> (IcmpHeader, Self) {
        let header = IcmpHeader {
            icmp_type: IcmpType::DestinationUnreachable,
            code: code.to_u8(),
            checksum: 0,
            rest_of_header: [0, 0, (next_hop_mtu >> 8) as u8, next_hop_mtu as u8],
        };
        let msg = Self {
            code,
            next_hop_mtu,
            original_data,
        };
        (header, msg)
    }

    /// Parses from header and payload.
    #[inline]
    pub fn parse(header: &IcmpHeader, payload: &'a [u8]) -> NetResult<Self> {
        if header.icmp_type != IcmpType::DestinationUnreachable {
            return Err(NetError::InvalidIcmpHeader);
        }

        let code = DestUnreachableCode::from_u8(header.code);
        let next_hop_mtu =
            u16::from_be_bytes([header.rest_of_header[2], header.rest_of_header[3]]);

        Ok(Self {
            code,
            next_hop_mtu,
            original_data: payload,
        })
    }
}

/// Time Exceeded message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IcmpTimeExceeded<'a> {
    /// Time exceeded code.
    pub code: TimeExceededCode,
    /// Original IP header + first 8 bytes of original datagram.
    pub original_data: &'a [u8],
}

impl<'a> IcmpTimeExceeded<'a> {
    /// Creates a new Time Exceeded message.
    #[inline]
    #[must_use]
    pub const fn new(code: TimeExceededCode, original_data: &'a [u8]) -> (IcmpHeader, Self) {
        let header = IcmpHeader {
            icmp_type: IcmpType::TimeExceeded,
            code: code.to_u8(),
            checksum: 0,
            rest_of_header: [0, 0, 0, 0],
        };
        let msg = Self {
            code,
            original_data,
        };
        (header, msg)
    }

    /// Parses from header and payload.
    #[inline]
    pub fn parse(header: &IcmpHeader, payload: &'a [u8]) -> NetResult<Self> {
        if header.icmp_type != IcmpType::TimeExceeded {
            return Err(NetError::InvalidIcmpHeader);
        }

        let code = TimeExceededCode::from_u8(header.code);

        Ok(Self {
            code,
            original_data: payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icmp_type_conversion() {
        assert_eq!(IcmpType::from_u8(0), IcmpType::EchoReply);
        assert_eq!(IcmpType::from_u8(8), IcmpType::EchoRequest);
        assert_eq!(IcmpType::from_u8(3), IcmpType::DestinationUnreachable);
        assert!(matches!(IcmpType::from_u8(99), IcmpType::Unknown(99)));

        assert_eq!(IcmpType::EchoReply.to_u8(), 0);
        assert_eq!(IcmpType::EchoRequest.to_u8(), 8);
    }

    #[test]
    fn test_icmp_header_parse() {
        #[rustfmt::skip]
        let data = [
            8,    // Type: Echo Request
            0,    // Code
            0x00, 0x00, // Checksum (will be wrong)
            0x00, 0x01, // Identifier
            0x00, 0x02, // Sequence
            0xDE, 0xAD, // Payload
        ];

        let (header, payload) = IcmpHeader::parse(&data).unwrap();
        assert_eq!(header.icmp_type, IcmpType::EchoRequest);
        assert_eq!(header.code, 0);
        assert_eq!(payload, &[0xDE, 0xAD]);
    }

    #[test]
    fn test_icmp_header_serialize() {
        let header = IcmpHeader {
            icmp_type: IcmpType::EchoReply,
            code: 0,
            checksum: 0x1234,
            rest_of_header: [0x00, 0x01, 0x00, 0x02],
        };

        let mut buf = [0u8; 16];
        let len = header.serialize(&mut buf).unwrap();

        assert_eq!(len, 8);
        assert_eq!(buf[0], 0); // Echo Reply type
        assert_eq!(buf[1], 0); // Code
        assert_eq!(&buf[2..4], &[0x12, 0x34]); // Checksum
    }

    #[test]
    fn test_icmp_echo_request() {
        let data = b"ping";
        let (header, echo) = IcmpEcho::request(1, 2, data);

        assert_eq!(header.icmp_type, IcmpType::EchoRequest);
        assert_eq!(echo.identifier, 1);
        assert_eq!(echo.sequence, 2);
        assert_eq!(echo.data, data);
    }

    #[test]
    fn test_icmp_echo_serialize_parse() {
        let data = b"hello";
        let echo = IcmpEcho {
            identifier: 0x1234,
            sequence: 0x0001,
            data,
        };

        let mut buf = [0u8; 64];
        let len = echo.serialize(true, &mut buf).unwrap();

        // Parse it back
        let (header, payload) = IcmpHeader::parse(&buf[..len]).unwrap();
        assert_eq!(header.icmp_type, IcmpType::EchoRequest);

        let parsed = IcmpEcho::parse(&header, payload).unwrap();
        assert_eq!(parsed.identifier, 0x1234);
        assert_eq!(parsed.sequence, 0x0001);
        assert_eq!(parsed.data, data);

        // Verify checksum
        assert!(IcmpHeader::verify_checksum(&buf[..len]));
    }

    #[test]
    fn test_icmp_echo_to_reply() {
        let (req_header, req_echo) = IcmpEcho::request(1, 2, b"test");
        let (rep_header, rep_echo) = req_echo.to_reply();

        assert_eq!(req_header.icmp_type, IcmpType::EchoRequest);
        assert_eq!(rep_header.icmp_type, IcmpType::EchoReply);
        assert_eq!(req_echo.identifier, rep_echo.identifier);
        assert_eq!(req_echo.sequence, rep_echo.sequence);
    }

    #[test]
    fn test_dest_unreachable_codes() {
        assert_eq!(
            DestUnreachableCode::from_u8(0),
            DestUnreachableCode::NetworkUnreachable
        );
        assert_eq!(
            DestUnreachableCode::from_u8(3),
            DestUnreachableCode::PortUnreachable
        );
        assert_eq!(
            DestUnreachableCode::from_u8(4),
            DestUnreachableCode::FragmentationNeeded
        );
    }

    #[test]
    fn test_dest_unreachable_message() {
        let original = [0u8; 28]; // Original IP header + 8 bytes
        let (header, msg) =
            IcmpDestUnreachable::new(DestUnreachableCode::PortUnreachable, 0, &original);

        assert_eq!(header.icmp_type, IcmpType::DestinationUnreachable);
        assert_eq!(header.code, 3);
        assert_eq!(msg.code, DestUnreachableCode::PortUnreachable);
    }

    #[test]
    fn test_time_exceeded_codes() {
        assert_eq!(
            TimeExceededCode::from_u8(0),
            TimeExceededCode::TtlExceeded
        );
        assert_eq!(
            TimeExceededCode::from_u8(1),
            TimeExceededCode::FragmentReassemblyExceeded
        );
    }

    #[test]
    fn test_time_exceeded_message() {
        let original = [0u8; 28];
        let (header, msg) = IcmpTimeExceeded::new(TimeExceededCode::TtlExceeded, &original);

        assert_eq!(header.icmp_type, IcmpType::TimeExceeded);
        assert_eq!(header.code, 0);
        assert_eq!(msg.code, TimeExceededCode::TtlExceeded);
    }

    #[test]
    fn test_icmp_checksum() {
        // Create a valid ICMP echo request
        let echo = IcmpEcho {
            identifier: 1,
            sequence: 1,
            data: &[0xDE, 0xAD, 0xBE, 0xEF],
        };

        let mut buf = [0u8; 64];
        let len = echo.serialize(true, &mut buf).unwrap();

        // Checksum should be valid
        assert!(IcmpHeader::verify_checksum(&buf[..len]));

        // Corrupt the message
        buf[len - 1] ^= 0xFF;

        // Checksum should now be invalid
        assert!(!IcmpHeader::verify_checksum(&buf[..len]));
    }

    #[test]
    fn test_icmp_header_too_short() {
        let short = [0u8; 4];
        assert_eq!(IcmpHeader::parse(&short), Err(NetError::PacketTooShort));
    }
}
