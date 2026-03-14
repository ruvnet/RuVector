//! User Datagram Protocol (UDP) handling.
//!
//! This module provides types and functions for parsing and serializing
//! UDP datagrams as per RFC 768.
//!
//! ## Header Format
//!
//! ```text
//! 0      7 8     15 16    23 24    31
//! +--------+--------+--------+--------+
//! |     Source      |   Destination   |
//! |      Port       |      Port       |
//! +--------+--------+--------+--------+
//! |     Length      |    Checksum     |
//! +--------+--------+--------+--------+
//! |          data octets ...          |
//! +-----------------------------------+
//! ```

use crate::error::{NetError, NetResult};
use crate::ipv4::Ipv4Addr;
use crate::UDP_HEADER_SIZE;

/// UDP header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdpHeader {
    /// Source port.
    pub src_port: u16,
    /// Destination port.
    pub dst_port: u16,
    /// Length of UDP header and data.
    pub length: u16,
    /// Checksum (0 means no checksum).
    pub checksum: u16,
}

impl UdpHeader {
    /// Parses a UDP header from a byte buffer.
    ///
    /// # Errors
    ///
    /// Returns `NetError::PacketTooShort` if buffer is smaller than 8 bytes.
    #[inline]
    pub fn parse(bytes: &[u8]) -> NetResult<(Self, &[u8])> {
        if bytes.len() < UDP_HEADER_SIZE {
            return Err(NetError::PacketTooShort);
        }

        let src_port = u16::from_be_bytes([bytes[0], bytes[1]]);
        let dst_port = u16::from_be_bytes([bytes[2], bytes[3]]);
        let length = u16::from_be_bytes([bytes[4], bytes[5]]);
        let checksum = u16::from_be_bytes([bytes[6], bytes[7]]);

        let header = Self {
            src_port,
            dst_port,
            length,
            checksum,
        };

        // Calculate payload bounds
        let data_len = (length as usize).saturating_sub(UDP_HEADER_SIZE);
        let payload_end = UDP_HEADER_SIZE + data_len.min(bytes.len() - UDP_HEADER_SIZE);
        let payload = &bytes[UDP_HEADER_SIZE..payload_end];

        Ok((header, payload))
    }

    /// Serializes the UDP header into a buffer.
    ///
    /// Returns the number of bytes written (always 8 on success).
    ///
    /// # Errors
    ///
    /// Returns `NetError::BufferTooSmall` if buffer is smaller than 8 bytes.
    #[inline]
    pub fn serialize(&self, buf: &mut [u8]) -> NetResult<usize> {
        if buf.len() < UDP_HEADER_SIZE {
            return Err(NetError::BufferTooSmall);
        }

        buf[0..2].copy_from_slice(&self.src_port.to_be_bytes());
        buf[2..4].copy_from_slice(&self.dst_port.to_be_bytes());
        buf[4..6].copy_from_slice(&self.length.to_be_bytes());
        buf[6..8].copy_from_slice(&self.checksum.to_be_bytes());

        Ok(UDP_HEADER_SIZE)
    }

    /// Creates a new UDP header.
    #[inline]
    #[must_use]
    pub const fn new(src_port: u16, dst_port: u16, payload_len: u16) -> Self {
        Self {
            src_port,
            dst_port,
            length: UDP_HEADER_SIZE as u16 + payload_len,
            checksum: 0, // Will be computed if needed
        }
    }

    /// Returns the payload length (total length minus header).
    #[inline]
    #[must_use]
    pub const fn payload_len(&self) -> u16 {
        self.length.saturating_sub(UDP_HEADER_SIZE as u16)
    }

    /// Computes the UDP checksum including the pseudo-header.
    ///
    /// The checksum covers a pseudo-header (source/dest IP, protocol, UDP length)
    /// plus the UDP header and payload.
    #[must_use]
    pub fn compute_checksum(
        src_ip: Ipv4Addr,
        dst_ip: Ipv4Addr,
        header: &UdpHeader,
        payload: &[u8],
    ) -> u16 {
        let mut sum: u32 = 0;

        // Pseudo-header
        // Source IP
        sum += u32::from(u16::from_be_bytes([src_ip.0[0], src_ip.0[1]]));
        sum += u32::from(u16::from_be_bytes([src_ip.0[2], src_ip.0[3]]));
        // Destination IP
        sum += u32::from(u16::from_be_bytes([dst_ip.0[0], dst_ip.0[1]]));
        sum += u32::from(u16::from_be_bytes([dst_ip.0[2], dst_ip.0[3]]));
        // Protocol (UDP = 17)
        sum += 17u32;
        // UDP length
        sum += u32::from(header.length);

        // UDP header (with checksum field as 0)
        sum += u32::from(header.src_port);
        sum += u32::from(header.dst_port);
        sum += u32::from(header.length);
        // Checksum field is 0 during calculation

        // Payload
        for chunk in payload.chunks(2) {
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
        let result = !sum as u16;
        // UDP checksum of 0 means "no checksum", use 0xFFFF instead
        if result == 0 {
            0xFFFF
        } else {
            result
        }
    }

    /// Verifies the UDP checksum.
    #[must_use]
    pub fn verify_checksum(
        src_ip: Ipv4Addr,
        dst_ip: Ipv4Addr,
        header: &UdpHeader,
        payload: &[u8],
    ) -> bool {
        // Checksum of 0 means checksum was not computed
        if header.checksum == 0 {
            return true;
        }

        // Recompute checksum including the stored checksum field
        let mut sum: u32 = 0;

        // Pseudo-header
        sum += u32::from(u16::from_be_bytes([src_ip.0[0], src_ip.0[1]]));
        sum += u32::from(u16::from_be_bytes([src_ip.0[2], src_ip.0[3]]));
        sum += u32::from(u16::from_be_bytes([dst_ip.0[0], dst_ip.0[1]]));
        sum += u32::from(u16::from_be_bytes([dst_ip.0[2], dst_ip.0[3]]));
        sum += 17u32;
        sum += u32::from(header.length);

        // UDP header including checksum
        sum += u32::from(header.src_port);
        sum += u32::from(header.dst_port);
        sum += u32::from(header.length);
        sum += u32::from(header.checksum);

        // Payload
        for chunk in payload.chunks(2) {
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

        // Should be 0xFFFF if valid
        sum as u16 == 0xFFFF
    }
}

/// UDP socket state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UdpSocketState {
    /// Socket is unbound.
    Unbound,
    /// Socket is bound to a local port.
    Bound,
    /// Socket is connected to a remote address.
    Connected,
}

/// UDP socket for sending and receiving datagrams.
///
/// This is a minimal socket implementation for `no_std` environments.
#[derive(Debug)]
pub struct UdpSocket {
    /// Local IP address.
    local_ip: Ipv4Addr,
    /// Local port (0 if unbound).
    local_port: u16,
    /// Remote IP address (for connected sockets).
    remote_ip: Ipv4Addr,
    /// Remote port (for connected sockets).
    remote_port: u16,
    /// Socket state.
    state: UdpSocketState,
}

impl UdpSocket {
    /// Creates a new unbound UDP socket.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            local_ip: Ipv4Addr::UNSPECIFIED,
            local_port: 0,
            remote_ip: Ipv4Addr::UNSPECIFIED,
            remote_port: 0,
            state: UdpSocketState::Unbound,
        }
    }

    /// Binds the socket to a local address and port.
    ///
    /// # Errors
    ///
    /// Returns `NetError::InvalidPort` if port is 0.
    pub fn bind(&mut self, local_ip: Ipv4Addr, local_port: u16) -> NetResult<()> {
        if local_port == 0 {
            return Err(NetError::InvalidPort);
        }

        self.local_ip = local_ip;
        self.local_port = local_port;
        self.state = UdpSocketState::Bound;

        Ok(())
    }

    /// Connects the socket to a remote address.
    ///
    /// This doesn't actually establish a connection (UDP is connectionless),
    /// but sets the default destination for `send()`.
    pub fn connect(&mut self, remote_ip: Ipv4Addr, remote_port: u16) -> NetResult<()> {
        if remote_port == 0 {
            return Err(NetError::InvalidPort);
        }

        self.remote_ip = remote_ip;
        self.remote_port = remote_port;
        self.state = UdpSocketState::Connected;

        Ok(())
    }

    /// Builds a UDP datagram for sending.
    ///
    /// Returns the total size of the datagram (header + payload).
    ///
    /// # Errors
    ///
    /// - `SocketNotBound` if socket is not bound
    /// - `InvalidPort` if destination port is 0
    /// - `BufferTooSmall` if buffer is too small
    pub fn build_datagram(
        &self,
        dst_ip: Ipv4Addr,
        dst_port: u16,
        payload: &[u8],
        buf: &mut [u8],
    ) -> NetResult<usize> {
        if self.state == UdpSocketState::Unbound {
            return Err(NetError::SocketNotBound);
        }

        if dst_port == 0 {
            return Err(NetError::InvalidPort);
        }

        let total_len = UDP_HEADER_SIZE + payload.len();
        if buf.len() < total_len {
            return Err(NetError::BufferTooSmall);
        }

        let header = UdpHeader::new(self.local_port, dst_port, payload.len() as u16);

        // Serialize header
        header.serialize(buf)?;

        // Copy payload
        buf[UDP_HEADER_SIZE..total_len].copy_from_slice(payload);

        // Compute and fill in checksum
        let checksum = UdpHeader::compute_checksum(self.local_ip, dst_ip, &header, payload);
        buf[6..8].copy_from_slice(&checksum.to_be_bytes());

        Ok(total_len)
    }

    /// Builds a UDP datagram to the connected remote address.
    ///
    /// # Errors
    ///
    /// - `SocketNotBound` if socket is not connected
    /// - `BufferTooSmall` if buffer is too small
    pub fn build_send(&self, payload: &[u8], buf: &mut [u8]) -> NetResult<usize> {
        if self.state != UdpSocketState::Connected {
            return Err(NetError::SocketNotBound);
        }

        self.build_datagram(self.remote_ip, self.remote_port, payload, buf)
    }

    /// Parses a received UDP datagram.
    ///
    /// Returns the source address, port, and payload.
    ///
    /// # Errors
    ///
    /// - `SocketNotBound` if socket is not bound
    /// - `PacketTooShort` if packet is too short
    pub fn parse_recv<'a>(
        &self,
        src_ip: Ipv4Addr,
        data: &'a [u8],
    ) -> NetResult<(Ipv4Addr, u16, &'a [u8])> {
        if self.state == UdpSocketState::Unbound {
            return Err(NetError::SocketNotBound);
        }

        let (header, payload) = UdpHeader::parse(data)?;

        // Verify destination port matches our bound port
        if header.dst_port != self.local_port {
            return Err(NetError::DestinationUnreachable);
        }

        // Verify checksum if present
        if header.checksum != 0
            && !UdpHeader::verify_checksum(src_ip, self.local_ip, &header, payload)
        {
            return Err(NetError::UdpChecksumError);
        }

        Ok((src_ip, header.src_port, payload))
    }

    /// Returns the local address and port.
    #[inline]
    #[must_use]
    pub const fn local_addr(&self) -> (Ipv4Addr, u16) {
        (self.local_ip, self.local_port)
    }

    /// Returns the remote address and port (for connected sockets).
    #[inline]
    #[must_use]
    pub const fn remote_addr(&self) -> Option<(Ipv4Addr, u16)> {
        if matches!(self.state, UdpSocketState::Connected) {
            Some((self.remote_ip, self.remote_port))
        } else {
            None
        }
    }

    /// Returns the socket state.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> UdpSocketState {
        self.state
    }

    /// Returns true if the socket is bound.
    #[inline]
    #[must_use]
    pub const fn is_bound(&self) -> bool {
        !matches!(self.state, UdpSocketState::Unbound)
    }

    /// Returns true if the socket is connected.
    #[inline]
    #[must_use]
    pub const fn is_connected(&self) -> bool {
        matches!(self.state, UdpSocketState::Connected)
    }
}

impl Default for UdpSocket {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing UDP datagrams.
pub struct UdpDatagramBuilder {
    src_port: u16,
    dst_port: u16,
    src_ip: Ipv4Addr,
    dst_ip: Ipv4Addr,
    compute_checksum: bool,
}

impl UdpDatagramBuilder {
    /// Creates a new datagram builder.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            src_port: 0,
            dst_port: 0,
            src_ip: Ipv4Addr::UNSPECIFIED,
            dst_ip: Ipv4Addr::UNSPECIFIED,
            compute_checksum: true,
        }
    }

    /// Sets the source port.
    #[inline]
    #[must_use]
    pub const fn src_port(mut self, port: u16) -> Self {
        self.src_port = port;
        self
    }

    /// Sets the destination port.
    #[inline]
    #[must_use]
    pub const fn dst_port(mut self, port: u16) -> Self {
        self.dst_port = port;
        self
    }

    /// Sets the source IP (for checksum calculation).
    #[inline]
    #[must_use]
    pub const fn src_ip(mut self, ip: Ipv4Addr) -> Self {
        self.src_ip = ip;
        self
    }

    /// Sets the destination IP (for checksum calculation).
    #[inline]
    #[must_use]
    pub const fn dst_ip(mut self, ip: Ipv4Addr) -> Self {
        self.dst_ip = ip;
        self
    }

    /// Disables checksum computation.
    #[inline]
    #[must_use]
    pub const fn no_checksum(mut self) -> Self {
        self.compute_checksum = false;
        self
    }

    /// Builds the datagram with the given payload.
    ///
    /// Returns the total size written to the buffer.
    pub fn build(&self, payload: &[u8], buf: &mut [u8]) -> NetResult<usize> {
        let total_len = UDP_HEADER_SIZE + payload.len();
        if buf.len() < total_len {
            return Err(NetError::BufferTooSmall);
        }

        let mut header = UdpHeader::new(self.src_port, self.dst_port, payload.len() as u16);

        if self.compute_checksum {
            header.checksum =
                UdpHeader::compute_checksum(self.src_ip, self.dst_ip, &header, payload);
        }

        header.serialize(buf)?;
        buf[UDP_HEADER_SIZE..total_len].copy_from_slice(payload);

        Ok(total_len)
    }
}

impl Default for UdpDatagramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_udp_header_new() {
        let header = UdpHeader::new(12345, 80, 100);
        assert_eq!(header.src_port, 12345);
        assert_eq!(header.dst_port, 80);
        assert_eq!(header.length, 108); // 8 + 100
        assert_eq!(header.payload_len(), 100);
    }

    #[test]
    fn test_udp_header_parse() {
        #[rustfmt::skip]
        let data = [
            0x30, 0x39, // Source port: 12345
            0x00, 0x50, // Dest port: 80
            0x00, 0x0C, // Length: 12 (8 header + 4 payload)
            0x00, 0x00, // Checksum: 0
            0xDE, 0xAD, 0xBE, 0xEF, // Payload
        ];

        let (header, payload) = UdpHeader::parse(&data).unwrap();
        assert_eq!(header.src_port, 12345);
        assert_eq!(header.dst_port, 80);
        assert_eq!(header.length, 12);
        assert_eq!(payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_udp_header_serialize() {
        let header = UdpHeader::new(12345, 80, 4);

        let mut buf = [0u8; 16];
        let len = header.serialize(&mut buf).unwrap();

        assert_eq!(len, 8);
        assert_eq!(&buf[0..2], &[0x30, 0x39]); // Source port
        assert_eq!(&buf[2..4], &[0x00, 0x50]); // Dest port
        assert_eq!(&buf[4..6], &[0x00, 0x0C]); // Length
    }

    #[test]
    fn test_udp_header_roundtrip() {
        let original = UdpHeader::new(54321, 443, 256);

        let mut buf = [0u8; 32];
        original.serialize(&mut buf).unwrap();

        let (parsed, _) = UdpHeader::parse(&buf).unwrap();
        assert_eq!(original.src_port, parsed.src_port);
        assert_eq!(original.dst_port, parsed.dst_port);
        assert_eq!(original.length, parsed.length);
    }

    #[test]
    fn test_udp_checksum() {
        let header = UdpHeader::new(12345, 80, 4);
        let payload = [0xDE, 0xAD, 0xBE, 0xEF];
        let src_ip = Ipv4Addr::new(192, 168, 1, 1);
        let dst_ip = Ipv4Addr::new(192, 168, 1, 2);

        let checksum = UdpHeader::compute_checksum(src_ip, dst_ip, &header, &payload);
        assert_ne!(checksum, 0);

        // Verify checksum
        let mut header_with_checksum = header;
        header_with_checksum.checksum = checksum;
        assert!(UdpHeader::verify_checksum(
            src_ip,
            dst_ip,
            &header_with_checksum,
            &payload
        ));
    }

    #[test]
    fn test_udp_checksum_zero_means_no_checksum() {
        let header = UdpHeader {
            src_port: 12345,
            dst_port: 80,
            length: 12,
            checksum: 0,
        };
        let payload = [0u8; 4];
        let src_ip = Ipv4Addr::LOCALHOST;
        let dst_ip = Ipv4Addr::LOCALHOST;

        // Zero checksum should always verify as valid (means no checksum)
        assert!(UdpHeader::verify_checksum(src_ip, dst_ip, &header, &payload));
    }

    #[test]
    fn test_udp_socket_new() {
        let socket = UdpSocket::new();
        assert_eq!(socket.state(), UdpSocketState::Unbound);
        assert!(!socket.is_bound());
        assert!(!socket.is_connected());
    }

    #[test]
    fn test_udp_socket_bind() {
        let mut socket = UdpSocket::new();
        socket.bind(Ipv4Addr::new(192, 168, 1, 1), 8080).unwrap();

        assert!(socket.is_bound());
        assert_eq!(socket.state(), UdpSocketState::Bound);
        assert_eq!(socket.local_addr(), (Ipv4Addr::new(192, 168, 1, 1), 8080));
    }

    #[test]
    fn test_udp_socket_bind_invalid_port() {
        let mut socket = UdpSocket::new();
        let result = socket.bind(Ipv4Addr::LOCALHOST, 0);
        assert_eq!(result, Err(NetError::InvalidPort));
    }

    #[test]
    fn test_udp_socket_connect() {
        let mut socket = UdpSocket::new();
        socket.bind(Ipv4Addr::LOCALHOST, 8080).unwrap();
        socket.connect(Ipv4Addr::new(10, 0, 0, 1), 443).unwrap();

        assert!(socket.is_connected());
        assert_eq!(
            socket.remote_addr(),
            Some((Ipv4Addr::new(10, 0, 0, 1), 443))
        );
    }

    #[test]
    fn test_udp_socket_build_datagram() {
        let mut socket = UdpSocket::new();
        socket.bind(Ipv4Addr::new(192, 168, 1, 1), 12345).unwrap();

        let payload = b"Hello, UDP!";
        let mut buf = [0u8; 64];

        let len = socket
            .build_datagram(Ipv4Addr::new(192, 168, 1, 2), 80, payload, &mut buf)
            .unwrap();

        assert_eq!(len, UDP_HEADER_SIZE + payload.len());

        // Parse it back
        let (header, data) = UdpHeader::parse(&buf[..len]).unwrap();
        assert_eq!(header.src_port, 12345);
        assert_eq!(header.dst_port, 80);
        assert_eq!(data, payload);
    }

    #[test]
    fn test_udp_socket_unbound_error() {
        let socket = UdpSocket::new();
        let mut buf = [0u8; 64];

        let result = socket.build_datagram(Ipv4Addr::LOCALHOST, 80, &[], &mut buf);
        assert_eq!(result, Err(NetError::SocketNotBound));
    }

    #[test]
    fn test_udp_datagram_builder() {
        let payload = b"Test";
        let mut buf = [0u8; 64];

        let len = UdpDatagramBuilder::new()
            .src_port(12345)
            .dst_port(80)
            .src_ip(Ipv4Addr::LOCALHOST)
            .dst_ip(Ipv4Addr::LOCALHOST)
            .build(payload, &mut buf)
            .unwrap();

        assert_eq!(len, 12);

        let (header, data) = UdpHeader::parse(&buf[..len]).unwrap();
        assert_eq!(header.src_port, 12345);
        assert_eq!(header.dst_port, 80);
        assert_ne!(header.checksum, 0); // Checksum should be computed
        assert_eq!(data, payload);
    }

    #[test]
    fn test_udp_datagram_builder_no_checksum() {
        let payload = b"Test";
        let mut buf = [0u8; 64];

        let len = UdpDatagramBuilder::new()
            .src_port(12345)
            .dst_port(80)
            .no_checksum()
            .build(payload, &mut buf)
            .unwrap();

        let (header, _) = UdpHeader::parse(&buf[..len]).unwrap();
        assert_eq!(header.checksum, 0);
    }

    #[test]
    fn test_udp_header_too_short() {
        let short = [0u8; 4];
        assert_eq!(UdpHeader::parse(&short), Err(NetError::PacketTooShort));
    }
}
