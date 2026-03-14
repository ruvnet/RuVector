//! Network stack integration.
//!
//! This module provides the `NetworkStack` type that combines all network
//! layers (Ethernet, ARP, IPv4, UDP, ICMP) into a cohesive stack.

use crate::arp::{ArpCache, ArpPacket};
use crate::device::NetworkDevice;
use crate::error::{NetError, NetResult};
use crate::ethernet::{EtherType, EthernetFrame, MacAddress};
use crate::icmp::{IcmpEcho, IcmpHeader, IcmpType};
use crate::ipv4::{Ipv4Addr, Ipv4Header, Protocol};
use crate::udp::UdpHeader;
use crate::{ETHERNET_HEADER_SIZE, IPV4_HEADER_MIN_SIZE, MTU, UDP_HEADER_SIZE};

/// Network stack configuration.
#[derive(Debug, Clone, Copy)]
pub struct StackConfig {
    /// Local IP address.
    pub ip_addr: Ipv4Addr,
    /// Subnet mask.
    pub subnet_mask: Ipv4Addr,
    /// Default gateway.
    pub gateway: Ipv4Addr,
    /// ARP cache timeout (in abstract time units).
    pub arp_timeout: u64,
    /// Default TTL for outgoing packets.
    pub default_ttl: u8,
    /// Enable ICMP echo reply (ping).
    pub enable_ping: bool,
}

impl Default for StackConfig {
    fn default() -> Self {
        Self {
            ip_addr: Ipv4Addr::UNSPECIFIED,
            subnet_mask: Ipv4Addr::new(255, 255, 255, 0),
            gateway: Ipv4Addr::UNSPECIFIED,
            arp_timeout: 300,
            default_ttl: 64,
            enable_ping: true,
        }
    }
}

/// Network stack state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackState {
    /// Stack is not initialized.
    Uninitialized,
    /// Stack is initializing.
    Initializing,
    /// Stack is ready to send/receive.
    Ready,
    /// Stack has encountered an error.
    Error,
}

/// Received packet information.
#[derive(Debug, Clone, Copy)]
pub struct ReceivedPacket {
    /// Source MAC address.
    pub src_mac: MacAddress,
    /// Source IP address.
    pub src_ip: Ipv4Addr,
    /// Source port (for UDP).
    pub src_port: u16,
    /// Destination port (for UDP).
    pub dst_port: u16,
    /// Protocol.
    pub protocol: Protocol,
    /// Payload offset in buffer.
    pub payload_offset: usize,
    /// Payload length.
    pub payload_len: usize,
}

/// Minimal network stack for `no_std` environments.
///
/// This stack supports:
/// - Ethernet II frames
/// - ARP resolution
/// - IPv4 routing
/// - UDP datagrams
/// - ICMP echo (ping)
pub struct NetworkStack<D: NetworkDevice> {
    /// Underlying network device.
    device: D,
    /// Stack configuration.
    config: StackConfig,
    /// ARP cache.
    arp_cache: ArpCache,
    /// Stack state.
    state: StackState,
    /// Packet identification counter.
    ip_id: u16,
    /// Current time (abstract units, must be updated externally).
    current_time: u64,
}

impl<D: NetworkDevice> NetworkStack<D> {
    /// Creates a new network stack.
    #[inline]
    #[must_use]
    pub fn new(device: D, config: StackConfig) -> Self {
        Self {
            device,
            config,
            arp_cache: ArpCache::with_timeout(config.arp_timeout),
            state: StackState::Ready,
            ip_id: 0,
            current_time: 0,
        }
    }

    /// Returns the MAC address of the device.
    #[inline]
    #[must_use]
    pub fn mac_address(&self) -> MacAddress {
        self.device.mac_address()
    }

    /// Returns the IP address.
    #[inline]
    #[must_use]
    pub const fn ip_address(&self) -> Ipv4Addr {
        self.config.ip_addr
    }

    /// Returns the stack state.
    #[inline]
    #[must_use]
    pub const fn state(&self) -> StackState {
        self.state
    }

    /// Returns a reference to the ARP cache.
    #[inline]
    #[must_use]
    pub const fn arp_cache(&self) -> &ArpCache {
        &self.arp_cache
    }

    /// Updates the current time (must be called periodically).
    #[inline]
    pub fn tick(&mut self, current_time: u64) {
        self.current_time = current_time;
        self.arp_cache.expire_stale(current_time);
    }

    /// Gets the next IP identification number.
    #[inline]
    fn next_ip_id(&mut self) -> u16 {
        let id = self.ip_id;
        self.ip_id = self.ip_id.wrapping_add(1);
        id
    }

    /// Determines if a destination is on the local network.
    #[inline]
    #[must_use]
    pub fn is_local(&self, dest: Ipv4Addr) -> bool {
        self.config
            .ip_addr
            .is_same_subnet(&dest, &self.config.subnet_mask)
    }

    /// Gets the next-hop MAC address for a destination.
    ///
    /// If the destination is local, resolves via ARP.
    /// If remote, returns the gateway's MAC.
    pub fn resolve_next_hop(&mut self, dest: Ipv4Addr) -> NetResult<Option<MacAddress>> {
        let next_hop = if self.is_local(dest) {
            dest
        } else {
            self.config.gateway
        };

        // Check ARP cache
        if let Some(mac) = self.arp_cache.resolve(next_hop, self.current_time) {
            return Ok(Some(mac));
        }

        // Send ARP request
        self.send_arp_request(next_hop)?;

        Ok(None)
    }

    /// Sends an ARP request for the given IP.
    pub fn send_arp_request(&mut self, target_ip: Ipv4Addr) -> NetResult<()> {
        let arp = ArpPacket::request(self.device.mac_address(), self.config.ip_addr, target_ip);

        let mut buf = [0u8; 64];

        // Build Ethernet frame
        let frame = EthernetFrame::new(
            MacAddress::BROADCAST,
            self.device.mac_address(),
            EtherType::Arp,
            &[], // Payload will be added below
        );

        let eth_len = frame.serialize(&mut buf)?;

        // Serialize ARP packet
        let arp_len = arp.serialize(&mut buf[ETHERNET_HEADER_SIZE..])?;

        // Send frame
        self.device.send(&buf[..ETHERNET_HEADER_SIZE + arp_len])?;

        // Mark as pending in cache
        self.arp_cache.mark_pending(target_ip, self.current_time)?;

        // Fix unused variable warning
        let _ = eth_len;

        Ok(())
    }

    /// Sends a UDP datagram.
    ///
    /// Returns `NetError::ArpNotFound` if the destination MAC is not cached.
    /// In this case, an ARP request has been sent and the caller should retry.
    pub fn send_udp(
        &mut self,
        src_port: u16,
        dst_ip: Ipv4Addr,
        dst_port: u16,
        payload: &[u8],
    ) -> NetResult<()> {
        // Check payload size
        let max_payload = MTU - IPV4_HEADER_MIN_SIZE - UDP_HEADER_SIZE;
        if payload.len() > max_payload {
            return Err(NetError::PacketTooLarge);
        }

        // Resolve destination MAC
        let dst_mac = match self.resolve_next_hop(dst_ip)? {
            Some(mac) => mac,
            None => return Err(NetError::ArpNotFound),
        };

        let mut buf = [0u8; 1536];
        let mut offset = 0;

        // Leave space for Ethernet header
        offset += ETHERNET_HEADER_SIZE;

        // Build IPv4 header
        let ip_payload_len = UDP_HEADER_SIZE + payload.len();
        let mut ip_header =
            Ipv4Header::new(self.config.ip_addr, dst_ip, Protocol::Udp, ip_payload_len as u16);
        ip_header.identification = self.next_ip_id();
        ip_header.ttl = self.config.default_ttl;

        let ip_len = ip_header.serialize(&mut buf[offset..])?;
        offset += ip_len;

        // Build UDP header
        let udp_header = UdpHeader::new(src_port, dst_port, payload.len() as u16);
        let udp_len = udp_header.serialize(&mut buf[offset..])?;

        // Copy payload
        buf[offset + udp_len..offset + udp_len + payload.len()].copy_from_slice(payload);

        // Compute and fill UDP checksum
        let checksum = UdpHeader::compute_checksum(
            self.config.ip_addr,
            dst_ip,
            &udp_header,
            payload,
        );
        buf[offset + 6..offset + 8].copy_from_slice(&checksum.to_be_bytes());

        offset += udp_len + payload.len();

        // Build Ethernet header directly
        let src_mac = self.device.mac_address();
        buf[0..6].copy_from_slice(&dst_mac.0);
        buf[6..12].copy_from_slice(&src_mac.0);
        buf[12..14].copy_from_slice(&EtherType::Ipv4.to_u16().to_be_bytes());

        // Send
        self.device.send(&buf[..offset])
    }

    /// Sends an ICMP echo request (ping).
    pub fn send_ping(
        &mut self,
        dst_ip: Ipv4Addr,
        identifier: u16,
        sequence: u16,
        data: &[u8],
    ) -> NetResult<()> {
        // Resolve destination MAC
        let dst_mac = match self.resolve_next_hop(dst_ip)? {
            Some(mac) => mac,
            None => return Err(NetError::ArpNotFound),
        };

        let mut buf = [0u8; 1536];
        let mut offset = 0;

        // Leave space for Ethernet header
        offset += ETHERNET_HEADER_SIZE;

        // Build ICMP echo request
        let echo = IcmpEcho {
            identifier,
            sequence,
            data,
        };
        let icmp_start = offset + IPV4_HEADER_MIN_SIZE;
        let icmp_len = echo.serialize(true, &mut buf[icmp_start..])?;

        // Build IPv4 header
        let mut ip_header =
            Ipv4Header::new(self.config.ip_addr, dst_ip, Protocol::Icmp, icmp_len as u16);
        ip_header.identification = self.next_ip_id();
        ip_header.ttl = self.config.default_ttl;

        ip_header.serialize(&mut buf[offset..])?;
        offset = icmp_start + icmp_len;

        // Build Ethernet header directly
        let src_mac = self.device.mac_address();
        buf[0..6].copy_from_slice(&dst_mac.0);
        buf[6..12].copy_from_slice(&src_mac.0);
        buf[12..14].copy_from_slice(&EtherType::Ipv4.to_u16().to_be_bytes());

        // Send
        self.device.send(&buf[..offset])
    }

    /// Processes a received frame.
    ///
    /// Returns information about the received packet if it contains
    /// application-layer data (e.g., UDP payload).
    pub fn receive(&mut self, buf: &mut [u8]) -> NetResult<Option<ReceivedPacket>> {
        // Try to receive a frame
        let frame_len = match self.device.receive(buf)? {
            Some(len) => len,
            None => return Ok(None),
        };

        if frame_len < ETHERNET_HEADER_SIZE {
            return Ok(None);
        }

        // Parse Ethernet frame header manually to avoid borrow issues
        let dest_mac = MacAddress::parse(&buf[0..6])?;
        let src_mac = MacAddress::parse(&buf[6..12])?;
        let ether_type = EtherType::from_u16(u16::from_be_bytes([buf[12], buf[13]]));

        // Check if it's for us
        let our_mac = self.device.mac_address();
        if !dest_mac.is_broadcast()
            && dest_mac != our_mac
            && !dest_mac.is_multicast()
        {
            return Ok(None);
        }

        match ether_type {
            EtherType::Arp => {
                self.handle_arp(&buf[ETHERNET_HEADER_SIZE..frame_len])?;
                Ok(None)
            }
            EtherType::Ipv4 => self.handle_ipv4_owned(src_mac, buf, frame_len),
            _ => Ok(None),
        }
    }

    /// Handles an ARP packet.
    fn handle_arp(&mut self, payload: &[u8]) -> NetResult<()> {
        let arp = ArpPacket::parse(payload)?;

        // Update cache with sender's info
        self.arp_cache
            .insert(arp.sender_ip, arp.sender_mac, self.current_time)?;

        // If this is a request for our IP, send a reply
        if arp.is_request() && arp.target_ip == self.config.ip_addr {
            let reply =
                ArpPacket::reply(self.device.mac_address(), self.config.ip_addr, arp.sender_mac, arp.sender_ip);

            let mut buf = [0u8; 64];

            // Ethernet header (overwritten)
            let frame = EthernetFrame::new(
                arp.sender_mac,
                self.device.mac_address(),
                EtherType::Arp,
                &[],
            );
            frame.serialize(&mut buf)?;

            // ARP reply
            let arp_len = reply.serialize(&mut buf[ETHERNET_HEADER_SIZE..])?;

            self.device.send(&buf[..ETHERNET_HEADER_SIZE + arp_len])?;
        }

        Ok(())
    }

    /// Handles an IPv4 packet (owns buffer to avoid borrow issues).
    fn handle_ipv4_owned(
        &mut self,
        src_mac: MacAddress,
        buf: &mut [u8],
        frame_len: usize,
    ) -> NetResult<Option<ReceivedPacket>> {
        let (ip_header, ip_payload) = Ipv4Header::parse(&buf[ETHERNET_HEADER_SIZE..frame_len])?;

        // Check destination
        if ip_header.dst_addr != self.config.ip_addr
            && !ip_header.dst_addr.is_broadcast()
            && !ip_header.dst_addr.is_multicast()
        {
            return Ok(None);
        }

        // Check for fragmentation
        if ip_header.flags.more_fragments || ip_header.fragment_offset != 0 {
            return Err(NetError::FragmentationNotSupported);
        }

        match ip_header.protocol {
            Protocol::Icmp => {
                self.handle_icmp(src_mac, &ip_header, ip_payload)?;
                Ok(None)
            }
            Protocol::Udp => self.handle_udp(src_mac, &ip_header, ip_payload),
            _ => Ok(None),
        }
    }

    /// Handles an ICMP packet.
    fn handle_icmp(
        &mut self,
        _src_mac: MacAddress,
        ip_header: &Ipv4Header,
        payload: &[u8],
    ) -> NetResult<()> {
        let (icmp_header, icmp_payload) = IcmpHeader::parse(payload)?;

        // Verify checksum
        if !IcmpHeader::verify_checksum(payload) {
            return Ok(()); // Silently drop
        }

        // Handle echo request
        if self.config.enable_ping && icmp_header.icmp_type == IcmpType::EchoRequest {
            let echo = IcmpEcho::parse(&icmp_header, icmp_payload)?;

            // Send echo reply
            self.send_icmp_echo_reply(ip_header.src_addr, echo.identifier, echo.sequence, echo.data)?;
        }

        Ok(())
    }

    /// Sends an ICMP echo reply.
    fn send_icmp_echo_reply(
        &mut self,
        dst_ip: Ipv4Addr,
        identifier: u16,
        sequence: u16,
        data: &[u8],
    ) -> NetResult<()> {
        // Resolve destination MAC
        let dst_mac = match self.arp_cache.resolve(dst_ip, self.current_time) {
            Some(mac) => mac,
            None => return Ok(()), // Don't have MAC, silently drop
        };

        let mut buf = [0u8; 1536];
        let mut offset = 0;

        // Leave space for Ethernet header
        offset += ETHERNET_HEADER_SIZE;

        // Build ICMP echo reply
        let echo = IcmpEcho {
            identifier,
            sequence,
            data,
        };
        let icmp_start = offset + IPV4_HEADER_MIN_SIZE;
        let icmp_len = echo.serialize(false, &mut buf[icmp_start..])?; // false = reply

        // Build IPv4 header
        let mut ip_header =
            Ipv4Header::new(self.config.ip_addr, dst_ip, Protocol::Icmp, icmp_len as u16);
        ip_header.identification = self.next_ip_id();
        ip_header.ttl = self.config.default_ttl;

        ip_header.serialize(&mut buf[offset..])?;
        offset = icmp_start + icmp_len;

        // Build Ethernet header directly
        let src_mac = self.device.mac_address();
        buf[0..6].copy_from_slice(&dst_mac.0);
        buf[6..12].copy_from_slice(&src_mac.0);
        buf[12..14].copy_from_slice(&EtherType::Ipv4.to_u16().to_be_bytes());

        // Send
        self.device.send(&buf[..offset])
    }

    /// Handles a UDP packet.
    fn handle_udp(
        &mut self,
        src_mac: MacAddress,
        ip_header: &Ipv4Header,
        payload: &[u8],
    ) -> NetResult<Option<ReceivedPacket>> {
        let (udp_header, udp_payload) = UdpHeader::parse(payload)?;

        // Verify checksum if present
        if udp_header.checksum != 0 {
            if !UdpHeader::verify_checksum(
                ip_header.src_addr,
                ip_header.dst_addr,
                &udp_header,
                udp_payload,
            ) {
                return Err(NetError::UdpChecksumError);
            }
        }

        Ok(Some(ReceivedPacket {
            src_mac,
            src_ip: ip_header.src_addr,
            src_port: udp_header.src_port,
            dst_port: udp_header.dst_port,
            protocol: Protocol::Udp,
            payload_offset: ETHERNET_HEADER_SIZE
                + ip_header.header_len()
                + UDP_HEADER_SIZE,
            payload_len: udp_payload.len(),
        }))
    }

    /// Returns a reference to the underlying device.
    #[inline]
    #[must_use]
    pub const fn device(&self) -> &D {
        &self.device
    }

    /// Returns a mutable reference to the underlying device.
    #[inline]
    pub fn device_mut(&mut self) -> &mut D {
        &mut self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::LoopbackDevice;

    fn create_test_stack() -> NetworkStack<LoopbackDevice> {
        let device = LoopbackDevice::new();
        let config = StackConfig {
            ip_addr: Ipv4Addr::new(192, 168, 1, 1),
            subnet_mask: Ipv4Addr::new(255, 255, 255, 0),
            gateway: Ipv4Addr::new(192, 168, 1, 254),
            ..Default::default()
        };
        NetworkStack::new(device, config)
    }

    #[test]
    fn test_stack_creation() {
        let stack = create_test_stack();
        assert_eq!(stack.ip_address(), Ipv4Addr::new(192, 168, 1, 1));
        assert_eq!(stack.state(), StackState::Ready);
    }

    #[test]
    fn test_is_local() {
        let stack = create_test_stack();

        // Same subnet
        assert!(stack.is_local(Ipv4Addr::new(192, 168, 1, 100)));

        // Different subnet
        assert!(!stack.is_local(Ipv4Addr::new(10, 0, 0, 1)));
    }

    #[test]
    fn test_ip_id_counter() {
        let mut stack = create_test_stack();

        let id1 = stack.next_ip_id();
        let id2 = stack.next_ip_id();
        let id3 = stack.next_ip_id();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
    }

    #[test]
    fn test_tick() {
        let mut stack = create_test_stack();

        // Add an ARP entry
        stack
            .arp_cache
            .insert(
                Ipv4Addr::new(192, 168, 1, 2),
                MacAddress::new([0, 1, 2, 3, 4, 5]),
                0,
            )
            .unwrap();

        // Should resolve
        assert!(stack
            .arp_cache
            .resolve(Ipv4Addr::new(192, 168, 1, 2), 0)
            .is_some());

        // After timeout, should not resolve
        stack.tick(1000);
        assert!(stack
            .arp_cache
            .resolve(Ipv4Addr::new(192, 168, 1, 2), 1000)
            .is_none());
    }

    #[test]
    fn test_arp_request_send() {
        let mut stack = create_test_stack();

        // This should send an ARP request
        let result = stack.send_arp_request(Ipv4Addr::new(192, 168, 1, 2));
        assert!(result.is_ok());

        // Entry should be pending
        let entry = stack.arp_cache.lookup(Ipv4Addr::new(192, 168, 1, 2));
        assert!(entry.is_some());
    }

    #[test]
    fn test_receive_empty() {
        let mut stack = create_test_stack();
        let mut buf = [0u8; 1536];

        // Should return None when no packets
        let result = stack.receive(&mut buf).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_stack_config_default() {
        let config = StackConfig::default();
        assert_eq!(config.default_ttl, 64);
        assert!(config.enable_ping);
        assert_eq!(config.arp_timeout, 300);
    }
}
