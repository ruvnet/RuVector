//! Address Resolution Protocol (ARP) handling.
//!
//! This module provides types and functions for parsing and serializing
//! ARP packets as per RFC 826, as well as a simple ARP cache.
//!
//! ## Packet Format
//!
//! ```text
//! +----------------+----------------+
//! | Hardware Type  | Protocol Type  |
//! +----------------+----------------+
//! | HW Len | P Len |   Operation    |
//! +----------------+----------------+
//! |     Sender Hardware Address     |
//! +----------------------------------+
//! |     Sender Protocol Address     |
//! +----------------------------------+
//! |     Target Hardware Address     |
//! +----------------------------------+
//! |     Target Protocol Address     |
//! +----------------------------------+
//! ```

use crate::error::{NetError, NetResult};
use crate::ethernet::MacAddress;
use crate::ipv4::Ipv4Addr;
use crate::ARP_PACKET_SIZE;

/// ARP hardware type for Ethernet.
pub const HARDWARE_TYPE_ETHERNET: u16 = 1;

/// ARP protocol type for IPv4.
pub const PROTOCOL_TYPE_IPV4: u16 = 0x0800;

/// ARP operation codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum ArpOperation {
    /// ARP Request - who has IP X? Tell IP Y.
    Request = 1,
    /// ARP Reply - IP X is at MAC Z.
    Reply = 2,
    /// Unknown operation.
    Unknown(u16) = 0xFFFF,
}

impl ArpOperation {
    /// Converts from a raw u16 value.
    #[inline]
    #[must_use]
    pub const fn from_u16(value: u16) -> Self {
        match value {
            1 => Self::Request,
            2 => Self::Reply,
            other => Self::Unknown(other),
        }
    }

    /// Converts to a raw u16 value.
    #[inline]
    #[must_use]
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::Request => 1,
            Self::Reply => 2,
            Self::Unknown(v) => v,
        }
    }
}

/// ARP packet for Ethernet/IPv4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArpPacket {
    /// Hardware type (1 for Ethernet).
    pub hardware_type: u16,
    /// Protocol type (0x0800 for IPv4).
    pub protocol_type: u16,
    /// Hardware address length (6 for Ethernet).
    pub hardware_len: u8,
    /// Protocol address length (4 for IPv4).
    pub protocol_len: u8,
    /// ARP operation (request or reply).
    pub operation: ArpOperation,
    /// Sender hardware (MAC) address.
    pub sender_mac: MacAddress,
    /// Sender protocol (IP) address.
    pub sender_ip: Ipv4Addr,
    /// Target hardware (MAC) address.
    pub target_mac: MacAddress,
    /// Target protocol (IP) address.
    pub target_ip: Ipv4Addr,
}

impl ArpPacket {
    /// Creates a new ARP request packet.
    #[inline]
    #[must_use]
    pub const fn request(sender_mac: MacAddress, sender_ip: Ipv4Addr, target_ip: Ipv4Addr) -> Self {
        Self {
            hardware_type: HARDWARE_TYPE_ETHERNET,
            protocol_type: PROTOCOL_TYPE_IPV4,
            hardware_len: 6,
            protocol_len: 4,
            operation: ArpOperation::Request,
            sender_mac,
            sender_ip,
            target_mac: MacAddress::ZERO,
            target_ip,
        }
    }

    /// Creates a new ARP reply packet.
    #[inline]
    #[must_use]
    pub const fn reply(
        sender_mac: MacAddress,
        sender_ip: Ipv4Addr,
        target_mac: MacAddress,
        target_ip: Ipv4Addr,
    ) -> Self {
        Self {
            hardware_type: HARDWARE_TYPE_ETHERNET,
            protocol_type: PROTOCOL_TYPE_IPV4,
            hardware_len: 6,
            protocol_len: 4,
            operation: ArpOperation::Reply,
            sender_mac,
            sender_ip,
            target_mac,
            target_ip,
        }
    }

    /// Parses an ARP packet from a byte buffer.
    ///
    /// # Errors
    ///
    /// Returns `NetError::PacketTooShort` if buffer is smaller than 28 bytes.
    /// Returns `NetError::InvalidArpPacket` if hardware/protocol types are invalid.
    #[inline]
    pub fn parse(bytes: &[u8]) -> NetResult<Self> {
        if bytes.len() < ARP_PACKET_SIZE {
            return Err(NetError::PacketTooShort);
        }

        let hardware_type = u16::from_be_bytes([bytes[0], bytes[1]]);
        let protocol_type = u16::from_be_bytes([bytes[2], bytes[3]]);
        let hardware_len = bytes[4];
        let protocol_len = bytes[5];
        let operation = ArpOperation::from_u16(u16::from_be_bytes([bytes[6], bytes[7]]));

        // Validate for Ethernet/IPv4
        if hardware_type != HARDWARE_TYPE_ETHERNET
            || protocol_type != PROTOCOL_TYPE_IPV4
            || hardware_len != 6
            || protocol_len != 4
        {
            return Err(NetError::InvalidArpPacket);
        }

        let sender_mac = MacAddress::parse(&bytes[8..14])?;
        let sender_ip = Ipv4Addr::parse(&bytes[14..18])?;
        let target_mac = MacAddress::parse(&bytes[18..24])?;
        let target_ip = Ipv4Addr::parse(&bytes[24..28])?;

        Ok(Self {
            hardware_type,
            protocol_type,
            hardware_len,
            protocol_len,
            operation,
            sender_mac,
            sender_ip,
            target_mac,
            target_ip,
        })
    }

    /// Serializes the ARP packet into a buffer.
    ///
    /// Returns the number of bytes written (always 28 on success).
    ///
    /// # Errors
    ///
    /// Returns `NetError::BufferTooSmall` if buffer is smaller than 28 bytes.
    #[inline]
    pub fn serialize(&self, buf: &mut [u8]) -> NetResult<usize> {
        if buf.len() < ARP_PACKET_SIZE {
            return Err(NetError::BufferTooSmall);
        }

        buf[0..2].copy_from_slice(&self.hardware_type.to_be_bytes());
        buf[2..4].copy_from_slice(&self.protocol_type.to_be_bytes());
        buf[4] = self.hardware_len;
        buf[5] = self.protocol_len;
        buf[6..8].copy_from_slice(&self.operation.to_u16().to_be_bytes());
        buf[8..14].copy_from_slice(&self.sender_mac.0);
        buf[14..18].copy_from_slice(&self.sender_ip.0);
        buf[18..24].copy_from_slice(&self.target_mac.0);
        buf[24..28].copy_from_slice(&self.target_ip.0);

        Ok(ARP_PACKET_SIZE)
    }

    /// Checks if this is a request.
    #[inline]
    #[must_use]
    pub const fn is_request(&self) -> bool {
        matches!(self.operation, ArpOperation::Request)
    }

    /// Checks if this is a reply.
    #[inline]
    #[must_use]
    pub const fn is_reply(&self) -> bool {
        matches!(self.operation, ArpOperation::Reply)
    }
}

/// ARP cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArpCacheEntry {
    /// IP address.
    pub ip: Ipv4Addr,
    /// MAC address.
    pub mac: MacAddress,
    /// Timestamp when entry was created/updated (in abstract time units).
    pub timestamp: u64,
    /// Entry state.
    pub state: ArpEntryState,
}

/// State of an ARP cache entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArpEntryState {
    /// Entry is valid and can be used.
    Valid,
    /// Entry is pending resolution (request sent, waiting for reply).
    Pending,
    /// Entry has expired and needs refresh.
    Stale,
}

/// Default ARP cache timeout in time units (e.g., seconds or ticks).
pub const ARP_CACHE_TIMEOUT: u64 = 300; // 5 minutes

/// Maximum number of entries in the ARP cache.
pub const ARP_CACHE_MAX_ENTRIES: usize = 64;

/// Simple ARP cache with fixed-size storage.
///
/// This cache is designed for `no_std` environments without heap allocation.
/// It uses a fixed-size array with LRU-like eviction.
#[derive(Debug)]
pub struct ArpCache {
    entries: [Option<ArpCacheEntry>; ARP_CACHE_MAX_ENTRIES],
    count: usize,
    timeout: u64,
}

impl ArpCache {
    /// Creates a new empty ARP cache.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: [None; ARP_CACHE_MAX_ENTRIES],
            count: 0,
            timeout: ARP_CACHE_TIMEOUT,
        }
    }

    /// Creates a new ARP cache with a custom timeout.
    #[inline]
    #[must_use]
    pub const fn with_timeout(timeout: u64) -> Self {
        Self {
            entries: [None; ARP_CACHE_MAX_ENTRIES],
            count: 0,
            timeout,
        }
    }

    /// Resolves an IP address to a MAC address.
    ///
    /// Returns `Some(mac)` if the entry exists and is valid, `None` otherwise.
    #[inline]
    pub fn resolve(&self, ip: Ipv4Addr, current_time: u64) -> Option<MacAddress> {
        for entry in self.entries.iter().flatten() {
            if entry.ip == ip {
                match entry.state {
                    ArpEntryState::Valid => {
                        if current_time.saturating_sub(entry.timestamp) < self.timeout {
                            return Some(entry.mac);
                        }
                    }
                    ArpEntryState::Pending | ArpEntryState::Stale => {}
                }
            }
        }
        None
    }

    /// Looks up an entry by IP address.
    #[inline]
    pub fn lookup(&self, ip: Ipv4Addr) -> Option<&ArpCacheEntry> {
        for entry in self.entries.iter().flatten() {
            if entry.ip == ip {
                return Some(entry);
            }
        }
        None
    }

    /// Inserts or updates an ARP cache entry.
    ///
    /// If the cache is full, evicts the oldest entry.
    ///
    /// # Errors
    ///
    /// Returns `NetError::ArpCacheFull` if eviction fails (should not happen).
    pub fn insert(&mut self, ip: Ipv4Addr, mac: MacAddress, current_time: u64) -> NetResult<()> {
        // First, try to find an existing entry for this IP
        for entry in self.entries.iter_mut().flatten() {
            if entry.ip == ip {
                entry.mac = mac;
                entry.timestamp = current_time;
                entry.state = ArpEntryState::Valid;
                return Ok(());
            }
        }

        // Find an empty slot
        for slot in &mut self.entries {
            if slot.is_none() {
                *slot = Some(ArpCacheEntry {
                    ip,
                    mac,
                    timestamp: current_time,
                    state: ArpEntryState::Valid,
                });
                self.count += 1;
                return Ok(());
            }
        }

        // Cache is full, evict oldest entry
        let oldest_idx = self.find_oldest_entry();
        if let Some(idx) = oldest_idx {
            self.entries[idx] = Some(ArpCacheEntry {
                ip,
                mac,
                timestamp: current_time,
                state: ArpEntryState::Valid,
            });
            return Ok(());
        }

        Err(NetError::ArpCacheFull)
    }

    /// Marks an entry as pending (waiting for ARP reply).
    pub fn mark_pending(&mut self, ip: Ipv4Addr, current_time: u64) -> NetResult<()> {
        // Check if entry already exists
        for entry in self.entries.iter_mut().flatten() {
            if entry.ip == ip {
                entry.state = ArpEntryState::Pending;
                entry.timestamp = current_time;
                return Ok(());
            }
        }

        // Create new pending entry
        for slot in &mut self.entries {
            if slot.is_none() {
                *slot = Some(ArpCacheEntry {
                    ip,
                    mac: MacAddress::ZERO,
                    timestamp: current_time,
                    state: ArpEntryState::Pending,
                });
                self.count += 1;
                return Ok(());
            }
        }

        // Cache is full, evict oldest
        let oldest_idx = self.find_oldest_entry();
        if let Some(idx) = oldest_idx {
            self.entries[idx] = Some(ArpCacheEntry {
                ip,
                mac: MacAddress::ZERO,
                timestamp: current_time,
                state: ArpEntryState::Pending,
            });
            return Ok(());
        }

        Err(NetError::ArpCacheFull)
    }

    /// Processes an ARP reply and updates the cache.
    pub fn process_reply(&mut self, packet: &ArpPacket, current_time: u64) -> NetResult<()> {
        if !packet.is_reply() {
            return Ok(());
        }

        self.insert(packet.sender_ip, packet.sender_mac, current_time)
    }

    /// Removes an entry by IP address.
    pub fn remove(&mut self, ip: Ipv4Addr) -> bool {
        for slot in &mut self.entries {
            if let Some(entry) = slot {
                if entry.ip == ip {
                    *slot = None;
                    self.count = self.count.saturating_sub(1);
                    return true;
                }
            }
        }
        false
    }

    /// Clears all entries from the cache.
    pub fn clear(&mut self) {
        for slot in &mut self.entries {
            *slot = None;
        }
        self.count = 0;
    }

    /// Returns the number of entries in the cache.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the cache is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Expires stale entries based on current time.
    pub fn expire_stale(&mut self, current_time: u64) {
        for slot in &mut self.entries {
            if let Some(entry) = slot {
                if current_time.saturating_sub(entry.timestamp) >= self.timeout {
                    if entry.state == ArpEntryState::Valid {
                        entry.state = ArpEntryState::Stale;
                    } else {
                        // Remove pending or already stale entries that are too old
                        *slot = None;
                        self.count = self.count.saturating_sub(1);
                    }
                }
            }
        }
    }

    /// Finds the index of the oldest entry for eviction.
    fn find_oldest_entry(&self) -> Option<usize> {
        let mut oldest_idx = None;
        let mut oldest_time = u64::MAX;

        for (i, entry) in self.entries.iter().enumerate() {
            if let Some(e) = entry {
                if e.timestamp < oldest_time {
                    oldest_time = e.timestamp;
                    oldest_idx = Some(i);
                }
            }
        }

        oldest_idx
    }

    /// Returns an iterator over valid entries.
    pub fn iter(&self) -> impl Iterator<Item = &ArpCacheEntry> {
        self.entries.iter().filter_map(|e| e.as_ref())
    }
}

impl Default for ArpCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arp_operation_conversion() {
        assert_eq!(ArpOperation::from_u16(1), ArpOperation::Request);
        assert_eq!(ArpOperation::from_u16(2), ArpOperation::Reply);
        assert!(matches!(ArpOperation::from_u16(99), ArpOperation::Unknown(99)));

        assert_eq!(ArpOperation::Request.to_u16(), 1);
        assert_eq!(ArpOperation::Reply.to_u16(), 2);
    }

    #[test]
    fn test_arp_packet_request() {
        let sender_mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        let sender_ip = Ipv4Addr::new(192, 168, 1, 1);
        let target_ip = Ipv4Addr::new(192, 168, 1, 2);

        let packet = ArpPacket::request(sender_mac, sender_ip, target_ip);

        assert!(packet.is_request());
        assert!(!packet.is_reply());
        assert_eq!(packet.sender_mac, sender_mac);
        assert_eq!(packet.sender_ip, sender_ip);
        assert_eq!(packet.target_ip, target_ip);
        assert!(packet.target_mac.is_zero());
    }

    #[test]
    fn test_arp_packet_reply() {
        let sender_mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        let sender_ip = Ipv4Addr::new(192, 168, 1, 1);
        let target_mac = MacAddress::new([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
        let target_ip = Ipv4Addr::new(192, 168, 1, 2);

        let packet = ArpPacket::reply(sender_mac, sender_ip, target_mac, target_ip);

        assert!(!packet.is_request());
        assert!(packet.is_reply());
        assert_eq!(packet.target_mac, target_mac);
    }

    #[test]
    fn test_arp_packet_serialize_parse() {
        let original = ArpPacket::request(
            MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]),
            Ipv4Addr::new(192, 168, 1, 1),
            Ipv4Addr::new(192, 168, 1, 2),
        );

        let mut buf = [0u8; 64];
        let len = original.serialize(&mut buf).unwrap();
        assert_eq!(len, ARP_PACKET_SIZE);

        let parsed = ArpPacket::parse(&buf[..len]).unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn test_arp_packet_parse_too_short() {
        let short = [0u8; 10];
        assert_eq!(ArpPacket::parse(&short), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_arp_packet_parse_invalid_type() {
        let mut buf = [0u8; 28];
        // Invalid hardware type
        buf[0] = 0;
        buf[1] = 2; // Not Ethernet
        assert_eq!(ArpPacket::parse(&buf), Err(NetError::InvalidArpPacket));
    }

    #[test]
    fn test_arp_cache_insert_resolve() {
        let mut cache = ArpCache::new();
        let ip = Ipv4Addr::new(192, 168, 1, 1);
        let mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);

        cache.insert(ip, mac, 0).unwrap();
        assert_eq!(cache.len(), 1);

        let resolved = cache.resolve(ip, 0);
        assert_eq!(resolved, Some(mac));
    }

    #[test]
    fn test_arp_cache_timeout() {
        let mut cache = ArpCache::with_timeout(100);
        let ip = Ipv4Addr::new(192, 168, 1, 1);
        let mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);

        cache.insert(ip, mac, 0).unwrap();

        // Should resolve within timeout
        assert_eq!(cache.resolve(ip, 50), Some(mac));

        // Should not resolve after timeout
        assert_eq!(cache.resolve(ip, 150), None);
    }

    #[test]
    fn test_arp_cache_update() {
        let mut cache = ArpCache::new();
        let ip = Ipv4Addr::new(192, 168, 1, 1);
        let mac1 = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);
        let mac2 = MacAddress::new([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);

        cache.insert(ip, mac1, 0).unwrap();
        assert_eq!(cache.resolve(ip, 0), Some(mac1));

        cache.insert(ip, mac2, 0).unwrap();
        assert_eq!(cache.resolve(ip, 0), Some(mac2));
        assert_eq!(cache.len(), 1); // Should not create duplicate
    }

    #[test]
    fn test_arp_cache_remove() {
        let mut cache = ArpCache::new();
        let ip = Ipv4Addr::new(192, 168, 1, 1);
        let mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);

        cache.insert(ip, mac, 0).unwrap();
        assert_eq!(cache.len(), 1);

        assert!(cache.remove(ip));
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.resolve(ip, 0), None);

        // Remove non-existent
        assert!(!cache.remove(ip));
    }

    #[test]
    fn test_arp_cache_clear() {
        let mut cache = ArpCache::new();

        for i in 0..10 {
            cache
                .insert(
                    Ipv4Addr::new(192, 168, 1, i),
                    MacAddress::new([0, 0, 0, 0, 0, i]),
                    0,
                )
                .unwrap();
        }
        assert_eq!(cache.len(), 10);

        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_arp_cache_pending() {
        let mut cache = ArpCache::new();
        let ip = Ipv4Addr::new(192, 168, 1, 1);

        cache.mark_pending(ip, 0).unwrap();

        let entry = cache.lookup(ip).unwrap();
        assert_eq!(entry.state, ArpEntryState::Pending);

        // Pending entries should not resolve
        assert_eq!(cache.resolve(ip, 0), None);
    }

    #[test]
    fn test_arp_cache_process_reply() {
        let mut cache = ArpCache::new();
        let ip = Ipv4Addr::new(192, 168, 1, 1);
        let mac = MacAddress::new([0x00, 0x11, 0x22, 0x33, 0x44, 0x55]);

        cache.mark_pending(ip, 0).unwrap();

        let reply = ArpPacket::reply(mac, ip, MacAddress::ZERO, Ipv4Addr::UNSPECIFIED);
        cache.process_reply(&reply, 1).unwrap();

        let entry = cache.lookup(ip).unwrap();
        assert_eq!(entry.state, ArpEntryState::Valid);
        assert_eq!(cache.resolve(ip, 1), Some(mac));
    }

    #[test]
    fn test_arp_cache_expire_stale() {
        let mut cache = ArpCache::with_timeout(100);

        cache
            .insert(
                Ipv4Addr::new(192, 168, 1, 1),
                MacAddress::new([0, 0, 0, 0, 0, 1]),
                0,
            )
            .unwrap();
        cache
            .insert(
                Ipv4Addr::new(192, 168, 1, 2),
                MacAddress::new([0, 0, 0, 0, 0, 2]),
                60,
            )
            .unwrap();

        cache.expire_stale(150);

        // First entry should be stale (150 - 0 = 150 >= 100)
        let entry1 = cache.lookup(Ipv4Addr::new(192, 168, 1, 1)).unwrap();
        assert_eq!(entry1.state, ArpEntryState::Stale);

        // Second entry should still be valid (150 - 60 = 90 < 100)
        let entry2 = cache.lookup(Ipv4Addr::new(192, 168, 1, 2)).unwrap();
        assert_eq!(entry2.state, ArpEntryState::Valid);
    }

    #[test]
    fn test_arp_cache_eviction() {
        let mut cache = ArpCache::new();

        // Fill the cache
        for i in 0..ARP_CACHE_MAX_ENTRIES {
            cache
                .insert(
                    Ipv4Addr::from_u32(i as u32),
                    MacAddress::new([0, 0, 0, 0, (i >> 8) as u8, i as u8]),
                    i as u64,
                )
                .unwrap();
        }

        assert_eq!(cache.len(), ARP_CACHE_MAX_ENTRIES);

        // Insert one more - should evict oldest (timestamp 0)
        let new_ip = Ipv4Addr::new(10, 0, 0, 1);
        let new_mac = MacAddress::new([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        cache
            .insert(new_ip, new_mac, ARP_CACHE_MAX_ENTRIES as u64)
            .unwrap();

        // New entry should exist
        assert_eq!(
            cache.resolve(new_ip, ARP_CACHE_MAX_ENTRIES as u64),
            Some(new_mac)
        );

        // Oldest entry (IP 0.0.0.0) should be evicted
        assert!(cache.lookup(Ipv4Addr::from_u32(0)).is_none());
    }
}
