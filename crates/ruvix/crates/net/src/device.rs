//! Network device abstraction.
//!
//! This module provides the `NetworkDevice` trait for hardware abstraction,
//! allowing the network stack to work with different network interfaces.

use crate::error::NetResult;
use crate::ethernet::MacAddress;

/// Network device capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceCapabilities {
    /// Maximum transmission unit (bytes).
    pub mtu: u16,
    /// Device supports hardware checksum offload for TX.
    pub tx_checksum_offload: bool,
    /// Device supports hardware checksum offload for RX.
    pub rx_checksum_offload: bool,
    /// Device supports scatter-gather I/O.
    pub scatter_gather: bool,
    /// Device supports VLAN tag insertion/removal.
    pub vlan_offload: bool,
    /// Maximum number of TX descriptors.
    pub max_tx_descriptors: u16,
    /// Maximum number of RX descriptors.
    pub max_rx_descriptors: u16,
}

impl DeviceCapabilities {
    /// Default capabilities for a standard Ethernet device.
    pub const DEFAULT: Self = Self {
        mtu: 1500,
        tx_checksum_offload: false,
        rx_checksum_offload: false,
        scatter_gather: false,
        vlan_offload: false,
        max_tx_descriptors: 256,
        max_rx_descriptors: 256,
    };
}

/// Network device statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceStats {
    /// Total bytes transmitted.
    pub tx_bytes: u64,
    /// Total packets transmitted.
    pub tx_packets: u64,
    /// Transmission errors.
    pub tx_errors: u64,
    /// Packets dropped on transmit.
    pub tx_dropped: u64,
    /// Total bytes received.
    pub rx_bytes: u64,
    /// Total packets received.
    pub rx_packets: u64,
    /// Receive errors.
    pub rx_errors: u64,
    /// Packets dropped on receive.
    pub rx_dropped: u64,
    /// Multicast packets received.
    pub rx_multicast: u64,
    /// Broadcast packets received.
    pub rx_broadcast: u64,
}

impl DeviceStats {
    /// Creates empty statistics.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            tx_bytes: 0,
            tx_packets: 0,
            tx_errors: 0,
            tx_dropped: 0,
            rx_bytes: 0,
            rx_packets: 0,
            rx_errors: 0,
            rx_dropped: 0,
            rx_multicast: 0,
            rx_broadcast: 0,
        }
    }

    /// Resets all statistics to zero.
    #[inline]
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Link status of a network device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkStatus {
    /// Link is down (no carrier).
    Down,
    /// Link is up at the specified speed.
    Up(LinkSpeed),
}

/// Link speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkSpeed {
    /// 10 Mbps.
    Speed10M,
    /// 100 Mbps.
    Speed100M,
    /// 1 Gbps.
    Speed1G,
    /// 2.5 Gbps.
    Speed2_5G,
    /// 5 Gbps.
    Speed5G,
    /// 10 Gbps.
    Speed10G,
    /// Unknown speed.
    Unknown,
}

impl LinkSpeed {
    /// Returns the speed in bits per second.
    #[inline]
    #[must_use]
    pub const fn bits_per_second(&self) -> u64 {
        match self {
            Self::Speed10M => 10_000_000,
            Self::Speed100M => 100_000_000,
            Self::Speed1G => 1_000_000_000,
            Self::Speed2_5G => 2_500_000_000,
            Self::Speed5G => 5_000_000_000,
            Self::Speed10G => 10_000_000_000,
            Self::Unknown => 0,
        }
    }
}

/// Network device trait.
///
/// This trait abstracts the hardware-specific details of a network interface,
/// allowing the network stack to work with different drivers.
pub trait NetworkDevice {
    /// Returns the MAC address of the device.
    fn mac_address(&self) -> MacAddress;

    /// Returns the device capabilities.
    fn capabilities(&self) -> DeviceCapabilities;

    /// Returns the current link status.
    fn link_status(&self) -> LinkStatus;

    /// Returns device statistics.
    fn stats(&self) -> DeviceStats;

    /// Sends a frame over the network.
    ///
    /// The frame should be a complete Ethernet frame including header.
    ///
    /// # Errors
    ///
    /// Returns an error if the transmission fails.
    fn send(&mut self, frame: &[u8]) -> NetResult<()>;

    /// Receives a frame from the network.
    ///
    /// Returns `None` if no frame is available (non-blocking).
    /// Returns `Some(len)` with the frame data written to `buf`.
    ///
    /// # Errors
    ///
    /// Returns an error if reception fails.
    fn receive(&mut self, buf: &mut [u8]) -> NetResult<Option<usize>>;

    /// Sets the device into promiscuous mode.
    fn set_promiscuous(&mut self, enabled: bool) -> NetResult<()>;

    /// Enables or disables the device.
    fn set_enabled(&mut self, enabled: bool) -> NetResult<()>;

    /// Returns true if the device is enabled.
    fn is_enabled(&self) -> bool;

    /// Resets the device.
    fn reset(&mut self) -> NetResult<()>;
}

/// Loopback device for testing.
///
/// This device echoes all sent frames back for reception.
#[derive(Debug)]
pub struct LoopbackDevice {
    mac: MacAddress,
    enabled: bool,
    stats: DeviceStats,
    /// Ring buffer for loopback frames.
    buffer: [[u8; 2048]; 16],
    /// Frame lengths in the buffer.
    lengths: [usize; 16],
    /// Write index.
    write_idx: usize,
    /// Read index.
    read_idx: usize,
    /// Number of frames in buffer.
    count: usize,
}

impl LoopbackDevice {
    /// Creates a new loopback device.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mac: MacAddress::new([0x02, 0x00, 0x00, 0x00, 0x00, 0x01]),
            enabled: true,
            stats: DeviceStats::new(),
            buffer: [[0u8; 2048]; 16],
            lengths: [0; 16],
            write_idx: 0,
            read_idx: 0,
            count: 0,
        }
    }

    /// Creates a loopback device with a custom MAC address.
    #[inline]
    #[must_use]
    pub const fn with_mac(mac: MacAddress) -> Self {
        Self {
            mac,
            enabled: true,
            stats: DeviceStats::new(),
            buffer: [[0u8; 2048]; 16],
            lengths: [0; 16],
            write_idx: 0,
            read_idx: 0,
            count: 0,
        }
    }
}

impl Default for LoopbackDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkDevice for LoopbackDevice {
    fn mac_address(&self) -> MacAddress {
        self.mac
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            mtu: 65535, // Loopback can handle larger frames
            tx_checksum_offload: false,
            rx_checksum_offload: false,
            scatter_gather: false,
            vlan_offload: false,
            max_tx_descriptors: 16,
            max_rx_descriptors: 16,
        }
    }

    fn link_status(&self) -> LinkStatus {
        if self.enabled {
            LinkStatus::Up(LinkSpeed::Speed10G)
        } else {
            LinkStatus::Down
        }
    }

    fn stats(&self) -> DeviceStats {
        self.stats
    }

    fn send(&mut self, frame: &[u8]) -> NetResult<()> {
        use crate::error::NetError;

        if !self.enabled {
            return Err(NetError::DeviceError);
        }

        if frame.len() > 2048 {
            return Err(NetError::PacketTooLarge);
        }

        if self.count >= 16 {
            self.stats.tx_dropped += 1;
            return Err(NetError::DeviceError);
        }

        // Copy frame to buffer
        self.buffer[self.write_idx][..frame.len()].copy_from_slice(frame);
        self.lengths[self.write_idx] = frame.len();
        self.write_idx = (self.write_idx + 1) % 16;
        self.count += 1;

        self.stats.tx_packets += 1;
        self.stats.tx_bytes += frame.len() as u64;

        Ok(())
    }

    fn receive(&mut self, buf: &mut [u8]) -> NetResult<Option<usize>> {
        use crate::error::NetError;

        if !self.enabled {
            return Err(NetError::DeviceError);
        }

        if self.count == 0 {
            return Ok(None);
        }

        let len = self.lengths[self.read_idx];
        if buf.len() < len {
            return Err(NetError::BufferTooSmall);
        }

        buf[..len].copy_from_slice(&self.buffer[self.read_idx][..len]);
        self.read_idx = (self.read_idx + 1) % 16;
        self.count -= 1;

        self.stats.rx_packets += 1;
        self.stats.rx_bytes += len as u64;

        Ok(Some(len))
    }

    fn set_promiscuous(&mut self, _enabled: bool) -> NetResult<()> {
        // Loopback always accepts all frames
        Ok(())
    }

    fn set_enabled(&mut self, enabled: bool) -> NetResult<()> {
        self.enabled = enabled;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn reset(&mut self) -> NetResult<()> {
        self.stats.reset();
        self.write_idx = 0;
        self.read_idx = 0;
        self.count = 0;
        Ok(())
    }
}

/// Null device that discards all frames.
///
/// Useful for testing or when no actual network is needed.
#[derive(Debug, Default)]
pub struct NullDevice {
    mac: MacAddress,
    enabled: bool,
    stats: DeviceStats,
}

impl NullDevice {
    /// Creates a new null device.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            mac: MacAddress::new([0x02, 0x00, 0x00, 0x00, 0x00, 0x00]),
            enabled: true,
            stats: DeviceStats::new(),
        }
    }
}

impl NetworkDevice for NullDevice {
    fn mac_address(&self) -> MacAddress {
        self.mac
    }

    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities::DEFAULT
    }

    fn link_status(&self) -> LinkStatus {
        if self.enabled {
            LinkStatus::Up(LinkSpeed::Speed1G)
        } else {
            LinkStatus::Down
        }
    }

    fn stats(&self) -> DeviceStats {
        self.stats
    }

    fn send(&mut self, frame: &[u8]) -> NetResult<()> {
        if self.enabled {
            self.stats.tx_packets += 1;
            self.stats.tx_bytes += frame.len() as u64;
            Ok(())
        } else {
            Err(crate::error::NetError::DeviceError)
        }
    }

    fn receive(&mut self, _buf: &mut [u8]) -> NetResult<Option<usize>> {
        Ok(None)
    }

    fn set_promiscuous(&mut self, _enabled: bool) -> NetResult<()> {
        Ok(())
    }

    fn set_enabled(&mut self, enabled: bool) -> NetResult<()> {
        self.enabled = enabled;
        Ok(())
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn reset(&mut self) -> NetResult<()> {
        self.stats.reset();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_capabilities_default() {
        let caps = DeviceCapabilities::DEFAULT;
        assert_eq!(caps.mtu, 1500);
        assert!(!caps.tx_checksum_offload);
    }

    #[test]
    fn test_device_stats() {
        let mut stats = DeviceStats::new();
        assert_eq!(stats.tx_packets, 0);

        stats.tx_packets = 100;
        stats.reset();
        assert_eq!(stats.tx_packets, 0);
    }

    #[test]
    fn test_link_speed() {
        assert_eq!(LinkSpeed::Speed10M.bits_per_second(), 10_000_000);
        assert_eq!(LinkSpeed::Speed1G.bits_per_second(), 1_000_000_000);
        assert_eq!(LinkSpeed::Unknown.bits_per_second(), 0);
    }

    #[test]
    fn test_loopback_device() {
        let mut device = LoopbackDevice::new();

        assert!(device.is_enabled());
        assert!(matches!(device.link_status(), LinkStatus::Up(_)));

        // Send a frame
        let frame = [0u8; 64];
        device.send(&frame).unwrap();

        assert_eq!(device.stats().tx_packets, 1);

        // Receive it back
        let mut buf = [0u8; 128];
        let len = device.receive(&mut buf).unwrap().unwrap();
        assert_eq!(len, 64);

        assert_eq!(device.stats().rx_packets, 1);
    }

    #[test]
    fn test_loopback_multiple_frames() {
        let mut device = LoopbackDevice::new();

        // Send multiple frames
        for i in 0..5 {
            let mut frame = [0u8; 64];
            frame[0] = i;
            device.send(&frame).unwrap();
        }

        // Receive them in order
        let mut buf = [0u8; 128];
        for i in 0..5 {
            let len = device.receive(&mut buf).unwrap().unwrap();
            assert_eq!(len, 64);
            assert_eq!(buf[0], i);
        }

        // No more frames
        assert!(device.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_loopback_disabled() {
        let mut device = LoopbackDevice::new();
        device.set_enabled(false).unwrap();

        assert!(!device.is_enabled());
        assert!(matches!(device.link_status(), LinkStatus::Down));

        let frame = [0u8; 64];
        assert!(device.send(&frame).is_err());
    }

    #[test]
    fn test_null_device() {
        let mut device = NullDevice::new();

        // Send succeeds but is discarded
        let frame = [0u8; 64];
        device.send(&frame).unwrap();
        assert_eq!(device.stats().tx_packets, 1);

        // Receive always returns None
        let mut buf = [0u8; 128];
        assert!(device.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_loopback_reset() {
        let mut device = LoopbackDevice::new();

        let frame = [0u8; 64];
        device.send(&frame).unwrap();

        device.reset().unwrap();

        assert_eq!(device.stats().tx_packets, 0);

        // Frame should be gone
        let mut buf = [0u8; 128];
        assert!(device.receive(&mut buf).unwrap().is_none());
    }
}
