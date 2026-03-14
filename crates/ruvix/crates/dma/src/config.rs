//! DMA configuration structures.

use crate::DmaDirection;

/// DMA transfer width (data bus width per transaction).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DmaTransferWidth {
    /// 8-bit transfers.
    Byte = 0,
    /// 16-bit transfers.
    HalfWord = 1,
    /// 32-bit transfers.
    #[default]
    Word = 2,
    /// 64-bit transfers.
    DoubleWord = 3,
    /// 128-bit transfers.
    QuadWord = 4,
}

impl DmaTransferWidth {
    /// Get the width in bytes.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::Byte => 1,
            Self::HalfWord => 2,
            Self::Word => 4,
            Self::DoubleWord => 8,
            Self::QuadWord => 16,
        }
    }

    /// Get the width in bits.
    #[must_use]
    pub const fn bits(self) -> usize {
        self.bytes() * 8
    }

    /// Create from byte count.
    #[must_use]
    pub const fn from_bytes(bytes: usize) -> Option<Self> {
        match bytes {
            1 => Some(Self::Byte),
            2 => Some(Self::HalfWord),
            4 => Some(Self::Word),
            8 => Some(Self::DoubleWord),
            16 => Some(Self::QuadWord),
            _ => None,
        }
    }
}

/// DMA burst size (number of transfers per burst).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum DmaBurstSize {
    /// Single transfer (no burst).
    Single = 0,
    /// 4 transfers per burst.
    #[default]
    Burst4 = 1,
    /// 8 transfers per burst.
    Burst8 = 2,
    /// 16 transfers per burst.
    Burst16 = 3,
    /// 32 transfers per burst.
    Burst32 = 4,
    /// 64 transfers per burst.
    Burst64 = 5,
    /// 128 transfers per burst.
    Burst128 = 6,
    /// 256 transfers per burst.
    Burst256 = 7,
}

impl DmaBurstSize {
    /// Get the number of transfers per burst.
    #[must_use]
    pub const fn count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::Burst4 => 4,
            Self::Burst8 => 8,
            Self::Burst16 => 16,
            Self::Burst32 => 32,
            Self::Burst64 => 64,
            Self::Burst128 => 128,
            Self::Burst256 => 256,
        }
    }

    /// Create from burst count.
    #[must_use]
    pub const fn from_count(count: usize) -> Option<Self> {
        match count {
            1 => Some(Self::Single),
            4 => Some(Self::Burst4),
            8 => Some(Self::Burst8),
            16 => Some(Self::Burst16),
            32 => Some(Self::Burst32),
            64 => Some(Self::Burst64),
            128 => Some(Self::Burst128),
            256 => Some(Self::Burst256),
            _ => None,
        }
    }

    /// Get the total bytes per burst for a given transfer width.
    #[must_use]
    pub const fn bytes_per_burst(self, width: DmaTransferWidth) -> usize {
        self.count() * width.bytes()
    }
}

/// Configuration for a DMA transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaConfig {
    /// Transfer direction.
    pub direction: DmaDirection,
    /// Source address.
    pub src_addr: u64,
    /// Destination address.
    pub dst_addr: u64,
    /// Total transfer length in bytes.
    pub length: u64,
    /// Transfer width (data bus width).
    pub width: DmaTransferWidth,
    /// Burst size.
    pub burst_size: DmaBurstSize,
    /// Increment source address after each transfer.
    pub src_increment: bool,
    /// Increment destination address after each transfer.
    pub dst_increment: bool,
    /// Channel priority (0 = lowest, 7 = highest).
    pub priority: u8,
    /// Enable circular mode (auto-restart when complete).
    pub circular: bool,
    /// Enable interrupt on completion.
    pub interrupt_on_complete: bool,
    /// Enable interrupt on half-transfer.
    pub interrupt_on_half: bool,
    /// Enable interrupt on error.
    pub interrupt_on_error: bool,
    /// Timeout in microseconds (0 = no timeout).
    pub timeout_us: u64,
}

impl Default for DmaConfig {
    fn default() -> Self {
        Self {
            direction: DmaDirection::MemToMem,
            src_addr: 0,
            dst_addr: 0,
            length: 0,
            width: DmaTransferWidth::Word,
            burst_size: DmaBurstSize::Burst4,
            src_increment: true,
            dst_increment: true,
            priority: 0,
            circular: false,
            interrupt_on_complete: true,
            interrupt_on_half: false,
            interrupt_on_error: true,
            timeout_us: crate::DEFAULT_TRANSFER_TIMEOUT_US,
        }
    }
}

impl DmaConfig {
    /// Create a new DMA configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            direction: DmaDirection::MemToMem,
            src_addr: 0,
            dst_addr: 0,
            length: 0,
            width: DmaTransferWidth::Word,
            burst_size: DmaBurstSize::Burst4,
            src_increment: true,
            dst_increment: true,
            priority: 0,
            circular: false,
            interrupt_on_complete: true,
            interrupt_on_half: false,
            interrupt_on_error: true,
            timeout_us: crate::DEFAULT_TRANSFER_TIMEOUT_US,
        }
    }

    /// Create a memory-to-memory transfer configuration.
    #[must_use]
    pub const fn mem_to_mem(src_addr: u64, dst_addr: u64, length: u64) -> Self {
        Self {
            direction: DmaDirection::MemToMem,
            src_addr,
            dst_addr,
            length,
            width: DmaTransferWidth::DoubleWord,
            burst_size: DmaBurstSize::Burst16,
            src_increment: true,
            dst_increment: true,
            priority: 0,
            circular: false,
            interrupt_on_complete: true,
            interrupt_on_half: false,
            interrupt_on_error: true,
            timeout_us: crate::DEFAULT_TRANSFER_TIMEOUT_US,
        }
    }

    /// Create a memory-to-device transfer configuration.
    #[must_use]
    pub const fn mem_to_device(src_addr: u64, device_addr: u64, length: u64) -> Self {
        Self {
            direction: DmaDirection::MemToDevice,
            src_addr,
            dst_addr: device_addr,
            length,
            width: DmaTransferWidth::Word,
            burst_size: DmaBurstSize::Burst4,
            src_increment: true,
            dst_increment: false, // Device registers typically don't increment
            priority: 0,
            circular: false,
            interrupt_on_complete: true,
            interrupt_on_half: false,
            interrupt_on_error: true,
            timeout_us: crate::DEFAULT_TRANSFER_TIMEOUT_US,
        }
    }

    /// Create a device-to-memory transfer configuration.
    #[must_use]
    pub const fn device_to_mem(device_addr: u64, dst_addr: u64, length: u64) -> Self {
        Self {
            direction: DmaDirection::DeviceToMem,
            src_addr: device_addr,
            dst_addr,
            length,
            width: DmaTransferWidth::Word,
            burst_size: DmaBurstSize::Burst4,
            src_increment: false, // Device registers typically don't increment
            dst_increment: true,
            priority: 0,
            circular: false,
            interrupt_on_complete: true,
            interrupt_on_half: false,
            interrupt_on_error: true,
            timeout_us: crate::DEFAULT_TRANSFER_TIMEOUT_US,
        }
    }

    /// Set the transfer width.
    #[must_use]
    pub const fn with_width(mut self, width: DmaTransferWidth) -> Self {
        self.width = width;
        self
    }

    /// Set the burst size.
    #[must_use]
    pub const fn with_burst_size(mut self, burst_size: DmaBurstSize) -> Self {
        self.burst_size = burst_size;
        self
    }

    /// Set the priority level.
    #[must_use]
    pub const fn with_priority(mut self, priority: u8) -> Self {
        self.priority = if priority > 7 { 7 } else { priority };
        self
    }

    /// Enable circular mode.
    #[must_use]
    pub const fn with_circular(mut self, circular: bool) -> Self {
        self.circular = circular;
        self
    }

    /// Set timeout in microseconds.
    #[must_use]
    pub const fn with_timeout(mut self, timeout_us: u64) -> Self {
        self.timeout_us = timeout_us;
        self
    }

    /// Configure interrupt settings.
    #[must_use]
    pub const fn with_interrupts(
        mut self,
        on_complete: bool,
        on_half: bool,
        on_error: bool,
    ) -> Self {
        self.interrupt_on_complete = on_complete;
        self.interrupt_on_half = on_half;
        self.interrupt_on_error = on_error;
        self
    }

    /// Validate the configuration.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        // Length must be non-zero
        if self.length == 0 {
            return false;
        }

        // Length must be aligned to transfer width
        if self.length % (self.width.bytes() as u64) != 0 {
            return false;
        }

        // Source must be aligned to transfer width
        if self.src_addr % (self.width.bytes() as u64) != 0 {
            return false;
        }

        // Destination must be aligned to transfer width
        if self.dst_addr % (self.width.bytes() as u64) != 0 {
            return false;
        }

        true
    }

    /// Calculate the number of transfers required.
    #[must_use]
    pub const fn transfer_count(&self) -> u64 {
        self.length / (self.width.bytes() as u64)
    }

    /// Calculate the number of bursts required.
    #[must_use]
    pub const fn burst_count(&self) -> u64 {
        let transfers = self.transfer_count();
        let burst_size = self.burst_size.count() as u64;
        (transfers + burst_size - 1) / burst_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_width() {
        assert_eq!(DmaTransferWidth::Byte.bytes(), 1);
        assert_eq!(DmaTransferWidth::HalfWord.bytes(), 2);
        assert_eq!(DmaTransferWidth::Word.bytes(), 4);
        assert_eq!(DmaTransferWidth::DoubleWord.bytes(), 8);
        assert_eq!(DmaTransferWidth::QuadWord.bytes(), 16);

        assert_eq!(DmaTransferWidth::Word.bits(), 32);

        assert_eq!(DmaTransferWidth::from_bytes(4), Some(DmaTransferWidth::Word));
        assert_eq!(DmaTransferWidth::from_bytes(3), None);
    }

    #[test]
    fn test_burst_size() {
        assert_eq!(DmaBurstSize::Single.count(), 1);
        assert_eq!(DmaBurstSize::Burst16.count(), 16);
        assert_eq!(DmaBurstSize::Burst256.count(), 256);

        assert_eq!(
            DmaBurstSize::Burst8.bytes_per_burst(DmaTransferWidth::Word),
            32
        );

        assert_eq!(DmaBurstSize::from_count(16), Some(DmaBurstSize::Burst16));
        assert_eq!(DmaBurstSize::from_count(15), None);
    }

    #[test]
    fn test_config_mem_to_mem() {
        let config = DmaConfig::mem_to_mem(0x1000, 0x2000, 4096);

        assert_eq!(config.direction, DmaDirection::MemToMem);
        assert_eq!(config.src_addr, 0x1000);
        assert_eq!(config.dst_addr, 0x2000);
        assert_eq!(config.length, 4096);
        assert!(config.src_increment);
        assert!(config.dst_increment);
    }

    #[test]
    fn test_config_mem_to_device() {
        let config = DmaConfig::mem_to_device(0x1000, 0x4000_0000, 256);

        assert_eq!(config.direction, DmaDirection::MemToDevice);
        assert!(config.src_increment);
        assert!(!config.dst_increment);
    }

    #[test]
    fn test_config_device_to_mem() {
        let config = DmaConfig::device_to_mem(0x4000_0000, 0x1000, 256);

        assert_eq!(config.direction, DmaDirection::DeviceToMem);
        assert!(!config.src_increment);
        assert!(config.dst_increment);
    }

    #[test]
    fn test_config_builders() {
        let config = DmaConfig::new()
            .with_width(DmaTransferWidth::DoubleWord)
            .with_burst_size(DmaBurstSize::Burst16)
            .with_priority(5)
            .with_circular(true)
            .with_timeout(500_000)
            .with_interrupts(true, true, true);

        assert_eq!(config.width, DmaTransferWidth::DoubleWord);
        assert_eq!(config.burst_size, DmaBurstSize::Burst16);
        assert_eq!(config.priority, 5);
        assert!(config.circular);
        assert_eq!(config.timeout_us, 500_000);
        assert!(config.interrupt_on_half);
    }

    #[test]
    fn test_config_priority_clamping() {
        let config = DmaConfig::new().with_priority(10);
        assert_eq!(config.priority, 7);
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid = DmaConfig::mem_to_mem(0x1000, 0x2000, 4096);
        assert!(valid.is_valid());

        // Zero length is invalid
        let zero_len = DmaConfig::mem_to_mem(0x1000, 0x2000, 0);
        assert!(!zero_len.is_valid());

        // Unaligned length
        let unaligned_len = DmaConfig::mem_to_mem(0x1000, 0x2000, 7);
        assert!(!unaligned_len.is_valid());

        // Unaligned source
        let unaligned_src = DmaConfig::mem_to_mem(0x1001, 0x2000, 4096);
        assert!(!unaligned_src.is_valid());
    }

    #[test]
    fn test_transfer_count() {
        let config = DmaConfig::mem_to_mem(0x1000, 0x2000, 4096)
            .with_width(DmaTransferWidth::DoubleWord);
        assert_eq!(config.transfer_count(), 512); // 4096 / 8
    }

    #[test]
    fn test_burst_count() {
        let config = DmaConfig::mem_to_mem(0x1000, 0x2000, 4096)
            .with_width(DmaTransferWidth::Word)
            .with_burst_size(DmaBurstSize::Burst8);

        // 4096 / 4 = 1024 transfers, 1024 / 8 = 128 bursts
        assert_eq!(config.burst_count(), 128);
    }
}
