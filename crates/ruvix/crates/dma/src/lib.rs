//! # RuVix DMA Controller Abstraction
//!
//! This crate provides a hardware-agnostic DMA (Direct Memory Access) controller
//! abstraction for the RuVix Cognition Kernel. It enables efficient zero-copy
//! data transfers between memory regions and peripheral devices.
//!
//! ## Design Principles
//!
//! - **No unsafe code in public API** - All unsafe operations are encapsulated
//! - **No std dependency** - `#![no_std]` only
//! - **Cache coherent buffers** - Proper cache management for DMA operations
//! - **Scatter-gather support** - Descriptor chains for non-contiguous transfers
//! - **Platform-agnostic** - Works across ARM64, RISC-V, x86_64
//!
//! ## Architecture
//!
//! The DMA subsystem consists of:
//!
//! - **DmaChannel** - Represents a single DMA channel with its state
//! - **DmaBuffer** - Cache-coherent memory buffer for DMA transfers
//! - **DmaDescriptor** - Scatter-gather descriptor for linked transfers
//! - **DmaController** - Trait defining the DMA controller interface
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_dma::{DmaController, DmaConfig, DmaDirection, DmaBuffer};
//!
//! fn transfer_to_device<D: DmaController>(
//!     dma: &mut D,
//!     buffer: &DmaBuffer,
//!     device_addr: u64,
//! ) -> Result<(), DmaError> {
//!     let channel = dma.allocate_channel()?;
//!
//!     let config = DmaConfig {
//!         direction: DmaDirection::MemToDevice,
//!         src_addr: buffer.physical_addr(),
//!         dst_addr: device_addr,
//!         length: buffer.len(),
//!         ..Default::default()
//!     };
//!
//!     dma.configure(&channel, &config)?;
//!     dma.start_transfer(&channel, None)?;
//!
//!     // Poll for completion
//!     loop {
//!         match dma.poll_completion(&channel)? {
//!             DmaStatus::Complete => break,
//!             DmaStatus::Error(e) => return Err(e),
//!             _ => continue,
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod buffer;
mod channel;
mod config;
mod controller;
mod descriptor;
mod error;

pub use buffer::{DmaBuffer, DmaBufferFlags};
pub use channel::{DmaChannel, DmaChannelId};
pub use config::{DmaConfig, DmaBurstSize, DmaTransferWidth};
pub use controller::DmaController;
pub use descriptor::{DmaDescriptor, DmaDescriptorChain, DmaDescriptorFlags};
pub use error::{DmaError, DmaResult};

/// DMA crate version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Direction of DMA transfer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DmaDirection {
    /// Memory to peripheral device transfer
    MemToDevice = 0,
    /// Peripheral device to memory transfer
    DeviceToMem = 1,
    /// Memory to memory transfer (memcpy acceleration)
    MemToMem = 2,
}

impl DmaDirection {
    /// Returns true if transfer reads from memory
    #[must_use]
    pub const fn reads_memory(self) -> bool {
        matches!(self, Self::MemToDevice | Self::MemToMem)
    }

    /// Returns true if transfer writes to memory
    #[must_use]
    pub const fn writes_memory(self) -> bool {
        matches!(self, Self::DeviceToMem | Self::MemToMem)
    }

    /// Create from raw u8 value
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::MemToDevice),
            1 => Some(Self::DeviceToMem),
            2 => Some(Self::MemToMem),
            _ => None,
        }
    }
}

/// Status of a DMA transfer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaStatus {
    /// Channel is idle, no transfer in progress
    Idle,
    /// Transfer is currently running
    Running,
    /// Transfer completed successfully
    Complete,
    /// Transfer encountered an error
    Error(DmaErrorKind),
    /// Transfer was paused
    Paused,
    /// Transfer was aborted
    Aborted,
}

impl DmaStatus {
    /// Returns true if the channel is busy (running or paused)
    #[must_use]
    pub const fn is_busy(self) -> bool {
        matches!(self, Self::Running | Self::Paused)
    }

    /// Returns true if the transfer finished (complete, error, or aborted)
    #[must_use]
    pub const fn is_finished(self) -> bool {
        matches!(self, Self::Complete | Self::Error(_) | Self::Aborted)
    }

    /// Returns true if the transfer completed successfully
    #[must_use]
    pub const fn is_success(self) -> bool {
        matches!(self, Self::Complete)
    }
}

/// Kind of DMA error that occurred
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaErrorKind {
    /// Bus error during transfer
    BusError,
    /// Address alignment error
    AlignmentError,
    /// Transfer timeout
    Timeout,
    /// Descriptor chain error
    DescriptorError,
    /// Configuration error
    ConfigError,
    /// Permission denied
    PermissionDenied,
    /// Unknown or unspecified error
    Unknown,
}

/// Maximum number of DMA channels supported
pub const MAX_DMA_CHANNELS: usize = 16;

/// Maximum descriptor chain length
pub const MAX_DESCRIPTOR_CHAIN_LENGTH: usize = 256;

/// Default DMA transfer timeout in microseconds
pub const DEFAULT_TRANSFER_TIMEOUT_US: u64 = 1_000_000;

/// Minimum alignment for DMA buffers (typically cache line size)
pub const DMA_BUFFER_ALIGNMENT: usize = 64;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::{
        DmaBuffer, DmaChannel, DmaConfig, DmaController, DmaDescriptor, DmaDirection, DmaError,
        DmaErrorKind, DmaResult, DmaStatus,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dma_direction_reads_memory() {
        assert!(DmaDirection::MemToDevice.reads_memory());
        assert!(!DmaDirection::DeviceToMem.reads_memory());
        assert!(DmaDirection::MemToMem.reads_memory());
    }

    #[test]
    fn test_dma_direction_writes_memory() {
        assert!(!DmaDirection::MemToDevice.writes_memory());
        assert!(DmaDirection::DeviceToMem.writes_memory());
        assert!(DmaDirection::MemToMem.writes_memory());
    }

    #[test]
    fn test_dma_direction_from_u8() {
        assert_eq!(DmaDirection::from_u8(0), Some(DmaDirection::MemToDevice));
        assert_eq!(DmaDirection::from_u8(1), Some(DmaDirection::DeviceToMem));
        assert_eq!(DmaDirection::from_u8(2), Some(DmaDirection::MemToMem));
        assert_eq!(DmaDirection::from_u8(3), None);
    }

    #[test]
    fn test_dma_status_is_busy() {
        assert!(!DmaStatus::Idle.is_busy());
        assert!(DmaStatus::Running.is_busy());
        assert!(!DmaStatus::Complete.is_busy());
        assert!(!DmaStatus::Error(DmaErrorKind::BusError).is_busy());
        assert!(DmaStatus::Paused.is_busy());
        assert!(!DmaStatus::Aborted.is_busy());
    }

    #[test]
    fn test_dma_status_is_finished() {
        assert!(!DmaStatus::Idle.is_finished());
        assert!(!DmaStatus::Running.is_finished());
        assert!(DmaStatus::Complete.is_finished());
        assert!(DmaStatus::Error(DmaErrorKind::BusError).is_finished());
        assert!(!DmaStatus::Paused.is_finished());
        assert!(DmaStatus::Aborted.is_finished());
    }

    #[test]
    fn test_dma_status_is_success() {
        assert!(!DmaStatus::Idle.is_success());
        assert!(!DmaStatus::Running.is_success());
        assert!(DmaStatus::Complete.is_success());
        assert!(!DmaStatus::Error(DmaErrorKind::BusError).is_success());
    }
}
