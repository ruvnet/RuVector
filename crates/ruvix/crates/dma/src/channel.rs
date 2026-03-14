//! DMA channel representation and management.

use crate::{DmaDirection, DmaStatus};
use core::sync::atomic::{AtomicU8, Ordering};

/// Unique identifier for a DMA channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DmaChannelId(u8);

impl DmaChannelId {
    /// Create a new channel ID.
    ///
    /// # Panics
    ///
    /// Panics if `id` is greater than or equal to `MAX_DMA_CHANNELS`.
    #[must_use]
    pub const fn new(id: u8) -> Self {
        assert!(id < crate::MAX_DMA_CHANNELS as u8);
        Self(id)
    }

    /// Create a new channel ID without bounds checking.
    ///
    /// # Safety
    ///
    /// The caller must ensure `id` is a valid channel number.
    #[must_use]
    pub const fn new_unchecked(id: u8) -> Self {
        Self(id)
    }

    /// Get the raw channel ID value.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }

    /// Get the channel ID as usize for indexing.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u8> for DmaChannelId {
    fn from(id: u8) -> Self {
        Self::new(id)
    }
}

impl From<DmaChannelId> for u8 {
    fn from(id: DmaChannelId) -> Self {
        id.0
    }
}

/// Internal channel state representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum ChannelState {
    Free = 0,
    Allocated = 1,
    Configured = 2,
    Running = 3,
    Paused = 4,
    Complete = 5,
    Error = 6,
}

impl ChannelState {
    const fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Free,
            1 => Self::Allocated,
            2 => Self::Configured,
            3 => Self::Running,
            4 => Self::Paused,
            5 => Self::Complete,
            6 => Self::Error,
            _ => Self::Free,
        }
    }
}

/// Represents a DMA channel with its current configuration and state.
#[derive(Debug)]
pub struct DmaChannel {
    /// Channel identifier.
    id: DmaChannelId,
    /// Current channel state (atomic for thread safety).
    state: AtomicU8,
    /// Direction of transfer.
    direction: DmaDirection,
    /// Number of bytes transferred in current/last operation.
    bytes_transferred: u64,
    /// Source address for current transfer.
    src_addr: u64,
    /// Destination address for current transfer.
    dst_addr: u64,
    /// Total transfer length.
    transfer_length: u64,
    /// Priority level (0 = lowest, 7 = highest).
    priority: u8,
    /// Whether this channel supports scatter-gather.
    scatter_gather: bool,
}

impl DmaChannel {
    /// Create a new DMA channel.
    #[must_use]
    pub const fn new(id: DmaChannelId) -> Self {
        Self {
            id,
            state: AtomicU8::new(ChannelState::Free as u8),
            direction: DmaDirection::MemToMem,
            bytes_transferred: 0,
            src_addr: 0,
            dst_addr: 0,
            transfer_length: 0,
            priority: 0,
            scatter_gather: false,
        }
    }

    /// Create a new DMA channel with scatter-gather support.
    #[must_use]
    pub const fn with_scatter_gather(id: DmaChannelId) -> Self {
        Self {
            id,
            state: AtomicU8::new(ChannelState::Free as u8),
            direction: DmaDirection::MemToMem,
            bytes_transferred: 0,
            src_addr: 0,
            dst_addr: 0,
            transfer_length: 0,
            priority: 0,
            scatter_gather: true,
        }
    }

    /// Get the channel ID.
    #[must_use]
    pub const fn id(&self) -> DmaChannelId {
        self.id
    }

    /// Get the current channel status.
    #[must_use]
    pub fn status(&self) -> DmaStatus {
        let state = ChannelState::from_u8(self.state.load(Ordering::Acquire));
        match state {
            ChannelState::Free | ChannelState::Allocated => DmaStatus::Idle,
            ChannelState::Configured => DmaStatus::Idle,
            ChannelState::Running => DmaStatus::Running,
            ChannelState::Paused => DmaStatus::Paused,
            ChannelState::Complete => DmaStatus::Complete,
            ChannelState::Error => DmaStatus::Error(crate::DmaErrorKind::Unknown),
        }
    }

    /// Check if the channel is currently allocated.
    #[must_use]
    pub fn is_allocated(&self) -> bool {
        let state = ChannelState::from_u8(self.state.load(Ordering::Acquire));
        !matches!(state, ChannelState::Free)
    }

    /// Check if the channel is busy (running or paused).
    #[must_use]
    pub fn is_busy(&self) -> bool {
        let state = ChannelState::from_u8(self.state.load(Ordering::Acquire));
        matches!(state, ChannelState::Running | ChannelState::Paused)
    }

    /// Check if the channel supports scatter-gather.
    #[must_use]
    pub const fn supports_scatter_gather(&self) -> bool {
        self.scatter_gather
    }

    /// Get the transfer direction.
    #[must_use]
    pub const fn direction(&self) -> DmaDirection {
        self.direction
    }

    /// Get the number of bytes transferred.
    #[must_use]
    pub const fn bytes_transferred(&self) -> u64 {
        self.bytes_transferred
    }

    /// Get the source address.
    #[must_use]
    pub const fn src_addr(&self) -> u64 {
        self.src_addr
    }

    /// Get the destination address.
    #[must_use]
    pub const fn dst_addr(&self) -> u64 {
        self.dst_addr
    }

    /// Get the total transfer length.
    #[must_use]
    pub const fn transfer_length(&self) -> u64 {
        self.transfer_length
    }

    /// Get the channel priority.
    #[must_use]
    pub const fn priority(&self) -> u8 {
        self.priority
    }

    /// Set the transfer direction.
    pub fn set_direction(&mut self, direction: DmaDirection) {
        self.direction = direction;
    }

    /// Set the source address.
    pub fn set_src_addr(&mut self, addr: u64) {
        self.src_addr = addr;
    }

    /// Set the destination address.
    pub fn set_dst_addr(&mut self, addr: u64) {
        self.dst_addr = addr;
    }

    /// Set the transfer length.
    pub fn set_transfer_length(&mut self, length: u64) {
        self.transfer_length = length;
    }

    /// Set the channel priority (0-7).
    pub fn set_priority(&mut self, priority: u8) {
        self.priority = priority.min(7);
    }

    /// Mark the channel as allocated.
    pub(crate) fn mark_allocated(&self) {
        self.state
            .store(ChannelState::Allocated as u8, Ordering::Release);
    }

    /// Mark the channel as configured.
    pub(crate) fn mark_configured(&self) {
        self.state
            .store(ChannelState::Configured as u8, Ordering::Release);
    }

    /// Mark the channel as running.
    pub(crate) fn mark_running(&self) {
        self.state
            .store(ChannelState::Running as u8, Ordering::Release);
    }

    /// Mark the channel as paused.
    pub(crate) fn mark_paused(&self) {
        self.state
            .store(ChannelState::Paused as u8, Ordering::Release);
    }

    /// Mark the channel as complete.
    pub(crate) fn mark_complete(&self) {
        self.state
            .store(ChannelState::Complete as u8, Ordering::Release);
    }

    /// Mark the channel as having an error.
    pub(crate) fn mark_error(&self) {
        self.state
            .store(ChannelState::Error as u8, Ordering::Release);
    }

    /// Release the channel back to free state.
    pub(crate) fn release(&self) {
        self.state
            .store(ChannelState::Free as u8, Ordering::Release);
    }

    /// Update bytes transferred count.
    pub(crate) fn update_bytes_transferred(&mut self, bytes: u64) {
        self.bytes_transferred = bytes;
    }

    /// Reset transfer counters.
    pub(crate) fn reset_counters(&mut self) {
        self.bytes_transferred = 0;
    }
}

impl Clone for DmaChannel {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            state: AtomicU8::new(self.state.load(Ordering::Acquire)),
            direction: self.direction,
            bytes_transferred: self.bytes_transferred,
            src_addr: self.src_addr,
            dst_addr: self.dst_addr,
            transfer_length: self.transfer_length,
            priority: self.priority,
            scatter_gather: self.scatter_gather,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_id_creation() {
        let id = DmaChannelId::new(5);
        assert_eq!(id.as_u8(), 5);
        assert_eq!(id.as_usize(), 5);
    }

    #[test]
    fn test_channel_id_from_u8() {
        let id: DmaChannelId = 7u8.into();
        assert_eq!(id.as_u8(), 7);
    }

    #[test]
    fn test_channel_creation() {
        let channel = DmaChannel::new(DmaChannelId::new(0));
        assert_eq!(channel.id().as_u8(), 0);
        assert_eq!(channel.status(), DmaStatus::Idle);
        assert!(!channel.is_allocated());
        assert!(!channel.is_busy());
    }

    #[test]
    fn test_channel_with_scatter_gather() {
        let channel = DmaChannel::with_scatter_gather(DmaChannelId::new(1));
        assert!(channel.supports_scatter_gather());
    }

    #[test]
    fn test_channel_state_transitions() {
        let channel = DmaChannel::new(DmaChannelId::new(2));

        channel.mark_allocated();
        assert!(channel.is_allocated());
        assert_eq!(channel.status(), DmaStatus::Idle);

        channel.mark_configured();
        assert_eq!(channel.status(), DmaStatus::Idle);

        channel.mark_running();
        assert!(channel.is_busy());
        assert_eq!(channel.status(), DmaStatus::Running);

        channel.mark_paused();
        assert!(channel.is_busy());
        assert_eq!(channel.status(), DmaStatus::Paused);

        channel.mark_complete();
        assert!(!channel.is_busy());
        assert_eq!(channel.status(), DmaStatus::Complete);

        channel.release();
        assert!(!channel.is_allocated());
    }

    #[test]
    fn test_channel_configuration() {
        let mut channel = DmaChannel::new(DmaChannelId::new(3));

        channel.set_direction(DmaDirection::MemToDevice);
        assert_eq!(channel.direction(), DmaDirection::MemToDevice);

        channel.set_src_addr(0x1000);
        assert_eq!(channel.src_addr(), 0x1000);

        channel.set_dst_addr(0x2000);
        assert_eq!(channel.dst_addr(), 0x2000);

        channel.set_transfer_length(4096);
        assert_eq!(channel.transfer_length(), 4096);

        channel.set_priority(5);
        assert_eq!(channel.priority(), 5);

        // Test priority clamping
        channel.set_priority(10);
        assert_eq!(channel.priority(), 7);
    }

    #[test]
    fn test_channel_clone() {
        let channel = DmaChannel::new(DmaChannelId::new(4));
        channel.mark_running();

        let cloned = channel.clone();
        assert_eq!(cloned.id(), channel.id());
        assert_eq!(cloned.status(), DmaStatus::Running);
    }
}
