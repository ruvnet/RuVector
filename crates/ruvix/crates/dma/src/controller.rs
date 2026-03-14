//! DMA controller trait definition.

use crate::{DmaChannel, DmaChannelId, DmaConfig, DmaDescriptorChain, DmaResult, DmaStatus};

/// Trait defining the interface for a DMA controller.
///
/// Platform-specific implementations must implement this trait to provide
/// DMA functionality to the RuVix kernel.
pub trait DmaController {
    /// Allocate a free DMA channel.
    ///
    /// # Returns
    ///
    /// A newly allocated DMA channel, or an error if no channels are available.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if no channels are available or allocation fails.
    fn allocate_channel(&mut self) -> DmaResult<DmaChannel>;

    /// Allocate a specific DMA channel by ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The specific channel ID to allocate.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the channel is not available or already in use.
    fn allocate_channel_by_id(&mut self, id: DmaChannelId) -> DmaResult<DmaChannel>;

    /// Release a DMA channel back to the pool.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to release.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the channel is currently busy.
    fn release_channel(&mut self, channel: &DmaChannel) -> DmaResult<()>;

    /// Configure a DMA channel for a transfer.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to configure.
    /// * `config` - The transfer configuration.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if configuration is invalid or the channel is busy.
    fn configure(&mut self, channel: &DmaChannel, config: &DmaConfig) -> DmaResult<()>;

    /// Start a DMA transfer on a channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to start.
    /// * `descriptors` - Optional descriptor chain for scatter-gather transfers.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the channel is not configured or already running.
    fn start_transfer(
        &mut self,
        channel: &DmaChannel,
        descriptors: Option<&DmaDescriptorChain>,
    ) -> DmaResult<()>;

    /// Poll the completion status of a transfer.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to poll.
    ///
    /// # Returns
    ///
    /// The current status of the transfer.
    fn poll_completion(&self, channel: &DmaChannel) -> DmaStatus;

    /// Abort an in-progress transfer.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to abort.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if abort fails.
    fn abort(&mut self, channel: &DmaChannel) -> DmaResult<()>;

    /// Pause a running transfer.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to pause.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the channel is not running or pause is not supported.
    fn pause(&mut self, channel: &DmaChannel) -> DmaResult<()> {
        // Default implementation aborts
        self.abort(channel)
    }

    /// Resume a paused transfer.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to resume.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the channel is not paused.
    fn resume(&mut self, channel: &DmaChannel) -> DmaResult<()> {
        // Default implementation restarts
        self.start_transfer(channel, None)
    }

    /// Get the number of bytes transferred on a channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to query.
    ///
    /// # Returns
    ///
    /// The number of bytes transferred so far.
    fn bytes_transferred(&self, channel: &DmaChannel) -> u64;

    /// Get the number of available channels.
    fn available_channels(&self) -> usize;

    /// Get the total number of channels.
    fn total_channels(&self) -> usize;

    /// Check if a channel supports scatter-gather.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to query.
    fn supports_scatter_gather(&self, channel: &DmaChannel) -> bool;

    /// Synchronize cache for a transfer.
    ///
    /// # Arguments
    ///
    /// * `addr` - Physical address to sync.
    /// * `size` - Size in bytes.
    /// * `for_device` - True if syncing for device access, false for CPU access.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if sync fails.
    fn sync_cache(&mut self, addr: u64, size: usize, for_device: bool) -> DmaResult<()>;

    /// Enable interrupts for a channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to enable interrupts for.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if interrupt enable fails.
    fn enable_interrupt(&mut self, channel: &DmaChannel) -> DmaResult<()>;

    /// Disable interrupts for a channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to disable interrupts for.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if interrupt disable fails.
    fn disable_interrupt(&mut self, channel: &DmaChannel) -> DmaResult<()>;

    /// Clear pending interrupt for a channel.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to clear interrupt for.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if clear fails.
    fn clear_interrupt(&mut self, channel: &DmaChannel) -> DmaResult<()>;

    /// Check if a channel has a pending interrupt.
    ///
    /// # Arguments
    ///
    /// * `channel` - The channel to check.
    fn has_pending_interrupt(&self, channel: &DmaChannel) -> bool;

    /// Reset the DMA controller.
    ///
    /// This aborts all transfers and reinitializes the controller.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if reset fails.
    fn reset(&mut self) -> DmaResult<()>;
}

/// Extension trait for DMA controllers with additional convenience methods.
pub trait DmaControllerExt: DmaController {
    /// Perform a simple memory-to-memory transfer.
    ///
    /// # Arguments
    ///
    /// * `src` - Source physical address.
    /// * `dst` - Destination physical address.
    /// * `len` - Transfer length in bytes.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the transfer fails.
    fn memcpy(&mut self, src: u64, dst: u64, len: u64) -> DmaResult<()> {
        let channel = self.allocate_channel()?;
        let config = DmaConfig::mem_to_mem(src, dst, len);

        self.configure(&channel, &config)?;
        self.start_transfer(&channel, None)?;

        // Poll until complete
        loop {
            match self.poll_completion(&channel) {
                DmaStatus::Complete => break,
                DmaStatus::Error(kind) => {
                    self.release_channel(&channel)?;
                    return Err(kind.into());
                }
                DmaStatus::Aborted => {
                    self.release_channel(&channel)?;
                    return Err(crate::DmaError::config_error());
                }
                _ => continue,
            }
        }

        self.release_channel(&channel)?;
        Ok(())
    }

    /// Perform a transfer to a device with automatic channel management.
    ///
    /// # Arguments
    ///
    /// * `src` - Source memory physical address.
    /// * `device` - Device register physical address.
    /// * `len` - Transfer length in bytes.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the transfer fails.
    fn write_to_device(&mut self, src: u64, device: u64, len: u64) -> DmaResult<()> {
        let channel = self.allocate_channel()?;
        let config = DmaConfig::mem_to_device(src, device, len);

        self.sync_cache(src, len as usize, true)?;
        self.configure(&channel, &config)?;
        self.start_transfer(&channel, None)?;

        loop {
            match self.poll_completion(&channel) {
                DmaStatus::Complete => break,
                DmaStatus::Error(kind) => {
                    self.release_channel(&channel)?;
                    return Err(kind.into());
                }
                DmaStatus::Aborted => {
                    self.release_channel(&channel)?;
                    return Err(crate::DmaError::config_error());
                }
                _ => continue,
            }
        }

        self.release_channel(&channel)?;
        Ok(())
    }

    /// Perform a transfer from a device with automatic channel management.
    ///
    /// # Arguments
    ///
    /// * `device` - Device register physical address.
    /// * `dst` - Destination memory physical address.
    /// * `len` - Transfer length in bytes.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the transfer fails.
    fn read_from_device(&mut self, device: u64, dst: u64, len: u64) -> DmaResult<()> {
        let channel = self.allocate_channel()?;
        let config = DmaConfig::device_to_mem(device, dst, len);

        self.configure(&channel, &config)?;
        self.start_transfer(&channel, None)?;

        loop {
            match self.poll_completion(&channel) {
                DmaStatus::Complete => break,
                DmaStatus::Error(kind) => {
                    self.release_channel(&channel)?;
                    return Err(kind.into());
                }
                DmaStatus::Aborted => {
                    self.release_channel(&channel)?;
                    return Err(crate::DmaError::config_error());
                }
                _ => continue,
            }
        }

        self.sync_cache(dst, len as usize, false)?;
        self.release_channel(&channel)?;
        Ok(())
    }
}

// Blanket implementation for all DmaController implementations
impl<T: DmaController> DmaControllerExt for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DmaChannelId;

    // Mock DMA controller for testing
    struct MockDmaController {
        channels: [bool; 8], // true = allocated
    }

    impl MockDmaController {
        fn new() -> Self {
            Self { channels: [false; 8] }
        }
    }

    impl DmaController for MockDmaController {
        fn allocate_channel(&mut self) -> DmaResult<DmaChannel> {
            for (i, allocated) in self.channels.iter_mut().enumerate() {
                if !*allocated {
                    *allocated = true;
                    let ch = DmaChannel::new(DmaChannelId::new(i as u8));
                    ch.mark_allocated();
                    return Ok(ch);
                }
            }
            Err(crate::DmaError::channel_not_available())
        }

        fn allocate_channel_by_id(&mut self, id: DmaChannelId) -> DmaResult<DmaChannel> {
            let idx = id.as_usize();
            if idx >= self.channels.len() || self.channels[idx] {
                return Err(crate::DmaError::channel_not_available());
            }
            self.channels[idx] = true;
            let ch = DmaChannel::new(id);
            ch.mark_allocated();
            Ok(ch)
        }

        fn release_channel(&mut self, channel: &DmaChannel) -> DmaResult<()> {
            let idx = channel.id().as_usize();
            if idx < self.channels.len() {
                self.channels[idx] = false;
            }
            Ok(())
        }

        fn configure(&mut self, _channel: &DmaChannel, _config: &DmaConfig) -> DmaResult<()> {
            Ok(())
        }

        fn start_transfer(
            &mut self,
            _channel: &DmaChannel,
            _descriptors: Option<&DmaDescriptorChain>,
        ) -> DmaResult<()> {
            Ok(())
        }

        fn poll_completion(&self, _channel: &DmaChannel) -> DmaStatus {
            DmaStatus::Complete
        }

        fn abort(&mut self, _channel: &DmaChannel) -> DmaResult<()> {
            Ok(())
        }

        fn bytes_transferred(&self, _channel: &DmaChannel) -> u64 {
            0
        }

        fn available_channels(&self) -> usize {
            self.channels.iter().filter(|&&x| !x).count()
        }

        fn total_channels(&self) -> usize {
            self.channels.len()
        }

        fn supports_scatter_gather(&self, _channel: &DmaChannel) -> bool {
            true
        }

        fn sync_cache(&mut self, _addr: u64, _size: usize, _for_device: bool) -> DmaResult<()> {
            Ok(())
        }

        fn enable_interrupt(&mut self, _channel: &DmaChannel) -> DmaResult<()> {
            Ok(())
        }

        fn disable_interrupt(&mut self, _channel: &DmaChannel) -> DmaResult<()> {
            Ok(())
        }

        fn clear_interrupt(&mut self, _channel: &DmaChannel) -> DmaResult<()> {
            Ok(())
        }

        fn has_pending_interrupt(&self, _channel: &DmaChannel) -> bool {
            false
        }

        fn reset(&mut self) -> DmaResult<()> {
            self.channels = [false; 8];
            Ok(())
        }
    }

    #[test]
    fn test_mock_controller_allocate() {
        let mut ctrl = MockDmaController::new();

        assert_eq!(ctrl.available_channels(), 8);

        let ch = ctrl.allocate_channel().unwrap();
        assert_eq!(ctrl.available_channels(), 7);
        assert!(ch.is_allocated());

        ctrl.release_channel(&ch).unwrap();
        assert_eq!(ctrl.available_channels(), 8);
    }

    #[test]
    fn test_mock_controller_allocate_by_id() {
        let mut ctrl = MockDmaController::new();

        let ch = ctrl.allocate_channel_by_id(DmaChannelId::new(3)).unwrap();
        assert_eq!(ch.id().as_u8(), 3);

        // Can't allocate same channel again
        assert!(ctrl.allocate_channel_by_id(DmaChannelId::new(3)).is_err());
    }

    #[test]
    fn test_mock_controller_memcpy() {
        let mut ctrl = MockDmaController::new();

        ctrl.memcpy(0x1000, 0x2000, 4096).unwrap();
        assert_eq!(ctrl.available_channels(), 8);
    }

    #[test]
    fn test_mock_controller_device_transfers() {
        let mut ctrl = MockDmaController::new();

        ctrl.write_to_device(0x1000, 0x4000_0000, 256).unwrap();
        ctrl.read_from_device(0x4000_0000, 0x2000, 256).unwrap();
    }

    #[test]
    fn test_mock_controller_reset() {
        let mut ctrl = MockDmaController::new();

        ctrl.allocate_channel().unwrap();
        ctrl.allocate_channel().unwrap();
        assert_eq!(ctrl.available_channels(), 6);

        ctrl.reset().unwrap();
        assert_eq!(ctrl.available_channels(), 8);
    }
}
