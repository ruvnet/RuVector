//! Kernel queue implementation.
//!
//! This module provides the high-level `KernelQueue` struct that implements
//! the ADR-087 queue IPC primitive.

use core::sync::atomic::{AtomicU32, Ordering};

use ruvix_types::{KernelError, MsgPriority, RegionHandle, RegionPolicy};

use crate::descriptor::{DescriptorValidator, MessageDescriptor};
use crate::ring::{RingBuffer, RingEntry};
use crate::{Duration, Result};

/// Configuration for a kernel queue.
#[derive(Debug, Clone, Copy)]
pub struct QueueConfig {
    /// Ring buffer size (must be power of 2).
    pub ring_size: u32,
    /// Maximum message size in bytes.
    pub max_msg_size: u32,
    /// WIT type ID for message schema validation (0 = no validation).
    pub schema: u32,
}

impl QueueConfig {
    /// Create a new queue configuration.
    ///
    /// # Arguments
    ///
    /// * `ring_size` - Number of ring entries (must be power of 2)
    /// * `max_msg_size` - Maximum message size in bytes
    #[inline]
    pub const fn new(ring_size: u32, max_msg_size: u32) -> Self {
        Self {
            ring_size,
            max_msg_size,
            schema: 0,
        }
    }

    /// Create a configuration with schema validation.
    #[inline]
    pub const fn with_schema(ring_size: u32, max_msg_size: u32, schema: u32) -> Self {
        Self {
            ring_size,
            max_msg_size,
            schema,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        // Ring size must be power of 2
        if self.ring_size == 0 || (self.ring_size & (self.ring_size - 1)) != 0 {
            return Err(KernelError::InvalidArgument);
        }

        // Max message size must be reasonable
        if self.max_msg_size == 0 || self.max_msg_size > 1024 * 1024 {
            return Err(KernelError::InvalidArgument);
        }

        Ok(())
    }

    /// Calculate the required backing memory size.
    pub fn required_memory(&self) -> usize {
        let entry_size = RingEntry::HEADER_SIZE + self.max_msg_size as usize;
        (self.ring_size as usize) * entry_size
    }
}

impl Default for QueueConfig {
    fn default() -> Self {
        Self {
            ring_size: 64,
            max_msg_size: 4096,
            schema: 0,
        }
    }
}

/// A kernel queue for inter-task communication.
///
/// Implements io_uring-style ring buffer IPC as specified in ADR-087 Section 7.
/// Features:
///
/// - Lock-free send/recv using atomic head/tail pointers
/// - Zero-copy descriptor-based messaging for shared regions
/// - Priority support (messages can have Low, Normal, High, Urgent priority)
/// - Optional WIT schema validation
///
/// # Thread Safety
///
/// KernelQueue is `Send` and `Sync`. Multiple tasks can send to the same queue
/// concurrently. Receives are typically done by a single consumer task.
#[cfg(feature = "std")]
pub struct KernelQueue {
    /// The ring buffer.
    ring: RingBuffer,

    /// Queue configuration.
    config: QueueConfig,

    /// Descriptor validator.
    validator: DescriptorValidator,

    /// Number of send operations.
    send_count: AtomicU32,

    /// Number of recv operations.
    recv_count: AtomicU32,
}

#[cfg(feature = "std")]
impl KernelQueue {
    /// Create a new kernel queue.
    ///
    /// # Arguments
    ///
    /// * `config` - Queue configuration
    /// * `region` - Handle to the backing region
    /// * `buffer` - Pointer to the backing memory
    /// * `buffer_len` - Length of the backing memory
    ///
    /// # Errors
    ///
    /// Returns `InvalidParameter` if the configuration is invalid.
    /// Returns `OutOfMemory` if the buffer is too small.
    pub fn new(
        config: QueueConfig,
        region: RegionHandle,
        buffer: *mut u8,
        buffer_len: usize,
    ) -> Result<Self> {
        config.validate()?;

        let ring = RingBuffer::new(
            region,
            config.ring_size,
            config.max_msg_size,
            buffer,
            buffer_len,
        )?;

        Ok(Self {
            ring,
            config,
            validator: DescriptorValidator::new(),
            send_count: AtomicU32::new(0),
            recv_count: AtomicU32::new(0),
        })
    }

    /// Create a queue with heap-allocated backing memory.
    ///
    /// This is a convenience method for testing and std environments.
    pub fn new_heap(config: QueueConfig) -> Result<(Self, Vec<u8>)> {
        config.validate()?;

        let required_size = config.required_memory();
        let mut buffer = vec![0u8; required_size];

        let ring = RingBuffer::new(
            RegionHandle::null(),
            config.ring_size,
            config.max_msg_size,
            buffer.as_mut_ptr(),
            buffer.len(),
        )?;

        let queue = Self {
            ring,
            config,
            validator: DescriptorValidator::new(),
            send_count: AtomicU32::new(0),
            recv_count: AtomicU32::new(0),
        };

        Ok((queue, buffer))
    }

    /// Send a message to the queue.
    ///
    /// # Arguments
    ///
    /// * `msg` - Message payload bytes
    /// * `priority` - Message priority
    ///
    /// # Errors
    ///
    /// Returns `QueueFull` if the queue is full.
    /// Returns `MessageTooLarge` if the message exceeds `max_msg_size`.
    pub fn send(&mut self, msg: &[u8], priority: MsgPriority) -> Result<()> {
        if msg.len() > self.config.max_msg_size as usize {
            return Err(KernelError::MessageTooLarge);
        }

        self.ring.enqueue(msg, priority)?;
        self.send_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Send a zero-copy message via descriptor.
    ///
    /// Instead of copying data, this sends a reference to data in a shared region.
    /// The receiver must read the data from the shared region using the descriptor.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - Reference to data in a shared region
    /// * `region_policy` - Policy of the region referenced by the descriptor
    /// * `region_size` - Size of the region
    /// * `priority` - Message priority
    ///
    /// # Errors
    ///
    /// Returns `QueueFull` if the queue is full.
    /// Returns `InvalidDescriptorRegion` if the region policy doesn't allow descriptors.
    /// Returns `InvalidParameter` if the descriptor references out-of-bounds memory.
    ///
    /// # TOCTOU Protection
    ///
    /// Only Immutable or AppendOnly regions are allowed. This prevents the sender
    /// from modifying the data after sending but before the receiver processes it.
    pub fn send_descriptor(
        &mut self,
        descriptor: &MessageDescriptor,
        region_policy: &RegionPolicy,
        region_size: usize,
        priority: MsgPriority,
    ) -> Result<()> {
        // Validate the descriptor and region policy
        self.validator
            .validate(descriptor, region_policy, region_size)?;

        self.ring.enqueue_descriptor(descriptor, priority)?;
        self.send_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Receive a message from the queue (non-blocking).
    ///
    /// # Arguments
    ///
    /// * `buf` - Buffer to receive the message data
    ///
    /// # Returns
    ///
    /// On success, returns the number of bytes received. For descriptor messages,
    /// the descriptor is written to `buf` and you should use `MessageDescriptor::from_bytes`
    /// to parse it.
    ///
    /// # Errors
    ///
    /// Returns `QueueEmpty` if no messages are available.
    pub fn recv(&mut self, buf: &mut [u8]) -> Result<usize> {
        let entry = self.ring.dequeue(buf)?;
        self.recv_count.fetch_add(1, Ordering::Relaxed);
        Ok(entry.length as usize)
    }

    /// Receive a message from the queue with timeout.
    ///
    /// # Arguments
    ///
    /// * `buf` - Buffer to receive the message data
    /// * `timeout` - Maximum time to wait for a message
    ///
    /// # Returns
    ///
    /// On success, returns the number of bytes received.
    ///
    /// # Errors
    ///
    /// Returns `Timeout` if no message arrived within the timeout period.
    pub fn recv_timeout(&mut self, buf: &mut [u8], timeout: Duration) -> Result<usize> {
        let start = std::time::Instant::now();

        loop {
            match self.recv(buf) {
                Ok(len) => return Ok(len),
                Err(KernelError::QueueEmpty) => {
                    if start.elapsed() >= timeout {
                        return Err(KernelError::Timeout);
                    }
                    // Spin briefly before retrying
                    std::hint::spin_loop();
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Receive a message, distinguishing between inline data and descriptors.
    ///
    /// # Returns
    ///
    /// Returns `ReceivedMessage` which indicates whether the data is inline
    /// or a descriptor reference.
    pub fn recv_typed(&mut self, buf: &mut [u8]) -> Result<ReceivedMessage> {
        let entry = self.ring.dequeue(buf)?;
        self.recv_count.fetch_add(1, Ordering::Relaxed);

        if entry.is_descriptor() {
            let descriptor =
                MessageDescriptor::from_bytes(buf).ok_or(KernelError::InternalError)?;
            Ok(ReceivedMessage::Descriptor {
                descriptor,
                priority: entry.priority(),
            })
        } else {
            Ok(ReceivedMessage::Inline {
                length: entry.length as usize,
                priority: entry.priority(),
            })
        }
    }

    /// Peek at the next message without removing it.
    pub fn peek(&self) -> Option<RingEntry> {
        self.ring.peek()
    }

    /// Check if the queue is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Check if the queue is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.ring.is_full()
    }

    /// Get the number of messages currently in the queue.
    #[inline]
    pub fn len(&self) -> u32 {
        self.ring.len()
    }

    /// Get the number of available slots.
    #[inline]
    pub fn available(&self) -> u32 {
        self.ring.available()
    }

    /// Get the queue configuration.
    #[inline]
    pub fn config(&self) -> &QueueConfig {
        &self.config
    }

    /// Get the backing region handle.
    #[inline]
    pub fn region(&self) -> RegionHandle {
        self.ring.region()
    }

    /// Get the total number of send operations.
    #[inline]
    pub fn send_count(&self) -> u32 {
        self.send_count.load(Ordering::Relaxed)
    }

    /// Get the total number of recv operations.
    #[inline]
    pub fn recv_count(&self) -> u32 {
        self.recv_count.load(Ordering::Relaxed)
    }

    /// Get ring buffer statistics.
    #[cfg(feature = "stats")]
    pub fn stats(&self) -> &crate::ring::RingStats {
        self.ring.stats()
    }
}

/// Result of a typed receive operation.
#[derive(Debug, Clone)]
pub enum ReceivedMessage {
    /// Inline data was received and is in the buffer.
    Inline {
        /// Length of the data in bytes.
        length: usize,
        /// Message priority.
        priority: MsgPriority,
    },
    /// A descriptor was received (zero-copy reference).
    Descriptor {
        /// The message descriptor.
        descriptor: MessageDescriptor,
        /// Message priority.
        priority: MsgPriority,
    },
}

impl ReceivedMessage {
    /// Check if this is a descriptor message.
    #[inline]
    pub fn is_descriptor(&self) -> bool {
        matches!(self, ReceivedMessage::Descriptor { .. })
    }

    /// Get the message priority.
    #[inline]
    pub fn priority(&self) -> MsgPriority {
        match self {
            ReceivedMessage::Inline { priority, .. } => *priority,
            ReceivedMessage::Descriptor { priority, .. } => *priority,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_config_validate() {
        // Valid config
        assert!(QueueConfig::new(64, 4096).validate().is_ok());
        assert!(QueueConfig::new(128, 1024).validate().is_ok());

        // Invalid ring size (not power of 2)
        assert!(QueueConfig::new(63, 4096).validate().is_err());
        assert!(QueueConfig::new(0, 4096).validate().is_err());

        // Invalid message size
        assert!(QueueConfig::new(64, 0).validate().is_err());
        assert!(QueueConfig::new(64, 2 * 1024 * 1024).validate().is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_basic() {
        let config = QueueConfig::new(16, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        assert!(queue.is_empty());
        assert!(!queue.is_full());

        // Send a message
        queue.send(b"hello world", MsgPriority::Normal).unwrap();
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        // Receive the message
        let mut buf = [0u8; 256];
        let len = queue.recv(&mut buf).unwrap();
        assert_eq!(len, 11);
        assert_eq!(&buf[..len], b"hello world");
        assert!(queue.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_send_count() {
        let config = QueueConfig::new(16, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        for i in 0..5 {
            let msg = format!("msg{}", i);
            queue.send(msg.as_bytes(), MsgPriority::Normal).unwrap();
        }

        assert_eq!(queue.send_count(), 5);
        assert_eq!(queue.recv_count(), 0);

        let mut buf = [0u8; 256];
        for _ in 0..3 {
            queue.recv(&mut buf).unwrap();
        }

        assert_eq!(queue.recv_count(), 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_full() {
        let config = QueueConfig::new(4, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        // Fill the queue
        for i in 0..4 {
            let msg = format!("msg{}", i);
            queue.send(msg.as_bytes(), MsgPriority::Normal).unwrap();
        }

        assert!(queue.is_full());

        // Should fail
        let result = queue.send(b"overflow", MsgPriority::Normal);
        assert!(matches!(result, Err(KernelError::QueueFull)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_message_too_large() {
        let config = QueueConfig::new(16, 64);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        let large_msg = vec![0u8; 128];
        let result = queue.send(&large_msg, MsgPriority::Normal);
        assert!(matches!(result, Err(KernelError::MessageTooLarge)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_recv_empty() {
        let config = QueueConfig::new(16, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        let mut buf = [0u8; 256];
        let result = queue.recv(&mut buf);
        assert!(matches!(result, Err(KernelError::QueueEmpty)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_recv_timeout() {
        let config = QueueConfig::new(16, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        let mut buf = [0u8; 256];
        let start = std::time::Instant::now();
        let result = queue.recv_timeout(&mut buf, Duration::from_millis(10));
        let elapsed = start.elapsed();

        assert!(matches!(result, Err(KernelError::Timeout)));
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_descriptor_slab_rejected() {
        let config = QueueConfig::new(16, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        use ruvix_types::Handle;
        let desc = MessageDescriptor::new(RegionHandle(Handle::new(1, 0)), 0, 100);

        // Slab regions should be rejected
        let result = queue.send_descriptor(
            &desc,
            &RegionPolicy::Slab {
                slot_size: 64,
                slot_count: 16,
            },
            1024,
            MsgPriority::Normal,
        );

        assert!(matches!(result, Err(KernelError::InvalidDescriptorRegion)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_descriptor_immutable_ok() {
        let config = QueueConfig::new(16, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        use ruvix_types::Handle;
        let desc = MessageDescriptor::new(RegionHandle(Handle::new(1, 0)), 0, 100);

        // Immutable regions should be allowed
        let result =
            queue.send_descriptor(&desc, &RegionPolicy::Immutable, 1024, MsgPriority::Normal);

        assert!(result.is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_queue_descriptor_out_of_bounds() {
        let config = QueueConfig::new(16, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        use ruvix_types::Handle;
        let desc = MessageDescriptor::new(RegionHandle(Handle::new(1, 0)), 900, 200);

        // Offset + length exceeds region size
        let result =
            queue.send_descriptor(&desc, &RegionPolicy::Immutable, 1000, MsgPriority::Normal);

        assert!(matches!(result, Err(KernelError::InvalidArgument)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_received_message_types() {
        let config = QueueConfig::new(16, 256);
        let (mut queue, _buffer) = KernelQueue::new_heap(config).unwrap();

        // Send inline message
        queue.send(b"inline", MsgPriority::High).unwrap();

        let mut buf = [0u8; 256];
        let msg = queue.recv_typed(&mut buf).unwrap();

        match msg {
            ReceivedMessage::Inline { length, priority } => {
                assert_eq!(length, 6);
                assert_eq!(priority, MsgPriority::High);
                assert_eq!(&buf[..length], b"inline");
            }
            _ => panic!("Expected inline message"),
        }
    }
}
