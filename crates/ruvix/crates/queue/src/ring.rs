//! Lock-free ring buffer implementation.
//!
//! This module provides the core ring buffer data structure used by kernel queues.
//! The design is inspired by io_uring's submission/completion queues.

use core::sync::atomic::{AtomicU32, Ordering};

use ruvix_types::{KernelError, MsgPriority, RegionHandle};

use crate::Result;

/// A lock-free ring buffer entry.
///
/// Each entry contains a header followed by payload data. The header includes
/// length, priority, flags, and sequence number for ordering.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RingEntry {
    /// Length of the payload in bytes.
    pub length: u32,
    /// Message priority (0-3).
    pub priority: u8,
    /// Entry flags.
    pub flags: u8,
    /// Sequence number for ordering.
    pub sequence: u16,
}

impl RingEntry {
    /// Size of a ring entry header in bytes.
    pub const HEADER_SIZE: usize = core::mem::size_of::<Self>();

    /// Flag indicating this entry contains a descriptor instead of inline data.
    pub const FLAG_DESCRIPTOR: u8 = 1 << 0;

    /// Flag indicating this entry is part of a multi-entry message.
    pub const FLAG_CONTINUATION: u8 = 1 << 1;

    /// Flag indicating this is the final entry of a multi-entry message.
    pub const FLAG_FINAL: u8 = 1 << 2;

    /// Create a new ring entry for inline data.
    #[inline]
    pub const fn new_inline(length: u32, priority: MsgPriority, sequence: u16) -> Self {
        Self {
            length,
            priority: priority as u8,
            flags: 0,
            sequence,
        }
    }

    /// Create a new ring entry for a descriptor.
    #[inline]
    pub const fn new_descriptor(priority: MsgPriority, sequence: u16) -> Self {
        Self {
            length: 24, // Size of MessageDescriptor
            priority: priority as u8,
            flags: Self::FLAG_DESCRIPTOR,
            sequence,
        }
    }

    /// Check if this entry contains a descriptor.
    #[inline]
    pub const fn is_descriptor(&self) -> bool {
        (self.flags & Self::FLAG_DESCRIPTOR) != 0
    }

    /// Check if this is a continuation entry.
    #[inline]
    pub const fn is_continuation(&self) -> bool {
        (self.flags & Self::FLAG_CONTINUATION) != 0
    }

    /// Get the priority as MsgPriority.
    #[inline]
    pub fn priority(&self) -> MsgPriority {
        MsgPriority::from_u8(self.priority).unwrap_or(MsgPriority::Normal)
    }
}

/// Statistics for ring buffer operations.
#[derive(Debug, Clone, Default)]
pub struct RingStats {
    /// Total messages enqueued.
    pub enqueued: u64,
    /// Total messages dequeued.
    pub dequeued: u64,
    /// Total bytes enqueued.
    pub bytes_enqueued: u64,
    /// Total bytes dequeued.
    pub bytes_dequeued: u64,
    /// Number of times the ring was full.
    pub full_count: u64,
    /// Number of times the ring was empty.
    pub empty_count: u64,
}

/// A lock-free ring buffer for message passing.
///
/// The ring buffer uses atomic head/tail pointers for lock-free operation.
/// Messages are stored inline after the entry header, or as descriptors
/// pointing to shared region data.
pub struct RingBuffer {
    /// The backing memory region handle.
    region: RegionHandle,

    /// Ring size (must be power of 2).
    size: u32,

    /// Mask for index calculation (size - 1).
    mask: u32,

    /// Maximum message size in bytes.
    max_msg_size: u32,

    /// Entry size (header + max payload).
    entry_size: u32,

    /// Submission queue head (producer writes).
    sq_head: AtomicU32,

    /// Submission queue tail (consumer advances).
    sq_tail: AtomicU32,

    /// Next sequence number.
    sequence: AtomicU32,

    /// Pointer to the ring buffer memory.
    #[cfg(feature = "std")]
    buffer: *mut u8,

    /// Buffer length.
    #[cfg(feature = "std")]
    buffer_len: usize,

    /// Statistics.
    #[cfg(feature = "stats")]
    stats: RingStats,
}

// SAFETY: RingBuffer uses atomic operations for thread safety.
// The buffer pointer is only accessed with proper synchronization.
unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    /// Create a new ring buffer.
    ///
    /// # Arguments
    ///
    /// * `region` - Handle to the backing region
    /// * `size` - Number of entries (must be power of 2)
    /// * `max_msg_size` - Maximum message size in bytes
    /// * `buffer` - Pointer to the ring buffer memory
    /// * `buffer_len` - Length of the buffer
    ///
    /// # Errors
    ///
    /// Returns `InvalidParameter` if size is not a power of 2.
    #[cfg(feature = "std")]
    pub fn new(
        region: RegionHandle,
        size: u32,
        max_msg_size: u32,
        buffer: *mut u8,
        buffer_len: usize,
    ) -> Result<Self> {
        // Validate size is power of 2
        if size == 0 || (size & (size - 1)) != 0 {
            return Err(KernelError::InvalidArgument);
        }

        let entry_size = RingEntry::HEADER_SIZE as u32 + max_msg_size;
        let required_size = (size as usize) * (entry_size as usize);

        if buffer_len < required_size {
            return Err(KernelError::OutOfMemory);
        }

        Ok(Self {
            region,
            size,
            mask: size - 1,
            max_msg_size,
            entry_size,
            sq_head: AtomicU32::new(0),
            sq_tail: AtomicU32::new(0),
            sequence: AtomicU32::new(0),
            buffer,
            buffer_len,
            #[cfg(feature = "stats")]
            stats: RingStats::default(),
        })
    }

    /// Returns the region handle.
    #[inline]
    pub fn region(&self) -> RegionHandle {
        self.region
    }

    /// Returns the ring size.
    #[inline]
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Returns the maximum message size.
    #[inline]
    pub fn max_msg_size(&self) -> u32 {
        self.max_msg_size
    }

    /// Returns the number of entries currently in the ring.
    #[inline]
    pub fn len(&self) -> u32 {
        let head = self.sq_head.load(Ordering::Acquire);
        let tail = self.sq_tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Returns true if the ring is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the ring is full.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() >= self.size
    }

    /// Returns the number of free slots.
    #[inline]
    pub fn available(&self) -> u32 {
        self.size.saturating_sub(self.len())
    }

    /// Enqueue a message into the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `data` - Message payload
    /// * `priority` - Message priority
    ///
    /// # Errors
    ///
    /// Returns `QueueFull` if the ring is full.
    /// Returns `MessageTooLarge` if the message exceeds max_msg_size.
    #[cfg(feature = "std")]
    pub fn enqueue(&mut self, data: &[u8], priority: MsgPriority) -> Result<()> {
        if data.len() > self.max_msg_size as usize {
            return Err(KernelError::MessageTooLarge);
        }

        // Check if there's space
        let head = self.sq_head.load(Ordering::Relaxed);
        let tail = self.sq_tail.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.size {
            #[cfg(feature = "stats")]
            {
                self.stats.full_count += 1;
            }
            return Err(KernelError::QueueFull);
        }

        // Calculate entry offset
        let index = head & self.mask;
        let offset = (index as usize) * (self.entry_size as usize);

        // Get sequence number
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed) as u16;

        // Create entry header
        let entry = RingEntry::new_inline(data.len() as u32, priority, seq);

        // Write header and data
        // SAFETY: We've verified bounds and have exclusive write access
        unsafe {
            let entry_ptr = self.buffer.add(offset);
            core::ptr::write(entry_ptr as *mut RingEntry, entry);

            let data_ptr = entry_ptr.add(RingEntry::HEADER_SIZE);
            core::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, data.len());
        }

        // Publish the entry
        self.sq_head.store(head.wrapping_add(1), Ordering::Release);

        #[cfg(feature = "stats")]
        {
            self.stats.enqueued += 1;
            self.stats.bytes_enqueued += data.len() as u64;
        }

        Ok(())
    }

    /// Enqueue a descriptor into the ring buffer.
    ///
    /// This is used for zero-copy message passing when the data is in a shared region.
    ///
    /// # Arguments
    ///
    /// * `descriptor` - The message descriptor (region, offset, length)
    /// * `priority` - Message priority
    ///
    /// # Errors
    ///
    /// Returns `QueueFull` if the ring is full.
    #[cfg(feature = "std")]
    pub fn enqueue_descriptor(
        &mut self,
        descriptor: &crate::MessageDescriptor,
        priority: MsgPriority,
    ) -> Result<()> {
        // Check if there's space
        let head = self.sq_head.load(Ordering::Relaxed);
        let tail = self.sq_tail.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= self.size {
            #[cfg(feature = "stats")]
            {
                self.stats.full_count += 1;
            }
            return Err(KernelError::QueueFull);
        }

        // Calculate entry offset
        let index = head & self.mask;
        let offset = (index as usize) * (self.entry_size as usize);

        // Get sequence number
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed) as u16;

        // Create entry header
        let entry = RingEntry::new_descriptor(priority, seq);

        // Write header and descriptor
        // SAFETY: We've verified bounds and have exclusive write access
        unsafe {
            let entry_ptr = self.buffer.add(offset);
            core::ptr::write(entry_ptr as *mut RingEntry, entry);

            let desc_ptr = entry_ptr.add(RingEntry::HEADER_SIZE);
            core::ptr::write(desc_ptr as *mut crate::MessageDescriptor, *descriptor);
        }

        // Publish the entry
        self.sq_head.store(head.wrapping_add(1), Ordering::Release);

        #[cfg(feature = "stats")]
        {
            self.stats.enqueued += 1;
            self.stats.bytes_enqueued += descriptor.length as u64;
        }

        Ok(())
    }

    /// Dequeue a message from the ring buffer.
    ///
    /// # Arguments
    ///
    /// * `buf` - Buffer to receive the message data
    ///
    /// # Returns
    ///
    /// On success, returns the ring entry header. For inline data, the actual
    /// data is copied to `buf`. For descriptors, the descriptor is in `buf`.
    ///
    /// # Errors
    ///
    /// Returns `QueueEmpty` if the ring is empty.
    #[cfg(feature = "std")]
    pub fn dequeue(&mut self, buf: &mut [u8]) -> Result<RingEntry> {
        // Check if there's data
        let head = self.sq_head.load(Ordering::Acquire);
        let tail = self.sq_tail.load(Ordering::Relaxed);

        if head == tail {
            #[cfg(feature = "stats")]
            {
                self.stats.empty_count += 1;
            }
            return Err(KernelError::QueueEmpty);
        }

        // Calculate entry offset
        let index = tail & self.mask;
        let offset = (index as usize) * (self.entry_size as usize);

        // Read entry
        // SAFETY: We've verified there's a valid entry at this position
        let entry = unsafe {
            let entry_ptr = self.buffer.add(offset);
            core::ptr::read(entry_ptr as *const RingEntry)
        };

        // Read payload
        let payload_len = entry.length as usize;
        if payload_len > buf.len() {
            return Err(KernelError::MessageTooLarge);
        }

        // SAFETY: We've verified bounds
        unsafe {
            let data_ptr = self.buffer.add(offset + RingEntry::HEADER_SIZE);
            core::ptr::copy_nonoverlapping(data_ptr, buf.as_mut_ptr(), payload_len);
        }

        // Advance tail
        self.sq_tail.store(tail.wrapping_add(1), Ordering::Release);

        #[cfg(feature = "stats")]
        {
            self.stats.dequeued += 1;
            self.stats.bytes_dequeued += payload_len as u64;
        }

        Ok(entry)
    }

    /// Peek at the next entry without removing it.
    #[cfg(feature = "std")]
    pub fn peek(&self) -> Option<RingEntry> {
        let head = self.sq_head.load(Ordering::Acquire);
        let tail = self.sq_tail.load(Ordering::Relaxed);

        if head == tail {
            return None;
        }

        let index = tail & self.mask;
        let offset = (index as usize) * (self.entry_size as usize);

        // SAFETY: We've verified there's a valid entry
        let entry = unsafe {
            let entry_ptr = self.buffer.add(offset);
            core::ptr::read(entry_ptr as *const RingEntry)
        };

        Some(entry)
    }

    /// Get ring buffer statistics.
    #[cfg(feature = "stats")]
    pub fn stats(&self) -> &RingStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "std")]
    #[test]
    fn test_ring_entry_size() {
        assert_eq!(RingEntry::HEADER_SIZE, 8);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_ring_buffer_basic() {
        // Need: size * (header_size + max_msg_size) = 64 * (8 + 4096) = 262656 bytes
        let mut backing = vec![0u8; 64 * (RingEntry::HEADER_SIZE + 4096)];
        let mut ring = RingBuffer::new(
            RegionHandle::null(),
            64,
            4096,
            backing.as_mut_ptr(),
            backing.len(),
        )
        .unwrap();

        assert!(ring.is_empty());
        assert!(!ring.is_full());
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.available(), 64);

        // Enqueue
        ring.enqueue(b"hello", MsgPriority::Normal).unwrap();
        assert_eq!(ring.len(), 1);
        assert!(!ring.is_empty());

        // Dequeue
        let mut buf = [0u8; 4096];
        let entry = ring.dequeue(&mut buf).unwrap();
        assert_eq!(entry.length, 5);
        assert_eq!(&buf[..5], b"hello");
        assert!(ring.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_ring_buffer_full() {
        let mut backing = vec![0u8; 4 * 1024]; // Small buffer for 4 entries
        let mut ring = RingBuffer::new(
            RegionHandle::null(),
            4,
            256,
            backing.as_mut_ptr(),
            backing.len(),
        )
        .unwrap();

        // Fill the ring
        for i in 0..4 {
            let msg = format!("msg{}", i);
            ring.enqueue(msg.as_bytes(), MsgPriority::Normal).unwrap();
        }

        assert!(ring.is_full());

        // Should fail on next enqueue
        let result = ring.enqueue(b"overflow", MsgPriority::Normal);
        assert!(matches!(result, Err(KernelError::QueueFull)));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_ring_buffer_wraparound() {
        let mut backing = vec![0u8; 4 * 1024];
        let mut ring = RingBuffer::new(
            RegionHandle::null(),
            4,
            256,
            backing.as_mut_ptr(),
            backing.len(),
        )
        .unwrap();

        let mut buf = [0u8; 256];

        // Fill and drain multiple times to test wraparound
        for round in 0..10 {
            for i in 0..4 {
                let msg = format!("r{}m{}", round, i);
                ring.enqueue(msg.as_bytes(), MsgPriority::Normal).unwrap();
            }

            for i in 0..4 {
                let entry = ring.dequeue(&mut buf).unwrap();
                let expected = format!("r{}m{}", round, i);
                assert_eq!(&buf[..entry.length as usize], expected.as_bytes());
            }

            assert!(ring.is_empty());
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_ring_buffer_priority() {
        let mut backing = vec![0u8; 8 * 1024];
        let mut ring = RingBuffer::new(
            RegionHandle::null(),
            8,
            256,
            backing.as_mut_ptr(),
            backing.len(),
        )
        .unwrap();

        // Enqueue with different priorities
        ring.enqueue(b"low", MsgPriority::Low).unwrap();
        ring.enqueue(b"high", MsgPriority::High).unwrap();
        ring.enqueue(b"urgent", MsgPriority::Urgent).unwrap();

        // Note: This basic ring doesn't reorder by priority.
        // Priority ordering is handled at a higher level in KernelQueue.
        let mut buf = [0u8; 256];

        let e1 = ring.dequeue(&mut buf).unwrap();
        assert_eq!(e1.priority(), MsgPriority::Low);

        let e2 = ring.dequeue(&mut buf).unwrap();
        assert_eq!(e2.priority(), MsgPriority::High);

        let e3 = ring.dequeue(&mut buf).unwrap();
        assert_eq!(e3.priority(), MsgPriority::Urgent);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_ring_buffer_invalid_size() {
        let mut backing = vec![0u8; 1024];

        // Size not power of 2
        let result = RingBuffer::new(
            RegionHandle::null(),
            3,
            256,
            backing.as_mut_ptr(),
            backing.len(),
        );
        assert!(matches!(result, Err(KernelError::InvalidArgument)));

        // Size is 0
        let result = RingBuffer::new(
            RegionHandle::null(),
            0,
            256,
            backing.as_mut_ptr(),
            backing.len(),
        );
        assert!(matches!(result, Err(KernelError::InvalidArgument)));
    }
}
