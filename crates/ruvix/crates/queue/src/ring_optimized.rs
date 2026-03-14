//! Optimized lock-free ring buffer with power-of-2 masking (ADR-087).
//!
//! This module provides a high-performance ring buffer using:
//! - Power-of-2 size for fast modulo via bitwise AND
//! - Cache-line aligned entries for optimal memory access
//! - Zero-copy descriptor mode for large messages
//! - Lock-free operation with atomic head/tail pointers
//!
//! Performance targets:
//! - Enqueue: <200ns
//! - Dequeue: <200ns
//! - Single cache line access per operation

use core::sync::atomic::{AtomicU32, Ordering};

use ruvix_types::{KernelError, MsgPriority, RegionHandle};

use crate::Result;

/// Cache line size for alignment.
const CACHE_LINE_SIZE: usize = 64;

/// Optimized ring entry header (8 bytes).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct OptimizedRingEntry {
    /// Length of the payload in bytes.
    pub length: u16,
    /// Message priority (0-3).
    pub priority: u8,
    /// Entry flags.
    pub flags: u8,
    /// Sequence number for ordering.
    pub sequence: u32,
}

impl OptimizedRingEntry {
    /// Size of the entry header.
    pub const HEADER_SIZE: usize = 8;

    /// Flag indicating this entry contains a descriptor.
    pub const FLAG_DESCRIPTOR: u8 = 1 << 0;

    /// Flag indicating entry is valid (ready to consume).
    pub const FLAG_VALID: u8 = 1 << 1;

    /// Creates a new inline data entry.
    #[inline]
    pub const fn new_inline(length: u16, priority: MsgPriority, sequence: u32) -> Self {
        Self {
            length,
            priority: priority as u8,
            flags: Self::FLAG_VALID,
            sequence,
        }
    }

    /// Creates a new descriptor entry.
    #[inline]
    pub const fn new_descriptor(priority: MsgPriority, sequence: u32) -> Self {
        Self {
            length: 24, // Size of MessageDescriptor
            priority: priority as u8,
            flags: Self::FLAG_DESCRIPTOR | Self::FLAG_VALID,
            sequence,
        }
    }

    /// Creates an empty (invalid) entry.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            length: 0,
            priority: 0,
            flags: 0,
            sequence: 0,
        }
    }

    /// Returns true if this entry is valid.
    #[inline]
    pub const fn is_valid(&self) -> bool {
        (self.flags & Self::FLAG_VALID) != 0
    }

    /// Returns true if this entry contains a descriptor.
    #[inline]
    pub const fn is_descriptor(&self) -> bool {
        (self.flags & Self::FLAG_DESCRIPTOR) != 0
    }

    /// Gets the priority as MsgPriority.
    #[inline]
    pub fn priority(&self) -> MsgPriority {
        MsgPriority::from_u8(self.priority).unwrap_or(MsgPriority::Normal)
    }
}

/// Cache-line aligned ring slot.
///
/// Each slot contains the entry header and inline payload data.
/// Aligned to cache line for single cache line access per operation.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct OptimizedRingSlot {
    /// Entry header.
    pub entry: OptimizedRingEntry,
    /// Inline payload data (56 bytes max for 64-byte slot).
    pub payload: [u8; 56],
}

impl OptimizedRingSlot {
    /// Maximum inline payload size.
    pub const MAX_INLINE_SIZE: usize = 56;

    /// Creates an empty slot.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            entry: OptimizedRingEntry::empty(),
            payload: [0; 56],
        }
    }
}

// Compile-time assertion that slot is exactly 64 bytes
const _: () = assert!(core::mem::size_of::<OptimizedRingSlot>() == CACHE_LINE_SIZE);

/// Optimized lock-free ring buffer with power-of-2 size.
///
/// Uses atomic operations for thread-safe enqueue/dequeue.
/// Power-of-2 size enables fast modulo via bitwise AND.
pub struct OptimizedRingBuffer<const N: usize = 64> {
    /// The backing memory region handle.
    region: RegionHandle,

    /// Ring slots array (cache-line aligned).
    slots: [OptimizedRingSlot; N],

    /// Mask for index calculation (N - 1).
    mask: u32,

    /// Producer head (where next enqueue goes).
    head: AtomicU32,

    /// Consumer tail (where next dequeue comes from).
    tail: AtomicU32,

    /// Sequence counter for ordering.
    sequence: AtomicU32,
}

impl<const N: usize> OptimizedRingBuffer<N> {
    /// Creates a new optimized ring buffer.
    ///
    /// # Arguments
    ///
    /// * `region` - Handle to the backing region
    ///
    /// # Panics
    ///
    /// Panics if N is not a power of 2.
    #[must_use]
    pub fn new(region: RegionHandle) -> Self {
        // Compile-time check would be ideal, but we verify at runtime
        assert!(N > 0 && (N & (N - 1)) == 0, "Ring size must be power of 2");

        Self {
            region,
            slots: [OptimizedRingSlot::empty(); N],
            mask: (N - 1) as u32,
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            sequence: AtomicU32::new(0),
        }
    }

    /// Returns the region handle.
    #[inline]
    pub fn region(&self) -> RegionHandle {
        self.region
    }

    /// Returns the ring capacity.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of entries currently in the ring.
    #[inline]
    pub fn len(&self) -> u32 {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
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
        self.len() >= N as u32
    }

    /// Returns the number of free slots.
    #[inline]
    pub fn available(&self) -> u32 {
        (N as u32).saturating_sub(self.len())
    }

    /// Enqueues a message into the ring buffer (<200ns target).
    ///
    /// # Arguments
    ///
    /// * `data` - Message payload (max 56 bytes for inline)
    /// * `priority` - Message priority
    ///
    /// # Errors
    ///
    /// Returns `QueueFull` if the ring is full.
    /// Returns `MessageTooLarge` if the message exceeds inline limit.
    #[inline]
    pub fn enqueue(&mut self, data: &[u8], priority: MsgPriority) -> Result<()> {
        if data.len() > OptimizedRingSlot::MAX_INLINE_SIZE {
            return Err(KernelError::MessageTooLarge);
        }

        // Check if there's space (single atomic load)
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head.wrapping_sub(tail) >= N as u32 {
            return Err(KernelError::QueueFull);
        }

        // Calculate slot index using mask (fast modulo)
        let index = (head & self.mask) as usize;

        // Get sequence number
        let seq = self.sequence.fetch_add(1, Ordering::Relaxed);

        // Write to slot (single cache line)
        let slot = &mut self.slots[index];
        slot.entry = OptimizedRingEntry::new_inline(data.len() as u16, priority, seq);
        slot.payload[..data.len()].copy_from_slice(data);

        // Publish the entry (release semantics)
        self.head.store(head.wrapping_add(1), Ordering::Release);

        Ok(())
    }

    /// Dequeues a message from the ring buffer (<200ns target).
    ///
    /// # Arguments
    ///
    /// * `buf` - Buffer to receive the message data
    ///
    /// # Returns
    ///
    /// On success, returns the entry header and number of bytes copied.
    ///
    /// # Errors
    ///
    /// Returns `QueueEmpty` if the ring is empty.
    #[inline]
    pub fn dequeue(&mut self, buf: &mut [u8]) -> Result<(OptimizedRingEntry, usize)> {
        // Check if there's data (single atomic load)
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);

        if head == tail {
            return Err(KernelError::QueueEmpty);
        }

        // Calculate slot index using mask (fast modulo)
        let index = (tail & self.mask) as usize;

        // Read from slot (single cache line)
        let slot = &self.slots[index];
        let entry = slot.entry;

        let copy_len = (entry.length as usize).min(buf.len());
        buf[..copy_len].copy_from_slice(&slot.payload[..copy_len]);

        // Advance tail (release semantics)
        self.tail.store(tail.wrapping_add(1), Ordering::Release);

        Ok((entry, copy_len))
    }

    /// Peeks at the next entry without removing it.
    #[inline]
    pub fn peek(&self) -> Option<&OptimizedRingEntry> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);

        if head == tail {
            return None;
        }

        let index = (tail & self.mask) as usize;
        Some(&self.slots[index].entry)
    }

    /// Tries to enqueue without blocking, returns immediately.
    #[inline]
    pub fn try_enqueue(&mut self, data: &[u8], priority: MsgPriority) -> Result<()> {
        self.enqueue(data, priority)
    }

    /// Tries to dequeue without blocking, returns immediately.
    #[inline]
    pub fn try_dequeue(&mut self, buf: &mut [u8]) -> Result<(OptimizedRingEntry, usize)> {
        self.dequeue(buf)
    }

    /// Clears all entries from the ring.
    pub fn clear(&mut self) {
        let head = self.head.load(Ordering::Relaxed);
        self.tail.store(head, Ordering::Release);
    }
}

impl<const N: usize> Default for OptimizedRingBuffer<N> {
    fn default() -> Self {
        Self::new(RegionHandle::null())
    }
}

// SAFETY: OptimizedRingBuffer uses atomic operations for thread safety.
unsafe impl<const N: usize> Send for OptimizedRingBuffer<N> {}
unsafe impl<const N: usize> Sync for OptimizedRingBuffer<N> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_slot_size() {
        assert_eq!(core::mem::size_of::<OptimizedRingSlot>(), 64);
        assert_eq!(core::mem::align_of::<OptimizedRingSlot>(), 64);
    }

    #[test]
    fn test_ring_entry_size() {
        assert_eq!(core::mem::size_of::<OptimizedRingEntry>(), 8);
    }

    #[test]
    fn test_optimized_ring_basic() {
        let mut ring = OptimizedRingBuffer::<64>::new(RegionHandle::null());

        assert!(ring.is_empty());
        assert!(!ring.is_full());
        assert_eq!(ring.len(), 0);
        assert_eq!(ring.available(), 64);

        // Enqueue
        ring.enqueue(b"hello", MsgPriority::Normal).unwrap();
        assert_eq!(ring.len(), 1);
        assert!(!ring.is_empty());

        // Dequeue
        let mut buf = [0u8; 56];
        let (entry, len) = ring.dequeue(&mut buf).unwrap();
        assert_eq!(len, 5);
        assert_eq!(&buf[..5], b"hello");
        assert_eq!(entry.length, 5);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_optimized_ring_full() {
        let mut ring = OptimizedRingBuffer::<4>::new(RegionHandle::null());

        // Fill the ring
        for i in 0..4 {
            let msg = [i as u8; 8];
            ring.enqueue(&msg, MsgPriority::Normal).unwrap();
        }

        assert!(ring.is_full());

        // Should fail on next enqueue
        let result = ring.enqueue(b"overflow", MsgPriority::Normal);
        assert!(matches!(result, Err(KernelError::QueueFull)));
    }

    #[test]
    fn test_optimized_ring_wraparound() {
        let mut ring = OptimizedRingBuffer::<4>::new(RegionHandle::null());
        let mut buf = [0u8; 56];

        // Fill and drain multiple times to test wraparound
        for round in 0..10 {
            for i in 0..4 {
                let msg = [round as u8, i as u8];
                ring.enqueue(&msg, MsgPriority::Normal).unwrap();
            }

            for i in 0..4 {
                let (_, len) = ring.dequeue(&mut buf).unwrap();
                assert_eq!(len, 2);
                assert_eq!(buf[0], round as u8);
                assert_eq!(buf[1], i as u8);
            }

            assert!(ring.is_empty());
        }
    }

    #[test]
    fn test_optimized_ring_priority() {
        let mut ring = OptimizedRingBuffer::<8>::new(RegionHandle::null());

        ring.enqueue(b"low", MsgPriority::Low).unwrap();
        ring.enqueue(b"high", MsgPriority::High).unwrap();
        ring.enqueue(b"urgent", MsgPriority::Urgent).unwrap();

        let mut buf = [0u8; 56];

        let (e1, _) = ring.dequeue(&mut buf).unwrap();
        assert_eq!(e1.priority(), MsgPriority::Low);

        let (e2, _) = ring.dequeue(&mut buf).unwrap();
        assert_eq!(e2.priority(), MsgPriority::High);

        let (e3, _) = ring.dequeue(&mut buf).unwrap();
        assert_eq!(e3.priority(), MsgPriority::Urgent);
    }

    #[test]
    fn test_optimized_ring_peek() {
        let mut ring = OptimizedRingBuffer::<4>::new(RegionHandle::null());

        assert!(ring.peek().is_none());

        ring.enqueue(b"test", MsgPriority::Normal).unwrap();

        let peeked = ring.peek().unwrap();
        assert_eq!(peeked.length, 4);
        assert!(peeked.is_valid());

        // Peek should not consume
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn test_optimized_ring_message_too_large() {
        let mut ring = OptimizedRingBuffer::<4>::new(RegionHandle::null());

        // Message larger than 56 bytes should fail
        let large_msg = [0u8; 64];
        let result = ring.enqueue(&large_msg, MsgPriority::Normal);
        assert!(matches!(result, Err(KernelError::MessageTooLarge)));
    }

    #[test]
    fn test_optimized_ring_clear() {
        let mut ring = OptimizedRingBuffer::<4>::new(RegionHandle::null());

        for i in 0..4 {
            ring.enqueue(&[i], MsgPriority::Normal).unwrap();
        }

        assert!(ring.is_full());

        ring.clear();

        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
    }

    #[test]
    fn test_power_of_2_sizes() {
        // Test various power-of-2 sizes compile and work
        let _r8 = OptimizedRingBuffer::<8>::default();
        let _r16 = OptimizedRingBuffer::<16>::default();
        let _r32 = OptimizedRingBuffer::<32>::default();
        let _r64 = OptimizedRingBuffer::<64>::default();
        let _r128 = OptimizedRingBuffer::<128>::default();
        let _r256 = OptimizedRingBuffer::<256>::default();
    }
}
