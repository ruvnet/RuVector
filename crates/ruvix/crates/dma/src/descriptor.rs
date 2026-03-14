//! DMA descriptor structures for scatter-gather transfers.

use crate::{DmaError, DmaResult, MAX_DESCRIPTOR_CHAIN_LENGTH};

/// Flags for DMA descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DmaDescriptorFlags {
    /// This descriptor is the last in the chain.
    pub end_of_chain: bool,
    /// Generate interrupt when this descriptor completes.
    pub interrupt_on_complete: bool,
    /// Pause transfer after this descriptor.
    pub pause_after: bool,
    /// Link to another descriptor chain after this one.
    pub chain_link: bool,
    /// Descriptor has been processed.
    pub completed: bool,
    /// Descriptor encountered an error.
    pub error: bool,
}

impl DmaDescriptorFlags {
    /// Create flags for a middle-of-chain descriptor.
    #[must_use]
    pub const fn middle() -> Self {
        Self {
            end_of_chain: false,
            interrupt_on_complete: false,
            pause_after: false,
            chain_link: false,
            completed: false,
            error: false,
        }
    }

    /// Create flags for the last descriptor in a chain.
    #[must_use]
    pub const fn last() -> Self {
        Self {
            end_of_chain: true,
            interrupt_on_complete: true,
            pause_after: false,
            chain_link: false,
            completed: false,
            error: false,
        }
    }

    /// Create flags with interrupt enabled.
    #[must_use]
    pub const fn with_interrupt(mut self) -> Self {
        self.interrupt_on_complete = true;
        self
    }

    /// Pack flags into a u8.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        let mut flags = 0u8;
        if self.end_of_chain {
            flags |= 1 << 0;
        }
        if self.interrupt_on_complete {
            flags |= 1 << 1;
        }
        if self.pause_after {
            flags |= 1 << 2;
        }
        if self.chain_link {
            flags |= 1 << 3;
        }
        if self.completed {
            flags |= 1 << 4;
        }
        if self.error {
            flags |= 1 << 5;
        }
        flags
    }

    /// Unpack flags from a u8.
    #[must_use]
    pub const fn from_u8(flags: u8) -> Self {
        Self {
            end_of_chain: flags & (1 << 0) != 0,
            interrupt_on_complete: flags & (1 << 1) != 0,
            pause_after: flags & (1 << 2) != 0,
            chain_link: flags & (1 << 3) != 0,
            completed: flags & (1 << 4) != 0,
            error: flags & (1 << 5) != 0,
        }
    }
}

/// A DMA descriptor for scatter-gather transfers.
///
/// Descriptors are linked together to form a chain, allowing
/// the DMA controller to process multiple non-contiguous
/// memory regions in a single operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct DmaDescriptor {
    /// Source address for this transfer segment.
    pub src_addr: u64,
    /// Destination address for this transfer segment.
    pub dst_addr: u64,
    /// Length of this transfer segment in bytes.
    pub length: u32,
    /// Reserved for alignment.
    reserved: u16,
    /// Flags controlling descriptor behavior.
    pub flags: DmaDescriptorFlags,
    /// Reserved byte for future use.
    reserved2: u8,
    /// Physical address of the next descriptor (0 if none).
    pub next: u64,
}

impl DmaDescriptor {
    /// Size of a descriptor in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Create a new DMA descriptor.
    #[must_use]
    pub const fn new(src_addr: u64, dst_addr: u64, length: u32) -> Self {
        Self {
            src_addr,
            dst_addr,
            length,
            reserved: 0,
            flags: DmaDescriptorFlags::last(),
            reserved2: 0,
            next: 0,
        }
    }

    /// Create a descriptor with custom flags.
    #[must_use]
    pub const fn with_flags(
        src_addr: u64,
        dst_addr: u64,
        length: u32,
        flags: DmaDescriptorFlags,
    ) -> Self {
        Self {
            src_addr,
            dst_addr,
            length,
            reserved: 0,
            flags,
            reserved2: 0,
            next: 0,
        }
    }

    /// Set the next descriptor pointer.
    #[must_use]
    pub const fn with_next(mut self, next_addr: u64) -> Self {
        self.next = next_addr;
        self.flags.end_of_chain = false;
        self
    }

    /// Check if this is the last descriptor in the chain.
    #[must_use]
    pub const fn is_last(&self) -> bool {
        self.flags.end_of_chain || self.next == 0
    }

    /// Check if the descriptor has been completed.
    #[must_use]
    pub const fn is_completed(&self) -> bool {
        self.flags.completed
    }

    /// Check if the descriptor has an error.
    #[must_use]
    pub const fn has_error(&self) -> bool {
        self.flags.error
    }

    /// Mark the descriptor as completed.
    pub fn mark_completed(&mut self) {
        self.flags.completed = true;
    }

    /// Mark the descriptor as having an error.
    pub fn mark_error(&mut self) {
        self.flags.error = true;
    }

    /// Reset the descriptor status flags.
    pub fn reset_status(&mut self) {
        self.flags.completed = false;
        self.flags.error = false;
    }

    /// Validate the descriptor.
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        // Length must be non-zero
        if self.length == 0 {
            return false;
        }

        // Addresses should be 4-byte aligned at minimum
        if self.src_addr % 4 != 0 || self.dst_addr % 4 != 0 {
            return false;
        }

        true
    }
}

impl Default for DmaDescriptor {
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

/// A chain of DMA descriptors stored in a fixed-size array.
///
/// This structure manages a linked list of descriptors for
/// scatter-gather DMA operations.
#[derive(Debug)]
pub struct DmaDescriptorChain {
    /// Array of descriptors.
    descriptors: [DmaDescriptor; MAX_DESCRIPTOR_CHAIN_LENGTH],
    /// Number of descriptors in use.
    count: usize,
    /// Physical base address of the descriptor array.
    base_addr: u64,
}

impl DmaDescriptorChain {
    /// Create a new empty descriptor chain.
    ///
    /// # Arguments
    ///
    /// * `base_addr` - Physical address where the descriptor array is mapped.
    #[must_use]
    pub const fn new(base_addr: u64) -> Self {
        Self {
            descriptors: [DmaDescriptor::new(0, 0, 0); MAX_DESCRIPTOR_CHAIN_LENGTH],
            count: 0,
            base_addr,
        }
    }

    /// Get the number of descriptors in the chain.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    /// Check if the chain is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Check if the chain is full.
    #[must_use]
    pub const fn is_full(&self) -> bool {
        self.count >= MAX_DESCRIPTOR_CHAIN_LENGTH
    }

    /// Get the physical address of the first descriptor.
    #[must_use]
    pub const fn head_addr(&self) -> u64 {
        self.base_addr
    }

    /// Get the physical address of a descriptor by index.
    #[must_use]
    pub const fn descriptor_addr(&self, index: usize) -> u64 {
        self.base_addr + (index * DmaDescriptor::SIZE) as u64
    }

    /// Add a descriptor to the chain.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the chain is full.
    pub fn push(&mut self, mut descriptor: DmaDescriptor) -> DmaResult<()> {
        if self.is_full() {
            return Err(DmaError::descriptor_error());
        }

        // Update the previous descriptor to point to this one
        if self.count > 0 {
            let prev_idx = self.count - 1;
            self.descriptors[prev_idx].next = self.descriptor_addr(self.count);
            self.descriptors[prev_idx].flags.end_of_chain = false;
        }

        // Mark this descriptor as the last
        descriptor.next = 0;
        descriptor.flags.end_of_chain = true;

        self.descriptors[self.count] = descriptor;
        self.count += 1;

        Ok(())
    }

    /// Add a transfer segment to the chain.
    ///
    /// # Errors
    ///
    /// Returns `DmaError` if the chain is full.
    pub fn add_segment(&mut self, src: u64, dst: u64, length: u32) -> DmaResult<()> {
        self.push(DmaDescriptor::new(src, dst, length))
    }

    /// Get a reference to a descriptor by index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&DmaDescriptor> {
        if index < self.count {
            Some(&self.descriptors[index])
        } else {
            None
        }
    }

    /// Get a mutable reference to a descriptor by index.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DmaDescriptor> {
        if index < self.count {
            Some(&mut self.descriptors[index])
        } else {
            None
        }
    }

    /// Clear the chain.
    pub fn clear(&mut self) {
        for desc in &mut self.descriptors[..self.count] {
            *desc = DmaDescriptor::default();
        }
        self.count = 0;
    }

    /// Reset all status flags in the chain.
    pub fn reset_status(&mut self) {
        for desc in &mut self.descriptors[..self.count] {
            desc.reset_status();
        }
    }

    /// Calculate total transfer length.
    #[must_use]
    pub fn total_length(&self) -> u64 {
        self.descriptors[..self.count]
            .iter()
            .map(|d| u64::from(d.length))
            .sum()
    }

    /// Check if all descriptors have completed.
    #[must_use]
    pub fn is_completed(&self) -> bool {
        self.count > 0 && self.descriptors[..self.count].iter().all(|d| d.is_completed())
    }

    /// Check if any descriptor has an error.
    #[must_use]
    pub fn has_error(&self) -> bool {
        self.descriptors[..self.count].iter().any(|d| d.has_error())
    }

    /// Validate the entire chain.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        if self.is_empty() {
            return false;
        }

        // Check all descriptors are valid
        for desc in &self.descriptors[..self.count] {
            if !desc.is_valid() {
                return false;
            }
        }

        // Check the last descriptor is marked as end
        if let Some(last) = self.descriptors.get(self.count.saturating_sub(1)) {
            if !last.is_last() {
                return false;
            }
        }

        true
    }

    /// Iterate over descriptors.
    pub fn iter(&self) -> impl Iterator<Item = &DmaDescriptor> {
        self.descriptors[..self.count].iter()
    }

    /// Iterate over descriptors mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut DmaDescriptor> {
        self.descriptors[..self.count].iter_mut()
    }
}

impl Default for DmaDescriptorChain {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor_flags() {
        let middle = DmaDescriptorFlags::middle();
        assert!(!middle.end_of_chain);
        assert!(!middle.interrupt_on_complete);

        let last = DmaDescriptorFlags::last();
        assert!(last.end_of_chain);
        assert!(last.interrupt_on_complete);

        let with_int = middle.with_interrupt();
        assert!(with_int.interrupt_on_complete);
    }

    #[test]
    fn test_descriptor_flags_pack_unpack() {
        let flags = DmaDescriptorFlags {
            end_of_chain: true,
            interrupt_on_complete: true,
            pause_after: false,
            chain_link: true,
            completed: false,
            error: true,
        };

        let packed = flags.to_u8();
        let unpacked = DmaDescriptorFlags::from_u8(packed);

        assert_eq!(flags, unpacked);
    }

    #[test]
    fn test_descriptor_creation() {
        let desc = DmaDescriptor::new(0x1000, 0x2000, 4096);

        assert_eq!(desc.src_addr, 0x1000);
        assert_eq!(desc.dst_addr, 0x2000);
        assert_eq!(desc.length, 4096);
        assert!(desc.is_last());
        assert!(!desc.is_completed());
        assert!(!desc.has_error());
    }

    #[test]
    fn test_descriptor_with_next() {
        let desc = DmaDescriptor::new(0x1000, 0x2000, 4096).with_next(0x3000);

        assert_eq!(desc.next, 0x3000);
        assert!(!desc.is_last());
    }

    #[test]
    fn test_descriptor_status() {
        let mut desc = DmaDescriptor::new(0x1000, 0x2000, 4096);

        desc.mark_completed();
        assert!(desc.is_completed());

        desc.mark_error();
        assert!(desc.has_error());

        desc.reset_status();
        assert!(!desc.is_completed());
        assert!(!desc.has_error());
    }

    #[test]
    fn test_descriptor_validation() {
        let valid = DmaDescriptor::new(0x1000, 0x2000, 4096);
        assert!(valid.is_valid());

        let zero_len = DmaDescriptor::new(0x1000, 0x2000, 0);
        assert!(!zero_len.is_valid());

        let unaligned = DmaDescriptor::new(0x1001, 0x2000, 4096);
        assert!(!unaligned.is_valid());
    }

    #[test]
    fn test_chain_basic_operations() {
        let mut chain = DmaDescriptorChain::new(0x1_0000);

        assert!(chain.is_empty());
        assert!(!chain.is_full());

        chain.add_segment(0x1000, 0x2000, 4096).unwrap();
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());

        chain.add_segment(0x2000, 0x3000, 2048).unwrap();
        assert_eq!(chain.len(), 2);

        assert_eq!(chain.total_length(), 6144);
    }

    #[test]
    fn test_chain_linking() {
        let mut chain = DmaDescriptorChain::new(0x1_0000);

        chain.add_segment(0x1000, 0x2000, 4096).unwrap();
        chain.add_segment(0x2000, 0x3000, 2048).unwrap();

        // First descriptor should point to second
        let first = chain.get(0).unwrap();
        assert!(!first.is_last());
        assert_eq!(first.next, chain.descriptor_addr(1));

        // Second descriptor should be last
        let second = chain.get(1).unwrap();
        assert!(second.is_last());
    }

    #[test]
    fn test_chain_clear() {
        let mut chain = DmaDescriptorChain::new(0x1_0000);

        chain.add_segment(0x1000, 0x2000, 4096).unwrap();
        chain.add_segment(0x2000, 0x3000, 2048).unwrap();

        chain.clear();
        assert!(chain.is_empty());
    }

    #[test]
    fn test_chain_validation() {
        let mut chain = DmaDescriptorChain::new(0x1_0000);

        // Empty chain is invalid
        assert!(!chain.is_valid());

        chain.add_segment(0x1000, 0x2000, 4096).unwrap();
        assert!(chain.is_valid());
    }

    #[test]
    fn test_chain_completion() {
        let mut chain = DmaDescriptorChain::new(0x1_0000);

        chain.add_segment(0x1000, 0x2000, 4096).unwrap();
        chain.add_segment(0x2000, 0x3000, 2048).unwrap();

        assert!(!chain.is_completed());

        chain.get_mut(0).unwrap().mark_completed();
        assert!(!chain.is_completed());

        chain.get_mut(1).unwrap().mark_completed();
        assert!(chain.is_completed());
    }

    #[test]
    fn test_chain_error_detection() {
        let mut chain = DmaDescriptorChain::new(0x1_0000);

        chain.add_segment(0x1000, 0x2000, 4096).unwrap();
        chain.add_segment(0x2000, 0x3000, 2048).unwrap();

        assert!(!chain.has_error());

        chain.get_mut(1).unwrap().mark_error();
        assert!(chain.has_error());
    }
}
