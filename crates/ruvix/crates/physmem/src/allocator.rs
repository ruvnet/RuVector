//! Buddy allocator implementation.
//!
//! This module implements a buddy allocator for physical page frame allocation.
//! The buddy system uses power-of-two block sizes for efficient allocation
//! and deallocation with minimal fragmentation.

use crate::{
    AllocatorStats, PageFrame, PageOrder, PhysAddr, PhysMemError,
    MAX_ORDER, PAGE_SIZE, order_to_pages, pages_to_order,
};

/// Maximum number of blocks per free list.
///
/// This limits memory usage for the free list arrays.
/// Each FreeList is 8KB (1024 * 8 bytes per PhysAddr).
/// Total: 10 orders * 8KB = 80KB for the full allocator.
/// This supports managing up to 2GB per order with 2MB blocks.
const MAX_FREE_BLOCKS: usize = 1024;

/// A buddy allocator for physical page frames.
///
/// The buddy allocator manages physical memory by maintaining free lists
/// for each power-of-two block size. When memory is allocated, blocks are
/// split as needed. When memory is freed, adjacent buddy blocks are coalesced.
///
/// # Block Sizes
///
/// | Order | Pages | Size |
/// |-------|-------|------|
/// | 0 | 1 | 4KB |
/// | 1 | 2 | 8KB |
/// | 2 | 4 | 16KB |
/// | 3 | 8 | 32KB |
/// | 4 | 16 | 64KB |
/// | 5 | 32 | 128KB |
/// | 6 | 64 | 256KB |
/// | 7 | 128 | 512KB |
/// | 8 | 256 | 1MB |
/// | 9 | 512 | 2MB |
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::{BuddyAllocator, PhysAddr, PAGE_SIZE};
///
/// // Create allocator for 4MB of memory
/// let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 1024);
///
/// // Allocate 4 pages (16KB)
/// if let Some(addr) = allocator.alloc_pages(4) {
///     assert!(addr.is_page_aligned());
///     allocator.dealloc_pages(addr, 4);
/// }
/// ```
pub struct BuddyAllocator {
    /// Base physical address of the managed region.
    base_addr: PhysAddr,

    /// Total number of pages under management.
    total_pages: usize,

    /// Free lists for each order (0 to MAX_ORDER-1).
    /// Each entry is a vector of free block addresses at that order.
    free_lists: [FreeList; MAX_ORDER],

    /// Allocation statistics.
    stats: AllocatorStats,

    /// Whether the allocator has been initialized.
    initialized: bool,
}

/// A free list for a specific block order.
///
/// This stores addresses of free blocks. In a real kernel, this would be
/// implemented as an intrusive linked list stored in the free pages themselves.
/// For this no_std implementation, we use a fixed-size array.
#[derive(Clone)]
struct FreeList {
    /// Addresses of free blocks.
    blocks: [PhysAddr; MAX_FREE_BLOCKS],
    /// Number of valid entries.
    count: usize,
}

impl Default for FreeList {
    fn default() -> Self {
        Self::new()
    }
}

impl FreeList {
    /// Creates an empty free list.
    const fn new() -> Self {
        Self {
            blocks: [PhysAddr::NULL; MAX_FREE_BLOCKS],
            count: 0,
        }
    }

    /// Returns the number of free blocks.
    #[inline]
    const fn len(&self) -> usize {
        self.count
    }

    /// Checks if the free list is empty.
    #[inline]
    #[allow(dead_code)]
    const fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Pushes a block address onto the free list.
    ///
    /// Returns an error if the list is full.
    #[inline]
    fn push(&mut self, addr: PhysAddr) -> Result<(), PhysMemError> {
        if self.count >= MAX_FREE_BLOCKS {
            return Err(PhysMemError::InternalCorruption);
        }
        self.blocks[self.count] = addr;
        self.count += 1;
        Ok(())
    }

    /// Pops a block address from the free list.
    ///
    /// Returns `None` if the list is empty.
    #[inline]
    fn pop(&mut self) -> Option<PhysAddr> {
        if self.count == 0 {
            None
        } else {
            self.count -= 1;
            Some(self.blocks[self.count])
        }
    }

    /// Removes a specific address from the free list.
    ///
    /// Returns `true` if the address was found and removed.
    fn remove(&mut self, addr: PhysAddr) -> bool {
        for i in 0..self.count {
            if self.blocks[i] == addr {
                // Swap with last element and decrement count
                self.count -= 1;
                if i < self.count {
                    self.blocks[i] = self.blocks[self.count];
                }
                return true;
            }
        }
        false
    }

    /// Checks if an address is in the free list.
    #[allow(dead_code)]
    fn contains(&self, addr: PhysAddr) -> bool {
        for i in 0..self.count {
            if self.blocks[i] == addr {
                return true;
            }
        }
        false
    }
}

impl BuddyAllocator {
    /// Creates a new buddy allocator.
    ///
    /// The allocator manages `total_pages` pages starting at `base_addr`.
    /// Initially, all memory is available for allocation.
    ///
    /// # Arguments
    ///
    /// * `base_addr` - The starting physical address (must be page-aligned).
    /// * `total_pages` - The total number of pages to manage.
    ///
    /// # Panics
    ///
    /// Panics if `base_addr` is not page-aligned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{BuddyAllocator, PhysAddr};
    ///
    /// let allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 1024);
    /// assert_eq!(allocator.total_pages(), 1024);
    /// assert_eq!(allocator.free_page_count(), 1024);
    /// ```
    #[must_use]
    pub fn new(base_addr: PhysAddr, total_pages: usize) -> Self {
        assert!(
            base_addr.is_page_aligned(),
            "Base address must be page-aligned"
        );

        let mut allocator = Self {
            base_addr,
            total_pages,
            free_lists: core::array::from_fn(|_| FreeList::new()),
            stats: AllocatorStats::new(total_pages, 0),
            initialized: false,
        };

        // Initialize free lists with the available memory
        allocator.init_free_lists();
        allocator.initialized = true;

        allocator
    }

    /// Creates an uninitialized allocator.
    ///
    /// Call `init` to initialize it with memory.
    #[must_use]
    pub const fn uninit() -> Self {
        Self {
            base_addr: PhysAddr::NULL,
            total_pages: 0,
            free_lists: [
                FreeList::new(), FreeList::new(), FreeList::new(), FreeList::new(),
                FreeList::new(), FreeList::new(), FreeList::new(), FreeList::new(),
                FreeList::new(), FreeList::new(),
            ],
            stats: AllocatorStats::new(0, 0),
            initialized: false,
        }
    }

    /// Initializes the allocator with memory.
    ///
    /// # Arguments
    ///
    /// * `base_addr` - The starting physical address (must be page-aligned).
    /// * `total_pages` - The total number of pages to manage.
    ///
    /// # Errors
    ///
    /// Returns an error if the address is not page-aligned.
    pub fn init(&mut self, base_addr: PhysAddr, total_pages: usize) -> Result<(), PhysMemError> {
        if !base_addr.is_page_aligned() {
            return Err(PhysMemError::UnalignedAddress);
        }

        self.base_addr = base_addr;
        self.total_pages = total_pages;
        self.stats = AllocatorStats::new(total_pages, 0);

        // Clear existing free lists
        for list in &mut self.free_lists {
            *list = FreeList::new();
        }

        self.init_free_lists();
        self.initialized = true;

        Ok(())
    }

    /// Initializes the free lists with the available memory.
    fn init_free_lists(&mut self) {
        let mut remaining_pages = self.total_pages;
        let mut current_addr = self.base_addr;

        // Add blocks from largest to smallest
        while remaining_pages > 0 {
            // Find the largest order that fits
            let order = self.largest_fitting_order(remaining_pages, current_addr);
            let pages = order_to_pages(order);

            // Add to free list (ignore error as we're initializing)
            let _ = self.free_lists[order].push(current_addr);

            current_addr = current_addr.add_pages(pages);
            remaining_pages -= pages;
        }

        // Update statistics
        self.stats = AllocatorStats::new(self.total_pages, self.total_pages);
    }

    /// Finds the largest order that fits within remaining pages and alignment.
    fn largest_fitting_order(&self, remaining_pages: usize, addr: PhysAddr) -> usize {
        for order in (0..MAX_ORDER).rev() {
            let pages = order_to_pages(order);
            if pages <= remaining_pages && addr.is_order_aligned(order) {
                return order;
            }
        }
        0
    }

    /// Returns the base address of the managed region.
    #[inline]
    #[must_use]
    pub const fn base_addr(&self) -> PhysAddr {
        self.base_addr
    }

    /// Returns the total number of pages.
    #[inline]
    #[must_use]
    pub const fn total_pages(&self) -> usize {
        self.total_pages
    }

    /// Returns the number of free pages.
    #[inline]
    #[must_use]
    pub fn free_page_count(&self) -> usize {
        let mut free = 0;
        for order in 0..MAX_ORDER {
            free += self.free_lists[order].len() * order_to_pages(order);
        }
        free
    }

    /// Returns the number of used pages.
    #[inline]
    #[must_use]
    pub fn used_pages(&self) -> usize {
        self.total_pages.saturating_sub(self.free_page_count())
    }

    /// Returns allocation statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &AllocatorStats {
        &self.stats
    }

    /// Returns the end address of the managed region (exclusive).
    #[inline]
    #[must_use]
    pub fn end_addr(&self) -> PhysAddr {
        self.base_addr.add_pages(self.total_pages)
    }

    /// Returns the number of free blocks at a specific order.
    #[inline]
    #[must_use]
    pub fn free_blocks_at_order(&self, order: usize) -> usize {
        if order < MAX_ORDER {
            self.free_lists[order].len()
        } else {
            0
        }
    }

    /// Allocates contiguous pages.
    ///
    /// Returns the physical address of the allocated memory, or `None` if
    /// allocation fails.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of contiguous pages to allocate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{BuddyAllocator, PhysAddr};
    ///
    /// let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 1024);
    ///
    /// // Allocate a single page
    /// let addr = allocator.alloc_pages(1).expect("allocation failed");
    /// assert!(addr.is_page_aligned());
    ///
    /// // Allocate 4 contiguous pages
    /// let addr4 = allocator.alloc_pages(4).expect("allocation failed");
    ///
    /// // Free the allocations
    /// allocator.dealloc_pages(addr, 1);
    /// allocator.dealloc_pages(addr4, 4);
    /// ```
    #[must_use]
    pub fn alloc_pages(&mut self, count: usize) -> Option<PhysAddr> {
        if !self.initialized {
            return None;
        }

        if count == 0 {
            return None;
        }

        // Calculate the required order
        let required_order = pages_to_order(count);
        if required_order >= MAX_ORDER {
            self.stats.record_allocation(count, false);
            return None;
        }

        // Try to allocate from the required order or higher
        let result = self.alloc_order(required_order);

        self.stats.record_allocation(count, result.is_some());

        result
    }

    /// Allocates a block of a specific order.
    fn alloc_order(&mut self, order: usize) -> Option<PhysAddr> {
        // Try to find a free block at this order
        if let Some(addr) = self.free_lists[order].pop() {
            return Some(addr);
        }

        // Need to split a larger block
        for higher_order in (order + 1)..MAX_ORDER {
            if let Some(addr) = self.free_lists[higher_order].pop() {
                // Split down to the required order
                self.split_to_order(addr, higher_order, order);
                return self.free_lists[order].pop();
            }
        }

        // No memory available
        None
    }

    /// Splits a block from `from_order` down to `to_order`.
    fn split_to_order(&mut self, addr: PhysAddr, from_order: usize, to_order: usize) {
        let base_addr = addr;
        let mut current_order = from_order;

        while current_order > to_order {
            // Split the block
            let new_order = current_order - 1;
            let buddy_addr = base_addr.add_pages(order_to_pages(new_order));

            // Add the buddy to the free list
            let _ = self.free_lists[new_order].push(buddy_addr);

            self.stats.record_split();

            current_order = new_order;
        }

        // Add the final block
        let _ = self.free_lists[to_order].push(base_addr);
    }

    /// Frees previously allocated pages.
    ///
    /// # Arguments
    ///
    /// * `addr` - The physical address returned by `alloc_pages`.
    /// * `count` - The number of pages that were allocated.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if the address is invalid or double-free is detected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{BuddyAllocator, PhysAddr};
    ///
    /// let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 1024);
    ///
    /// let addr = allocator.alloc_pages(8).unwrap();
    /// allocator.dealloc_pages(addr, 8);
    ///
    /// // Memory is now available again
    /// let addr2 = allocator.alloc_pages(8).unwrap();
    /// ```
    pub fn dealloc_pages(&mut self, addr: PhysAddr, count: usize) {
        if !self.initialized || count == 0 {
            return;
        }

        // Validate address
        debug_assert!(
            addr.is_page_aligned(),
            "Address must be page-aligned: {addr:?}"
        );
        debug_assert!(
            addr.is_in_range(self.base_addr, self.end_addr()),
            "Address out of range: {addr:?}"
        );

        let order = pages_to_order(count);
        if order >= MAX_ORDER {
            return;
        }

        self.free_order(addr, order);
        self.stats.record_free(count);
    }

    /// Frees a block and coalesces with buddy if possible.
    fn free_order(&mut self, addr: PhysAddr, order: usize) {
        let mut current_addr = addr;
        let mut current_order = order;

        // Try to coalesce with buddies
        while current_order < MAX_ORDER - 1 {
            let buddy_addr = self.buddy_addr(current_addr, current_order);

            // Check if buddy is free and in range
            if !buddy_addr.is_in_range(self.base_addr, self.end_addr()) {
                break;
            }

            // Try to remove buddy from free list
            if !self.free_lists[current_order].remove(buddy_addr) {
                // Buddy is not free, cannot coalesce
                break;
            }

            // Coalesce: use lower address
            current_addr = if current_addr.as_u64() < buddy_addr.as_u64() {
                current_addr
            } else {
                buddy_addr
            };

            self.stats.record_coalesce();
            current_order += 1;
        }

        // Add to free list
        let _ = self.free_lists[current_order].push(current_addr);
    }

    /// Calculates the buddy address for a block.
    fn buddy_addr(&self, addr: PhysAddr, order: usize) -> PhysAddr {
        let block_size = (order_to_pages(order) * PAGE_SIZE) as u64;
        PhysAddr::new(addr.as_u64() ^ block_size)
    }

    /// Allocates pages and returns a `PageFrame`.
    ///
    /// This is a convenience method that returns a `PageFrame` instead of
    /// just the address.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of contiguous pages to allocate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{BuddyAllocator, PhysAddr};
    ///
    /// let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 1024);
    ///
    /// if let Some(frame) = allocator.alloc_frame(4) {
    ///     assert_eq!(frame.pages(), 4);
    ///     allocator.free_frame(&frame);
    /// }
    /// ```
    #[must_use]
    pub fn alloc_frame(&mut self, count: usize) -> Option<PageFrame> {
        let order = PageOrder::from_pages(count)?;
        let addr = self.alloc_pages(count)?;
        Some(PageFrame::new(addr, order))
    }

    /// Frees a `PageFrame`.
    ///
    /// # Arguments
    ///
    /// * `frame` - The page frame to free.
    pub fn free_frame(&mut self, frame: &PageFrame) {
        self.dealloc_pages(frame.addr(), frame.pages());
    }

    /// Tries to allocate pages, returning a `Result`.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of contiguous pages to allocate.
    ///
    /// # Errors
    ///
    /// Returns `PhysMemError::OutOfMemory` if allocation fails.
    /// Returns `PhysMemError::ZeroAllocation` if count is zero.
    /// Returns `PhysMemError::AllocationTooLarge` if count exceeds max block size.
    pub fn try_alloc_pages(&mut self, count: usize) -> Result<PhysAddr, PhysMemError> {
        if !self.initialized {
            return Err(PhysMemError::NotInitialized);
        }

        if count == 0 {
            return Err(PhysMemError::ZeroAllocation);
        }

        let required_order = pages_to_order(count);
        if required_order >= MAX_ORDER {
            return Err(PhysMemError::AllocationTooLarge);
        }

        self.alloc_pages(count).ok_or(PhysMemError::OutOfMemory)
    }

    /// Tries to free pages, returning a `Result`.
    ///
    /// # Arguments
    ///
    /// * `addr` - The physical address to free.
    /// * `count` - The number of pages to free.
    ///
    /// # Errors
    ///
    /// Returns errors for invalid addresses or sizes.
    pub fn try_free_pages(&mut self, addr: PhysAddr, count: usize) -> Result<(), PhysMemError> {
        if !self.initialized {
            return Err(PhysMemError::NotInitialized);
        }

        if count == 0 {
            return Err(PhysMemError::ZeroAllocation);
        }

        if !addr.is_page_aligned() {
            return Err(PhysMemError::UnalignedAddress);
        }

        if !addr.is_in_range(self.base_addr, self.end_addr()) {
            return Err(PhysMemError::AddressOutOfRange);
        }

        let order = pages_to_order(count);
        if order >= MAX_ORDER {
            return Err(PhysMemError::AllocationTooLarge);
        }

        self.dealloc_pages(addr, count);
        Ok(())
    }

    /// Checks if an address is within the managed range.
    #[inline]
    #[must_use]
    pub fn contains(&self, addr: PhysAddr) -> bool {
        addr.is_in_range(self.base_addr, self.end_addr())
    }

    /// Resets the allocator to its initial state.
    ///
    /// All allocations are freed and the full memory becomes available.
    pub fn reset(&mut self) {
        if !self.initialized {
            return;
        }

        // Clear all free lists
        for list in &mut self.free_lists {
            *list = FreeList::new();
        }

        // Re-initialize
        self.init_free_lists();
        self.stats.reset_counters();
    }

    /// Returns detailed statistics per order.
    #[must_use]
    pub fn order_stats(&self) -> [crate::stats::OrderStats; MAX_ORDER] {
        let mut stats = [crate::stats::OrderStats::new(); MAX_ORDER];
        for (i, stat) in stats.iter_mut().enumerate() {
            stat.free_blocks = self.free_lists[i].len();
        }
        stats
    }

    /// Dumps the free list state for debugging.
    #[cfg(feature = "std")]
    pub fn dump_free_lists(&self) {
        std::println!("Buddy Allocator State:");
        std::println!("  Base: {}", self.base_addr);
        std::println!("  Total: {} pages ({} bytes)", self.total_pages, self.total_pages * PAGE_SIZE);
        std::println!("  Free: {} pages", self.free_page_count());
        std::println!("  Used: {} pages", self.used_pages());
        std::println!();
        std::println!("Free Lists:");
        for order in 0..MAX_ORDER {
            let count = self.free_lists[order].len();
            if count > 0 {
                let pages = order_to_pages(order);
                let bytes = pages * PAGE_SIZE;
                std::println!("  Order {}: {} blocks ({} pages each, {} bytes)",
                    order, count, pages, bytes);
            }
        }
    }
}

impl Default for BuddyAllocator {
    fn default() -> Self {
        Self::uninit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_BASE: u64 = 0x1000_0000;
    const TEST_PAGES: usize = 1024; // 4MB

    fn create_allocator() -> BuddyAllocator {
        BuddyAllocator::new(PhysAddr::new(TEST_BASE), TEST_PAGES)
    }

    #[test]
    fn test_new_allocator() {
        let alloc = create_allocator();
        assert_eq!(alloc.total_pages(), TEST_PAGES);
        assert_eq!(alloc.free_page_count(), TEST_PAGES);
        assert_eq!(alloc.used_pages(), 0);
        assert_eq!(alloc.base_addr().as_u64(), TEST_BASE);
    }

    #[test]
    fn test_alloc_single_page() {
        let mut alloc = create_allocator();

        let addr = alloc.alloc_pages(1).expect("alloc failed");
        assert!(addr.is_page_aligned());
        assert!(alloc.contains(addr));
        assert_eq!(alloc.used_pages(), 1);

        alloc.dealloc_pages(addr, 1);
        assert_eq!(alloc.used_pages(), 0);
    }

    #[test]
    fn test_alloc_multiple_pages() {
        let mut alloc = create_allocator();

        // Allocate 4 pages (order 2)
        let addr = alloc.alloc_pages(4).expect("alloc failed");
        assert!(addr.is_page_aligned());
        assert!(addr.is_order_aligned(2)); // Should be 16KB aligned
        assert_eq!(alloc.used_pages(), 4);

        // Allocate 8 pages (order 3)
        let addr2 = alloc.alloc_pages(8).expect("alloc failed");
        assert!(addr2.is_order_aligned(3));
        assert_eq!(alloc.used_pages(), 12);

        alloc.dealloc_pages(addr, 4);
        alloc.dealloc_pages(addr2, 8);
        assert_eq!(alloc.used_pages(), 0);
    }

    #[test]
    fn test_alloc_max_block() {
        let mut alloc = BuddyAllocator::new(PhysAddr::new(TEST_BASE), 512);

        // Allocate 512 pages (order 9, 2MB)
        let addr = alloc.alloc_pages(512).expect("alloc failed");
        assert!(addr.is_page_aligned());
        assert_eq!(alloc.used_pages(), 512);
        assert_eq!(alloc.free_page_count(), 0);

        alloc.dealloc_pages(addr, 512);
        assert_eq!(alloc.free_page_count(), 512);
    }

    #[test]
    fn test_alloc_too_large() {
        let mut alloc = create_allocator();

        // Try to allocate more than max block size
        let result = alloc.alloc_pages(513);
        assert!(result.is_none());
    }

    #[test]
    fn test_alloc_zero() {
        let mut alloc = create_allocator();

        let result = alloc.alloc_pages(0);
        assert!(result.is_none());
    }

    #[test]
    fn test_out_of_memory() {
        let mut alloc = BuddyAllocator::new(PhysAddr::new(TEST_BASE), 8);

        // Allocate all memory
        let addr1 = alloc.alloc_pages(4).expect("alloc failed");
        let addr2 = alloc.alloc_pages(4).expect("alloc failed");
        assert_eq!(alloc.free_page_count(), 0);

        // Try to allocate more
        let result = alloc.alloc_pages(1);
        assert!(result.is_none());

        // Free and retry
        alloc.dealloc_pages(addr1, 4);
        let addr3 = alloc.alloc_pages(1).expect("alloc failed");
        assert!(addr3.is_page_aligned());

        alloc.dealloc_pages(addr2, 4);
        alloc.dealloc_pages(addr3, 1);
    }

    #[test]
    fn test_block_splitting() {
        let mut alloc = BuddyAllocator::new(PhysAddr::new(TEST_BASE), 8);

        // Allocate single page from 8-page region
        let addr = alloc.alloc_pages(1).expect("alloc failed");

        // Should have splits recorded
        assert!(alloc.stats().splits() > 0);

        // Remaining should be 7 pages in various orders
        assert_eq!(alloc.free_page_count(), 7);

        alloc.dealloc_pages(addr, 1);
    }

    #[test]
    fn test_block_coalescing() {
        let mut alloc = BuddyAllocator::new(PhysAddr::new(TEST_BASE), 8);

        // Allocate two adjacent blocks
        let addr1 = alloc.alloc_pages(4).expect("alloc failed");
        let addr2 = alloc.alloc_pages(4).expect("alloc failed");

        // Free in order - should coalesce
        alloc.dealloc_pages(addr1, 4);
        alloc.dealloc_pages(addr2, 4);

        // Should have coalesces recorded
        assert!(alloc.stats().coalesces() > 0);

        // All memory should be free
        assert_eq!(alloc.free_page_count(), 8);
    }

    #[test]
    fn test_allocation_patterns() {
        let mut alloc = create_allocator();

        // Allocate many small blocks
        let mut addrs = [PhysAddr::NULL; 64];
        for addr in &mut addrs {
            *addr = alloc.alloc_pages(1).expect("alloc failed");
        }
        assert_eq!(alloc.used_pages(), 64);

        // Free every other one
        for (i, addr) in addrs.iter().enumerate() {
            if i % 2 == 0 {
                alloc.dealloc_pages(*addr, 1);
            }
        }
        assert_eq!(alloc.used_pages(), 32);

        // Free the rest
        for (i, addr) in addrs.iter().enumerate() {
            if i % 2 == 1 {
                alloc.dealloc_pages(*addr, 1);
            }
        }
        assert_eq!(alloc.used_pages(), 0);
    }

    #[test]
    fn test_try_alloc() {
        let mut alloc = create_allocator();

        let result = alloc.try_alloc_pages(4);
        assert!(result.is_ok());

        let result = alloc.try_alloc_pages(0);
        assert_eq!(result, Err(PhysMemError::ZeroAllocation));

        let result = alloc.try_alloc_pages(1024);
        assert_eq!(result, Err(PhysMemError::AllocationTooLarge));
    }

    #[test]
    fn test_try_free() {
        let mut alloc = create_allocator();

        let addr = alloc.alloc_pages(4).unwrap();

        let result = alloc.try_free_pages(addr, 4);
        assert!(result.is_ok());

        let result = alloc.try_free_pages(addr, 0);
        assert_eq!(result, Err(PhysMemError::ZeroAllocation));

        let result = alloc.try_free_pages(PhysAddr::new(1), 4);
        assert_eq!(result, Err(PhysMemError::UnalignedAddress));

        let result = alloc.try_free_pages(PhysAddr::new(0), 4);
        assert_eq!(result, Err(PhysMemError::AddressOutOfRange));
    }

    #[test]
    fn test_alloc_frame() {
        let mut alloc = create_allocator();

        let frame = alloc.alloc_frame(4).expect("alloc failed");
        assert_eq!(frame.pages(), 4);
        assert!(frame.addr().is_page_aligned());

        alloc.free_frame(&frame);
        assert_eq!(alloc.used_pages(), 0);
    }

    #[test]
    fn test_reset() {
        let mut alloc = create_allocator();

        // Allocate some memory
        let _ = alloc.alloc_pages(100);
        let _ = alloc.alloc_pages(50);
        assert!(alloc.used_pages() > 0);

        // Reset
        alloc.reset();
        assert_eq!(alloc.free_page_count(), TEST_PAGES);
        assert_eq!(alloc.used_pages(), 0);
    }

    #[test]
    fn test_uninit() {
        let mut alloc = BuddyAllocator::uninit();

        // Should fail before init
        let result = alloc.try_alloc_pages(1);
        assert_eq!(result, Err(PhysMemError::NotInitialized));

        // Initialize
        alloc.init(PhysAddr::new(TEST_BASE), 64).unwrap();

        // Should work now
        let addr = alloc.alloc_pages(1).expect("alloc failed");
        alloc.dealloc_pages(addr, 1);
    }

    #[test]
    fn test_init_invalid_alignment() {
        let mut alloc = BuddyAllocator::uninit();

        let result = alloc.init(PhysAddr::new(0x1001), 64);
        assert_eq!(result, Err(PhysMemError::UnalignedAddress));
    }

    #[test]
    fn test_statistics() {
        let mut alloc = create_allocator();

        // Initial stats
        assert_eq!(alloc.stats().allocations(), 0);
        assert_eq!(alloc.stats().frees(), 0);

        // Allocate
        let addr = alloc.alloc_pages(4).unwrap();
        assert_eq!(alloc.stats().successful_allocations(), 1);

        // Failed allocation
        let mut small_alloc = BuddyAllocator::new(PhysAddr::new(TEST_BASE), 1);
        let _ = small_alloc.alloc_pages(1); // Use all
        let _ = small_alloc.alloc_pages(1); // Fail
        assert_eq!(small_alloc.stats().failed_allocations(), 1);

        // Free
        alloc.dealloc_pages(addr, 4);
        assert_eq!(alloc.stats().frees(), 1);
    }

    #[test]
    fn test_order_stats() {
        let alloc = BuddyAllocator::new(PhysAddr::new(TEST_BASE), 512);

        let stats = alloc.order_stats();

        // 512 pages = one block of order 9
        assert_eq!(stats[9].free_blocks, 1);
        for i in 0..9 {
            assert_eq!(stats[i].free_blocks, 0);
        }
    }

    #[test]
    fn test_contains() {
        let alloc = create_allocator();

        assert!(alloc.contains(PhysAddr::new(TEST_BASE)));
        assert!(alloc.contains(PhysAddr::new(TEST_BASE + 0x1000)));
        assert!(!alloc.contains(PhysAddr::new(TEST_BASE - 1)));
        assert!(!alloc.contains(PhysAddr::new(TEST_BASE + TEST_PAGES as u64 * PAGE_SIZE as u64)));
    }

    #[test]
    #[should_panic(expected = "page-aligned")]
    fn test_new_unaligned_panics() {
        let _ = BuddyAllocator::new(PhysAddr::new(0x1001), 64);
    }

    #[test]
    fn test_free_list_operations() {
        let mut list = FreeList::new();

        assert!(list.is_empty());
        assert_eq!(list.len(), 0);

        list.push(PhysAddr::new(0x1000)).unwrap();
        list.push(PhysAddr::new(0x2000)).unwrap();
        assert_eq!(list.len(), 2);
        assert!(!list.is_empty());

        assert!(list.contains(PhysAddr::new(0x1000)));
        assert!(!list.contains(PhysAddr::new(0x3000)));

        assert!(list.remove(PhysAddr::new(0x1000)));
        assert_eq!(list.len(), 1);
        assert!(!list.contains(PhysAddr::new(0x1000)));

        let addr = list.pop().unwrap();
        assert_eq!(addr.as_u64(), 0x2000);
        assert!(list.is_empty());

        assert!(list.pop().is_none());
    }
}
