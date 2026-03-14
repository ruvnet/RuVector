//! Allocator statistics.
//!
//! This module provides types for tracking allocation statistics.

use core::fmt;

/// Statistics for the buddy allocator.
///
/// These statistics help monitor memory usage and allocation patterns.
/// They are updated atomically and can be retrieved at any time.
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::AllocatorStats;
///
/// let stats = AllocatorStats::new(4096, 4096);
/// assert_eq!(stats.total_pages(), 4096);
/// assert_eq!(stats.free_pages(), 4096);
/// assert_eq!(stats.used_pages(), 0);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AllocatorStats {
    /// Total number of pages under management.
    total_pages: usize,
    /// Number of currently free pages.
    free_pages: usize,
    /// Total number of allocation requests.
    allocations: u64,
    /// Total number of successful allocations.
    successful_allocations: u64,
    /// Total number of free operations.
    frees: u64,
    /// Total number of block splits during allocation.
    splits: u64,
    /// Total number of block coalesces during free.
    coalesces: u64,
    /// Peak number of pages ever used.
    peak_used_pages: usize,
}

impl AllocatorStats {
    /// Creates new statistics.
    ///
    /// # Arguments
    ///
    /// * `total_pages` - Total pages under management.
    /// * `free_pages` - Initial free pages.
    #[inline]
    #[must_use]
    pub const fn new(total_pages: usize, free_pages: usize) -> Self {
        Self {
            total_pages,
            free_pages,
            allocations: 0,
            successful_allocations: 0,
            frees: 0,
            splits: 0,
            coalesces: 0,
            peak_used_pages: total_pages.saturating_sub(free_pages),
        }
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
    pub const fn free_pages(&self) -> usize {
        self.free_pages
    }

    /// Returns the number of used pages.
    #[inline]
    #[must_use]
    pub const fn used_pages(&self) -> usize {
        self.total_pages.saturating_sub(self.free_pages)
    }

    /// Returns the total memory in bytes.
    #[inline]
    #[must_use]
    pub const fn total_bytes(&self) -> usize {
        self.total_pages * crate::PAGE_SIZE
    }

    /// Returns the free memory in bytes.
    #[inline]
    #[must_use]
    pub const fn free_bytes(&self) -> usize {
        self.free_pages * crate::PAGE_SIZE
    }

    /// Returns the used memory in bytes.
    #[inline]
    #[must_use]
    pub const fn used_bytes(&self) -> usize {
        self.used_pages() * crate::PAGE_SIZE
    }

    /// Returns the memory utilization as a percentage (0-100).
    #[inline]
    #[must_use]
    pub const fn utilization_percent(&self) -> u8 {
        if self.total_pages == 0 {
            0
        } else {
            ((self.used_pages() * 100) / self.total_pages) as u8
        }
    }

    /// Returns the total number of allocation requests.
    #[inline]
    #[must_use]
    pub const fn allocations(&self) -> u64 {
        self.allocations
    }

    /// Returns the total number of successful allocations.
    #[inline]
    #[must_use]
    pub const fn successful_allocations(&self) -> u64 {
        self.successful_allocations
    }

    /// Returns the number of failed allocations.
    #[inline]
    #[must_use]
    pub const fn failed_allocations(&self) -> u64 {
        self.allocations.saturating_sub(self.successful_allocations)
    }

    /// Returns the total number of free operations.
    #[inline]
    #[must_use]
    pub const fn frees(&self) -> u64 {
        self.frees
    }

    /// Returns the total number of block splits.
    #[inline]
    #[must_use]
    pub const fn splits(&self) -> u64 {
        self.splits
    }

    /// Returns the total number of block coalesces.
    #[inline]
    #[must_use]
    pub const fn coalesces(&self) -> u64 {
        self.coalesces
    }

    /// Returns the peak number of used pages.
    #[inline]
    #[must_use]
    pub const fn peak_used_pages(&self) -> usize {
        self.peak_used_pages
    }

    /// Returns the peak memory usage in bytes.
    #[inline]
    #[must_use]
    pub const fn peak_used_bytes(&self) -> usize {
        self.peak_used_pages * crate::PAGE_SIZE
    }

    /// Records an allocation attempt.
    ///
    /// # Arguments
    ///
    /// * `pages` - Number of pages requested.
    /// * `success` - Whether the allocation succeeded.
    #[inline]
    pub fn record_allocation(&mut self, pages: usize, success: bool) {
        self.allocations += 1;
        if success {
            self.successful_allocations += 1;
            self.free_pages = self.free_pages.saturating_sub(pages);
            let used = self.used_pages();
            if used > self.peak_used_pages {
                self.peak_used_pages = used;
            }
        }
    }

    /// Records a free operation.
    ///
    /// # Arguments
    ///
    /// * `pages` - Number of pages freed.
    #[inline]
    pub fn record_free(&mut self, pages: usize) {
        self.frees += 1;
        self.free_pages = (self.free_pages + pages).min(self.total_pages);
    }

    /// Records a block split.
    #[inline]
    pub fn record_split(&mut self) {
        self.splits += 1;
    }

    /// Records a block coalesce.
    #[inline]
    pub fn record_coalesce(&mut self) {
        self.coalesces += 1;
    }

    /// Resets counters while preserving current state.
    #[inline]
    pub fn reset_counters(&mut self) {
        self.allocations = 0;
        self.successful_allocations = 0;
        self.frees = 0;
        self.splits = 0;
        self.coalesces = 0;
        self.peak_used_pages = self.used_pages();
    }

    /// Returns the allocation success rate as a percentage (0-100).
    #[inline]
    #[must_use]
    pub const fn success_rate_percent(&self) -> u8 {
        if self.allocations == 0 {
            100
        } else {
            ((self.successful_allocations * 100) / self.allocations) as u8
        }
    }

    /// Returns the average splits per allocation.
    #[inline]
    #[must_use]
    pub fn splits_per_allocation(&self) -> f64 {
        if self.successful_allocations == 0 {
            0.0
        } else {
            self.splits as f64 / self.successful_allocations as f64
        }
    }

    /// Returns the average coalesces per free.
    #[inline]
    #[must_use]
    pub fn coalesces_per_free(&self) -> f64 {
        if self.frees == 0 {
            0.0
        } else {
            self.coalesces as f64 / self.frees as f64
        }
    }
}

impl fmt::Debug for AllocatorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AllocatorStats")
            .field("total_pages", &self.total_pages)
            .field("free_pages", &self.free_pages)
            .field("used_pages", &self.used_pages())
            .field("utilization_percent", &self.utilization_percent())
            .field("allocations", &self.allocations)
            .field("successful", &self.successful_allocations)
            .field("frees", &self.frees)
            .field("splits", &self.splits)
            .field("coalesces", &self.coalesces)
            .finish()
    }
}

impl fmt::Display for AllocatorStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Memory: {}/{} pages ({}%), Allocs: {} ({} failed), Frees: {}",
            self.used_pages(),
            self.total_pages,
            self.utilization_percent(),
            self.successful_allocations,
            self.failed_allocations(),
            self.frees
        )
    }
}

/// Statistics per order level.
///
/// Tracks allocation patterns at each block size.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct OrderStats {
    /// Number of blocks currently in the free list.
    pub free_blocks: usize,
    /// Total allocations at this order.
    pub allocations: u64,
    /// Total frees at this order.
    pub frees: u64,
}

impl OrderStats {
    /// Creates new order statistics.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            free_blocks: 0,
            allocations: 0,
            frees: 0,
        }
    }
}

impl fmt::Debug for OrderStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OrderStats")
            .field("free_blocks", &self.free_blocks)
            .field("allocations", &self.allocations)
            .field("frees", &self.frees)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::format;
    use super::*;

    #[test]
    fn test_new_stats() {
        let stats = AllocatorStats::new(1024, 1024);
        assert_eq!(stats.total_pages(), 1024);
        assert_eq!(stats.free_pages(), 1024);
        assert_eq!(stats.used_pages(), 0);
        assert_eq!(stats.utilization_percent(), 0);
    }

    #[test]
    fn test_record_allocation() {
        let mut stats = AllocatorStats::new(1024, 1024);

        stats.record_allocation(10, true);
        assert_eq!(stats.allocations(), 1);
        assert_eq!(stats.successful_allocations(), 1);
        assert_eq!(stats.free_pages(), 1014);
        assert_eq!(stats.used_pages(), 10);

        stats.record_allocation(5, false);
        assert_eq!(stats.allocations(), 2);
        assert_eq!(stats.successful_allocations(), 1);
        assert_eq!(stats.failed_allocations(), 1);
        assert_eq!(stats.free_pages(), 1014); // Unchanged on failure
    }

    #[test]
    fn test_record_free() {
        let mut stats = AllocatorStats::new(1024, 1000);

        stats.record_free(10);
        assert_eq!(stats.frees(), 1);
        assert_eq!(stats.free_pages(), 1010);

        // Cannot exceed total
        stats.record_free(100);
        assert_eq!(stats.free_pages(), 1024);
    }

    #[test]
    fn test_peak_tracking() {
        let mut stats = AllocatorStats::new(100, 100);

        stats.record_allocation(30, true);
        assert_eq!(stats.peak_used_pages(), 30);

        stats.record_allocation(20, true);
        assert_eq!(stats.peak_used_pages(), 50);

        stats.record_free(40);
        assert_eq!(stats.peak_used_pages(), 50); // Peak unchanged

        stats.record_allocation(60, true);
        assert_eq!(stats.peak_used_pages(), 70);
    }

    #[test]
    fn test_splits_coalesces() {
        let mut stats = AllocatorStats::new(1024, 1024);

        stats.record_split();
        stats.record_split();
        stats.record_coalesce();

        assert_eq!(stats.splits(), 2);
        assert_eq!(stats.coalesces(), 1);
    }

    #[test]
    fn test_utilization() {
        let stats = AllocatorStats::new(100, 50);
        assert_eq!(stats.utilization_percent(), 50);

        let stats = AllocatorStats::new(100, 100);
        assert_eq!(stats.utilization_percent(), 0);

        let stats = AllocatorStats::new(100, 0);
        assert_eq!(stats.utilization_percent(), 100);

        let stats = AllocatorStats::new(0, 0);
        assert_eq!(stats.utilization_percent(), 0);
    }

    #[test]
    fn test_success_rate() {
        let mut stats = AllocatorStats::new(1024, 1024);
        assert_eq!(stats.success_rate_percent(), 100); // No attempts

        stats.record_allocation(10, true);
        assert_eq!(stats.success_rate_percent(), 100);

        stats.record_allocation(10, false);
        assert_eq!(stats.success_rate_percent(), 50);

        stats.record_allocation(10, true);
        assert_eq!(stats.success_rate_percent(), 66); // 2/3
    }

    #[test]
    fn test_bytes_calculations() {
        let stats = AllocatorStats::new(256, 128);

        assert_eq!(stats.total_bytes(), 256 * 4096);
        assert_eq!(stats.free_bytes(), 128 * 4096);
        assert_eq!(stats.used_bytes(), 128 * 4096);
    }

    #[test]
    fn test_reset_counters() {
        let mut stats = AllocatorStats::new(100, 50);
        stats.allocations = 100;
        stats.successful_allocations = 90;
        stats.frees = 50;
        stats.splits = 20;
        stats.coalesces = 10;
        stats.peak_used_pages = 80;

        stats.reset_counters();

        assert_eq!(stats.allocations(), 0);
        assert_eq!(stats.successful_allocations(), 0);
        assert_eq!(stats.frees(), 0);
        assert_eq!(stats.splits(), 0);
        assert_eq!(stats.coalesces(), 0);
        assert_eq!(stats.peak_used_pages(), 50); // Reset to current usage
        assert_eq!(stats.free_pages(), 50); // Preserved
    }

    #[test]
    fn test_display() {
        let mut stats = AllocatorStats::new(1024, 512);
        stats.allocations = 10;
        stats.successful_allocations = 8;
        stats.frees = 3;

        let s = format!("{stats}");
        assert!(s.contains("512/1024"));
        assert!(s.contains("50%"));
        assert!(s.contains("8"));
        assert!(s.contains("2 failed"));
    }

    #[test]
    fn test_order_stats() {
        let mut os = OrderStats::new();
        assert_eq!(os.free_blocks, 0);
        assert_eq!(os.allocations, 0);
        assert_eq!(os.frees, 0);

        os.free_blocks = 5;
        os.allocations = 10;
        os.frees = 3;

        assert_eq!(os.free_blocks, 5);
    }
}
