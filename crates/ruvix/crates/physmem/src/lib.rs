//! # RuVix Physical Memory Allocator
//!
//! This crate provides a buddy allocator for physical page frame allocation
//! as part of the RuVix Cognition Kernel (ADR-087).
//!
//! ## Overview
//!
//! The buddy allocator manages physical memory using power-of-two block sizes,
//! enabling efficient allocation and deallocation with minimal fragmentation.
//! It supports block sizes from 4KB (single page) to 2MB (512 pages).
//!
//! ## Architecture
//!
//! ```text
//! +------------------+
//! | BuddyAllocator   |
//! |------------------|
//! | free_lists[10]   | <- One list per order (0-9)
//! | base_addr        | <- Start of managed memory
//! | total_pages      | <- Total pages under management
//! | stats            | <- Allocation statistics
//! +------------------+
//!          |
//!          v
//! +------------------+
//! | Free Lists       |
//! |------------------|
//! | Order 0: 4KB     | <- Single pages
//! | Order 1: 8KB     | <- 2 pages
//! | Order 2: 16KB    | <- 4 pages
//! | ...              |
//! | Order 9: 2MB     | <- 512 pages
//! +------------------+
//! ```
//!
//! ## Features
//!
//! - `std`: Enable standard library support
//! - `alloc`: Enable alloc crate support
//! - `stats`: Enable detailed statistics collection
//! - `debug-alloc`: Enable debug assertions for allocation tracking
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvix_physmem::{BuddyAllocator, PhysAddr, PAGE_SIZE};
//!
//! // Create allocator for 16MB of memory starting at physical address 0x1000_0000
//! let mut allocator = BuddyAllocator::new(PhysAddr::new(0x1000_0000), 4096);
//!
//! // Allocate 4 contiguous pages (16KB)
//! if let Some(addr) = allocator.alloc_pages(4) {
//!     // Use the memory...
//!     allocator.dealloc_pages(addr, 4);
//! }
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

mod addr;
mod allocator;
mod error;
mod frame;
mod stats;

pub use addr::PhysAddr;
pub use allocator::BuddyAllocator;
pub use error::PhysMemError;
pub use frame::{PageFrame, PageOrder};
pub use stats::AllocatorStats;

// Re-export kernel error for convenience
pub use ruvix_types::KernelError;

/// Page size in bytes (4KB).
pub const PAGE_SIZE: usize = 4096;

/// Page size shift (log2(PAGE_SIZE)).
pub const PAGE_SHIFT: usize = 12;

/// Maximum order for buddy allocation (2^9 = 512 pages = 2MB).
pub const MAX_ORDER: usize = 10;

/// Minimum allocation unit (single page).
pub const MIN_ORDER: usize = 0;

/// Maximum block size in pages (2^(MAX_ORDER-1) = 512 pages).
pub const MAX_BLOCK_PAGES: usize = 1 << (MAX_ORDER - 1);

/// Maximum block size in bytes (2MB).
pub const MAX_BLOCK_SIZE: usize = MAX_BLOCK_PAGES * PAGE_SIZE;

/// Calculates the order required for a given number of pages.
///
/// Returns the smallest order `n` such that `2^n >= pages`.
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::pages_to_order;
///
/// assert_eq!(pages_to_order(1), 0);  // 2^0 = 1 page
/// assert_eq!(pages_to_order(2), 1);  // 2^1 = 2 pages
/// assert_eq!(pages_to_order(3), 2);  // 2^2 = 4 pages (rounds up)
/// assert_eq!(pages_to_order(4), 2);  // 2^2 = 4 pages
/// assert_eq!(pages_to_order(5), 3);  // 2^3 = 8 pages (rounds up)
/// ```
#[inline]
#[must_use]
pub const fn pages_to_order(pages: usize) -> usize {
    if pages == 0 {
        return 0;
    }

    // Calculate ceiling of log2(pages)
    let leading_zeros = (pages - 1).leading_zeros() as usize;
    let bits = core::mem::size_of::<usize>() * 8;

    if leading_zeros >= bits {
        0
    } else {
        bits - leading_zeros
    }
}

/// Calculates the number of pages for a given order.
///
/// Returns `2^order` pages.
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::order_to_pages;
///
/// assert_eq!(order_to_pages(0), 1);   // 2^0 = 1 page
/// assert_eq!(order_to_pages(1), 2);   // 2^1 = 2 pages
/// assert_eq!(order_to_pages(2), 4);   // 2^2 = 4 pages
/// assert_eq!(order_to_pages(9), 512); // 2^9 = 512 pages
/// ```
#[inline]
#[must_use]
pub const fn order_to_pages(order: usize) -> usize {
    1 << order
}

/// Calculates the block size in bytes for a given order.
///
/// Returns `2^order * PAGE_SIZE` bytes.
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::{order_to_bytes, PAGE_SIZE};
///
/// assert_eq!(order_to_bytes(0), PAGE_SIZE);       // 4KB
/// assert_eq!(order_to_bytes(1), 2 * PAGE_SIZE);   // 8KB
/// assert_eq!(order_to_bytes(9), 512 * PAGE_SIZE); // 2MB
/// ```
#[inline]
#[must_use]
pub const fn order_to_bytes(order: usize) -> usize {
    order_to_pages(order) * PAGE_SIZE
}

/// Checks if an address is page-aligned.
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::is_page_aligned;
///
/// assert!(is_page_aligned(0));
/// assert!(is_page_aligned(4096));
/// assert!(is_page_aligned(0x1000_0000));
/// assert!(!is_page_aligned(1));
/// assert!(!is_page_aligned(4097));
/// ```
#[inline]
#[must_use]
pub const fn is_page_aligned(addr: u64) -> bool {
    addr & ((PAGE_SIZE as u64) - 1) == 0
}

/// Aligns an address down to the nearest page boundary.
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::align_down;
///
/// assert_eq!(align_down(0), 0);
/// assert_eq!(align_down(4095), 0);
/// assert_eq!(align_down(4096), 4096);
/// assert_eq!(align_down(4097), 4096);
/// assert_eq!(align_down(8192), 8192);
/// ```
#[inline]
#[must_use]
pub const fn align_down(addr: u64) -> u64 {
    addr & !((PAGE_SIZE as u64) - 1)
}

/// Aligns an address up to the nearest page boundary.
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::align_up;
///
/// assert_eq!(align_up(0), 0);
/// assert_eq!(align_up(1), 4096);
/// assert_eq!(align_up(4095), 4096);
/// assert_eq!(align_up(4096), 4096);
/// assert_eq!(align_up(4097), 8192);
/// ```
#[inline]
#[must_use]
pub const fn align_up(addr: u64) -> u64 {
    let page_mask = (PAGE_SIZE as u64) - 1;
    (addr + page_mask) & !page_mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pages_to_order() {
        assert_eq!(pages_to_order(0), 0);
        assert_eq!(pages_to_order(1), 0);
        assert_eq!(pages_to_order(2), 1);
        assert_eq!(pages_to_order(3), 2);
        assert_eq!(pages_to_order(4), 2);
        assert_eq!(pages_to_order(5), 3);
        assert_eq!(pages_to_order(8), 3);
        assert_eq!(pages_to_order(9), 4);
        assert_eq!(pages_to_order(16), 4);
        assert_eq!(pages_to_order(17), 5);
        assert_eq!(pages_to_order(256), 8);
        assert_eq!(pages_to_order(512), 9);
    }

    #[test]
    fn test_order_to_pages() {
        assert_eq!(order_to_pages(0), 1);
        assert_eq!(order_to_pages(1), 2);
        assert_eq!(order_to_pages(2), 4);
        assert_eq!(order_to_pages(3), 8);
        assert_eq!(order_to_pages(4), 16);
        assert_eq!(order_to_pages(9), 512);
    }

    #[test]
    fn test_order_to_bytes() {
        assert_eq!(order_to_bytes(0), 4096);
        assert_eq!(order_to_bytes(1), 8192);
        assert_eq!(order_to_bytes(2), 16384);
        assert_eq!(order_to_bytes(9), 2 * 1024 * 1024);
    }

    #[test]
    fn test_page_alignment() {
        assert!(is_page_aligned(0));
        assert!(is_page_aligned(PAGE_SIZE as u64));
        assert!(is_page_aligned(0x1000_0000));
        assert!(!is_page_aligned(1));
        assert!(!is_page_aligned(PAGE_SIZE as u64 - 1));
    }

    #[test]
    fn test_align_down() {
        assert_eq!(align_down(0), 0);
        assert_eq!(align_down(1), 0);
        assert_eq!(align_down(4095), 0);
        assert_eq!(align_down(4096), 4096);
        assert_eq!(align_down(4097), 4096);
        assert_eq!(align_down(8191), 4096);
        assert_eq!(align_down(8192), 8192);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 4096);
        assert_eq!(align_up(4095), 4096);
        assert_eq!(align_up(4096), 4096);
        assert_eq!(align_up(4097), 8192);
        assert_eq!(align_up(8192), 8192);
    }

    #[test]
    fn test_roundtrip_order_pages() {
        for order in 0..MAX_ORDER {
            let pages = order_to_pages(order);
            let back = pages_to_order(pages);
            assert_eq!(back, order, "Order {order} -> {pages} pages -> order {back}");
        }
    }
}
