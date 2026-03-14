//! Physical address types.
//!
//! This module provides the `PhysAddr` newtype wrapper for physical addresses,
//! ensuring type safety when working with physical memory.

use core::fmt;
use core::ops::{Add, AddAssign, Sub, SubAssign};

use crate::{align_down, align_up, is_page_aligned, PAGE_SIZE};

/// A physical memory address.
///
/// This is a newtype wrapper around `u64` that provides type safety and
/// utility methods for working with physical addresses.
///
/// # Invariants
///
/// Physical addresses should typically be page-aligned when used for
/// allocation. The allocator enforces this constraint.
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::PhysAddr;
///
/// let addr = PhysAddr::new(0x1000_0000);
/// assert_eq!(addr.as_u64(), 0x1000_0000);
/// assert!(addr.is_page_aligned());
///
/// let next_page = addr.add_pages(1);
/// assert_eq!(next_page.as_u64(), 0x1000_1000);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct PhysAddr(u64);

impl PhysAddr {
    /// The null physical address (0x0).
    pub const NULL: Self = Self(0);

    /// Creates a new physical address.
    ///
    /// # Arguments
    ///
    /// * `addr` - The raw physical address value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x1000);
    /// assert_eq!(addr.as_u64(), 0x1000);
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(addr: u64) -> Self {
        Self(addr)
    }

    /// Creates a new physical address from a page frame number.
    ///
    /// The page frame number is the physical address divided by `PAGE_SIZE`.
    ///
    /// # Arguments
    ///
    /// * `pfn` - The page frame number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::from_pfn(1);
    /// assert_eq!(addr.as_u64(), 0x1000); // Page 1 = 4096
    ///
    /// let addr = PhysAddr::from_pfn(256);
    /// assert_eq!(addr.as_u64(), 0x10_0000); // 256 * 4096 = 1MB
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_pfn(pfn: u64) -> Self {
        Self(pfn * PAGE_SIZE as u64)
    }

    /// Returns the raw physical address value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x1234_5000);
    /// assert_eq!(addr.as_u64(), 0x1234_5000);
    /// ```
    #[inline]
    #[must_use]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Returns the page frame number for this address.
    ///
    /// This is the physical address divided by `PAGE_SIZE`, truncating any
    /// offset within the page.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x1000);
    /// assert_eq!(addr.pfn(), 1);
    ///
    /// let addr = PhysAddr::new(0x1234);
    /// assert_eq!(addr.pfn(), 1); // Offset 0x234 is truncated
    /// ```
    #[inline]
    #[must_use]
    pub const fn pfn(self) -> u64 {
        self.0 / PAGE_SIZE as u64
    }

    /// Returns the offset within the page.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x1234);
    /// assert_eq!(addr.page_offset(), 0x234);
    ///
    /// let addr = PhysAddr::new(0x1000);
    /// assert_eq!(addr.page_offset(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub const fn page_offset(self) -> u64 {
        self.0 & (PAGE_SIZE as u64 - 1)
    }

    /// Checks if the address is page-aligned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// assert!(PhysAddr::new(0).is_page_aligned());
    /// assert!(PhysAddr::new(0x1000).is_page_aligned());
    /// assert!(!PhysAddr::new(0x1001).is_page_aligned());
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_page_aligned(self) -> bool {
        is_page_aligned(self.0)
    }

    /// Checks if the address is null (zero).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// assert!(PhysAddr::NULL.is_null());
    /// assert!(PhysAddr::new(0).is_null());
    /// assert!(!PhysAddr::new(0x1000).is_null());
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Aligns the address down to the nearest page boundary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x1234);
    /// assert_eq!(addr.align_down().as_u64(), 0x1000);
    ///
    /// let addr = PhysAddr::new(0x2000);
    /// assert_eq!(addr.align_down().as_u64(), 0x2000);
    /// ```
    #[inline]
    #[must_use]
    pub const fn align_down(self) -> Self {
        Self(align_down(self.0))
    }

    /// Aligns the address up to the nearest page boundary.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x1001);
    /// assert_eq!(addr.align_up().as_u64(), 0x2000);
    ///
    /// let addr = PhysAddr::new(0x2000);
    /// assert_eq!(addr.align_up().as_u64(), 0x2000);
    /// ```
    #[inline]
    #[must_use]
    pub const fn align_up(self) -> Self {
        Self(align_up(self.0))
    }

    /// Adds a number of pages to the address.
    ///
    /// # Arguments
    ///
    /// * `pages` - The number of pages to add.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x1000);
    /// assert_eq!(addr.add_pages(1).as_u64(), 0x2000);
    /// assert_eq!(addr.add_pages(4).as_u64(), 0x5000);
    /// ```
    #[inline]
    #[must_use]
    pub const fn add_pages(self, pages: usize) -> Self {
        Self(self.0 + (pages as u64 * PAGE_SIZE as u64))
    }

    /// Subtracts a number of pages from the address.
    ///
    /// # Arguments
    ///
    /// * `pages` - The number of pages to subtract.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x5000);
    /// assert_eq!(addr.sub_pages(1).as_u64(), 0x4000);
    /// assert_eq!(addr.sub_pages(4).as_u64(), 0x1000);
    /// ```
    #[inline]
    #[must_use]
    pub const fn sub_pages(self, pages: usize) -> Self {
        Self(self.0 - (pages as u64 * PAGE_SIZE as u64))
    }

    /// Adds a byte offset to the address.
    ///
    /// # Arguments
    ///
    /// * `offset` - The byte offset to add.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x1000);
    /// assert_eq!(addr.add_bytes(0x100).as_u64(), 0x1100);
    /// ```
    #[inline]
    #[must_use]
    pub const fn add_bytes(self, offset: u64) -> Self {
        Self(self.0 + offset)
    }

    /// Calculates the number of pages between two addresses.
    ///
    /// Returns the number of complete pages between `self` and `other`.
    /// Both addresses should be page-aligned for meaningful results.
    ///
    /// # Arguments
    ///
    /// * `other` - The other address to compare with.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let start = PhysAddr::new(0x1000);
    /// let end = PhysAddr::new(0x5000);
    /// assert_eq!(start.pages_to(end), 4);
    /// ```
    #[inline]
    #[must_use]
    pub const fn pages_to(self, other: Self) -> usize {
        if other.0 >= self.0 {
            ((other.0 - self.0) / PAGE_SIZE as u64) as usize
        } else {
            0
        }
    }

    /// Checks if this address is within a range.
    ///
    /// # Arguments
    ///
    /// * `start` - The start of the range (inclusive).
    /// * `end` - The end of the range (exclusive).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let start = PhysAddr::new(0x1000);
    /// let end = PhysAddr::new(0x5000);
    ///
    /// assert!(PhysAddr::new(0x1000).is_in_range(start, end));
    /// assert!(PhysAddr::new(0x3000).is_in_range(start, end));
    /// assert!(!PhysAddr::new(0x5000).is_in_range(start, end));
    /// assert!(!PhysAddr::new(0x0000).is_in_range(start, end));
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_in_range(self, start: Self, end: Self) -> bool {
        self.0 >= start.0 && self.0 < end.0
    }

    /// Checks if the address is aligned to a power-of-two order.
    ///
    /// An address is order-aligned if it is aligned to `2^order * PAGE_SIZE`.
    ///
    /// # Arguments
    ///
    /// * `order` - The order to check alignment for.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PhysAddr;
    ///
    /// let addr = PhysAddr::new(0x4000); // 16KB
    /// assert!(addr.is_order_aligned(0)); // Aligned to 4KB
    /// assert!(addr.is_order_aligned(1)); // Aligned to 8KB
    /// assert!(addr.is_order_aligned(2)); // Aligned to 16KB
    /// assert!(!addr.is_order_aligned(3)); // Not aligned to 32KB
    /// ```
    #[inline]
    #[must_use]
    pub const fn is_order_aligned(self, order: usize) -> bool {
        let alignment = (PAGE_SIZE as u64) << order;
        self.0 & (alignment - 1) == 0
    }
}

impl Add<u64> for PhysAddr {
    type Output = Self;

    #[inline]
    fn add(self, rhs: u64) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl AddAssign<u64> for PhysAddr {
    #[inline]
    fn add_assign(&mut self, rhs: u64) {
        self.0 += rhs;
    }
}

impl Sub<u64> for PhysAddr {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: u64) -> Self::Output {
        Self(self.0 - rhs)
    }
}

impl SubAssign<u64> for PhysAddr {
    #[inline]
    fn sub_assign(&mut self, rhs: u64) {
        self.0 -= rhs;
    }
}

impl Sub<PhysAddr> for PhysAddr {
    type Output = u64;

    #[inline]
    fn sub(self, rhs: PhysAddr) -> Self::Output {
        self.0.saturating_sub(rhs.0)
    }
}

impl fmt::Debug for PhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PhysAddr({:#x})", self.0)
    }
}

impl fmt::Display for PhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

impl fmt::LowerHex for PhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl fmt::UpperHex for PhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

impl From<u64> for PhysAddr {
    #[inline]
    fn from(addr: u64) -> Self {
        Self::new(addr)
    }
}

impl From<PhysAddr> for u64 {
    #[inline]
    fn from(addr: PhysAddr) -> Self {
        addr.0
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::format;
    use super::*;

    #[test]
    fn test_new_and_as_u64() {
        let addr = PhysAddr::new(0x1234_5678);
        assert_eq!(addr.as_u64(), 0x1234_5678);
    }

    #[test]
    fn test_from_pfn() {
        assert_eq!(PhysAddr::from_pfn(0).as_u64(), 0);
        assert_eq!(PhysAddr::from_pfn(1).as_u64(), 0x1000);
        assert_eq!(PhysAddr::from_pfn(256).as_u64(), 0x10_0000);
    }

    #[test]
    fn test_pfn() {
        assert_eq!(PhysAddr::new(0).pfn(), 0);
        assert_eq!(PhysAddr::new(0x1000).pfn(), 1);
        assert_eq!(PhysAddr::new(0x1FFF).pfn(), 1);
        assert_eq!(PhysAddr::new(0x2000).pfn(), 2);
    }

    #[test]
    fn test_page_offset() {
        assert_eq!(PhysAddr::new(0x1000).page_offset(), 0);
        assert_eq!(PhysAddr::new(0x1001).page_offset(), 1);
        assert_eq!(PhysAddr::new(0x1FFF).page_offset(), 0xFFF);
    }

    #[test]
    fn test_is_page_aligned() {
        assert!(PhysAddr::new(0).is_page_aligned());
        assert!(PhysAddr::new(0x1000).is_page_aligned());
        assert!(!PhysAddr::new(0x1001).is_page_aligned());
        assert!(!PhysAddr::new(0xFFF).is_page_aligned());
    }

    #[test]
    fn test_is_null() {
        assert!(PhysAddr::NULL.is_null());
        assert!(PhysAddr::new(0).is_null());
        assert!(!PhysAddr::new(1).is_null());
    }

    #[test]
    fn test_align_down() {
        assert_eq!(PhysAddr::new(0).align_down().as_u64(), 0);
        assert_eq!(PhysAddr::new(0x1000).align_down().as_u64(), 0x1000);
        assert_eq!(PhysAddr::new(0x1001).align_down().as_u64(), 0x1000);
        assert_eq!(PhysAddr::new(0x1FFF).align_down().as_u64(), 0x1000);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(PhysAddr::new(0).align_up().as_u64(), 0);
        assert_eq!(PhysAddr::new(0x1000).align_up().as_u64(), 0x1000);
        assert_eq!(PhysAddr::new(0x1001).align_up().as_u64(), 0x2000);
        assert_eq!(PhysAddr::new(0x1FFF).align_up().as_u64(), 0x2000);
    }

    #[test]
    fn test_add_sub_pages() {
        let addr = PhysAddr::new(0x1000);
        assert_eq!(addr.add_pages(1).as_u64(), 0x2000);
        assert_eq!(addr.add_pages(4).as_u64(), 0x5000);
        assert_eq!(addr.add_pages(0).as_u64(), 0x1000);

        let addr = PhysAddr::new(0x5000);
        assert_eq!(addr.sub_pages(1).as_u64(), 0x4000);
        assert_eq!(addr.sub_pages(4).as_u64(), 0x1000);
    }

    #[test]
    fn test_pages_to() {
        let start = PhysAddr::new(0x1000);
        let end = PhysAddr::new(0x5000);
        assert_eq!(start.pages_to(end), 4);
        assert_eq!(end.pages_to(start), 0);
        assert_eq!(start.pages_to(start), 0);
    }

    #[test]
    fn test_is_in_range() {
        let start = PhysAddr::new(0x1000);
        let end = PhysAddr::new(0x5000);

        assert!(PhysAddr::new(0x1000).is_in_range(start, end));
        assert!(PhysAddr::new(0x2000).is_in_range(start, end));
        assert!(PhysAddr::new(0x4FFF).is_in_range(start, end));
        assert!(!PhysAddr::new(0x5000).is_in_range(start, end));
        assert!(!PhysAddr::new(0x0FFF).is_in_range(start, end));
    }

    #[test]
    fn test_is_order_aligned() {
        // 0x4000 = 16KB = 4 pages
        let addr = PhysAddr::new(0x4000);
        assert!(addr.is_order_aligned(0)); // 4KB
        assert!(addr.is_order_aligned(1)); // 8KB
        assert!(addr.is_order_aligned(2)); // 16KB
        assert!(!addr.is_order_aligned(3)); // 32KB

        // 0x8000 = 32KB = 8 pages
        let addr = PhysAddr::new(0x8000);
        assert!(addr.is_order_aligned(0));
        assert!(addr.is_order_aligned(1));
        assert!(addr.is_order_aligned(2));
        assert!(addr.is_order_aligned(3));
        assert!(!addr.is_order_aligned(4)); // 64KB
    }

    #[test]
    fn test_arithmetic_ops() {
        let mut addr = PhysAddr::new(0x1000);
        assert_eq!((addr + 0x100).as_u64(), 0x1100);
        assert_eq!((addr - 0x100).as_u64(), 0x0F00);

        addr += 0x1000;
        assert_eq!(addr.as_u64(), 0x2000);

        addr -= 0x500;
        assert_eq!(addr.as_u64(), 0x1B00);
    }

    #[test]
    fn test_addr_subtraction() {
        let a = PhysAddr::new(0x5000);
        let b = PhysAddr::new(0x1000);
        assert_eq!(a - b, 0x4000);
        assert_eq!(b - a, 0); // saturating
    }

    #[test]
    fn test_display() {
        let addr = PhysAddr::new(0x1234_5000);
        assert_eq!(format!("{addr}"), "0x12345000");
        assert_eq!(format!("{addr:?}"), "PhysAddr(0x12345000)");
        assert_eq!(format!("{addr:x}"), "12345000");
        assert_eq!(format!("{addr:X}"), "12345000");
    }

    #[test]
    fn test_from_into() {
        let addr: PhysAddr = 0x1234u64.into();
        assert_eq!(addr.as_u64(), 0x1234);

        let val: u64 = addr.into();
        assert_eq!(val, 0x1234);
    }

    #[test]
    fn test_ordering() {
        let a = PhysAddr::new(0x1000);
        let b = PhysAddr::new(0x2000);
        let c = PhysAddr::new(0x1000);

        assert!(a < b);
        assert!(b > a);
        assert!(a == c);
        assert!(a <= c);
        assert!(a >= c);
    }
}
