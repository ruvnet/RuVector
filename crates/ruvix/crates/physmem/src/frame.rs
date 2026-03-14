//! Page frame types.
//!
//! This module provides types for representing physical page frames
//! and their allocation orders.

use core::fmt;

use crate::{order_to_bytes, order_to_pages, PhysAddr, MAX_ORDER, PAGE_SIZE};

/// The order of a page frame block.
///
/// Order `n` represents `2^n` contiguous pages:
/// - Order 0: 1 page (4KB)
/// - Order 1: 2 pages (8KB)
/// - Order 2: 4 pages (16KB)
/// - ...
/// - Order 9: 512 pages (2MB)
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::PageOrder;
///
/// let order = PageOrder::new(2).unwrap();
/// assert_eq!(order.pages(), 4);
/// assert_eq!(order.bytes(), 16384);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct PageOrder(u8);

impl PageOrder {
    /// The minimum order (single page, 4KB).
    pub const MIN: Self = Self(0);

    /// The maximum order (512 pages, 2MB).
    pub const MAX: Self = Self((MAX_ORDER - 1) as u8);

    /// Creates a new page order.
    ///
    /// Returns `None` if the order is greater than or equal to `MAX_ORDER`.
    ///
    /// # Arguments
    ///
    /// * `order` - The order value (0 to `MAX_ORDER - 1`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PageOrder;
    ///
    /// assert!(PageOrder::new(0).is_some());
    /// assert!(PageOrder::new(9).is_some());
    /// assert!(PageOrder::new(10).is_none()); // MAX_ORDER = 10
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(order: usize) -> Option<Self> {
        if order < MAX_ORDER {
            Some(Self(order as u8))
        } else {
            None
        }
    }

    /// Creates a new page order without bounds checking.
    ///
    /// # Safety
    ///
    /// This is safe because we only store a u8 and the buddy allocator
    /// will validate the order. However, using an invalid order will
    /// result in unexpected behavior.
    ///
    /// # Arguments
    ///
    /// * `order` - The order value.
    #[inline]
    #[must_use]
    pub const fn new_unchecked(order: usize) -> Self {
        Self(order as u8)
    }

    /// Returns the order as a `usize`.
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Returns the number of pages for this order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PageOrder;
    ///
    /// assert_eq!(PageOrder::new(0).unwrap().pages(), 1);
    /// assert_eq!(PageOrder::new(1).unwrap().pages(), 2);
    /// assert_eq!(PageOrder::new(2).unwrap().pages(), 4);
    /// assert_eq!(PageOrder::new(9).unwrap().pages(), 512);
    /// ```
    #[inline]
    #[must_use]
    pub const fn pages(self) -> usize {
        order_to_pages(self.0 as usize)
    }

    /// Returns the block size in bytes for this order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PageOrder;
    ///
    /// assert_eq!(PageOrder::new(0).unwrap().bytes(), 4096);
    /// assert_eq!(PageOrder::new(1).unwrap().bytes(), 8192);
    /// assert_eq!(PageOrder::new(9).unwrap().bytes(), 2 * 1024 * 1024);
    /// ```
    #[inline]
    #[must_use]
    pub const fn bytes(self) -> usize {
        order_to_bytes(self.0 as usize)
    }

    /// Returns the next higher order, if it exists.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PageOrder;
    ///
    /// let order = PageOrder::new(2).unwrap();
    /// assert_eq!(order.next().map(|o| o.as_usize()), Some(3));
    ///
    /// let max = PageOrder::MAX;
    /// assert!(max.next().is_none());
    /// ```
    #[inline]
    #[must_use]
    pub const fn next(self) -> Option<Self> {
        if (self.0 as usize) < MAX_ORDER - 1 {
            Some(Self(self.0 + 1))
        } else {
            None
        }
    }

    /// Returns the next lower order, if it exists.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PageOrder;
    ///
    /// let order = PageOrder::new(2).unwrap();
    /// assert_eq!(order.prev().map(|o| o.as_usize()), Some(1));
    ///
    /// let min = PageOrder::MIN;
    /// assert!(min.prev().is_none());
    /// ```
    #[inline]
    #[must_use]
    pub const fn prev(self) -> Option<Self> {
        if self.0 > 0 {
            Some(Self(self.0 - 1))
        } else {
            None
        }
    }

    /// Creates an order from a page count, rounding up if necessary.
    ///
    /// Returns `None` if the resulting order would exceed `MAX_ORDER - 1`.
    ///
    /// # Arguments
    ///
    /// * `pages` - The number of pages needed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PageOrder;
    ///
    /// assert_eq!(PageOrder::from_pages(1).map(|o| o.as_usize()), Some(0));
    /// assert_eq!(PageOrder::from_pages(2).map(|o| o.as_usize()), Some(1));
    /// assert_eq!(PageOrder::from_pages(3).map(|o| o.as_usize()), Some(2)); // rounds up
    /// assert_eq!(PageOrder::from_pages(4).map(|o| o.as_usize()), Some(2));
    /// assert_eq!(PageOrder::from_pages(512).map(|o| o.as_usize()), Some(9));
    /// assert!(PageOrder::from_pages(513).is_none()); // Too large
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_pages(pages: usize) -> Option<Self> {
        if pages == 0 {
            return None;
        }

        let order = crate::pages_to_order(pages);
        Self::new(order)
    }

    /// Creates an order from a byte size, rounding up if necessary.
    ///
    /// Returns `None` if the resulting order would exceed `MAX_ORDER - 1`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The number of bytes needed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::PageOrder;
    ///
    /// assert_eq!(PageOrder::from_bytes(1).map(|o| o.as_usize()), Some(0));
    /// assert_eq!(PageOrder::from_bytes(4096).map(|o| o.as_usize()), Some(0));
    /// assert_eq!(PageOrder::from_bytes(4097).map(|o| o.as_usize()), Some(1));
    /// assert_eq!(PageOrder::from_bytes(8192).map(|o| o.as_usize()), Some(1));
    /// ```
    #[inline]
    #[must_use]
    pub const fn from_bytes(bytes: usize) -> Option<Self> {
        if bytes == 0 {
            return None;
        }

        let pages = (bytes + PAGE_SIZE - 1) / PAGE_SIZE;
        Self::from_pages(pages)
    }
}

impl fmt::Debug for PageOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PageOrder({})", self.0)
    }
}

impl fmt::Display for PageOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "order {} ({} pages, {} bytes)", self.0, self.pages(), self.bytes())
    }
}

/// A physical page frame.
///
/// Represents a contiguous block of physical memory with a specific
/// physical address and order (size).
///
/// # Examples
///
/// ```rust
/// use ruvix_physmem::{PageFrame, PhysAddr, PageOrder};
///
/// let frame = PageFrame::new(PhysAddr::new(0x1000), PageOrder::new(2).unwrap());
/// assert_eq!(frame.addr().as_u64(), 0x1000);
/// assert_eq!(frame.pages(), 4);
/// assert_eq!(frame.bytes(), 16384);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageFrame {
    /// The physical address of the frame's start.
    addr: PhysAddr,
    /// The order of the frame (determines size).
    order: PageOrder,
}

impl PageFrame {
    /// Creates a new page frame.
    ///
    /// # Arguments
    ///
    /// * `addr` - The physical address of the frame's start.
    /// * `order` - The order determining the frame's size.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{PageFrame, PhysAddr, PageOrder};
    ///
    /// let frame = PageFrame::new(
    ///     PhysAddr::new(0x1000_0000),
    ///     PageOrder::new(3).unwrap()
    /// );
    /// assert_eq!(frame.pages(), 8);
    /// ```
    #[inline]
    #[must_use]
    pub const fn new(addr: PhysAddr, order: PageOrder) -> Self {
        Self { addr, order }
    }

    /// Creates a page frame for a single page.
    ///
    /// # Arguments
    ///
    /// * `addr` - The physical address of the page.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{PageFrame, PhysAddr};
    ///
    /// let frame = PageFrame::single_page(PhysAddr::new(0x1000));
    /// assert_eq!(frame.pages(), 1);
    /// assert_eq!(frame.order().as_usize(), 0);
    /// ```
    #[inline]
    #[must_use]
    pub const fn single_page(addr: PhysAddr) -> Self {
        Self {
            addr,
            order: PageOrder::MIN,
        }
    }

    /// Returns the physical address of the frame's start.
    #[inline]
    #[must_use]
    pub const fn addr(&self) -> PhysAddr {
        self.addr
    }

    /// Returns the order of the frame.
    #[inline]
    #[must_use]
    pub const fn order(&self) -> PageOrder {
        self.order
    }

    /// Returns the number of pages in the frame.
    #[inline]
    #[must_use]
    pub const fn pages(&self) -> usize {
        self.order.pages()
    }

    /// Returns the size of the frame in bytes.
    #[inline]
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.order.bytes()
    }

    /// Returns the physical address of the frame's end (exclusive).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{PageFrame, PhysAddr, PageOrder};
    ///
    /// let frame = PageFrame::new(
    ///     PhysAddr::new(0x1000),
    ///     PageOrder::new(2).unwrap()
    /// );
    /// assert_eq!(frame.end_addr().as_u64(), 0x5000); // 0x1000 + 4*0x1000
    /// ```
    #[inline]
    #[must_use]
    pub const fn end_addr(&self) -> PhysAddr {
        self.addr.add_pages(self.pages())
    }

    /// Returns the page frame number of the first page.
    #[inline]
    #[must_use]
    pub const fn pfn(&self) -> u64 {
        self.addr.pfn()
    }

    /// Checks if an address is contained within this frame.
    ///
    /// # Arguments
    ///
    /// * `addr` - The address to check.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{PageFrame, PhysAddr, PageOrder};
    ///
    /// let frame = PageFrame::new(
    ///     PhysAddr::new(0x1000),
    ///     PageOrder::new(2).unwrap() // 4 pages: 0x1000..0x5000
    /// );
    ///
    /// assert!(frame.contains(PhysAddr::new(0x1000)));
    /// assert!(frame.contains(PhysAddr::new(0x2500)));
    /// assert!(frame.contains(PhysAddr::new(0x4FFF)));
    /// assert!(!frame.contains(PhysAddr::new(0x5000)));
    /// assert!(!frame.contains(PhysAddr::new(0x0FFF)));
    /// ```
    #[inline]
    #[must_use]
    pub const fn contains(&self, addr: PhysAddr) -> bool {
        addr.is_in_range(self.addr, self.end_addr())
    }

    /// Splits the frame into two buddy frames of the next lower order.
    ///
    /// Returns `None` if the frame is already at the minimum order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{PageFrame, PhysAddr, PageOrder};
    ///
    /// let frame = PageFrame::new(
    ///     PhysAddr::new(0x1000),
    ///     PageOrder::new(2).unwrap() // 4 pages
    /// );
    ///
    /// if let Some((left, right)) = frame.split() {
    ///     assert_eq!(left.pages(), 2);
    ///     assert_eq!(right.pages(), 2);
    ///     assert_eq!(left.addr().as_u64(), 0x1000);
    ///     assert_eq!(right.addr().as_u64(), 0x3000);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub const fn split(&self) -> Option<(Self, Self)> {
        match self.order.prev() {
            Some(new_order) => {
                let left = Self {
                    addr: self.addr,
                    order: new_order,
                };
                let right = Self {
                    addr: self.addr.add_pages(new_order.pages()),
                    order: new_order,
                };
                Some((left, right))
            }
            None => None,
        }
    }

    /// Returns the buddy address for this frame.
    ///
    /// The buddy is the adjacent block of the same size that can be
    /// merged with this block to form a larger block.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{PageFrame, PhysAddr, PageOrder};
    ///
    /// // Frame at 0x1000, order 1 (2 pages)
    /// let frame = PageFrame::new(
    ///     PhysAddr::new(0x1000),
    ///     PageOrder::new(1).unwrap()
    /// );
    /// // Buddy is at 0x3000 (XOR with block size)
    /// assert_eq!(frame.buddy_addr().as_u64(), 0x3000);
    ///
    /// // Frame at 0x3000, order 1 (2 pages)
    /// let frame = PageFrame::new(
    ///     PhysAddr::new(0x3000),
    ///     PageOrder::new(1).unwrap()
    /// );
    /// // Buddy is at 0x1000
    /// assert_eq!(frame.buddy_addr().as_u64(), 0x1000);
    /// ```
    #[inline]
    #[must_use]
    pub const fn buddy_addr(&self) -> PhysAddr {
        let block_size = self.bytes() as u64;
        PhysAddr::new(self.addr.as_u64() ^ block_size)
    }

    /// Merges this frame with its buddy to create a larger frame.
    ///
    /// Returns `None` if the merge would exceed the maximum order.
    ///
    /// # Arguments
    ///
    /// * `buddy` - The buddy frame to merge with.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ruvix_physmem::{PageFrame, PhysAddr, PageOrder};
    ///
    /// let left = PageFrame::new(PhysAddr::new(0x1000), PageOrder::new(1).unwrap());
    /// let right = PageFrame::new(PhysAddr::new(0x3000), PageOrder::new(1).unwrap());
    ///
    /// if let Some(merged) = left.merge(&right) {
    ///     assert_eq!(merged.pages(), 4);
    ///     assert_eq!(merged.addr().as_u64(), 0x1000);
    /// }
    /// ```
    #[inline]
    #[must_use]
    pub const fn merge(&self, buddy: &Self) -> Option<Self> {
        // Buddies must have the same order
        if self.order.0 != buddy.order.0 {
            return None;
        }

        // Check that they are actually buddies
        if self.buddy_addr().as_u64() != buddy.addr.as_u64() {
            return None;
        }

        // Get the next order
        match self.order.next() {
            Some(new_order) => {
                // Use the lower address as the merged block's address
                let addr = if self.addr.as_u64() < buddy.addr.as_u64() {
                    self.addr
                } else {
                    buddy.addr
                };
                Some(Self {
                    addr,
                    order: new_order,
                })
            }
            None => None,
        }
    }
}

impl fmt::Debug for PageFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PageFrame")
            .field("addr", &self.addr)
            .field("order", &self.order.0)
            .field("pages", &self.pages())
            .finish()
    }
}

impl fmt::Display for PageFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PageFrame[{} - {}, {} pages]",
            self.addr,
            self.end_addr(),
            self.pages()
        )
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::format;
    use super::*;

    #[test]
    fn test_page_order_new() {
        assert!(PageOrder::new(0).is_some());
        assert!(PageOrder::new(9).is_some());
        assert!(PageOrder::new(10).is_none());
        assert!(PageOrder::new(100).is_none());
    }

    #[test]
    fn test_page_order_pages() {
        assert_eq!(PageOrder::new(0).unwrap().pages(), 1);
        assert_eq!(PageOrder::new(1).unwrap().pages(), 2);
        assert_eq!(PageOrder::new(2).unwrap().pages(), 4);
        assert_eq!(PageOrder::new(9).unwrap().pages(), 512);
    }

    #[test]
    fn test_page_order_bytes() {
        assert_eq!(PageOrder::new(0).unwrap().bytes(), 4096);
        assert_eq!(PageOrder::new(1).unwrap().bytes(), 8192);
        assert_eq!(PageOrder::new(9).unwrap().bytes(), 2 * 1024 * 1024);
    }

    #[test]
    fn test_page_order_next_prev() {
        let order = PageOrder::new(5).unwrap();
        assert_eq!(order.next().map(|o| o.as_usize()), Some(6));
        assert_eq!(order.prev().map(|o| o.as_usize()), Some(4));

        assert!(PageOrder::MIN.prev().is_none());
        assert!(PageOrder::MAX.next().is_none());
    }

    #[test]
    fn test_page_order_from_pages() {
        assert_eq!(PageOrder::from_pages(0), None);
        assert_eq!(PageOrder::from_pages(1).map(|o| o.as_usize()), Some(0));
        assert_eq!(PageOrder::from_pages(2).map(|o| o.as_usize()), Some(1));
        assert_eq!(PageOrder::from_pages(3).map(|o| o.as_usize()), Some(2));
        assert_eq!(PageOrder::from_pages(4).map(|o| o.as_usize()), Some(2));
        assert_eq!(PageOrder::from_pages(512).map(|o| o.as_usize()), Some(9));
        assert!(PageOrder::from_pages(513).is_none());
    }

    #[test]
    fn test_page_order_from_bytes() {
        assert_eq!(PageOrder::from_bytes(0), None);
        assert_eq!(PageOrder::from_bytes(1).map(|o| o.as_usize()), Some(0));
        assert_eq!(PageOrder::from_bytes(4096).map(|o| o.as_usize()), Some(0));
        assert_eq!(PageOrder::from_bytes(4097).map(|o| o.as_usize()), Some(1));
        assert_eq!(PageOrder::from_bytes(8192).map(|o| o.as_usize()), Some(1));
    }

    #[test]
    fn test_page_frame_new() {
        let frame = PageFrame::new(
            PhysAddr::new(0x1000_0000),
            PageOrder::new(3).unwrap(),
        );
        assert_eq!(frame.addr().as_u64(), 0x1000_0000);
        assert_eq!(frame.order().as_usize(), 3);
        assert_eq!(frame.pages(), 8);
        assert_eq!(frame.bytes(), 8 * 4096);
    }

    #[test]
    fn test_page_frame_single_page() {
        let frame = PageFrame::single_page(PhysAddr::new(0x5000));
        assert_eq!(frame.pages(), 1);
        assert_eq!(frame.order().as_usize(), 0);
    }

    #[test]
    fn test_page_frame_end_addr() {
        let frame = PageFrame::new(
            PhysAddr::new(0x1000),
            PageOrder::new(2).unwrap(),
        );
        assert_eq!(frame.end_addr().as_u64(), 0x5000);
    }

    #[test]
    fn test_page_frame_contains() {
        let frame = PageFrame::new(
            PhysAddr::new(0x1000),
            PageOrder::new(2).unwrap(),
        );

        assert!(frame.contains(PhysAddr::new(0x1000)));
        assert!(frame.contains(PhysAddr::new(0x2000)));
        assert!(frame.contains(PhysAddr::new(0x4FFF)));
        assert!(!frame.contains(PhysAddr::new(0x5000)));
        assert!(!frame.contains(PhysAddr::new(0x0FFF)));
    }

    #[test]
    fn test_page_frame_split() {
        let frame = PageFrame::new(
            PhysAddr::new(0x4000),
            PageOrder::new(2).unwrap(),
        );

        let (left, right) = frame.split().unwrap();
        assert_eq!(left.order().as_usize(), 1);
        assert_eq!(right.order().as_usize(), 1);
        assert_eq!(left.addr().as_u64(), 0x4000);
        assert_eq!(right.addr().as_u64(), 0x6000);
        assert_eq!(left.pages(), 2);
        assert_eq!(right.pages(), 2);

        // Cannot split a single page
        let single = PageFrame::single_page(PhysAddr::new(0x1000));
        assert!(single.split().is_none());
    }

    #[test]
    fn test_page_frame_buddy_addr() {
        // Order 1 block at 0x2000 (2 pages = 0x2000 bytes)
        let frame = PageFrame::new(
            PhysAddr::new(0x2000),
            PageOrder::new(1).unwrap(),
        );
        // XOR with 0x2000 gives 0x0
        assert_eq!(frame.buddy_addr().as_u64(), 0x0);

        // Order 1 block at 0x0
        let frame = PageFrame::new(
            PhysAddr::new(0x0),
            PageOrder::new(1).unwrap(),
        );
        // XOR with 0x2000 gives 0x2000
        assert_eq!(frame.buddy_addr().as_u64(), 0x2000);

        // Order 2 block at 0x4000 (4 pages = 0x4000 bytes)
        let frame = PageFrame::new(
            PhysAddr::new(0x4000),
            PageOrder::new(2).unwrap(),
        );
        // XOR with 0x4000 gives 0x0
        assert_eq!(frame.buddy_addr().as_u64(), 0x0);
    }

    #[test]
    fn test_page_frame_merge() {
        let left = PageFrame::new(PhysAddr::new(0x0), PageOrder::new(1).unwrap());
        let right = PageFrame::new(PhysAddr::new(0x2000), PageOrder::new(1).unwrap());

        let merged = left.merge(&right).unwrap();
        assert_eq!(merged.order().as_usize(), 2);
        assert_eq!(merged.addr().as_u64(), 0x0);
        assert_eq!(merged.pages(), 4);

        // Verify merge works in both directions
        let merged2 = right.merge(&left).unwrap();
        assert_eq!(merged2.addr().as_u64(), 0x0);

        // Cannot merge non-buddies
        let non_buddy = PageFrame::new(PhysAddr::new(0x4000), PageOrder::new(1).unwrap());
        assert!(left.merge(&non_buddy).is_none());

        // Cannot merge different orders
        let diff_order = PageFrame::new(PhysAddr::new(0x2000), PageOrder::new(2).unwrap());
        assert!(left.merge(&diff_order).is_none());
    }

    #[test]
    fn test_page_frame_display() {
        let frame = PageFrame::new(
            PhysAddr::new(0x1000),
            PageOrder::new(2).unwrap(),
        );
        let s = format!("{frame}");
        assert!(s.contains("0x1000"));
        assert!(s.contains("0x5000"));
        assert!(s.contains("4 pages"));
    }
}
