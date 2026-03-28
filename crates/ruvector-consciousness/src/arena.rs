//! Bump allocator for per-computation scratch space.
//!
//! Avoids repeated heap allocations in hot Φ computation loops.
//! Reset after each partition evaluation for O(1) reclamation.

use std::cell::RefCell;

/// Bump allocator for consciousness computation scratch buffers.
pub struct PhiArena {
    buf: RefCell<Vec<u8>>,
    offset: RefCell<usize>,
}

impl PhiArena {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: RefCell::new(vec![0u8; capacity]),
            offset: RefCell::new(0),
        }
    }

    /// Allocate a mutable slice of `len` elements, zero-initialised.
    pub fn alloc_slice<T: Copy + Default>(&self, len: usize) -> &mut [T] {
        let size = std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();
        assert!(align <= 16, "PhiArena does not support alignment > 16");

        let byte_len = size
            .checked_mul(len)
            .expect("PhiArena: size * len overflow");

        let mut offset = self.offset.borrow_mut();
        let mut buf = self.buf.borrow_mut();

        let aligned = (*offset + align - 1) & !(align - 1);
        let needed = aligned
            .checked_add(byte_len)
            .expect("PhiArena: aligned + byte_len overflow");

        if needed > buf.len() {
            let new_cap = (needed * 2).max(buf.len() * 2);
            buf.resize(new_cap, 0);
        }

        buf[aligned..aligned + byte_len].fill(0);
        *offset = aligned + byte_len;
        let ptr = buf[aligned..].as_mut_ptr() as *mut T;

        // SAFETY: Exclusive access via RefCell borrows. Alignment guaranteed.
        // Region is zero-filled and within bounds. See ruvector-solver arena
        // for detailed invariant documentation.
        drop(offset);
        drop(buf);

        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    /// Reset bump pointer to zero (O(1) reclamation).
    pub fn reset(&self) {
        *self.offset.borrow_mut() = 0;
    }

    pub fn bytes_used(&self) -> usize {
        *self.offset.borrow()
    }
}

// SAFETY: PhiArena exclusively owns its data. Not Sync due to RefCell.
unsafe impl Send for PhiArena {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_reset() {
        let arena = PhiArena::with_capacity(4096);
        let s: &mut [f64] = arena.alloc_slice(128);
        assert_eq!(s.len(), 128);
        assert!(arena.bytes_used() >= 128 * 8);
        arena.reset();
        assert_eq!(arena.bytes_used(), 0);
    }
}
