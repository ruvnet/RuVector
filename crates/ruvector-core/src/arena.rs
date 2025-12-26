//! Arena allocator for batch operations
//!
//! This module provides arena-based memory allocation to reduce allocation
//! overhead in hot paths and improve memory locality.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::RefCell;
use std::ptr;

/// Arena allocator for temporary allocations
///
/// Use this for batch operations where many temporary allocations
/// are needed and can be freed all at once.
pub struct Arena {
    chunks: RefCell<Vec<Chunk>>,
    chunk_size: usize,
}

struct Chunk {
    data: *mut u8,
    capacity: usize,
    used: usize,
}

impl Arena {
    /// Create a new arena with the specified chunk size
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: RefCell::new(Vec::new()),
            chunk_size,
        }
    }

    /// Create an arena with a default 1MB chunk size
    pub fn with_default_chunk_size() -> Self {
        Self::new(1024 * 1024) // 1MB
    }

    /// Allocate a buffer of the specified size
    pub fn alloc_vec<T>(&self, count: usize) -> ArenaVec<T> {
        let size = count * std::mem::size_of::<T>();
        let align = std::mem::align_of::<T>();

        let ptr = self.alloc_raw(size, align);

        ArenaVec {
            ptr: ptr as *mut T,
            len: 0,
            capacity: count,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Allocate raw bytes with specified alignment
    fn alloc_raw(&self, size: usize, align: usize) -> *mut u8 {
        // SECURITY: Validate alignment is a power of 2 and size is reasonable
        assert!(
            align > 0 && align.is_power_of_two(),
            "Alignment must be a power of 2"
        );
        assert!(size > 0, "Cannot allocate zero bytes");
        assert!(size <= isize::MAX as usize, "Allocation size too large");

        let mut chunks = self.chunks.borrow_mut();

        // Try to allocate from the last chunk
        if let Some(chunk) = chunks.last_mut() {
            // Align the current position
            let current = chunk.used;
            let aligned = (current + align - 1) & !(align - 1);

            // SECURITY: Check for overflow in alignment calculation
            if aligned < current {
                panic!("Alignment calculation overflow");
            }

            let needed = aligned
                .checked_add(size)
                .expect("Arena allocation size overflow");

            if needed <= chunk.capacity {
                chunk.used = needed;
                return unsafe {
                    // SECURITY: Verify pointer arithmetic doesn't overflow
                    let ptr = chunk.data.add(aligned);
                    debug_assert!(ptr as usize >= chunk.data as usize, "Pointer underflow");
                    ptr
                };
            }
        }

        // Need a new chunk
        let chunk_size = self.chunk_size.max(size + align);
        let layout = Layout::from_size_align(chunk_size, 64).unwrap();
        let data = unsafe { alloc(layout) };

        let aligned = align;
        let chunk = Chunk {
            data,
            capacity: chunk_size,
            used: aligned + size,
        };

        let ptr = unsafe { data.add(aligned) };
        chunks.push(chunk);

        ptr
    }

    /// Reset the arena, allowing reuse of allocated memory
    pub fn reset(&self) {
        let mut chunks = self.chunks.borrow_mut();
        for chunk in chunks.iter_mut() {
            chunk.used = 0;
        }
    }

    /// Get total allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        let chunks = self.chunks.borrow();
        chunks.iter().map(|c| c.capacity).sum()
    }

    /// Get used bytes
    pub fn used_bytes(&self) -> usize {
        let chunks = self.chunks.borrow();
        chunks.iter().map(|c| c.used).sum()
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let chunks = self.chunks.borrow();
        for chunk in chunks.iter() {
            let layout = Layout::from_size_align(chunk.capacity, 64).unwrap();
            unsafe {
                dealloc(chunk.data, layout);
            }
        }
    }
}

/// Vector allocated from an arena
pub struct ArenaVec<T> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ArenaVec<T> {
    /// Push an element (panics if capacity exceeded)
    pub fn push(&mut self, value: T) {
        // SECURITY: Bounds check before pointer arithmetic
        assert!(self.len < self.capacity, "ArenaVec capacity exceeded");
        assert!(!self.ptr.is_null(), "ArenaVec pointer is null");

        unsafe {
            // Additional safety: verify the pointer offset is within bounds
            let offset_ptr = self.ptr.add(self.len);
            debug_assert!(
                offset_ptr as usize >= self.ptr as usize,
                "Pointer arithmetic overflow"
            );
            ptr::write(offset_ptr, value);
        }
        self.len += 1;
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        // SECURITY: Bounds check before creating slice
        assert!(self.len <= self.capacity, "Length exceeds capacity");
        assert!(!self.ptr.is_null(), "Cannot create slice from null pointer");

        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SECURITY: Bounds check before creating slice
        assert!(self.len <= self.capacity, "Length exceeds capacity");
        assert!(!self.ptr.is_null(), "Cannot create slice from null pointer");

        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T> std::ops::Deref for ArenaVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> std::ops::DerefMut for ArenaVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// Thread-local arena for per-thread allocations
thread_local! {
    static THREAD_ARENA: RefCell<Arena> = RefCell::new(Arena::with_default_chunk_size());
}

/// Get the thread-local arena
/// Note: Commented out due to lifetime issues with RefCell::borrow() escaping closure
/// Use THREAD_ARENA.with(|arena| { ... }) directly instead
/*
pub fn thread_arena() -> impl std::ops::Deref<Target = Arena> {
    THREAD_ARENA.with(|arena| {
        arena.borrow()
    })
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_alloc() {
        let arena = Arena::new(1024);

        let mut vec1 = arena.alloc_vec::<f32>(10);
        vec1.push(1.0);
        vec1.push(2.0);
        vec1.push(3.0);

        assert_eq!(vec1.len(), 3);
        assert_eq!(vec1[0], 1.0);
        assert_eq!(vec1[1], 2.0);
        assert_eq!(vec1[2], 3.0);
    }

    #[test]
    fn test_arena_multiple_allocs() {
        let arena = Arena::new(1024);

        let vec1 = arena.alloc_vec::<u32>(100);
        let vec2 = arena.alloc_vec::<u64>(50);
        let vec3 = arena.alloc_vec::<f32>(200);

        assert_eq!(vec1.capacity(), 100);
        assert_eq!(vec2.capacity(), 50);
        assert_eq!(vec3.capacity(), 200);
    }

    #[test]
    fn test_arena_reset() {
        let arena = Arena::new(1024);

        {
            let _vec1 = arena.alloc_vec::<f32>(100);
            let _vec2 = arena.alloc_vec::<f32>(100);
        }

        let used_before = arena.used_bytes();
        arena.reset();
        let used_after = arena.used_bytes();

        assert!(used_after < used_before);
    }
}
