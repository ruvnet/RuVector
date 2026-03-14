//! Arena allocator for scratch allocations in hot paths (ADR-103 A8).
//!
//! Provides a simple bump allocator that avoids per-allocation heap overhead
//! for short-lived data such as line formatting buffers, grep result
//! accumulation, and glob result building.

const DEFAULT_CHUNK_SIZE: usize = 8 * 1024; // 8 KiB

/// Simple bump arena for temporary allocations in hot paths.
///
/// Allocations are carved out of pre-allocated chunks. When the current chunk
/// is exhausted a new one is allocated. [`reset`](Arena::reset) reclaims all
/// memory without deallocating the underlying chunks, making the arena
/// reusable across iterations of a hot loop.
pub struct Arena {
    chunks: Vec<Vec<u8>>,
    current: usize,
    offset: usize,
}

impl Arena {
    /// Create a new arena with the default chunk size (8 KiB).
    pub fn new() -> Self {
        Self {
            chunks: vec![vec![0u8; DEFAULT_CHUNK_SIZE]],
            current: 0,
            offset: 0,
        }
    }

    /// Create a new arena whose first chunk has at least `cap` bytes.
    pub fn with_capacity(cap: usize) -> Self {
        let cap = cap.max(64);
        Self {
            chunks: vec![vec![0u8; cap]],
            current: 0,
            offset: 0,
        }
    }

    /// Allocate `size` bytes from the arena, returning a mutable slice.
    ///
    /// The returned slice is zero-initialized only for the first use of a
    /// chunk; after [`reset`](Arena::reset) it may contain stale data.
    pub fn alloc(&mut self, size: usize) -> &mut [u8] {
        if size == 0 {
            return &mut [];
        }

        // Try to fit in the current chunk.
        if self.current < self.chunks.len() {
            let remaining = self.chunks[self.current].len() - self.offset;
            if size <= remaining {
                let start = self.offset;
                self.offset += size;
                return &mut self.chunks[self.current][start..start + size];
            }
        }

        // Move to the next existing chunk or allocate a new one.
        self.current += 1;
        self.offset = 0;

        if self.current < self.chunks.len() {
            // Reuse an existing chunk if it is large enough.
            if self.chunks[self.current].len() >= size {
                self.offset = size;
                return &mut self.chunks[self.current][..size];
            }
            // Existing chunk is too small — replace it.
            self.chunks[self.current] = vec![0u8; size.max(DEFAULT_CHUNK_SIZE)];
            self.offset = size;
            return &mut self.chunks[self.current][..size];
        }

        // Allocate a brand-new chunk.
        let chunk_size = size.max(DEFAULT_CHUNK_SIZE);
        self.chunks.push(vec![0u8; chunk_size]);
        self.offset = size;
        &mut self.chunks[self.current][..size]
    }

    /// Allocate a copy of the string `s` inside the arena and return a `&str`
    /// reference with the arena's lifetime.
    ///
    /// This is useful for interning short strings during hot-path processing
    /// without going through the global allocator.
    pub fn alloc_str(&mut self, s: &str) -> &str {
        let bytes = self.alloc(s.len());
        bytes.copy_from_slice(s.as_bytes());
        // SAFETY: we just copied valid UTF-8 bytes.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    /// Reset the arena so all previously allocated memory can be reused.
    ///
    /// This does **not** deallocate the underlying chunks — it simply resets
    /// the bump pointer to the beginning, making the next series of
    /// allocations reuse existing memory.
    pub fn reset(&mut self) {
        self.current = 0;
        self.offset = 0;
    }

    /// Total bytes currently in use (allocated but not yet reset).
    pub fn bytes_used(&self) -> usize {
        if self.chunks.is_empty() {
            return 0;
        }
        let full_chunks: usize = self.chunks[..self.current]
            .iter()
            .map(|c| c.len())
            .sum();
        full_chunks + self.offset
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_default() {
        let a = Arena::new();
        assert_eq!(a.bytes_used(), 0);
        assert_eq!(a.chunks.len(), 1);
    }

    #[test]
    fn test_with_capacity() {
        let a = Arena::with_capacity(1024);
        assert!(a.chunks[0].len() >= 1024);
    }
}
