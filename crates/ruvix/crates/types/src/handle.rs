//! Generic handle type for kernel objects.
//!
//! All kernel objects are accessed through handles which contain a unique
//! identifier and generation counter for use-after-free detection.

/// A generic handle to a kernel object.
///
/// Handles are unforgeable kernel-managed tokens that identify resources.
/// The generation counter prevents use-after-free attacks when handles
/// are recycled after object destruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Handle {
    /// Unique identifier within the object pool.
    pub id: u32,
    /// Generation counter for stale handle detection.
    /// Incremented each time the slot is reused.
    pub generation: u32,
}

impl Handle {
    /// Creates a new handle with the given id and generation.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self { id, generation }
    }

    /// Creates a null handle (invalid).
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self {
            id: u32::MAX,
            generation: 0,
        }
    }

    /// Checks if this handle is null (invalid).
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.id == u32::MAX
    }

    /// Returns the raw 64-bit representation of this handle.
    #[inline]
    #[must_use]
    pub const fn to_raw(&self) -> u64 {
        ((self.generation as u64) << 32) | (self.id as u64)
    }

    /// Creates a handle from a raw 64-bit representation.
    #[inline]
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self {
            id: raw as u32,
            generation: (raw >> 32) as u32,
        }
    }
}

impl Default for Handle {
    fn default() -> Self {
        Self::null()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_new() {
        let h = Handle::new(42, 7);
        assert_eq!(h.id, 42);
        assert_eq!(h.generation, 7);
    }

    #[test]
    fn test_handle_null() {
        let h = Handle::null();
        assert!(h.is_null());
    }

    #[test]
    fn test_handle_roundtrip() {
        let h = Handle::new(12345, 67890);
        let raw = h.to_raw();
        let h2 = Handle::from_raw(raw);
        assert_eq!(h, h2);
    }
}
