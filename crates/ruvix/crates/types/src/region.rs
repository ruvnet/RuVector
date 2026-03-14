//! Memory region types.
//!
//! RuVix replaces virtual memory with regions. A region is a contiguous,
//! capability-protected memory object with one of three policies:
//! Immutable, AppendOnly, or Slab.

use crate::handle::Handle;

/// Handle to a memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct RegionHandle(pub Handle);

impl RegionHandle {
    /// Creates a new region handle.
    #[inline]
    #[must_use]
    pub const fn new(id: u32, generation: u32) -> Self {
        Self(Handle::new(id, generation))
    }

    /// Creates a null (invalid) region handle.
    #[inline]
    #[must_use]
    pub const fn null() -> Self {
        Self(Handle::null())
    }

    /// Checks if this handle is null.
    #[inline]
    #[must_use]
    pub const fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns the raw handle.
    #[inline]
    #[must_use]
    pub const fn raw(&self) -> Handle {
        self.0
    }
}

impl Default for RegionHandle {
    fn default() -> Self {
        Self::null()
    }
}

/// Memory region access policy.
///
/// RuVix does not implement demand paging. All regions are physically backed
/// at `region_map` time. This eliminates page faults, swap, and copy-on-write
/// complexity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionPolicy {
    /// Contents are set once at creation and never modified.
    ///
    /// The kernel may deduplicate identical immutable regions.
    /// Ideal for: RVF component code, trained model weights, lookup tables.
    Immutable,

    /// Contents can only be appended, never overwritten or truncated.
    ///
    /// A monotonic write cursor tracks the append position.
    /// Ideal for: witness logs, event streams, time-series vectors.
    AppendOnly {
        /// Maximum size in bytes before the region is considered full.
        max_size: usize,
    },

    /// Fixed-size slots allocated from a free list.
    ///
    /// Slots can be freed and reused. No fragmentation by construction.
    /// Ideal for: task control blocks, capability tables, queue ring buffers.
    Slab {
        /// Size of each slot in bytes.
        slot_size: usize,
        /// Total number of slots.
        slot_count: usize,
    },
}

impl RegionPolicy {
    /// Returns the policy name as a string.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Immutable => "Immutable",
            Self::AppendOnly { .. } => "AppendOnly",
            Self::Slab { .. } => "Slab",
        }
    }

    /// Returns true if this policy allows writes.
    #[inline]
    #[must_use]
    pub const fn is_writable(&self) -> bool {
        match self {
            Self::Immutable => false,
            Self::AppendOnly { .. } | Self::Slab { .. } => true,
        }
    }

    /// Returns true if this policy allows in-place modifications.
    ///
    /// Only Slab regions allow overwriting existing data.
    /// AppendOnly regions only permit appending.
    #[inline]
    #[must_use]
    pub const fn allows_overwrite(&self) -> bool {
        matches!(self, Self::Slab { .. })
    }

    /// Returns the total capacity in bytes.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> Option<usize> {
        match self {
            Self::Immutable => None,
            Self::AppendOnly { max_size } => Some(*max_size),
            Self::Slab {
                slot_size,
                slot_count,
            } => Some(*slot_size * *slot_count),
        }
    }

    /// Creates an immutable region policy.
    #[inline]
    #[must_use]
    pub const fn immutable() -> Self {
        Self::Immutable
    }

    /// Creates an append-only region policy.
    #[inline]
    #[must_use]
    pub const fn append_only(max_size: usize) -> Self {
        Self::AppendOnly { max_size }
    }

    /// Creates a slab region policy.
    #[inline]
    #[must_use]
    pub const fn slab(slot_size: usize, slot_count: usize) -> Self {
        Self::Slab {
            slot_size,
            slot_count,
        }
    }
}

impl Default for RegionPolicy {
    fn default() -> Self {
        Self::Immutable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_region_handle() {
        let h = RegionHandle::new(5, 10);
        assert!(!h.is_null());
        assert_eq!(h.raw().id, 5);
    }

    #[test]
    fn test_region_policy_immutable() {
        let policy = RegionPolicy::immutable();
        assert!(!policy.is_writable());
        assert!(!policy.allows_overwrite());
        assert!(policy.capacity().is_none());
    }

    #[test]
    fn test_region_policy_append_only() {
        let policy = RegionPolicy::append_only(1024);
        assert!(policy.is_writable());
        assert!(!policy.allows_overwrite());
        assert_eq!(policy.capacity(), Some(1024));
    }

    #[test]
    fn test_region_policy_slab() {
        let policy = RegionPolicy::slab(64, 100);
        assert!(policy.is_writable());
        assert!(policy.allows_overwrite());
        assert_eq!(policy.capacity(), Some(6400));
    }
}
