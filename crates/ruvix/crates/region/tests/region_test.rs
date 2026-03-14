//! Comprehensive integration tests for ruvix-region.
//!
//! Tests all three region policies: Immutable, AppendOnly, and Slab.
//! Also tests memory backing implementations.

use ruvix_region::{
    append_only::AppendOnlyRegion,
    backing::{MemoryBacking, StaticBacking},
    immutable::ImmutableRegion,
    slab::{SlabAllocator, SlabRegion, SlotHandle},
};
use ruvix_types::{KernelError, RegionHandle};

// ============================================================================
// Immutable Region Tests
// ============================================================================

mod immutable_tests {
    use super::*;

    #[test]
    fn test_immutable_region_creation() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let data = b"Test data for immutable region";

        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        assert_eq!(region.len(), data.len());
        assert!(!region.is_empty());
        assert_eq!(region.handle(), handle);
    }

    #[test]
    fn test_immutable_region_empty() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);

        let region = ImmutableRegion::new(backing, &[], handle).unwrap();

        assert!(region.is_empty());
        assert_eq!(region.len(), 0);
        assert_eq!(region.as_slice(), &[]);
    }

    #[test]
    fn test_immutable_region_as_slice() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let data = b"Slice access test";

        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        assert_eq!(region.as_slice(), data);
    }

    #[test]
    fn test_immutable_region_read() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let data = b"Read test data";

        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        let mut buf = [0u8; 64];
        let read_len = region.read(0, &mut buf).unwrap();

        assert_eq!(read_len, data.len());
        assert_eq!(&buf[..read_len], data);
    }

    #[test]
    fn test_immutable_region_read_with_offset() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let data = b"Hello World!";

        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        let mut buf = [0u8; 64];
        let read_len = region.read(6, &mut buf).unwrap();

        assert_eq!(read_len, 6); // "World!"
        assert_eq!(&buf[..read_len], b"World!");
    }

    #[test]
    fn test_immutable_region_read_out_of_bounds() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let data = b"Short";

        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        let mut buf = [0u8; 64];
        // Reading at data.len() should fail
        let result = region.read(data.len(), &mut buf);
        assert!(matches!(result, Err(KernelError::InvalidArgument)));
    }

    #[test]
    fn test_immutable_region_content_hash() {
        let backing1 = StaticBacking::<1024>::new();
        let backing2 = StaticBacking::<1024>::new();
        let backing3 = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let data = b"Identical content";

        let region1 = ImmutableRegion::new(backing1, data, handle).unwrap();
        let region2 = ImmutableRegion::new(backing2, data, handle).unwrap();
        let region3 = ImmutableRegion::new(backing3, b"Different", handle).unwrap();

        // Same content should have same hash
        assert!(region1.content_equals(&region2));

        // Different content should have different hash
        assert!(!region1.content_equals(&region3));
    }

    #[test]
    fn test_immutable_region_get_byte() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let data = b"ABCDE";

        let region = ImmutableRegion::new(backing, data, handle).unwrap();

        assert_eq!(region.get(0), Some(b'A'));
        assert_eq!(region.get(4), Some(b'E'));
        assert_eq!(region.get(5), None);
    }

    #[test]
    fn test_immutable_region_read_u64() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);

        let mut data = [0u8; 16];
        data[0..8].copy_from_slice(&0x123456789ABCDEFu64.to_le_bytes());
        data[8..12].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());

        let region = ImmutableRegion::new(backing, &data, handle).unwrap();

        assert_eq!(region.read_u64(0).unwrap(), 0x123456789ABCDEF);
        assert_eq!(region.read_u32(8).unwrap(), 0xDEADBEEF);
    }
}

// ============================================================================
// Append-Only Region Tests
// ============================================================================

mod append_only_tests {
    use super::*;

    #[test]
    fn test_append_only_creation() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);

        let region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        assert!(region.is_empty());
        assert_eq!(region.max_size(), 256);
        assert_eq!(region.remaining(), 256);
        assert!(!region.is_full());
    }

    #[test]
    fn test_append_only_zero_size_error() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);

        let result = AppendOnlyRegion::new(backing, 0, handle);
        assert!(matches!(result, Err(KernelError::InvalidArgument)));
    }

    #[test]
    fn test_append_only_single_append() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        let data = b"First append";
        let offset = region.append(data).unwrap();

        assert_eq!(offset, 0);
        assert_eq!(region.len(), data.len());
        assert!(!region.is_empty());
    }

    #[test]
    fn test_append_only_multiple_appends() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        let data1 = b"First";
        let data2 = b"Second";
        let data3 = b"Third";

        let off1 = region.append(data1).unwrap();
        let off2 = region.append(data2).unwrap();
        let off3 = region.append(data3).unwrap();

        assert_eq!(off1, 0);
        assert_eq!(off2, data1.len());
        assert_eq!(off3, data1.len() + data2.len());

        // Verify as_slice returns concatenation
        assert_eq!(region.as_slice(), b"FirstSecondThird");
    }

    #[test]
    fn test_append_only_empty_append() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        // Empty append should return current cursor
        let offset = region.append(&[]).unwrap();
        assert_eq!(offset, 0);
        assert!(region.is_empty());
    }

    #[test]
    fn test_append_only_fill_to_capacity() {
        let backing = StaticBacking::<128>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 64, handle).unwrap();

        let data = [0xABu8; 64];
        region.append(&data).unwrap();

        assert!(region.is_full());
        assert_eq!(region.remaining(), 0);
    }

    #[test]
    fn test_append_only_overflow_error() {
        let backing = StaticBacking::<128>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 64, handle).unwrap();

        region.append(&[0u8; 32]).unwrap();

        // This should fail - not enough space
        let result = region.append(&[0u8; 48]);
        assert!(matches!(result, Err(KernelError::RegionFull)));
    }

    #[test]
    fn test_append_only_read() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        let data = b"Test data for reading";
        region.append(data).unwrap();

        let mut buf = [0u8; 64];
        let read_len = region.read(0, &mut buf).unwrap();

        assert_eq!(read_len, data.len());
        assert_eq!(&buf[..read_len], data);
    }

    #[test]
    fn test_append_only_read_with_offset() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        region.append(b"Hello World").unwrap();

        let mut buf = [0u8; 64];
        let read_len = region.read(6, &mut buf).unwrap();

        assert_eq!(read_len, 5); // "World"
        assert_eq!(&buf[..read_len], b"World");
    }

    #[test]
    fn test_append_only_read_beyond_cursor() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        region.append(&[0u8; 10]).unwrap();

        let mut buf = [0u8; 10];
        // Reading at cursor position should fail
        assert!(region.read(10, &mut buf).is_err());
        // Reading beyond cursor should fail
        assert!(region.read(20, &mut buf).is_err());
    }

    #[test]
    fn test_append_only_fill_ratio() {
        let backing = StaticBacking::<256>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 100, handle).unwrap();

        assert!((region.fill_ratio() - 0.0).abs() < 0.01);

        region.append(&[0u8; 25]).unwrap();
        assert!((region.fill_ratio() - 0.25).abs() < 0.01);

        region.append(&[0u8; 25]).unwrap();
        assert!((region.fill_ratio() - 0.50).abs() < 0.01);

        region.append(&[0u8; 50]).unwrap();
        assert!((region.fill_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_append_only_u64_operations() {
        let backing = StaticBacking::<256>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 100, handle).unwrap();

        region.append_u64(0x123456789ABCDEF0).unwrap();
        region.append_u32(0xDEADBEEF).unwrap();

        assert_eq!(region.read_u64(0).unwrap(), 0x123456789ABCDEF0);
        assert_eq!(region.read_u32(8).unwrap(), 0xDEADBEEF);
    }

    #[test]
    fn test_append_only_read_all() {
        let backing = StaticBacking::<1024>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = AppendOnlyRegion::new(backing, 256, handle).unwrap();

        let data = b"Complete data";
        region.append(data).unwrap();

        let mut buf = [0u8; 64];
        let read_len = region.read_all(&mut buf).unwrap();

        assert_eq!(read_len, data.len());
        assert_eq!(&buf[..read_len], data);
    }
}

// ============================================================================
// Slab Allocator Tests
// ============================================================================

mod slab_tests {
    use super::*;

    #[test]
    fn test_slab_creation() {
        let backing = StaticBacking::<4096>::new();
        let slab = SlabAllocator::new(backing, 64, 16).unwrap();

        assert_eq!(slab.slot_size(), 64);
        assert_eq!(slab.slot_count(), 16);
        assert_eq!(slab.allocated_count(), 0);
        assert_eq!(slab.free_count(), 16);
        assert!(!slab.is_full());
        assert!(slab.is_empty());
    }

    #[test]
    fn test_slab_invalid_params() {
        let backing1 = StaticBacking::<4096>::new();
        let backing2 = StaticBacking::<4096>::new();

        // Zero slot size
        let result = SlabAllocator::new(backing1, 0, 16);
        assert!(matches!(result, Err(KernelError::InvalidArgument)));

        // Zero slot count
        let result = SlabAllocator::new(backing2, 64, 0);
        assert!(matches!(result, Err(KernelError::InvalidArgument)));
    }

    #[test]
    fn test_slab_basic_alloc_free() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let handle = slab.alloc().unwrap();
        assert!(!handle.is_invalid());
        assert_eq!(slab.allocated_count(), 1);
        assert_eq!(slab.free_count(), 15);

        slab.free(handle).unwrap();
        assert_eq!(slab.allocated_count(), 0);
        assert_eq!(slab.free_count(), 16);
    }

    #[test]
    fn test_slab_multiple_allocs() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 8).unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            handles.push(slab.alloc().unwrap());
        }

        assert!(slab.is_full());
        assert_eq!(slab.allocated_count(), 8);

        // All handles should be unique
        for i in 0..handles.len() {
            for j in (i + 1)..handles.len() {
                assert_ne!(handles[i].index, handles[j].index);
            }
        }
    }

    #[test]
    fn test_slab_alloc_when_full() {
        let backing = StaticBacking::<512>::new();
        let mut slab = SlabAllocator::new(backing, 64, 4).unwrap();

        // Allocate all slots
        for _ in 0..4 {
            slab.alloc().unwrap();
        }

        // Should fail when full
        let result = slab.alloc();
        assert!(matches!(result, Err(KernelError::SlabFull)));
    }

    #[test]
    fn test_slab_reuse_after_free() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 4).unwrap();

        // Allocate all
        let mut handles = Vec::new();
        for _ in 0..4 {
            handles.push(slab.alloc().unwrap());
        }

        // Free one
        slab.free(handles[2]).unwrap();

        // Allocate should succeed
        let new_handle = slab.alloc().unwrap();
        assert_eq!(new_handle.index, handles[2].index);
    }

    #[test]
    fn test_slab_generation_counter() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let handle1 = slab.alloc().unwrap();
        slab.free(handle1).unwrap();

        // Old handle should be invalid
        assert!(slab.free(handle1).is_err());

        // New allocation at same slot has different generation
        let handle2 = slab.alloc().unwrap();
        assert_eq!(handle1.index, handle2.index);
        assert_ne!(handle1.generation, handle2.generation);
    }

    #[test]
    fn test_slab_read_write() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let handle = slab.alloc().unwrap();

        let data = b"Hello, RuVix Slab!";
        let written = slab.write(handle, data).unwrap();
        assert_eq!(written, data.len());

        let mut buf = [0u8; 64];
        let read = slab.read(handle, &mut buf).unwrap();
        assert_eq!(read, 64);
        assert_eq!(&buf[..data.len()], data);
    }

    #[test]
    fn test_slab_write_too_large() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 32, 16).unwrap();

        let handle = slab.alloc().unwrap();

        let large_data = [0u8; 64];
        let result = slab.write(handle, &large_data);
        assert!(matches!(result, Err(KernelError::BufferTooSmall)));
    }

    #[test]
    fn test_slab_zero_on_free() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let handle = slab.alloc().unwrap();
        slab.write(handle, &[0xAB; 64]).unwrap();
        slab.free(handle).unwrap();

        // Reallocate same slot
        let handle2 = slab.alloc().unwrap();

        // Should be zeroed
        let mut buf = [0xFF; 64];
        slab.read(handle2, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_slab_stale_handle_operations() {
        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let handle = slab.alloc().unwrap();
        slab.free(handle).unwrap();

        // All operations with stale handle should fail
        let mut buf = [0u8; 64];
        assert!(slab.read(handle, &mut buf).is_err());
        assert!(slab.write(handle, &[0u8; 64]).is_err());
        assert!(slab.slot_ptr(handle).is_err());
    }

    #[test]
    fn test_slab_invalid_handle() {
        let backing = StaticBacking::<4096>::new();
        let slab = SlabAllocator::new(backing, 64, 16).unwrap();

        let invalid = SlotHandle::invalid();
        let mut buf = [0u8; 64];

        assert!(slab.read(invalid, &mut buf).is_err());
    }

    #[test]
    fn test_slot_handle_default() {
        let handle = SlotHandle::default();
        assert!(handle.is_invalid());
    }
}

// ============================================================================
// Slab Region Tests
// ============================================================================

mod slab_region_tests {
    use super::*;

    #[test]
    fn test_slab_region_creation() {
        let backing = StaticBacking::<4096>::new();
        let handle = RegionHandle::new(1, 0);
        let region = SlabRegion::new(backing, 64, 16, handle).unwrap();

        assert_eq!(region.handle(), handle);
        assert_eq!(region.slot_size(), 64);
        assert_eq!(region.slot_count(), 16);
        assert_eq!(region.allocated_count(), 0);
    }

    #[test]
    fn test_slab_region_alloc_free() {
        let backing = StaticBacking::<4096>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = SlabRegion::new(backing, 64, 16, handle).unwrap();

        let slot = region.alloc().unwrap();
        assert_eq!(region.allocated_count(), 1);

        region.free(slot).unwrap();
        assert_eq!(region.allocated_count(), 0);
    }

    #[test]
    fn test_slab_region_read_write() {
        let backing = StaticBacking::<4096>::new();
        let handle = RegionHandle::new(1, 0);
        let mut region = SlabRegion::new(backing, 64, 16, handle).unwrap();

        let slot = region.alloc().unwrap();

        let data = b"Region slot data";
        region.write(slot, data).unwrap();

        let mut buf = [0u8; 64];
        region.read(slot, &mut buf).unwrap();

        assert_eq!(&buf[..data.len()], data);
    }
}

// ============================================================================
// Memory Backing Tests
// ============================================================================

mod backing_tests {
    use super::*;

    #[test]
    fn test_static_backing_creation() {
        let backing = StaticBacking::<1024>::new();
        assert_eq!(backing.capacity(), 1024);
        assert_eq!(backing.allocated(), 0);
    }

    #[test]
    fn test_static_backing_allocation() {
        let mut backing = StaticBacking::<1024>::new();

        let (ptr1, size1) = backing.allocate(100).unwrap();
        assert!(!ptr1.is_null());
        assert!(size1 >= 100);
        assert!(backing.allocated() >= 100);

        let (ptr2, size2) = backing.allocate(200).unwrap();
        assert!(!ptr2.is_null());
        assert!(size2 >= 200);
        assert!(ptr2 > ptr1);
    }

    #[test]
    fn test_static_backing_out_of_memory() {
        let mut backing = StaticBacking::<256>::new();

        backing.allocate(200).unwrap();

        // Should fail - not enough space
        let result = backing.allocate(100);
        assert!(matches!(result, Err(KernelError::OutOfMemory)));
    }

    #[test]
    fn test_static_backing_alignment() {
        let mut backing = StaticBacking::<1024>::new();

        // Allocate odd sizes and verify 8-byte alignment
        let (ptr1, _) = backing.allocate(13).unwrap();
        assert_eq!(ptr1 as usize % 8, 0);

        let (ptr2, _) = backing.allocate(7).unwrap();
        assert_eq!(ptr2 as usize % 8, 0);
    }

    #[cfg(feature = "std")]
    mod heap_backing_tests {
        use ruvix_region::backing::{HeapBacking, MemoryBacking};
        use ruvix_types::KernelError;

        #[test]
        fn test_heap_backing_allocation() {
            let mut backing = HeapBacking::new(4096);

            let (ptr, size) = backing.allocate(100).unwrap();
            assert!(!ptr.is_null());
            assert!(size >= 100);

            unsafe {
                backing.deallocate(ptr, size).unwrap();
            }
        }

        #[test]
        fn test_heap_backing_out_of_memory() {
            let mut backing = HeapBacking::new(256);

            backing.allocate(200).unwrap();

            let result = backing.allocate(100);
            assert!(matches!(result, Err(KernelError::OutOfMemory)));
        }
    }
}

// ============================================================================
// Cross-Region Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_multiple_regions_same_backing_type() {
        // Create multiple regions of different types
        let backing1 = StaticBacking::<1024>::new();
        let backing2 = StaticBacking::<1024>::new();
        let backing3 = StaticBacking::<2048>::new();

        let immutable = ImmutableRegion::new(
            backing1,
            b"Immutable data",
            RegionHandle::new(1, 0),
        )
        .unwrap();

        let mut append_only =
            AppendOnlyRegion::new(backing2, 256, RegionHandle::new(2, 0)).unwrap();

        let mut slab = SlabRegion::new(backing3, 64, 16, RegionHandle::new(3, 0)).unwrap();

        // Use all regions
        assert_eq!(immutable.as_slice(), b"Immutable data");

        append_only.append(b"Appended").unwrap();
        assert_eq!(append_only.as_slice(), b"Appended");

        let slot = slab.alloc().unwrap();
        slab.write(slot, b"Slab data").unwrap();
        let mut buf = [0u8; 64];
        slab.read(slot, &mut buf).unwrap();
        assert_eq!(&buf[..9], b"Slab data");
    }

    #[test]
    fn test_region_handles_uniqueness() {
        let handle1 = RegionHandle::new(1, 0);
        let handle2 = RegionHandle::new(2, 0);
        let handle3 = RegionHandle::new(1, 1);

        assert_ne!(handle1.raw().to_raw(), handle2.raw().to_raw());
        assert_ne!(handle1.raw().to_raw(), handle3.raw().to_raw());
    }

    #[test]
    fn test_witness_log_simulation() {
        // Simulate witness log using append-only region
        let backing = StaticBacking::<4096>::new();
        let mut log = AppendOnlyRegion::new(backing, 1024, RegionHandle::new(1, 0)).unwrap();

        // Append multiple log entries
        for i in 0u64..10 {
            let offset = log.append_u64(i).unwrap();
            assert_eq!(offset, i as usize * 8);
        }

        // Verify all entries
        for i in 0u64..10 {
            assert_eq!(log.read_u64(i as usize * 8).unwrap(), i);
        }
    }

    #[test]
    fn test_task_control_blocks_simulation() {
        // Simulate task control blocks using slab allocator
        #[repr(C)]
        struct TaskControlBlock {
            id: u32,
            priority: u8,
            state: u8,
            _padding: [u8; 2],
        }

        let backing = StaticBacking::<4096>::new();
        let mut slab = SlabAllocator::new(
            backing,
            core::mem::size_of::<TaskControlBlock>(),
            16,
        )
        .unwrap();

        let mut tasks = Vec::new();

        // Create tasks
        for i in 0..8 {
            let handle = slab.alloc().unwrap();
            let tcb = TaskControlBlock {
                id: i,
                priority: (i % 4) as u8,
                state: 1,
                _padding: [0; 2],
            };

            let bytes = unsafe {
                core::slice::from_raw_parts(
                    &tcb as *const TaskControlBlock as *const u8,
                    core::mem::size_of::<TaskControlBlock>(),
                )
            };
            slab.write(handle, bytes).unwrap();
            tasks.push(handle);
        }

        // Verify tasks
        for (i, &handle) in tasks.iter().enumerate() {
            let mut buf = [0u8; 8];
            slab.read(handle, &mut buf).unwrap();

            let id = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
            assert_eq!(id, i as u32);
        }
    }
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn append_only_never_loses_data(
            chunks in prop::collection::vec(prop::collection::vec(0u8..=255, 1..32), 1..10)
        ) {
            let total_size: usize = chunks.iter().map(|c| c.len()).sum();
            if total_size > 1024 {
                return Ok(());
            }

            let backing = StaticBacking::<2048>::new();
            let mut region = AppendOnlyRegion::new(
                backing,
                1024,
                RegionHandle::new(1, 0)
            ).unwrap();

            let mut expected: Vec<u8> = Vec::new();
            for chunk in &chunks {
                region.append(chunk).unwrap();
                expected.extend(chunk);
            }

            assert_eq!(region.as_slice(), expected.as_slice());
        }

        #[test]
        fn slab_alloc_free_consistent(ops in prop::collection::vec(0u8..=1, 1..100)) {
            let backing = StaticBacking::<4096>::new();
            let mut slab = SlabAllocator::new(backing, 64, 32).unwrap();
            let mut active_handles = Vec::new();

            for op in ops {
                if op == 0 && !active_handles.is_empty() {
                    // Free random handle
                    let idx = active_handles.len() - 1;
                    let handle = active_handles.remove(idx);
                    slab.free(handle).unwrap();
                } else if op == 1 && slab.free_count() > 0 {
                    // Allocate
                    let handle = slab.alloc().unwrap();
                    active_handles.push(handle);
                }
            }

            assert_eq!(slab.allocated_count(), active_handles.len());
        }

        #[test]
        fn immutable_region_hash_deterministic(data in prop::collection::vec(0u8..=255, 0..256)) {
            let backing1 = StaticBacking::<512>::new();
            let backing2 = StaticBacking::<512>::new();

            let region1 = ImmutableRegion::new(
                backing1,
                &data,
                RegionHandle::new(1, 0)
            ).unwrap();
            let region2 = ImmutableRegion::new(
                backing2,
                &data,
                RegionHandle::new(1, 0)
            ).unwrap();

            assert!(region1.content_equals(&region2));
        }
    }
}
