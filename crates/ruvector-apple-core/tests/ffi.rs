use std::ptr;

use ruvector_apple_core::{
    ruvector_apple_core_abi_version, ruvector_apple_core_bytes_data,
    ruvector_apple_core_bytes_free, ruvector_apple_core_index_create,
    ruvector_apple_core_index_destroy, ruvector_apple_core_index_info,
    ruvector_apple_core_index_remove, ruvector_apple_core_index_restore,
    ruvector_apple_core_index_search, ruvector_apple_core_index_snapshot,
    ruvector_apple_core_index_upsert, ruvector_apple_core_results_data,
    ruvector_apple_core_results_free, RvAppleIndex, RvIndexInfo, RvOwnedBytes, RvOwnedResults,
    RvSearchResult, RvStatus, RVECTOR_APPLE_CORE_ABI_VERSION,
};

#[test]
fn fixed_layout_and_version_are_stable() {
    assert_eq!(
        ruvector_apple_core_abi_version(),
        RVECTOR_APPLE_CORE_ABI_VERSION
    );
    assert_eq!(std::mem::size_of::<RvSearchResult>(), 16);
    assert_eq!(std::mem::align_of::<RvSearchResult>(), 8);
    assert_eq!(std::mem::size_of::<RvIndexInfo>(), 20);
}

#[test]
fn ffi_lifecycle_search_snapshot_restore_and_frees() {
    unsafe {
        let mut index: *mut RvAppleIndex = ptr::null_mut();
        assert_eq!(
            ruvector_apple_core_index_create(2, 4, 1, &mut index),
            RvStatus::Ok as i32
        );
        assert!(!index.is_null());
        assert_eq!(
            ruvector_apple_core_index_upsert(index, 1, [1.0_f32, 0.0].as_ptr(), 2),
            RvStatus::Ok as i32
        );
        assert_eq!(
            ruvector_apple_core_index_upsert(index, 2, [0.0_f32, 1.0].as_ptr(), 2),
            RvStatus::Ok as i32
        );

        let mut info = RvIndexInfo {
            abi_version: 0,
            dimensions: 0,
            capacity: 0,
            count: 0,
            metric: 0,
        };
        assert_eq!(
            ruvector_apple_core_index_info(index, &mut info),
            RvStatus::Ok as i32
        );
        assert_eq!(info.count, 2);
        assert_eq!(info.dimensions, 2);

        let mut results: *mut RvOwnedResults = ptr::null_mut();
        let mut results_len = 0;
        assert_eq!(
            ruvector_apple_core_index_search(index, [1.0_f32, 0.0].as_ptr(), 2, 2, &mut results,),
            RvStatus::Ok as i32
        );
        let mut result_values: *const RvSearchResult = ptr::null();
        assert_eq!(
            ruvector_apple_core_results_data(results, &mut result_values, &mut results_len),
            RvStatus::Ok as i32
        );
        assert_eq!(results_len, 2);
        assert_eq!((*result_values).id, 1);
        assert_eq!(
            ruvector_apple_core_results_free(results),
            RvStatus::Ok as i32
        );

        let mut bytes: *mut RvOwnedBytes = ptr::null_mut();
        let mut byte_values: *const u8 = ptr::null();
        let mut bytes_len = 0;
        assert_eq!(
            ruvector_apple_core_index_snapshot(index, &mut bytes),
            RvStatus::Ok as i32
        );
        assert_eq!(
            ruvector_apple_core_bytes_data(bytes, &mut byte_values, &mut bytes_len),
            RvStatus::Ok as i32
        );
        let mut restored: *mut RvAppleIndex = ptr::null_mut();
        assert_eq!(
            ruvector_apple_core_index_restore(byte_values, bytes_len, &mut restored),
            RvStatus::Ok as i32
        );
        assert_eq!(ruvector_apple_core_bytes_free(bytes), RvStatus::Ok as i32);
        assert_eq!(
            ruvector_apple_core_index_info(restored, &mut info),
            RvStatus::Ok as i32
        );
        assert_eq!(info.count, 2);

        let mut removed = 0;
        assert_eq!(
            ruvector_apple_core_index_remove(restored, 2, &mut removed),
            RvStatus::Ok as i32
        );
        assert_eq!(removed, 1);
        assert_eq!(
            ruvector_apple_core_index_destroy(restored),
            RvStatus::Ok as i32
        );
        assert_eq!(
            ruvector_apple_core_index_destroy(index),
            RvStatus::Ok as i32
        );
    }
}

#[test]
fn ffi_rejects_null_invalid_and_corrupt_inputs_with_reset_outputs() {
    unsafe {
        let mut index = 1_usize as *mut RvAppleIndex;
        assert_eq!(
            ruvector_apple_core_index_create(0, 1, 1, &mut index),
            RvStatus::InvalidArgument as i32
        );
        assert!(index.is_null());
        assert_eq!(
            ruvector_apple_core_index_create(2, 1, 99, &mut index),
            RvStatus::InvalidMetric as i32
        );
        assert_eq!(
            ruvector_apple_core_index_create(2, 1, 1, ptr::null_mut()),
            RvStatus::NullPointer as i32
        );
        assert_eq!(
            ruvector_apple_core_index_restore([1_u8, 2, 3].as_ptr(), 3, &mut index),
            RvStatus::CorruptSnapshot as i32
        );
        assert!(index.is_null());
        assert_eq!(
            ruvector_apple_core_results_free(ptr::null_mut()),
            RvStatus::Ok as i32
        );
        assert_eq!(
            ruvector_apple_core_bytes_free(ptr::null_mut()),
            RvStatus::Ok as i32
        );
        assert_eq!(
            ruvector_apple_core_index_destroy(ptr::null_mut()),
            RvStatus::Ok as i32
        );
    }
}

#[test]
fn opaque_owners_validate_accessors_without_caller_supplied_allocation_lengths() {
    unsafe {
        let mut values: *const RvSearchResult = 1_usize as *const RvSearchResult;
        let mut value_count = 99;
        assert_eq!(
            ruvector_apple_core_results_data(ptr::null(), &mut values, &mut value_count),
            RvStatus::NullPointer as i32
        );

        let mut bytes: *const u8 = 1_usize as *const u8;
        let mut byte_count = 99;
        assert_eq!(
            ruvector_apple_core_bytes_data(ptr::null(), &mut bytes, &mut byte_count),
            RvStatus::NullPointer as i32
        );
    }
}
