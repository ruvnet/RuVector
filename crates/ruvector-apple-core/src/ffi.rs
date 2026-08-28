use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;
use std::sync::RwLock;

use crate::{
    DistanceMetric, ExactVectorIndex, IndexConfig, IndexError, SearchHit, MAX_SNAPSHOT_BYTES,
    RVECTOR_APPLE_CORE_ABI_VERSION,
};

/// Status code returned by C ABI operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum RvStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// A required pointer was null.
    NullPointer = 1,
    /// A scalar argument or resource bound was invalid.
    InvalidArgument = 2,
    /// A metric tag was invalid.
    InvalidMetric = 3,
    /// A vector length did not match the index dimensions.
    DimensionMismatch = 4,
    /// A vector contained NaN or infinity.
    NonFiniteValue = 5,
    /// A cosine vector had zero norm.
    ZeroNormVector = 6,
    /// The index has no capacity for another ID.
    CapacityExceeded = 7,
    /// Snapshot integrity or structure validation failed.
    CorruptSnapshot = 8,
    /// A snapshot or output exceeded its compiled bound.
    ResourceLimit = 9,
    /// A synchronization primitive was poisoned.
    LockPoisoned = 10,
    /// A Rust panic was caught before it crossed the ABI boundary.
    Panic = 255,
}

/// Opaque, thread-safe native index handle.
#[repr(C)]
pub struct RvAppleIndex {
    inner: RwLock<ExactVectorIndex>,
}

/// Opaque owner for one bounded search result allocation.
#[repr(C)]
pub struct RvOwnedResults {
    values: Box<[RvSearchResult]>,
}

/// Opaque owner for one bounded snapshot allocation.
#[repr(C)]
pub struct RvOwnedBytes {
    values: Box<[u8]>,
}

/// Fixed-layout search result returned to C callers.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct RvSearchResult {
    /// Application-defined vector ID.
    pub id: u64,
    /// Metric score, ordered with higher values first.
    pub score: f64,
}

/// Fixed-layout metadata for an opaque index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct RvIndexInfo {
    /// ABI version implemented by the library.
    pub abi_version: u32,
    /// Vector dimensions.
    pub dimensions: u32,
    /// Maximum vector count.
    pub capacity: u32,
    /// Current vector count.
    pub count: u32,
    /// Metric tag: cosine=1, L2=2, dot=3.
    pub metric: u32,
}

/// Returns the native ABI version, or zero only if an unexpected panic occurs.
#[no_mangle]
pub extern "C" fn ruvector_apple_core_abi_version() -> u32 {
    catch_unwind(|| RVECTOR_APPLE_CORE_ABI_VERSION).unwrap_or(0)
}

/// Creates a bounded index and transfers its ownership to the caller.
///
/// # Safety
///
/// `out_index` must be valid for one pointer write. The caller must eventually
/// pass a successful result to [`ruvector_apple_core_index_destroy`].
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_index_create(
    dimensions: u32,
    capacity: u32,
    metric: u32,
    out_index: *mut *mut RvAppleIndex,
) -> i32 {
    ffi_guard(|| {
        require_pointer(out_index)?;
        unsafe { out_index.write(ptr::null_mut()) };
        let metric = DistanceMetric::try_from(metric).map_err(map_error)?;
        let index = ExactVectorIndex::new(IndexConfig {
            dimensions,
            capacity,
            metric,
        })
        .map_err(map_error)?;
        let handle = Box::new(RvAppleIndex {
            inner: RwLock::new(index),
        });
        unsafe { out_index.write(Box::into_raw(handle)) };
        Ok(())
    })
}

/// Destroys an index returned by create or restore. A null pointer is a no-op.
///
/// # Safety
///
/// `index` must be null or an owned pointer returned by this library that has
/// not previously been destroyed.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_index_destroy(index: *mut RvAppleIndex) -> i32 {
    ffi_guard(|| {
        if !index.is_null() {
            unsafe { drop(Box::from_raw(index)) };
        }
        Ok(())
    })
}

/// Inserts or replaces one vector.
///
/// # Safety
///
/// `index` must be a live library handle. For non-zero `values_len`, `values`
/// must reference that many readable `f32` values for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_index_upsert(
    index: *const RvAppleIndex,
    id: u64,
    values: *const f32,
    values_len: u32,
) -> i32 {
    ffi_guard(|| {
        let handle = unsafe { index_ref(index) }?;
        let mut index = handle.inner.write().map_err(|_| RvStatus::LockPoisoned)?;
        if values_len != index.config().dimensions {
            return Err(RvStatus::DimensionMismatch);
        }
        let values = unsafe { input_slice(values, u64::from(values_len)) }?;
        index.upsert(id, values).map_err(map_error)
    })
}

/// Removes one ID and writes `1` when it existed or `0` otherwise.
///
/// # Safety
///
/// `index` must be a live library handle and `out_removed` must be valid for
/// one `u8` write.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_index_remove(
    index: *const RvAppleIndex,
    id: u64,
    out_removed: *mut u8,
) -> i32 {
    ffi_guard(|| {
        require_pointer(out_removed)?;
        unsafe { out_removed.write(0) };
        let handle = unsafe { index_ref(index) }?;
        let removed = handle
            .inner
            .write()
            .map_err(|_| RvStatus::LockPoisoned)?
            .remove(id);
        unsafe { out_removed.write(u8::from(removed)) };
        Ok(())
    })
}

/// Reads index metadata.
///
/// # Safety
///
/// `index` must be a live library handle and `out_info` must be valid for one
/// [`RvIndexInfo`] write.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_index_info(
    index: *const RvAppleIndex,
    out_info: *mut RvIndexInfo,
) -> i32 {
    ffi_guard(|| {
        require_pointer(out_info)?;
        unsafe {
            out_info.write(RvIndexInfo {
                abi_version: 0,
                dimensions: 0,
                capacity: 0,
                count: 0,
                metric: 0,
            })
        };
        let handle = unsafe { index_ref(index) }?;
        let index = handle.inner.read().map_err(|_| RvStatus::LockPoisoned)?;
        let config = index.config();
        let count = u32::try_from(index.len()).map_err(|_| RvStatus::ResourceLimit)?;
        unsafe {
            out_info.write(RvIndexInfo {
                abi_version: RVECTOR_APPLE_CORE_ABI_VERSION,
                dimensions: config.dimensions,
                capacity: config.capacity,
                count,
                metric: config.metric as u32,
            })
        };
        Ok(())
    })
}

/// Runs exact search and transfers an opaque result owner to the caller.
///
/// # Safety
///
/// All input pointers must satisfy the same requirements as upsert.
/// `out_results` must be valid for one write. A non-null result owner must be
/// inspected through [`ruvector_apple_core_results_data`] and released with
/// [`ruvector_apple_core_results_free`].
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_index_search(
    index: *const RvAppleIndex,
    query: *const f32,
    query_len: u32,
    limit: u32,
    out_results: *mut *mut RvOwnedResults,
) -> i32 {
    ffi_guard(|| {
        require_pointer(out_results)?;
        unsafe { out_results.write(ptr::null_mut()) };
        let handle = unsafe { index_ref(index) }?;
        let index = handle.inner.read().map_err(|_| RvStatus::LockPoisoned)?;
        if query_len != index.config().dimensions {
            return Err(RvStatus::DimensionMismatch);
        }
        let query = unsafe { input_slice(query, u64::from(query_len)) }?;
        let hits = index.search(query, limit as usize).map_err(map_error)?;
        if hits.is_empty() {
            return Ok(());
        }
        let values: Box<[RvSearchResult]> = hits
            .into_iter()
            .map(|hit: SearchHit| RvSearchResult {
                id: hit.id,
                score: hit.score,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let owner = Box::new(RvOwnedResults { values });
        unsafe { out_results.write(Box::into_raw(owner)) };
        Ok(())
    })
}

/// Serializes and transfers an opaque checksummed snapshot owner to the caller.
///
/// # Safety
///
/// `index` must be live and `out_bytes` must be valid for one write. A non-null
/// owner must be inspected through [`ruvector_apple_core_bytes_data`] and
/// released with [`ruvector_apple_core_bytes_free`].
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_index_snapshot(
    index: *const RvAppleIndex,
    out_bytes: *mut *mut RvOwnedBytes,
) -> i32 {
    ffi_guard(|| {
        require_pointer(out_bytes)?;
        unsafe { out_bytes.write(ptr::null_mut()) };
        let handle = unsafe { index_ref(index) }?;
        let snapshot = handle
            .inner
            .read()
            .map_err(|_| RvStatus::LockPoisoned)?
            .snapshot()
            .map_err(map_error)?;
        let owner = Box::new(RvOwnedBytes {
            values: snapshot.into_boxed_slice(),
        });
        unsafe { out_bytes.write(Box::into_raw(owner)) };
        Ok(())
    })
}

/// Restores a checksummed snapshot and transfers a new index to the caller.
///
/// # Safety
///
/// For non-zero `bytes_len`, `bytes` must reference that many readable bytes.
/// `out_index` must be valid for one pointer write.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_index_restore(
    bytes: *const u8,
    bytes_len: u64,
    out_index: *mut *mut RvAppleIndex,
) -> i32 {
    ffi_guard(|| {
        require_pointer(out_index)?;
        unsafe { out_index.write(ptr::null_mut()) };
        if bytes_len > MAX_SNAPSHOT_BYTES {
            return Err(RvStatus::ResourceLimit);
        }
        let bytes = unsafe { input_slice(bytes, bytes_len) }?;
        let index = ExactVectorIndex::from_snapshot(bytes).map_err(map_error)?;
        let handle = Box::new(RvAppleIndex {
            inner: RwLock::new(index),
        });
        unsafe { out_index.write(Box::into_raw(handle)) };
        Ok(())
    })
}

/// Borrows the immutable result data owned by a search result handle.
///
/// # Safety
///
/// `results` must be a live result owner returned by this library. `out_values`
/// and `out_len` must be valid for one write. The returned pointer is valid only
/// until its owner is freed and must not be mutated or freed by the caller.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_results_data(
    results: *const RvOwnedResults,
    out_values: *mut *const RvSearchResult,
    out_len: *mut u32,
) -> i32 {
    ffi_guard(|| {
        require_pointer(results)?;
        require_pointer(out_values)?;
        require_pointer(out_len)?;
        let owner = unsafe { &*results };
        let length = u32::try_from(owner.values.len()).map_err(|_| RvStatus::ResourceLimit)?;
        unsafe {
            out_values.write(owner.values.as_ptr());
            out_len.write(length);
        }
        Ok(())
    })
}

/// Frees a result owner returned by search. A null pointer is a no-op.
///
/// # Safety
///
/// A non-null pointer must be an owned live result handle returned by this
/// library and must not have been freed before.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_results_free(results: *mut RvOwnedResults) -> i32 {
    ffi_guard(|| {
        if !results.is_null() {
            unsafe { drop(Box::from_raw(results)) };
        }
        Ok(())
    })
}

/// Borrows the immutable bytes owned by a snapshot handle.
///
/// # Safety
///
/// `bytes` must be a live snapshot owner returned by this library. Output
/// pointers must each be valid for one write. The borrowed data is valid only
/// until the owner is freed and must not be mutated or freed by the caller.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_bytes_data(
    bytes: *const RvOwnedBytes,
    out_values: *mut *const u8,
    out_len: *mut u64,
) -> i32 {
    ffi_guard(|| {
        require_pointer(bytes)?;
        require_pointer(out_values)?;
        require_pointer(out_len)?;
        let owner = unsafe { &*bytes };
        unsafe {
            out_values.write(owner.values.as_ptr());
            out_len.write(owner.values.len() as u64);
        }
        Ok(())
    })
}

/// Frees a snapshot owner returned by snapshot. A null pointer is a no-op.
///
/// # Safety
///
/// A non-null pointer must be an owned live snapshot handle returned by this
/// library and must not have been freed before.
#[no_mangle]
pub unsafe extern "C" fn ruvector_apple_core_bytes_free(bytes: *mut RvOwnedBytes) -> i32 {
    ffi_guard(|| {
        if !bytes.is_null() {
            unsafe { drop(Box::from_raw(bytes)) };
        }
        Ok(())
    })
}

fn ffi_guard(operation: impl FnOnce() -> Result<(), RvStatus>) -> i32 {
    match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(Ok(())) => RvStatus::Ok as i32,
        Ok(Err(status)) => status as i32,
        Err(_) => RvStatus::Panic as i32,
    }
}

fn require_pointer<T>(pointer: *const T) -> Result<(), RvStatus> {
    if pointer.is_null() {
        Err(RvStatus::NullPointer)
    } else {
        Ok(())
    }
}

unsafe fn index_ref<'a>(index: *const RvAppleIndex) -> Result<&'a RvAppleIndex, RvStatus> {
    unsafe { index.as_ref() }.ok_or(RvStatus::NullPointer)
}

unsafe fn input_slice<'a, T>(values: *const T, length: u64) -> Result<&'a [T], RvStatus> {
    let length = usize::try_from(length).map_err(|_| RvStatus::ResourceLimit)?;
    if length == 0 {
        return Ok(&[]);
    }
    require_pointer(values)?;
    Ok(unsafe { slice::from_raw_parts(values, length) })
}

fn map_error(error: IndexError) -> RvStatus {
    match error {
        IndexError::InvalidMetric(_) => RvStatus::InvalidMetric,
        IndexError::DimensionMismatch { .. } => RvStatus::DimensionMismatch,
        IndexError::NonFiniteValue => RvStatus::NonFiniteValue,
        IndexError::ZeroNormVector => RvStatus::ZeroNormVector,
        IndexError::CapacityExceeded(_) => RvStatus::CapacityExceeded,
        IndexError::CorruptSnapshot(_) => RvStatus::CorruptSnapshot,
        IndexError::SnapshotTooLarge { .. }
        | IndexError::SearchLimitExceeded(_)
        | IndexError::MemoryBudgetExceeded { .. } => RvStatus::ResourceLimit,
        IndexError::InvalidDimensions(_) | IndexError::InvalidCapacity(_) => {
            RvStatus::InvalidArgument
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_is_contained() {
        let status = ffi_guard(|| -> Result<(), RvStatus> { panic!("contained") });
        assert_eq!(status, RvStatus::Panic as i32);
    }
}
