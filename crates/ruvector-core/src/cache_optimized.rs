//! Cache-optimized data structures using Structure-of-Arrays (SoA) layout
//!
//! This module provides cache-friendly layouts for vector storage to minimize
//! cache misses and improve memory access patterns.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

/// Cache line size (typically 64 bytes on modern CPUs)
const CACHE_LINE_SIZE: usize = 64;

/// Structure-of-Arrays layout for vectors
///
/// Instead of storing vectors as Vec<Vec<f32>>, we store all components
/// separately to improve cache locality during SIMD operations.
#[repr(align(64))] // Align to cache line boundary
pub struct SoAVectorStorage {
    /// Number of vectors
    count: usize,
    /// Dimensions per vector
    dimensions: usize,
    /// Capacity (allocated vectors)
    capacity: usize,
    /// Storage for each dimension separately
    /// Layout: [dim0_vec0, dim0_vec1, ..., dim0_vecN, dim1_vec0, ...]
    data: *mut f32,
}

impl SoAVectorStorage {
    /// Maximum allowed dimensions to prevent overflow
    const MAX_DIMENSIONS: usize = 65536;
    /// Maximum allowed capacity to prevent overflow
    const MAX_CAPACITY: usize = 1 << 24; // ~16M vectors

    /// Create a new SoA vector storage
    ///
    /// # Panics
    /// Panics if dimensions or capacity exceed safe limits or would cause overflow.
    pub fn new(dimensions: usize, initial_capacity: usize) -> Self {
        // Security: Validate inputs to prevent integer overflow
        assert!(
            dimensions > 0 && dimensions <= Self::MAX_DIMENSIONS,
            "dimensions must be between 1 and {}",
            Self::MAX_DIMENSIONS
        );
        assert!(
            initial_capacity <= Self::MAX_CAPACITY,
            "initial_capacity exceeds maximum of {}",
            Self::MAX_CAPACITY
        );

        let capacity = initial_capacity.next_power_of_two();

        // Security: Use checked arithmetic to prevent overflow
        let total_elements = dimensions
            .checked_mul(capacity)
            .expect("dimensions * capacity overflow");
        let total_bytes = total_elements
            .checked_mul(std::mem::size_of::<f32>())
            .expect("total size overflow");

        let layout =
            Layout::from_size_align(total_bytes, CACHE_LINE_SIZE).expect("invalid memory layout");

        let data = unsafe { alloc(layout) as *mut f32 };

        // Zero initialize
        unsafe {
            ptr::write_bytes(data, 0, total_elements);
        }

        Self {
            count: 0,
            dimensions,
            capacity,
            data,
        }
    }

    /// Add a vector to the storage
    pub fn push(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dimensions);

        if self.count >= self.capacity {
            self.grow();
        }

        // Store each dimension separately
        for (dim_idx, &value) in vector.iter().enumerate() {
            let offset = dim_idx * self.capacity + self.count;
            unsafe {
                *self.data.add(offset) = value;
            }
        }

        self.count += 1;
    }

    /// Get a vector by index (copies to output buffer)
    pub fn get(&self, index: usize, output: &mut [f32]) {
        assert!(index < self.count);
        assert_eq!(output.len(), self.dimensions);

        for dim_idx in 0..self.dimensions {
            let offset = dim_idx * self.capacity + index;
            output[dim_idx] = unsafe { *self.data.add(offset) };
        }
    }

    /// Get a slice of a specific dimension across all vectors
    /// This allows efficient SIMD operations on a single dimension
    pub fn dimension_slice(&self, dim_idx: usize) -> &[f32] {
        assert!(dim_idx < self.dimensions);
        let offset = dim_idx * self.capacity;
        unsafe { std::slice::from_raw_parts(self.data.add(offset), self.count) }
    }

    /// Get a mutable slice of a specific dimension
    pub fn dimension_slice_mut(&mut self, dim_idx: usize) -> &mut [f32] {
        assert!(dim_idx < self.dimensions);
        let offset = dim_idx * self.capacity;
        unsafe { std::slice::from_raw_parts_mut(self.data.add(offset), self.count) }
    }

    /// Number of vectors stored
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Dimensions per vector
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Grow the storage capacity
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;

        // Security: Use checked arithmetic to prevent overflow
        let new_total_elements = self
            .dimensions
            .checked_mul(new_capacity)
            .expect("dimensions * new_capacity overflow");
        let new_total_bytes = new_total_elements
            .checked_mul(std::mem::size_of::<f32>())
            .expect("total size overflow in grow");

        let new_layout = Layout::from_size_align(new_total_bytes, CACHE_LINE_SIZE)
            .expect("invalid memory layout in grow");

        let new_data = unsafe { alloc(new_layout) as *mut f32 };

        // Copy old data dimension by dimension
        for dim_idx in 0..self.dimensions {
            let old_offset = dim_idx * self.capacity;
            let new_offset = dim_idx * new_capacity;

            unsafe {
                ptr::copy_nonoverlapping(
                    self.data.add(old_offset),
                    new_data.add(new_offset),
                    self.count,
                );
            }
        }

        // Deallocate old data
        let old_layout = Layout::from_size_align(
            self.dimensions * self.capacity * std::mem::size_of::<f32>(),
            CACHE_LINE_SIZE,
        )
        .unwrap();

        unsafe {
            dealloc(self.data as *mut u8, old_layout);
        }

        self.data = new_data;
        self.capacity = new_capacity;
    }

    /// Compute distance from query to all stored vectors using dimension-wise operations
    /// This takes advantage of the SoA layout for better cache utilization
    pub fn batch_euclidean_distances(&self, query: &[f32], output: &mut [f32]) {
        assert_eq!(query.len(), self.dimensions);
        assert_eq!(output.len(), self.count);

        // Initialize output with zeros
        output.fill(0.0);

        // Process dimension by dimension
        for dim_idx in 0..self.dimensions {
            let dim_slice = self.dimension_slice(dim_idx);
            let query_val = query[dim_idx];

            // Compute squared differences for this dimension
            for vec_idx in 0..self.count {
                let diff = dim_slice[vec_idx] - query_val;
                output[vec_idx] += diff * diff;
            }
        }

        // Take square root
        for distance in output.iter_mut() {
            *distance = distance.sqrt();
        }
    }
}

impl Drop for SoAVectorStorage {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(
            self.dimensions * self.capacity * std::mem::size_of::<f32>(),
            CACHE_LINE_SIZE,
        )
        .unwrap();

        unsafe {
            dealloc(self.data as *mut u8, layout);
        }
    }
}

unsafe impl Send for SoAVectorStorage {}
unsafe impl Sync for SoAVectorStorage {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_storage() {
        let mut storage = SoAVectorStorage::new(3, 4);

        storage.push(&[1.0, 2.0, 3.0]);
        storage.push(&[4.0, 5.0, 6.0]);

        assert_eq!(storage.len(), 2);

        let mut output = vec![0.0; 3];
        storage.get(0, &mut output);
        assert_eq!(output, vec![1.0, 2.0, 3.0]);

        storage.get(1, &mut output);
        assert_eq!(output, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dimension_slice() {
        let mut storage = SoAVectorStorage::new(3, 4);

        storage.push(&[1.0, 2.0, 3.0]);
        storage.push(&[4.0, 5.0, 6.0]);
        storage.push(&[7.0, 8.0, 9.0]);

        // Get all values for dimension 0
        let dim0 = storage.dimension_slice(0);
        assert_eq!(dim0, &[1.0, 4.0, 7.0]);

        // Get all values for dimension 1
        let dim1 = storage.dimension_slice(1);
        assert_eq!(dim1, &[2.0, 5.0, 8.0]);
    }

    #[test]
    fn test_batch_distances() {
        let mut storage = SoAVectorStorage::new(3, 4);

        storage.push(&[1.0, 0.0, 0.0]);
        storage.push(&[0.0, 1.0, 0.0]);
        storage.push(&[0.0, 0.0, 1.0]);

        let query = vec![1.0, 0.0, 0.0];
        let mut distances = vec![0.0; 3];

        storage.batch_euclidean_distances(&query, &mut distances);

        assert!((distances[0] - 0.0).abs() < 0.001);
        assert!((distances[1] - 1.414).abs() < 0.01);
        assert!((distances[2] - 1.414).abs() < 0.01);
    }
}
