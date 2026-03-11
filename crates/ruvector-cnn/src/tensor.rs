//! Tensor type for CNN operations
//!
//! Uses NHWC (batch, height, width, channels) memory layout for optimal
//! cache utilization with convolutional operations.

use crate::error::{CnnError, CnnResult};

/// A multi-dimensional tensor with NHWC layout
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Raw data storage
    data: Vec<f32>,
    /// Shape: [batch, height, width, channels] for 4D tensors
    shape: Vec<usize>,
    /// Strides for each dimension
    strides: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with the given shape, initialized to zeros
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let data = vec![0.0; numel];
        let strides = Self::compute_strides(shape);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Create a new tensor with the given shape, initialized to ones
    pub fn ones(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let data = vec![1.0; numel];
        let strides = Self::compute_strides(shape);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Create a tensor from raw data with the given shape
    pub fn from_data(data: Vec<f32>, shape: &[usize]) -> CnnResult<Self> {
        let expected_numel: usize = shape.iter().product();
        if data.len() != expected_numel {
            return Err(CnnError::invalid_shape(
                format!("data length {}", expected_numel),
                format!("data length {}", data.len()),
            ));
        }

        let strides = Self::compute_strides(shape);

        Ok(Self {
            data,
            shape: shape.to_vec(),
            strides,
        })
    }

    /// Create a tensor filled with a constant value
    pub fn full(shape: &[usize], value: f32) -> Self {
        let numel: usize = shape.iter().product();
        let data = vec![value; numel];
        let strides = Self::compute_strides(shape);

        Self {
            data,
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Compute strides for row-major (NHWC) layout
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Get the shape of the tensor
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of the tensor
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the raw data
    #[inline]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get a mutable reference to the raw data
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get element at index (for 4D NHWC tensor)
    #[inline]
    pub fn get_4d(&self, n: usize, h: usize, w: usize, c: usize) -> f32 {
        debug_assert!(self.shape.len() == 4);
        let idx = n * self.strides[0] + h * self.strides[1] + w * self.strides[2] + c;
        self.data[idx]
    }

    /// Set element at index (for 4D NHWC tensor)
    #[inline]
    pub fn set_4d(&mut self, n: usize, h: usize, w: usize, c: usize, value: f32) {
        debug_assert!(self.shape.len() == 4);
        let idx = n * self.strides[0] + h * self.strides[1] + w * self.strides[2] + c;
        self.data[idx] = value;
    }

    /// Get batch size (first dimension)
    #[inline]
    pub fn batch_size(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape[0]
        }
    }

    /// Get height (second dimension for NHWC)
    #[inline]
    pub fn height(&self) -> usize {
        if self.shape.len() < 2 {
            1
        } else {
            self.shape[1]
        }
    }

    /// Get width (third dimension for NHWC)
    #[inline]
    pub fn width(&self) -> usize {
        if self.shape.len() < 3 {
            1
        } else {
            self.shape[2]
        }
    }

    /// Get channels (fourth dimension for NHWC)
    #[inline]
    pub fn channels(&self) -> usize {
        if self.shape.len() < 4 {
            1
        } else {
            self.shape[3]
        }
    }

    /// Reshape the tensor to a new shape
    pub fn reshape(&self, new_shape: &[usize]) -> CnnResult<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(CnnError::invalid_shape(
                format!("numel {}", self.numel()),
                format!("numel {}", new_numel),
            ));
        }

        Self::from_data(self.data.clone(), new_shape)
    }

    /// Clone with a new shape (must have same numel)
    pub fn view(&self, new_shape: &[usize]) -> CnnResult<Self> {
        self.reshape(new_shape)
    }

    /// Get a slice of the tensor along the batch dimension
    pub fn slice_batch(&self, start: usize, end: usize) -> CnnResult<Self> {
        if self.shape.is_empty() {
            return Err(CnnError::invalid_shape("non-empty tensor", "empty tensor"));
        }

        if start >= end || end > self.shape[0] {
            return Err(CnnError::IndexOutOfBounds {
                index: end,
                size: self.shape[0],
            });
        }

        let batch_stride = self.strides[0];
        let start_idx = start * batch_stride;
        let end_idx = end * batch_stride;

        let mut new_shape = self.shape.clone();
        new_shape[0] = end - start;

        Self::from_data(self.data[start_idx..end_idx].to_vec(), &new_shape)
    }

    /// Apply a function element-wise
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let data: Vec<f32> = self.data.iter().map(|&x| f(x)).collect();
        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    /// Apply a function element-wise in place
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(f32) -> f32,
    {
        for x in &mut self.data {
            *x = f(*x);
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> CnnResult<Self> {
        if self.shape != other.shape {
            return Err(CnnError::shape_mismatch(format!(
                "add: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Self::from_data(data, &self.shape)
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> CnnResult<Self> {
        if self.shape != other.shape {
            return Err(CnnError::shape_mismatch(format!(
                "mul: {:?} vs {:?}",
                self.shape, other.shape
            )));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Self::from_data(data, &self.shape)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Self {
        self.map(|x| x * scalar)
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        self.sum() / self.numel() as f32
    }

    /// Maximum element
    pub fn max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Minimum element
    pub fn min(&self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }
}

impl Default for Tensor {
    fn default() -> Self {
        Self::zeros(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(&[2, 3, 4, 5]);
        assert_eq!(t.shape(), &[2, 3, 4, 5]);
        assert_eq!(t.numel(), 2 * 3 * 4 * 5);
        assert!(t.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::ones(&[2, 2, 2, 2]);
        assert!(t.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_tensor_strides() {
        let t = Tensor::zeros(&[2, 3, 4, 5]);
        assert_eq!(t.strides(), &[60, 20, 5, 1]); // NHWC row-major
    }

    #[test]
    fn test_tensor_get_set_4d() {
        let mut t = Tensor::zeros(&[2, 3, 4, 5]);
        t.set_4d(1, 2, 3, 4, 42.0);
        assert_eq!(t.get_4d(1, 2, 3, 4), 42.0);
    }

    #[test]
    fn test_tensor_reshape() {
        let t = Tensor::ones(&[2, 3, 4, 5]);
        let reshaped = t.reshape(&[6, 4, 5]).unwrap();
        assert_eq!(reshaped.shape(), &[6, 4, 5]);
        assert_eq!(reshaped.numel(), t.numel());
    }

    #[test]
    fn test_tensor_map() {
        let t = Tensor::full(&[2, 2], 2.0);
        let squared = t.map(|x| x * x);
        assert!(squared.data().iter().all(|&x| x == 4.0));
    }

    #[test]
    fn test_tensor_add() {
        let a = Tensor::ones(&[2, 2]);
        let b = Tensor::ones(&[2, 2]);
        let c = a.add(&b).unwrap();
        assert!(c.data().iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_tensor_sum_mean() {
        let t = Tensor::ones(&[2, 3]);
        assert_eq!(t.sum(), 6.0);
        assert_eq!(t.mean(), 1.0);
    }
}
