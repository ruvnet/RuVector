//! Vector storage trait for abstract access to vector data.

/// Provides access to vectors by ID without requiring a specific storage layout.
///
/// Implementations may back this with in-memory arrays, mmap'd VEC_SEGs,
/// or any other source of vector data.
pub trait VectorStore {
    /// Return the vector for the given node ID, or `None` if not present.
    fn get_vector(&self, id: u64) -> Option<&[f32]>;

    /// The dimensionality of all vectors in this store.
    fn dimension(&self) -> usize;

    /// Call `f(id, vector)` for every present id in `start..end`.
    ///
    /// The default walks the numeric range, which is right for dense id
    /// spaces. Stores that can enumerate their ids override this so the cost
    /// is bounded by the entries present, not by the width of the range.
    fn for_each_in_range(&self, start: u64, end: u64, f: &mut dyn FnMut(u64, &[f32])) {
        let mut id = start;
        while id < end {
            if let Some(v) = self.get_vector(id) {
                f(id, v);
            }
            id = match id.checked_add(1) {
                Some(n) => n,
                None => break,
            };
        }
    }
}

/// Simple in-memory vector store backed by a `Vec<Vec<f32>>`.
///
/// IDs are assumed to be contiguous starting from 0.
#[cfg(feature = "std")]
pub struct InMemoryVectorStore {
    vectors: Vec<Vec<f32>>,
    dim: usize,
}

#[cfg(feature = "std")]
impl InMemoryVectorStore {
    /// Create a new store from a collection of vectors.
    ///
    /// # Panics
    ///
    /// Panics if `vectors` is empty.
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        let dim = vectors.first().map_or(0, |v| v.len());
        Self { vectors, dim }
    }
}

#[cfg(feature = "std")]
impl VectorStore for InMemoryVectorStore {
    fn get_vector(&self, id: u64) -> Option<&[f32]> {
        self.vectors.get(id as usize).map(|v| v.as_slice())
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}
