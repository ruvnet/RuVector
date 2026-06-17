/// A single time-sliced snapshot of an agent's active memory vectors.
#[derive(Debug, Clone)]
pub struct VectorWindow {
    pub window_id: u64,
    pub vectors: Vec<Vec<f32>>,
}

impl VectorWindow {
    pub fn new(window_id: u64, vectors: Vec<Vec<f32>>) -> Self {
        Self { window_id, vectors }
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.vectors.first().map(|v| v.len()).unwrap_or(0)
    }
}
