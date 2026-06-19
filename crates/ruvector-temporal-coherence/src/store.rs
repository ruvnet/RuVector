//! In-memory vector store for agent memories.

pub type MemoryId = u64;

#[derive(Clone, Debug)]
pub struct MemoryMetadata {
    pub timestamp: u64,
    pub source: String,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct MemoryRecord {
    pub id: MemoryId,
    pub vec: Vec<f32>,
    pub metadata: MemoryMetadata,
}

/// Append-only, flat vector store.
/// For large corpora this is O(n) search — the variants add scoring layers
/// rather than a graph index, keeping the PoC self-contained and fair.
pub struct MemoryStore {
    records: Vec<MemoryRecord>,
    dims: usize,
    next_id: MemoryId,
}

impl MemoryStore {
    pub fn new(dims: usize) -> Self {
        Self {
            records: Vec::new(),
            dims,
            next_id: 0,
        }
    }

    pub fn insert(&mut self, vec: Vec<f32>, metadata: MemoryMetadata) -> MemoryId {
        assert_eq!(vec.len(), self.dims, "dimension mismatch");
        let id = self.next_id;
        self.next_id += 1;
        self.records.push(MemoryRecord { id, vec, metadata });
        id
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    pub fn dims(&self) -> usize {
        self.dims
    }

    pub fn records(&self) -> impl Iterator<Item = &MemoryRecord> {
        self.records.iter()
    }

    pub fn get(&self, id: MemoryId) -> Option<&MemoryRecord> {
        self.records.get(id as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_retrieve() {
        let mut s = MemoryStore::new(4);
        let id = s.insert(
            vec![1.0, 2.0, 3.0, 4.0],
            MemoryMetadata {
                timestamp: 42,
                source: "test".into(),
                tags: vec![],
            },
        );
        assert_eq!(id, 0);
        assert_eq!(s.len(), 1);
        let r = s.get(0).unwrap();
        assert_eq!(r.vec[0], 1.0);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn wrong_dims_panics() {
        let mut s = MemoryStore::new(4);
        s.insert(
            vec![1.0, 2.0],
            MemoryMetadata {
                timestamp: 0,
                source: "".into(),
                tags: vec![],
            },
        );
    }
}
