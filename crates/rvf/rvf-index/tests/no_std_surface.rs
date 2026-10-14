//! Exercise the public index path with the library built without `std`.

#[cfg(not(feature = "std"))]
mod no_std {
    use rvf_index::{cosine_distance, l2_distance, HnswConfig, HnswGraph, VectorStore};

    struct OneVector([f32; 2]);

    impl VectorStore for OneVector {
        fn get_vector(&self, id: u64) -> Option<&[f32]> {
            (id == 0).then_some(&self.0)
        }

        fn dimension(&self) -> usize {
            2
        }
    }

    #[test]
    fn scalar_distance_and_hnsw_level_selection_work_without_std() {
        assert!((cosine_distance(&[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-6);

        let vectors = OneVector([1.0, 0.0]);
        let mut graph = HnswGraph::new(&HnswConfig::default());
        graph.insert(0, 0.5, &vectors, &l2_distance);
        assert_eq!(graph.entry_point, Some(0));
        assert!(graph.layers[0].contains(0));
    }
}
