//! Integration tests for the learning module

#[cfg(test)]
mod learning_tests {
    use ruvector_postgres::learning::{
        OptimizationTarget, PatternExtractor, QueryTrajectory, ReasoningBank, SearchOptimizer,
        TrajectoryTracker, LEARNING_MANAGER,
    };

    #[test]
    fn test_end_to_end_learning_workflow() {
        // 1. Enable learning for a table
        LEARNING_MANAGER.enable_for_table("test_e2e", 1000);

        // 2. Record some query trajectories
        let tracker = LEARNING_MANAGER.get_tracker("test_e2e").unwrap();

        for i in 0u64..50 {
            let trajectory = QueryTrajectory::new(
                vec![i as f32 / 10.0, (i % 10) as f32],
                vec![i, i + 1],
                1000 + i * 10,
                (50 + (i % 3) * 10) as usize,
                (10 + (i % 2) * 5) as usize,
            );
            tracker.record(trajectory);
        }

        // 3. Extract patterns
        let patterns_extracted = LEARNING_MANAGER.extract_patterns("test_e2e", 5).unwrap();
        assert!(patterns_extracted > 0);

        // 4. Optimize a query
        let optimizer = LEARNING_MANAGER.get_optimizer("test_e2e").unwrap();
        let query = vec![2.5, 5.0];
        let params = optimizer.optimize(&query);

        assert!(params.ef_search > 0);
        assert!(params.probes > 0);
        assert!(params.confidence >= 0.0 && params.confidence <= 1.0);
    }

    #[test]
    fn test_trajectory_tracking_ring_buffer() {
        let tracker = TrajectoryTracker::new(10);

        // Fill the ring buffer
        for i in 0..15 {
            tracker.record(QueryTrajectory::new(vec![i as f32], vec![i], 1000, 50, 10));
        }

        let all = tracker.get_all();
        assert_eq!(all.len(), 10); // Ring buffer size

        let recent = tracker.get_recent(5);
        assert_eq!(recent.len(), 5);
    }

    #[test]
    fn test_pattern_extraction_with_clusters() {
        let mut trajectories = Vec::new();

        // Create two distinct clusters
        for i in 0..20 {
            // Cluster 1: vectors around [1.0, 0.0]
            trajectories.push(QueryTrajectory::new(
                vec![1.0 + (i as f32 * 0.01), 0.0],
                vec![i],
                1000,
                50,
                10,
            ));

            // Cluster 2: vectors around [0.0, 1.0]
            trajectories.push(QueryTrajectory::new(
                vec![0.0, 1.0 + (i as f32 * 0.01)],
                vec![i + 100],
                2000,
                60,
                15,
            ));
        }

        let extractor = PatternExtractor::new(2);
        let patterns = extractor.extract_patterns(&trajectories);

        assert_eq!(patterns.len(), 2);
        assert!(patterns[0].sample_count > 0);
        assert!(patterns[1].sample_count > 0);
    }

    #[test]
    fn test_reasoning_bank_consolidation() {
        let bank = ReasoningBank::new();

        // Store similar patterns
        for i in 0..5 {
            let pattern = ruvector_postgres::learning::LearnedPattern::new(
                vec![1.0 + i as f32 * 0.01, 0.0],
                50,
                10,
                0.9,
                100,
                1000.0,
                Some(0.95),
            );
            bank.store(pattern);
        }

        assert_eq!(bank.len(), 5);

        let merged = bank.consolidate(0.99);
        assert!(merged > 0);
        assert!(bank.len() < 5);
    }

    #[test]
    fn test_search_optimization_with_target() {
        let bank = std::sync::Arc::new(ReasoningBank::new());

        // Store test pattern
        let pattern = ruvector_postgres::learning::LearnedPattern::new(
            vec![1.0, 0.0, 0.0],
            50,
            10,
            0.9,
            100,
            1000.0,
            Some(0.95),
        );
        bank.store(pattern);

        let optimizer = SearchOptimizer::new(bank);

        let query = vec![1.0, 0.0, 0.0];

        let speed_params = optimizer.optimize_with_target(&query, OptimizationTarget::Speed);
        let accuracy_params = optimizer.optimize_with_target(&query, OptimizationTarget::Accuracy);

        // Speed should use lower parameters than accuracy
        assert!(speed_params.ef_search <= accuracy_params.ef_search);
    }

    #[test]
    fn test_trajectory_feedback() {
        let mut traj = QueryTrajectory::new(vec![1.0, 2.0], vec![1, 2, 3, 4, 5], 1000, 50, 10);

        traj.add_feedback(vec![1, 2, 6], vec![3, 4]);

        let precision = traj.precision().unwrap();
        let recall = traj.recall().unwrap();

        // 2 out of 5 results are relevant
        assert!((precision - 0.4).abs() < 0.01);
        // 2 out of 3 total relevant retrieved
        assert!((recall - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_pattern_similarity() {
        let pattern = ruvector_postgres::learning::LearnedPattern::new(
            vec![1.0, 0.0, 0.0],
            50,
            10,
            0.9,
            100,
            1000.0,
            Some(0.95),
        );

        let similar_query = vec![0.9, 0.1, 0.0];
        let dissimilar_query = vec![0.0, 1.0, 0.0];

        let sim1 = pattern.similarity(&similar_query);
        let sim2 = pattern.similarity(&dissimilar_query);

        assert!(sim1 > sim2);
        assert!(sim1 > 0.8);
        assert!(sim2 < 0.2);
    }

    #[test]
    fn test_learning_manager_lifecycle() {
        LEARNING_MANAGER.enable_for_table("test_lifecycle", 500);

        assert!(LEARNING_MANAGER.get_tracker("test_lifecycle").is_some());
        assert!(LEARNING_MANAGER
            .get_reasoning_bank("test_lifecycle")
            .is_some());
        assert!(LEARNING_MANAGER.get_optimizer("test_lifecycle").is_some());

        // Record some trajectories
        let tracker = LEARNING_MANAGER.get_tracker("test_lifecycle").unwrap();
        for i in 0..20 {
            tracker.record(QueryTrajectory::new(vec![i as f32], vec![i], 1000, 50, 10));
        }

        // Extract patterns
        let count = LEARNING_MANAGER
            .extract_patterns("test_lifecycle", 3)
            .unwrap();
        assert!(count > 0);

        // Verify patterns are stored
        let bank = LEARNING_MANAGER
            .get_reasoning_bank("test_lifecycle")
            .unwrap();
        assert!(bank.len() > 0);
    }

    #[test]
    fn test_performance_estimation() {
        let bank = std::sync::Arc::new(ReasoningBank::new());

        let pattern = ruvector_postgres::learning::LearnedPattern::new(
            vec![1.0, 0.0],
            50,
            10,
            0.9,
            100,
            1500.0,
            Some(0.95),
        );
        bank.store(pattern);

        let optimizer = SearchOptimizer::new(bank);

        let query = vec![0.9, 0.1];
        let params = ruvector_postgres::learning::SearchParams::new(50, 10, 0.9);

        let estimate = optimizer.estimate_performance(&query, &params);

        assert!(estimate.estimated_latency_us > 0.0);
        assert!(estimate.confidence > 0.0);
    }

    #[test]
    fn test_bank_pruning() {
        let bank = ReasoningBank::new();

        // Store patterns with varying confidence
        for i in 0..10 {
            let confidence = if i % 2 == 0 { 0.9 } else { 0.3 };
            let mut pattern = ruvector_postgres::learning::LearnedPattern::new(
                vec![i as f32],
                50,
                10,
                confidence,
                100,
                1000.0,
                Some(0.95),
            );
            bank.store(pattern);
        }

        assert_eq!(bank.len(), 10);

        // Prune low confidence patterns
        let pruned = bank.prune(0, 0.5);

        assert_eq!(pruned, 5); // Half should be pruned
        assert_eq!(bank.len(), 5);
    }

    #[test]
    fn test_trajectory_statistics() {
        let tracker = TrajectoryTracker::new(100);

        for i in 0..10 {
            let mut traj =
                QueryTrajectory::new(vec![i as f32], vec![i, i + 1], 1000 + i * 100, 50, 10);

            if i % 2 == 0 {
                traj.add_feedback(vec![i], vec![i + 1]);
            }

            tracker.record(traj);
        }

        let stats = tracker.stats();

        assert_eq!(stats.total_trajectories, 10);
        assert_eq!(stats.trajectories_with_feedback, 5);
        assert!(stats.avg_latency_us > 1000.0);
    }

    #[test]
    fn test_search_recommendations() {
        let bank = std::sync::Arc::new(ReasoningBank::new());

        // Store multiple patterns
        for i in 0..5 {
            let pattern = ruvector_postgres::learning::LearnedPattern::new(
                vec![i as f32, 0.0],
                50 + i * 5,
                10 + i,
                0.8 + i as f64 * 0.02,
                100,
                1000.0 + i as f64 * 100.0,
                Some(0.9),
            );
            bank.store(pattern);
        }

        let optimizer = SearchOptimizer::new(bank);
        let query = vec![2.0, 0.0];

        let recommendations = optimizer.recommendations(&query);

        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().all(|r| r.confidence >= 0.5));
    }
}
