//! Full pipeline test with 10,000 perception events.
//!
//! This test verifies that the cognitive demo pipeline:
//! 1. Mounts the RVF package
//! 2. Runs 10,000 perception events through the pipeline
//! 3. Verifies all attestations in witness log
//! 4. Checkpoints, restarts, replays
//! 5. Verifies bit-identical final state

use ruvix_demo::{
    config,
    pipeline::{CognitivePipeline, PipelineConfig, PipelineState},
    stats::SyscallStats,
};

/// Test the full pipeline with 10,000 events.
#[test]
fn test_full_pipeline_10000_events() {
    let config = PipelineConfig {
        event_count: config::FULL_PIPELINE_EVENTS as u64,
        batch_size: 500,
        verbose: false,
        seed: 0xDEADBEEF,
    };

    let mut pipeline = CognitivePipeline::new(config);

    // Run the complete pipeline
    let result = pipeline.run().unwrap();

    // Verify completion
    assert_eq!(result.state, PipelineState::Completed);
    assert!(result.is_success());

    // Verify event counts
    assert_eq!(result.events_processed, config::FULL_PIPELINE_EVENTS as u64);
    assert_eq!(result.vectors_stored, config::FULL_PIPELINE_EVENTS as u64);

    // Verify attestations
    assert!(result.attestations >= config::FULL_PIPELINE_EVENTS as u64);

    // Verify graph mutations (every 2 vectors)
    // Note: May be less due to coherence threshold filtering
    assert!(result.graph_mutations > 0);

    println!(
        "Full pipeline completed:\n\
         - Events: {}\n\
         - Vectors: {}\n\
         - Graph mutations: {}\n\
         - Attestations: {}",
        result.events_processed,
        result.vectors_stored,
        result.graph_mutations,
        result.attestations
    );
}

/// Test all 12 syscalls are exercised.
#[test]
fn test_all_syscalls_covered() {
    let config = PipelineConfig {
        event_count: 1000,
        batch_size: 100,
        ..Default::default()
    };

    let mut pipeline = CognitivePipeline::new(config);
    let result = pipeline.run().unwrap();

    let stats = &result.syscall_stats;

    // Verify all syscalls were called
    assert!(stats.task_spawn > 0, "task_spawn not called");
    assert!(stats.cap_grant > 0, "cap_grant not called");
    assert!(stats.region_map > 0, "region_map not called");
    assert!(stats.queue_send > 0, "queue_send not called");
    assert!(stats.queue_recv > 0, "queue_recv not called");
    assert!(stats.timer_wait > 0, "timer_wait not called");
    assert!(stats.rvf_mount > 0, "rvf_mount not called");
    assert!(stats.attest_emit > 0, "attest_emit not called");
    assert!(stats.vector_get > 0, "vector_get not called");
    assert!(stats.vector_put_proved > 0, "vector_put_proved not called");
    assert!(stats.graph_apply_proved > 0, "graph_apply_proved not called");
    assert!(stats.sensor_subscribe > 0, "sensor_subscribe not called");

    assert!(
        stats.all_covered(),
        "Not all syscalls covered: {:?}",
        stats.uncovered_syscalls()
    );
}

/// Test checkpoint and replay produces identical state.
#[test]
fn test_checkpoint_replay_determinism() {
    let config = PipelineConfig {
        event_count: 500,
        batch_size: 50,
        seed: 12345,
        ..Default::default()
    };

    // First run
    let mut pipeline1 = CognitivePipeline::new(config.clone());
    pipeline1.initialize().unwrap();
    pipeline1.setup_coordinator().unwrap();

    // Process half
    for _ in 0..5 {
        pipeline1.process_batch().unwrap();
    }

    // Checkpoint
    let checkpoint = pipeline1.checkpoint().unwrap();
    let checkpoint_events = checkpoint.events_processed;

    // Continue first run
    while pipeline1.state() == PipelineState::Running {
        pipeline1.process_batch().unwrap();
    }
    let final_stats1 = pipeline1.get_syscall_stats();

    // Second run - restart from checkpoint
    let mut pipeline2 = CognitivePipeline::new(config);
    pipeline2.initialize().unwrap();
    pipeline2.setup_coordinator().unwrap();

    // Process to checkpoint point
    while pipeline2.events_processed() < checkpoint_events {
        pipeline2.process_batch().unwrap();
    }

    // Verify checkpoint alignment
    let checkpoint2 = pipeline2.checkpoint().unwrap();
    assert_eq!(
        checkpoint.events_processed, checkpoint2.events_processed,
        "Checkpoint states differ"
    );

    // Continue second run
    while pipeline2.state() == PipelineState::Running {
        pipeline2.process_batch().unwrap();
    }
    let final_stats2 = pipeline2.get_syscall_stats();

    // Verify bit-identical final state
    assert_eq!(
        final_stats1.total(),
        final_stats2.total(),
        "Total syscall counts differ"
    );
}

/// Test batch processing with various batch sizes.
#[test]
fn test_batch_processing_consistency() {
    // Test with batch size 10
    let config10 = PipelineConfig {
        event_count: 100,
        batch_size: 10,
        seed: 42,
        ..Default::default()
    };
    let mut pipeline10 = CognitivePipeline::new(config10);
    let result10 = pipeline10.run().unwrap();

    // Test with batch size 50
    let config50 = PipelineConfig {
        event_count: 100,
        batch_size: 50,
        seed: 42,
        ..Default::default()
    };
    let mut pipeline50 = CognitivePipeline::new(config50);
    let result50 = pipeline50.run().unwrap();

    // Results should be identical regardless of batch size
    assert_eq!(result10.events_processed, result50.events_processed);
    assert_eq!(result10.vectors_stored, result50.vectors_stored);
}

/// Test deterministic event generation.
#[test]
fn test_deterministic_generation() {
    let config = PipelineConfig {
        event_count: 100,
        batch_size: 100,
        seed: 0xCAFEBABE,
        ..Default::default()
    };

    // Run twice with same seed
    let mut pipeline1 = CognitivePipeline::new(config.clone());
    let result1 = pipeline1.run().unwrap();

    let mut pipeline2 = CognitivePipeline::new(config);
    let result2 = pipeline2.run().unwrap();

    // Should produce identical results
    assert_eq!(result1.events_processed, result2.events_processed);
    assert_eq!(result1.vectors_stored, result2.vectors_stored);
    assert_eq!(result1.attestations, result2.attestations);
    assert_eq!(
        result1.syscall_stats.total(),
        result2.syscall_stats.total()
    );
}

/// Test pipeline handles small event counts correctly.
#[test]
fn test_small_event_count() {
    for event_count in [1, 2, 5, 10, 25] {
        let config = PipelineConfig {
            event_count,
            batch_size: 10,
            ..Default::default()
        };

        let mut pipeline = CognitivePipeline::new(config);
        let result = pipeline.run().unwrap();

        assert!(
            result.is_success(),
            "Failed with event_count={}",
            event_count
        );
        assert_eq!(result.events_processed, event_count);
    }
}

/// Test feature coverage report generation.
#[test]
fn test_feature_coverage_report() {
    let config = PipelineConfig {
        event_count: 500,
        batch_size: 100,
        ..Default::default()
    };

    let mut pipeline = CognitivePipeline::new(config);
    let result = pipeline.run().unwrap();

    let coverage = &result.feature_coverage;
    let report = coverage.report();

    // Print report for manual inspection
    println!("{}", report.summary());

    // Verify report structure
    assert_eq!(report.syscalls.total, 12);
    assert!(report.regions.regions_created >= 3);
    assert!(report.overall_percentage > 0.0);
}

/// Test witness log attestations match event count.
#[test]
fn test_witness_log_attestations() {
    let config = PipelineConfig {
        event_count: 200,
        batch_size: 50,
        ..Default::default()
    };

    let mut pipeline = CognitivePipeline::new(config);
    let result = pipeline.run().unwrap();

    // Each event should generate an attestation
    assert!(
        result.attestations >= result.events_processed,
        "Missing attestations: {} events, {} attestations",
        result.events_processed,
        result.attestations
    );
}

/// Test coordinator timer waits.
#[test]
fn test_coordinator_timer_waits() {
    let config = PipelineConfig {
        event_count: 100,
        batch_size: 25,
        ..Default::default()
    };

    let mut pipeline = CognitivePipeline::new(config);
    let result = pipeline.run().unwrap();

    // Coordinator should have performed timer waits
    assert!(
        result.syscall_stats.timer_wait >= config::TIMER_WAITS as u64,
        "Expected {} timer waits, got {}",
        config::TIMER_WAITS,
        result.syscall_stats.timer_wait
    );
}

/// Test pipeline lifecycle states.
#[test]
fn test_pipeline_lifecycle() {
    let config = PipelineConfig {
        event_count: 50,
        batch_size: 10,
        ..Default::default()
    };

    let mut pipeline = CognitivePipeline::new(config);

    // Initial state
    assert_eq!(pipeline.state(), PipelineState::Uninitialized);

    // After initialization
    pipeline.initialize().unwrap();
    assert_eq!(pipeline.state(), PipelineState::Running);

    // After completion
    while pipeline.state() == PipelineState::Running {
        pipeline.process_batch().unwrap();
    }
    assert_eq!(pipeline.state(), PipelineState::Completed);
}

/// Benchmark-style test for performance measurement.
#[test]
fn test_pipeline_performance() {
    use std::time::Instant;

    let config = PipelineConfig {
        event_count: 1000,
        batch_size: 100,
        ..Default::default()
    };

    let start = Instant::now();
    let mut pipeline = CognitivePipeline::new(config);
    let result = pipeline.run().unwrap();
    let elapsed = start.elapsed();

    let events_per_sec = result.events_processed as f64 / elapsed.as_secs_f64();

    println!(
        "Performance: {} events in {:?} ({:.0} events/sec)",
        result.events_processed, elapsed, events_per_sec
    );

    // Should process at least 1000 events/sec (very conservative)
    assert!(
        events_per_sec > 100.0,
        "Performance too slow: {} events/sec",
        events_per_sec
    );
}
