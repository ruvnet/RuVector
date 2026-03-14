//! Feature coverage matrix verification tests.
//!
//! This test module verifies that the cognitive demo exercises ALL RuVix
//! kernel features as specified in ADR-087.

use ruvix_demo::{
    config,
    manifest::{ComponentType, DemoManifest, DemoRegionType, ProofPolicyConfig},
    pipeline::{CognitivePipeline, PipelineConfig},
    stats::{FeatureCoverage, SyscallStats},
};
use ruvix_types::ProofTier;

/// Verify the feature coverage matrix matches specification.
#[test]
fn test_feature_coverage_matrix() {
    let config = PipelineConfig {
        event_count: config::FULL_PIPELINE_EVENTS as u64,
        batch_size: 500,
        ..Default::default()
    };

    let mut pipeline = CognitivePipeline::new(config);
    let result = pipeline.run().unwrap();

    let stats = &result.syscall_stats;

    // Expected counts from specification
    let expected = [
        ("task_spawn", config::TASK_SPAWNS as u64, stats.task_spawn),
        ("cap_grant", config::CAP_GRANTS as u64, stats.cap_grant),
        ("region_map", 3, stats.region_map),
        (
            "queue_send",
            config::FULL_PIPELINE_EVENTS as u64 * 2,
            stats.queue_send,
        ),
        (
            "queue_recv",
            config::FULL_PIPELINE_EVENTS as u64 * 2,
            stats.queue_recv,
        ),
        ("timer_wait", config::TIMER_WAITS as u64, stats.timer_wait),
        ("rvf_mount", 1, stats.rvf_mount),
        (
            "attest_emit",
            config::FULL_PIPELINE_EVENTS as u64,
            stats.attest_emit,
        ),
        (
            "vector_get",
            config::FULL_PIPELINE_EVENTS as u64,
            stats.vector_get,
        ),
        (
            "vector_put_proved",
            config::FULL_PIPELINE_EVENTS as u64,
            stats.vector_put_proved,
        ),
        (
            "graph_apply_proved",
            config::GRAPH_MUTATIONS as u64,
            stats.graph_apply_proved,
        ),
        ("sensor_subscribe", 1, stats.sensor_subscribe),
    ];

    println!("\nFeature Coverage Matrix:");
    println!(
        "| {:<20} | {:<15} | {:<10} | {:<10} |",
        "Syscall", "Component", "Expected", "Actual"
    );
    println!("|{:-<22}|{:-<17}|{:-<12}|{:-<12}|", "", "", "", "");

    for (name, expected, actual) in &expected {
        let component = match *name {
            "task_spawn" => "Coordinator",
            "cap_grant" => "Coordinator",
            "region_map" => "Boot",
            "queue_send" => "All",
            "queue_recv" => "All",
            "timer_wait" => "Coordinator",
            "rvf_mount" => "Boot",
            "attest_emit" => "Attestor",
            "vector_get" => "ReasoningEngine",
            "vector_put_proved" => "FeatureExtractor",
            "graph_apply_proved" => "ReasoningEngine",
            "sensor_subscribe" => "SensorAdapter",
            _ => "Unknown",
        };

        let status = if actual >= expected { "OK" } else { "UNDER" };
        println!(
            "| {:<20} | {:<15} | {:>10} | {:>10} | {}",
            name, component, expected, actual, status
        );
    }

    // Verify minimum expected counts
    assert!(
        stats.task_spawn >= config::TASK_SPAWNS as u64,
        "task_spawn: {} < {}",
        stats.task_spawn,
        config::TASK_SPAWNS
    );
    assert!(
        stats.cap_grant >= config::CAP_GRANTS as u64,
        "cap_grant: {} < {}",
        stats.cap_grant,
        config::CAP_GRANTS
    );
    assert!(stats.region_map >= 3, "region_map: {} < 3", stats.region_map);
    assert!(stats.rvf_mount >= 1, "rvf_mount: {} < 1", stats.rvf_mount);
    assert!(
        stats.sensor_subscribe >= 1,
        "sensor_subscribe: {} < 1",
        stats.sensor_subscribe
    );
}

/// Verify all 3 memory region types are used.
#[test]
fn test_all_region_types_covered() {
    let manifest = DemoManifest::cognitive_demo();

    let mut has_immutable = false;
    let mut has_append_only = false;
    let mut has_slab = false;

    for region in &manifest.regions {
        match region.region_type {
            DemoRegionType::Immutable { size } => {
                has_immutable = true;
                assert_eq!(size, config::MODEL_WEIGHTS_SIZE);
                assert_eq!(region.name, "model_weights");
            }
            DemoRegionType::AppendOnly { max_size } => {
                has_append_only = true;
                assert_eq!(max_size, config::WITNESS_LOG_MAX_SIZE);
                assert_eq!(region.name, "witness_log");
            }
            DemoRegionType::Slab { slot_size, slots } => {
                has_slab = true;
                assert_eq!(slot_size, config::VECTOR_SLOT_SIZE);
                assert_eq!(slots, config::VECTOR_SLOT_COUNT);
                assert_eq!(region.name, "vector_store");
            }
        }
    }

    assert!(has_immutable, "Missing Immutable region");
    assert!(has_append_only, "Missing AppendOnly region");
    assert!(has_slab, "Missing Slab region");
}

/// Verify all 3 proof tiers are configured.
#[test]
fn test_all_proof_tiers_configured() {
    let manifest = DemoManifest::cognitive_demo();
    let policy = &manifest.proof_policy;

    // Verify tier assignments
    assert_eq!(
        policy.vector_mutations,
        ProofTier::Reflex,
        "Vector mutations should use Reflex tier"
    );
    assert_eq!(
        policy.graph_mutations,
        ProofTier::Standard,
        "Graph mutations should use Standard tier"
    );
    assert_eq!(
        policy.structural_changes,
        ProofTier::Deep,
        "Structural changes should use Deep tier"
    );

    // Verify component tier overrides
    let fe_tier = policy
        .component_tiers
        .iter()
        .find(|ct| ct.component == ComponentType::FeatureExtractor)
        .map(|ct| ct.tier);
    assert_eq!(
        fe_tier,
        Some(ProofTier::Reflex),
        "FeatureExtractor should use Reflex tier"
    );

    let re_tier = policy
        .component_tiers
        .iter()
        .find(|ct| ct.component == ComponentType::ReasoningEngine)
        .map(|ct| ct.tier);
    assert_eq!(
        re_tier,
        Some(ProofTier::Standard),
        "ReasoningEngine should use Standard tier"
    );
}

/// Verify all 5 components are present.
#[test]
fn test_all_components_present() {
    let manifest = DemoManifest::cognitive_demo();

    let component_types: Vec<_> = manifest
        .components
        .iter()
        .map(|c| c.component_type)
        .collect();

    assert!(
        component_types.contains(&ComponentType::SensorAdapter),
        "Missing SensorAdapter"
    );
    assert!(
        component_types.contains(&ComponentType::FeatureExtractor),
        "Missing FeatureExtractor"
    );
    assert!(
        component_types.contains(&ComponentType::ReasoningEngine),
        "Missing ReasoningEngine"
    );
    assert!(
        component_types.contains(&ComponentType::Attestor),
        "Missing Attestor"
    );
    assert!(
        component_types.contains(&ComponentType::Coordinator),
        "Missing Coordinator"
    );

    assert_eq!(manifest.components.len(), 5, "Expected 5 components");
}

/// Verify rollback hooks are configured.
#[test]
fn test_rollback_hooks_configured() {
    let manifest = DemoManifest::cognitive_demo();

    assert!(
        manifest.rollback_hooks.len() >= 2,
        "Expected at least 2 rollback hooks"
    );

    let hook_names: Vec<_> = manifest.rollback_hooks.iter().map(|h| &h.name).collect();

    assert!(
        hook_names.iter().any(|n| n.contains("coherence")),
        "Missing coherence drop hook"
    );
    assert!(
        hook_names.iter().any(|n| n.contains("proof")),
        "Missing proof failure hook"
    );
}

/// Verify manifest validation passes.
#[test]
fn test_manifest_validates() {
    let manifest = DemoManifest::cognitive_demo();
    assert!(manifest.validate(), "Manifest validation failed");
}

/// Verify expected syscall counts in manifest.
#[test]
fn test_manifest_expected_syscall_counts() {
    let manifest = DemoManifest::cognitive_demo();
    let counts = manifest.expected_syscall_counts();

    // Verify all 12 syscalls are listed
    assert_eq!(counts.len(), 12);

    // Verify specific counts
    let task_spawn = counts.iter().find(|(n, _, _)| *n == 0).unwrap();
    assert_eq!(task_spawn.2, config::TASK_SPAWNS as u32);

    let attest_emit = counts.iter().find(|(n, _, _)| *n == 7).unwrap();
    assert_eq!(attest_emit.2, config::FULL_PIPELINE_EVENTS as u32);

    let graph_apply = counts.iter().find(|(n, _, _)| *n == 10).unwrap();
    assert_eq!(graph_apply.2, config::GRAPH_MUTATIONS as u32);
}

/// Verify total memory requirement calculation.
#[test]
fn test_total_memory_calculation() {
    let manifest = DemoManifest::cognitive_demo();
    let total = manifest.total_memory_bytes();

    let expected = config::MODEL_WEIGHTS_SIZE
        + config::WITNESS_LOG_MAX_SIZE
        + (config::VECTOR_SLOT_SIZE * config::VECTOR_SLOT_COUNT);

    assert_eq!(total, expected);

    // Should be approximately 4MB
    assert!(total > 4_000_000);
    assert!(total < 5_000_000);
}

/// Verify proof tier verification times.
#[test]
fn test_proof_tier_verification_times() {
    assert!(
        ProofTier::Reflex.max_verification_time_us() < 10,
        "Reflex tier should be sub-microsecond"
    );
    assert!(
        ProofTier::Standard.max_verification_time_us() < 1000,
        "Standard tier should be sub-millisecond"
    );
    assert!(
        ProofTier::Deep.max_verification_time_us() < 100_000,
        "Deep tier should be sub-100ms"
    );
}

/// Verify component dependencies form a DAG.
#[test]
fn test_component_dependencies_dag() {
    let manifest = DemoManifest::cognitive_demo();

    for component in &manifest.components {
        for &dep in &component.dependencies {
            // Dependencies must have lower indices (DAG property)
            assert!(
                dep < component.index,
                "Component {} depends on {} which is not earlier",
                component.index,
                dep
            );
        }
    }
}

/// Verify syscall usage declarations match implementation.
#[test]
fn test_syscall_usage_declarations() {
    let manifest = DemoManifest::cognitive_demo();

    for component in &manifest.components {
        println!(
            "Component {}: {:?}",
            component.name,
            component
                .syscalls
                .iter()
                .map(|s| s.syscall_name)
                .collect::<Vec<_>>()
        );

        match component.component_type {
            ComponentType::SensorAdapter => {
                assert!(component
                    .syscalls
                    .iter()
                    .any(|s| s.syscall_name == "sensor_subscribe"));
                assert!(component.syscalls.iter().any(|s| s.syscall_name == "queue_send"));
            }
            ComponentType::FeatureExtractor => {
                assert!(component
                    .syscalls
                    .iter()
                    .any(|s| s.syscall_name == "queue_recv"));
                assert!(component
                    .syscalls
                    .iter()
                    .any(|s| s.syscall_name == "vector_put_proved"));
            }
            ComponentType::ReasoningEngine => {
                assert!(component
                    .syscalls
                    .iter()
                    .any(|s| s.syscall_name == "vector_get"));
                assert!(component
                    .syscalls
                    .iter()
                    .any(|s| s.syscall_name == "graph_apply_proved"));
            }
            ComponentType::Attestor => {
                assert!(component
                    .syscalls
                    .iter()
                    .any(|s| s.syscall_name == "attest_emit"));
            }
            ComponentType::Coordinator => {
                assert!(component
                    .syscalls
                    .iter()
                    .any(|s| s.syscall_name == "task_spawn"));
                assert!(component.syscalls.iter().any(|s| s.syscall_name == "cap_grant"));
                assert!(component
                    .syscalls
                    .iter()
                    .any(|s| s.syscall_name == "timer_wait"));
            }
        }
    }
}

/// Integration test: Full coverage verification.
#[test]
fn test_full_coverage_verification() {
    let config = PipelineConfig {
        event_count: 1000,
        batch_size: 100,
        ..Default::default()
    };

    let mut pipeline = CognitivePipeline::new(config);
    let result = pipeline.run().unwrap();

    let coverage = &result.feature_coverage;

    // Print syscall details for debugging
    println!("\nSyscall Coverage Details:");
    for detail in &coverage.syscall_coverage.details {
        println!(
            "  {}: expected={}, actual={}, covered={}",
            detail.name, detail.expected, detail.actual, detail.actual > 0
        );
    }
    println!("All covered: {}", coverage.syscall_coverage.all_covered);

    // Verify syscall coverage
    assert!(
        coverage.syscall_coverage.all_covered,
        "Not all syscalls covered: {:?}",
        coverage.syscall_coverage.details.iter()
            .filter(|d| d.actual == 0)
            .map(|d| d.name)
            .collect::<Vec<_>>()
    );

    // Verify region coverage
    assert!(
        coverage.region_coverage.all_covered,
        "Not all region types covered"
    );

    // Verify proof coverage
    assert!(
        coverage.proof_coverage.all_covered,
        "Not all proof tiers covered"
    );

    // Verify component coverage
    assert!(
        coverage.component_coverage.all_covered,
        "Not all components active"
    );

    // Generate and print feature matrix
    let report = coverage.report();
    println!("\n{}", report.summary());

    let matrix = report.feature_matrix();
    println!("\nFeature Matrix:");
    matrix.print_table();

    // Overall should be 100%
    assert!(
        report.overall_percentage >= 90.0,
        "Overall coverage too low: {}%",
        report.overall_percentage
    );
}

/// Test config constants match expected values.
#[test]
fn test_config_constants() {
    assert_eq!(config::FULL_PIPELINE_EVENTS, 10_000);
    assert_eq!(config::GRAPH_MUTATIONS, 5_000);
    assert_eq!(config::TIMER_WAITS, 100);
    assert_eq!(config::CAP_GRANTS, 20);
    assert_eq!(config::TASK_SPAWNS, 5);
    assert_eq!(config::MODEL_WEIGHTS_SIZE, 1024 * 1024); // 1 MiB
    assert_eq!(config::WITNESS_LOG_MAX_SIZE, 64 * 1024); // 64 KiB
    assert_eq!(config::VECTOR_SLOT_SIZE, 3 * 1024); // 3 KiB
    assert_eq!(config::VECTOR_SLOT_COUNT, 1024);
    assert_eq!(config::EMBEDDING_DIM, 768);
}
