//! Cognitive Demo Example
//!
//! This example demonstrates the complete cognitive pipeline with all
//! RuVix kernel features. Run with:
//!
//! ```bash
//! cargo run --example cognitive_demo
//! ```

use ruvix_demo::{
    config,
    manifest::DemoManifest,
    pipeline::{CognitivePipeline, PipelineConfig, PipelineState},
};
use std::time::Instant;

fn main() {
    println!("=== RuVix Cognitive Demo ===\n");

    // Print manifest information
    print_manifest_info();

    // Run the pipeline
    run_pipeline();
}

fn print_manifest_info() {
    let manifest = DemoManifest::cognitive_demo();

    println!("Manifest: cognitive_demo.rvf");
    println!(
        "Version: {}.{}.{}",
        manifest.version.major, manifest.version.minor, manifest.version.patch
    );
    println!();

    println!("Components ({}):", manifest.components.len());
    for comp in &manifest.components {
        println!(
            "  [{}] {} - entry: {}",
            comp.index, comp.name, comp.entry_point
        );
        print!("      syscalls: ");
        for (i, syscall) in comp.syscalls.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", syscall.syscall_name);
        }
        println!();
    }
    println!();

    println!("Memory Regions ({}):", manifest.regions.len());
    for region in &manifest.regions {
        println!(
            "  [{}] {} - {} ({} bytes)",
            region.index,
            region.name,
            region.policy.as_str(),
            region.region_type.size_bytes()
        );
    }
    println!(
        "  Total memory: {} bytes ({:.2} MiB)",
        manifest.total_memory_bytes(),
        manifest.total_memory_bytes() as f64 / (1024.0 * 1024.0)
    );
    println!();

    println!("Proof Policy:");
    println!(
        "  Vector mutations: {:?}",
        manifest.proof_policy.vector_mutations
    );
    println!(
        "  Graph mutations: {:?}",
        manifest.proof_policy.graph_mutations
    );
    println!(
        "  Structural changes: {:?}",
        manifest.proof_policy.structural_changes
    );
    println!();

    println!("Rollback Hooks ({}):", manifest.rollback_hooks.len());
    for hook in &manifest.rollback_hooks {
        println!("  {} -> {}", hook.name, hook.function);
    }
    println!();

    println!("Expected Syscall Counts:");
    println!(
        "| {:<20} | {:<15} | {:>10} |",
        "Syscall", "Component", "Count"
    );
    println!("|{:-<22}|{:-<17}|{:-<12}|", "", "", "");
    for (num, name, count) in manifest.expected_syscall_counts() {
        let component = match num {
            0 => "Coordinator",
            1 => "Coordinator",
            2 => "Boot",
            3 | 4 => "All",
            5 => "Coordinator",
            6 => "Boot",
            7 => "Attestor",
            8 => "ReasoningEngine",
            9 => "FeatureExtractor",
            10 => "ReasoningEngine",
            11 => "SensorAdapter",
            _ => "Unknown",
        };
        println!("| {:<20} | {:<15} | {:>10} |", name, component, count);
    }
    println!();
}

fn run_pipeline() {
    println!("Running Pipeline...\n");

    let config = PipelineConfig {
        event_count: config::FULL_PIPELINE_EVENTS as u64,
        batch_size: 500,
        verbose: false,
        seed: 0xDEADBEEF,
    };

    println!("Configuration:");
    println!("  Events: {}", config.event_count);
    println!("  Batch size: {}", config.batch_size);
    println!("  Seed: 0x{:X}", config.seed);
    println!();

    let start = Instant::now();
    let mut pipeline = CognitivePipeline::new(config);

    // Initialize
    println!("Initializing...");
    pipeline.initialize().expect("Failed to initialize");
    pipeline
        .setup_coordinator()
        .expect("Failed to setup coordinator");
    println!("  Regions mapped: {}", pipeline.kernel().stats.region_map);
    println!(
        "  Sensor subscribed: {}",
        pipeline.kernel().stats.sensor_subscribe
    );
    println!("  RVF mounted: {}", pipeline.kernel().stats.rvf_mount);

    // Process events
    println!("\nProcessing events...");
    let mut batch_count = 0;
    while pipeline.state() == PipelineState::Running {
        pipeline.process_batch().expect("Batch failed");
        batch_count += 1;

        // Progress indicator every 10 batches
        if batch_count % 10 == 0 {
            let stats = pipeline.get_syscall_stats();
            print!(
                "\r  Processed {} events ({} vectors, {} mutations)...",
                stats.queue_send / 2,
                stats.vector_put_proved,
                stats.graph_apply_proved
            );
        }
    }
    println!();

    // Complete timer waits
    while pipeline.kernel().stats.timer_wait < config::TIMER_WAITS as u64 {
        // Would call coordinator.wait_timer() here
    }

    let elapsed = start.elapsed();

    // Results
    println!("\n=== Results ===\n");

    let stats = pipeline.get_syscall_stats();
    let coverage = pipeline.get_feature_coverage();

    println!("Execution Time: {:?}", elapsed);
    println!(
        "Events/second: {:.0}",
        stats.queue_send as f64 / 2.0 / elapsed.as_secs_f64()
    );
    println!();

    println!("Syscall Statistics:");
    println!("| {:<20} | {:>10} |", "Syscall", "Count");
    println!("|{:-<22}|{:-<12}|", "", "");
    for (_, name, count) in stats.as_list() {
        println!("| {:<20} | {:>10} |", name, count);
    }
    println!("| {:<20} | {:>10} |", "TOTAL", stats.total());
    println!();

    println!("Feature Coverage:");
    let report = coverage.report();
    println!(
        "  Syscalls: {}/{} ({:.1}%)",
        report.syscalls.covered,
        report.syscalls.total,
        report.syscalls.percentage()
    );
    println!(
        "  Regions: {} types ({:.1}%)",
        (report.regions.immutable as u32)
            + (report.regions.append_only as u32)
            + (report.regions.slab as u32),
        report.regions.percentage()
    );
    println!(
        "  Proofs: {} tiers ({:.1}%)",
        (report.proofs.reflex as u32)
            + (report.proofs.standard as u32)
            + (report.proofs.deep as u32),
        report.proofs.percentage()
    );
    println!(
        "  Components: {} active ({:.1}%)",
        (report.components.sensor_adapter as u32)
            + (report.components.feature_extractor as u32)
            + (report.components.reasoning_engine as u32)
            + (report.components.attestor as u32)
            + (report.components.coordinator as u32),
        report.components.percentage()
    );
    println!();
    println!("  Overall: {:.1}%", report.overall_percentage);
    println!("  Fully Covered: {}", report.fully_covered);

    // Final state check
    println!();
    if pipeline.state() == PipelineState::Completed && stats.all_covered() {
        println!("SUCCESS: All RuVix kernel features demonstrated!");
    } else {
        println!("WARNING: Some features may not be fully exercised");
        let uncovered = stats.uncovered_syscalls();
        if !uncovered.is_empty() {
            println!("  Uncovered syscalls: {:?}", uncovered);
        }
    }
}
