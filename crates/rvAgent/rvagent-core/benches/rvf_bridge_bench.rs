//! Criterion benchmarks for rvf_bridge: witness header serialization,
//! mount table operations, path parsing, and manifest filtering (ADR-106).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use rvagent_core::rvf_bridge::{
    GovernanceMode, MountTable, PolicyCheck, RvfBridgeConfig, RvfManifest, RvfManifestEntry,
    RvfManifestEntryType, RvfToolCallEntry, RvfVerifyStatus, RvfWitnessHeader, TaskOutcome,
    WIT_HAS_TRACE, WIT_SIGNED,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sample_manifest(name: &str, tool_count: usize, skill_count: usize) -> RvfManifest {
    let mut manifest = RvfManifest::new(name, "0.1.0");
    for i in 0..tool_count {
        manifest.entries.push(RvfManifestEntry {
            name: format!("tool_{}", i),
            entry_type: RvfManifestEntryType::Tool,
            description: format!("Tool {} description", i),
            version: "0.1.0".into(),
            parameters_schema: Some(serde_json::json!({"type": "object"})),
            content_hash: None,
            required_capabilities: vec![],
        });
    }
    for i in 0..skill_count {
        manifest.entries.push(RvfManifestEntry {
            name: format!("skill_{}", i),
            entry_type: RvfManifestEntryType::Skill,
            description: format!("Skill {} description", i),
            version: "0.1.0".into(),
            parameters_schema: None,
            content_hash: None,
            required_capabilities: vec!["execute".into()],
        });
    }
    manifest
}

fn make_witness_header() -> RvfWitnessHeader {
    RvfWitnessHeader {
        version: 1,
        flags: WIT_SIGNED | WIT_HAS_TRACE,
        task_id: [0x42; 16],
        policy_hash: [0xAA; 8],
        created_ns: 1_700_000_000_000_000_000,
        outcome: TaskOutcome::Solved,
        governance_mode: GovernanceMode::Approved,
        tool_call_count: 25,
        total_cost_microdollars: 15_000,
        total_latency_ms: 4_500,
        total_tokens: 8_000,
        retry_count: 1,
        section_count: 3,
        total_bundle_size: 4096,
    }
}

// ---------------------------------------------------------------------------
// Benchmark: WitnessHeader serialization roundtrip
// ---------------------------------------------------------------------------

fn bench_witness_header_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("witness_header");

    let header = make_witness_header();

    group.bench_function("to_bytes", |b| {
        b.iter(|| {
            let bytes = black_box(&header).to_bytes();
            black_box(bytes);
        })
    });

    let bytes = header.to_bytes();
    group.bench_function("from_bytes", |b| {
        b.iter(|| {
            let hdr = RvfWitnessHeader::from_bytes(black_box(&bytes)).unwrap();
            black_box(hdr);
        })
    });

    group.bench_function("roundtrip", |b| {
        b.iter(|| {
            let bytes = black_box(&header).to_bytes();
            let decoded = RvfWitnessHeader::from_bytes(&bytes).unwrap();
            black_box(decoded);
        })
    });

    // Compare with serde JSON roundtrip
    group.bench_function("serde_json_roundtrip", |b| {
        b.iter(|| {
            let json = serde_json::to_vec(black_box(&header)).unwrap();
            let decoded: RvfWitnessHeader = serde_json::from_slice(&json).unwrap();
            black_box(decoded);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: MountTable operations
// ---------------------------------------------------------------------------

fn bench_mount_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("mount_table");

    // Mount operation
    group.bench_function("mount_single", |b| {
        b.iter(|| {
            let mut table = MountTable::new();
            let manifest = sample_manifest("pkg", 5, 2);
            let handle = table.mount(black_box(manifest), RvfVerifyStatus::SignatureValid);
            black_box(handle);
        })
    });

    // Lookup by handle in tables of varying size
    for count in [1, 10, 50] {
        let mut table = MountTable::new();
        let mut handles = Vec::new();
        for i in 0..count {
            let manifest = sample_manifest(&format!("pkg-{}", i), 5, 2);
            handles.push(table.mount(manifest, RvfVerifyStatus::SignatureValid));
        }
        // Look up the last handle (worst case for linear scan)
        let target = *handles.last().unwrap();

        group.bench_with_input(
            BenchmarkId::new("get_by_handle", count),
            &(table.clone(), target),
            |b, (table, target)| {
                b.iter(|| {
                    let entry = table.get(black_box(*target));
                    black_box(entry);
                })
            },
        );
    }

    // Lookup by name (linear scan through entries)
    for count in [1, 10, 50] {
        let mut table = MountTable::new();
        for i in 0..count {
            let manifest = sample_manifest(&format!("pkg-{}", i), 5, 2);
            table.mount(manifest, RvfVerifyStatus::SignatureValid);
        }
        let target_name = format!("pkg-{}", count - 1);

        group.bench_with_input(
            BenchmarkId::new("find_by_name_linear", count),
            &(table.clone(), target_name.clone()),
            |b, (table, name)| {
                b.iter(|| {
                    let found = table.list().iter().find(|e| e.package_name == *name);
                    black_box(found);
                })
            },
        );
    }

    // all_tools collection
    for count in [1, 10, 50] {
        let mut table = MountTable::new();
        for i in 0..count {
            let manifest = sample_manifest(&format!("pkg-{}", i), 5, 2);
            table.mount(manifest, RvfVerifyStatus::SignatureValid);
        }

        group.bench_with_input(BenchmarkId::new("all_tools", count), &table, |b, table| {
            b.iter(|| {
                let tools = table.all_tools();
                black_box(tools);
            })
        });
    }

    // Unmount (retain operation)
    for count in [10, 50] {
        group.bench_with_input(
            BenchmarkId::new("unmount_middle", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    let mut table = MountTable::new();
                    let mut handles = Vec::new();
                    for i in 0..count {
                        let manifest = sample_manifest(&format!("pkg-{}", i), 3, 1);
                        handles.push(table.mount(manifest, RvfVerifyStatus::SignatureValid));
                    }
                    // Unmount from the middle
                    let target = handles[count / 2];
                    table.unmount(black_box(target));
                    black_box(table);
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Manifest filtering (tools/skills)
// ---------------------------------------------------------------------------

fn bench_manifest_filtering(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifest_filtering");

    for (tools, skills) in [(5, 2), (20, 10), (50, 25)] {
        let manifest = sample_manifest("pkg", tools, skills);

        group.bench_with_input(
            BenchmarkId::new("tools", tools + skills),
            &manifest,
            |b, manifest| {
                b.iter(|| {
                    let tools = manifest.tools();
                    black_box(tools);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("skills", tools + skills),
            &manifest,
            |b, manifest| {
                b.iter(|| {
                    let skills = manifest.skills();
                    black_box(skills);
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: ToolCallEntry serialization
// ---------------------------------------------------------------------------

fn bench_tool_call_entry(c: &mut Criterion) {
    let mut group = c.benchmark_group("tool_call_entry");

    let entry = RvfToolCallEntry {
        action: "read_file".into(),
        args_hash: [0x11; 8],
        result_hash: [0x22; 8],
        latency_ms: 150,
        cost_microdollars: 500,
        tokens: 200,
        policy_check: PolicyCheck::Allowed,
    };

    group.bench_function("serde_json_roundtrip", |b| {
        b.iter(|| {
            let json = serde_json::to_vec(black_box(&entry)).unwrap();
            let back: RvfToolCallEntry = serde_json::from_slice(&json).unwrap();
            black_box(back);
        })
    });

    // Batch of entries
    let entries: Vec<RvfToolCallEntry> = (0..50)
        .map(|i| RvfToolCallEntry {
            action: format!("tool_{}", i),
            args_hash: [i as u8; 8],
            result_hash: [(i * 2) as u8; 8],
            latency_ms: 100 + i * 10,
            cost_microdollars: 50 * i,
            tokens: 20 * i,
            policy_check: PolicyCheck::Allowed,
        })
        .collect();

    group.bench_function("serde_json_batch_50", |b| {
        b.iter(|| {
            let json = serde_json::to_vec(black_box(&entries)).unwrap();
            let back: Vec<RvfToolCallEntry> = serde_json::from_slice(&json).unwrap();
            black_box(back);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: RvfBridgeConfig serialization
// ---------------------------------------------------------------------------

fn bench_config_serde(c: &mut Criterion) {
    let config = RvfBridgeConfig {
        enabled: true,
        package_dir: Some("/opt/rvf/packages".into()),
        verify_signatures: true,
        rvf_witness: true,
        governance_mode: GovernanceMode::Autonomous,
    };

    c.bench_function("bridge_config_serde_roundtrip", |b| {
        b.iter(|| {
            let json = serde_json::to_vec(black_box(&config)).unwrap();
            let back: RvfBridgeConfig = serde_json::from_slice(&json).unwrap();
            black_box(back);
        })
    });
}

criterion_group!(
    benches,
    bench_witness_header_serialization,
    bench_mount_table,
    bench_manifest_filtering,
    bench_tool_call_entry,
    bench_config_serde,
);
criterion_main!(benches);
