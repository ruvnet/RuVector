//! Full pipeline benchmarks.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ruvector_decompiler::{decompile, DecompileConfig};

/// Generate a synthetic minified JS bundle of approximately `target_bytes` size.
fn generate_bundle(target_bytes: usize) -> String {
    let mut bundle = String::with_capacity(target_bytes + 256);
    let mut var_idx = 0u32;

    while bundle.len() < target_bytes {
        let name = format!("_{:x}", var_idx);
        match var_idx % 4 {
            0 => {
                bundle.push_str(&format!(
                    r#"var {}=function(){{return"str_{}"+"path/to/module_{}"}};"#,
                    name, var_idx, var_idx
                ));
            }
            1 => {
                let prev = format!("_{:x}", var_idx.saturating_sub(1));
                bundle.push_str(&format!(
                    "const {}=function(x){{return {}(x)+x.toString()}};",
                    name, prev
                ));
            }
            2 => {
                bundle.push_str(&format!(
                    r#"var {}=class{{constructor(){{this.name="cls_{}";this.type="entity"}}get(){{return this.name}}}};{}"#,
                    name, var_idx, ""
                ));
            }
            3 => {
                bundle.push_str(&format!(
                    r#"var {}={{value:{},handler:function(a){{return a+{}}},description:"item_{}"}};"#,
                    name, var_idx, var_idx, var_idx
                ));
            }
            _ => unreachable!(),
        }
        var_idx += 1;
    }

    bundle
}

fn bench_full_pipeline(c: &mut Criterion) {
    let sizes: &[(usize, &str)] = &[
        (1_000, "1KB"),
        (10_000, "10KB"),
        (100_000, "100KB"),
    ];

    let mut group = c.benchmark_group("pipeline");
    group.sample_size(10);

    for &(size, label) in sizes {
        let bundle = generate_bundle(size);

        // With all features enabled.
        group.bench_with_input(
            BenchmarkId::new("full_pipeline", label),
            &bundle,
            |b, source| {
                let config = DecompileConfig::default();
                b.iter(|| {
                    let result = decompile(black_box(source), &config);
                    black_box(result).ok();
                });
            },
        );

        // Parse-only (no witness, no source maps).
        group.bench_with_input(
            BenchmarkId::new("parse_only", label),
            &bundle,
            |b, source| {
                let config = DecompileConfig {
                    generate_source_maps: false,
                    generate_witness: false,
                    ..DecompileConfig::default()
                };
                b.iter(|| {
                    let result = decompile(black_box(source), &config);
                    black_box(result).ok();
                });
            },
        );
    }

    group.finish();
}

fn bench_pipeline_phases(c: &mut Criterion) {
    let bundle = generate_bundle(100_000);
    let mut group = c.benchmark_group("phases_100KB");
    group.sample_size(10);

    // Phase 1: Parse
    group.bench_function("parse", |b| {
        b.iter(|| {
            let result = ruvector_decompiler::parser::parse_bundle(black_box(&bundle));
            black_box(result).ok();
        });
    });

    // Phase 2: Graph
    let decls = ruvector_decompiler::parser::parse_bundle(&bundle).unwrap();
    let decls_clone = decls.clone();
    group.bench_function("graph", |b| {
        b.iter(|| {
            let graph = ruvector_decompiler::graph::build_reference_graph(
                black_box(decls_clone.clone()),
            );
            black_box(graph);
        });
    });

    // Phase 3: Partition
    let graph = ruvector_decompiler::graph::build_reference_graph(decls);
    group.bench_function("partition", |b| {
        b.iter(|| {
            let result = ruvector_decompiler::partitioner::partition_modules(
                black_box(&graph),
                Some(5),
            );
            black_box(result).ok();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_full_pipeline, bench_pipeline_phases);
criterion_main!(benches);
