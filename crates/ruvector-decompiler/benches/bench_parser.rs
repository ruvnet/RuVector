//! Parser benchmarks for various bundle sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ruvector_decompiler::parser::parse_bundle;

/// Generate a synthetic minified JS bundle of approximately `target_bytes` size.
fn generate_bundle(target_bytes: usize) -> String {
    let mut bundle = String::with_capacity(target_bytes + 256);
    let mut var_idx = 0u32;

    while bundle.len() < target_bytes {
        let name = format!("_{:x}", var_idx);
        // Mix of declaration types to exercise different parser paths.
        match var_idx % 4 {
            0 => {
                // var with function body and string literals
                bundle.push_str(&format!(
                    r#"var {}=function(){{return"str_{}"+"path/to/module_{}"}};"#,
                    name, var_idx, var_idx
                ));
            }
            1 => {
                // const with arrow-like expression referencing previous var
                let prev = format!("_{:x}", var_idx.saturating_sub(1));
                bundle.push_str(&format!(
                    "const {}=function(x){{return {}(x)+x.toString()}};",
                    name, prev
                ));
            }
            2 => {
                // class with methods and property accesses
                bundle.push_str(&format!(
                    r#"var {}=class{{constructor(){{this.name="cls_{}";this.type="entity"}}get(){{return this.name}}}};{}"#,
                    name, var_idx, ""
                ));
            }
            3 => {
                // Simple variable with object literal
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

fn bench_parse(c: &mut Criterion) {
    let sizes: &[(usize, &str)] = &[
        (1_000, "1KB"),
        (10_000, "10KB"),
        (100_000, "100KB"),
        (1_000_000, "1MB"),
    ];

    let mut group = c.benchmark_group("parser");
    group.sample_size(20);

    for &(size, label) in sizes {
        let bundle = generate_bundle(size);
        let actual_size = bundle.len();

        group.bench_with_input(
            BenchmarkId::new("parse_bundle", label),
            &bundle,
            |b, source| {
                b.iter(|| {
                    let result = parse_bundle(black_box(source));
                    black_box(result).ok();
                });
            },
        );

        // Also benchmark just the declaration extraction (without error handling).
        let decl_count = parse_bundle(&bundle).map(|d| d.len()).unwrap_or(0);
        println!(
            "  {} bundle: {} actual bytes, {} declarations",
            label, actual_size, decl_count
        );
    }

    group.finish();
}

criterion_group!(benches, bench_parse);
criterion_main!(benches);
