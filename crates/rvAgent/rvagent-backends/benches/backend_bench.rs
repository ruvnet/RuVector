//! Benchmarks for rvagent-backends (ADR-103 A9).
//!
//! Tests:
//! - format_content_with_line_numbers (100, 1000, 10000 lines)
//! - Path resolution latency
//! - Grep literal search performance

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::path::PathBuf;

fn generate_content(num_lines: usize) -> String {
    (0..num_lines)
        .map(|i| format!("line {} with some content for benchmarking purposes here", i))
        .collect::<Vec<_>>()
        .join("\n")
}

fn bench_format_content(c: &mut Criterion) {
    let content_100 = generate_content(100);
    let content_1000 = generate_content(1000);
    let content_10000 = generate_content(10000);

    c.bench_function("format_content_100_lines", |b| {
        b.iter(|| {
            rvagent_backends::utils::format_content_with_line_numbers(
                black_box(&content_100),
                1,
                2000,
            )
        })
    });

    c.bench_function("format_content_1000_lines", |b| {
        b.iter(|| {
            rvagent_backends::utils::format_content_with_line_numbers(
                black_box(&content_1000),
                1,
                2000,
            )
        })
    });

    c.bench_function("format_content_10000_lines", |b| {
        b.iter(|| {
            rvagent_backends::utils::format_content_with_line_numbers(
                black_box(&content_10000),
                1,
                2000,
            )
        })
    });
}

fn bench_path_resolution(c: &mut Criterion) {
    let backend = rvagent_backends::FilesystemBackend::new(PathBuf::from("/tmp"));

    c.bench_function("resolve_path_simple", |b| {
        b.iter(|| backend.resolve_path(black_box("src/main.rs")))
    });

    c.bench_function("resolve_path_nested", |b| {
        b.iter(|| backend.resolve_path(black_box("src/deep/nested/path/to/file.rs")))
    });

    c.bench_function("resolve_path_absolute_in_virtual", |b| {
        b.iter(|| backend.resolve_path(black_box("/absolute/path/file.txt")))
    });
}

fn bench_grep_literal(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Set up a state backend with files for grepping
    let backend = rvagent_backends::StateBackend::new();
    rt.block_on(async {
        for i in 0..100 {
            let content = (0..50)
                .map(|j| {
                    if j % 10 == 0 {
                        format!("fn process_item_{}_{}() {{}}", i, j)
                    } else {
                        format!("let x_{} = {};", j, j * 42)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            backend
                .write_file(&format!("src/file_{}.rs", i), &content)
                .await;
        }
    });

    c.bench_function("grep_literal_100_files", |b| {
        b.iter(|| {
            rt.block_on(async {
                backend
                    .grep(black_box("fn process_item"), None, None)
                    .await
                    .unwrap()
            })
        })
    });

    c.bench_function("grep_literal_no_match", |b| {
        b.iter(|| {
            rt.block_on(async {
                backend
                    .grep(black_box("NONEXISTENT_PATTERN_XYZ"), None, None)
                    .await
                    .unwrap()
            })
        })
    });
}

criterion_group!(
    benches,
    bench_format_content,
    bench_path_resolution,
    bench_grep_literal
);
criterion_main!(benches);
