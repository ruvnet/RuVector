//! Criterion benchmarks for rvagent-backends: line formatting, path resolution,
//! grep, and unicode detection (ADR-103 A9).

use criterion::{criterion_group, criterion_main, Criterion, black_box, BenchmarkId};

use rvagent_backends::utils::{
    contains_traversal, format_content_with_line_numbers, is_safe_path_component,
};
use rvagent_backends::unicode_security::{
    detect_dangerous_unicode, strip_dangerous_unicode, validate_ascii_identifier,
    detect_confusables, check_url_safety,
};

// ---------------------------------------------------------------------------
// Helpers — generate content at various sizes
// ---------------------------------------------------------------------------

fn make_lines(n: usize) -> String {
    let mut content = String::with_capacity(n * 80);
    for i in 0..n {
        if i > 0 {
            content.push('\n');
        }
        // Realistic source-code-like lines (~60-80 chars)
        content.push_str(&format!(
            "    pub fn function_{}(arg: &str) -> Result<String, Error> {{ /* body */ }}",
            i
        ));
    }
    content
}

fn make_content_bytes(target_bytes: usize) -> String {
    let line = "use std::collections::HashMap; // typical import line padding to ~70 chars here\n";
    let repeats = target_bytes / line.len() + 1;
    line.repeat(repeats)
}

// ---------------------------------------------------------------------------
// Benchmark: format_content_with_line_numbers (ADR-103 A7)
// ---------------------------------------------------------------------------

fn bench_format_content_with_line_numbers(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_line_numbers");

    for line_count in [100, 1000, 10_000] {
        let content = make_lines(line_count);

        // Optimized: pre-allocated single String::with_capacity
        group.bench_with_input(
            BenchmarkId::new("optimized", line_count),
            &content,
            |b, content| {
                b.iter(|| {
                    let result =
                        format_content_with_line_numbers(black_box(content), 1, 2000);
                    black_box(result);
                })
            },
        );

        // Naive: per-line String allocation and push
        group.bench_with_input(
            BenchmarkId::new("naive_push_per_line", line_count),
            &content,
            |b, content| {
                b.iter(|| {
                    let lines: Vec<&str> = content.lines().collect();
                    let mut out = String::new();
                    for (i, line) in lines.iter().enumerate() {
                        if i > 0 {
                            out.push('\n');
                        }
                        out.push_str(&format!("{:>6}\t{}", i + 1, line));
                    }
                    black_box(out);
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: path resolution / safety checking
// ---------------------------------------------------------------------------

fn bench_path_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("path_resolution");

    let paths = vec![
        ("simple_file", "main.rs"),
        ("nested_path", "src/handlers/auth/middleware.rs"),
        ("with_dots", "src/config.prod.yaml"),
        ("traversal_attack", "../../../etc/passwd"),
        ("windows_traversal", "foo\\..\\bar"),
        ("deep_nesting", "a/b/c/d/e/f/g/h/i/j/k/l.rs"),
        ("unicode_path", "src/caf\u{00E9}/main.rs"),
    ];

    // contains_traversal checks
    for (name, path) in &paths {
        group.bench_with_input(
            BenchmarkId::new("contains_traversal", name),
            path,
            |b, path| {
                b.iter(|| {
                    let result = contains_traversal(black_box(path));
                    black_box(result);
                })
            },
        );
    }

    // is_safe_path_component on individual segments
    let components = vec![
        ("normal", "src"),
        ("dotdot", ".."),
        ("dot", "."),
        ("empty", ""),
        ("with_null", "file\0.rs"),
        ("long_name", "a_very_long_directory_name_that_might_appear_in_real_projects"),
    ];

    for (name, component) in &components {
        group.bench_with_input(
            BenchmarkId::new("is_safe_component", name),
            component,
            |b, comp| {
                b.iter(|| {
                    let result = is_safe_path_component(black_box(comp));
                    black_box(result);
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: grep (literal string search) at various content sizes
// ---------------------------------------------------------------------------

fn bench_grep_literal(c: &mut Criterion) {
    let mut group = c.benchmark_group("grep_literal");

    for (label, size_bytes) in [("10KB", 10_000), ("100KB", 100_000), ("1MB", 1_000_000)] {
        let content = make_content_bytes(size_bytes);
        let lines: Vec<&str> = content.lines().collect();

        // Pattern that appears frequently (should match most lines)
        group.bench_with_input(
            BenchmarkId::new("frequent_match", label),
            &lines,
            |b, lines| {
                b.iter(|| {
                    let mut matches = Vec::new();
                    for (i, line) in lines.iter().enumerate() {
                        if line.contains(black_box("HashMap")) {
                            matches.push((i + 1, *line));
                        }
                    }
                    black_box(matches);
                })
            },
        );

        // Pattern that appears rarely (should match few/no lines)
        group.bench_with_input(
            BenchmarkId::new("rare_match", label),
            &lines,
            |b, lines| {
                b.iter(|| {
                    let mut matches = Vec::new();
                    for (i, line) in lines.iter().enumerate() {
                        if line.contains(black_box("XYZZY_NONEXISTENT_PATTERN")) {
                            matches.push((i + 1, *line));
                        }
                    }
                    black_box(matches);
                })
            },
        );

        // Pattern at end of line (worst case for naive contains)
        group.bench_with_input(
            BenchmarkId::new("end_of_line_match", label),
            &lines,
            |b, lines| {
                b.iter(|| {
                    let mut matches = Vec::new();
                    for (i, line) in lines.iter().enumerate() {
                        if line.contains(black_box("here")) {
                            matches.push((i + 1, *line));
                        }
                    }
                    black_box(matches);
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Unicode security detection
// ---------------------------------------------------------------------------

fn bench_unicode_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("unicode_detection");

    // Clean ASCII text — common case, should be fast
    let clean_ascii = "fn main() {\n    println!(\"Hello, world!\");\n}\n".repeat(100);
    group.bench_function("detect_dangerous/clean_ascii_4KB", |b| {
        b.iter(|| {
            let issues = detect_dangerous_unicode(black_box(&clean_ascii));
            black_box(issues);
        })
    });

    // Clean text with safe Unicode (accents, CJK)
    let safe_unicode = "caf\u{00E9} r\u{00E9}sum\u{00E9} na\u{00EF}ve \u{4F60}\u{597D} \u{3053}\u{3093}\u{306B}\u{3061}\u{306F}\n".repeat(100);
    group.bench_function("detect_dangerous/safe_unicode_4KB", |b| {
        b.iter(|| {
            let issues = detect_dangerous_unicode(black_box(&safe_unicode));
            black_box(issues);
        })
    });

    // Text with scattered dangerous codepoints (BiDi + zero-width)
    let mut dangerous_text = String::with_capacity(5000);
    for i in 0..100 {
        dangerous_text.push_str(&format!("line {} normal text ", i));
        if i % 10 == 0 {
            dangerous_text.push('\u{202E}'); // RTL override
        }
        if i % 15 == 0 {
            dangerous_text.push('\u{200B}'); // zero-width space
        }
        dangerous_text.push('\n');
    }
    group.bench_function("detect_dangerous/scattered_dangerous", |b| {
        b.iter(|| {
            let issues = detect_dangerous_unicode(black_box(&dangerous_text));
            black_box(issues);
        })
    });

    // Strip dangerous — clean text (no-op path)
    group.bench_function("strip_dangerous/clean_ascii", |b| {
        b.iter(|| {
            let result = strip_dangerous_unicode(black_box(&clean_ascii));
            black_box(result);
        })
    });

    // Strip dangerous — text with dangerous chars
    group.bench_function("strip_dangerous/with_dangerous", |b| {
        b.iter(|| {
            let result = strip_dangerous_unicode(black_box(&dangerous_text));
            black_box(result);
        })
    });

    // Confusable detection
    let mixed_text = "Hello \u{0410}\u{0412}\u{0421} world \u{0391}\u{0392} end".repeat(50);
    group.bench_function("detect_confusables/mixed_scripts", |b| {
        b.iter(|| {
            let results = detect_confusables(black_box(&mixed_text));
            black_box(results);
        })
    });

    // validate_ascii_identifier — valid names
    group.bench_function("validate_identifier/valid", |b| {
        b.iter(|| {
            let r1 = validate_ascii_identifier(black_box("my-skill-name"));
            let r2 = validate_ascii_identifier(black_box("deploy_prod_v2"));
            let r3 = validate_ascii_identifier(black_box("a"));
            black_box((r1, r2, r3));
        })
    });

    // validate_ascii_identifier — invalid names (various rejection paths)
    group.bench_function("validate_identifier/invalid", |b| {
        b.iter(|| {
            let r1 = validate_ascii_identifier(black_box(""));
            let r2 = validate_ascii_identifier(black_box("123abc"));
            let r3 = validate_ascii_identifier(black_box("Hello"));
            let r4 = validate_ascii_identifier(black_box("na\u{0441}me"));
            black_box((r1, r2, r3, r4));
        })
    });

    // URL safety checking
    group.bench_function("check_url_safety/safe", |b| {
        b.iter(|| {
            let result = check_url_safety(black_box("https://example.com/path/to/resource"));
            black_box(result);
        })
    });

    group.bench_function("check_url_safety/mixed_script", |b| {
        b.iter(|| {
            let result = check_url_safety(black_box("https://exam\u{0440}le.com/path"));
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_format_content_with_line_numbers,
    bench_path_resolution,
    bench_grep_literal,
    bench_unicode_detection
);
criterion_main!(benches);
