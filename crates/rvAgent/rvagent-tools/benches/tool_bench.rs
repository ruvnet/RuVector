//! Benchmarks for tool dispatch latency (ADR-103 A9).
//!
//! Measures:
//! - Enum dispatch overhead for each built-in tool
//! - AnyTool dispatch (builtin vs dynamic)
//! - Tool resolution by name
//! - format_content_with_line_numbers
//! - write_todos invocation

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rvagent_tools::*;
use std::collections::HashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Bench backend — minimal mock for benchmarking
// ---------------------------------------------------------------------------

struct BenchBackend {
    files: HashMap<String, String>,
}

impl BenchBackend {
    fn new() -> Self {
        let mut files = HashMap::new();
        files.insert(
            "/bench.txt".to_string(),
            "line1\nline2\nline3".to_string(),
        );
        Self { files }
    }
}

impl Backend for BenchBackend {
    fn ls_info(&self, _path: &str) -> Result<Vec<FileInfo>, String> {
        Ok(vec![FileInfo {
            name: "bench.txt".into(),
            file_type: "file".into(),
            permissions: "-rw-r--r--".into(),
            size: 17,
        }])
    }

    fn read(
        &self,
        path: &str,
        _offset: usize,
        _limit: usize,
    ) -> Result<String, String> {
        self.files
            .get(path)
            .cloned()
            .ok_or_else(|| "not found".into())
    }

    fn write(&self, _path: &str, _content: &str) -> WriteResult {
        WriteResult::default()
    }

    fn edit(
        &self,
        _path: &str,
        _old: &str,
        _new: &str,
        _all: bool,
    ) -> WriteResult {
        WriteResult {
            occurrences: Some(1),
            ..Default::default()
        }
    }

    fn glob_info(
        &self,
        _pattern: &str,
        _path: &str,
    ) -> Result<Vec<String>, String> {
        Ok(vec!["/bench.txt".to_string()])
    }

    fn grep_raw(
        &self,
        _pattern: &str,
        _path: Option<&str>,
        _include: Option<&str>,
    ) -> Result<Vec<GrepMatch>, String> {
        Ok(vec![GrepMatch {
            file: "/bench.txt".into(),
            line_number: 1,
            text: "line1".into(),
        }])
    }

    fn execute(
        &self,
        command: &str,
        _timeout: u32,
    ) -> Result<ExecuteResponse, String> {
        Ok(ExecuteResponse {
            output: format!("ok: {}", command),
            exit_code: 0,
        })
    }
}

fn bench_runtime() -> ToolRuntime {
    ToolRuntime::new(Arc::new(BenchBackend::new()))
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_builtin_dispatch(c: &mut Criterion) {
    let runtime = bench_runtime();
    let tools = builtin_tools();

    let mut group = c.benchmark_group("builtin_dispatch");

    group.bench_function("ls", |b| {
        let args = serde_json::json!({"path": "/"});
        b.iter(|| {
            black_box(tools[0].invoke(black_box(args.clone()), &runtime));
        });
    });

    group.bench_function("read_file", |b| {
        let args = serde_json::json!({"file_path": "/bench.txt"});
        b.iter(|| {
            black_box(tools[1].invoke(black_box(args.clone()), &runtime));
        });
    });

    group.bench_function("grep", |b| {
        let args = serde_json::json!({"pattern": "line1"});
        b.iter(|| {
            black_box(tools[5].invoke(black_box(args.clone()), &runtime));
        });
    });

    group.bench_function("glob", |b| {
        let args = serde_json::json!({"pattern": "*.txt"});
        b.iter(|| {
            black_box(tools[4].invoke(black_box(args.clone()), &runtime));
        });
    });

    group.bench_function("execute", |b| {
        let args = serde_json::json!({"command": "echo hi"});
        b.iter(|| {
            black_box(tools[6].invoke(black_box(args.clone()), &runtime));
        });
    });

    group.finish();
}

fn bench_any_tool_dispatch(c: &mut Criterion) {
    let runtime = bench_runtime();
    let args = serde_json::json!({"path": "/"});

    let builtin = AnyTool::Builtin(BuiltinTool::Ls(LsTool));

    struct DynLs;
    #[async_trait::async_trait]
    impl Tool for DynLs {
        fn name(&self) -> &str {
            "ls"
        }
        fn description(&self) -> &str {
            "ls"
        }
        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        fn invoke(
            &self,
            _args: serde_json::Value,
            _runtime: &ToolRuntime,
        ) -> ToolResult {
            ToolResult::Text("ok".into())
        }
    }

    let dynamic = AnyTool::Dynamic(Box::new(DynLs));

    let mut group = c.benchmark_group("any_tool_dispatch");

    group.bench_function("builtin", |b| {
        b.iter(|| {
            black_box(builtin.invoke(black_box(args.clone()), &runtime));
        });
    });

    group.bench_function("dynamic", |b| {
        b.iter(|| {
            black_box(dynamic.invoke(black_box(args.clone()), &runtime));
        });
    });

    group.finish();
}

fn bench_resolve_tool(c: &mut Criterion) {
    let tools = builtin_tools();

    let mut group = c.benchmark_group("resolve");

    group.bench_function("resolve_tool_by_name", |b| {
        b.iter(|| {
            black_box(resolve_tool(black_box("grep"), &tools));
        });
    });

    group.bench_function("resolve_builtin_by_name", |b| {
        b.iter(|| {
            black_box(resolve_builtin(black_box("grep")));
        });
    });

    group.finish();
}

fn bench_format_line_numbers(c: &mut Criterion) {
    let mut group = c.benchmark_group("format_line_numbers");

    let content_100: String = (0..100)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n");
    group.bench_function("100_lines", |b| {
        b.iter(|| {
            black_box(format_content_with_line_numbers(
                black_box(&content_100),
                1,
            ));
        });
    });

    let content_1000: String = (0..1000)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n");
    group.bench_function("1000_lines", |b| {
        b.iter(|| {
            black_box(format_content_with_line_numbers(
                black_box(&content_1000),
                1,
            ));
        });
    });

    let content_10000: String = (0..10000)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n");
    group.bench_function("10000_lines", |b| {
        b.iter(|| {
            black_box(format_content_with_line_numbers(
                black_box(&content_10000),
                1,
            ));
        });
    });

    group.finish();
}

fn bench_write_todos(c: &mut Criterion) {
    let runtime = bench_runtime();
    let tool = WriteTodosTool;
    let args = serde_json::json!({
        "todos": [
            {"content": "Task 1", "status": "pending", "activeForm": "Doing 1"},
            {"content": "Task 2", "status": "in_progress", "activeForm": "Doing 2"},
            {"content": "Task 3", "status": "completed", "activeForm": "Done 3"},
        ]
    });

    c.bench_function("write_todos_3_items", |b| {
        b.iter(|| {
            black_box(tool.invoke(black_box(args.clone()), &runtime));
        });
    });
}

criterion_group!(
    benches,
    bench_builtin_dispatch,
    bench_any_tool_dispatch,
    bench_resolve_tool,
    bench_format_line_numbers,
    bench_write_todos,
);
criterion_main!(benches);
