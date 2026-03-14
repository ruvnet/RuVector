//! Criterion benchmarks for rvagent-tools: enum dispatch vs trait object dispatch,
//! and parallel vs sequential tool execution (ADR-103 A6, A2, A9).

use criterion::{criterion_group, criterion_main, Criterion, black_box};

// ---------------------------------------------------------------------------
// Simulated tool dispatch types (ADR-103 A6)
//
// The actual BuiltinTool enum and Box<dyn Tool> are defined in rvagent-tools.
// Since the crate's source may not be fully populated yet, we inline realistic
// simulations that match the ADR-103 A6 architecture to benchmark the dispatch
// pattern itself.
// ---------------------------------------------------------------------------

/// Simulated built-in tool enum (ADR-103 A6: enum dispatch for hot-path tools).
#[derive(Clone, Copy)]
enum BuiltinTool {
    Ls,
    ReadFile,
    WriteFile,
    EditFile,
    Glob,
    Grep,
    Execute,
    WriteTodos,
    Task,
}

impl BuiltinTool {
    /// Simulate tool invocation — the interesting part is the dispatch cost.
    #[inline(never)]
    fn invoke(&self, input: &str) -> String {
        match self {
            BuiltinTool::Ls => format!("ls: {} entries", input.len()),
            BuiltinTool::ReadFile => format!("read: {} bytes", input.len()),
            BuiltinTool::WriteFile => format!("write: {} bytes", input.len()),
            BuiltinTool::EditFile => format!("edit: {} chars changed", input.len()),
            BuiltinTool::Glob => format!("glob: {} matches", input.len() % 10),
            BuiltinTool::Grep => format!("grep: {} matches", input.len() % 5),
            BuiltinTool::Execute => format!("exec: exit {}", input.len() % 2),
            BuiltinTool::WriteTodos => format!("todos: {} items", input.len() % 8),
            BuiltinTool::Task => format!("task: spawned {}", input.len() % 3),
        }
    }
}

/// Simulated trait object dispatch (the pre-ADR-103 approach).
trait DynTool: Send + Sync {
    fn name(&self) -> &str;
    fn invoke(&self, input: &str) -> String;
}

struct DynReadFile;
impl DynTool for DynReadFile {
    fn name(&self) -> &str { "read_file" }
    #[inline(never)]
    fn invoke(&self, input: &str) -> String {
        format!("read: {} bytes", input.len())
    }
}

struct DynGrep;
impl DynTool for DynGrep {
    fn name(&self) -> &str { "grep" }
    #[inline(never)]
    fn invoke(&self, input: &str) -> String {
        format!("grep: {} matches", input.len() % 5)
    }
}

struct DynGlob;
impl DynTool for DynGlob {
    fn name(&self) -> &str { "glob" }
    #[inline(never)]
    fn invoke(&self, input: &str) -> String {
        format!("glob: {} matches", input.len() % 10)
    }
}

struct DynLs;
impl DynTool for DynLs {
    fn name(&self) -> &str { "ls" }
    #[inline(never)]
    fn invoke(&self, input: &str) -> String {
        format!("ls: {} entries", input.len())
    }
}

/// Combined dispatch (ADR-103 A6): try enum first, fall back to trait object.
enum AnyTool {
    Builtin(BuiltinTool),
    Dynamic(Box<dyn DynTool>),
}

impl AnyTool {
    fn invoke(&self, input: &str) -> String {
        match self {
            AnyTool::Builtin(b) => b.invoke(input),
            AnyTool::Dynamic(d) => d.invoke(input),
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark: enum dispatch vs Box<dyn Tool> dispatch
// ---------------------------------------------------------------------------

fn bench_tool_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("tool_dispatch");
    let input = "src/main.rs";

    // Enum dispatch — direct match, no vtable
    let builtin_tools = vec![
        BuiltinTool::ReadFile,
        BuiltinTool::Grep,
        BuiltinTool::Glob,
        BuiltinTool::Ls,
    ];
    group.bench_function("enum_dispatch_4_tools", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(4);
            for tool in &builtin_tools {
                results.push(tool.invoke(black_box(input)));
            }
            black_box(results);
        })
    });

    // Trait object dispatch — vtable indirection
    let dyn_tools: Vec<Box<dyn DynTool>> = vec![
        Box::new(DynReadFile),
        Box::new(DynGrep),
        Box::new(DynGlob),
        Box::new(DynLs),
    ];
    group.bench_function("dyn_dispatch_4_tools", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(4);
            for tool in &dyn_tools {
                results.push(tool.invoke(black_box(input)));
            }
            black_box(results);
        })
    });

    // AnyTool enum wrapping builtin — should be same perf as enum_dispatch
    let any_builtin: Vec<AnyTool> = vec![
        AnyTool::Builtin(BuiltinTool::ReadFile),
        AnyTool::Builtin(BuiltinTool::Grep),
        AnyTool::Builtin(BuiltinTool::Glob),
        AnyTool::Builtin(BuiltinTool::Ls),
    ];
    group.bench_function("any_tool_builtin_4_tools", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(4);
            for tool in &any_builtin {
                results.push(tool.invoke(black_box(input)));
            }
            black_box(results);
        })
    });

    // AnyTool wrapping dynamic — should match dyn_dispatch
    let any_dynamic: Vec<AnyTool> = vec![
        AnyTool::Dynamic(Box::new(DynReadFile)),
        AnyTool::Dynamic(Box::new(DynGrep)),
        AnyTool::Dynamic(Box::new(DynGlob)),
        AnyTool::Dynamic(Box::new(DynLs)),
    ];
    group.bench_function("any_tool_dynamic_4_tools", |b| {
        b.iter(|| {
            let mut results = Vec::with_capacity(4);
            for tool in &any_dynamic {
                results.push(tool.invoke(black_box(input)));
            }
            black_box(results);
        })
    });

    // Single tool dispatch — enum vs dyn (isolate per-call overhead)
    group.bench_function("single_enum_dispatch", |b| {
        let tool = BuiltinTool::Grep;
        b.iter(|| {
            let result = tool.invoke(black_box(input));
            black_box(result);
        })
    });

    group.bench_function("single_dyn_dispatch", |b| {
        let tool: Box<dyn DynTool> = Box::new(DynGrep);
        b.iter(|| {
            let result = tool.invoke(black_box(input));
            black_box(result);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: parallel vs sequential tool execution (ADR-103 A2)
// ---------------------------------------------------------------------------

fn bench_parallel_vs_sequential_tools(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");

    // Simulate 4 tool calls with CPU-bound work (string processing).
    // Real I/O-bound tools would show much larger parallel speedup; here we
    // measure the coordination overhead and baseline comparison.

    let inputs: Vec<String> = (0..4)
        .map(|i| format!("input_data_{}", "x".repeat(1000 * (i + 1))))
        .collect();

    // Sequential execution
    group.bench_function("sequential_4_tools", |b| {
        let tools = vec![
            BuiltinTool::ReadFile,
            BuiltinTool::Grep,
            BuiltinTool::Glob,
            BuiltinTool::Ls,
        ];
        b.iter(|| {
            let mut results = Vec::with_capacity(4);
            for (tool, input) in tools.iter().zip(inputs.iter()) {
                results.push(tool.invoke(black_box(input)));
            }
            black_box(results);
        })
    });

    // Parallel execution using tokio (measures spawn + join overhead)
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .build()
        .unwrap();

    group.bench_function("parallel_4_tools_tokio", |b| {
        let tools = vec![
            BuiltinTool::ReadFile,
            BuiltinTool::Grep,
            BuiltinTool::Glob,
            BuiltinTool::Ls,
        ];
        b.iter(|| {
            rt.block_on(async {
                let mut set = tokio::task::JoinSet::new();
                for (tool, input) in tools.iter().zip(inputs.iter()) {
                    let t = *tool;
                    let inp = input.clone();
                    set.spawn(async move { t.invoke(black_box(&inp)) });
                }
                let mut results = Vec::with_capacity(4);
                while let Some(result) = set.join_next().await {
                    results.push(result.unwrap());
                }
                black_box(results);
            })
        })
    });

    // Sequential with 8 tools (larger batch)
    group.bench_function("sequential_8_tools", |b| {
        let tools = vec![
            BuiltinTool::ReadFile,
            BuiltinTool::Grep,
            BuiltinTool::Glob,
            BuiltinTool::Ls,
            BuiltinTool::Execute,
            BuiltinTool::WriteFile,
            BuiltinTool::EditFile,
            BuiltinTool::WriteTodos,
        ];
        let inputs_8: Vec<String> = (0..8)
            .map(|i| format!("input_{}", "y".repeat(500 * (i + 1))))
            .collect();
        b.iter(|| {
            let mut results = Vec::with_capacity(8);
            for (tool, input) in tools.iter().zip(inputs_8.iter()) {
                results.push(tool.invoke(black_box(input)));
            }
            black_box(results);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_tool_dispatch,
    bench_parallel_vs_sequential_tools
);
criterion_main!(benches);
