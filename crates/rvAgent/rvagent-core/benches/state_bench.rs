//! Criterion benchmarks for rvagent-core: AgentState, Message serialization,
//! and SystemPromptBuilder (ADR-103 A9).

use criterion::{criterion_group, criterion_main, Criterion, black_box, BenchmarkId};
use std::collections::HashMap;
use std::sync::Arc;

use rvagent_core::messages::{Message, ToolCall};
use rvagent_core::prompt::SystemPromptBuilder;
use rvagent_core::state::{AgentState, FileData, SkillMetadata, TodoItem, TodoStatus};

// ---------------------------------------------------------------------------
// Helpers — build realistic state payloads
// ---------------------------------------------------------------------------

fn make_messages(n: usize) -> Vec<Message> {
    let mut msgs = Vec::with_capacity(n);
    msgs.push(Message::system("You are a helpful coding assistant."));
    for i in 0..n.saturating_sub(1) {
        if i % 3 == 0 {
            msgs.push(Message::human(format!(
                "Please read the file src/module_{}.rs and summarize it.",
                i
            )));
        } else if i % 3 == 1 {
            msgs.push(Message::ai_with_tools(
                format!("Let me read that file for you (step {}).", i),
                vec![ToolCall {
                    id: format!("call_{}", i),
                    name: "read_file".into(),
                    args: serde_json::json!({"path": format!("src/module_{}.rs", i)}),
                }],
            ));
        } else {
            msgs.push(Message::tool(
                format!("call_{}", i - 1),
                format!("     1\tpub fn func_{}() {{}}", i),
            ));
        }
    }
    msgs
}

fn make_files(n: usize) -> HashMap<String, FileData> {
    (0..n)
        .map(|i| {
            (
                format!("src/module_{}.rs", i),
                FileData {
                    content: format!("pub fn func_{}() {{}}\n// line 2\n// line 3", i),
                    encoding: "utf-8".into(),
                    modified_at: None,
                },
            )
        })
        .collect()
}

fn make_todos(n: usize) -> Vec<TodoItem> {
    (0..n)
        .map(|i| TodoItem {
            content: format!("Implement feature {}", i),
            status: if i % 3 == 0 {
                TodoStatus::Completed
            } else if i % 3 == 1 {
                TodoStatus::InProgress
            } else {
                TodoStatus::Pending
            },
            active_form: format!("Implementing feature {}", i),
        })
        .collect()
}

fn make_populated_state(msg_count: usize, file_count: usize, todo_count: usize) -> AgentState {
    let mut state = AgentState::new();
    state.messages = Arc::new(make_messages(msg_count));
    state.todos = Arc::new(make_todos(todo_count));
    state.files = Arc::new(make_files(file_count));
    state.skills_metadata = Some(Arc::new(vec![
        SkillMetadata {
            name: "deploy".into(),
            description: "Deploy the application to production".into(),
            parameters: serde_json::json!({"target": "string", "env": "string"}),
        },
        SkillMetadata {
            name: "test-runner".into(),
            description: "Run the test suite".into(),
            parameters: serde_json::json!({"filter": "string"}),
        },
    ]));
    state
}

// ---------------------------------------------------------------------------
// Benchmark: AgentState clone (Arc O(1) vs deep clone simulation)
// ---------------------------------------------------------------------------

fn bench_state_clone(c: &mut Criterion) {
    let state = make_populated_state(100, 10, 5);

    let mut group = c.benchmark_group("state_clone");

    // Arc clone — the real implementation (should be near-instant)
    group.bench_function("arc_clone_100msg_10files_5todos", |b| {
        b.iter(|| {
            let cloned = black_box(&state).clone();
            black_box(cloned);
        })
    });

    // Simulate deep clone by serializing/deserializing (what HashMap<String, Value> would need)
    group.bench_function("deep_clone_via_serde_100msg_10files_5todos", |b| {
        let json = serde_json::to_vec(&*state.messages).unwrap();
        let files_json = serde_json::to_vec(&*state.files).unwrap();
        let todos_json = serde_json::to_vec(&*state.todos).unwrap();
        b.iter(|| {
            let msgs: Vec<Message> =
                serde_json::from_slice(black_box(&json)).unwrap();
            let files: HashMap<String, FileData> =
                serde_json::from_slice(black_box(&files_json)).unwrap();
            let todos: Vec<TodoItem> =
                serde_json::from_slice(black_box(&todos_json)).unwrap();
            black_box((msgs, files, todos));
        })
    });

    // Arc clone with larger state
    let large_state = make_populated_state(1000, 50, 20);
    group.bench_function("arc_clone_1000msg_50files_20todos", |b| {
        b.iter(|| {
            let cloned = black_box(&large_state).clone();
            black_box(cloned);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Message serialization round-trip
// ---------------------------------------------------------------------------

fn bench_message_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("message_serialization");

    for count in [10, 100, 1000] {
        let messages = make_messages(count);
        let json_bytes = serde_json::to_vec(&messages).unwrap();

        group.bench_with_input(
            BenchmarkId::new("serialize", count),
            &messages,
            |b, msgs| {
                b.iter(|| {
                    let bytes = serde_json::to_vec(black_box(msgs)).unwrap();
                    black_box(bytes);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("deserialize", count),
            &json_bytes,
            |b, bytes| {
                b.iter(|| {
                    let msgs: Vec<Message> =
                        serde_json::from_slice(black_box(bytes)).unwrap();
                    black_box(msgs);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("roundtrip", count),
            &messages,
            |b, msgs| {
                b.iter(|| {
                    let bytes = serde_json::to_vec(black_box(msgs)).unwrap();
                    let back: Vec<Message> = serde_json::from_slice(&bytes).unwrap();
                    black_box(back);
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: SystemPromptBuilder vs naive String concatenation
// ---------------------------------------------------------------------------

fn bench_system_prompt_builder(c: &mut Criterion) {
    let segments: Vec<String> = vec![
        "You are a helpful coding assistant.".into(),
        "## Memory\nYou have access to the following memory contents:\n- auth patterns: JWT\n- db patterns: PostgreSQL".into(),
        "## Skills\nAvailable skills:\n- deploy: Deploy to production\n- test: Run tests\n- lint: Run linter".into(),
        "## Filesystem\nCurrent working directory: /home/user/project\nFiles: src/main.rs, src/lib.rs, Cargo.toml".into(),
        "## SubAgents\nYou can spawn subagents for parallel work.".into(),
        "## Summarization\nConversation is within token limits.".into(),
        "## Security\nDo not expose secrets. Validate all paths.".into(),
        "## Output\nBe concise. Use absolute file paths.".into(),
    ];

    let mut group = c.benchmark_group("system_prompt_builder");

    // SystemPromptBuilder (single allocation)
    group.bench_function("builder_8_segments", |b| {
        b.iter(|| {
            let mut builder = SystemPromptBuilder::new();
            for seg in &segments {
                builder.append(seg.clone());
            }
            let result = builder.build();
            black_box(result);
        })
    });

    // Naive sequential format!() concatenation
    group.bench_function("naive_format_8_segments", |b| {
        b.iter(|| {
            let mut result = segments[0].clone();
            for seg in &segments[1..] {
                result = format!("{}\n\n{}", result, seg);
            }
            black_box(result);
        })
    });

    // Naive String push_str
    group.bench_function("naive_push_str_8_segments", |b| {
        b.iter(|| {
            let mut result = String::new();
            for (i, seg) in segments.iter().enumerate() {
                if i > 0 {
                    result.push_str("\n\n");
                }
                result.push_str(seg);
            }
            black_box(result);
        })
    });

    // Builder with borrowed &'static str (best case — no clone needed)
    group.bench_function("builder_static_segments", |b| {
        let static_segs: &[&str] = &[
            "Segment one: system prompt",
            "Segment two: memory",
            "Segment three: skills",
            "Segment four: filesystem",
            "Segment five: subagents",
            "Segment six: summarization",
            "Segment seven: security",
            "Segment eight: output format",
        ];
        b.iter(|| {
            let mut builder = SystemPromptBuilder::new();
            for seg in static_segs {
                builder.append(*seg);
            }
            let result = builder.build();
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_state_clone,
    bench_message_serialization,
    bench_system_prompt_builder
);
criterion_main!(benches);
