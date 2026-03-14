//! Criterion benchmarks for rvagent-middleware: pipeline throughput,
//! SystemPromptBuilder, and skill name validation (ADR-103 A9).

use criterion::{criterion_group, criterion_main, Criterion, black_box, BenchmarkId};
use std::borrow::Cow;
use std::collections::HashMap;

use rvagent_middleware::{
    AgentState, Message, ModelHandler, ModelRequest, ModelResponse, MiddlewarePipeline,
    PipelineConfig, Runtime, RunnableConfig, ToolDefinition,
    build_default_pipeline, SystemPromptBuilder,
};

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Passthrough model handler — returns immediately with a fixed response.
struct NoOpHandler;
impl ModelHandler for NoOpHandler {
    fn call(&self, request: ModelRequest) -> ModelResponse {
        ModelResponse::text(format!("ok: {} msgs", request.messages.len()))
    }
}

fn make_state_with_messages(n: usize) -> AgentState {
    let mut state = AgentState::default();
    state.messages.push(Message::system("You are a helpful assistant."));
    for i in 0..n {
        state.messages.push(Message::user(format!("Message {}", i)));
        state.messages.push(Message::assistant(format!("Response {}", i)));
    }
    state
}

fn make_request(msg_count: usize) -> ModelRequest {
    let mut messages = vec![Message::user("test query")];
    for i in 0..msg_count.saturating_sub(1) {
        messages.push(Message::assistant(format!("response {}", i)));
    }
    ModelRequest::new(messages)
        .with_system(Some("Base system prompt for testing.".into()))
}

// ---------------------------------------------------------------------------
// Benchmark: Full middleware pipeline pass-through (target: <1ms)
// ---------------------------------------------------------------------------

fn bench_middleware_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("middleware_pipeline");

    // Full 11-middleware pipeline (ADR-095 ordering)
    let config_full = PipelineConfig {
        memory_sources: Some(vec!["AGENTS.md".into()]),
        skill_sources: Some(vec![".skills".into()]),
        interrupt_on: Some(vec!["execute".into()]),
        enable_witness: true,
    };
    let pipeline_full = build_default_pipeline(&config_full);
    assert_eq!(pipeline_full.len(), 11, "Expected 11 middlewares in full pipeline");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    // Full pipeline: run_before_agent + modify_request + wrap_model_call
    group.bench_function("full_11mw_pipeline_run", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut state = make_state_with_messages(10);
                let runtime = Runtime::new();
                let config = RunnableConfig::default();
                let request = make_request(5);
                let response = pipeline_full
                    .run(&mut state, &runtime, &config, request, &NoOpHandler)
                    .await;
                black_box(response);
            })
        })
    });

    // Just wrap_model_call chaining (no before_agent, no tool collection)
    group.bench_function("full_11mw_wrap_model_call", |b| {
        b.iter(|| {
            let request = make_request(5);
            let response =
                pipeline_full.run_wrap_model_call(black_box(request), &NoOpHandler);
            black_box(response);
        })
    });

    // Just modify_request through pipeline
    group.bench_function("full_11mw_modify_request", |b| {
        b.iter(|| {
            let request = make_request(5);
            let modified = pipeline_full.run_modify_request(black_box(request));
            black_box(modified);
        })
    });

    // Minimal pipeline (7 middlewares — no memory, skills, witness, hitl)
    let config_minimal = PipelineConfig::default();
    let pipeline_minimal = build_default_pipeline(&config_minimal);

    group.bench_function("minimal_7mw_pipeline_run", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut state = make_state_with_messages(10);
                let runtime = Runtime::new();
                let config = RunnableConfig::default();
                let request = make_request(5);
                let response = pipeline_minimal
                    .run(&mut state, &runtime, &config, request, &NoOpHandler)
                    .await;
                black_box(response);
            })
        })
    });

    // Empty pipeline baseline (pure handler call)
    let pipeline_empty = MiddlewarePipeline::empty();
    group.bench_function("empty_pipeline_baseline", |b| {
        b.iter(|| {
            let request = make_request(5);
            let response =
                pipeline_empty.run_wrap_model_call(black_box(request), &NoOpHandler);
            black_box(response);
        })
    });

    // Pipeline with varying message counts
    for msg_count in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("full_11mw_run_msgs", msg_count),
            &msg_count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let mut state = make_state_with_messages(count);
                        let runtime = Runtime::new();
                        let config = RunnableConfig::default();
                        let request = make_request(count);
                        let response = pipeline_full
                            .run(&mut state, &runtime, &config, request, &NoOpHandler)
                            .await;
                        black_box(response);
                    })
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: SystemPromptBuilder.build() vs format!() with 8 segments
// ---------------------------------------------------------------------------

fn bench_prompt_builder_vs_concat(c: &mut Criterion) {
    let mut group = c.benchmark_group("prompt_builder_vs_concat");

    let segments: Vec<&str> = vec![
        "You are an expert coding assistant powered by RuVector.",
        "## Memory\nRecent patterns:\n- JWT authentication with refresh tokens\n- PostgreSQL connection pooling\n- Redis caching layer",
        "## Skills\nAvailable:\n- deploy (Deploy app)\n- test (Run tests)\n- lint (Run linter)\n- format (Format code)",
        "## Filesystem\nCWD: /home/user/project\nKey files: src/main.rs, src/lib.rs, Cargo.toml, .env",
        "## SubAgents\nYou can spawn subagents for:\n- Parallel file operations\n- Background test runs",
        "## Summarization\nConversation: 45,000/100,000 tokens used.",
        "## Security\n- Never expose API keys\n- Validate file paths\n- Use sandbox for execution",
        "## Output\n- Be concise\n- Use absolute paths\n- Show code only when relevant",
    ];

    // SystemPromptBuilder with borrowed static segments
    group.bench_function("builder_8_static_segments", |b| {
        b.iter(|| {
            let mut builder = SystemPromptBuilder::new();
            for seg in &segments {
                builder.append(black_box(*seg));
            }
            let result = builder.build();
            black_box(result);
        })
    });

    // SystemPromptBuilder with owned String segments
    let owned_segments: Vec<String> = segments.iter().map(|s| s.to_string()).collect();
    group.bench_function("builder_8_owned_segments", |b| {
        b.iter(|| {
            let mut builder = SystemPromptBuilder::new();
            for seg in &owned_segments {
                builder.append(seg.clone());
            }
            let result = builder.build();
            black_box(result);
        })
    });

    // Naive: sequential format!() — 7 intermediate allocations
    group.bench_function("naive_format_chain_8_segments", |b| {
        b.iter(|| {
            let mut result = segments[0].to_string();
            for seg in &segments[1..] {
                result = format!("{}\n\n{}", result, seg);
            }
            black_box(result);
        })
    });

    // Naive: push_str without pre-allocated capacity
    group.bench_function("naive_push_str_no_capacity", |b| {
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

    // Cow-based builder (what the real implementation uses)
    group.bench_function("builder_cow_mixed", |b| {
        b.iter(|| {
            let mut builder = SystemPromptBuilder::new();
            builder.append(Cow::Borrowed(segments[0]));
            builder.append(Cow::Borrowed(segments[1]));
            builder.append(Cow::Owned(format!("## Dynamic\nTimestamp: {}", 1234567890)));
            builder.append(Cow::Borrowed(segments[3]));
            builder.append(Cow::Borrowed(segments[4]));
            builder.append(Cow::Owned(format!("## Tokens\nUsed: {}/100000", 45000)));
            builder.append(Cow::Borrowed(segments[6]));
            builder.append(Cow::Borrowed(segments[7]));
            let result = builder.build();
            black_box(result);
        })
    });

    // Vec::join comparison (standard library approach)
    group.bench_function("vec_join_8_segments", |b| {
        b.iter(|| {
            let result = segments.join("\n\n");
            black_box(result);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Skill name validation (ADR-103 C10)
//
// Uses the same validation logic as validate_ascii_identifier from
// rvagent-backends, inlined here since middleware calls it for skill
// registration.
// ---------------------------------------------------------------------------

/// Validate skill name: ASCII lowercase letters, digits, hyphens, underscores.
/// Must start with a letter.
fn validate_skill_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }
    for c in chars {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_' {
            continue;
        }
        return false;
    }
    true
}

fn bench_skill_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("skill_validation");

    // Valid skill names of various lengths
    let valid_names = vec![
        "a",
        "deploy",
        "test-runner",
        "my_custom_skill_v2",
        "really-long-skill-name-that-describes-what-it-does-in-detail",
    ];

    group.bench_function("valid_names_5_checks", |b| {
        b.iter(|| {
            let mut all_valid = true;
            for name in &valid_names {
                all_valid &= validate_skill_name(black_box(name));
            }
            black_box(all_valid);
        })
    });

    // Invalid skill names — various rejection paths
    let invalid_names = vec![
        "",                    // empty
        "123abc",              // starts with digit
        "Hello",              // uppercase
        "-start",             // starts with hyphen
        "na\u{0441}me",      // Cyrillic
        "caf\u{00E9}",       // non-ASCII accent
        "has space",          // space
        "has.dot",            // dot
        "ALLCAPS",            // all uppercase
    ];

    group.bench_function("invalid_names_9_checks", |b| {
        b.iter(|| {
            let mut any_valid = false;
            for name in &invalid_names {
                any_valid |= validate_skill_name(black_box(name));
            }
            black_box(any_valid);
        })
    });

    // Batch validation — typical middleware startup scanning 50 skills
    let skill_batch: Vec<String> = (0..50)
        .map(|i| format!("skill-{}-handler", i))
        .collect();

    group.bench_function("batch_50_skills", |b| {
        b.iter(|| {
            let results: Vec<bool> = skill_batch
                .iter()
                .map(|name| validate_skill_name(black_box(name)))
                .collect();
            black_box(results);
        })
    });

    // Single validation hot path
    group.bench_function("single_valid_short", |b| {
        b.iter(|| {
            let result = validate_skill_name(black_box("deploy"));
            black_box(result);
        })
    });

    group.bench_function("single_invalid_empty", |b| {
        b.iter(|| {
            let result = validate_skill_name(black_box(""));
            black_box(result);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_middleware_pipeline,
    bench_prompt_builder_vs_concat,
    bench_skill_validation
);
criterion_main!(benches);
