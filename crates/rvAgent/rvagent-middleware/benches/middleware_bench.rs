//! Benchmarks for rvagent-middleware pipeline.
//!
//! Tests:
//! - Full 11-middleware pipeline throughput (target <1ms)
//! - SystemPromptBuilder vs naive concatenation
//! - Skill name validation

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use rvagent_middleware::{
    build_default_pipeline, Message, ModelHandler, ModelRequest, ModelResponse,
    PipelineConfig, SystemPromptBuilder,
};
use rvagent_middleware::skills::validate_skill_name;

/// A no-op handler that returns immediately.
struct NoOpHandler;
impl ModelHandler for NoOpHandler {
    fn call(&self, _request: ModelRequest) -> ModelResponse {
        ModelResponse::text("ok")
    }
}

fn bench_full_pipeline(c: &mut Criterion) {
    let config = PipelineConfig {
        memory_sources: Some(vec!["AGENTS.md".into()]),
        skill_sources: Some(vec![".skills".into()]),
        interrupt_on: Some(vec!["execute".into()]),
        enable_witness: true,
    };
    let pipeline = build_default_pipeline(&config);
    let handler = NoOpHandler;

    c.bench_function("full_11_middleware_pipeline", |b| {
        b.iter(|| {
            let request = ModelRequest::new(vec![
                Message::user("Hello"),
                Message::assistant("Hi there"),
                Message::user("Write some code"),
            ]);
            let response = pipeline.run_wrap_model_call(black_box(request), &handler);
            black_box(response);
        });
    });
}

fn bench_system_prompt_builder(c: &mut Criterion) {
    let segments: Vec<String> = (0..8)
        .map(|i| format!("Segment {} with some content that represents a typical middleware injection of about 100 characters of text for testing purposes.", i))
        .collect();

    c.bench_function("system_prompt_builder", |b| {
        b.iter(|| {
            let mut builder = SystemPromptBuilder::new();
            for seg in &segments {
                builder.append(seg.clone());
            }
            black_box(builder.build());
        });
    });

    c.bench_function("naive_string_concat", |b| {
        b.iter(|| {
            let mut result = String::new();
            for (i, seg) in segments.iter().enumerate() {
                if i > 0 {
                    result.push_str("\n\n");
                }
                result.push_str(seg);
            }
            black_box(result);
        });
    });
}

fn bench_skill_name_validation(c: &mut Criterion) {
    c.bench_function("validate_skill_name_valid", |b| {
        b.iter(|| {
            let _ = black_box(validate_skill_name(
                black_box("my-cool-skill-123"),
                black_box("my-cool-skill-123"),
            ));
        });
    });

    c.bench_function("validate_skill_name_invalid_unicode", |b| {
        b.iter(|| {
            let _ = black_box(validate_skill_name(
                black_box("my-skíll"),
                black_box("my-skíll"),
            ));
        });
    });

    c.bench_function("validate_skill_name_max_length", |b| {
        let name = "a".repeat(64);
        b.iter(|| {
            let _ = black_box(validate_skill_name(
                black_box(&name),
                black_box(&name),
            ));
        });
    });
}

fn bench_pipeline_modify_request(c: &mut Criterion) {
    let config = PipelineConfig {
        memory_sources: Some(vec!["AGENTS.md".into()]),
        skill_sources: Some(vec![".skills".into()]),
        interrupt_on: None,
        enable_witness: false,
    };
    let pipeline = build_default_pipeline(&config);

    c.bench_function("pipeline_modify_request", |b| {
        b.iter(|| {
            let request = ModelRequest::new(vec![Message::user("test")])
                .with_system(Some("You are helpful.".into()));
            let modified = pipeline.run_modify_request(black_box(request));
            black_box(modified);
        });
    });
}

fn bench_pipeline_collect_tools(c: &mut Criterion) {
    let config = PipelineConfig::default();
    let pipeline = build_default_pipeline(&config);

    c.bench_function("pipeline_collect_tools", |b| {
        b.iter(|| {
            let tools = pipeline.collect_tools();
            black_box(tools);
        });
    });
}

criterion_group!(
    benches,
    bench_full_pipeline,
    bench_system_prompt_builder,
    bench_skill_name_validation,
    bench_pipeline_modify_request,
    bench_pipeline_collect_tools,
);
criterion_main!(benches);
