//! Midstream Platform integration for real-time streaming analysis (ADR-077).
//!
//! Wraps four midstream crates behind feature flags:
//! - **nanosecond-scheduler**: Background task scheduling with nanosecond precision
//! - **temporal-attractor-studio**: Lyapunov exponent analysis for embedding trajectories
//! - **temporal-neural-solver**: Certified temporal predictions with solver gates
//! - **strange-loop**: Recursive meta-cognitive reasoning with safety bounds

use crate::types::AppState;

// ── Attractor Analysis (temporal-attractor-studio) ─────────────────────

/// Compute Lyapunov exponent for a category's embedding trajectory.
/// Positive λ = chaotic (knowledge diverging), negative = stable (converging).
/// Returns None if trajectory is too short (need ≥10 points for meaningful estimate).
pub fn analyze_category_attractor(
    embeddings: &[Vec<f32>],
) -> Option<temporal_attractor_studio::LyapunovResult> {
    if embeddings.len() < 10 {
        return None;
    }
    // Convert f32 embeddings to f64 trajectories for FTLE computation
    let trajectory: Vec<Vec<f64>> = embeddings
        .iter()
        .map(|e| e.iter().map(|&v| v as f64).collect())
        .collect();

    temporal_attractor_studio::estimate_lyapunov_default(&trajectory).ok()
}

/// Compute a stability score from a Lyapunov result for search ranking.
/// Returns a small additive bonus (0.0 to 0.05) for memories in stable categories.
/// Stable categories (negative λ) get a boost; chaotic categories get zero.
pub fn attractor_stability_score(result: &temporal_attractor_studio::LyapunovResult) -> f32 {
    if result.lambda < 0.0 {
        // Negative Lyapunov = converging (stable knowledge domain)
        // Scale: λ=-1.0 → 0.05 bonus, λ=0.0 → 0.0
        ((-result.lambda).min(1.0) * 0.05) as f32
    } else {
        0.0
    }
}

// ── Temporal Neural Solver (temporal-neural-solver) ────────────────────
// Note: This crate requires x86_64 SIMD — disabled on ARM/Apple Silicon

/// Score a search result using the temporal solver's prediction confidence.
/// Returns a small additive bonus (0.0 to 0.04) based on the certificate confidence.
#[cfg(feature = "x86-simd")]
pub fn solver_confidence_score(certificate: &temporal_neural_solver::Certificate) -> f32 {
    if certificate.gate_pass {
        // Certificate passed solver gate — high confidence prediction
        (certificate.confidence.min(1.0) * 0.04) as f32
    } else {
        0.0
    }
}

/// Stub for non-x86 platforms
#[cfg(not(feature = "x86-simd"))]
pub mod temporal_neural_solver_stub {
    /// Stub certificate for non-x86 platforms
    pub struct Certificate {
        pub gate_pass: bool,
        pub confidence: f64,
    }
}

// ── Strange Loop Meta-Cognition (strange-loop) ─────────────────────────

/// Create a default StrangeLoop engine for meta-cognitive reasoning.
pub fn create_strange_loop() -> strange_loop::StrangeLoop<
    strange_loop::ScalarReasoner,
    strange_loop::SimpleCritic,
    strange_loop::SafeReflector,
> {
    let reasoner = strange_loop::ScalarReasoner::new(0.0, 1.0);
    let critic = strange_loop::SimpleCritic::new();
    let reflector = strange_loop::SafeReflector::new();
    let config = strange_loop::LoopConfig {
        max_iterations: 10,
        max_duration_ns: 5_000_000, // 5ms budget for meta-cognition
        convergence_threshold: 0.01,
        lipschitz_constant: 0.9,
        enable_consciousness: false,
        enable_quantum: false,
        enable_simd: false,
    };
    strange_loop::StrangeLoop::new(reasoner, critic, reflector, config)
}

/// Run a meta-cognitive evaluation on a search context.
/// Returns a small additive bonus (0.0 to 0.04) based on the loop's convergence.
pub fn strange_loop_score(
    loop_engine: &mut strange_loop::StrangeLoop<
        strange_loop::ScalarReasoner,
        strange_loop::SimpleCritic,
        strange_loop::SafeReflector,
    >,
    query_relevance: f64,
    memory_quality: f64,
) -> f32 {
    let mut ctx = strange_loop::Context::new();
    ctx.insert("relevance".to_string(), query_relevance);
    ctx.insert("quality".to_string(), memory_quality);
    // Run the loop — bounded by max_iterations and max_duration_ns
    match loop_engine.run(&mut ctx) {
        Ok(_result) => {
            // Extract composite score from context after meta-cognitive evaluation
            let composite = ctx.get("relevance").copied().unwrap_or(0.0)
                * ctx.get("quality").copied().unwrap_or(0.0);
            (composite.min(1.0).max(0.0) * 0.04) as f32
        }
        Err(_) => 0.0,
    }
}

// ── Nanosecond Scheduler ───────────────────────────────────────────────

/// Create a default scheduler for background brain tasks.
pub fn create_scheduler() -> nanosecond_scheduler::Scheduler {
    let config = nanosecond_scheduler::Config {
        max_tasks_per_tick: 50,
        ..nanosecond_scheduler::Config::default()
    };
    nanosecond_scheduler::Scheduler::new(config)
}

// ── Status / Diagnostics ───────────────────────────────────────────────

/// Midstream diagnostics for the /v1/status and /v1/midstream endpoints.
#[derive(Debug, serde::Serialize)]
pub struct MidstreamStatus {
    pub scheduler_total_ticks: u64,
    pub scheduler_tasks_per_sec: f64,
    pub attractor_categories_analyzed: usize,
    pub solver_input_size: usize,
    pub strange_loop_version: String,
}

/// Collect midstream diagnostics from AppState.
pub fn collect_status(state: &AppState) -> MidstreamStatus {
    let metrics = state.nano_scheduler.metrics();
    let attractor_count = state.attractor_results.read().len();
    let solver = state.temporal_solver.read();
    // TemporalSolver doesn't expose input_size directly; use a sentinel
    let _ = &solver; // read lock held briefly

    MidstreamStatus {
        scheduler_total_ticks: metrics.total_ticks,
        scheduler_tasks_per_sec: metrics.tasks_per_second,
        attractor_categories_analyzed: attractor_count,
        solver_input_size: crate::embeddings::EMBED_DIM,
        strange_loop_version: strange_loop::VERSION.to_string(),
    }
}
