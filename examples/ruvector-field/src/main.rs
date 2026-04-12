//! RuVector field subsystem — runnable demo.
//!
//! This example mirrors `docs/research/ruvector-field/SPEC.md`. It builds a
//! tiny field engine with four shells, binds geometric and semantic antipodes,
//! recomputes coherence, promotes shells, runs a shell-aware retrieval with a
//! contradiction frontier, checks drift, and issues a routing hint.
//!
//! Run with:
//!     cargo run --manifest-path examples/ruvector-field/Cargo.toml

#![allow(dead_code)]

mod engine;
mod types;

use engine::FieldEngine;
use types::*;

fn main() {
    println!("=== RuVector Field Subsystem Demo ===\n");

    let mut engine = FieldEngine::new();

    // --- 1. Ingest a handful of interactions into the Event shell ---
    let e1 = engine.ingest(
        NodeKind::Interaction,
        "User reports authentication timeout after 30s of idle",
        Embedding::new(vec![0.9, 0.1, 0.0, 0.2, 0.0]),
        AxisScores::new(0.7, 0.6, 0.5, 0.8),
        0b0001,
    );
    let e2 = engine.ingest(
        NodeKind::Interaction,
        "User reports authentication timeout on mobile device",
        Embedding::new(vec![0.85, 0.15, 0.05, 0.2, 0.0]),
        AxisScores::new(0.7, 0.6, 0.55, 0.8),
        0b0001,
    );
    let e3 = engine.ingest(
        NodeKind::Interaction,
        "Session refresh silently fails after JWT expiry",
        Embedding::new(vec![0.8, 0.2, 0.1, 0.15, 0.05]),
        AxisScores::new(0.7, 0.55, 0.5, 0.85),
        0b0001,
    );

    // A summary pattern that generalizes these events
    let p1 = engine.ingest(
        NodeKind::Summary,
        "Pattern: idle timeout causes silent JWT refresh failure",
        Embedding::new(vec![0.85, 0.15, 0.05, 0.18, 0.02]),
        AxisScores::new(0.8, 0.7, 0.6, 0.85),
        0b0001,
    );

    // A concept: refresh token lifecycle
    let c1 = engine.ingest(
        NodeKind::Summary,
        "Concept: refresh tokens must be rotated before access token expiry",
        Embedding::new(vec![0.82, 0.18, 0.08, 0.2, 0.05]),
        AxisScores::new(0.85, 0.75, 0.65, 0.9),
        0b0001,
    );

    // A policy (principle shell target)
    let r1 = engine.ingest(
        NodeKind::Policy,
        "Principle: sessions shall never silently fail — always surface auth errors",
        Embedding::new(vec![0.8, 0.2, 0.1, 0.22, 0.06]),
        AxisScores::new(0.95, 0.9, 0.8, 0.95),
        0b1111,
    );

    // An opposing claim to drive the contradiction frontier
    let opposite = engine.ingest(
        NodeKind::Summary,
        "Claim: idle timeouts are harmless; clients always retry on 401",
        Embedding::new(vec![0.75, 0.25, 0.15, 0.3, 0.1]),
        AxisScores::new(0.3, 0.2, 0.4, 0.5),
        0b0001,
    );

    // --- 2. Wire up relationships ---
    engine.add_edge(e1, p1, EdgeKind::DerivedFrom, 0.9);
    engine.add_edge(e2, p1, EdgeKind::DerivedFrom, 0.9);
    engine.add_edge(e3, p1, EdgeKind::DerivedFrom, 0.85);
    engine.add_edge(p1, c1, EdgeKind::Refines, 0.9);
    engine.add_edge(p1, c1, EdgeKind::Supports, 0.9);
    engine.add_edge(c1, r1, EdgeKind::Supports, 0.95);
    engine.add_edge(c1, r1, EdgeKind::Refines, 0.95);
    engine.add_edge(p1, r1, EdgeKind::Supports, 0.9);
    engine.add_edge(e1, p1, EdgeKind::Supports, 0.9);
    engine.add_edge(e2, p1, EdgeKind::Supports, 0.9);

    // Semantic antipode — explicit contradiction, not just vector flip
    engine.bind_semantic_antipode(r1, opposite, 0.95);

    // --- 3. Recompute coherence and promote shells ---
    engine.recompute_coherence();
    let promotions = engine.promote_candidates();
    println!("Shell promotions:");
    for (id, before, after) in &promotions {
        println!("  node {:>3}: {:?} → {:?}", id, before, after);
    }
    if promotions.is_empty() {
        println!("  (none — ingest more support edges to trigger promotion)");
    }

    println!("\nCurrent nodes:");
    let mut nodes: Vec<&FieldNode> = engine.nodes.values().collect();
    nodes.sort_by_key(|n| n.id);
    for n in &nodes {
        println!(
            "  id={:>3} shell={:<9?} coherence={:.3} resonance={:.3} text={:?}",
            n.id,
            n.shell,
            n.coherence,
            n.resonance,
            truncate(&n.text, 60)
        );
    }

    // --- 4. Shell-aware retrieval with contradiction frontier ---
    let query = Embedding::new(vec![0.88, 0.12, 0.05, 0.2, 0.02]);
    let result = engine.retrieve(&query, &[Shell::Pattern, Shell::Concept, Shell::Principle], 3);

    println!("\nRetrieval:");
    println!("  selected nodes: {:?}", result.selected);
    println!("  contradiction frontier: {:?}", result.contradiction_frontier);
    println!("  explanation trace:");
    for line in &result.explanation {
        println!("    - {}", line);
    }

    // --- 5. Drift signal against a synthetic baseline centroid ---
    let baseline = Embedding::new(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    let drift = engine.drift(&baseline);
    println!(
        "\nDrift: semantic={:.3} structural={:.3} policy={:.3} identity={:.3} total={:.3}",
        drift.semantic, drift.structural, drift.policy, drift.identity, drift.total
    );
    if drift.total > 0.4 && [drift.semantic, drift.structural, drift.policy, drift.identity]
        .iter()
        .filter(|c| **c > 0.05)
        .count()
        >= 2
    {
        println!("  >> drift alert: at least two channels agree past threshold");
    } else {
        println!("  (no alert — threshold not crossed or not enough agreeing channels)");
    }

    // --- 6. Routing hint based on role embeddings ---
    let roles = vec![
        (
            1001_u64,
            "constraint",
            Embedding::new(vec![0.9, 0.1, 0.0, 0.2, 0.0]),
        ),
        (
            1002_u64,
            "structuring",
            Embedding::new(vec![0.5, 0.5, 0.0, 0.2, 0.0]),
        ),
        (
            1003_u64,
            "synthesis",
            Embedding::new(vec![0.3, 0.3, 0.3, 0.3, 0.3]),
        ),
        (
            1004_u64,
            "verification",
            Embedding::new(vec![0.8, 0.2, 0.1, 0.25, 0.05]),
        ),
    ];
    match engine.route(&query, &roles) {
        Some(hint) => {
            println!(
                "\nRouting hint: agent={:?} gain={:.3} cost={:.3} ttl={} reason={:?}",
                hint.target_agent, hint.gain_estimate, hint.cost_estimate, hint.ttl_epochs, hint.reason
            );
            println!("  note: hint is advisory — privileged mutations must still pass proof + witness gates");
        }
        None => println!("\nNo routing hint available"),
    }

    // --- 7. Phi-scaled budgets for each shell ---
    let base_budget = 1024.0;
    println!("\nShell budgets (base = {base_budget}):");
    for s in [Shell::Event, Shell::Pattern, Shell::Concept, Shell::Principle] {
        println!("  {:<9?} → {:.1}", s, s.budget(base_budget));
    }

    println!("\nDone.");
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_string()
    } else {
        format!("{}...", &s[..n])
    }
}
