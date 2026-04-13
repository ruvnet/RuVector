//! RuVector Field Subsystem — runnable demo binary.
//!
//! A thin CLI that exercises the library implementation end-to-end. Run:
//!
//! ```text
//! cargo run --bin field_demo -- --nodes 16 --query "authentication timeout"
//! cargo run --bin field_demo -- --help
//! ```

use std::env;
use std::process::ExitCode;

use ruvector_field::prelude::*;
use ruvector_field::engine::route::RoutingAgent;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let opts = match parse_args(&args) {
        Ok(opts) => opts,
        Err(msg) => {
            eprintln!("error: {}", msg);
            print_usage(&args[0]);
            return ExitCode::from(2);
        }
    };
    if opts.help {
        print_usage(&args[0]);
        return ExitCode::SUCCESS;
    }

    println!("=== RuVector Field Subsystem Demo ===");
    println!("(nodes={}, seed={}, query={:?})\n", opts.nodes, opts.seed, opts.query);

    let provider = HashEmbeddingProvider::new(32);
    let mut engine = FieldEngine::new();
    seed_policies(&mut engine);

    let corpus = build_corpus(opts.nodes, opts.seed);
    let mut ids = Vec::new();
    for (kind, text, axes, mask) in &corpus {
        let emb = provider.embed(text);
        let id = engine
            .ingest(*kind, text.clone(), emb, *axes, *mask)
            .expect("ingest");
        ids.push(id);
    }

    // Wire edges: every odd node derives from its predecessor; every fifth
    // node supports the one before it; the final node contradicts the first.
    for (i, id) in ids.iter().enumerate() {
        if i > 0 && i % 2 == 1 {
            engine
                .add_edge(ids[i - 1], *id, EdgeKind::DerivedFrom, 0.9)
                .expect("edge");
        }
        if i > 0 && i % 3 == 0 {
            engine.add_edge(*id, ids[i - 1], EdgeKind::Supports, 0.85).expect("edge");
        }
        if i > 0 && i % 5 == 0 {
            engine.add_edge(*id, ids[i - 1], EdgeKind::Refines, 0.8).expect("edge");
        }
    }
    if ids.len() >= 2 {
        engine
            .bind_semantic_antipode(ids[0], ids[ids.len() - 1], 0.9)
            .expect("antipode");
    }

    // Force tick + two promotion passes so hysteresis can fire.
    engine.tick();
    for _ in 0..3 {
        let _ = engine.promote_candidates();
    }
    let final_promotions = engine.promote_candidates();

    println!("Shell promotions (final pass):");
    if final_promotions.is_empty() {
        println!("  (none this pass)");
    } else {
        for rec in &final_promotions {
            println!("  {}", rec);
        }
    }

    println!("\nCurrent nodes:");
    let mut nodes: Vec<&FieldNode> = engine.nodes.values().collect();
    nodes.sort_by_key(|n| n.id);
    for n in &nodes {
        println!("  {}", n);
    }

    // Retrieval with the parsed query.
    let query_emb = provider.embed(&opts.query);
    let shells = if opts.shells.is_empty() {
        vec![Shell::Event, Shell::Pattern, Shell::Concept, Shell::Principle]
    } else {
        opts.shells.clone()
    };
    let result = engine.retrieve(&query_emb, &shells, 3, None);
    println!("\nRetrieval {}", result);
    for line in &result.explanation {
        println!("  {}", line);
    }

    // Drift
    if opts.show_drift {
        let baseline = provider.embed("baseline reference corpus drift");
        let drift = engine.drift(&baseline);
        println!("\n{}", drift);
        if drift.agreement_fires(0.4, 0.1) {
            println!("  >> drift alert: two or more channels agree past threshold");
        } else {
            println!("  (no alert — threshold not crossed or channels do not agree)");
        }
    }

    // Routing
    let agents = vec![
        RoutingAgent {
            agent_id: 1001,
            role: "constraint".into(),
            capability: provider.embed("constraint guardrail limit"),
            role_embedding: provider.embed("constraint"),
            home_node: ids.first().copied(),
            home_shell: Shell::Principle,
        },
        RoutingAgent {
            agent_id: 1002,
            role: "synthesis".into(),
            capability: provider.embed("synthesis bridge combine"),
            role_embedding: provider.embed("synthesis"),
            home_node: ids.get(ids.len() / 2).copied(),
            home_shell: Shell::Concept,
        },
        RoutingAgent {
            agent_id: 1003,
            role: "verification".into(),
            capability: provider.embed("verification audit check"),
            role_embedding: provider.embed("verification"),
            home_node: ids.last().copied(),
            home_shell: Shell::Concept,
        },
    ];
    if let Some(hint) = engine.route(&query_emb, Shell::Concept, &agents, ids.first().copied(), false) {
        println!("\nRouting hint: {}", hint);
    }

    // Phi budgets
    let base = 1024.0;
    println!("\nShell budgets (base = {base}):");
    for s in Shell::all() {
        println!("  {:<9} -> {:.1}", format!("{}", s), s.budget(base));
    }

    if opts.show_witness {
        println!("\nWitness events:");
        for ev in engine.witness.events() {
            println!("  {} {:?}", ev.tag(), ev);
        }
    }

    println!("\nDone.");
    ExitCode::SUCCESS
}

struct Opts {
    nodes: usize,
    query: String,
    shells: Vec<Shell>,
    show_witness: bool,
    show_drift: bool,
    seed: u64,
    help: bool,
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut opts = Opts {
        nodes: 8,
        query: "authentication timeout".to_string(),
        shells: Vec::new(),
        show_witness: false,
        show_drift: true,
        seed: 42,
        help: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => opts.help = true,
            "--nodes" => {
                i += 1;
                opts.nodes = args
                    .get(i)
                    .ok_or("--nodes requires a value")?
                    .parse::<usize>()
                    .map_err(|e| format!("--nodes: {}", e))?;
            }
            "--query" => {
                i += 1;
                opts.query = args.get(i).ok_or("--query requires a value")?.clone();
            }
            "--shells" => {
                i += 1;
                let raw = args.get(i).ok_or("--shells requires a value")?;
                for part in raw.split(',') {
                    let s: Shell = part.parse().map_err(|e: &str| e.to_string())?;
                    opts.shells.push(s);
                }
            }
            "--show-witness" => opts.show_witness = true,
            "--show-drift" => opts.show_drift = true,
            "--no-drift" => opts.show_drift = false,
            "--seed" => {
                i += 1;
                opts.seed = args
                    .get(i)
                    .ok_or("--seed requires a value")?
                    .parse::<u64>()
                    .map_err(|e| format!("--seed: {}", e))?;
            }
            other => return Err(format!("unknown flag: {}", other)),
        }
        i += 1;
    }
    Ok(opts)
}

fn print_usage(argv0: &str) {
    println!("Usage: {} [flags]", argv0);
    println!();
    println!("Flags:");
    println!("  --nodes N           Number of synthetic nodes to seed (default 8)");
    println!("  --query TEXT        Retrieval query text (default \"authentication timeout\")");
    println!("  --shells S1,S2,..   Allowed shells (event,pattern,concept,principle)");
    println!("  --show-witness      Print the full witness event list");
    println!("  --show-drift        Print drift analysis (default on)");
    println!("  --no-drift          Disable drift printout");
    println!("  --seed N            Deterministic seed (default 42)");
    println!("  --help              Show this help");
}

fn seed_policies(engine: &mut FieldEngine) {
    let mut reg = PolicyRegistry::new();
    reg.register(Policy {
        id: 1,
        name: "safety".into(),
        mask: 0b0001,
        required_axes: AxisConstraints {
            limit: AxisConstraint::min(0.4),
            care: AxisConstraint::min(0.4),
            bridge: AxisConstraint::any(),
            clarity: AxisConstraint::min(0.3),
        },
    });
    engine.set_policy_registry(reg);
}

fn build_corpus(n: usize, seed: u64) -> Vec<(NodeKind, String, AxisScores, u64)> {
    let templates = [
        ("user reports authentication timeout after idle", NodeKind::Interaction),
        ("session refresh silently fails after JWT expiry", NodeKind::Interaction),
        ("mobile client hits auth timeout on weak network", NodeKind::Interaction),
        ("pattern: idle timeout causes refresh failure", NodeKind::Summary),
        ("pattern: retry loop cures transient auth outage", NodeKind::Summary),
        ("concept: refresh tokens must rotate before access expiry", NodeKind::Summary),
        ("concept: silent failures must never reach the user", NodeKind::Summary),
        ("principle: sessions shall surface auth errors", NodeKind::Policy),
        ("principle: refresh token compromise forces re-auth", NodeKind::Policy),
        ("claim: idle timeouts are harmless; clients always retry", NodeKind::Summary),
    ];
    let mut out = Vec::new();
    for i in 0..n {
        let t = templates[(i + seed as usize) % templates.len()];
        let mix = ((i as f32) * 0.01) % 1.0;
        out.push((
            t.1,
            format!("{} #{}", t.0, i),
            AxisScores::new(0.6 + mix, 0.55, 0.5, 0.7),
            0b0001,
        ));
    }
    out
}
