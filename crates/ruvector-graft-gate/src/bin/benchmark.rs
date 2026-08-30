//! Benchmark harness for the mincut-gated-insertion nightly (ADR-340).
//!
//! Run: `cargo run --release -p ruvector-graft-gate --bin benchmark`
//!
//! Pipeline: build a clean bootstrapped index (ungated), generate target
//! queries and a synthetic poison attack pool, interleave poison with
//! ordinary corpus growth in one fixed deterministic order, replay that
//! exact order against three gate variants (NoGate, CoherenceRatio,
//! MinCut) starting from cloned copies of the same bootstrapped index,
//! then measure gate latency, poison catch rate, legitimate false-reject
//! rate, attack success rate, and recall@10 for each.

use ruvector_graft_gate::config::*;
use ruvector_graft_gate::data::{
    gen_ball_point, gen_centroids, gen_poison_vector, shuffle_indices, InsertItem,
};
use ruvector_graft_gate::graph_index::brute_force_top_k;
use ruvector_graft_gate::{evaluate_gate, GateConfig, GateVariant, GraphIndex, Xorshift64};
use std::collections::HashMap;
use std::time::Instant;

struct VariantOutcome {
    name: &'static str,
    gate_ns: Vec<u64>,
    poison_admitted: usize,
    poison_total: usize,
    legit_admitted: usize,
    legit_total: usize,
    attack_success_unconditional: usize,
    attack_success_conditional_num: usize,
    attack_success_conditional_den: usize,
    recall_sum: f32,
    recall_n: usize,
    total_insert_wall_ns: u128,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean_u64(v: &[u64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<u64>() as f64 / v.len() as f64
    }
}

fn run_variant(
    variant: GateVariant,
    name: &'static str,
    base_index: &GraphIndex,
    items: &[InsertItem],
    gate_cfg: &GateConfig,
    ground_truth_vectors: &[Vec<f32>],
    target_queries: &[(usize, Vec<f32>)], // (target_id, vector)
) -> VariantOutcome {
    let mut index = base_index.clone();
    // content_of[node_id] = content_id, for every node in `index`
    // (clean-corpus nodes get content_id == node_id by construction).
    let mut content_of: Vec<u64> = (0..index.len() as u64).collect();
    // poison content_id -> target_id, recorded only for admitted poison.
    let mut poison_target_by_content: HashMap<u64, usize> = HashMap::new();

    let mut gate_ns = Vec::with_capacity(items.len());
    let mut poison_admitted = 0usize;
    let mut poison_total = 0usize;
    let mut legit_admitted = 0usize;
    let mut legit_total = 0usize;
    let insert_start = Instant::now();

    for item in items {
        let vector = item.vector().to_vec();
        let search_result = index.search(&vector, EF_CONSTRUCTION);

        let t0 = Instant::now();
        let decision = evaluate_gate(variant, gate_cfg, &index, &search_result);
        gate_ns.push(t0.elapsed().as_nanos() as u64);

        match item {
            InsertItem::Legit { content_id, .. } => {
                legit_total += 1;
                if decision.admit {
                    legit_admitted += 1;
                    let id = index.insert_with_neighbors(vector, &search_result);
                    content_of.push(0); // placeholder, fixed below
                    let idx = content_of.len() - 1;
                    debug_assert_eq!(idx as u32, id);
                    content_of[idx] = *content_id;
                }
            }
            InsertItem::Poison {
                content_id,
                target_id,
                ..
            } => {
                poison_total += 1;
                if decision.admit {
                    poison_admitted += 1;
                    let id = index.insert_with_neighbors(vector, &search_result);
                    content_of.push(0);
                    let idx = content_of.len() - 1;
                    debug_assert_eq!(idx as u32, id);
                    content_of[idx] = *content_id;
                    poison_target_by_content.insert(*content_id, *target_id);
                }
            }
        }
    }
    let total_insert_wall_ns = insert_start.elapsed().as_nanos();

    // Which targets have at least one surviving (admitted) poison?
    let mut targets_with_surviving_poison: HashMap<usize, bool> = HashMap::new();
    for &tid in poison_target_by_content.values() {
        targets_with_surviving_poison.insert(tid, true);
    }

    let mut attack_success_unconditional = 0usize;
    let mut attack_success_conditional_num = 0usize;
    let mut attack_success_conditional_den = 0usize;
    let mut recall_sum = 0f32;
    let mut recall_n = 0usize;

    for (target_id, query) in target_queries {
        let gt = brute_force_top_k(ground_truth_vectors, query, TOP_K);
        let actual: Vec<(u32, f32)> = index.search(query, EF_SEARCH);
        let actual_top: Vec<u32> = actual.into_iter().take(TOP_K).map(|(id, _)| id).collect();

        let mut poison_hit = false;
        let mut matched_content_ids: Vec<u64> = Vec::with_capacity(actual_top.len());
        for &node_id in &actual_top {
            let cid = content_of[node_id as usize];
            matched_content_ids.push(cid);
            if let Some(&tid) = poison_target_by_content.get(&cid) {
                if tid == *target_id {
                    poison_hit = true;
                }
            }
        }
        if poison_hit {
            attack_success_unconditional += 1;
        }
        if targets_with_surviving_poison.contains_key(target_id) {
            attack_success_conditional_den += 1;
            if poison_hit {
                attack_success_conditional_num += 1;
            }
        }

        let overlap = gt
            .iter()
            .filter(|&&g| matched_content_ids.contains(&(g as u64)))
            .count();
        recall_sum += overlap as f32 / TOP_K as f32;
        recall_n += 1;
    }

    VariantOutcome {
        name,
        gate_ns,
        poison_admitted,
        poison_total,
        legit_admitted,
        legit_total,
        attack_success_unconditional,
        attack_success_conditional_num,
        attack_success_conditional_den,
        recall_sum,
        recall_n,
        total_insert_wall_ns,
    }
}

fn main() {
    println!("=== ruvector-graft-gate benchmark ===");
    println!(
        "dim={DIM} clusters={NUM_CLUSTERS} n_clean={N_CLEAN} n_additional_legit={N_ADDITIONAL_LEGIT} \
         n_target_queries={N_TARGET_QUERIES} poison_per_target={POISON_PER_TARGET} (total_poison={}) \
         alpha={POISON_ALPHA} m={GRAPH_M} ef_construction={EF_CONSTRUCTION} ef_search={EF_SEARCH}",
        N_TARGET_QUERIES * POISON_PER_TARGET
    );
    println!(
        "gate_k={GATE_K} peakedness_threshold={PEAKEDNESS_THRESHOLD} mincut_edge_factor={MINCUT_EDGE_FACTOR} \
         mincut_reject_below={MINCUT_REJECT_BELOW}"
    );

    // 1. Clusters + clean bootstrapped corpus (ungated, matches the
    //    hypothesis's "bootstrapped with 5,000 ... vectors" clause).
    let mut rng_centroids = Xorshift64::new(SEED_CENTROIDS);
    let centroids = gen_centroids(&mut rng_centroids, NUM_CLUSTERS, DIM);

    let mut rng_clean = Xorshift64::new(SEED_CLEAN_CORPUS);
    let mut clean_vectors: Vec<Vec<f32>> = Vec::with_capacity(N_CLEAN);
    let mut base_index = GraphIndex::new(DIM, GRAPH_M);
    let t_ingest = Instant::now();
    for i in 0..N_CLEAN {
        let c = &centroids[i % NUM_CLUSTERS];
        let v = gen_ball_point(&mut rng_clean, c, CLUSTER_SIGMA);
        clean_vectors.push(v.clone());
        let sr = base_index.search(&v, EF_CONSTRUCTION);
        base_index.insert_with_neighbors(v, &sr);
    }
    let ingest_ms = t_ingest.elapsed().as_secs_f64() * 1000.0;
    println!(
        "bootstrap ingest: {N_CLEAN} clean vectors in {ingest_ms:.3} ms, entry_points={}",
        base_index.entry_points.len()
    );

    // 2. Target queries (one per cluster round-robin, representing real
    //    topics an attacker wants to manipulate).
    let mut rng_queries = Xorshift64::new(SEED_TARGET_QUERIES);
    let mut target_queries: Vec<(usize, Vec<f32>)> = Vec::with_capacity(N_TARGET_QUERIES);
    for t in 0..N_TARGET_QUERIES {
        let c = &centroids[t % NUM_CLUSTERS];
        let q = gen_ball_point(&mut rng_queries, c, CLUSTER_SIGMA);
        target_queries.push((t, q));
    }

    // 3. Poison pool: POISON_PER_TARGET attempts per target query.
    let mut rng_poison = Xorshift64::new(SEED_POISON);
    let mut poison_items: Vec<InsertItem> =
        Vec::with_capacity(N_TARGET_QUERIES * POISON_PER_TARGET);
    let mut next_content_id: u64 = N_CLEAN as u64 + N_ADDITIONAL_LEGIT as u64;
    for (target_id, query) in &target_queries {
        for _ in 0..POISON_PER_TARGET {
            let v = gen_poison_vector(&mut rng_poison, query, POISON_ALPHA);
            poison_items.push(InsertItem::Poison {
                content_id: next_content_id,
                target_id: *target_id,
                vector: v,
            });
            next_content_id += 1;
        }
    }

    // 4. Additional legit insertions (ordinary corpus growth).
    let mut rng_legit_extra = Xorshift64::new(SEED_ADDITIONAL_LEGIT);
    let mut additional_legit: Vec<Vec<f32>> = Vec::with_capacity(N_ADDITIONAL_LEGIT);
    let mut legit_items: Vec<InsertItem> = Vec::with_capacity(N_ADDITIONAL_LEGIT);
    for i in 0..N_ADDITIONAL_LEGIT {
        let c = &centroids[i % NUM_CLUSTERS];
        let v = gen_ball_point(&mut rng_legit_extra, c, CLUSTER_SIGMA);
        additional_legit.push(v.clone());
        legit_items.push(InsertItem::Legit {
            content_id: N_CLEAN as u64 + i as u64,
            vector: v,
        });
    }

    // 5. Fixed deterministic interleave, shared across all three variants.
    let mut all_items: Vec<InsertItem> = Vec::with_capacity(legit_items.len() + poison_items.len());
    all_items.extend(legit_items);
    all_items.extend(poison_items);
    let mut rng_shuffle = Xorshift64::new(SEED_INTERLEAVE_SHUFFLE);
    let order = shuffle_indices(&mut rng_shuffle, all_items.len());
    let interleaved: Vec<InsertItem> = order.into_iter().map(|i| all_items[i].clone()).collect();

    // Ground truth universe: every legitimate vector that exists, whether
    // or not a given variant happened to admit it — this is the fixed
    // yardstick recall is measured against for all three variants.
    let mut ground_truth_vectors: Vec<Vec<f32>> = Vec::with_capacity(N_CLEAN + N_ADDITIONAL_LEGIT);
    ground_truth_vectors.extend(clean_vectors.iter().cloned());
    ground_truth_vectors.extend(additional_legit.iter().cloned());

    let gate_cfg = GateConfig::default();
    let variants = [
        (GateVariant::NoGate, "NoGate"),
        (GateVariant::CoherenceRatio, "CoherenceRatio"),
        (GateVariant::MinCut, "MinCut"),
    ];

    let mut outcomes = Vec::new();
    for (variant, name) in variants {
        let o = run_variant(
            variant,
            name,
            &base_index,
            &interleaved,
            &gate_cfg,
            &ground_truth_vectors,
            &target_queries,
        );
        outcomes.push(o);
    }

    println!();
    println!(
        "{:<15} {:>12} {:>10} {:>12} {:>12} {:>10} {:>10} {:>12} {:>16} {:>10} {:>12}",
        "variant",
        "gate_mean_ns",
        "gate_p50",
        "gate_p95",
        "poison_catch",
        "catch_%",
        "legit_fr",
        "legit_fr_%",
        "attack_unc_%",
        "attack_c_%",
        "recall@10"
    );
    for o in &outcomes {
        let mut sorted = o.gate_ns.clone();
        sorted.sort_unstable();
        let mean = mean_u64(&o.gate_ns);
        let p50 = percentile(&sorted, 0.50);
        let p95 = percentile(&sorted, 0.95);
        let poison_rejected = o.poison_total - o.poison_admitted;
        let catch_pct = 100.0 * poison_rejected as f64 / o.poison_total as f64;
        let legit_rejected = o.legit_total - o.legit_admitted;
        let legit_fr_pct = 100.0 * legit_rejected as f64 / o.legit_total as f64;
        let attack_unc_pct = 100.0 * o.attack_success_unconditional as f64 / o.recall_n as f64;
        let attack_c_pct = if o.attack_success_conditional_den > 0 {
            100.0 * o.attack_success_conditional_num as f64
                / o.attack_success_conditional_den as f64
        } else {
            f64::NAN
        };
        let recall = o.recall_sum / o.recall_n as f32;
        println!(
            "{:<15} {:>12.1} {:>10} {:>12} {:>12}/{:<3} {:>9.1} {:>10}/{:<5} {:>10.1} {:>15.1} {:>10.1} {:>12.4}",
            o.name, mean, p50, p95, poison_rejected, o.poison_total, catch_pct, legit_rejected, o.legit_total,
            legit_fr_pct, attack_unc_pct, attack_c_pct, recall
        );
    }

    println!();
    println!("total_insert_wall_time_ms:");
    for o in &outcomes {
        println!(
            "  {:<15} {:>10.3} ms",
            o.name,
            o.total_insert_wall_ns as f64 / 1_000_000.0
        );
    }

    // === Acceptance evaluation against the pre-registered hypothesis ===
    let no_gate = outcomes.iter().find(|o| o.name == "NoGate").unwrap();
    let coherence = outcomes
        .iter()
        .find(|o| o.name == "CoherenceRatio")
        .unwrap();
    let mincut = outcomes.iter().find(|o| o.name == "MinCut").unwrap();

    let catch_rate = |o: &VariantOutcome| {
        100.0 * (o.poison_total - o.poison_admitted) as f64 / o.poison_total as f64
    };
    let legit_fr_rate = |o: &VariantOutcome| {
        100.0 * (o.legit_total - o.legit_admitted) as f64 / o.legit_total as f64
    };
    let attack_c_rate = |o: &VariantOutcome| {
        if o.attack_success_conditional_den > 0 {
            100.0 * o.attack_success_conditional_num as f64
                / o.attack_success_conditional_den as f64
        } else {
            f64::NAN
        }
    };
    let recall = |o: &VariantOutcome| o.recall_sum / o.recall_n as f32;

    let mincut_catch = catch_rate(mincut);
    let coherence_catch = catch_rate(coherence);
    let clause_a = mincut_catch - coherence_catch >= CATCH_RATE_GAP_BUDGET_PP as f64;

    let coherence_fr = legit_fr_rate(coherence);
    let mincut_fr = legit_fr_rate(mincut);
    let clause_b = coherence_fr <= LEGIT_FALSE_REJECT_BUDGET_PCT as f64
        && mincut_fr <= LEGIT_FALSE_REJECT_BUDGET_PCT as f64;

    let no_gate_attack_c = attack_c_rate(no_gate);
    let mincut_attack_c = attack_c_rate(mincut);
    let clause_c = if no_gate_attack_c.is_nan() || mincut_attack_c.is_nan() {
        false
    } else {
        no_gate_attack_c - mincut_attack_c >= ATTACK_SUCCESS_GAP_BUDGET_PP as f64
    };

    let coherence_mean_ns = mean_u64(&coherence.gate_ns);
    let mincut_mean_ns = mean_u64(&mincut.gate_ns);
    let latency_ok = coherence_mean_ns <= LATENCY_BUDGET_NS && mincut_mean_ns <= LATENCY_BUDGET_NS;

    let no_gate_recall = recall(no_gate);
    let coherence_recall = recall(coherence);
    let mincut_recall = recall(mincut);
    let recall_ok = (no_gate_recall - coherence_recall) * 100.0 <= RECALL_DROP_BUDGET_PP
        && (no_gate_recall - mincut_recall) * 100.0 <= RECALL_DROP_BUDGET_PP;

    println!();
    println!("=== acceptance ===");
    println!(
        "(a) mincut_catch({mincut_catch:.1}%) - coherence_catch({coherence_catch:.1}%) >= {CATCH_RATE_GAP_BUDGET_PP}pp: {clause_a}"
    );
    println!(
        "(b) legit false-reject <= {LEGIT_FALSE_REJECT_BUDGET_PCT}%: coherence={coherence_fr:.2}% mincut={mincut_fr:.2}% -> {clause_b}"
    );
    println!(
        "(c) attack_success_conditional: no_gate={no_gate_attack_c:.1}% mincut={mincut_attack_c:.1}% gap>={ATTACK_SUCCESS_GAP_BUDGET_PP}pp: {clause_c}"
    );
    println!(
        "subject-to latency: coherence_mean={coherence_mean_ns:.0}ns mincut_mean={mincut_mean_ns:.0}ns budget={LATENCY_BUDGET_NS:.0}ns -> {latency_ok}"
    );
    println!(
        "subject-to recall drop <= {RECALL_DROP_BUDGET_PP}pp: no_gate={no_gate_recall:.4} coherence={coherence_recall:.4} mincut={mincut_recall:.4} -> {recall_ok}"
    );

    // "Subject to" clauses are validity conditions on the experiment
    // itself (per ADR-340): if they fail, the a/b/c measurements were
    // taken outside the regime the hypothesis assumed, so the result is
    // INCONCLUSIVE rather than a clean falsification. If they hold,
    // clauses (a)/(b)/(c) decide ACCEPT vs REJECT directly.
    let subject_to_ok = latency_ok && recall_ok;
    let verdict = if !subject_to_ok {
        "INCONCLUSIVE"
    } else if clause_a && clause_b && clause_c {
        "ACCEPT"
    } else {
        "REJECT"
    };
    println!();
    println!("ACCEPTANCE RESULT: {verdict}");
}
