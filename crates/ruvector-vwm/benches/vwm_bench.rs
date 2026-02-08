//! Criterion benchmark suite for the ruvector-vwm crate.
//!
//! Covers Gaussian evaluation, tile encoding/decoding, draw list construction,
//! coherence gating, entity graph operations, active masks, bandwidth budgeting,
//! and depth sorting of screen-space Gaussians.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use ruvector_vwm::coherence::{CoherenceGate, CoherenceInput, PermissionLevel};
use ruvector_vwm::draw_list::{DrawList, OpacityMode};
use ruvector_vwm::entity::{Edge, EdgeType, Entity, EntityGraph, EntityType};
use ruvector_vwm::gaussian::{Gaussian4D, ScreenGaussian};
use ruvector_vwm::streaming::{ActiveMask, BandwidthBudget};
use ruvector_vwm::tile::{PrimitiveBlock, QuantTier};

// ---------------------------------------------------------------------------
// Helpers: realistic test data generators
// ---------------------------------------------------------------------------

/// Generate a vector of Gaussians with varied positions, velocities, and time ranges.
fn make_gaussians(n: usize) -> Vec<Gaussian4D> {
    (0..n)
        .map(|i| {
            let fi = i as f32;
            let mut g = Gaussian4D::new(
                [fi * 0.1, (fi * 0.7).sin() * 5.0, -10.0 + (fi * 0.3).cos()],
                i as u32,
            );
            g.velocity = [0.01 * fi.sin(), 0.02 * fi.cos(), 0.005];
            g.time_range = [0.0, 100.0];
            g.opacity = 0.5 + 0.5 * ((fi * 0.1).sin()).abs();
            g.sh_coeffs = [
                0.3 + 0.2 * (fi * 0.05).sin(),
                0.4 + 0.1 * (fi * 0.07).cos(),
                0.35,
            ];
            g.scale = [
                0.5 + 0.5 * (fi * 0.03).sin().abs(),
                0.5 + 0.5 * (fi * 0.04).cos().abs(),
                0.5 + 0.5 * (fi * 0.05).sin().abs(),
            ];
            g
        })
        .collect()
}

/// Build a simple perspective-like view-projection matrix.
fn make_view_proj() -> [f32; 16] {
    // Near-identity with perspective-like w component
    [
        1.0, 0.0, 0.0, 0.0, // col 0
        0.0, 1.0, 0.0, 0.0, // col 1
        0.0, 0.0, 1.0, 0.1, // col 2 (w gets depth contribution)
        0.0, 0.0, 0.0, 1.0, // col 3
    ]
}

/// Generate screen gaussians by projecting real Gaussian4D data.
fn make_screen_gaussians(n: usize) -> Vec<ScreenGaussian> {
    let vp = make_view_proj();
    let gaussians = make_gaussians(n * 2); // project more to ensure we get enough
    let mut result: Vec<ScreenGaussian> = gaussians
        .iter()
        .filter_map(|g| g.project(&vp, 50.0))
        .collect();
    result.truncate(n);
    // If we didn't get enough (unlikely), pad with synthetic data
    while result.len() < n {
        let i = result.len();
        result.push(ScreenGaussian {
            center_screen: [i as f32 * 0.01, i as f32 * 0.02],
            depth: 1.0 + i as f32 * 0.1,
            conic: [1.0, 0.0, 1.0],
            color: [0.5, 0.5, 0.5],
            opacity: 0.8,
            radius: 2.0,
            id: i as u32,
        });
    }
    result
}

/// Build coherence inputs with realistic variation.
fn make_coherence_inputs(n: usize) -> Vec<CoherenceInput> {
    (0..n)
        .map(|i| {
            let fi = i as f32;
            CoherenceInput {
                tile_disagreement: 0.05 + 0.1 * (fi * 0.1).sin().abs(),
                entity_continuity: 0.7 + 0.3 * (fi * 0.07).cos().abs(),
                sensor_confidence: 0.8 + 0.2 * (fi * 0.03).sin().abs(),
                sensor_freshness_ms: 50 + (i as u64 % 200),
                budget_pressure: 0.1 + 0.3 * (fi * 0.05).cos().abs(),
                permission_level: match i % 4 {
                    0 => PermissionLevel::Standard,
                    1 => PermissionLevel::Elevated,
                    2 => PermissionLevel::Standard,
                    _ => PermissionLevel::Standard,
                },
            }
        })
        .collect()
}

/// Create an entity with a small embedding.
fn make_entity(id: u64, class: &str, time: [f32; 2]) -> Entity {
    Entity {
        id,
        entity_type: EntityType::Object {
            class: class.to_string(),
        },
        time_span: time,
        embedding: (0..16).map(|j| (id as f32 + j as f32) * 0.1).collect(),
        confidence: 0.9,
        privacy_tags: vec![],
        attributes: vec![],
        gaussian_ids: vec![id as u32],
    }
}

// ---------------------------------------------------------------------------
// Gaussian benchmarks
// ---------------------------------------------------------------------------

fn bench_gaussian_position_at(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_position_at");
    for &size in &[1_000usize, 10_000, 100_000] {
        let gaussians = make_gaussians(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &gaussians, |b, gs| {
            b.iter(|| {
                for g in gs {
                    black_box(g.position_at(black_box(42.5)));
                }
            });
        });
    }
    group.finish();
}

fn bench_gaussian_project(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_project");
    let vp = make_view_proj();
    for &size in &[1_000usize, 10_000] {
        let gaussians = make_gaussians(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &gaussians, |b, gs| {
            b.iter(|| {
                for g in gs {
                    black_box(g.project(black_box(&vp), black_box(50.0)));
                }
            });
        });
    }
    group.finish();
}

fn bench_gaussian_is_active(c: &mut Criterion) {
    let gaussians = make_gaussians(100_000);
    c.bench_function("gaussian_is_active_100k", |b| {
        b.iter(|| {
            for g in &gaussians {
                black_box(g.is_active_at(black_box(50.0)));
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Tile / PrimitiveBlock benchmarks
// ---------------------------------------------------------------------------

fn bench_primitive_block_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive_block_encode");
    for &size in &[100usize, 1_000, 10_000] {
        let gaussians = make_gaussians(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &gaussians, |b, gs| {
            b.iter(|| {
                black_box(PrimitiveBlock::encode(black_box(gs), QuantTier::Hot8));
            });
        });
    }
    group.finish();
}

fn bench_primitive_block_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive_block_decode");
    for &size in &[100usize, 1_000, 10_000] {
        let gaussians = make_gaussians(size);
        let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
        group.bench_with_input(BenchmarkId::from_parameter(size), &block, |b, blk| {
            b.iter(|| {
                black_box(blk.decode());
            });
        });
    }
    group.finish();
}

fn bench_primitive_block_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitive_block_roundtrip");
    for &size in &[100usize, 1_000, 10_000] {
        let gaussians = make_gaussians(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &gaussians, |b, gs| {
            b.iter(|| {
                let block = PrimitiveBlock::encode(black_box(gs), QuantTier::Hot8);
                black_box(block.decode());
            });
        });
    }
    group.finish();
}

fn bench_checksum(c: &mut Criterion) {
    let mut group = c.benchmark_group("checksum");
    for &size in &[100usize, 1_000, 10_000] {
        let gaussians = make_gaussians(size);
        let block = PrimitiveBlock::encode(&gaussians, QuantTier::Hot8);
        group.bench_with_input(BenchmarkId::from_parameter(size), &block, |b, blk| {
            b.iter(|| {
                black_box(blk.compute_checksum());
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Draw list benchmarks
// ---------------------------------------------------------------------------

fn bench_draw_list_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_list_build");
    for &size in &[100u32, 1_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let mut dl = DrawList::new(1, 0, 0);
                for i in 0..n {
                    dl.bind_tile(i as u64, i, QuantTier::Hot8);
                    dl.draw_block(i, i as f32 * 0.1, OpacityMode::AlphaBlend);
                }
                black_box(&dl);
            });
        });
    }
    group.finish();
}

fn bench_draw_list_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_list_serialize");
    for &size in &[100u32, 1_000] {
        let mut dl = DrawList::new(1, 0, 0);
        for i in 0..size {
            dl.bind_tile(i as u64, i, QuantTier::Hot8);
            dl.draw_block(i, i as f32 * 0.1, OpacityMode::AlphaBlend);
        }
        dl.finalize();
        group.bench_with_input(BenchmarkId::from_parameter(size), &dl, |b, d| {
            b.iter(|| {
                black_box(d.to_bytes());
            });
        });
    }
    group.finish();
}

fn bench_draw_list_finalize(c: &mut Criterion) {
    let mut group = c.benchmark_group("draw_list_finalize");
    for &size in &[100u32, 1_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let mut dl = DrawList::new(1, 0, 0);
                for i in 0..n {
                    dl.bind_tile(i as u64, i, QuantTier::Hot8);
                    dl.draw_block(i, i as f32 * 0.1, OpacityMode::AlphaBlend);
                }
                black_box(dl.finalize());
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Coherence gate benchmarks
// ---------------------------------------------------------------------------

fn bench_coherence_evaluate(c: &mut Criterion) {
    let gate = CoherenceGate::with_defaults();
    let inputs = make_coherence_inputs(10_000);
    c.bench_function("coherence_evaluate_10k", |b| {
        b.iter(|| {
            for input in &inputs {
                black_box(gate.evaluate(black_box(input)));
            }
        });
    });
}

fn bench_coherence_batch(c: &mut Criterion) {
    let gate = CoherenceGate::with_defaults();
    // Diverse inputs: mix of accept, defer, freeze, rollback triggers
    let mut inputs = Vec::with_capacity(5_000);
    for i in 0..5_000usize {
        let fi = i as f32;
        inputs.push(CoherenceInput {
            tile_disagreement: match i % 5 {
                0 => 0.05,                      // low -> accept path
                1 => 0.5,                       // moderate
                2 => 0.85,                      // freeze
                3 => 0.96,                      // rollback
                _ => 0.2 + 0.1 * fi.sin().abs(),
            },
            entity_continuity: 0.3 + 0.7 * (fi * 0.03).cos().abs(),
            sensor_confidence: 0.6 + 0.4 * (fi * 0.05).sin().abs(),
            sensor_freshness_ms: if i % 10 == 0 { 8000 } else { 100 },
            budget_pressure: if i % 8 == 0 { 0.95 } else { 0.2 },
            permission_level: match i % 20 {
                0 => PermissionLevel::Admin,
                1 => PermissionLevel::ReadOnly,
                2 => PermissionLevel::Elevated,
                _ => PermissionLevel::Standard,
            },
        });
    }
    c.bench_function("coherence_batch_diverse_5k", |b| {
        b.iter(|| {
            for input in &inputs {
                black_box(gate.evaluate(black_box(input)));
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Entity graph benchmarks
// ---------------------------------------------------------------------------

fn bench_entity_graph_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_graph_add");
    let classes = ["car", "person", "tree", "building", "sign"];
    for &size in &[1_000usize, 10_000] {
        let entities: Vec<Entity> = (0..size)
            .map(|i| {
                make_entity(
                    i as u64,
                    classes[i % classes.len()],
                    [i as f32 * 0.1, i as f32 * 0.1 + 5.0],
                )
            })
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(size), &entities, |b, ents| {
            b.iter(|| {
                let mut graph = EntityGraph::new();
                for e in ents {
                    graph.add_entity(e.clone());
                }
                black_box(graph.entity_count());
            });
        });
    }
    group.finish();
}

fn bench_entity_graph_query_type(c: &mut Criterion) {
    let classes = ["car", "person", "tree", "building", "sign"];
    let mut graph = EntityGraph::new();
    for i in 0..10_000u64 {
        graph.add_entity(make_entity(
            i,
            classes[i as usize % classes.len()],
            [i as f32 * 0.1, i as f32 * 0.1 + 5.0],
        ));
    }
    c.bench_function("entity_graph_query_type_10k", |b| {
        b.iter(|| {
            black_box(graph.query_by_type(black_box("car")));
        });
    });
}

fn bench_entity_graph_query_time(c: &mut Criterion) {
    let classes = ["car", "person", "tree", "building", "sign"];
    let mut graph = EntityGraph::new();
    for i in 0..10_000u64 {
        graph.add_entity(make_entity(
            i,
            classes[i as usize % classes.len()],
            [i as f32 * 0.1, i as f32 * 0.1 + 5.0],
        ));
    }
    c.bench_function("entity_graph_query_time_10k", |b| {
        b.iter(|| {
            black_box(graph.query_time_range(black_box(200.0), black_box(300.0)));
        });
    });
}

fn bench_entity_graph_neighbors(c: &mut Criterion) {
    // Dense graph: 1K entities, ~5K edges
    let classes = ["car", "person", "tree"];
    let mut graph = EntityGraph::new();
    for i in 0..1_000u64 {
        graph.add_entity(make_entity(
            i,
            classes[i as usize % classes.len()],
            [0.0, 100.0],
        ));
    }
    // Create a dense edge set: each entity connects to 5 neighbors
    for i in 0..1_000u64 {
        for offset in 1..=5u64 {
            let target = (i + offset) % 1_000;
            graph.add_edge(Edge {
                source: i,
                target,
                edge_type: EdgeType::Adjacency,
                weight: 1.0,
                time_range: None,
            });
        }
    }
    c.bench_function("entity_graph_neighbors_dense", |b| {
        b.iter(|| {
            // Query neighbors for entities spread across the graph
            for id in (0..1_000u64).step_by(100) {
                black_box(graph.neighbors(black_box(id)));
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Active mask benchmarks
// ---------------------------------------------------------------------------

fn bench_active_mask_set(c: &mut Criterion) {
    let mut group = c.benchmark_group("active_mask_set");
    for &size in &[100_000u32, 500_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &n| {
            b.iter(|| {
                let mut mask = ActiveMask::new(n);
                for i in 0..n {
                    mask.set(i, i % 3 == 0);
                }
                black_box(&mask);
            });
        });
    }
    group.finish();
}

fn bench_active_mask_count(c: &mut Criterion) {
    let mut mask = ActiveMask::new(500_000);
    for i in 0..500_000u32 {
        mask.set(i, i % 3 == 0);
    }
    c.bench_function("active_mask_count_500k", |b| {
        b.iter(|| {
            black_box(mask.active_count());
        });
    });
}

// ---------------------------------------------------------------------------
// Streaming / bandwidth budget benchmarks
// ---------------------------------------------------------------------------

fn bench_bandwidth_budget(c: &mut Criterion) {
    c.bench_function("bandwidth_budget_check_record_10k", |b| {
        b.iter(|| {
            let mut budget = BandwidthBudget::new(10_000_000);
            budget.reset_window(0);
            for i in 0..10_000u64 {
                let now = i; // each iteration is 1ms
                let can = budget.can_send(100, now);
                black_box(can);
                if can {
                    budget.record_sent(100, now);
                }
            }
            black_box(budget.utilization());
        });
    });
}

// ---------------------------------------------------------------------------
// Depth sort benchmarks
// ---------------------------------------------------------------------------

fn bench_depth_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("depth_sort");
    for &size in &[10_000usize, 50_000, 100_000] {
        let screen_gs = make_screen_gaussians(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &screen_gs,
            |b, sgs| {
                b.iter(|| {
                    let mut sorted = sgs.clone();
                    sorted.sort_by(|a, b| {
                        a.depth.partial_cmp(&b.depth).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    black_box(&sorted);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups and main
// ---------------------------------------------------------------------------

criterion_group!(
    gaussian_benches,
    bench_gaussian_position_at,
    bench_gaussian_project,
    bench_gaussian_is_active,
);

criterion_group!(
    tile_benches,
    bench_primitive_block_encode,
    bench_primitive_block_decode,
    bench_primitive_block_roundtrip,
    bench_checksum,
);

criterion_group!(
    draw_list_benches,
    bench_draw_list_build,
    bench_draw_list_serialize,
    bench_draw_list_finalize,
);

criterion_group!(
    coherence_benches,
    bench_coherence_evaluate,
    bench_coherence_batch,
);

criterion_group!(
    entity_benches,
    bench_entity_graph_add,
    bench_entity_graph_query_type,
    bench_entity_graph_query_time,
    bench_entity_graph_neighbors,
);

criterion_group!(
    mask_benches,
    bench_active_mask_set,
    bench_active_mask_count,
);

criterion_group!(streaming_benches, bench_bandwidth_budget,);

criterion_group!(sort_benches, bench_depth_sort,);

criterion_main!(
    gaussian_benches,
    tile_benches,
    draw_list_benches,
    coherence_benches,
    entity_benches,
    mask_benches,
    streaming_benches,
    sort_benches,
);
