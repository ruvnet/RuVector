//! Benchmarks for the ruvix-sched scheduler.
//!
//! Run with: `cargo bench -p ruvix-sched`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ruvix_cap::CapRights;
use ruvix_sched::{
    compute_priority, Instant, NoveltyTracker, PriorityConfig, Scheduler, SchedulerConfig,
    TaskControlBlock,
};
use ruvix_types::{TaskHandle, TaskPriority};

fn make_task(id: u32, partition: u32) -> TaskControlBlock {
    TaskControlBlock::new(
        TaskHandle::new(id, 0),
        CapRights::READ,
        TaskPriority::Normal,
        partition,
    )
}

fn make_task_with_deadline(id: u32, partition: u32, deadline_us: u64) -> TaskControlBlock {
    make_task(id, partition).with_deadline(Instant::from_micros(deadline_us))
}

/// Benchmark task creation
fn bench_task_creation(c: &mut Criterion) {
    c.bench_function("task_creation", |b| {
        b.iter(|| {
            black_box(TaskControlBlock::new(
                TaskHandle::new(1, 0),
                CapRights::READ,
                TaskPriority::Normal,
                0,
            ))
        })
    });
}

/// Benchmark priority computation
fn bench_priority_computation(c: &mut Criterion) {
    let config = PriorityConfig::default();
    let now = Instant::from_micros(1_000_000);

    let task_no_deadline = make_task(1, 0);
    let task_with_deadline = make_task_with_deadline(2, 0, 2_000_000);
    let task_with_novelty = make_task(3, 0).with_novelty(0.8);
    let task_with_risk = make_task(4, 0).with_coherence_delta(-0.5);

    let mut group = c.benchmark_group("priority_computation");

    group.bench_function("no_deadline", |b| {
        b.iter(|| black_box(compute_priority(&task_no_deadline, now, &config)))
    });

    group.bench_function("with_deadline", |b| {
        b.iter(|| black_box(compute_priority(&task_with_deadline, now, &config)))
    });

    group.bench_function("with_novelty", |b| {
        b.iter(|| black_box(compute_priority(&task_with_novelty, now, &config)))
    });

    group.bench_function("with_risk", |b| {
        b.iter(|| black_box(compute_priority(&task_with_risk, now, &config)))
    });

    group.finish();
}

/// Benchmark scheduler operations
fn bench_scheduler_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduler_operations");

    // Benchmark add_task
    group.bench_function("add_task", |b| {
        b.iter_batched(
            || Scheduler::<64, 8>::with_default_config(),
            |mut scheduler| {
                black_box(scheduler.add_task(make_task(1, 0)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Benchmark select_next_task with varying task counts
    for task_count in [1, 8, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::new("select_next_task", task_count),
            task_count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let mut scheduler = Scheduler::<64, 8>::with_default_config();
                        for i in 0..count {
                            scheduler.add_task(make_task(i, i % 4)).unwrap();
                        }
                        scheduler
                    },
                    |mut scheduler| {
                        black_box(scheduler.select_next_task_at(Instant::from_micros(0)));
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    // Benchmark yield_task
    group.bench_function("yield_task", |b| {
        b.iter_batched(
            || {
                let mut scheduler = Scheduler::<64, 8>::with_default_config();
                scheduler.add_task(make_task(1, 0)).unwrap();
                scheduler.select_next_task_at(Instant::from_micros(0));
                scheduler
            },
            |mut scheduler| {
                black_box(scheduler.yield_task().unwrap());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    // Benchmark block/unblock cycle
    group.bench_function("block_unblock_cycle", |b| {
        b.iter_batched(
            || {
                let mut scheduler = Scheduler::<64, 8>::with_default_config();
                scheduler.add_task(make_task(1, 0)).unwrap();
                scheduler.select_next_task_at(Instant::from_micros(0));
                scheduler
            },
            |mut scheduler| {
                scheduler.block_task().unwrap();
                black_box(scheduler.unblock_task(TaskHandle::new(1, 0)).unwrap());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

/// Benchmark novelty tracking
fn bench_novelty_tracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("novelty_tracking");

    // Small vectors (16 dimensions)
    group.bench_function("compute_novelty_16d", |b| {
        let mut tracker: NoveltyTracker<16> =
            NoveltyTracker::new(ruvix_sched::NoveltyConfig::default().with_dimensions(16));
        let input: [f32; 16] = [1.0; 16];
        tracker.update(&input);

        b.iter(|| {
            let different = [2.0f32; 16];
            black_box(tracker.compute_novelty(&different))
        })
    });

    // Medium vectors (128 dimensions)
    group.bench_function("compute_novelty_128d", |b| {
        let mut tracker: NoveltyTracker<128> =
            NoveltyTracker::new(ruvix_sched::NoveltyConfig::default().with_dimensions(128));
        let input: [f32; 128] = [1.0; 128];
        tracker.update(&input);

        b.iter(|| {
            let different = [2.0f32; 128];
            black_box(tracker.compute_novelty(&different))
        })
    });

    // Process (compute + update)
    group.bench_function("process_16d", |b| {
        let mut tracker: NoveltyTracker<16> =
            NoveltyTracker::new(ruvix_sched::NoveltyConfig::default().with_dimensions(16));

        b.iter(|| {
            let input = [1.5f32; 16];
            black_box(tracker.process(&input))
        })
    });

    group.finish();
}

/// Benchmark partition scheduling
fn bench_partition_scheduling(c: &mut Criterion) {
    use ruvix_sched::PartitionManager;

    let mut group = c.benchmark_group("partition_scheduling");

    // Select partition with varying partition counts
    for partition_count in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("select_partition", partition_count),
            partition_count,
            |b, &count| {
                b.iter_batched(
                    || {
                        let mut manager: PartitionManager<8> = PartitionManager::new(10_000);
                        for i in 0..count {
                            manager.add_partition(i, 5000).unwrap();
                            if let Some(p) = manager.get_partition_mut(i) {
                                p.task_ready();
                            }
                        }
                        manager
                    },
                    |mut manager| {
                        black_box(manager.select_next_partition(Instant::from_micros(0)));
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }

    group.finish();
}

/// Benchmark scheduling throughput (tasks per second)
fn bench_scheduling_throughput(c: &mut Criterion) {
    c.bench_function("scheduling_throughput", |b| {
        b.iter_batched(
            || {
                let mut scheduler = Scheduler::<64, 8>::with_default_config();
                for i in 0..32 {
                    scheduler.add_task(make_task(i, i % 4)).unwrap();
                }
                scheduler
            },
            |mut scheduler| {
                let now = Instant::from_micros(0);
                // Simulate 100 scheduling cycles
                for _ in 0..100 {
                    if let Some(_) = scheduler.select_next_task_at(now) {
                        scheduler.yield_task().unwrap();
                    }
                }
                black_box(scheduler.stats().tasks_scheduled)
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    bench_task_creation,
    bench_priority_computation,
    bench_scheduler_operations,
    bench_novelty_tracking,
    bench_partition_scheduling,
    bench_scheduling_throughput,
);

criterion_main!(benches);
