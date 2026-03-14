# ruvix-sched

Coherence-aware scheduler for the RuVix Cognition Kernel (ADR-087).

## Overview

`ruvix-sched` implements a real-time scheduler that combines three scheduling signals to determine task priority:

1. **Deadline Pressure (EDF)**: Tasks with tighter deadlines receive higher urgency
2. **Novelty Signal**: Tasks processing genuinely new information get a priority boost
3. **Structural Risk**: Tasks that might destabilize system coherence receive a penalty

The priority formula from ADR-087 Section 5.1:

```
score = deadline_urgency + novelty_boost - risk_penalty
```

## Features

- **No-std compatible**: Works in embedded and kernel environments
- **Partition scheduling**: Guaranteed time slices per RVF mount
- **Capability-based**: Integrates with `ruvix-cap` for access control
- **Zero allocation**: Uses fixed-size arrays for predictable memory usage
- **Novelty tracking**: EMA-based centroid tracking for input novelty detection

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
ruvix-sched = { version = "0.1.0", path = "../ruvix-sched" }
```

## Usage

```rust
use ruvix_sched::{Scheduler, SchedulerConfig, TaskControlBlock, Instant};
use ruvix_types::{TaskHandle, TaskPriority};
use ruvix_cap::CapRights;

// Create scheduler
let config = SchedulerConfig::default();
let mut scheduler: Scheduler<64, 8> = Scheduler::new(config);

// Add a task
let task = TaskControlBlock::new(
    TaskHandle::new(1, 0),
    CapRights::READ,
    TaskPriority::Normal,
    0,  // partition_id
);
scheduler.add_task(task).unwrap();

// Schedule
let now = Instant::from_micros(0);
if let Some(handle) = scheduler.select_next_task_at(now) {
    println!("Selected task: {:?}", handle);
    
    // Task runs...
    
    // Yield when done
    scheduler.yield_task().unwrap();
}
```

### Setting Deadlines

```rust
let task = TaskControlBlock::new(/* ... */)
    .with_deadline(Instant::from_micros(1_000_000)); // 1 second deadline
```

### Novelty Boosting

```rust
// High novelty = higher priority
let task = TaskControlBlock::new(/* ... */)
    .with_novelty(0.8);  // 80% novelty
```

### Risk Penalty

```rust
// Negative coherence delta = risk penalty
let task = TaskControlBlock::new(/* ... */)
    .with_coherence_delta(-0.5);  // Will receive priority penalty
```

## Scheduling Guarantees

From ADR-087 Section 5.2:

1. **No priority inversion**: High-priority tasks always preempt lower-priority tasks at queue boundaries
2. **Bounded preemption**: Preemption only occurs at queue boundaries, not mid-operation
3. **Partition scheduling**: Each partition receives its configured time slice per epoch

## Architecture

```
ruvix-sched/
├── src/
│   ├── lib.rs        # Module exports and time types
│   ├── error.rs      # Error types
│   ├── task.rs       # TaskControlBlock and TaskState
│   ├── priority.rs   # Priority computation (ADR-087 formula)
│   ├── novelty.rs    # NoveltyTracker for input novelty
│   ├── partition.rs  # Partition scheduling
│   └── scheduler.rs  # Main Scheduler implementation
└── benches/
    └── scheduler_bench.rs  # Performance benchmarks
```

## Feature Flags

- `std`: Enable standard library features (default: off)
- `alloc`: Enable allocation support (default: off)
- `coherence`: Enable integration with `ruvector-coherence` for spectral scoring
- `metrics`: Enable metrics collection (requires `alloc`)
- `audit-log`: Enable audit logging (requires `alloc`)

## Benchmarks

Run benchmarks with:

```bash
cargo bench -p ruvix-sched
```

## Testing

```bash
cargo test -p ruvix-sched
```

## License

MIT OR Apache-2.0
