//! Criterion benchmarks comparing RuVix syscalls against Linux equivalents.
//!
//! Run with: cargo bench --bench linux_comparison

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ruvix_nucleus::{
    Kernel, KernelConfig, Syscall, VectorStoreConfig, TaskPriority, ProofTier,
    VectorKey, RegionPolicy, MsgPriority, TimerSpec, SensorDescriptor, QueueHandle,
    GraphMutation, CapHandle, CapRights, RvfMountHandle, RvfComponentId,
};
use ruvix_types::{TaskHandle, ObjectType};
use std::time::Duration;

fn setup_kernel() -> Kernel {
    let mut kernel = Kernel::new(KernelConfig::default());
    kernel.boot(0, [0u8; 32]).expect("Boot failed");
    kernel.set_current_time(1_000_000);
    kernel
}

// ============================================================================
// RuVix Benchmarks for Comparison
// ============================================================================

fn bench_ruvix_cap_grant(c: &mut Criterion) {
    let mut kernel = setup_kernel();
    let root_task = TaskHandle::new(1, 0);
    let cap = kernel.create_root_capability(0, ObjectType::RvfMount, root_task).unwrap();

    c.bench_function("ruvix/cap_grant", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::CapGrant {
                target: TaskHandle::new(2, 0),
                cap,
                rights: CapRights::READ,
            }))
        })
    });
}

fn bench_ruvix_queue_send(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    c.bench_function("ruvix/queue_send", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::QueueSend {
                queue: QueueHandle::new(1, 0),
                msg: vec![1, 2, 3, 4, 5, 6, 7, 8],
                priority: MsgPriority::Normal,
            }))
        })
    });
}

fn bench_ruvix_region_map(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    c.bench_function("ruvix/region_map", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::RegionMap {
                size: 4096,
                policy: RegionPolicy::Immutable,
                cap: CapHandle::null(),
            }))
        })
    });
}

fn bench_ruvix_timer_wait(c: &mut Criterion) {
    let mut kernel = setup_kernel();

    c.bench_function("ruvix/timer_wait", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::TimerWait {
                deadline: TimerSpec::from_millis(0),
            }))
        })
    });
}

// ============================================================================
// Linux Benchmarks for Comparison
// ============================================================================

#[cfg(unix)]
fn bench_linux_getuid(c: &mut Criterion) {
    c.bench_function("linux/getuid", |b| {
        b.iter(|| {
            black_box(unsafe { libc::getuid() })
        })
    });
}

#[cfg(unix)]
fn bench_linux_capability_simulation(c: &mut Criterion) {
    // Simulate capability check overhead with multiple syscalls
    c.bench_function("linux/capability_check", |b| {
        b.iter(|| {
            black_box(unsafe {
                libc::getuid();
                libc::geteuid();
                libc::getgid();
                libc::getegid()
            })
        })
    });
}

#[cfg(unix)]
fn bench_linux_pipe_write(c: &mut Criterion) {
    // Create pipe
    let mut fds: [libc::c_int; 2] = [0; 2];
    unsafe { libc::pipe(fds.as_mut_ptr()); }

    let write_fd = fds[1];
    let read_fd = fds[0];
    let data = [1u8, 2, 3, 4, 5, 6, 7, 8];

    // Spawn reader thread to prevent buffer fill
    let reader = std::thread::spawn(move || {
        let mut buf = [0u8; 1024];
        loop {
            let n = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
            if n <= 0 { break; }
        }
    });

    c.bench_function("linux/pipe_write", |b| {
        b.iter(|| {
            black_box(unsafe {
                libc::write(write_fd, data.as_ptr() as *const libc::c_void, data.len())
            })
        })
    });

    unsafe { libc::close(write_fd); }
    let _ = reader.join();
}

#[cfg(unix)]
fn bench_linux_mmap(c: &mut Criterion) {
    let page_size = 4096usize;

    c.bench_function("linux/mmap", |b| {
        b.iter(|| {
            let ptr = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    page_size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if ptr != libc::MAP_FAILED {
                unsafe { libc::munmap(ptr, page_size) };
            }
            black_box(ptr)
        })
    });
}

#[cfg(unix)]
fn bench_linux_clock_gettime(c: &mut Criterion) {
    c.bench_function("linux/clock_gettime", |b| {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        b.iter(|| {
            black_box(unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) })
        })
    });
}

#[cfg(unix)]
fn bench_linux_selinux_simulation(c: &mut Criterion) {
    // Simulate SELinux policy check overhead (typically 10+ lookups)
    c.bench_function("linux/selinux_simulation", |b| {
        b.iter(|| {
            for _ in 0..10 {
                black_box(unsafe { libc::getuid() });
            }
        })
    });
}

// ============================================================================
// Comparison Groups
// ============================================================================

fn compare_capability(c: &mut Criterion) {
    let mut group = c.benchmark_group("capability_comparison");

    // RuVix cap_grant
    let mut kernel = setup_kernel();
    let root_task = TaskHandle::new(1, 0);
    let cap = kernel.create_root_capability(0, ObjectType::RvfMount, root_task).unwrap();

    group.bench_function("ruvix", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::CapGrant {
                target: TaskHandle::new(2, 0),
                cap,
                rights: CapRights::READ,
            }))
        })
    });

    // Linux capability check
    #[cfg(unix)]
    group.bench_function("linux", |b| {
        b.iter(|| {
            black_box(unsafe {
                libc::getuid();
                libc::geteuid();
                libc::getgid();
                libc::getegid()
            })
        })
    });

    group.finish();
}

fn compare_ipc(c: &mut Criterion) {
    let mut group = c.benchmark_group("ipc_comparison");

    // RuVix queue_send
    let mut kernel = setup_kernel();

    group.bench_function("ruvix", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::QueueSend {
                queue: QueueHandle::new(1, 0),
                msg: vec![1, 2, 3, 4, 5, 6, 7, 8],
                priority: MsgPriority::Normal,
            }))
        })
    });

    // Linux pipe write
    #[cfg(unix)]
    {
        let mut fds: [libc::c_int; 2] = [0; 2];
        unsafe { libc::pipe(fds.as_mut_ptr()); }

        let write_fd = fds[1];
        let read_fd = fds[0];
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];

        let reader = std::thread::spawn(move || {
            let mut buf = [0u8; 1024];
            loop {
                let n = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
                if n <= 0 { break; }
            }
        });

        group.bench_function("linux", |b| {
            b.iter(|| {
                black_box(unsafe {
                    libc::write(write_fd, data.as_ptr() as *const libc::c_void, data.len())
                })
            })
        });

        unsafe { libc::close(write_fd); }
        let _ = reader.join();
    }

    group.finish();
}

fn compare_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_comparison");

    // RuVix region_map
    let mut kernel = setup_kernel();

    group.bench_function("ruvix", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::RegionMap {
                size: 4096,
                policy: RegionPolicy::Immutable,
                cap: CapHandle::null(),
            }))
        })
    });

    // Linux mmap
    #[cfg(unix)]
    {
        let page_size = 4096usize;
        group.bench_function("linux", |b| {
            b.iter(|| {
                let ptr = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        page_size,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                        -1,
                        0,
                    )
                };
                if ptr != libc::MAP_FAILED {
                    unsafe { libc::munmap(ptr, page_size) };
                }
                black_box(ptr)
            })
        });
    }

    group.finish();
}

fn compare_timer(c: &mut Criterion) {
    let mut group = c.benchmark_group("timer_comparison");

    // RuVix timer_wait
    let mut kernel = setup_kernel();

    group.bench_function("ruvix", |b| {
        b.iter(|| {
            kernel.dispatch(black_box(Syscall::TimerWait {
                deadline: TimerSpec::from_millis(0),
            }))
        })
    });

    // Linux clock_gettime
    #[cfg(unix)]
    {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        group.bench_function("linux", |b| {
            b.iter(|| {
                black_box(unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) })
            })
        });
    }

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

#[cfg(unix)]
criterion_group!(
    ruvix_benches,
    bench_ruvix_cap_grant,
    bench_ruvix_queue_send,
    bench_ruvix_region_map,
    bench_ruvix_timer_wait,
);

#[cfg(unix)]
criterion_group!(
    linux_benches,
    bench_linux_getuid,
    bench_linux_capability_simulation,
    bench_linux_pipe_write,
    bench_linux_mmap,
    bench_linux_clock_gettime,
    bench_linux_selinux_simulation,
);

criterion_group!(
    comparison_benches,
    compare_capability,
    compare_ipc,
    compare_memory,
    compare_timer,
);

#[cfg(unix)]
criterion_main!(ruvix_benches, linux_benches, comparison_benches);

#[cfg(not(unix))]
criterion_group!(
    ruvix_only,
    bench_ruvix_cap_grant,
    bench_ruvix_queue_send,
    bench_ruvix_region_map,
    bench_ruvix_timer_wait,
);

#[cfg(not(unix))]
criterion_main!(ruvix_only, comparison_benches);
