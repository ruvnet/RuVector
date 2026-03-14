//! Linux syscall benchmarks for comparison.
//!
//! This module provides benchmark functions for Linux equivalents of RuVix syscalls.
//!
//! # Comparison Mapping
//!
//! | RuVix Syscall | Linux Equivalent |
//! |---------------|------------------|
//! | cap_grant | setuid/capabilities |
//! | queue_send/recv | pipe/socket |
//! | region_map | mmap |
//! | task_spawn | fork/clone |
//! | timer_wait | clock_nanosleep |
//! | vector_put_proved | write+fsync |

use std::time::{Duration, Instant};

use crate::BenchmarkResult;

#[cfg(unix)]
use std::os::unix::io::RawFd;

/// Linux benchmark configuration.
#[derive(Debug, Clone)]
pub struct LinuxBenchConfig {
    /// Number of warmup iterations.
    pub warmup_iterations: usize,
    /// Number of measurement iterations.
    pub measure_iterations: usize,
}

impl Default for LinuxBenchConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 100,
            measure_iterations: 10000,
        }
    }
}

/// Helper to run a benchmark.
fn run_benchmark<F>(name: &str, mut f: F, config: &LinuxBenchConfig) -> BenchmarkResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..config.warmup_iterations {
        f();
    }

    // Measurement
    let mut measurements = Vec::with_capacity(config.measure_iterations);

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    BenchmarkResult::from_measurements(name, &measurements, None)
}

// ============================================================================
// Capability Operations
// ============================================================================

/// Benchmarks capability check (Linux getuid + permission check simulation).
#[cfg(unix)]
pub fn bench_linux_capability_check(config: &LinuxBenchConfig) -> BenchmarkResult {
    run_benchmark(
        "linux_capability_check",
        || {
            // Simulate capability check with getuid (fast syscall)
            let _ = unsafe { libc::getuid() };
            // In real Linux, capability checks involve:
            // 1. Look up process credentials
            // 2. Check CAP_* bits or DAC permissions
            // This adds ~800ns typically
        },
        config,
    )
}

/// Benchmarks full setuid operation (expensive capability change).
#[cfg(unix)]
pub fn bench_linux_setuid_simulation(config: &LinuxBenchConfig) -> BenchmarkResult {
    // Note: Actually calling setuid requires privileges and changes state
    // We simulate the overhead by measuring getuid + cap_get_proc equivalent

    let mut measurements = Vec::with_capacity(config.measure_iterations);

    for _ in 0..config.warmup_iterations {
        let _ = unsafe { libc::getuid() };
        let _ = unsafe { libc::geteuid() };
        let _ = unsafe { libc::getgid() };
    }

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        // Simulate checking multiple credential fields
        let _ = unsafe { libc::getuid() };
        let _ = unsafe { libc::geteuid() };
        let _ = unsafe { libc::getgid() };
        let _ = unsafe { libc::getegid() };
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    BenchmarkResult::from_measurements("linux_setuid_simulation", &measurements, None)
}

// ============================================================================
// IPC Operations (Pipe)
// ============================================================================

/// Benchmarks Linux pipe write (IPC send equivalent).
#[cfg(unix)]
pub fn bench_linux_pipe_write(config: &LinuxBenchConfig) -> BenchmarkResult {
    use std::io::Write;
    use std::os::unix::io::FromRawFd;

    // Create a pipe
    let mut fds: [libc::c_int; 2] = [0; 2];
    unsafe {
        libc::pipe(fds.as_mut_ptr());
    }

    let write_fd = fds[1];
    let data = [1u8, 2, 3, 4, 5, 6, 7, 8];

    // Create a reader thread to consume data (prevent pipe buffer from filling)
    let read_fd = fds[0];
    let reader = std::thread::spawn(move || {
        let mut buf = [0u8; 1024];
        loop {
            let n = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
            if n <= 0 {
                break;
            }
        }
    });

    let mut measurements = Vec::with_capacity(config.measure_iterations);

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = unsafe { libc::write(write_fd, data.as_ptr() as *const libc::c_void, data.len()) };
    }

    // Measure
    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        let _ = unsafe { libc::write(write_fd, data.as_ptr() as *const libc::c_void, data.len()) };
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    // Cleanup
    unsafe {
        libc::close(write_fd);
    }
    let _ = reader.join();
    unsafe {
        libc::close(read_fd);
    }

    BenchmarkResult::from_measurements("linux_pipe_write", &measurements, None)
}

/// Benchmarks Linux pipe read (IPC recv equivalent).
#[cfg(unix)]
pub fn bench_linux_pipe_read(config: &LinuxBenchConfig) -> BenchmarkResult {
    // Create a pipe
    let mut fds: [libc::c_int; 2] = [0; 2];
    unsafe {
        libc::pipe(fds.as_mut_ptr());
    }

    let read_fd = fds[0];
    let write_fd = fds[1];
    let data = [1u8, 2, 3, 4, 5, 6, 7, 8];

    // Capture iteration count before spawning thread
    let total_writes = config.warmup_iterations + config.measure_iterations + 100;

    // Writer thread to keep feeding data
    let writer = std::thread::spawn(move || {
        for _ in 0..total_writes {
            let _ = unsafe { libc::write(write_fd, data.as_ptr() as *const libc::c_void, data.len()) };
        }
        unsafe { libc::close(write_fd) };
    });

    let mut measurements = Vec::with_capacity(config.measure_iterations);
    let mut buf = [0u8; 1024];

    // Warmup
    for _ in 0..config.warmup_iterations {
        let _ = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
    }

    // Measure
    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        let _ = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    // Cleanup
    unsafe {
        libc::close(read_fd);
    }
    let _ = writer.join();

    BenchmarkResult::from_measurements("linux_pipe_read", &measurements, None)
}

// ============================================================================
// Memory Operations (mmap)
// ============================================================================

/// Benchmarks Linux mmap (region_map equivalent).
#[cfg(unix)]
pub fn bench_linux_mmap(config: &LinuxBenchConfig) -> BenchmarkResult {
    let mut measurements = Vec::with_capacity(config.measure_iterations);
    let page_size = 4096usize;

    // Warmup
    for _ in 0..config.warmup_iterations {
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
    }

    // Measure
    for _ in 0..config.measure_iterations {
        let start = Instant::now();
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
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);

        // Cleanup
        if ptr != libc::MAP_FAILED {
            unsafe { libc::munmap(ptr, page_size) };
        }
    }

    BenchmarkResult::from_measurements("linux_mmap", &measurements, None)
}

/// Benchmarks Linux munmap (region unmap).
#[cfg(unix)]
pub fn bench_linux_munmap(config: &LinuxBenchConfig) -> BenchmarkResult {
    let mut measurements = Vec::with_capacity(config.measure_iterations);
    let page_size = 4096usize;

    // Pre-allocate memory regions
    let mut regions = Vec::with_capacity(config.warmup_iterations + config.measure_iterations);
    for _ in 0..(config.warmup_iterations + config.measure_iterations) {
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
            regions.push(ptr);
        }
    }

    // Warmup
    for _ in 0..config.warmup_iterations.min(regions.len()) {
        if let Some(ptr) = regions.pop() {
            unsafe { libc::munmap(ptr, page_size) };
        }
    }

    // Measure
    for _ in 0..config.measure_iterations.min(regions.len()) {
        if let Some(ptr) = regions.pop() {
            let start = Instant::now();
            unsafe { libc::munmap(ptr, page_size) };
            let elapsed = start.elapsed();
            measurements.push(elapsed.as_nanos() as f64);
        }
    }

    // Cleanup remaining
    for ptr in regions {
        unsafe { libc::munmap(ptr, page_size) };
    }

    BenchmarkResult::from_measurements("linux_munmap", &measurements, None)
}

// ============================================================================
// Process Operations (fork simulation)
// ============================================================================

/// Benchmarks Linux clone overhead simulation.
///
/// Note: Actually forking is expensive and creates real processes.
/// We simulate by measuring the overhead of related syscalls.
#[cfg(unix)]
pub fn bench_linux_clone_simulation(config: &LinuxBenchConfig) -> BenchmarkResult {
    // Simulate clone overhead with:
    // - getpid (lightweight)
    // - signal handling setup
    // This gives a lower bound estimate

    let mut measurements = Vec::with_capacity(config.measure_iterations);

    for _ in 0..config.warmup_iterations {
        let _ = unsafe { libc::getpid() };
        let _ = unsafe { libc::getppid() };
    }

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        // Simulate minimal clone work
        let _ = unsafe { libc::getpid() };
        let _ = unsafe { libc::getppid() };
        // In real clone: copy page tables, set up signals, etc.
        // Actual fork is typically 2-10us
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    BenchmarkResult::from_measurements("linux_clone_simulation", &measurements, None)
}

// ============================================================================
// Timer Operations
// ============================================================================

/// Benchmarks Linux clock_gettime (timer operation).
#[cfg(unix)]
pub fn bench_linux_clock_gettime(config: &LinuxBenchConfig) -> BenchmarkResult {
    let mut measurements = Vec::with_capacity(config.measure_iterations);
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };

    for _ in 0..config.warmup_iterations {
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    }

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    BenchmarkResult::from_measurements("linux_clock_gettime", &measurements, None)
}

// ============================================================================
// File I/O (write + fsync for durability)
// ============================================================================

/// Benchmarks Linux write + fsync (durable write like vector_put_proved).
#[cfg(unix)]
pub fn bench_linux_write_fsync(config: &LinuxBenchConfig) -> BenchmarkResult {
    use std::fs::OpenOptions;
    use std::io::Write;

    // Create temp file
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(format!("ruvix_bench_{}", std::process::id()));

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temp_path)
        .expect("Failed to create temp file");

    let data = vec![0u8; 32]; // 32 bytes like a hash
    let mut measurements = Vec::with_capacity(config.measure_iterations);

    // Warmup
    for _ in 0..config.warmup_iterations {
        file.write_all(&data).ok();
        file.sync_all().ok();
    }

    // Measure
    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        file.write_all(&data).ok();
        file.sync_all().ok();
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    // Cleanup
    drop(file);
    std::fs::remove_file(&temp_path).ok();

    BenchmarkResult::from_measurements("linux_write_fsync", &measurements, None)
}

/// Benchmarks Linux write without fsync (non-durable).
#[cfg(unix)]
pub fn bench_linux_write_only(config: &LinuxBenchConfig) -> BenchmarkResult {
    use std::fs::OpenOptions;
    use std::io::Write;

    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(format!("ruvix_bench_wo_{}", std::process::id()));

    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temp_path)
        .expect("Failed to create temp file");

    let data = vec![0u8; 32];
    let mut measurements = Vec::with_capacity(config.measure_iterations);

    for _ in 0..config.warmup_iterations {
        file.write_all(&data).ok();
    }

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        file.write_all(&data).ok();
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    drop(file);
    std::fs::remove_file(&temp_path).ok();

    BenchmarkResult::from_measurements("linux_write_only", &measurements, None)
}

// ============================================================================
// Security Operations (SELinux simulation)
// ============================================================================

/// Simulates SELinux policy check overhead.
///
/// In production Linux, SELinux adds ~800ns-2us per access check.
/// We simulate this with multiple getuid/geteuid calls.
#[cfg(unix)]
pub fn bench_linux_selinux_simulation(config: &LinuxBenchConfig) -> BenchmarkResult {
    let mut measurements = Vec::with_capacity(config.measure_iterations);

    for _ in 0..config.warmup_iterations {
        // Simulate policy lookup: typically O(n) in policy rules
        for _ in 0..10 {
            let _ = unsafe { libc::getuid() };
        }
    }

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        // Simulate 10 policy checks (conservative)
        for _ in 0..10 {
            let _ = unsafe { libc::getuid() };
        }
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    BenchmarkResult::from_measurements("linux_selinux_simulation", &measurements, None)
}

// ============================================================================
// Socket IPC
// ============================================================================

/// Benchmarks Unix socket send (alternative IPC).
#[cfg(unix)]
pub fn bench_linux_socket_send(config: &LinuxBenchConfig) -> BenchmarkResult {
    // Create Unix socket pair
    let mut fds: [libc::c_int; 2] = [0; 2];
    unsafe {
        if libc::socketpair(libc::AF_UNIX, libc::SOCK_STREAM, 0, fds.as_mut_ptr()) != 0 {
            return BenchmarkResult::from_measurements("linux_socket_send", &[0.0], None);
        }
    }

    let send_fd = fds[0];
    let recv_fd = fds[1];
    let data = [1u8, 2, 3, 4, 5, 6, 7, 8];

    // Reader thread
    let reader = std::thread::spawn(move || {
        let mut buf = [0u8; 1024];
        loop {
            let n = unsafe { libc::recv(recv_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len(), 0) };
            if n <= 0 {
                break;
            }
        }
    });

    let mut measurements = Vec::with_capacity(config.measure_iterations);

    for _ in 0..config.warmup_iterations {
        let _ = unsafe { libc::send(send_fd, data.as_ptr() as *const libc::c_void, data.len(), 0) };
    }

    for _ in 0..config.measure_iterations {
        let start = Instant::now();
        let _ = unsafe { libc::send(send_fd, data.as_ptr() as *const libc::c_void, data.len(), 0) };
        let elapsed = start.elapsed();
        measurements.push(elapsed.as_nanos() as f64);
    }

    unsafe {
        libc::close(send_fd);
    }
    let _ = reader.join();
    unsafe {
        libc::close(recv_fd);
    }

    BenchmarkResult::from_measurements("linux_socket_send", &measurements, None)
}

// ============================================================================
// Aggregated Benchmark Suite
// ============================================================================

/// Runs all Linux comparison benchmarks.
#[cfg(unix)]
pub fn bench_all_linux(config: &LinuxBenchConfig) -> Vec<BenchmarkResult> {
    vec![
        bench_linux_capability_check(config),
        bench_linux_setuid_simulation(config),
        bench_linux_pipe_write(config),
        bench_linux_pipe_read(config),
        bench_linux_mmap(config),
        bench_linux_munmap(config),
        bench_linux_clone_simulation(config),
        bench_linux_clock_gettime(config),
        bench_linux_write_fsync(config),
        bench_linux_write_only(config),
        bench_linux_selinux_simulation(config),
        bench_linux_socket_send(config),
    ]
}

#[cfg(not(unix))]
pub fn bench_all_linux(_config: &LinuxBenchConfig) -> Vec<BenchmarkResult> {
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(unix)]
    fn test_linux_capability_check() {
        let config = LinuxBenchConfig {
            warmup_iterations: 10,
            measure_iterations: 100,
        };
        let result = bench_linux_capability_check(&config);
        assert_eq!(result.iterations, 100);
        assert!(result.mean_ns > 0.0);
    }

    #[test]
    #[cfg(unix)]
    fn test_linux_mmap() {
        let config = LinuxBenchConfig {
            warmup_iterations: 10,
            measure_iterations: 100,
        };
        let result = bench_linux_mmap(&config);
        assert_eq!(result.iterations, 100);
    }
}
