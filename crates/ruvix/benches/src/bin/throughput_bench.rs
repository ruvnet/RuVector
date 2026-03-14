//! Throughput benchmark for RuVix operations.
//!
//! Measures operations per second for IPC, vector, and graph operations.
//!
//! Usage: cargo run --bin throughput-bench -- [OPTIONS]

use clap::Parser;
use std::time::{Duration, Instant};
use tabled::{Table, Tabled};

use ruvix_nucleus::{
    Kernel, KernelConfig, Syscall, VectorStoreConfig, ProofTier,
    VectorKey, MsgPriority, QueueHandle, GraphMutation,
};

#[derive(Parser, Debug)]
#[command(name = "throughput-bench")]
#[command(about = "Measure RuVix operation throughput")]
struct Args {
    /// Duration of each benchmark in seconds.
    #[arg(short, long, default_value = "5")]
    duration: u64,

    /// Target operations per second to verify.
    #[arg(short, long, default_value = "100000")]
    target: u64,

    /// Include Linux comparison.
    #[arg(long)]
    compare_linux: bool,

    /// Run quick benchmark.
    #[arg(long)]
    quick: bool,
}

#[derive(Tabled)]
struct ThroughputRow {
    #[tabled(rename = "Operation")]
    operation: String,
    #[tabled(rename = "Ops/sec")]
    ops_per_sec: String,
    #[tabled(rename = "Latency (mean)")]
    latency: String,
    #[tabled(rename = "Target")]
    target: String,
    #[tabled(rename = "Status")]
    status: String,
}

fn format_rate(rate: f64) -> String {
    if rate >= 1_000_000.0 {
        format!("{:.2}M", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.2}K", rate / 1_000.0)
    } else {
        format!("{:.0}", rate)
    }
}

fn format_duration(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.2}ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.2}us", ns / 1_000.0)
    } else {
        format!("{:.0}ns", ns)
    }
}

fn setup_kernel() -> Kernel {
    let mut kernel = Kernel::new(KernelConfig::default());
    kernel.boot(0, [0u8; 32]).expect("Boot failed");
    kernel.set_current_time(1_000_000);
    kernel
}

fn run_throughput_bench<F>(name: &str, duration_secs: u64, mut f: F) -> (u64, Duration, f64)
where
    F: FnMut() -> (),
{
    let target_duration = Duration::from_secs(duration_secs);
    let start = Instant::now();
    let mut ops = 0u64;

    while start.elapsed() < target_duration {
        f();
        ops += 1;
    }

    let elapsed = start.elapsed();
    let ops_per_sec = ops as f64 / elapsed.as_secs_f64();
    let latency_ns = elapsed.as_nanos() as f64 / ops as f64;

    println!("  {}: {} ops in {:.2}s = {} ops/sec ({}/op)",
        name, ops, elapsed.as_secs_f64(), format_rate(ops_per_sec), format_duration(latency_ns));

    (ops, elapsed, ops_per_sec)
}

fn main() {
    let args = Args::parse();

    let duration = if args.quick { 1 } else { args.duration };

    println!("RuVix Throughput Benchmark");
    println!("==========================");
    println!();
    println!("Configuration:");
    println!("  Duration: {} seconds per test", duration);
    println!("  Target: {} ops/sec", format_rate(args.target as f64));
    println!();

    let mut rows = Vec::new();

    // IPC throughput
    println!("IPC Throughput:");
    {
        let mut kernel = setup_kernel();
        let msg = vec![1u8, 2, 3, 4, 5, 6, 7, 8];

        let (_, _, ops_per_sec) = run_throughput_bench("queue_send (8B)", duration, || {
            let _ = kernel.dispatch(Syscall::QueueSend {
                queue: QueueHandle::new(1, 0),
                msg: msg.clone(),
                priority: MsgPriority::Normal,
            });
        });

        let meets_target = ops_per_sec >= args.target as f64;
        rows.push(ThroughputRow {
            operation: "IPC Send (8B)".to_string(),
            ops_per_sec: format!("{}/s", format_rate(ops_per_sec)),
            latency: format_duration(1_000_000_000.0 / ops_per_sec),
            target: format!("{}/s", format_rate(args.target as f64)),
            status: if meets_target { "PASS".to_string() } else { "FAIL".to_string() },
        });
    }
    println!();

    // Vector throughput
    println!("Vector Store Throughput:");
    {
        let mut kernel = setup_kernel();
        let config = VectorStoreConfig::new(4, 100000);
        let store = kernel.create_vector_store(config).unwrap();
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut nonce = 0u64;

        let (_, _, ops_per_sec) = run_throughput_bench("vector_put (4D)", duration, || {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

            let _ = kernel.dispatch(Syscall::VectorPutProved {
                store,
                key: VectorKey::new((nonce % 100000) as u64),
                data: data.clone(),
                proof,
            });
        });

        let meets_target = ops_per_sec >= args.target as f64;
        rows.push(ThroughputRow {
            operation: "Vector Put (4D)".to_string(),
            ops_per_sec: format!("{}/s", format_rate(ops_per_sec)),
            latency: format_duration(1_000_000_000.0 / ops_per_sec),
            target: format!("{}/s", format_rate(args.target as f64)),
            status: if meets_target { "PASS".to_string() } else { "FAIL".to_string() },
        });
    }
    {
        let mut kernel = setup_kernel();
        let config = VectorStoreConfig::new(768, 100000);
        let store = kernel.create_vector_store(config).unwrap();
        let data: Vec<f32> = (0..768).map(|i| i as f32 * 0.001).collect();
        let mut nonce = 0u64;

        let (_, _, ops_per_sec) = run_throughput_bench("vector_put (768D)", duration, || {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);

            let _ = kernel.dispatch(Syscall::VectorPutProved {
                store,
                key: VectorKey::new((nonce % 100000) as u64),
                data: data.clone(),
                proof,
            });
        });

        rows.push(ThroughputRow {
            operation: "Vector Put (768D)".to_string(),
            ops_per_sec: format!("{}/s", format_rate(ops_per_sec)),
            latency: format_duration(1_000_000_000.0 / ops_per_sec),
            target: format!("{}/s", format_rate(args.target as f64 / 10.0)),
            status: if ops_per_sec >= args.target as f64 / 10.0 { "PASS".to_string() } else { "FAIL".to_string() },
        });
    }
    println!();

    // Graph throughput
    println!("Graph Store Throughput:");
    {
        let mut kernel = setup_kernel();
        let graph = kernel.create_graph_store().unwrap();
        let mut nonce = 0u64;

        let (_, _, ops_per_sec) = run_throughput_bench("graph_apply", duration, || {
            nonce += 1;
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Standard, nonce);

            let _ = kernel.dispatch(Syscall::GraphApplyProved {
                graph,
                mutation: GraphMutation::add_node(nonce),
                proof,
            });
        });

        rows.push(ThroughputRow {
            operation: "Graph Apply".to_string(),
            ops_per_sec: format!("{}/s", format_rate(ops_per_sec)),
            latency: format_duration(1_000_000_000.0 / ops_per_sec),
            target: format!("{}/s", format_rate(args.target as f64)),
            status: if ops_per_sec >= args.target as f64 { "PASS".to_string() } else { "FAIL".to_string() },
        });
    }
    println!();

    // Pipeline throughput
    println!("Full Pipeline Throughput:");
    {
        let mut kernel = setup_kernel();
        let vector_config = VectorStoreConfig::new(4, 100000);
        let vector_store = kernel.create_vector_store(vector_config).unwrap();
        let graph = kernel.create_graph_store().unwrap();
        let mut nonce = 0u64;

        let (_, _, ops_per_sec) = run_throughput_bench("queue+vector+graph", duration, || {
            nonce += 1;

            // IPC
            let _ = kernel.dispatch(Syscall::QueueSend {
                queue: QueueHandle::new(1, 0),
                msg: vec![1, 2, 3, 4],
                priority: MsgPriority::Normal,
            });

            // Vector
            let mutation_hash = [nonce as u8; 32];
            let proof = kernel.create_proof(mutation_hash, ProofTier::Reflex, nonce);
            let _ = kernel.dispatch(Syscall::VectorPutProved {
                store: vector_store,
                key: VectorKey::new((nonce % 100000) as u64),
                data: vec![1.0, 2.0, 3.0, 4.0],
                proof,
            });

            // Graph
            let graph_hash = [(nonce + 1) as u8; 32];
            let graph_proof = kernel.create_proof(graph_hash, ProofTier::Standard, nonce + 1);
            let _ = kernel.dispatch(Syscall::GraphApplyProved {
                graph,
                mutation: GraphMutation::add_node(nonce),
                proof: graph_proof,
            });
        });

        rows.push(ThroughputRow {
            operation: "Full Pipeline".to_string(),
            ops_per_sec: format!("{}/s", format_rate(ops_per_sec)),
            latency: format_duration(1_000_000_000.0 / ops_per_sec),
            target: format!("{}/s", format_rate(args.target as f64 / 3.0)),
            status: if ops_per_sec >= args.target as f64 / 3.0 { "PASS".to_string() } else { "FAIL".to_string() },
        });
    }
    println!();

    // Linux comparison
    #[cfg(unix)]
    if args.compare_linux {
        println!("Linux Comparison:");

        // Pipe write
        {
            let mut fds: [libc::c_int; 2] = [0; 2];
            unsafe { libc::pipe(fds.as_mut_ptr()); }

            let write_fd = fds[1];
            let read_fd = fds[0];
            let msg = [1u8, 2, 3, 4, 5, 6, 7, 8];

            let reader = std::thread::spawn(move || {
                let mut buf = [0u8; 8192];
                loop {
                    let n = unsafe { libc::read(read_fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
                    if n <= 0 { break; }
                }
            });

            let (_, _, ops_per_sec) = run_throughput_bench("linux_pipe (8B)", duration, || {
                unsafe { libc::write(write_fd, msg.as_ptr() as *const libc::c_void, msg.len()) };
            });

            unsafe { libc::close(write_fd); }
            let _ = reader.join();

            rows.push(ThroughputRow {
                operation: "Linux Pipe (8B)".to_string(),
                ops_per_sec: format!("{}/s", format_rate(ops_per_sec)),
                latency: format_duration(1_000_000_000.0 / ops_per_sec),
                target: "-".to_string(),
                status: "-".to_string(),
            });
        }
        println!();
    }

    // Summary table
    println!("Results Summary:");
    println!("{}", Table::new(&rows));
    println!();

    // Calculate overall pass rate
    let passing = rows.iter().filter(|r| r.status == "PASS").count();
    let total = rows.iter().filter(|r| r.status != "-").count();
    println!("Overall: {} / {} benchmarks meet target ({:.0}%)",
        passing, total, 100.0 * passing as f64 / total as f64);
}
