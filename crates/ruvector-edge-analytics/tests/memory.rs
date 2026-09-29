//! Memory model check: the estimate must bound the allocator peak of
//! rebuild + query (measured with a counting global allocator; this file
//! is its own test binary so the allocator affects nothing else).
mod common;
use ruvector_edge_analytics::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Mutex;

struct Counting;
static CUR: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static SERIAL: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = System.alloc(l);
        if !p.is_null() {
            let now = CUR.fetch_add(l.size(), SeqCst) + l.size();
            PEAK.fetch_max(now, SeqCst);
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l);
        CUR.fetch_sub(l.size(), SeqCst);
    }
}

#[global_allocator]
static A: Counting = Counting;

/// Peak bytes above the baseline while `f` runs.
fn peak_of<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let base = CUR.load(SeqCst);
    PEAK.store(base, SeqCst);
    let out = f();
    (out, PEAK.load(SeqCst) - base)
}

fn unlimited(limits: GraphLimits) -> Profile {
    Profile {
        limits,
        budget: Budget {
            max_work: u64::MAX,
            max_memory_bytes: u64::MAX,
        },
    }
}

fn check(name: &str, edges: Vec<(u64, u64, f64)>, mode: QueryMode) {
    let _serial = SERIAL.lock().unwrap();
    // The graph the caller holds (decoded edge list) counts too.
    let (g, graph_bytes) =
        peak_of(|| TenantGraph::from_edges([2; 16], 1, &edges, &GraphLimits::JOB).unwrap());
    drop(edges);
    let (r, query_bytes) = peak_of(|| query(&g, &mode, &unlimited(GraphLimits::JOB)).unwrap());
    let measured = (graph_bytes.max(24 * g.edges().len()) + query_bytes) as u64;
    let est = r.estimate.memory_bytes;
    println!(
        "{name}: n={} m={} measured={measured} estimate={est} ratio={:.2}",
        g.vertex_count(),
        g.edge_count(),
        est as f64 / measured as f64
    );
    assert!(
        est >= measured,
        "{name}: estimate {est} < measured peak {measured}"
    );
}

#[test]
fn estimate_bounds_peak_certified() {
    check(
        "sparse 10k/50k",
        common::sparse_graph(10_000, 50_000, 1),
        QueryMode::Exact,
    );
    check(
        "sparse 50k/200k",
        common::sparse_graph(50_000, 200_000, 2),
        QueryMode::Exact,
    );
}

#[test]
fn estimate_bounds_peak_stoer_wagner() {
    check(
        "clusters 250/50k",
        common::two_clusters(250, 50_000, 5, 4),
        QueryMode::Exact,
    );
    check(
        "ring+chords 2k/6k",
        common::ring_chords(2_000, 6_000, 5),
        QueryMode::Exact,
    );
}

#[test]
/// Approximate requests are served by the exact solver, so the exact
/// estimate must bound them too.
fn estimate_bounds_peak_approximate_requests() {
    let mode = QueryMode::Approximate { epsilon: 0.2 };
    check(
        "approx 600/2.4k",
        common::two_clusters(300, 2_400, 3, 9),
        mode,
    );
    check(
        "approx 800/6k",
        common::two_clusters(400, 6_000, 3, 9),
        mode,
    );
}
