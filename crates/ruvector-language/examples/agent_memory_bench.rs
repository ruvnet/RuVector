//! Reproducible synthetic comparison; no model or external service required.
#[path = "support/memory.rs"]
mod memory;
use ruvector_language::{
    audit::{self, AuditOptions, Intervention},
    runtime,
};
use serde_json::json;
use std::{hint::black_box, time::Instant};
fn measure<T>(mut f: impl FnMut() -> T) -> serde_json::Value {
    for _ in 0..5 {
        black_box(f());
    }
    let mut samples = vec![];
    for _ in 0..30 {
        let start = Instant::now();
        black_box(f());
        samples.push(start.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    json!({"median_ns":samples[15],"p95_ns":samples[28],"samples_ns":samples})
}
fn main() {
    let mut results = vec![];
    for size in [128, 1024, 4096] {
        let r = memory::request(size, 42);
        let execute = || runtime::execute(&r.program, &r.snapshot, &r.context, None).unwrap();
        let receipt = execute();
        let prepared = runtime::PreparedSnapshot::new(&r.snapshot).unwrap();
        let warm = || prepared.execute(&r.program, &r.context, None).unwrap();
        assert_eq!(receipt, warm());
        assert_eq!(memory::reference(&r), memory::actual(&receipt));
        assert_eq!(receipt, execute());
        let options = AuditOptions {
            interventions: vec![
                Intervention::Source {
                    id: "primary".into(),
                },
                Intervention::Source {
                    id: "review".into(),
                },
                Intervention::Edge { id: "e0".into() },
            ],
            max_execution_work: 40_000_000,
        };
        let report = audit::run(&r, &options).unwrap();
        assert_eq!(report, audit::run(&r, &options).unwrap());
        results.push(json!({"nodes":size,"edges":size,"observations":r.snapshot.observations.len(),"seed":42,"equal_statuses":true,"exact_replay":true,
          "direct_status_only":measure(||memory::reference(&r)),"full_runtime":measure(execute),
          "preparation":measure(||runtime::PreparedSnapshot::new(&r.snapshot).unwrap()),
          "prepared_runtime":measure(warm),
          "audit_four_executions":measure(||audit::run(&r,&options).unwrap()),
          "receipt_hash":receipt.receipt_hash,"runtime_work":receipt.work,"audit_execution_work":report.execution_work,
          "audit_status_changes":report.trials.iter().map(|t| t.changes.len()).collect::<Vec<_>>() }));
    }
    println!("{}",serde_json::to_string_pretty(&json!({"schema_version":1,"workload":"synthetic agent memory, exact cosine top4, outgoing depth2, signed evidence and two rules","target_arch":std::env::consts::ARCH,"target_os":std::env::consts::OS,"warmups":5,"samples":30,"limitations":"Direct baseline returns identical statuses but omits validation, grants, resource accounting, provenance and hashes. No predictive accuracy, SOTA superiority, production latency or general speedup claim. Timings exclude parsing and subprocess startup.","results":results})).unwrap());
}
