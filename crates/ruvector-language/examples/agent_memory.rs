//! Runnable evidence-aware agent memory reference application.
#[path = "support/memory.rs"]
mod memory;
use ruvector_language::{
    audit::{self, AuditOptions, Intervention},
    runtime,
};
fn main() {
    let request = memory::request(32, 0);
    if std::env::args().nth(1).as_deref() == Some("request") {
        println!("{}", serde_json::to_string_pretty(&request).unwrap());
        return;
    }
    let receipt =
        runtime::execute(&request.program, &request.snapshot, &request.context, None).unwrap();
    assert_eq!(memory::actual(&receipt), memory::reference(&request));
    let audit = audit::run(
        &request,
        &AuditOptions {
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
        },
    )
    .unwrap();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"receipt":receipt,"audit":audit}))
            .unwrap()
    );
}
