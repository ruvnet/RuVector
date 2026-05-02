//! ruvllm-sidecar — minimal v1 skeleton.
//!
//! In v1 the trajectory persistence runs *embedded* inside the main process via
//! `PersistentTrajectoryStore`. This binary exists so that the
//! `[[bin]] required-features = ["persistence"]` wiring is in place and a
//! future v2 can host a UDS / IPC sidecar without re-touching the manifest.

fn main() {
    println!("ruvllm-sidecar v1 — embedded mode active, external IPC TBD");
}
