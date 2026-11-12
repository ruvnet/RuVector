//! `staged-workspace-adapter` — subprocess bridge for cross-language
//! consumers of the ADR-318 content-hash binding (see the crate's
//! `adapter` module for the frozen protocol and exit-code contract).
//!
//! Reads ONE JSON request from stdin, writes ONE JSON response to stdout,
//! and exits 0 (ok), 1 (integrity rejection), or 2 (adapter malfunction).

use std::io::Read;

fn main() {
    let mut input = String::new();
    if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        // No JSON contractually required here, but emit the standard
        // adapter-error shape anyway so consumers see a reason.
        println!(
            "{{\"ok\":false,\"error\":\"adapter-error\",\"detail\":\"cannot read stdin: {e}\"}}"
        );
        std::process::exit(2);
    }
    let outcome = ruvector_staged_workspace::adapter::handle_raw(&input);
    match serde_json::to_string(&outcome.response) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            println!(
                "{{\"ok\":false,\"error\":\"adapter-error\",\"detail\":\"cannot serialize response: {e}\"}}"
            );
            std::process::exit(2);
        }
    }
    std::process::exit(outcome.exit_code);
}
