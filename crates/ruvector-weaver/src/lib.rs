//! ruvector-weaver (ADR-355): the deterministic half of the ecosystem
//! fabric loop. It crawls the ruvnet package graph, scans consumer repos for
//! integration / stub / security signals, gates candidates, ranks a
//! one-lane-per-repo work queue with self-contained agent briefs, and keeps
//! the append-only outcome ledger the loop learns from. LLM agents (driven
//! by the `ecosystem-weaver` /loop skill) consume `queue.json` and report
//! outcomes back through `weaver ledger record`.

pub mod brief;
pub mod checks;
pub mod config;
pub mod gate;
pub mod graph;
pub mod http;
pub mod lanes;
pub mod ledger;
pub mod model;
pub mod propose;
pub mod prs;
pub mod queue;
pub mod readme;
pub mod registry;
pub mod release;
pub mod requeue;
pub mod resolve;
pub mod saturation;
pub mod scan;
pub mod semver;
pub mod seo;
pub mod svg;
pub mod util;
pub mod vuln;
