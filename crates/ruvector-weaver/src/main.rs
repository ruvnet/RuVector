//! `weaver` CLI. See ADR-355 and `.claude/skills/ecosystem-weaver/SKILL.md`.

mod cli;

fn main() {
    if let Err(e) = cli::run() {
        eprintln!("weaver: {e:#}");
        std::process::exit(1);
    }
}
