// RVForge Reader desktop entry point.
//
// Built only with `--features desktop` (see `required-features` in Cargo.toml).
// The core library — inspection, capability card, runtime selection, state
// layout — builds and tests without Tauri.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    rvforge_reader::run();
}
