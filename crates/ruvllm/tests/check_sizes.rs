use ruvector_sona::engine::SonaEngine;
use ruvector_sona::ewc::EwcPlusPlus;
use ruvector_sona::reasoning_bank::ReasoningBank;
use ruvllm::sona::{SonaConfig, SonaIntegration};
use std::mem::size_of;

#[test]
fn test_print_sizes() {
    println!("Size of SonaConfig: {} bytes", size_of::<SonaConfig>());
    println!("Size of SonaEngine: {} bytes", size_of::<SonaEngine>());
    println!("Size of EwcPlusPlus: {} bytes", size_of::<EwcPlusPlus>());
    println!(
        "Size of ReasoningBank: {} bytes",
        size_of::<ReasoningBank>()
    );
    println!(
        "Size of SonaIntegration: {} bytes",
        size_of::<SonaIntegration>()
    );
}
