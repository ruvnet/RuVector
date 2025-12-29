//! SONA: Self-Optimizing Neural Architecture for DAG Learning

mod engine;
mod micro_lora;
mod trajectory;
mod reasoning_bank;
mod ewc;

pub use engine::DagSonaEngine;
pub use micro_lora::{MicroLoRA, MicroLoRAConfig};
pub use trajectory::{DagTrajectory, DagTrajectoryBuffer};
pub use reasoning_bank::{DagReasoningBank, DagPattern, ReasoningBankConfig};
pub use ewc::{EwcPlusPlus, EwcConfig};
