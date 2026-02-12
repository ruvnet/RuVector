//! # ruqu-core -- Quantum Execution Intelligence Engine
//!
//! Pure Rust quantum simulation and execution engine for the ruVector stack.
//! Supports state-vector (up to 32 qubits), stabilizer (millions), Clifford+T
//! (moderate T-count), and tensor network backends with automatic routing,
//! noise modeling, error mitigation, and cryptographic witness logging.
//!
//! ## Quick Start
//!
//! ```
//! use ruqu_core::prelude::*;
//!
//! // Create a Bell state |00> + |11> (unnormalised notation)
//! let mut circuit = QuantumCircuit::new(2);
//! circuit.h(0).cnot(0, 1);
//! let result = Simulator::run(&circuit).unwrap();
//! let probs = result.state.probabilities();
//! // probs ~= [0.5, 0.0, 0.0, 0.5]
//! ```

// -- Core simulation layer --
pub mod types;
pub mod error;
pub mod gate;
pub mod state;
pub mod mixed_precision;
pub mod circuit;
pub mod simulator;
pub mod optimizer;
pub mod simd;
pub mod backend;
pub mod circuit_analyzer;
pub mod stabilizer;
pub mod tensor_network;

// -- Scientific instrument layer (ADR-QE-015) --
pub mod qasm;
pub mod noise;
pub mod mitigation;
pub mod hardware;
pub mod transpiler;
pub mod replay;
pub mod witness;
pub mod confidence;
pub mod verification;

// -- SOTA differentiation layer --
pub mod planner;
pub mod clifford_t;
pub mod decomposition;
pub mod pipeline;

// -- QEC control plane --
pub mod decoder;
pub mod subpoly_decoder;
pub mod qec_scheduler;
pub mod control_theory;

// -- Benchmark & proof suite --
pub mod benchmark;

/// Re-exports of the most commonly used items.
pub mod prelude {
    pub use crate::types::*;
    pub use crate::error::{QuantumError, Result};
    pub use crate::gate::Gate;
    pub use crate::state::QuantumState;
    pub use crate::circuit::QuantumCircuit;
    pub use crate::simulator::{SimConfig, SimulationResult, Simulator, ShotResult};
    pub use crate::qasm::to_qasm3;
    pub use crate::backend::BackendType;
}
