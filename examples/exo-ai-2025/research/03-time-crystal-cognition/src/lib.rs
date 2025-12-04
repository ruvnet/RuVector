// Time Crystal Cognition Library
// Cognitive Time Crystals: Discrete Time Translation Symmetry Breaking in Working Memory

pub mod discrete_time_crystal;
pub mod floquet_cognition;
pub mod temporal_memory;
pub mod simd_optimizations;

// Re-export main types
pub use discrete_time_crystal::{DiscreteTimeCrystal, DTCConfig};
pub use floquet_cognition::{FloquetCognitiveSystem, FloquetConfig, FloquetTrajectory, PhaseDiagram};
pub use temporal_memory::{TemporalMemory, TemporalMemoryConfig, MemoryItem, MemoryStats, WorkingMemoryTask};
pub use simd_optimizations::{SimdDTC, SimdFloquet, HierarchicalTimeCrystal, TopologicalTimeCrystal};
