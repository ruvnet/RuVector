//! Self-Healing System for Neural DAG Learning

mod anomaly;
mod index_health;
mod drift_detector;
mod strategies;
mod orchestrator;

pub use anomaly::{AnomalyDetector, Anomaly, AnomalyType, AnomalyConfig};
pub use index_health::{IndexHealthChecker, IndexHealth, IndexType, IndexThresholds, IndexCheckResult, HealthStatus};
pub use drift_detector::{LearningDriftDetector, DriftMetric, DriftTrend};
pub use strategies::{RepairStrategy, RepairResult, IndexRebalanceStrategy, PatternResetStrategy, CacheFlushStrategy};
pub use orchestrator::{HealingOrchestrator, HealingCycleResult};
