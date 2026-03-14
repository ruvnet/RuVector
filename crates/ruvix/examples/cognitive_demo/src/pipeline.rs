//! Cognitive pipeline orchestration.
//!
//! This module provides the [`CognitivePipeline`] struct that orchestrates
//! all 5 components to run the complete cognitive demo.

use crate::components::{
    Attestor, Component, Coordinator, FeatureExtractor, KernelInterface, ReasoningEngine,
    SensorAdapter,
};
use crate::manifest::DemoManifest;
use crate::stats::{FeatureCoverage, SyscallStats};
use crate::{config, PerceptionEvent, Result};
use ruvix_types::{
    CapHandle, GraphHandle, KernelError, ProofTier, QueueHandle, RegionHandle, VectorKey,
    VectorStoreHandle,
};

/// Pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Number of perception events to process.
    pub event_count: u64,

    /// Batch size for processing.
    pub batch_size: u32,

    /// Enable verbose logging.
    pub verbose: bool,

    /// Random seed for deterministic execution.
    pub seed: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            event_count: config::FULL_PIPELINE_EVENTS as u64,
            batch_size: 100,
            verbose: false,
            seed: 0xDEADBEEF,
        }
    }
}

/// Pipeline execution state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    /// Not initialized.
    Uninitialized,

    /// Initializing components.
    Initializing,

    /// Running the pipeline.
    Running,

    /// Pipeline completed successfully.
    Completed,

    /// Pipeline encountered an error.
    Error,
}

/// Cognitive pipeline orchestrating all 5 components.
pub struct CognitivePipeline {
    /// Pipeline configuration.
    config: PipelineConfig,

    /// Demo manifest.
    manifest: DemoManifest,

    /// Kernel interface for syscall simulation.
    kernel: KernelInterface,

    /// SensorAdapter component.
    sensor_adapter: SensorAdapter,

    /// FeatureExtractor component.
    feature_extractor: FeatureExtractor,

    /// ReasoningEngine component.
    reasoning_engine: ReasoningEngine,

    /// Attestor component.
    attestor: Attestor,

    /// Coordinator component.
    coordinator: Coordinator,

    /// Pipeline state.
    state: PipelineState,

    /// Events processed count.
    events_processed: u64,

    /// Region handles.
    regions: Vec<RegionHandle>,
}

impl CognitivePipeline {
    /// Creates a new cognitive pipeline with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        let manifest = DemoManifest::cognitive_demo();

        // Create kernel interface
        let kernel = KernelInterface::new();

        // Create handles for inter-component communication
        let sensor_to_extractor_queue = QueueHandle::new(0, 0);
        let extractor_to_engine_queue = QueueHandle::new(1, 0);
        let vector_store = VectorStoreHandle::null();
        let graph_store = GraphHandle::null();
        let witness_log = RegionHandle::new(1, 0);

        // Create components
        let sensor_adapter = SensorAdapter::new(
            sensor_to_extractor_queue,
            CapHandle::null(),
            CapHandle::null(),
        )
        .with_event_count(config.event_count)
        .with_seed(config.seed);

        let feature_extractor = FeatureExtractor::new(
            sensor_to_extractor_queue,
            extractor_to_engine_queue,
            vector_store,
            CapHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
        )
        .with_max_events(config.event_count);

        let reasoning_engine = ReasoningEngine::new(
            extractor_to_engine_queue,
            vector_store,
            graph_store,
            CapHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
        )
        .with_max_vectors(config.event_count)
        .with_mutation_frequency(2);

        let attestor = Attestor::new(witness_log, CapHandle::null())
            .with_max_attestations(config.event_count);

        let coordinator =
            Coordinator::new(CapHandle::null()).with_max_timer_waits(config::TIMER_WAITS as u64);

        Self {
            config,
            manifest,
            kernel,
            sensor_adapter,
            feature_extractor,
            reasoning_engine,
            attestor,
            coordinator,
            state: PipelineState::Uninitialized,
            events_processed: 0,
            regions: Vec::new(),
        }
    }

    /// Creates a pipeline with default configuration.
    pub fn default_pipeline() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Returns the current pipeline state.
    pub fn state(&self) -> PipelineState {
        self.state
    }

    /// Returns the manifest.
    pub fn manifest(&self) -> &DemoManifest {
        &self.manifest
    }

    /// Returns the kernel interface (for stats).
    pub fn kernel(&self) -> &KernelInterface {
        &self.kernel
    }

    /// Returns the number of events processed.
    pub fn events_processed(&self) -> u64 {
        self.events_processed
    }

    /// Initializes the pipeline.
    pub fn initialize(&mut self) -> Result<()> {
        self.state = PipelineState::Initializing;

        // Map regions
        let model_weights = self
            .kernel
            .region_map(config::MODEL_WEIGHTS_SIZE, ruvix_types::RegionPolicy::Immutable)?;
        let witness_log = self.kernel.region_map(
            config::WITNESS_LOG_MAX_SIZE,
            ruvix_types::RegionPolicy::AppendOnly {
                max_size: config::WITNESS_LOG_MAX_SIZE,
            },
        )?;
        let vector_store = self.kernel.region_map(
            config::VECTOR_SLOT_SIZE * config::VECTOR_SLOT_COUNT,
            ruvix_types::RegionPolicy::Slab {
                slot_size: config::VECTOR_SLOT_SIZE,
                slot_count: config::VECTOR_SLOT_COUNT,
            },
        )?;

        self.regions = vec![model_weights, witness_log, vector_store];

        // Initialize components
        self.sensor_adapter.initialize()?;
        self.feature_extractor.initialize()?;
        self.reasoning_engine.initialize()?;
        self.attestor.initialize()?;
        self.coordinator.initialize()?;

        // Subscribe sensor
        self.sensor_adapter.subscribe(&mut self.kernel)?;

        // Mount RVF (simulated)
        let manifest_bytes = self.manifest.to_bytes();
        self.kernel.rvf_mount(&manifest_bytes)?;

        self.state = PipelineState::Running;
        Ok(())
    }

    /// Runs the coordinator setup phase.
    pub fn setup_coordinator(&mut self) -> Result<()> {
        // Spawn tasks
        self.coordinator.spawn_tasks(&mut self.kernel)?;

        // Grant capabilities
        self.coordinator.grant_capabilities(&mut self.kernel)?;

        Ok(())
    }

    /// Processes one batch of events through the entire pipeline.
    pub fn process_batch(&mut self) -> Result<u32> {
        if self.state != PipelineState::Running {
            return Err(KernelError::NotPermitted);
        }

        let batch_size = self.config.batch_size;
        let mut processed = 0;

        // 1. Generate sensor events
        let sensor_processed = self.sensor_adapter.process_batch(&mut self.kernel, batch_size)?;

        // 2. Transfer events from sensor to extractor (simulate queue_recv)
        for i in 0..sensor_processed {
            // Simulate receiving from queue (exercises queue_recv syscall)
            let _ = self.kernel.queue_recv(QueueHandle::new(0, 0), 1_000_000)?;

            let event = PerceptionEvent::new(
                self.kernel.current_time_ns,
                1,
                self.events_processed + i as u64,
            );
            self.feature_extractor.queue_event(event);
        }

        // 3. Process in feature extractor
        let extractor_processed =
            self.feature_extractor
                .process_batch(&mut self.kernel, batch_size)?;

        // 4. Transfer embeddings to reasoning engine
        for i in 0..extractor_processed {
            let key = VectorKey::new(self.events_processed + i as u64);
            self.reasoning_engine.queue_vector(key, self.events_processed + i as u64, 0.75);
        }

        // 5. Process in reasoning engine
        let (engine_processed, mutations) =
            self.reasoning_engine.process_batch(&mut self.kernel, batch_size)?;

        // 6. Queue attestations for all operations
        for i in 0..sensor_processed {
            let hash = [i as u8; 32];
            self.attestor
                .queue_attestation(hash, ProofTier::Reflex, 0, self.events_processed + i as u64);
        }

        // 7. Process attestations
        let attested = self.attestor.process_batch(&mut self.kernel, batch_size)?;

        // 8. Timer coordination
        if self.coordinator.stats().timer_waits < config::TIMER_WAITS as u64 {
            self.coordinator.wait_timer(&mut self.kernel)?;
        }

        processed = sensor_processed.max(extractor_processed).max(engine_processed);
        self.events_processed += processed as u64;

        // Check for completion
        if self.events_processed >= self.config.event_count {
            self.state = PipelineState::Completed;
        }

        Ok(processed)
    }

    /// Runs the complete pipeline.
    pub fn run(&mut self) -> Result<PipelineResult> {
        // Initialize
        self.initialize()?;

        // Setup coordinator
        self.setup_coordinator()?;

        // Process all events
        while self.state == PipelineState::Running {
            self.process_batch()?;
        }

        // Complete remaining timer waits
        while self.coordinator.stats().timer_waits < config::TIMER_WAITS as u64 {
            self.coordinator.wait_timer(&mut self.kernel)?;
        }

        // Generate results
        let result = PipelineResult {
            events_processed: self.events_processed,
            vectors_stored: self.feature_extractor.operation_count(),
            graph_mutations: self.reasoning_engine.stats().mutations_applied,
            attestations: self.attestor.stats().attestations_emitted,
            syscall_stats: self.get_syscall_stats(),
            feature_coverage: self.get_feature_coverage(),
            state: self.state,
        };

        Ok(result)
    }

    /// Returns syscall statistics.
    pub fn get_syscall_stats(&self) -> SyscallStats {
        SyscallStats::from_kernel(&self.kernel)
    }

    /// Returns feature coverage.
    pub fn get_feature_coverage(&self) -> FeatureCoverage {
        FeatureCoverage::from_kernel(&self.kernel, &self.manifest)
    }

    /// Performs checkpoint (simulated).
    pub fn checkpoint(&mut self) -> Result<CheckpointData> {
        let checkpoint = CheckpointData {
            events_processed: self.events_processed,
            kernel_time_ns: self.kernel.current_time_ns,
            syscall_stats: self.get_syscall_stats(),
            state: self.state,
        };

        Ok(checkpoint)
    }

    /// Restores from checkpoint (simulated).
    pub fn restore(&mut self, checkpoint: &CheckpointData) -> Result<()> {
        self.events_processed = checkpoint.events_processed;
        self.kernel.current_time_ns = checkpoint.kernel_time_ns;
        self.state = checkpoint.state;
        Ok(())
    }
}

/// Pipeline execution result.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// Total events processed.
    pub events_processed: u64,

    /// Vectors stored in vector store.
    pub vectors_stored: u64,

    /// Graph mutations applied.
    pub graph_mutations: u64,

    /// Attestations emitted.
    pub attestations: u64,

    /// Syscall statistics.
    pub syscall_stats: SyscallStats,

    /// Feature coverage.
    pub feature_coverage: FeatureCoverage,

    /// Final pipeline state.
    pub state: PipelineState,
}

impl PipelineResult {
    /// Returns true if the pipeline completed successfully.
    pub fn is_success(&self) -> bool {
        self.state == PipelineState::Completed
    }

    /// Verifies the result matches expected counts.
    pub fn verify(&self) -> bool {
        let expected = config::FULL_PIPELINE_EVENTS as u64;

        // Check key metrics
        self.events_processed >= expected
            && self.vectors_stored >= expected
            && self.syscall_stats.sensor_subscribe >= 1
            && self.syscall_stats.rvf_mount >= 1
    }
}

/// Checkpoint data for deterministic replay.
#[derive(Debug, Clone)]
pub struct CheckpointData {
    /// Events processed at checkpoint.
    pub events_processed: u64,

    /// Kernel time at checkpoint.
    pub kernel_time_ns: u64,

    /// Syscall stats at checkpoint.
    pub syscall_stats: SyscallStats,

    /// Pipeline state at checkpoint.
    pub state: PipelineState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.event_count, config::FULL_PIPELINE_EVENTS as u64);
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = CognitivePipeline::default_pipeline();
        assert_eq!(pipeline.state(), PipelineState::Uninitialized);
    }

    #[test]
    fn test_pipeline_initialization() {
        let config = PipelineConfig {
            event_count: 100,
            batch_size: 10,
            ..Default::default()
        };
        let mut pipeline = CognitivePipeline::new(config);

        pipeline.initialize().unwrap();
        assert_eq!(pipeline.state(), PipelineState::Running);

        // Verify regions were created
        assert_eq!(pipeline.kernel.stats.region_map, 3);

        // Verify sensor was subscribed
        assert_eq!(pipeline.kernel.stats.sensor_subscribe, 1);

        // Verify RVF was mounted
        assert_eq!(pipeline.kernel.stats.rvf_mount, 1);
    }

    #[test]
    fn test_pipeline_batch_processing() {
        let config = PipelineConfig {
            event_count: 100,
            batch_size: 25,
            ..Default::default()
        };
        let mut pipeline = CognitivePipeline::new(config);

        pipeline.initialize().unwrap();
        pipeline.setup_coordinator().unwrap();

        // Process one batch
        let processed = pipeline.process_batch().unwrap();
        assert!(processed > 0);

        // Verify syscalls were made
        assert!(pipeline.kernel.stats.queue_send > 0);
        assert!(pipeline.kernel.stats.vector_put_proved > 0);
    }

    #[test]
    fn test_pipeline_small_run() {
        let config = PipelineConfig {
            event_count: 50,
            batch_size: 10,
            ..Default::default()
        };
        let mut pipeline = CognitivePipeline::new(config);

        let result = pipeline.run().unwrap();

        assert_eq!(result.state, PipelineState::Completed);
        assert!(result.events_processed >= 50);
        assert!(result.vectors_stored >= 50);
    }

    #[test]
    fn test_pipeline_checkpoint_restore() {
        let config = PipelineConfig {
            event_count: 100,
            batch_size: 20,
            ..Default::default()
        };
        let mut pipeline = CognitivePipeline::new(config);

        pipeline.initialize().unwrap();
        pipeline.process_batch().unwrap();

        // Create checkpoint
        let checkpoint = pipeline.checkpoint().unwrap();
        let saved_events = checkpoint.events_processed;

        // Process more
        pipeline.process_batch().unwrap();
        assert!(pipeline.events_processed > saved_events);

        // Restore
        pipeline.restore(&checkpoint).unwrap();
        assert_eq!(pipeline.events_processed, saved_events);
    }

    #[test]
    fn test_pipeline_result_verify() {
        let config = PipelineConfig {
            event_count: 100,
            batch_size: 25,
            ..Default::default()
        };
        let mut pipeline = CognitivePipeline::new(config);

        let result = pipeline.run().unwrap();

        // Should not verify with default config since we need 10000 events
        // But the basic structure should be correct
        assert!(result.is_success());
        assert!(result.events_processed >= 100);
    }
}
