//! Coordinator component - Demonstrates task_spawn, cap_grant, timer_wait.
//!
//! The Coordinator manages the pipeline lifecycle, spawning tasks, granting
//! capabilities, and managing timing for the cognitive pipeline.
//!
//! ## Syscalls Used
//!
//! - `task_spawn` (5 calls) - Spawn pipeline component tasks
//! - `cap_grant` (20 calls) - Grant capabilities to components
//! - `timer_wait` (100 calls) - Timing coordination
//!
//! ## Architecture
//!
//! ```text
//! +----------------------+
//! |     Coordinator      |
//! |----------------------|
//! |     task_spawn       |---> Spawn component tasks
//! |         |            |
//! |         v            |
//! |     cap_grant        |---> Distribute capabilities
//! |         |            |
//! |         v            |
//! |    timer_wait        |---> Periodic coordination
//! |         |            |
//! |         v            |
//! |  monitor_pipeline    |---> Track progress
//! +----------------------+
//! ```

use super::{Component, ComponentTickResult, KernelInterface};
use crate::{config, manifest::ComponentType, Result};
use ruvix_types::{
    CapHandle, CapRights, RegionHandle, RvfComponentId, RvfMountHandle, TaskHandle, TimerSpec,
};

/// Coordinator component for pipeline management.
pub struct Coordinator {
    /// Component name.
    name: &'static str,

    /// Timer capability.
    timer_cap: CapHandle,

    /// Spawned task handles.
    tasks: Vec<TaskHandle>,

    /// Granted capability handles.
    granted_caps: Vec<CapHandle>,

    /// Tasks spawned count.
    tasks_spawned: u64,

    /// Capabilities granted count.
    caps_granted: u64,

    /// Timer waits completed count.
    timer_waits: u64,

    /// Maximum timer waits.
    max_timer_waits: u64,

    /// Whether initialization is complete.
    initialized: bool,

    /// Whether component is in error state.
    error: bool,

    /// Pipeline state.
    pipeline_state: PipelineCoordinatorState,

    /// Timer interval in nanoseconds.
    timer_interval_ns: u64,

    /// RVF mount handle for component spawning.
    rvf_mount: RvfMountHandle,

    /// Region handles for capability grants.
    regions: Vec<RegionHandle>,
}

/// Pipeline coordinator state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineCoordinatorState {
    /// Not started.
    Idle,

    /// Spawning tasks.
    SpawningTasks,

    /// Granting capabilities.
    GrantingCapabilities,

    /// Running and monitoring.
    Running,

    /// Pipeline completed.
    Completed,

    /// Error state.
    Error,
}

impl Coordinator {
    /// Creates a new Coordinator.
    pub fn new(timer_cap: CapHandle) -> Self {
        Self {
            name: "Coordinator",
            timer_cap,
            tasks: Vec::with_capacity(config::TASK_SPAWNS),
            granted_caps: Vec::with_capacity(config::CAP_GRANTS),
            tasks_spawned: 0,
            caps_granted: 0,
            timer_waits: 0,
            max_timer_waits: config::TIMER_WAITS as u64,
            initialized: false,
            error: false,
            pipeline_state: PipelineCoordinatorState::Idle,
            timer_interval_ns: 10_000_000, // 10ms default
            rvf_mount: RvfMountHandle::null(),
            regions: Vec::new(),
        }
    }

    /// Sets the maximum timer waits.
    pub fn with_max_timer_waits(mut self, max: u64) -> Self {
        self.max_timer_waits = max;
        self
    }

    /// Sets the timer interval.
    pub fn with_timer_interval_ns(mut self, interval: u64) -> Self {
        self.timer_interval_ns = interval;
        self
    }

    /// Sets the RVF mount handle.
    pub fn with_rvf_mount(mut self, mount: RvfMountHandle) -> Self {
        self.rvf_mount = mount;
        self
    }

    /// Adds a region handle for capability grants.
    pub fn add_region(&mut self, region: RegionHandle) {
        self.regions.push(region);
    }

    /// Returns the current pipeline state.
    pub fn state(&self) -> PipelineCoordinatorState {
        self.pipeline_state
    }

    /// Spawns all pipeline component tasks.
    pub fn spawn_tasks(&mut self, kernel: &mut KernelInterface) -> Result<Vec<TaskHandle>> {
        let component_types = [
            ComponentType::SensorAdapter,
            ComponentType::FeatureExtractor,
            ComponentType::ReasoningEngine,
            ComponentType::Attestor,
            ComponentType::Coordinator, // Self - but in real kernel would be different
        ];

        let mut handles = Vec::with_capacity(component_types.len());

        for (i, _comp_type) in component_types.iter().enumerate() {
            let entry = RvfComponentId::new(self.rvf_mount, i as u32);

            // In a real kernel, we would pass entry point info
            // For simulation, we just spawn with empty caps
            let task = kernel.task_spawn(&[])?;

            handles.push(task);
            self.tasks_spawned += 1;
        }

        self.tasks = handles.clone();
        self.pipeline_state = PipelineCoordinatorState::SpawningTasks;

        Ok(handles)
    }

    /// Grants capabilities to pipeline tasks.
    pub fn grant_capabilities(&mut self, kernel: &mut KernelInterface) -> Result<Vec<CapHandle>> {
        if self.tasks.is_empty() {
            return Ok(Vec::new());
        }

        let mut granted = Vec::new();

        // Grant capabilities based on component requirements
        // Each component needs specific capabilities

        // SensorAdapter (task 0): sensor cap, output queue cap
        if self.tasks.len() > 0 {
            let cap1 = kernel.cap_grant(self.tasks[0], self.timer_cap, CapRights::READ)?;
            granted.push(cap1);
            self.caps_granted += 1;

            let cap2 = kernel.cap_grant(self.tasks[0], self.timer_cap, CapRights::WRITE)?;
            granted.push(cap2);
            self.caps_granted += 1;
        }

        // FeatureExtractor (task 1): input queue, output queue, vector store
        if self.tasks.len() > 1 {
            for _ in 0..3 {
                let cap = kernel.cap_grant(self.tasks[1], self.timer_cap, CapRights::READ)?;
                granted.push(cap);
                self.caps_granted += 1;
            }
        }

        // ReasoningEngine (task 2): input queue, vector store, graph store
        if self.tasks.len() > 2 {
            for _ in 0..3 {
                let cap = kernel.cap_grant(self.tasks[2], self.timer_cap, CapRights::READ)?;
                granted.push(cap);
                self.caps_granted += 1;
            }
        }

        // Attestor (task 3): witness log region
        if self.tasks.len() > 3 {
            let cap = kernel.cap_grant(self.tasks[3], self.timer_cap, CapRights::WRITE)?;
            granted.push(cap);
            self.caps_granted += 1;
        }

        // Coordinator (task 4): timer, all regions (for monitoring)
        if self.tasks.len() > 4 {
            for _ in 0..4 {
                let cap = kernel.cap_grant(self.tasks[4], self.timer_cap, CapRights::READ)?;
                granted.push(cap);
                self.caps_granted += 1;
            }
        }

        // Grant additional capabilities up to the required count
        while self.caps_granted < config::CAP_GRANTS as u64 && !self.tasks.is_empty() {
            let task_idx = (self.caps_granted as usize) % self.tasks.len();
            let cap = kernel.cap_grant(self.tasks[task_idx], self.timer_cap, CapRights::READ)?;
            granted.push(cap);
            self.caps_granted += 1;
        }

        self.granted_caps = granted.clone();
        self.pipeline_state = PipelineCoordinatorState::GrantingCapabilities;

        Ok(granted)
    }

    /// Performs a timer wait for coordination.
    pub fn wait_timer(&mut self, kernel: &mut KernelInterface) -> Result<()> {
        let deadline_ns = kernel.current_time_ns + self.timer_interval_ns;
        kernel.timer_wait(deadline_ns)?;
        self.timer_waits += 1;
        Ok(())
    }

    /// Runs one coordination cycle.
    pub fn coordinate_cycle(&mut self, kernel: &mut KernelInterface) -> Result<()> {
        match self.pipeline_state {
            PipelineCoordinatorState::Idle => {
                // Spawn tasks
                self.spawn_tasks(kernel)?;
            }
            PipelineCoordinatorState::SpawningTasks => {
                // Grant capabilities
                self.grant_capabilities(kernel)?;
                self.pipeline_state = PipelineCoordinatorState::Running;
            }
            PipelineCoordinatorState::GrantingCapabilities => {
                // Start running
                self.pipeline_state = PipelineCoordinatorState::Running;
            }
            PipelineCoordinatorState::Running => {
                // Timer wait for periodic coordination
                if self.timer_waits < self.max_timer_waits {
                    self.wait_timer(kernel)?;
                } else {
                    self.pipeline_state = PipelineCoordinatorState::Completed;
                }
            }
            PipelineCoordinatorState::Completed | PipelineCoordinatorState::Error => {
                // No action
            }
        }

        Ok(())
    }

    /// Runs the coordinator for the specified number of cycles.
    pub fn run_cycles(&mut self, kernel: &mut KernelInterface, cycles: u32) -> Result<u32> {
        let mut completed = 0;

        for _ in 0..cycles {
            if self.pipeline_state == PipelineCoordinatorState::Completed {
                break;
            }

            self.coordinate_cycle(kernel)?;
            completed += 1;
        }

        Ok(completed)
    }

    /// Returns coordinator statistics.
    pub fn stats(&self) -> CoordinatorStats {
        CoordinatorStats {
            tasks_spawned: self.tasks_spawned,
            caps_granted: self.caps_granted,
            timer_waits: self.timer_waits,
            pipeline_state: self.pipeline_state,
        }
    }
}

/// Coordinator statistics.
#[derive(Debug, Clone, Copy)]
pub struct CoordinatorStats {
    /// Tasks spawned.
    pub tasks_spawned: u64,

    /// Capabilities granted.
    pub caps_granted: u64,

    /// Timer waits completed.
    pub timer_waits: u64,

    /// Current pipeline state.
    pub pipeline_state: PipelineCoordinatorState,
}

impl Component for Coordinator {
    fn name(&self) -> &'static str {
        self.name
    }

    fn initialize(&mut self) -> Result<()> {
        self.initialized = true;
        Ok(())
    }

    fn tick(&mut self) -> Result<ComponentTickResult> {
        if self.error {
            return Ok(ComponentTickResult::Error);
        }

        if !self.initialized {
            return Ok(ComponentTickResult::Waiting);
        }

        match self.pipeline_state {
            PipelineCoordinatorState::Completed => Ok(ComponentTickResult::Finished),
            PipelineCoordinatorState::Error => Ok(ComponentTickResult::Error),
            _ => Ok(ComponentTickResult::Processed(1)),
        }
    }

    fn shutdown(&mut self) -> Result<()> {
        self.pipeline_state = PipelineCoordinatorState::Completed;
        Ok(())
    }

    fn operation_count(&self) -> u64 {
        self.tasks_spawned + self.caps_granted + self.timer_waits
    }

    fn is_error(&self) -> bool {
        self.error || self.pipeline_state == PipelineCoordinatorState::Error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_coordinator() -> Coordinator {
        Coordinator::new(CapHandle::null())
    }

    #[test]
    fn test_coordinator_creation() {
        let coordinator = create_coordinator();

        assert_eq!(coordinator.name(), "Coordinator");
        assert_eq!(coordinator.tasks_spawned, 0);
        assert_eq!(coordinator.caps_granted, 0);
        assert_eq!(coordinator.timer_waits, 0);
        assert_eq!(coordinator.state(), PipelineCoordinatorState::Idle);
    }

    #[test]
    fn test_spawn_tasks() {
        let mut coordinator = create_coordinator();
        let mut kernel = KernelInterface::new();

        coordinator.initialize().unwrap();

        let tasks = coordinator.spawn_tasks(&mut kernel).unwrap();

        assert_eq!(tasks.len(), 5);
        assert_eq!(coordinator.tasks_spawned, 5);
        assert_eq!(kernel.stats.task_spawn, 5);
        assert_eq!(coordinator.state(), PipelineCoordinatorState::SpawningTasks);
    }

    #[test]
    fn test_grant_capabilities() {
        let mut coordinator = create_coordinator();
        let mut kernel = KernelInterface::new();

        coordinator.initialize().unwrap();
        coordinator.spawn_tasks(&mut kernel).unwrap();

        let caps = coordinator.grant_capabilities(&mut kernel).unwrap();

        assert_eq!(caps.len(), config::CAP_GRANTS);
        assert_eq!(coordinator.caps_granted, config::CAP_GRANTS as u64);
        assert_eq!(kernel.stats.cap_grant, config::CAP_GRANTS as u64);
    }

    #[test]
    fn test_timer_wait() {
        let mut coordinator = create_coordinator().with_timer_interval_ns(1_000_000);
        let mut kernel = KernelInterface::new();

        coordinator.initialize().unwrap();

        let initial_time = kernel.current_time_ns;
        coordinator.wait_timer(&mut kernel).unwrap();

        assert_eq!(coordinator.timer_waits, 1);
        assert_eq!(kernel.stats.timer_wait, 1);
        assert!(kernel.current_time_ns >= initial_time + 1_000_000);
    }

    #[test]
    fn test_coordinate_cycle() {
        let mut coordinator = create_coordinator().with_max_timer_waits(5);
        let mut kernel = KernelInterface::new();

        coordinator.initialize().unwrap();

        // Cycle 1: Spawn tasks
        coordinator.coordinate_cycle(&mut kernel).unwrap();
        assert_eq!(coordinator.state(), PipelineCoordinatorState::SpawningTasks);

        // Cycle 2: Grant capabilities
        coordinator.coordinate_cycle(&mut kernel).unwrap();
        assert_eq!(coordinator.state(), PipelineCoordinatorState::Running);

        // Cycles 3-7: Timer waits (5 times) + 1 more cycle to transition to Completed
        for _i in 0..6 {
            coordinator.coordinate_cycle(&mut kernel).unwrap();
        }
        assert_eq!(coordinator.state(), PipelineCoordinatorState::Completed);
    }

    #[test]
    fn test_run_cycles() {
        let mut coordinator = create_coordinator().with_max_timer_waits(3);
        let mut kernel = KernelInterface::new();

        coordinator.initialize().unwrap();

        // Run enough cycles to complete
        let completed = coordinator.run_cycles(&mut kernel, 10).unwrap();

        assert!(completed >= 5); // At least spawn + grant + 3 timer waits
        assert_eq!(coordinator.state(), PipelineCoordinatorState::Completed);
    }

    #[test]
    fn test_stats() {
        let mut coordinator = create_coordinator().with_max_timer_waits(2);
        let mut kernel = KernelInterface::new();

        coordinator.initialize().unwrap();
        coordinator.run_cycles(&mut kernel, 10).unwrap();

        let stats = coordinator.stats();
        assert_eq!(stats.tasks_spawned, 5);
        assert_eq!(stats.caps_granted, config::CAP_GRANTS as u64);
        assert_eq!(stats.timer_waits, 2);
        assert_eq!(stats.pipeline_state, PipelineCoordinatorState::Completed);
    }

    #[test]
    fn test_component_tick() {
        let mut coordinator = create_coordinator();

        // Before initialization
        assert_eq!(coordinator.tick().unwrap(), ComponentTickResult::Waiting);

        coordinator.initialize().unwrap();

        // Running state
        assert_eq!(
            coordinator.tick().unwrap(),
            ComponentTickResult::Processed(1)
        );

        // After shutdown
        coordinator.shutdown().unwrap();
        assert_eq!(coordinator.tick().unwrap(), ComponentTickResult::Finished);
    }
}
