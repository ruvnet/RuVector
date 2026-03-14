//! SensorAdapter component - Demonstrates sensor_subscribe + queue IPC.
//!
//! The SensorAdapter subscribes to a simulated sensor and generates perception
//! events that are sent to the FeatureExtractor via queue_send.
//!
//! ## Syscalls Used
//!
//! - `sensor_subscribe` (1 call) - Subscribe to sensor events
//! - `queue_send` (10,000 calls) - Send perception events to pipeline
//!
//! ## Architecture
//!
//! ```text
//! +------------------+
//! |  SensorAdapter   |
//! |------------------|
//! | sensor_subscribe |---> Simulated Sensor
//! |                  |
//! | generate_event() |
//! |        |         |
//! |        v         |
//! |   queue_send     |---> Output Queue ---> FeatureExtractor
//! +------------------+
//! ```

use super::{Component, ComponentTickResult, KernelInterface, PipelineMessage};
use crate::{config, PerceptionEvent, Result};
use ruvix_types::{
    CapHandle, MsgPriority, QueueHandle, SensorDescriptor, SensorType, SubscriptionHandle,
};
use sha2::{Digest, Sha256};

/// SensorAdapter component for generating perception events.
pub struct SensorAdapter {
    /// Component name.
    name: &'static str,

    /// Output queue for perception events.
    output_queue: QueueHandle,

    /// Capability for sensor access.
    sensor_cap: CapHandle,

    /// Capability for queue write.
    queue_cap: CapHandle,

    /// Sensor subscription handle.
    subscription: Option<SubscriptionHandle>,

    /// Current event sequence number.
    sequence: u64,

    /// Total events to generate.
    total_events: u64,

    /// Events generated so far.
    events_generated: u64,

    /// Whether initialization is complete.
    initialized: bool,

    /// Whether component is in error state.
    error: bool,

    /// Simulated sensor type.
    sensor_type: SensorType,

    /// Random seed for deterministic event generation.
    seed: u64,
}

impl SensorAdapter {
    /// Creates a new SensorAdapter.
    pub fn new(output_queue: QueueHandle, sensor_cap: CapHandle, queue_cap: CapHandle) -> Self {
        Self {
            name: "SensorAdapter",
            output_queue,
            sensor_cap,
            queue_cap,
            subscription: None,
            sequence: 0,
            total_events: config::FULL_PIPELINE_EVENTS as u64,
            events_generated: 0,
            initialized: false,
            error: false,
            sensor_type: SensorType::Custom,
            seed: 0xDEADBEEF,
        }
    }

    /// Sets the total number of events to generate.
    pub fn with_event_count(mut self, count: u64) -> Self {
        self.total_events = count;
        self
    }

    /// Sets the sensor type.
    pub fn with_sensor_type(mut self, sensor_type: SensorType) -> Self {
        self.sensor_type = sensor_type;
        self
    }

    /// Sets the random seed for deterministic generation.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Generates a perception event with deterministic content.
    fn generate_event(&mut self, kernel: &mut KernelInterface) -> PerceptionEvent {
        // Deterministic pseudo-random using the seed and sequence
        let combined = self.seed.wrapping_add(self.sequence);
        let hash_input = combined.to_le_bytes();

        let mut hasher = Sha256::new();
        hasher.update(&hash_input);
        hasher.update(b"sensor_event");
        let hash_result: [u8; 32] = hasher.finalize().into();

        // Extract values from hash for deterministic pseudo-randomness
        let priority = hash_result[0];
        let coherence_raw = u16::from_le_bytes([hash_result[1], hash_result[2]]);
        let coherence_score = (coherence_raw % 10001).min(10000);

        PerceptionEvent::new(kernel.current_time_ns, self.sensor_type as u8, self.sequence)
            .with_priority(priority)
            .with_coherence(coherence_score)
            .with_data_hash(hash_result)
    }

    /// Subscribes to the sensor.
    pub fn subscribe(&mut self, kernel: &mut KernelInterface) -> Result<SubscriptionHandle> {
        let descriptor = SensorDescriptor::new(self.sensor_type, 0).with_sample_rate(0); // All events

        let handle = kernel.sensor_subscribe(descriptor, self.output_queue)?;
        self.subscription = Some(handle);
        Ok(handle)
    }

    /// Sends a perception event to the output queue.
    fn send_event(&mut self, event: &PerceptionEvent, kernel: &mut KernelInterface) -> Result<()> {
        let msg = PipelineMessage::PerceptionEvent(*event);
        let bytes = msg.to_bytes();

        // Determine priority based on event priority
        let priority = if event.priority > 200 {
            MsgPriority::High
        } else if event.priority > 100 {
            MsgPriority::Normal
        } else {
            MsgPriority::Low
        };

        kernel.queue_send(self.output_queue, &bytes, priority)?;
        Ok(())
    }

    /// Processes one batch of events (up to batch_size).
    pub fn process_batch(
        &mut self,
        kernel: &mut KernelInterface,
        batch_size: u32,
    ) -> Result<u32> {
        let mut processed = 0;

        while processed < batch_size && self.events_generated < self.total_events {
            // Advance time for realistic simulation
            kernel.advance_time(1_000_000); // 1ms per event

            // Generate and send event
            let event = self.generate_event(kernel);
            self.send_event(&event, kernel)?;

            self.sequence += 1;
            self.events_generated += 1;
            processed += 1;
        }

        Ok(processed)
    }
}

impl Component for SensorAdapter {
    fn name(&self) -> &'static str {
        self.name
    }

    fn initialize(&mut self) -> Result<()> {
        // In a real kernel, we would call sensor_subscribe here
        // For simulation, we just mark as initialized
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

        if self.events_generated >= self.total_events {
            return Ok(ComponentTickResult::Finished);
        }

        // In real implementation, we would process with kernel interface
        // For now, return Idle to indicate we need external processing
        Ok(ComponentTickResult::Idle)
    }

    fn shutdown(&mut self) -> Result<()> {
        self.subscription = None;
        Ok(())
    }

    fn operation_count(&self) -> u64 {
        self.events_generated
    }

    fn is_error(&self) -> bool {
        self.error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensor_adapter_creation() {
        let adapter = SensorAdapter::new(
            QueueHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
        );

        assert_eq!(adapter.name(), "SensorAdapter");
        assert_eq!(adapter.events_generated, 0);
        assert!(!adapter.initialized);
    }

    #[test]
    fn test_sensor_adapter_event_generation() {
        let mut adapter = SensorAdapter::new(
            QueueHandle::new(1, 0),
            CapHandle::null(),
            CapHandle::null(),
        )
        .with_seed(12345)
        .with_event_count(10);

        let mut kernel = KernelInterface::new();
        adapter.initialize().unwrap();

        // Generate first event
        let event1 = adapter.generate_event(&mut kernel);
        assert_eq!(event1.sequence, 0);

        adapter.sequence += 1;

        // Generate second event (should be different)
        let event2 = adapter.generate_event(&mut kernel);
        assert_eq!(event2.sequence, 1);
        assert_ne!(event1.data_hash, event2.data_hash);
    }

    #[test]
    fn test_sensor_adapter_determinism() {
        let mut adapter1 = SensorAdapter::new(
            QueueHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
        )
        .with_seed(42);

        let mut adapter2 = SensorAdapter::new(
            QueueHandle::null(),
            CapHandle::null(),
            CapHandle::null(),
        )
        .with_seed(42);

        let mut kernel = KernelInterface::new();

        // Same seed should produce same events
        let event1 = adapter1.generate_event(&mut kernel);
        let event2 = adapter2.generate_event(&mut kernel);

        assert_eq!(event1.data_hash, event2.data_hash);
        assert_eq!(event1.priority, event2.priority);
        assert_eq!(event1.coherence_score, event2.coherence_score);
    }

    #[test]
    fn test_sensor_adapter_batch_processing() {
        let queue = QueueHandle::new(1, 0);
        let mut adapter = SensorAdapter::new(queue, CapHandle::null(), CapHandle::null())
            .with_event_count(100);

        let mut kernel = KernelInterface::new();
        adapter.initialize().unwrap();

        // Process first batch
        let processed = adapter.process_batch(&mut kernel, 25).unwrap();
        assert_eq!(processed, 25);
        assert_eq!(adapter.events_generated, 25);
        assert_eq!(kernel.stats.queue_send, 25);

        // Process remaining
        let processed = adapter.process_batch(&mut kernel, 100).unwrap();
        assert_eq!(processed, 75);
        assert_eq!(adapter.events_generated, 100);
        assert_eq!(kernel.stats.queue_send, 100);

        // No more events
        let processed = adapter.process_batch(&mut kernel, 10).unwrap();
        assert_eq!(processed, 0);
    }

    #[test]
    fn test_sensor_subscription() {
        let queue = QueueHandle::new(1, 0);
        let mut adapter = SensorAdapter::new(queue, CapHandle::null(), CapHandle::null())
            .with_sensor_type(SensorType::Camera);

        let mut kernel = KernelInterface::new();

        let handle = adapter.subscribe(&mut kernel).unwrap();
        assert!(!handle.is_null());
        assert_eq!(kernel.stats.sensor_subscribe, 1);
        assert!(adapter.subscription.is_some());
    }
}
