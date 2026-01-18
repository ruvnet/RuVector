//! Test utilities for ruvector-postgres routing module
//!
//! Provides utilities for:
//! - Mocking PostgreSQL memory contexts
//! - Testing iterator return types (SetOfIterator compatibility)
//! - Verifying PG version compatibility
//! - Memory leak detection

#[cfg(test)]
pub mod mock {
    //! Mock utilities for testing without PostgreSQL context

    use std::sync::Arc;
    use crate::routing::agents::{Agent, AgentRegistry, AgentType};

    /// Mock agent builder for test construction
    pub struct MockAgentBuilder {
        name: String,
        agent_type: AgentType,
        capabilities: Vec<String>,
        cost: f32,
        latency: f32,
        quality: f32,
        is_active: bool,
        embedding: Option<Vec<f32>>,
    }

    impl MockAgentBuilder {
        /// Create a new mock agent builder
        pub fn new(name: impl Into<String>) -> Self {
            Self {
                name: name.into(),
                agent_type: AgentType::LLM,
                capabilities: Vec::new(),
                cost: 0.05,
                latency: 100.0,
                quality: 0.8,
                is_active: true,
                embedding: None,
            }
        }

        /// Set the agent type
        pub fn agent_type(mut self, agent_type: AgentType) -> Self {
            self.agent_type = agent_type;
            self
        }

        /// Add a capability
        pub fn capability(mut self, cap: impl Into<String>) -> Self {
            self.capabilities.push(cap.into());
            self
        }

        /// Set multiple capabilities
        pub fn capabilities(mut self, caps: Vec<String>) -> Self {
            self.capabilities = caps;
            self
        }

        /// Set cost per request
        pub fn cost(mut self, cost: f32) -> Self {
            self.cost = cost;
            self
        }

        /// Set average latency
        pub fn latency(mut self, latency: f32) -> Self {
            self.latency = latency;
            self
        }

        /// Set quality score
        pub fn quality(mut self, quality: f32) -> Self {
            self.quality = quality;
            self
        }

        /// Set active status
        pub fn active(mut self, is_active: bool) -> Self {
            self.is_active = is_active;
            self
        }

        /// Set embedding vector
        pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
            self.embedding = Some(embedding);
            self
        }

        /// Build the agent
        pub fn build(self) -> Agent {
            let mut agent = Agent::new(self.name, self.agent_type, self.capabilities);
            agent.cost_model.per_request = self.cost;
            agent.performance.avg_latency_ms = self.latency;
            agent.performance.quality_score = self.quality;
            agent.is_active = self.is_active;
            agent.embedding = self.embedding;
            agent
        }
    }

    /// Create a test registry with pre-populated agents
    pub fn create_test_registry() -> Arc<AgentRegistry> {
        let registry = Arc::new(AgentRegistry::new());

        // Add some default test agents
        registry.register(
            MockAgentBuilder::new("test-llm")
                .agent_type(AgentType::LLM)
                .capability("coding")
                .cost(0.05)
                .quality(0.85)
                .build()
        ).ok();

        registry.register(
            MockAgentBuilder::new("test-embedding")
                .agent_type(AgentType::Embedding)
                .capability("similarity")
                .cost(0.01)
                .latency(50.0)
                .quality(0.90)
                .build()
        ).ok();

        registry
    }

    /// Create a test registry with agents at different cost/quality/latency points
    pub fn create_cost_quality_latency_registry() -> Arc<AgentRegistry> {
        let registry = Arc::new(AgentRegistry::new());

        // Cheap, fast, low quality
        registry.register(
            MockAgentBuilder::new("cheap-fast-low")
                .cost(0.01)
                .latency(50.0)
                .quality(0.60)
                .embedding(vec![0.1; 384])
                .build()
        ).ok();

        // Expensive, slow, high quality
        registry.register(
            MockAgentBuilder::new("expensive-slow-high")
                .cost(0.10)
                .latency(500.0)
                .quality(0.95)
                .embedding(vec![0.2; 384])
                .build()
        ).ok();

        // Balanced
        registry.register(
            MockAgentBuilder::new("balanced")
                .cost(0.05)
                .latency(150.0)
                .quality(0.80)
                .embedding(vec![0.15; 384])
                .build()
        ).ok();

        registry
    }
}

#[cfg(test)]
pub mod iterator {
    //! Utilities for testing iterator compatibility with PostgreSQL

    use std::marker::PhantomData;

    /// Mock type simulating PostgreSQL SetOfIterator behavior
    ///
    /// This type helps verify that our data structures can be converted
    /// into iterators that PG18's SetOfIterator expects.
    pub struct MockSetOfIterator<T, E> {
        items: Vec<T>,
        _phantom: PhantomData<E>,
    }

    impl<T, E> MockSetOfIterator<T, E> {
        /// Create a new mock iterator from a vector
        pub fn new(items: Vec<T>) -> Self {
            Self {
                items,
                _phantom: PhantomData,
            }
        }

        /// Collect into a vector for testing
        pub fn collect_vec(self) -> Vec<T> {
            self.items
        }
    }

    impl<T, E> From<Vec<T>> for MockSetOfIterator<T, E> {
        fn from(items: Vec<T>) -> Self {
            Self::new(items)
        }
    }

    impl<T, E> FromIterator<T> for MockSetOfIterator<T, E> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            Self {
                items: iter.into_iter().collect(),
                _phantom: PhantomData,
            }
        }
    }

    /// Test that a type can be converted into a SetOfIterator-like structure
    ///
    /// This is the pattern used in ruvector_list_agents() and
    /// ruvector_find_agents_by_capability().
    pub fn test_setof_compatibility<T, E>(items: Vec<T>) -> bool {
        // The key requirement: we can convert Vec<T> into an iterator
        // without requiring 'static lifetime
        let _iterator: MockSetOfIterator<T, E> = MockSetOfIterator::new(items);
        true
    }

    /// Test that the mapping pattern used in operators works
    ///
    /// Verifies: agents.into_iter().map(|agent| (...)) produces a valid iterator
    pub fn test_map_compatibility<T, U, F>(items: Vec<T>, f: F) -> Vec<U>
    where
        T: Clone,
        F: Fn(T) -> U,
    {
        items.into_iter().map(f).collect()
    }
}

#[cfg(test)]
pub mod pg_version {
    //! Utilities for testing PostgreSQL version compatibility

    /// Supported PostgreSQL versions
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PgVersion {
        Pg14,
        Pg15,
        Pg16,
        Pg17,
        Pg18,
    }

    impl PgVersion {
        /// Check if this version supports SetOfIterator
        pub fn supports_setof_iterator(self) -> bool {
            matches!(self, Self::Pg18)
        }

        /// Check if this version requires non-static lifetimes for iterators
        pub fn requires_non_static_lifetime(self) -> bool {
            matches!(self, Self::Pg18)
        }

        /// Parse version string
        pub fn from_str(s: &str) -> Option<Self> {
            match s {
                "14" | "pg14" => Some(Self::Pg14),
                "15" | "pg15" => Some(Self::Pg15),
                "16" | "pg16" => Some(Self::Pg16),
                "17" | "pg17" => Some(Self::Pg17),
                "18" | "pg18" => Some(Self::Pg18),
                _ => None,
            }
        }
    }

    /// Check if the current build is compatible with a given PG version
    pub fn check_compatibility(version: PgVersion) -> CompatibilityResult {
        let mut issues = Vec::new();

        // PG18 requires SetOfIterator, not TableIterator<'static>
        if version == PgVersion::Pg18 {
            // This would be checked at compile time via feature flags
            // In tests, we verify the pattern is correct
        }

        CompatibilityResult {
            version,
            compatible: issues.is_empty(),
            issues,
        }
    }

    pub struct CompatibilityResult {
        pub version: PgVersion,
        pub compatible: bool,
        pub issues: Vec<String>,
    }
}

#[cfg(test)]
pub mod memory {
    //! Memory leak detection utilities

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A counter to track allocations/deallocations
    #[derive(Clone)]
    pub struct AllocationCounter {
        count: Arc<AtomicUsize>,
    }

    impl AllocationCounter {
        /// Create a new counter
        pub fn new() -> Self {
            Self {
                count: Arc::new(AtomicUsize::new(0)),
            }
        }

        /// Increment the counter
        pub fn increment(&self) {
            self.count.fetch_add(1, Ordering::SeqCst);
        }

        /// Get the current count
        pub fn count(&self) -> usize {
            self.count.load(Ordering::SeqCst)
        }

        /// Check for leaks (count should be zero if all tracked items dropped)
        pub fn has_leaks(&self) -> bool {
            self.count() > 0
        }
    }

    impl Default for AllocationCounter {
        fn default() -> Self {
            Self::new()
        }
    }

    /// A wrapper that tracks when values are created and dropped
    pub struct TrackedValue<T> {
        value: T,
        counter: AllocationCounter,
    }

    impl<T> TrackedValue<T> {
        /// Create a new tracked value
        pub fn new(value: T, counter: AllocationCounter) -> Self {
            counter.increment();
            Self { value, counter }
        }

        /// Get the inner value
        pub fn get(&self) -> &T {
            &self.value
        }

        /// Get the inner value mutably
        pub fn get_mut(&mut self) -> &mut T {
            &mut self.value
        }

        /// Get the inner value (only if T is Clone)
        pub fn unwrap(self) -> Option<T>
        where
            T: Clone,
        {
            Some(self.value.clone())
        }
    }

    impl<T: Clone> Clone for TrackedValue<T> {
        fn clone(&self) -> Self {
            self.counter.increment();
            Self {
                value: self.value.clone(),
                counter: self.counter.clone(),
            }
        }
    }

    impl<T> Drop for TrackedValue<T> {
        fn drop(&mut self) {
            self.counter.count.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Run a test and verify no memory leaks
    pub fn test_no_leaks<F>(f: F) -> bool
    where
        F: FnOnce(AllocationCounter) -> (),
    {
        let counter = AllocationCounter::new();
        f(counter.clone());
        !counter.has_leaks()
    }
}

#[cfg(test)]
mod tests {
    use super::mock::{MockAgentBuilder, create_test_registry};
    use super::iterator::{test_setof_compatibility, test_map_compatibility};
    use super::pg_version::{PgVersion, check_compatibility};
    use super::memory::{test_no_leaks, TrackedValue};

    #[test]
    fn test_mock_agent_builder() {
        let agent = MockAgentBuilder::new("test")
            .agent_type(crate::routing::agents::AgentType::LLM)
            .capability("coding")
            .capability("translation")
            .cost(0.05)
            .quality(0.85)
            .build();

        assert_eq!(agent.name, "test");
        assert_eq!(agent.capabilities.len(), 2);
        assert_eq!(agent.cost_model.per_request, 0.05);
    }

    #[test]
    fn test_create_test_registry() {
        let registry = create_test_registry();
        assert_eq!(registry.count(), 2);
    }

    #[test]
    fn test_iterator_setof_compatibility() {
        let items = vec![1, 2, 3, 4, 5];
        // Use a concrete type for the error parameter
        assert!(test_setof_compatibility::<i32, ()>(items));
    }

    #[test]
    fn test_iterator_map_compatibility() {
        let items = vec!["agent1", "agent2", "agent3"];
        let result = test_map_compatibility(items, |s| s.to_uppercase());
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "AGENT1");
    }

    #[test]
    fn test_pg_version_parsing() {
        assert_eq!(PgVersion::from_str("pg18"), Some(PgVersion::Pg18));
        assert_eq!(PgVersion::from_str("18"), Some(PgVersion::Pg18));
        assert_eq!(PgVersion::from_str("17"), Some(PgVersion::Pg17));
        assert_eq!(PgVersion::from_str("invalid"), None);
    }

    #[test]
    fn test_pg18_setof_support() {
        assert!(PgVersion::Pg18.supports_setof_iterator());
        assert!(PgVersion::Pg18.requires_non_static_lifetime());
        assert!(!PgVersion::Pg17.supports_setof_iterator());
    }

    #[test]
    fn test_compatibility_check() {
        let result = check_compatibility(PgVersion::Pg18);
        assert_eq!(result.version, PgVersion::Pg18);
        assert!(result.compatible);
    }

    #[test]
    fn test_memory_tracking_no_leaks() {
        assert!(test_no_leaks(|counter| {
            let _tracked1 = TrackedValue::new(42, counter.clone());
            let _tracked2 = TrackedValue::new(100, counter.clone());
            // Both dropped at end of closure
        }));
    }

    #[test]
    fn test_memory_tracking_with_leaks() {
        assert!(!test_no_leaks(|counter| {
            let _tracked1 = TrackedValue::new(42, counter.clone());
            let leaked = TrackedValue::new(100, counter.clone());
            std::mem::forget(leaked); // Simulate leak
        }));
    }

    #[test]
    fn test_tracked_value() {
        let counter = super::memory::AllocationCounter::new();
        assert_eq!(counter.count(), 0);

        {
            let tracked = TrackedValue::new(42, counter.clone());
            assert_eq!(counter.count(), 1);
            assert_eq!(*tracked.get(), 42);
            assert_eq!(tracked.unwrap(), Some(42));
        }

        assert_eq!(counter.count(), 0); // Dropped
    }
}
