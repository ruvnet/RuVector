//! Kernel object type enumeration.
//!
//! Every kernel object has a type that determines what operations are valid
//! and what capabilities are required to access it.

/// The type of a kernel object.
///
/// RuVix has a fixed set of kernel object types. All other abstractions
/// (file systems, networking, device drivers, vector indexes, graph engines,
/// AI inference) are RVF components running in user space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ObjectType {
    /// A task (unit of concurrent execution with capability set).
    Task = 0,

    /// A memory region with access policy (immutable, append-only, slab).
    Region = 1,

    /// A typed ring buffer for inter-task communication.
    Queue = 2,

    /// A deadline-driven scheduling primitive.
    Timer = 3,

    /// A kernel-resident vector store with HNSW indexing.
    VectorStore = 4,

    /// A kernel-resident graph store with mincut partitioning.
    GraphStore = 5,

    /// A mounted RVF package in the component namespace.
    RvfMount = 6,

    /// A capability table entry.
    Capability = 7,

    /// The kernel witness log (append-only attestation log).
    WitnessLog = 8,

    /// A sensor subscription for RuView perception events.
    Subscription = 9,
}

impl ObjectType {
    /// Returns a human-readable name for the object type.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Task => "Task",
            Self::Region => "Region",
            Self::Queue => "Queue",
            Self::Timer => "Timer",
            Self::VectorStore => "VectorStore",
            Self::GraphStore => "GraphStore",
            Self::RvfMount => "RvfMount",
            Self::Capability => "Capability",
            Self::WitnessLog => "WitnessLog",
            Self::Subscription => "Subscription",
        }
    }

    /// Converts from a raw u8 value.
    #[inline]
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Task),
            1 => Some(Self::Region),
            2 => Some(Self::Queue),
            3 => Some(Self::Timer),
            4 => Some(Self::VectorStore),
            5 => Some(Self::GraphStore),
            6 => Some(Self::RvfMount),
            7 => Some(Self::Capability),
            8 => Some(Self::WitnessLog),
            9 => Some(Self::Subscription),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_type_roundtrip() {
        for i in 0..=9 {
            let ot = ObjectType::from_u8(i).unwrap();
            assert_eq!(ot as u8, i);
        }
    }

    #[test]
    fn test_object_type_invalid() {
        assert!(ObjectType::from_u8(10).is_none());
        assert!(ObjectType::from_u8(255).is_none());
    }
}
