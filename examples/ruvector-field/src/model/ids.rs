//! Typed identifier newtypes.
//!
//! Every public API uses these instead of raw `u64` so a node id and an edge
//! id can never be mixed up at a call site.
//!
//! # Example
//!
//! ```
//! use ruvector_field::model::NodeId;
//! let a = NodeId(1);
//! let b = NodeId(1);
//! assert_eq!(a, b);
//! assert_eq!(format!("{}", a), "node#1");
//! ```

use core::fmt;

macro_rules! typed_id {
    ($name:ident, $prefix:literal) => {
        /// Strongly typed identifier newtype.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
        pub struct $name(pub u64);

        impl $name {
            /// Wrap a raw `u64`.
            pub const fn new(raw: u64) -> Self {
                Self(raw)
            }
            /// Extract the raw `u64`.
            pub const fn get(self) -> u64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", $prefix, self.0)
            }
        }

        impl From<u64> for $name {
            fn from(raw: u64) -> Self {
                Self(raw)
            }
        }
    };
}

typed_id!(NodeId, "node");
typed_id!(EdgeId, "edge");
typed_id!(HintId, "hint");
typed_id!(WitnessCursor, "witness");
