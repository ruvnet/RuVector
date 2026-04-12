//! Four logical shells with phi-scaled compression budgets.
//!
//! Spec section 5.1 and 9.3.
//!
//! # Example
//!
//! ```
//! use ruvector_field::model::Shell;
//! let s = Shell::Concept;
//! assert_eq!(s.depth(), 2);
//! let budget = Shell::Event.budget(1024.0);
//! assert!((budget - 1024.0).abs() < 1e-3);
//! ```

use core::fmt;
use core::str::FromStr;

/// Golden ratio constant used for shell budget scaling.
pub const PHI: f32 = 1.618_033_988;

/// Logical abstraction depth. Distinct from physical memory tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Shell {
    /// Raw observations, tool calls, sensor frames.
    Event,
    /// Recurring motifs and local summaries.
    Pattern,
    /// Durable concepts, templates, domain models.
    Concept,
    /// Policies, invariants, proved contracts.
    Principle,
}

impl Shell {
    /// Ordinal depth in `[0, 3]`.
    pub fn depth(self) -> u8 {
        match self {
            Shell::Event => 0,
            Shell::Pattern => 1,
            Shell::Concept => 2,
            Shell::Principle => 3,
        }
    }

    /// Phi-scaled compression budget `base / phi^depth`. Spec section 9.3.
    pub fn budget(self, base: f32) -> f32 {
        base / PHI.powi(self.depth() as i32)
    }

    /// Next deeper shell, or `None` if already `Principle`.
    pub fn promote(self) -> Option<Shell> {
        match self {
            Shell::Event => Some(Shell::Pattern),
            Shell::Pattern => Some(Shell::Concept),
            Shell::Concept => Some(Shell::Principle),
            Shell::Principle => None,
        }
    }

    /// Shallower shell, or `None` if already `Event`.
    pub fn demote(self) -> Option<Shell> {
        match self {
            Shell::Event => None,
            Shell::Pattern => Some(Shell::Event),
            Shell::Concept => Some(Shell::Pattern),
            Shell::Principle => Some(Shell::Concept),
        }
    }

    /// All four shells in order.
    pub fn all() -> [Shell; 4] {
        [
            Shell::Event,
            Shell::Pattern,
            Shell::Concept,
            Shell::Principle,
        ]
    }
}

impl fmt::Display for Shell {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Shell::Event => f.write_str("event"),
            Shell::Pattern => f.write_str("pattern"),
            Shell::Concept => f.write_str("concept"),
            Shell::Principle => f.write_str("principle"),
        }
    }
}

impl FromStr for Shell {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "event" | "e" => Ok(Shell::Event),
            "pattern" | "p" => Ok(Shell::Pattern),
            "concept" | "c" => Ok(Shell::Concept),
            "principle" | "r" => Ok(Shell::Principle),
            _ => Err("unknown shell"),
        }
    }
}
