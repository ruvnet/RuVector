//! The default-deny capability policy and its hash.
//!
//! ADR-286 §1 defines fifteen capability classes and one rule: nothing is
//! granted unless the policy declares it, and an absent declaration is a
//! denial rather than an unspecified case for a host to resolve. That is why
//! [`CapabilityPolicy::default`] is the *empty* policy and why
//! [`CapabilityPolicy::grants`] is keyed by class — a class that is not a key
//! is denied, with no third state.
//!
//! [`CapabilityPolicy::policy_hash`] is the `capabilityPolicyHash` field of
//! the runtime contract (ADR-291 §1). Because it is the hash of the canonical
//! JSON encoding, the granted set is reproducible from the hash and a host
//! cannot quietly widen it.

use crate::canonical::canonical_sha256_hex;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The fifteen capability classes of ADR-286 §1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CapabilityClass {
    /// Memory quotas; one component may never read another's memory.
    Memory,
    /// Declared filesystem paths only.
    Filesystem,
    /// Declared network destinations only.
    Network,
    /// Which model segments may be loaded, bounded by signed maximum size.
    Model,
    /// MCP tool surfaces, first-class rather than implied by network access.
    Mcp,
    /// Process creation and control.
    Process,
    /// Wall clock; deniable, with a deterministic virtual mode.
    Clock,
    /// Randomness; deniable, with a seeded deterministic mode.
    Randomness,
    /// GPU access.
    Gpu,
    /// Sensor access.
    Sensor,
    /// Display access.
    Display,
    /// Audio access.
    Audio,
    /// Clipboard access.
    Clipboard,
    /// Persistent state; bound to the base RVF lineage.
    PersistentState,
    /// Inter-agent messaging.
    InterAgentMessaging,
}

impl CapabilityClass {
    /// Every class, in declaration order.
    pub const ALL: [CapabilityClass; 15] = [
        Self::Memory,
        Self::Filesystem,
        Self::Network,
        Self::Model,
        Self::Mcp,
        Self::Process,
        Self::Clock,
        Self::Randomness,
        Self::Gpu,
        Self::Sensor,
        Self::Display,
        Self::Audio,
        Self::Clipboard,
        Self::PersistentState,
        Self::InterAgentMessaging,
    ];

    /// The kebab-case wire name, e.g. `persistent-state`.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::Filesystem => "filesystem",
            Self::Network => "network",
            Self::Model => "model",
            Self::Mcp => "mcp",
            Self::Process => "process",
            Self::Clock => "clock",
            Self::Randomness => "randomness",
            Self::Gpu => "gpu",
            Self::Sensor => "sensor",
            Self::Display => "display",
            Self::Audio => "audio",
            Self::Clipboard => "clipboard",
            Self::PersistentState => "persistent-state",
            Self::InterAgentMessaging => "inter-agent-messaging",
        }
    }

    /// Parse a wire name. Returns `None` for anything unrecognized; callers
    /// must treat that as a denial, never as a wildcard.
    pub fn from_wire(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.as_str() == s)
    }
}

impl std::fmt::Display for CapabilityClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What one class is opened to.
///
/// Constraints are a sorted set so that two policies granting the same access
/// hash identically regardless of the order the caller supplied them in.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityGrant {
    /// The scope this class is opened to: paths for `filesystem`, hosts for
    /// `network`, tool names for `mcp`. Empty means the class is granted with
    /// no further narrowing.
    pub scope: BTreeSet<String>,
}

impl CapabilityGrant {
    /// A grant narrowed to the given scope entries.
    pub fn scoped<I, S>(scope: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            scope: scope.into_iter().map(Into::into).collect(),
        }
    }

    /// A grant with no narrowing.
    pub fn unscoped() -> Self {
        Self::default()
    }
}

/// A default-deny capability policy.
///
/// The default value grants nothing. Every granted class is an explicit entry.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityPolicy {
    /// Explicitly granted classes. Absence from this map is a denial.
    pub grants: BTreeMap<CapabilityClass, CapabilityGrant>,
}

impl CapabilityPolicy {
    /// The empty, deny-everything policy.
    pub fn deny_all() -> Self {
        Self::default()
    }

    /// Grant `class` with the given scope, replacing any prior grant.
    pub fn grant(mut self, class: CapabilityClass, grant: CapabilityGrant) -> Self {
        self.grants.insert(class, grant);
        self
    }

    /// Whether `class` is granted.
    pub fn is_granted(&self, class: CapabilityClass) -> bool {
        self.grants.contains_key(&class)
    }

    /// The granted classes, in class order.
    pub fn granted_classes(&self) -> Vec<CapabilityClass> {
        self.grants.keys().copied().collect()
    }

    /// The `capabilityPolicyHash` for this policy: SHA-256 of its canonical
    /// JSON encoding, lowercase hex.
    ///
    /// # Errors
    ///
    /// [`crate::ForgeError::Manifest`] if the policy cannot be canonicalized.
    pub fn policy_hash(&self) -> Result<String> {
        canonical_sha256_hex(self)
    }

    /// Classes this policy does not grant that `declared` names — the classes
    /// an artifact asked for and the build policy withheld.
    pub fn undeclared_grants(&self, declared: &[CapabilityClass]) -> Vec<CapabilityClass> {
        declared
            .iter()
            .copied()
            .filter(|c| !self.is_granted(*c))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_grants_nothing() {
        let p = CapabilityPolicy::default();
        assert!(p.grants.is_empty());
        for class in CapabilityClass::ALL {
            assert!(!p.is_granted(class), "{class} should be denied by default");
        }
    }

    #[test]
    fn wire_names_round_trip_for_every_class() {
        for class in CapabilityClass::ALL {
            assert_eq!(CapabilityClass::from_wire(class.as_str()), Some(class));
        }
        assert_eq!(CapabilityClass::from_wire("teleportation"), None);
        assert_eq!(
            CapabilityClass::PersistentState.as_str(),
            "persistent-state"
        );
    }

    #[test]
    fn policy_hash_is_independent_of_insertion_order() {
        let a = CapabilityPolicy::deny_all()
            .grant(
                CapabilityClass::Network,
                CapabilityGrant::scoped(["b", "a"]),
            )
            .grant(CapabilityClass::Clock, CapabilityGrant::unscoped());
        let b = CapabilityPolicy::deny_all()
            .grant(CapabilityClass::Clock, CapabilityGrant::unscoped())
            .grant(
                CapabilityClass::Network,
                CapabilityGrant::scoped(["a", "b"]),
            );
        assert_eq!(a.policy_hash().unwrap(), b.policy_hash().unwrap());
    }

    #[test]
    fn policy_hash_changes_when_a_grant_changes() {
        let deny = CapabilityPolicy::deny_all();
        let open = CapabilityPolicy::deny_all()
            .grant(CapabilityClass::Network, CapabilityGrant::unscoped());
        let narrowed = CapabilityPolicy::deny_all().grant(
            CapabilityClass::Network,
            CapabilityGrant::scoped(["example.com"]),
        );
        let hashes = [
            deny.policy_hash().unwrap(),
            open.policy_hash().unwrap(),
            narrowed.policy_hash().unwrap(),
        ];
        assert_eq!(
            hashes.iter().collect::<BTreeSet<_>>().len(),
            3,
            "distinct policies must hash distinctly: {hashes:?}"
        );
        assert_eq!(hashes[0].len(), 64);
    }

    #[test]
    fn undeclared_grants_reports_the_withheld_classes() {
        let p = CapabilityPolicy::deny_all()
            .grant(CapabilityClass::Network, CapabilityGrant::unscoped());
        let withheld = p.undeclared_grants(&[CapabilityClass::Network, CapabilityClass::Gpu]);
        assert_eq!(withheld, vec![CapabilityClass::Gpu]);
    }

    #[test]
    fn classes_serialize_as_kebab_case_map_keys() {
        let p = CapabilityPolicy::deny_all().grant(
            CapabilityClass::PersistentState,
            CapabilityGrant::unscoped(),
        );
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"persistent-state\""), "{json}");
    }
}
