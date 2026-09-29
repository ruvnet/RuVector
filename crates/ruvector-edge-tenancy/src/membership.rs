//! Membership roles and capability intersection (ADR-351 §4.2, §5.3, §5.5).
//!
//! Capability = route requirement ∩ scope ∩ role, default-deny. A caller with
//! no membership row gets **no** data capability: only `AnyValidToken`
//! routes and `tenant:claim` remain reachable.

use ruvector_edge_auth::scopes::{capabilities_for, UPSTREAM_FIRST_PARTY_CAPS};
use ruvector_edge_auth::{Capability, CapabilitySet, RouteSurface, TokenKind};

/// A member's role in one tenant (`TenantLedger.memberships.role`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// Read.
    Viewer,
    /// Read, write, create.
    Editor,
    /// Everything an editor has, plus admin (members, deny entries, restore,
    /// audit), collection drop and public publish.
    Owner,
}

impl Role {
    /// Wire form stored in the ledger (`owner` / `editor` / `viewer`).
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Viewer => "viewer",
            Role::Editor => "editor",
            Role::Owner => "owner",
        }
    }

    /// Strictly parse the wire form (exact lowercase match).
    pub fn parse(input: &str) -> Option<Role> {
        match input {
            "viewer" => Some(Role::Viewer),
            "editor" => Some(Role::Editor),
            "owner" => Some(Role::Owner),
            _ => None,
        }
    }

    /// The capability ceiling this role allows (§4.2).
    pub fn ceiling(self) -> CapabilitySet {
        let caps: &[Capability] = match self {
            Role::Viewer => &[Capability::Read],
            Role::Editor => &[
                Capability::Read,
                Capability::Write,
                Capability::CreateCollection,
            ],
            Role::Owner => &[
                Capability::Read,
                Capability::Write,
                Capability::CreateCollection,
                Capability::Admin,
                Capability::PublishPublic,
            ],
        };
        set_of(caps)
    }
}

fn set_of(caps: &[Capability]) -> CapabilitySet {
    let mut set = CapabilitySet::EMPTY;
    for c in caps {
        set.insert(*c);
    }
    set
}

/// `a ∩ b`.
pub fn intersect(a: CapabilitySet, b: CapabilitySet) -> CapabilitySet {
    let mut out = CapabilitySet::EMPTY;
    for c in a.iter().filter(|c| b.contains(*c)) {
        out.insert(c);
    }
    out
}

/// Effective capabilities: scope-derived set (or the fixed §5.5 set for
/// upstream-first-party tokens) intersected with the role ceiling.
///
/// `role = None` (not a member) yields [`CapabilitySet::EMPTY`].
/// Callers must already have rejected upstream-first-party tokens on
/// non-REST surfaces; this function denies them anyway (defense in depth).
pub fn effective_capabilities(
    scopes: &[String],
    kind: TokenKind,
    surface: RouteSurface,
    role: Option<Role>,
) -> CapabilitySet {
    let Some(role) = role else {
        return CapabilitySet::EMPTY;
    };
    let from_token = match (kind, surface) {
        (TokenKind::EdgeIssued, _) => capabilities_for(scopes, kind, surface),
        (TokenKind::UpstreamFirstParty, RouteSurface::Rest) => set_of(UPSTREAM_FIRST_PARTY_CAPS),
        (TokenKind::UpstreamFirstParty, _) => CapabilitySet::EMPTY,
    };
    intersect(from_token, role.ceiling())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn role_wire_form_round_trips_strictly() {
        for r in [Role::Viewer, Role::Editor, Role::Owner] {
            assert_eq!(Role::parse(r.as_str()), Some(r));
        }
        for bad in ["Owner", "admin", "", " owner", "owner\0"] {
            assert_eq!(Role::parse(bad), None);
        }
    }

    #[test]
    fn upstream_on_mcp_is_empty_even_for_owner() {
        let caps = effective_capabilities(
            &s(&["mcp:invoke"]),
            TokenKind::UpstreamFirstParty,
            RouteSurface::Mcp,
            Some(Role::Owner),
        );
        assert_eq!(caps, CapabilitySet::EMPTY);
    }
}
