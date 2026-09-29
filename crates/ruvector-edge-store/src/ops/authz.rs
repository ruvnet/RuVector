//! Per-op authorization: capability = scope ∩ role, default-deny (§5.3).
//!
//! Scope is checked before role, so a missing scope is always the step-up
//! `insufficient_scope` (even for an owner), and a member without the role
//! gets `role_required`; a non-member of an unclaimed tenant gets
//! `not_claimed`.

use super::types::Op;
use crate::error::{ErrorCode, OpError};
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_tenancy::Role;

/// What an op needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpRequirement {
    /// Scope-derived capability, `None` for any valid token.
    pub capability: Option<Capability>,
    /// Minimum membership role, `None` for none.
    pub min_role: Option<Role>,
    /// Writes data.
    pub mutating: bool,
}

/// The versioned op table (every [`Op`] has a row: `match` is exhaustive).
pub fn requirement(op: Op) -> OpRequirement {
    let r = |c, role, mutating| OpRequirement {
        capability: Some(c),
        min_role: Some(role),
        mutating,
    };
    match op {
        Op::TenantMe => OpRequirement {
            capability: None,
            min_role: None,
            mutating: false,
        },
        Op::CollectionList | Op::VectorQuery | Op::VectorFetch | Op::UsageGet => {
            r(Capability::Read, Role::Viewer, false)
        }
        Op::CollectionCreate => r(Capability::CreateCollection, Role::Editor, true),
        Op::VectorUpsert | Op::VectorDelete => r(Capability::Write, Role::Editor, true),
    }
}

/// Authorize `op` for a caller with `scope_caps` whose ledger role is
/// `role` in a tenant that is (or is not) `claimed`.
pub fn authorize(
    op: Op,
    scope_caps: CapabilitySet,
    role: Option<Role>,
    claimed: bool,
) -> Result<(), OpError> {
    let req = requirement(op);
    if let Some(cap) = req.capability {
        if !scope_caps.contains(cap) {
            return Err(OpError {
                code: ErrorCode::InsufficientScope,
                detail: "insufficient scope",
                scope: Some(cap.satisfying_scope()),
            });
        }
    }
    if let Some(min) = req.min_role {
        match role {
            None if !claimed => {
                return Err(OpError::new(ErrorCode::NotClaimed, "tenant not claimed"))
            }
            None => return Err(OpError::new(ErrorCode::RoleRequired, "membership required")),
            Some(r) if r < min => {
                return Err(OpError::new(ErrorCode::RoleRequired, "role required"))
            }
            Some(_) => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(list: &[Capability]) -> CapabilitySet {
        let mut s = CapabilitySet::EMPTY;
        list.iter().for_each(|c| s.insert(*c));
        s
    }

    #[test]
    fn scope_before_role_and_default_deny() {
        let read = caps(&[Capability::Read]);
        let all = caps(&[
            Capability::Read,
            Capability::Write,
            Capability::CreateCollection,
        ]);
        // Owner without write scope → step-up, never role.
        let e = authorize(Op::VectorUpsert, read, Some(Role::Owner), true).unwrap_err();
        assert_eq!(
            (e.code, e.scope),
            (ErrorCode::InsufficientScope, Some("ruvector:write"))
        );
        // Viewer with write scope → role_required.
        assert_eq!(
            authorize(Op::VectorUpsert, all, Some(Role::Viewer), true)
                .unwrap_err()
                .code,
            ErrorCode::RoleRequired
        );
        assert_eq!(
            authorize(Op::CollectionCreate, all, Some(Role::Viewer), true)
                .unwrap_err()
                .code,
            ErrorCode::RoleRequired
        );
        // Non-member.
        assert_eq!(
            authorize(Op::VectorQuery, all, None, true)
                .unwrap_err()
                .code,
            ErrorCode::RoleRequired
        );
        assert_eq!(
            authorize(Op::VectorQuery, all, None, false)
                .unwrap_err()
                .code,
            ErrorCode::NotClaimed
        );
        // Allowed paths.
        assert!(authorize(Op::VectorQuery, read, Some(Role::Viewer), true).is_ok());
        assert!(authorize(Op::VectorUpsert, all, Some(Role::Editor), true).is_ok());
        assert!(authorize(Op::TenantMe, CapabilitySet::EMPTY, None, false).is_ok());
    }

    #[test]
    fn every_op_has_a_row_and_mutating_ops_need_editor() {
        for op in Op::ALL {
            let r = requirement(op);
            if r.mutating {
                assert!(r.min_role >= Some(Role::Editor), "{op:?}");
                assert!(matches!(
                    r.capability,
                    Some(Capability::Write | Capability::CreateCollection)
                ));
            }
        }
    }
}
