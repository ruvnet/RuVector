//! Exhaustive visibility × relation × capability × action table, checked
//! twice: against `authorize` directly and end-to-end through the registry
//! (get, pull, list, yank, publish, push). Cross-tenant access to a
//! non-public version is 404 whatever the capabilities.

mod common;

use common::*;
use ruvector_edge_auth::{Capability, CapabilitySet};
use ruvector_edge_registry::authz::{authorize, authorize_scope_push, Target};
use ruvector_edge_registry::registry::PageRequest;
use ruvector_edge_registry::{Action, Caller, Denial, RegistryError, Visibility};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rel {
    Uploader,
    Colleague,
    Outsider,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Pull,
    Yank,
    Publish,
    Push,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Want {
    Ok,
    NotFound,
    NotOwner,
    Missing(Capability),
}

fn subsets() -> Vec<Vec<Capability>> {
    (0..16u8)
        .map(|m| {
            ALL_CAPS
                .iter()
                .enumerate()
                .filter(|(i, _)| m & (1 << i) != 0)
                .map(|(_, c)| *c)
                .collect()
        })
        .collect()
}

fn who(rel: Rel, list: &[Capability]) -> Caller {
    match rel {
        Rel::Uploader => caller('a', "es1_alice", list),
        Rel::Colleague => caller('a', "es1_bob", list),
        Rel::Outsider => caller('b', "es1_carol", list),
    }
}

/// The specification, written as a table independent of the implementation.
fn spec(vis: Visibility, rel: Rel, c: CapabilitySet, op: Op) -> Want {
    let has = |x| c.contains(x);
    if op == Op::Push {
        // Decided on the scope (owned by tenant a), not on the version.
        return match (rel, has(Capability::Write)) {
            (Rel::Outsider, _) => Want::NotOwner,
            (_, false) => Want::Missing(Capability::Write),
            _ => Want::Ok,
        };
    }
    let visible = match (vis, rel) {
        (Visibility::Public, _) => true,
        (Visibility::Tenant, Rel::Uploader | Rel::Colleague) => true,
        (Visibility::Tenant, Rel::Outsider) => false,
        (Visibility::Private, Rel::Uploader) => true,
        (Visibility::Private, Rel::Colleague) => has(Capability::Admin),
        (Visibility::Private, Rel::Outsider) => false,
    };
    if !visible {
        return Want::NotFound;
    }
    match (op, rel) {
        (Op::Pull, _) if !has(Capability::Read) => Want::Missing(Capability::Read),
        (Op::Pull, _) => Want::Ok,
        (Op::Yank | Op::Publish, Rel::Outsider) => Want::NotOwner,
        (Op::Yank, _) if !has(Capability::Write) => Want::Missing(Capability::Write),
        (Op::Yank, Rel::Colleague) if !has(Capability::Admin) => Want::NotOwner,
        (Op::Yank, _) => Want::Ok,
        (Op::Publish, _) if !has(Capability::PublishPublic) => {
            Want::Missing(Capability::PublishPublic)
        }
        (Op::Publish, _) => Want::Ok,
        (Op::Push, _) => unreachable!(),
    }
}

fn from_denial(r: Result<(), Denial>) -> Want {
    match r {
        Ok(()) => Want::Ok,
        Err(Denial::NotFound) => Want::NotFound,
        Err(Denial::NotOwner) => Want::NotOwner,
        Err(Denial::Missing(c)) => Want::Missing(c),
    }
}

fn from_registry<T>(r: Result<T, RegistryError>) -> Want {
    match r {
        Ok(_) => Want::Ok,
        Err(RegistryError::NotFound) => Want::NotFound,
        Err(RegistryError::NotOwner) => Want::NotOwner,
        Err(RegistryError::Forbidden(c)) => Want::Missing(c),
        Err(e) => panic!("unexpected {e:?}"),
    }
}

const RELS: [Rel; 3] = [Rel::Uploader, Rel::Colleague, Rel::Outsider];
const OPS: [Op; 4] = [Op::Pull, Op::Yank, Op::Publish, Op::Push];

#[test]
fn authorize_matches_the_table_in_every_cell() {
    let owner = tenant('a');
    let mut cells = 0;
    for vis in Visibility::ALL {
        let t = Target {
            owner: &owner,
            created_by: "es1_alice",
            visibility: vis,
        };
        for rel in RELS {
            for list in subsets() {
                let c = who(rel, &list);
                for op in OPS {
                    let got = match op {
                        Op::Pull => from_denial(authorize(&c, Action::Pull, &t)),
                        Op::Yank => from_denial(authorize(&c, Action::Yank, &t)),
                        Op::Publish => from_denial(authorize(&c, Action::Publish, &t)),
                        Op::Push => from_denial(authorize_scope_push(&c, Some(&owner))),
                    };
                    assert_eq!(
                        got,
                        spec(vis, rel, c.caps, op),
                        "{vis:?} {rel:?} {list:?} {op:?}"
                    );
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 3 * 3 * 16 * 4);
}

fn seeded(vis: Visibility) -> (Reg, FakeR2) {
    let (reg, r2) = (registry(), FakeR2::default());
    let a = alice();
    let initial = if vis == Visibility::Public {
        Visibility::Tenant
    } else {
        vis
    };
    push(
        &reg,
        &r2,
        &a,
        "@acme/pkg",
        "1.0.0",
        initial,
        &small_store(1),
    )
    .unwrap();
    if vis == Visibility::Public {
        publish(&reg, &r2, &a, "@acme/pkg", "1.0.0").unwrap();
    }
    (reg, r2)
}

#[test]
fn registry_enforces_the_table_end_to_end() {
    let (n, v) = (name("@acme/pkg"), ver("1.0.0"));
    for vis in Visibility::ALL {
        for rel in RELS {
            for list in subsets() {
                let c = who(rel, &list);
                for op in OPS {
                    let (reg, r2) = seeded(vis);
                    let want = spec(vis, rel, c.caps, op);
                    let got = match op {
                        Op::Pull => {
                            let g = from_registry(reg.get(&c, &n, &v));
                            assert_eq!(from_registry(reg.pull(&c, &n, &v)), g);
                            assert_eq!(
                                from_registry(reg.list_versions(&c, &n, &PageRequest::default())),
                                g
                            );
                            g
                        }
                        Op::Yank => from_registry(reg.yank(&c, &n, &v, "bad build")),
                        Op::Publish => {
                            let p = from_registry(reg.publish_plan(&c, &n, &v));
                            let ev = evidence(&small_store(1));
                            assert_eq!(from_registry(reg.publish_commit(&c, &n, &v, ev)), p);
                            p
                        }
                        Op::Push => from_registry(push(
                            &reg,
                            &r2,
                            &c,
                            "@acme/pkg",
                            "2.0.0",
                            Visibility::Tenant,
                            &small_store(2),
                        )),
                    };
                    assert_eq!(got, want, "{vis:?} {rel:?} {list:?} {op:?}");
                    let status = match want {
                        Want::Ok => None,
                        Want::NotFound => Some(404),
                        Want::NotOwner | Want::Missing(_) => Some(403),
                    };
                    if let (Some(code), Some(err)) = (
                        status,
                        match op {
                            Op::Pull => reg.get(&c, &n, &v).err(),
                            _ => None,
                        },
                    ) {
                        assert_eq!(err.http_status(), code);
                    }
                }
            }
        }
    }
}

#[test]
fn cross_tenant_private_pull_is_404_even_with_every_capability() {
    let (reg, _) = seeded(Visibility::Private);
    let outsider = caller('b', "es1_carol", &ALL_CAPS);
    let e = reg
        .pull(&outsider, &name("@acme/pkg"), &ver("1.0.0"))
        .unwrap_err();
    assert_eq!(e, RegistryError::NotFound);
    assert_eq!(e.http_status(), 404);
    // Indistinguishable from a version that does not exist.
    let missing = reg
        .pull(&outsider, &name("@acme/pkg"), &ver("9.9.9"))
        .unwrap_err();
    assert_eq!(
        (e.http_status(), e.to_string()),
        (missing.http_status(), missing.to_string())
    );
    let e = reg
        .list_versions(&outsider, &name("@acme/pkg"), &PageRequest::default())
        .unwrap_err();
    assert_eq!(e, RegistryError::NotFound);
}

#[test]
fn forbidden_carries_the_step_up_scope() {
    let (reg, _) = seeded(Visibility::Tenant);
    let reader = caller('a', "es1_bob", &[Capability::Read]);
    let e = reg
        .yank(&reader, &name("@acme/pkg"), &ver("1.0.0"), "")
        .unwrap_err();
    assert_eq!(e.step_up_scope(), Some("ruvector:write"));
    let e = reg
        .publish_plan(&reader, &name("@acme/pkg"), &ver("1.0.0"))
        .unwrap_err();
    assert_eq!(e.step_up_scope(), Some("ruvector:publish"));
    assert_eq!(e.http_status(), 403);
}
