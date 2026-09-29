//! Mandatory inspection classes (PIR WP33, ADR-337).
//!
//! Five operation classes are inspected at 100% regardless of what the
//! value-of-information economics say. They are the classes where a missed
//! violation is not recoverable by a later check: privilege escalation,
//! network access, credential use, runtime mutation, and destructive
//! operations.
//!
//! # Why this is structural, not configuration
//!
//! The cheapest way to hit a monitoring-overhead target is to quietly stop
//! inspecting the expensive-to-inspect things. This module is arranged so
//! that is not reachable:
//!
//! - [`MandatoryClass`] is a closed enum. There is no "custom class" arm and
//!   no registry, so the set cannot be narrowed by adding configuration.
//! - [`Classification`] has a private field, **no public constructor**, and
//!   deliberately does **not** derive `Default` — a derived `Default` would
//!   hand any external crate a `Classification` reporting "in no mandatory
//!   class", which is precisely the forgery this is meant to prevent. The
//!   only way to obtain one is [`classify`], which is `pub(crate)`. The
//!   ladder also does not accept one as a parameter: it calls [`classify`]
//!   itself on the subject it was given.
//! - No type in [`crate::monitor`] carries a per-class enable/disable flag,
//!   an allowlist, or a bypass token. There is nothing to set.
//!
//! The remaining honest gap is *recognition*, not *enforcement*: an operation
//! this module fails to recognise as belonging to a class is not inspected as
//! one. That gap is **here**, in the marker lists below — not in the tiny
//! detector. The ladder consults [`classify`] directly and it never consults
//! a detector, so replacing [`crate::monitor::detector::KeywordDetector`]
//! with a real classifier does nothing for it. Closing it means improving
//! these markers, or replacing substring matching with something that
//! understands the operation.
//!
//! The matchers are therefore deliberately broad — a false positive costs one
//! inspection, a false negative costs the guarantee. They still only match
//! literal substrings: an operation that spells its verb differently, encodes
//! it, or expresses it purely through an opaque identifier is not recognised.

use super::detector::InspectionSubject;

/// An operation class that is always inspected, whatever the economics say.
///
/// Closed by construction: adding a class is a source change and a review,
/// not a configuration edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MandatoryClass {
    /// Acquiring or widening authority (sudo, setuid, capability grants,
    /// role assumption).
    PrivilegeEscalation,
    /// Reaching the network in either direction.
    NetworkAccess,
    /// Reading, writing, or presenting secrets.
    CredentialUse,
    /// Changing the running system's own code, policy, or capability set.
    RuntimeMutation,
    /// Irreversible removal or overwrite of data or infrastructure.
    DestructiveOperation,
}

impl MandatoryClass {
    /// Every class, in declaration order. Used by tests to prove none can be
    /// skipped, and by reporting to enumerate coverage.
    pub const ALL: [MandatoryClass; 5] = [
        MandatoryClass::PrivilegeEscalation,
        MandatoryClass::NetworkAccess,
        MandatoryClass::CredentialUse,
        MandatoryClass::RuntimeMutation,
        MandatoryClass::DestructiveOperation,
    ];

    /// Stable identifier for logs and receipts.
    pub const fn as_str(self) -> &'static str {
        match self {
            MandatoryClass::PrivilegeEscalation => "privilege_escalation",
            MandatoryClass::NetworkAccess => "network_access",
            MandatoryClass::CredentialUse => "credential_use",
            MandatoryClass::RuntimeMutation => "runtime_mutation",
            MandatoryClass::DestructiveOperation => "destructive_operation",
        }
    }

    /// Lowercase substrings that mark an operation as belonging to this
    /// class. Deliberately broad: a false positive costs one inspection, a
    /// false negative costs the guarantee this module exists to provide.
    const fn markers(self) -> &'static [&'static str] {
        match self {
            MandatoryClass::PrivilegeEscalation => &[
                "sudo",
                "setuid",
                "chown",
                "chmod",
                "privilege",
                "escalate",
                "grant",
                "role",
                "assume",
                "admin",
                "root",
                "capability",
                // Plain-ASCII privilege verbs that matched nothing until
                // listed: `setcap` grants file capabilities, `chattr` sets
                // immutable/append-only attributes, `impersonate` is the
                // ordinary spelling of assuming another identity.
                "setcap",
                "chattr",
                "impersonat",
            ],
            MandatoryClass::NetworkAccess => &[
                "http", "https", "socket", "curl", "wget", "fetch", "request", "download",
                "upload", "connect", "dns", "tcp", "udp", "webhook", "egress",
            ],
            MandatoryClass::CredentialUse => &[
                "secret",
                "token",
                "password",
                "credential",
                "api_key",
                "apikey",
                "private_key",
                "keychain",
                "auth",
                "bearer",
                ".env",
                "ssh",
            ],
            MandatoryClass::RuntimeMutation => &[
                "mount",
                "unmount",
                "install",
                "plugin",
                "patch",
                "hotfix",
                "reload",
                "policy",
                "config_change",
                "eval",
                "exec",
                "spawn",
                "load_module",
            ],
            MandatoryClass::DestructiveOperation => &[
                "delete",
                "remove",
                "drop",
                "truncate",
                "purge",
                "destroy",
                "wipe",
                "format",
                "rm -rf",
                "overwrite",
                "revoke",
                "terminate",
                "kill",
                // Plain-ASCII destructive commands. These are not obfuscation
                // evasions -- they are the ordinary spelling, and each one
                // classified as non-mandatory until it was listed here.
                "unlink",
                "shred",
                "rmdir",
                "wipefs",
                "mkfs",
                "dd if=",
                "dd of=",
                "if=/dev/",
                "of=/dev/",
            ],
        }
    }
}

/// The set of mandatory classes an operation belongs to.
///
/// Deliberately opaque: the private field and absent public constructor mean
/// this can only be produced by [`classify`], so no caller can assert "this
/// operation is in no mandatory class".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classification {
    matched: Vec<MandatoryClass>,
}

impl Classification {
    /// Classes this operation belongs to, sorted and deduplicated.
    pub fn matched(&self) -> &[MandatoryClass] {
        &self.matched
    }

    /// Whether inspection is mandatory for this operation.
    pub fn is_mandatory(&self) -> bool {
        !self.matched.is_empty()
    }
}

/// Classify a subject against every [`MandatoryClass`].
///
/// `pub(crate)` on purpose: [`crate::monitor::EscalationLadder`] calls this
/// on the subject it was handed, so a caller cannot substitute a
/// pre-computed classification. Matching is case-insensitive across the tool
/// name, the human description, and the serialized arguments — an operation
/// that hides its verb in an argument value is still caught.
pub(crate) fn classify(subject: &InspectionSubject) -> Classification {
    let haystack = subject.searchable_text();
    let mut matched: Vec<MandatoryClass> = MandatoryClass::ALL
        .iter()
        .copied()
        .filter(|class| {
            class
                .markers()
                .iter()
                .any(|marker| haystack.contains(marker))
        })
        .collect();
    matched.sort_unstable();
    matched.dedup();
    Classification { matched }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn subject(name: &str, description: &str, args: serde_json::Value) -> InspectionSubject {
        InspectionSubject::new(name, description, args)
    }

    #[test]
    fn plain_ascii_destructive_and_privilege_verbs_are_recognised() {
        // Each of these classified as NON-mandatory until it was added to the
        // marker lists. They are not obfuscation evasions -- they are the
        // ordinary spelling of the operation, which is what makes missing them
        // worse than missing an encoded one. Regression test so a future edit
        // to the marker arrays cannot quietly drop them again.
        let destructive = [
            "unlink",
            "shred",
            "rmdir",
            "wipefs",
            "mkfs",
            "dd if=/dev/zero of=/dev/sda",
        ];
        for verb in destructive {
            let c = classify(&subject(verb, "", json!({})));
            assert!(
                c.matched().contains(&MandatoryClass::DestructiveOperation),
                "{verb} was not classified as destructive: {:?}",
                c.matched()
            );
        }

        let privilege = ["setcap", "chattr", "impersonate"];
        for verb in privilege {
            let c = classify(&subject(verb, "", json!({})));
            assert!(
                c.matched().contains(&MandatoryClass::PrivilegeEscalation),
                "{verb} was not classified as privilege escalation: {:?}",
                c.matched()
            );
        }
    }

    #[test]
    fn dd_with_options_is_mandatory() {
        let destructive = [
            "dd bs=1M if=/dev/zero of=/dev/sda",
            "dd status=progress of=/dev/nvme0n1",
            "dd conv=notrunc if=/dev/urandom of=/tmp/disk.img",
        ];
        for command in destructive {
            let found = classify(&subject("run", command, json!({})));
            assert!(
                found
                    .matched()
                    .contains(&MandatoryClass::DestructiveOperation),
                "{command} was not classified as destructive"
            );
        }
    }

    #[test]
    fn impersonation_is_mandatory() {
        let found = classify(&subject(
            "run",
            "impersonation of a service account",
            json!({}),
        ));
        assert!(found
            .matched()
            .contains(&MandatoryClass::PrivilegeEscalation));
    }

    #[test]
    fn every_class_is_recognisable() {
        // One representative operation per class. If a class ever stops being
        // detectable at all, the guarantee is hollow for that class.
        let cases = [
            (
                MandatoryClass::PrivilegeEscalation,
                subject("run", "sudo the installer", json!({})),
            ),
            (
                MandatoryClass::NetworkAccess,
                subject("send", "post to an https endpoint", json!({})),
            ),
            (
                MandatoryClass::CredentialUse,
                subject("read", "load the api_key", json!({})),
            ),
            (
                MandatoryClass::RuntimeMutation,
                subject("apply", "mount a plugin at runtime", json!({})),
            ),
            (
                MandatoryClass::DestructiveOperation,
                subject("cleanup", "delete the bucket", json!({})),
            ),
        ];
        for (class, subj) in cases {
            let c = classify(&subj);
            assert!(
                c.matched().contains(&class),
                "{} was not recognised: {:?}",
                class.as_str(),
                c.matched()
            );
            assert!(c.is_mandatory());
        }
    }

    #[test]
    fn a_verb_hidden_in_arguments_is_still_caught() {
        // The name and description are innocuous; the destructive verb is
        // only in an argument value.
        let subj = subject(
            "apply_manifest",
            "routine reconciliation",
            json!({ "command": "rm -rf /var/data" }),
        );
        let c = classify(&subj);
        assert!(c.is_mandatory());
        assert!(c.matched().contains(&MandatoryClass::DestructiveOperation));
    }

    #[test]
    fn classification_is_case_insensitive() {
        let subj = subject("X", "SUDO Privilege Escalate", json!({}));
        assert!(classify(&subj).is_mandatory());
    }

    #[test]
    fn a_benign_operation_matches_nothing() {
        let subj = subject("sum", "add two integers", json!({ "a": 1, "b": 2 }));
        let c = classify(&subj);
        assert!(!c.is_mandatory(), "unexpected match: {:?}", c.matched());
    }

    #[test]
    fn an_operation_can_belong_to_several_classes() {
        let subj = subject(
            "rotate",
            "download the new token over https and delete the old one",
            json!({}),
        );
        let c = classify(&subj);
        assert!(c.matched().len() >= 3, "matched: {:?}", c.matched());
    }

    #[test]
    fn matched_classes_are_sorted_and_deduplicated() {
        let subj = subject("x", "delete delete destroy purge", json!({}));
        let c = classify(&subj);
        let mut sorted = c.matched().to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(c.matched(), sorted.as_slice());
    }
}
