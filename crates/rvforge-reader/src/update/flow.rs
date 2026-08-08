//! Planning, applying, and rolling back an update.

use std::path::PathBuf;

use crate::capability::{CapabilityCard, CardStatus};
use crate::install::{self, InstallRecord, InstallRequest};
use crate::library::{Library, LibraryState};

use super::diff::{diff_cards, SchemaChange, StateDisposition, UpdatePlan};
use super::{io, ReleaseDescriptor, UpdateApproval, UpdateError};

/// Compute the plan for applying `release` to an installed agent.
///
/// Verifies the new package, checks lineage, and diffs the contracts. Nothing
/// is written except the verification receipt and, when the update is real, the
/// library's [`LibraryState::Updates`] flag.
///
/// # Errors
///
/// [`UpdateError::NotVerified`], [`UpdateError::LineageMismatch`],
/// [`UpdateError::NotAnUpdate`], and any library error.
pub fn plan(
    library: &Library,
    install_id: &str,
    release: &ReleaseDescriptor,
) -> Result<UpdatePlan, UpdateError> {
    let entry = library.get(install_id)?;
    let installed = entry.record.base_identity.clone();

    // Lineage first: an update that does not descend from what is installed is
    // rejected before its contract is even shown, so a mismatched package can
    // never reach the approval screen.
    if release.predecessor_identity != installed {
        library.witness(
            install_id,
            "update",
            "rejected",
            &format!(
                "lineage mismatch: release supersedes {}, installed is {installed}",
                release.predecessor_identity
            ),
        );
        return Err(UpdateError::LineageMismatch {
            install_id: install_id.to_string(),
            installed,
            declared: release.predecessor_identity.clone(),
        });
    }

    let offer = library.offer(&release.rvf_path);
    if !offer.verified || offer.card.status != CardStatus::Derived {
        let reason = offer
            .refusal
            .unwrap_or_else(|| "the release did not verify".to_string());
        library.witness(install_id, "update", "rejected", &reason);
        return Err(UpdateError::NotVerified { reason });
    }
    let to_identity = offer
        .base_identity
        .clone()
        .ok_or_else(|| UpdateError::NotVerified {
            reason: "the release reported no identity".to_string(),
        })?;
    if to_identity == installed {
        return Err(UpdateError::NotAnUpdate {
            identity: to_identity,
        });
    }

    let before = installed_card(library, &entry.record);
    let mut diff = diff_cards(&before, &offer.card);
    if release.state_contract.state_schema_version != entry.record.state_contract.state_schema_version
    {
        diff.state_schema_change = Some(SchemaChange {
            from: entry.record.state_contract.state_schema_version,
            to: release.state_contract.state_schema_version,
        });
    }

    let plan = UpdatePlan {
        install_id: install_id.to_string(),
        version_label: release.version_label.clone(),
        from_identity: installed,
        to_identity,
        requires_approval: diff.requires_approval(),
        approval_classes: diff.expanding_classes(),
        rollback_available_after: !release.state_contract.rollback_unsafe,
        state_disposition: StateDisposition::RetainedUnderPredecessor,
        diff,
        new_card: offer.card,
    };

    if entry.state != LibraryState::Updates {
        let _ = library.mark_update_available(
            install_id,
            &format!("{} is available", plan.version_label),
        );
    }
    Ok(plan)
}

/// The contract the installed release is running under.
///
/// Reconstructed from the record rather than re-derived from the artifact: it
/// is what the user actually consented to, which is what an update is a
/// difference *from*. A class that was declared but not accepted is not part of
/// the current contract and so shows as an addition if the new release keeps it.
fn installed_card(library: &Library, record: &InstallRecord) -> CapabilityCard {
    let _ = library;
    let mut card = CapabilityCard::from_declared_classes(&record.granted_classes)
        .unwrap_or_else(|e| CapabilityCard::manifest_unavailable(&e.to_string()));
    // Restore the sentences the user was shown, so a diff renders the same
    // prose the install screen did.
    for (line, text) in card.requests.iter_mut().zip(record.granted_text.iter()) {
        line.text = text.clone();
    }
    card
}

/// Apply an update the user has approved.
///
/// # Errors
///
/// [`UpdateError::ApprovalRequired`] when an expanding class is not named in
/// the approval, plus everything [`plan`] and [`install`](crate::install)
/// raise. A refusal leaves the installed release exactly as it was.
pub fn apply(
    library: &Library,
    release: &ReleaseDescriptor,
    plan: &UpdatePlan,
    approval: &UpdateApproval,
) -> Result<InstallRecord, UpdateError> {
    let missing: Vec<String> = plan
        .approval_classes
        .iter()
        .filter(|c| !approval.approved_classes.iter().any(|a| a == *c))
        .cloned()
        .collect();
    if !missing.is_empty() {
        library.witness(
            &plan.install_id,
            "update",
            "refused",
            &format!("unapproved capability expansion: {}", missing.join(", ")),
        );
        return Err(UpdateError::ApprovalRequired { classes: missing });
    }

    let entry = library.get(&plan.install_id)?;
    let previous = snapshot_previous(&entry.record)?;

    // The accepted set is what the previous release held, plus every expansion
    // the user just approved. A class that appears in the new card but was not
    // granted before and not approved now stays denied.
    let mut accepted: Vec<String> = entry.record.granted_classes.clone();
    for class in &approval.approved_classes {
        if !accepted.contains(class) {
            accepted.push(class.clone());
        }
    }
    accepted.retain(|c| plan.new_card.requests.iter().any(|r| &r.class == c));

    let request = InstallRequest {
        rvf_path: release.rvf_path.clone(),
        agent_name: entry.record.agent_name.clone(),
        accepted_classes: accepted,
        state_contract: release.state_contract.clone(),
        install_id: Some(plan.install_id.clone()),
        predecessor_identity: Some(plan.from_identity.clone()),
    };

    let record = install::install(library.layout(), &request)?;
    library.witness(
        &plan.install_id,
        "update",
        "applied",
        &format!(
            "{} → {} ({}); approved [{}]; predecessor retained at {}",
            plan.from_identity,
            plan.to_identity,
            plan.version_label,
            approval.approved_classes.join(", "),
            previous.display()
        ),
    );
    let _ = library.move_to(&plan.install_id, LibraryState::Installed, None);
    Ok(record)
}

/// Return to the retained predecessor release.
///
/// # Errors
///
/// [`UpdateError::RollbackUnsafe`] when the installed release declared before
/// installation that rollback is unsafe, and [`UpdateError::NoPredecessor`]
/// when nothing was retained.
pub fn rollback(library: &Library, install_id: &str) -> Result<InstallRecord, UpdateError> {
    let entry = library.get(install_id)?;

    if entry.record.state_contract.rollback_unsafe {
        library.witness(
            install_id,
            "rollback",
            "refused",
            "the installed release declared its state migration makes rollback unsafe",
        );
        return Err(UpdateError::RollbackUnsafe {
            install_id: install_id.to_string(),
        });
    }

    let dir = previous_dir(&entry.record);
    let record_path = dir.join("install.json");
    let artifact = dir.join("base.rvf");
    if !record_path.is_file() || !artifact.is_file() {
        return Err(UpdateError::NoPredecessor {
            install_id: install_id.to_string(),
        });
    }

    let text = std::fs::read_to_string(&record_path).map_err(|e| io(&record_path, &e))?;
    let restored: InstallRecord =
        serde_json::from_str(&text).map_err(|e| UpdateError::Io {
            path: record_path.display().to_string(),
            message: format!("not a readable install record: {e}"),
        })?;

    // Reinstall from the retained artifact, which re-verifies it: a predecessor
    // that no longer verifies must not be restored just because it is older.
    let request = InstallRequest {
        rvf_path: artifact.clone(),
        agent_name: restored.agent_name.clone(),
        accepted_classes: restored.granted_classes.clone(),
        state_contract: restored.state_contract.clone(),
        install_id: Some(install_id.to_string()),
        predecessor_identity: restored.predecessor_identity.clone(),
    };
    let record = install::install(library.layout(), &request)?;

    library.witness(
        install_id,
        "rollback",
        "completed",
        &format!(
            "{} → {}",
            entry.record.base_identity, record.base_identity
        ),
    );
    let _ = library.move_to(install_id, LibraryState::Installed, None);
    Ok(record)
}

fn previous_dir(record: &InstallRecord) -> PathBuf {
    record.install_dir().join("previous")
}

/// Keep the current release beside the install so rollback has something to
/// return to. One level deep: rolling back twice is not supported, and the
/// second attempt reports [`UpdateError::NoPredecessor`] rather than silently
/// restoring the wrong release.
fn snapshot_previous(record: &InstallRecord) -> Result<PathBuf, UpdateError> {
    let dir = previous_dir(record);
    std::fs::create_dir_all(&dir).map_err(|e| io(&dir, &e))?;

    let artifact = dir.join("base.rvf");
    if artifact.exists() {
        let mut perms = std::fs::metadata(&artifact)
            .map_err(|e| io(&artifact, &e))?
            .permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        let _ = std::fs::set_permissions(&artifact, perms);
        std::fs::remove_file(&artifact).map_err(|e| io(&artifact, &e))?;
    }
    std::fs::copy(record.base_artifact(), &artifact).map_err(|e| io(&artifact, &e))?;

    let json = serde_json::to_string_pretty(record).map_err(|e| UpdateError::Io {
        path: dir.display().to_string(),
        message: e.to_string(),
    })?;
    let record_path = dir.join("install.json");
    std::fs::write(&record_path, json).map_err(|e| io(&record_path, &e))?;
    Ok(dir)
}
