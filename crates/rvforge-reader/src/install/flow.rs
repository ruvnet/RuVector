//! The install flow itself: offer the contract, then materialize the package.
//!
//! [`offer`] verifies and derives the P6 card; [`install`] repeats the same
//! refusals and, when they all pass, copies the verified bytes in and witnesses
//! the result. The refusals live here rather than in the command layer so a
//! caller cannot reach a decision by invoking a different entry point.

use std::path::Path;

use serde::Serialize;

use crate::capability::{CapabilityCard, CardStatus};
use crate::inspect::{self, SignatureStatus, VerificationStatus};
use crate::runtime::{self, HostProfile, SelectionStatus};
use crate::state::StateCapsule;

use super::{io, InstallError, InstallLayout, InstallRecord, InstallRequest, INSTALL_RECORD_SCHEMA};

/// What the user is asked to consent to, before anything is written.
///
/// [`Self::card`] is the P6 contract and [`Self::grantable_classes`] is exactly
/// what [`install`] will accept, so a UI cannot present one set and submit
/// another without the install refusing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InstallOffer {
    pub path: String,
    pub verified: bool,
    pub base_identity: Option<String>,
    pub signature: SignatureStatus,
    pub card: CapabilityCard,
    pub grantable_classes: Vec<String>,
    pub runtime_profile: Option<String>,
    pub isolation_claim: Option<String>,
    /// Why installation is not offered, when it is not.
    pub refusal: Option<String>,
}

/// Verify a package and derive the contract the user will be asked to accept.
///
/// This runs the same verification [`install`] runs and writes the same
/// receipt, so the card shown here is derived from bytes that passed, not from
/// a hopeful pre-read.
pub fn offer(layout: &InstallLayout, rvf_path: &Path) -> InstallOffer {
    offer_for_host(layout, rvf_path, &HostProfile::detect())
}

/// [`offer`], against an explicit host profile.
pub fn offer_for_host(layout: &InstallLayout, rvf_path: &Path, host: &HostProfile) -> InstallOffer {
    let outcome = inspect::verify_with_receipts(rvf_path, &layout.receipt_log());
    let card = inspect::card_for(&outcome, rvf_path);
    let verified = outcome.summary.verification == VerificationStatus::Verified;

    let choice = runtime::select_for_host(host).ok();
    let unsupported = choice
        .as_ref()
        .map(|c| c.status == SelectionStatus::Unsupported)
        .unwrap_or(true);

    let refusal = if !verified {
        Some(format!(
            "This package did not verify, so it cannot be installed: {}",
            outcome
                .summary
                .notes
                .last()
                .cloned()
                .unwrap_or_else(|| "verification failed".to_string())
        ))
    } else if card.status != CardStatus::Derived {
        Some("No capability contract could be derived for this package.".to_string())
    } else if unsupported {
        Some("No runtime on this host can run this package.".to_string())
    } else {
        None
    };

    InstallOffer {
        path: rvf_path.display().to_string(),
        verified,
        base_identity: outcome.summary.identity.clone().map(content_address),
        signature: outcome.summary.signature,
        grantable_classes: card.requests.iter().map(|r| r.class.clone()).collect(),
        runtime_profile: choice.as_ref().and_then(|c| c.profile.clone()),
        isolation_claim: choice.as_ref().and_then(|c| c.isolation_claim.clone()),
        card,
        refusal,
    }
}

/// Install a verified package, materializing it and witnessing the result.
///
/// # Errors
///
/// Every [`InstallError`]. A refusal writes no install directory; the refusals
/// that are decisions rather than I/O failures are witnessed in the lifecycle
/// log before they are returned.
pub fn install(layout: &InstallLayout, request: &InstallRequest) -> Result<InstallRecord, InstallError> {
    install_for_host(layout, request, &HostProfile::detect())
}

/// [`install`], against an explicit host profile.
pub fn install_for_host(
    layout: &InstallLayout,
    request: &InstallRequest,
    host: &HostProfile,
) -> Result<InstallRecord, InstallError> {
    let offer = offer_for_host(layout, &request.rvf_path, host);
    let log = layout.lifecycle_log();
    let subject = |id: &str| -> String { id.to_string() };

    // Refusals first, each witnessed against the id the install would have had.
    if !offer.verified {
        let reason = offer
            .refusal
            .clone()
            .unwrap_or_else(|| "verification failed".to_string());
        let _ = log.append(
            &request.rvf_path.display().to_string(),
            "install",
            "refused",
            &reason,
        );
        return Err(InstallError::NotVerified { reason });
    }
    if offer.card.status != CardStatus::Derived {
        return Err(InstallError::NoContract {
            reason: offer.card.notes.join(" "),
        });
    }
    let base_identity = offer
        .base_identity
        .clone()
        .ok_or_else(|| InstallError::NotVerified {
            reason: "the container reported no identity".to_string(),
        })?;

    let install_id = request
        .install_id
        .clone()
        .unwrap_or_else(|| derive_install_id(&base_identity));

    for class in &request.accepted_classes {
        let class = class.trim().to_ascii_lowercase();
        if !offer.grantable_classes.contains(&class) {
            let _ = log.append(
                &subject(&install_id),
                "install",
                "refused",
                &format!("'{class}' was accepted but the container does not declare it"),
            );
            return Err(InstallError::GrantNotDeclared { class });
        }
    }

    if offer.runtime_profile.is_none() {
        let order = runtime::select_for_host(host)
            .map(|c| c.order)
            .unwrap_or_default();
        let _ = log.append(
            &subject(&install_id),
            "install",
            "refused",
            "no compatible runtime on this host",
        );
        return Err(InstallError::UnsupportedRuntime { order });
    }

    if request.install_id.is_none() && layout.record_path(&install_id).is_file() {
        return Err(InstallError::AlreadyInstalled { install_id });
    }

    // The capsule constructor is the ADR-288 §2 check: it refuses a state root
    // inside the install root, so this runs before anything is written.
    let install_dir = layout.install_dir(&install_id);
    let capsule = StateCapsule::new(&install_dir, layout.state_root(), &base_identity)?;

    materialize(&request.rvf_path, &install_dir, &capsule)?;

    let granted_text = offer
        .card
        .requests
        .iter()
        .filter(|r| {
            request
                .accepted_classes
                .iter()
                .any(|c| c.trim().eq_ignore_ascii_case(&r.class))
        })
        .map(|r| r.text.clone())
        .collect();

    let mut record = InstallRecord {
        schema: INSTALL_RECORD_SCHEMA.to_string(),
        install_id: install_id.clone(),
        agent_name: request.agent_name.clone(),
        base_identity: base_identity.clone(),
        source_path: request.rvf_path.display().to_string(),
        installed_at_unix: now_unix(),
        install_dir: install_dir.display().to_string(),
        state_root: layout.state_root().display().to_string(),
        granted_classes: request
            .accepted_classes
            .iter()
            .map(|c| c.trim().to_ascii_lowercase())
            .collect(),
        granted_text,
        state_contract: request.state_contract.clone(),
        predecessor_identity: request.predecessor_identity.clone(),
        runtime_profile: offer.runtime_profile.clone(),
        isolation_claim: offer.isolation_claim.clone(),
        signature: offer.signature,
        receipt_id: None,
    };

    let details = format!(
        "installed {base_identity} as '{install_id}' granting [{}]; runtime {}; rollback {}",
        record.granted_classes.join(", "),
        record.runtime_profile.as_deref().unwrap_or("none"),
        if record.state_contract.rollback_unsafe {
            "declared unsafe"
        } else {
            "available"
        }
    );
    record.receipt_id = log
        .append(&subject(&install_id), "install", "accepted", &details)
        .ok()
        .map(|r| r.receipt_id);

    layout.write_record(&record)?;
    Ok(record)
}

/// Copy the verified bytes in, make them read-only, and create the capsule dir.
fn materialize(
    source: &Path,
    install_dir: &Path,
    capsule: &StateCapsule,
) -> Result<(), InstallError> {
    std::fs::create_dir_all(install_dir).map_err(|e| io(install_dir, &e))?;
    let dest = install_dir.join("base.rvf");

    // Remove first: copying onto a 0444 file fails, and an update rebinding an
    // install id replaces the base artifact rather than editing it.
    if dest.exists() {
        set_writable(&dest);
        std::fs::remove_file(&dest).map_err(|e| io(&dest, &e))?;
    }
    std::fs::copy(source, &dest).map_err(|e| io(&dest, &e))?;
    set_read_only(&dest);

    let capsule_dir = capsule.capsule_dir();
    std::fs::create_dir_all(&capsule_dir).map_err(|e| io(&capsule_dir, &e))?;
    Ok(())
}

/// Best effort: an installed artifact that stayed writable is still verified
/// on every open, so a failure to set the bit is not a reason to refuse.
fn set_read_only(path: &Path) {
    if let Ok(meta) = std::fs::metadata(path) {
        let mut perms = meta.permissions();
        perms.set_readonly(true);
        let _ = std::fs::set_permissions(path, perms);
    }
}

fn set_writable(path: &Path) {
    if let Ok(meta) = std::fs::metadata(path) {
        let mut perms = meta.permissions();
        #[allow(clippy::permissions_set_readonly_false)]
        perms.set_readonly(false);
        let _ = std::fs::set_permissions(path, perms);
    }
}

/// `rvf-forge-core` reports a bare hex digest; ADR-288 identities are content
/// addresses. Prefixing here keeps the reader's on-disk records in one shape.
fn content_address(hex: String) -> String {
    if hex.starts_with("sha256:") {
        hex
    } else {
        format!("sha256:{hex}")
    }
}

/// A stable, filesystem-safe id for a first install.
///
/// Derived from the base identity so two different packages never collide, and
/// short enough to be a readable directory name. Updates keep the id they were
/// installed under, so it identifies the *agent*, not the release.
pub fn derive_install_id(base_identity: &str) -> String {
    let hex = base_identity.strip_prefix("sha256:").unwrap_or(base_identity);
    format!("agent-{}", &hex[..hex.len().min(16)])
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
