//! The install-time capability contract (requirements P6, ADR-286).
//!
//! A [`CapabilityCard`] is derived from the signed `CapabilityManifest`
//! described in `docs/research/rvf-forge/registry-model.md`. Two rules govern
//! the derivation and both are enforced here rather than in the UI:
//!
//! - **Default deny.** Every one of the fifteen ADR-286 capability classes that
//!   the manifest does not request is rendered in the "cannot" list. A manifest
//!   that cannot be read produces a card where everything is denied.
//! - **No vague prose.** Broad scopes (`all-files`, `*`, `unrestricted`) are
//!   rejected at derivation time, as are banned phrases such as "access your
//!   computer". The card cannot render a permission it could not describe
//!   specifically.
//!
//! The authoritative source of *which classes* a package opens is the verified
//! container's own declaration, read by `rvf-forge-core`; see
//! [`CapabilityCard::from_declared_classes`]. That declaration carries class
//! names and no scopes, so an unsigned development sidecar may narrow it to
//! specific scopes — [`CapabilityCard::refined_with_manifest_json`] — but it can
//! only ever narrow: a sidecar cannot grant a class the container did not
//! declare.

mod text;

use serde::{Deserialize, Serialize};

use text::{check_phrases, denial_text, request_text, unscoped_request_text};

/// The fifteen ADR-286 capability classes.
pub const CAPABILITY_CLASSES: [&str; 15] = [
    "memory",
    "filesystem",
    "network",
    "model",
    "mcp",
    "process",
    "clock",
    "randomness",
    "gpu",
    "sensor",
    "display",
    "audio",
    "clipboard",
    "persistent-state",
    "inter-agent-messaging",
];

/// Scope tokens that describe too much to render honestly. ADR-294 §8 routes
/// these to manual review; requirements P6 forbids rendering them as prose.
const VAGUE_SCOPES: [&str; 12] = [
    "all",
    "all-files",
    "any",
    "anything",
    "arbitrary",
    "broad",
    "everything",
    "full",
    "full-access",
    "system",
    "unrestricted",
    "your-computer",
];

#[derive(Debug, Clone, Deserialize)]
pub struct CapabilityManifest {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u32,
    #[serde(rename = "manifestId")]
    pub manifest_id: String,
    #[serde(rename = "defaultPolicy")]
    pub default_policy: String,
    #[serde(default)]
    pub requests: Vec<CapabilityRequest>,
    #[serde(default)]
    pub denials: Vec<String>,
    #[serde(default, rename = "manualReviewTriggers")]
    pub manual_review_triggers: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CapabilityRequest {
    pub class: String,
    pub scope: String,
    #[serde(default)]
    pub rationale: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CardStatus {
    /// Derived from a manifest that parsed and passed the P6 rules.
    Derived,
    /// No manifest could be read. Nothing is granted.
    ManifestUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CardLine {
    pub class: String,
    pub scope: Option<String>,
    /// The exact sentence shown to the user.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CapabilityCard {
    pub status: CardStatus,
    pub manifest_id: Option<String>,
    pub default_policy: String,
    /// "This agent requests" — one line per granted capability.
    pub requests: Vec<CardLine>,
    /// "This agent cannot" — explicit denials plus every unrequested class.
    pub cannot: Vec<CardLine>,
    pub manual_review_triggers: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    /// ADR-286 requires a closed baseline.
    DefaultPolicyNotDeny(String),
    UnknownClass(String),
    VagueScope {
        class: String,
        scope: String,
    },
    BannedPhrase {
        class: String,
        phrase: String,
    },
    /// A class appears in both `requests` and `denials`.
    Contradiction(String),
    Parse(String),
}

impl std::fmt::Display for CapabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CapabilityError::DefaultPolicyNotDeny(p) => {
                write!(f, "defaultPolicy is '{p}', expected 'deny'")
            }
            CapabilityError::UnknownClass(c) => {
                write!(f, "'{c}' is not an ADR-286 capability class")
            }
            CapabilityError::VagueScope { class, scope } => write!(
                f,
                "scope '{scope}' on class '{class}' is too broad to describe specifically"
            ),
            CapabilityError::BannedPhrase { class, phrase } => {
                write!(
                    f,
                    "class '{class}' rationale contains banned phrase '{phrase}'"
                )
            }
            CapabilityError::Contradiction(c) => {
                write!(f, "class '{c}' is both requested and denied")
            }
            CapabilityError::Parse(e) => write!(f, "capability manifest parse error: {e}"),
        }
    }
}

impl std::error::Error for CapabilityError {}

impl CapabilityCard {
    /// The card shown when no manifest is available: nothing granted,
    /// every class denied.
    pub fn manifest_unavailable(reason: &str) -> Self {
        Self {
            status: CardStatus::ManifestUnavailable,
            manifest_id: None,
            default_policy: "deny".to_string(),
            requests: Vec::new(),
            cannot: CAPABILITY_CLASSES
                .iter()
                .map(|c| CardLine {
                    class: (*c).to_string(),
                    scope: None,
                    text: denial_text(c),
                })
                .collect(),
            manual_review_triggers: Vec::new(),
            notes: vec![
                reason.to_string(),
                "No capability is granted while the manifest is unreadable (default deny)."
                    .to_string(),
            ],
        }
    }

    /// Derive a card from manifest JSON.
    pub fn from_manifest_json(json: &str) -> Result<Self, CapabilityError> {
        let manifest: CapabilityManifest =
            serde_json::from_str(json).map_err(|e| CapabilityError::Parse(e.to_string()))?;
        Self::from_manifest(&manifest)
    }

    /// Derive a card from a parsed manifest, enforcing the P6 rules.
    pub fn from_manifest(manifest: &CapabilityManifest) -> Result<Self, CapabilityError> {
        if manifest.default_policy != "deny" {
            return Err(CapabilityError::DefaultPolicyNotDeny(
                manifest.default_policy.clone(),
            ));
        }

        let mut requests = Vec::new();
        let mut requested_classes: Vec<String> = Vec::new();

        for req in &manifest.requests {
            let class = req.class.to_ascii_lowercase();
            if !CAPABILITY_CLASSES.contains(&class.as_str()) {
                return Err(CapabilityError::UnknownClass(req.class.clone()));
            }
            check_scope(&class, &req.scope)?;
            if let Some(rationale) = &req.rationale {
                check_phrases(&class, rationale)?;
            }
            let text = request_text(&class, &req.scope);
            check_phrases(&class, &text)?;

            requested_classes.push(class.clone());
            requests.push(CardLine {
                class,
                scope: Some(req.scope.clone()),
                text,
            });
        }

        let mut cannot = Vec::new();
        let mut seen_denials: Vec<String> = Vec::new();

        for denial in &manifest.denials {
            let token = denial.to_ascii_lowercase();
            if requested_classes.contains(&token) {
                return Err(CapabilityError::Contradiction(token));
            }
            if seen_denials.contains(&token) {
                continue;
            }
            seen_denials.push(token.clone());
            cannot.push(CardLine {
                class: token.clone(),
                scope: None,
                text: denial_text(&token),
            });
        }

        // Default deny: everything not requested is denied, whether or not the
        // publisher bothered to list it.
        for class in CAPABILITY_CLASSES {
            if requested_classes.iter().any(|c| c == class)
                || seen_denials.iter().any(|d| d == class)
            {
                continue;
            }
            cannot.push(CardLine {
                class: class.to_string(),
                scope: None,
                text: denial_text(class),
            });
        }

        Ok(Self {
            status: CardStatus::Derived,
            manifest_id: Some(manifest.manifest_id.clone()),
            default_policy: manifest.default_policy.clone(),
            requests,
            cannot,
            manual_review_triggers: manifest.manual_review_triggers.clone(),
            notes: Vec::new(),
        })
    }

    /// Derive a card from the capability classes a verified container declares.
    ///
    /// The container's declaration is class names only — `rvf.capabilities=
    /// network,filesystem` — with no scope attached, so every line says which
    /// class is open and states plainly that the scope is undeclared. Each such
    /// class also raises a manual-review trigger (ADR-294 §8): a class opened
    /// without a stated limit is exactly what review exists to look at.
    ///
    /// # Errors
    ///
    /// [`CapabilityError::UnknownClass`] for a name outside the fifteen ADR-286
    /// classes, and [`CapabilityError::BannedPhrase`] if a generated sentence
    /// would carry banned prose.
    pub fn from_declared_classes(classes: &[String]) -> Result<Self, CapabilityError> {
        let mut requests = Vec::new();
        let mut declared: Vec<String> = Vec::new();
        let mut triggers = Vec::new();

        for name in classes {
            let class = name.trim().to_ascii_lowercase();
            if !CAPABILITY_CLASSES.contains(&class.as_str()) {
                return Err(CapabilityError::UnknownClass(name.clone()));
            }
            if declared.contains(&class) {
                continue;
            }
            let text = unscoped_request_text(&class);
            check_phrases(&class, &text)?;
            triggers.push(format!("class '{class}' is declared without a scope"));
            declared.push(class.clone());
            requests.push(CardLine {
                class,
                scope: None,
                text,
            });
        }

        Ok(Self {
            status: CardStatus::Derived,
            manifest_id: None,
            default_policy: "deny".to_string(),
            cannot: deny_all_except(&declared),
            requests,
            manual_review_triggers: triggers,
            notes: vec![
                "Granted classes come from the verified container's own declaration.".to_string(),
            ],
        })
    }

    /// Narrow a container's declared classes with scopes from a sidecar manifest.
    ///
    /// The sidecar is held to the full P6 rules, so a vague scope in it rejects
    /// the whole card rather than downgrading to the unscoped rendering — a
    /// publisher who wrote `all-files` does not get a friendlier card than one
    /// who wrote nothing. A sidecar request for a class the container did not
    /// declare is dropped into the "cannot" list with a note.
    ///
    /// # Errors
    ///
    /// Any [`CapabilityError`] the sidecar itself raises, plus those of
    /// [`Self::from_declared_classes`].
    pub fn refined_with_manifest_json(
        declared: &[String],
        json: &str,
    ) -> Result<Self, CapabilityError> {
        let sidecar = Self::from_manifest_json(json)?;
        let mut card = Self::from_declared_classes(declared)?;

        let mut widened: Vec<String> = Vec::new();
        for line in &sidecar.requests {
            match card.requests.iter_mut().find(|r| r.class == line.class) {
                Some(existing) => *existing = line.clone(),
                None => widened.push(line.class.clone()),
            }
        }

        // A class that now carries a scope no longer needs the undeclared-scope
        // trigger, so the trigger list is rebuilt rather than filtered.
        let mut triggers: Vec<String> = card
            .requests
            .iter()
            .filter(|r| r.scope.is_none())
            .map(|r| format!("class '{}' is declared without a scope", r.class))
            .collect();
        for trigger in sidecar.manual_review_triggers {
            if !triggers.contains(&trigger) {
                triggers.push(trigger);
            }
        }
        card.manual_review_triggers = triggers;

        // Denials the sidecar names explicitly — including tokens outside the
        // fifteen classes, such as `microphone` — are worth showing verbatim.
        for line in sidecar.cannot {
            if !card.cannot.iter().any(|c| c.class == line.class)
                && !card.requests.iter().any(|r| r.class == line.class)
            {
                card.cannot.push(line);
            }
        }

        card.manifest_id = sidecar.manifest_id;
        if !widened.is_empty() {
            card.notes.push(format!(
                "The sidecar requested {} the container does not declare; \
                 {} denied.",
                widened.join(", "),
                if widened.len() == 1 { "it stays" } else { "they stay" }
            ));
        }
        Ok(card)
    }
}

/// Every class not in `granted`, rendered as a denial.
fn deny_all_except(granted: &[String]) -> Vec<CardLine> {
    CAPABILITY_CLASSES
        .iter()
        .filter(|c| !granted.iter().any(|g| g == *c))
        .map(|c| CardLine {
            class: (*c).to_string(),
            scope: None,
            text: denial_text(c),
        })
        .collect()
}

fn check_scope(class: &str, scope: &str) -> Result<(), CapabilityError> {
    let normalized = scope.trim().to_ascii_lowercase();
    let vague = normalized.is_empty()
        || normalized.contains('*')
        || VAGUE_SCOPES.contains(&normalized.as_str())
        || VAGUE_SCOPES.iter().any(|v| {
            normalized.starts_with(&format!("{v} ")) || normalized.starts_with(&format!("{v}-"))
        });
    if vague {
        return Err(CapabilityError::VagueScope {
            class: class.to_string(),
            scope: scope.to_string(),
        });
    }
    Ok(())
}
