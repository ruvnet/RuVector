//! The sentences a capability card shows, and the phrases it may never show.
//!
//! Separated from the derivation rules in [`super`] because this is the part a
//! reviewer reads as prose: every branch here is a sentence a user will be
//! asked to consent to. There is no generic fallback that could read as
//! "access your computer", and [`check_phrases`] holds the line even for text
//! a publisher supplied.

use super::CapabilityError;

/// Phrases that must never reach the user, whatever the manifest says.
const BANNED_PHRASES: [&str; 6] = [
    "access your computer",
    "access to your computer",
    "control your computer",
    "full access",
    "unrestricted access",
    "all your files",
];

/// Reject text carrying a banned phrase.
///
/// # Errors
///
/// [`CapabilityError::BannedPhrase`], naming both the class and the phrase.
pub(super) fn check_phrases(class: &str, text: &str) -> Result<(), CapabilityError> {
    let lowered = text.to_ascii_lowercase();
    for phrase in BANNED_PHRASES {
        if lowered.contains(phrase) {
            return Err(CapabilityError::BannedPhrase {
                class: class.to_string(),
                phrase: phrase.to_string(),
            });
        }
    }
    Ok(())
}

/// Per-class, per-scope sentence. Every branch names the specific class and
/// the specific scope.
pub(super) fn request_text(class: &str, scope: &str) -> String {
    match (class, scope) {
        ("filesystem", "user-selected") => {
            "Read the files and folders you select, and nothing else".to_string()
        }
        ("persistent-state", "encrypted-local") => {
            "Keep encrypted memory on this computer between sessions".to_string()
        }
        ("model", "embedded") | ("model", "local") => {
            "Run its bundled model on this computer".to_string()
        }
        ("memory", s) => format!("Use up to {s} of memory"),
        ("filesystem", s) => format!("Read files under: {s}"),
        ("network", s) => format!("Open network connections to: {s}"),
        ("model", s) => format!("Run a model located at: {s}"),
        ("mcp", s) => format!("Call MCP tools limited to: {s}"),
        ("process", s) => format!("Start processes limited to: {s}"),
        ("clock", s) => format!("Read the runtime clock: {s}"),
        ("randomness", s) => format!("Draw randomness from: {s}"),
        ("gpu", s) => format!("Use the GPU for: {s}"),
        ("sensor", s) => format!("Read sensor data from: {s}"),
        ("display", s) => format!("Draw to the display: {s}"),
        ("audio", s) => format!("Use audio limited to: {s}"),
        ("clipboard", s) => format!("Use the clipboard limited to: {s}"),
        ("persistent-state", s) => format!("Store persistent state as: {s}"),
        ("inter-agent-messaging", s) => format!("Exchange messages with: {s}"),
        (c, s) => format!("Use the '{c}' capability limited to: {s}"),
    }
}

/// The sentence for a class the container declares without a scope.
///
/// Each branch names the specific class and says the limit is undeclared. None
/// claims a limit that was not stated, and none reaches for the broad prose
/// [`BANNED_PHRASES`] forbids.
pub(super) fn unscoped_request_text(class: &str) -> String {
    let subject = match class {
        "memory" => "Allocate memory",
        "filesystem" => "Read and write files",
        "network" => "Open network connections",
        "model" => "Run a model",
        "mcp" => "Call MCP tools",
        "process" => "Start other programs",
        "clock" => "Read the clock",
        "randomness" => "Draw randomness",
        "gpu" => "Use the GPU",
        "sensor" => "Read sensors",
        "display" => "Draw to the display",
        "audio" => "Record or play audio",
        "clipboard" => "Read and write the clipboard",
        "persistent-state" => "Keep memory between sessions",
        "inter-agent-messaging" => "Exchange messages with other agents",
        other => return format!("Use the '{other}' capability, with no limit declared"),
    };
    format!("{subject} — the package declares no limit on this")
}

/// Denial sentences. Tokens outside the fifteen classes appear in real
/// manifests (`microphone`, `background`), so they get specific phrasing too.
pub(super) fn denial_text(token: &str) -> String {
    match token {
        "network" => "Access the internet".to_string(),
        "filesystem" => "Read folders you have not selected".to_string(),
        "process" => "Start other programs".to_string(),
        "background" => "Run in the background".to_string(),
        "external-model-providers" => "Contact external model providers".to_string(),
        "microphone" => "Use the microphone".to_string(),
        "camera" => "Use the camera".to_string(),
        "audio" => "Record or play audio".to_string(),
        "clipboard" => "Read or write your clipboard".to_string(),
        "gpu" => "Use the GPU".to_string(),
        "sensor" => "Read sensors on this computer".to_string(),
        "display" => "Draw outside its own window".to_string(),
        "mcp" => "Call MCP tools".to_string(),
        "model" => "Run a model".to_string(),
        "memory" => "Allocate memory beyond its declared quota".to_string(),
        "clock" => "Read the system clock".to_string(),
        "randomness" => "Draw randomness outside the runtime".to_string(),
        "persistent-state" => "Keep memory between sessions".to_string(),
        "inter-agent-messaging" => "Exchange messages with other agents".to_string(),
        other => format!("Use the '{other}' capability"),
    }
}
