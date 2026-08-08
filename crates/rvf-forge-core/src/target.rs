//! Build targets and the compatibility matrix gate.
//!
//! [`Target`] enumerates exactly the rows of the ADR-291 §5 platform table.
//! That table is normative: "A target absent from this table is not buildable,
//! not because it is technically impossible but because no validated runtime
//! combination exists for it."
//!
//! One caveat worth knowing when reading this alongside the other documents:
//! ADR-283 §Decision and requirements §3 both list a Linux `.rpm` among the
//! outputs, but ADR-291 §5 has no `.rpm` row. This module follows ADR-291,
//! which is the later and more specific document, so [`Target::parse`] refuses
//! `linux-rpm` with an [`crate::ForgeError::UnsupportedTarget`] naming the
//! `.deb` row as the nearest supported alternative. If the matrix later gains
//! an RPM row, adding the variant here is the whole change.

use crate::error::{ForgeError, Result};
use serde::{Deserialize, Serialize};

/// A platform Forge can build for.
///
/// Ordering is the declaration order below and is what
/// [`crate::build_manifest`] sorts by, so it is part of the manifest's byte
/// stability. Insert new variants at the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Target {
    /// Windows 10 and 11 x64: NSIS `.exe` and `.msi`.
    WindowsX64,
    /// Windows 11 ARM64: NSIS `.exe` and `.msi`.
    WindowsArm64,
    /// macOS 13+ Intel: `.app` and `.dmg`, signed and notarized.
    MacosIntel,
    /// macOS 13+ Apple Silicon: `.app` and `.dmg`, signed and notarized.
    MacosArm64,
    /// macOS universal: Intel plus ARM64 in one package.
    MacosUniversal,
    /// Ubuntu 22.04 and 24.04: `.deb`, GPG signed.
    UbuntuDeb,
    /// Debian 12+: `.deb`, GPG signed.
    DebianDeb,
    /// Generic Linux: `.AppImage`.
    LinuxAppimage,
    /// Browser: WebAssembly execution.
    Browser,
    /// RVM appliances: native `.rvf` output (ADR-293).
    RvmAppliance,
}

impl Target {
    /// Every target in the compatibility matrix.
    pub const ALL: [Target; 10] = [
        Self::WindowsX64,
        Self::WindowsArm64,
        Self::MacosIntel,
        Self::MacosArm64,
        Self::MacosUniversal,
        Self::UbuntuDeb,
        Self::DebianDeb,
        Self::LinuxAppimage,
        Self::Browser,
        Self::RvmAppliance,
    ];

    /// The kebab-case wire name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WindowsX64 => "windows-x64",
            Self::WindowsArm64 => "windows-arm64",
            Self::MacosIntel => "macos-intel",
            Self::MacosArm64 => "macos-arm64",
            Self::MacosUniversal => "macos-universal",
            Self::UbuntuDeb => "ubuntu-deb",
            Self::DebianDeb => "debian-deb",
            Self::LinuxAppimage => "linux-appimage",
            Self::Browser => "browser",
            Self::RvmAppliance => "rvm-appliance",
        }
    }

    /// The artifact extensions this target produces.
    pub const fn artifact_extensions(self) -> &'static [&'static str] {
        match self {
            Self::WindowsX64 | Self::WindowsArm64 => &["exe", "msi"],
            Self::MacosIntel | Self::MacosArm64 | Self::MacosUniversal => &["app", "dmg"],
            Self::UbuntuDeb | Self::DebianDeb => &["deb"],
            Self::LinuxAppimage => &["AppImage"],
            Self::Browser => &["wasm"],
            Self::RvmAppliance => &["rvf", "img", "efi", "bundle"],
        }
    }

    /// Confirm this target is in the compatibility matrix.
    ///
    /// Every [`Target`] variant is, by construction — the check exists so that
    /// the gate has one call site that a future matrix revision can narrow
    /// without changing every caller.
    ///
    /// # Errors
    ///
    /// [`ForgeError::UnsupportedTarget`] if the target is not in the matrix.
    pub fn ensure_supported(self) -> Result<()> {
        if Self::ALL.contains(&self) {
            return Ok(());
        }
        Err(unsupported(
            self.as_str(),
            "absent from the published RVM compatibility matrix (ADR-291 §5)",
        ))
    }

    /// Parse a target from its wire name.
    ///
    /// # Errors
    ///
    /// [`ForgeError::UnsupportedTarget`] for any name not in the matrix, with
    /// a detail naming the nearest supported alternative where one exists
    /// (ADR-291 §2).
    pub fn parse(name: &str) -> Result<Self> {
        let normalized = name.trim().to_ascii_lowercase();
        if let Some(t) = Self::ALL.into_iter().find(|t| t.as_str() == normalized) {
            return Ok(t);
        }
        Err(unsupported(name, nearest_hint(&normalized)))
    }
}

impl std::fmt::Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Target {
    type Err = ForgeError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

fn unsupported(target: &str, detail: impl Into<String>) -> ForgeError {
    ForgeError::UnsupportedTarget {
        target: target.to_string(),
        detail: detail.into(),
    }
}

/// Point the caller at the closest matrix row for names that are plausible but
/// unsupported. Anything else gets the full list.
fn nearest_hint(normalized: &str) -> String {
    let nearest = match normalized {
        "linux-rpm" | "rpm" | "fedora" | "rhel" => Some("ubuntu-deb"),
        "android" | "ios" => None,
        n if n.starts_with("windows") => Some("windows-x64"),
        n if n.starts_with("macos") || n.starts_with("darwin") => Some("macos-universal"),
        n if n.starts_with("linux") || n.starts_with("ubuntu") || n.starts_with("debian") => {
            Some("linux-appimage")
        }
        _ => None,
    };
    match nearest {
        Some(hint) => format!(
            "absent from the published RVM compatibility matrix (ADR-291 §5); \
             the nearest supported target is {hint}"
        ),
        None => format!(
            "absent from the published RVM compatibility matrix (ADR-291 §5); \
             supported targets are {}",
            Target::ALL
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_target_round_trips_through_its_wire_name() {
        for t in Target::ALL {
            assert_eq!(Target::parse(t.as_str()).unwrap(), t);
            assert_eq!(t.to_string(), t.as_str());
            assert!(t.ensure_supported().is_ok());
        }
    }

    #[test]
    fn parsing_is_case_and_whitespace_insensitive() {
        assert_eq!(Target::parse("  Windows-X64 ").unwrap(), Target::WindowsX64);
    }

    #[test]
    fn rpm_is_refused_and_points_at_deb() {
        let err = Target::parse("linux-rpm").unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::UnsupportedTarget);
        assert!(err.to_string().contains("ubuntu-deb"), "{err}");
    }

    #[test]
    fn mobile_targets_are_refused_with_the_full_list() {
        let err = Target::parse("android").unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::UnsupportedTarget);
        assert!(err.to_string().contains("windows-x64"), "{err}");
        assert!(err.to_string().contains("rvm-appliance"), "{err}");
    }

    #[test]
    fn a_near_miss_gets_a_specific_hint() {
        let err = Target::parse("windows-x86").unwrap_err();
        assert!(err.to_string().contains("windows-x64"), "{err}");
    }

    #[test]
    fn unknown_names_are_refused() {
        let err = Target::parse("solaris").unwrap_err();
        assert_eq!(err.code(), crate::ErrorCode::UnsupportedTarget);
        assert!(err.to_string().contains("solaris"), "{err}");
    }

    #[test]
    fn wire_names_are_unique() {
        let mut names: Vec<&str> = Target::ALL.iter().map(|t| t.as_str()).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn ordering_is_declaration_order() {
        let mut sorted = vec![Target::RvmAppliance, Target::WindowsX64, Target::Browser];
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec![Target::WindowsX64, Target::Browser, Target::RvmAppliance]
        );
    }

    #[test]
    fn artifact_extensions_are_declared_for_every_target() {
        for t in Target::ALL {
            assert!(!t.artifact_extensions().is_empty(), "{t}");
        }
        assert_eq!(Target::WindowsX64.artifact_extensions(), &["exe", "msi"]);
    }

    #[test]
    fn targets_serialize_as_kebab_case() {
        let json = serde_json::to_string(&Target::MacosUniversal).unwrap();
        assert_eq!(json, "\"macos-universal\"");
    }
}
