//! FR004 selection order: rvm-native → os-isolation+wasm → wasm →
//! linux-microvm → unsupported.

use rvforge_reader::runtime::{
    self, CompatibilityMatrix, HostProfile, PolicySource, SelectionStatus,
};

fn host(os: &str, arch: &str, os_isolation: bool, kvm: bool, rvm: bool) -> HostProfile {
    HostProfile {
        os: os.to_string(),
        arch: arch.to_string(),
        os_isolation,
        kvm,
        rvm_measured_boot: rvm,
    }
}

fn matrix() -> CompatibilityMatrix {
    runtime::embedded_matrix().expect("vendored matrix must parse")
}

/// The vendored copy must stay in step with the canonical document.
#[test]
fn vendored_matrix_matches_canonical_copy() {
    let canonical = include_str!("../../../docs/research/rvf-forge/compatibility-matrix.json");
    assert_eq!(
        canonical,
        runtime::COMPATIBILITY_MATRIX_JSON,
        "vendored assets/compatibility-matrix.json has drifted from the canonical doc"
    );
}

#[test]
fn selection_order_is_the_fr004_order() {
    assert_eq!(
        matrix().selection_order,
        vec!["rvm-native", "os-isolation+wasm", "wasm", "linux-microvm"]
    );
}

#[test]
fn order_comes_from_embedded_policy_only() {
    let choice = runtime::select(&matrix(), &host("linux", "x64", true, true, false));
    assert_eq!(choice.policy_source, PolicySource::EmbeddedDefault);
    assert_eq!(choice.order, matrix().selection_order);
}

#[test]
fn os_isolation_wins_over_plain_wasm_when_the_adapter_can_engage_it() {
    for os in ["windows", "macos", "linux"] {
        for arch in ["x64", "aarch64"] {
            let choice = runtime::select(&matrix(), &host(os, arch, true, true, false));
            assert_eq!(choice.status, SelectionStatus::Selected, "{os}/{arch}");
            assert_eq!(
                choice.profile.as_deref(),
                Some("os-isolation+wasm"),
                "{os}/{arch}"
            );
            assert_eq!(
                choice.isolation_claim.as_deref(),
                Some("os-sandbox+wasm"),
                "hosted RVM must not claim bare-metal isolation (ADR-285)"
            );
            assert!(
                !choice.mechanisms.is_empty(),
                "{os}/{arch} should list mechanisms"
            );
        }
    }
}

#[test]
fn falls_back_to_wasm_when_os_confinement_is_unavailable() {
    for os in ["windows", "macos", "linux"] {
        let choice = runtime::select(&matrix(), &host(os, "x64", false, true, false));
        assert_eq!(choice.profile.as_deref(), Some("wasm"), "{os}");
        assert_eq!(
            choice.isolation_claim.as_deref(),
            Some("wasm-sandbox"),
            "{os}"
        );
    }
}

#[test]
fn microvm_is_never_chosen_over_wasm() {
    // linux-microvm sits below wasm in the order, so a KVM host that can also
    // run wasm still gets wasm. Preferring the microVM would be a reordering.
    let choice = runtime::select(&matrix(), &host("linux", "x64", false, true, false));
    assert_eq!(choice.profile.as_deref(), Some("wasm"));
}

#[test]
fn rvm_native_is_planned_and_therefore_not_selectable_yet() {
    // Even a host claiming measured boot cannot select it while the matrix
    // marks the profile `planned`.
    let choice = runtime::select(&matrix(), &host("rvm", "x64", true, true, true));
    assert_eq!(choice.status, SelectionStatus::Unsupported);
    let rvm = choice
        .evaluated
        .iter()
        .find(|e| e.profile == "rvm-native")
        .expect("rvm-native must be evaluated");
    assert!(!rvm.eligible);
    assert!(rvm.reason.contains("planned"), "reason was: {}", rvm.reason);
}

#[test]
fn browser_gets_wasm_only() {
    let choice = runtime::select(&matrix(), &host("browser", "wasm32", true, true, true));
    assert_eq!(choice.profile.as_deref(), Some("wasm"));
    let os_iso = choice
        .evaluated
        .iter()
        .find(|e| e.profile == "os-isolation+wasm")
        .unwrap();
    assert!(
        !os_iso.eligible,
        "browser has no os-isolation platform entry"
    );
}

#[test]
fn unsupported_is_terminal_for_an_unknown_platform() {
    let choice = runtime::select(&matrix(), &host("plan9", "x64", true, true, true));
    assert_eq!(choice.status, SelectionStatus::Unsupported);
    assert!(choice.profile.is_none());
    assert!(choice.isolation_claim.is_none());
    assert!(choice.evaluated.iter().all(|e| !e.eligible));
}

#[test]
fn unknown_architecture_is_not_selectable() {
    let choice = runtime::select(&matrix(), &host("linux", "riscv64", true, true, false));
    assert_eq!(choice.status, SelectionStatus::Unsupported);
}

/// Every reachable combination of the three host capability flags across every
/// OS in the matrix, checked against the order.
#[test]
fn all_host_flag_combinations_follow_the_order() {
    let m = matrix();
    for os in ["windows", "macos", "linux", "browser", "rvm"] {
        let arch = if os == "browser" { "wasm32" } else { "x64" };
        for os_isolation in [false, true] {
            for kvm in [false, true] {
                for rvm in [false, true] {
                    let h = host(os, arch, os_isolation, kvm, rvm);
                    let choice = runtime::select(&m, &h);
                    let expected = expected_profile(os, os_isolation, kvm);
                    assert_eq!(
                        choice.profile.as_deref(),
                        expected,
                        "{os} os_isolation={os_isolation} kvm={kvm} rvm={rvm}"
                    );
                }
            }
        }
    }
}

/// The order, restated independently of the implementation.
fn expected_profile(os: &str, os_isolation: bool, kvm: bool) -> Option<&'static str> {
    // rvm-native is `planned`, so it is never reachable regardless of the host.
    let os_iso_available = matches!(os, "windows" | "macos" | "linux") && os_isolation;
    let wasm_available = matches!(os, "windows" | "macos" | "linux" | "browser");
    let microvm_available = os == "linux" && kvm;
    if os_iso_available {
        Some("os-isolation+wasm")
    } else if wasm_available {
        Some("wasm")
    } else if microvm_available {
        // linux-microvm is `planned` too, so this arm is unreachable today; it
        // documents where the profile sits in the order.
        None
    } else {
        None
    }
}

#[test]
fn detected_host_claims_no_isolation_it_cannot_engage() {
    let detected = HostProfile::detect();
    assert!(!detected.os_isolation, "rvm-host adapters do not exist yet");
    assert!(!detected.rvm_measured_boot);
    assert!(!detected.kvm);
}
