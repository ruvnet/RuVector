//! Phi-scaled compression budgets.

use ruvector_field::model::Shell;
use ruvector_field::model::shell::PHI;

#[test]
fn budget_ratios_are_phi_powers() {
    let base = 1024.0_f32;
    let e = Shell::Event.budget(base);
    let p = Shell::Pattern.budget(base);
    let c = Shell::Concept.budget(base);
    let r = Shell::Principle.budget(base);

    assert!((e / base - 1.0).abs() < 1e-3);
    assert!((p / base - 1.0 / PHI).abs() < 1e-3);
    assert!((c / base - 1.0 / (PHI * PHI)).abs() < 1e-3);
    assert!((r / base - 1.0 / (PHI * PHI * PHI)).abs() < 1e-3);
}
