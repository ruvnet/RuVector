//! Argument-validation tests for the C-ABI exports.
//!
//! Every case here used to panic (a WASM `unreachable` trap in production).

extern crate std;

use super::*;

const INVALID: i32 = -2;

// The registry is a process-wide static; serialize tests that touch it.
static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_handle(f: impl FnOnce(i32)) {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let h = rvf_solver_create();
    assert!(h > 0, "create failed");
    f(h);
    assert_eq!(rvf_solver_destroy(h), 0);
}

#[test]
fn train_rejects_non_positive_count() {
    with_handle(|h| {
        assert_eq!(rvf_solver_train(h, -1, 1, 10, 1, 0), INVALID);
        assert_eq!(rvf_solver_train(h, 0, 1, 10, 1, 0), INVALID);
        assert_eq!(rvf_solver_train(h, i32::MIN, 1, 10, 1, 0), INVALID);
        // A valid call still works afterwards.
        assert!(rvf_solver_train(h, 3, 1, 10, 1, 0) >= 0);
    });
}

#[test]
fn train_clamps_difficulty_to_documented_range() {
    with_handle(|h| {
        // -1 used to wrap to difficulty 255 and 256 to 0; both are clamped now.
        assert!(rvf_solver_train(h, 2, -1, 256, 1, 0) >= 0);
        assert!(rvf_solver_train(h, 2, 50, 3, 1, 0) >= 0);
    });
}

#[test]
fn acceptance_rejects_degenerate_configs() {
    with_handle(|h| {
        assert_eq!(rvf_solver_acceptance(h, 5, 5, 0, 100, 1, 0), INVALID);
        assert_eq!(rvf_solver_acceptance(h, 5, 5, -1, 100, 1, 0), INVALID);
        assert_eq!(rvf_solver_acceptance(h, 0, 5, 1, 100, 1, 0), INVALID);
        assert_eq!(rvf_solver_acceptance(h, 5, -1, 1, 100, 1, 0), INVALID);
        assert_eq!(rvf_solver_acceptance(h, 5, 5, 1, 0, 1, 0), INVALID);
        assert!(rvf_solver_acceptance(h, 3, 3, 1, 100, 1, 0) >= 0);
    });
}

#[test]
fn acceptance_survives_max_seed() {
    // training_seed = seed + 1 = u64::MAX, then + cycle * 10_000 overflowed.
    with_handle(|h| {
        assert!(rvf_solver_acceptance(h, 2, 2, 2, 50, -2, -1) >= 0);
    });
}

#[test]
fn extreme_handles_are_invalid_not_overflow() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(rvf_solver_train(i32::MIN, 1, 1, 10, 1, 0), -1);
    assert_eq!(rvf_solver_result_len(i32::MIN), -1);
    assert_eq!(rvf_solver_destroy(i32::MIN), -1);
    assert_eq!(rvf_solver_destroy(0), -1);
}

#[test]
fn zero_training_per_cycle_is_allowed() {
    with_handle(|h| {
        assert!(rvf_solver_acceptance(h, 2, 0, 1, 50, 1, 0) >= 0);
        let len = rvf_solver_result_len(h);
        assert!(len > 0);
        let json = &registry().get(h).unwrap().last_result_json;
        let text = core::str::from_utf8(json).unwrap();
        assert!(
            !text.contains("null"),
            "no NaN may leak into the manifest: {text}"
        );
    });
}
