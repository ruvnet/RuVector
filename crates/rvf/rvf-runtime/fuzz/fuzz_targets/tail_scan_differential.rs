#![no_main]
//! Asserts the runtime's manifest location and rvf-wire's tail scanner agree
//! (invariants I1-I4 in `tests/support/differential.rs`) on arbitrary bytes.

use libfuzzer_sys::fuzz_target;

#[path = "../../tests/support/differential.rs"]
mod differential;

fuzz_target!(|data: &[u8]| {
    if let Err(msg) = differential::check(data) {
        panic!("tail scanners disagree: {msg}");
    }
});
