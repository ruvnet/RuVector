//! Regression for #1277: the memory-amplification tests must use a measure
//! that exists off Linux. Checks the portable RSS measures track real growth.
#[path = "support/rss.rs"]
mod rss;

/// `ru_maxrss` is a high-water mark that can be inherited across `exec`
/// (cargo's own peak), so allocate well past whatever it already reads.
#[cfg(unix)]
#[test]
fn getrusage_peak_rss_tracks_allocation() {
    let before = rss::rusage_maxrss_kb().expect("getrusage");
    let n = (before as usize + (64 << 10)) * 1024;
    let mut v = vec![0u8; n];
    for i in (0..v.len()).step_by(4096) {
        v[i] = 1;
    }
    std::hint::black_box(&v);
    let after = rss::rusage_maxrss_kb().expect("getrusage");
    assert!(
        after >= before + (48 << 10),
        "peak RSS grew only {} KiB for a touched {} KiB buffer",
        after.saturating_sub(before),
        n / 1024
    );
}

#[test]
fn rss_kb_is_available_on_supported_platforms() {
    if cfg!(unix) {
        assert!(rss::rss_kb().unwrap() > 0);
    }
}
