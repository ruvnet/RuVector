//! Portable resident-memory measure for the memory-amplification tests
//! (issue #1277). Not every OS has `/proc`.
//!
//! * Linux: current RSS from `/proc/self/statm` (exact, can fall again).
//! * macOS: current resident size from Mach `task_info(MACH_TASK_BASIC_INFO)`.
//! * Other unix (BSD): peak RSS from `getrusage(RUSAGE_SELF)`. This is a
//!   high-water mark that is also inherited across `exec` (it can already read
//!   ~100 MB at process start), so a before/after delta can only UNDER-report
//!   growth; it never over-reports, so an upper-bound assertion cannot
//!   spuriously fail (but is weaker than on Linux/macOS).
//! * Elsewhere: `None`; callers must skip with a message instead of passing
//!   silently.
#![allow(dead_code)]

/// Peak RSS in KiB via `getrusage` (unix only).
#[cfg(unix)]
pub fn rusage_maxrss_kb() -> Option<u64> {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `ru` is a valid, writable rusage.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return None;
    }
    let raw = ru.ru_maxrss as u64;
    // macOS reports bytes; Linux and the BSDs report KiB.
    Some(if cfg!(target_vendor = "apple") {
        raw / 1024
    } else {
        raw
    })
}

#[cfg(not(unix))]
pub fn rusage_maxrss_kb() -> Option<u64> {
    None
}

/// Current resident size in KiB from Mach `task_info`.
#[cfg(target_vendor = "apple")]
pub fn mach_resident_kb() -> Option<u64> {
    let mut info: libc::mach_task_basic_info = unsafe { std::mem::zeroed() };
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    // SAFETY: `info`/`count` are valid for MACH_TASK_BASIC_INFO.
    let kr = unsafe {
        libc::task_info(
            libc::mach_task_self(),
            libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as libc::task_info_t,
            &mut count,
        )
    };
    if kr == 0 {
        Some(info.resident_size / 1024)
    } else {
        None
    }
}

/// Resident set size in KiB, or `None` if this platform has no measure.
pub fn rss_kb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        // statm field 1 is resident pages; assume 4 KiB pages like the
        // original tests did, falling back to getrusage if unreadable.
        if let Ok(s) = std::fs::read_to_string("/proc/self/statm") {
            if let Some(p) = s
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse::<u64>().ok())
            {
                return Some(p * 4);
            }
        }
    }
    #[cfg(target_vendor = "apple")]
    {
        if let Some(v) = mach_resident_kb() {
            return Some(v);
        }
    }
    rusage_maxrss_kb()
}
