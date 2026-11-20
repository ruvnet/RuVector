use serde_json::{Value, json};

#[cfg(target_os = "linux")]
pub fn sample() -> Value {
    fn field(path: &str, name: &str) -> Option<u64> {
        std::fs::read_to_string(path)
            .ok()?
            .lines()
            .find_map(|line| {
                let mut words = line.split_whitespace();
                (words.next()? == name)
                    .then(|| words.next()?.parse().ok())
                    .flatten()
            })
    }
    json!({"os":"linux", "process_read_bytes":field("/proc/self/io","read_bytes:"),
        "peak_rss_bytes":field("/proc/self/status","VmHWM:").map(|v|v*1024),
        "process_swap_bytes":field("/proc/self/status","VmSwap:").map(|v|v*1024),
        "host_swapouts_pages":field("/proc/vmstat","pswpout")})
}

#[cfg(target_os = "macos")]
pub fn sample() -> Value {
    let pid = std::process::id().to_string();
    let helper = std::env::var_os("RUVLLM_DARWIN_METRICS").expect("missing Darwin metrics binary");
    let out = std::process::Command::new(helper)
        .arg(pid)
        .output()
        .expect("start Darwin metrics");
    assert!(out.status.success(), "Darwin process metrics unavailable");
    serde_json::from_slice(&out.stdout).expect("invalid Darwin metrics")
}
