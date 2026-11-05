//! Small shared helpers: bounded parallel map, time-boxed subprocesses,
//! timestamps.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Map `f` over `items` on `workers` threads, preserving order.
pub fn par_map<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(
    items: &[T],
    workers: usize,
    f: F,
) -> Vec<R> {
    let next = Mutex::new(0usize);
    let out: Mutex<Vec<Option<R>>> = Mutex::new((0..items.len()).map(|_| None).collect());
    std::thread::scope(|s| {
        for _ in 0..workers.max(1).min(items.len().max(1)) {
            s.spawn(|| loop {
                let i = {
                    let mut n = next.lock().expect("par_map index");
                    if *n >= items.len() {
                        break;
                    }
                    *n += 1;
                    *n - 1
                };
                let r = f(&items[i]);
                out.lock().expect("par_map out")[i] = Some(r);
            });
        }
    });
    out.into_inner()
        .expect("par_map out")
        .into_iter()
        .map(|r| r.expect("every index filled"))
        .collect()
}

pub struct CmdOutput {
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

/// Run a command with a wall-clock limit; output is captured, never echoed.
pub fn run_timeout(cmd: &str, args: &[&str], cwd: &Path, timeout: Duration) -> Result<CmdOutput> {
    let mut child = Command::new(cmd)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {cmd}"))?;
    let mut stdout = child.stdout.take().context("stdout")?;
    let mut stderr = child.stderr.take().context("stderr")?;
    let t_out = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = std::io::Read::read_to_string(&mut stdout, &mut s);
        s
    });
    let t_err = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = std::io::Read::read_to_string(&mut stderr, &mut s);
        s
    });
    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        if let Some(st) = child.try_wait()? {
            break st.code();
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            timed_out = true;
            break None;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    Ok(CmdOutput {
        status,
        stdout: t_out.join().unwrap_or_default(),
        stderr: t_err.join().unwrap_or_default(),
        timed_out,
    })
}

/// Truncate to `n` chars on a char boundary.
pub fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut t: String = s.chars().take(n).collect();
        t.push('…');
        t
    }
}

/// `owner/name` slug as a filesystem-safe key.
pub fn slug_key(slug: &str) -> String {
    slug.to_ascii_lowercase().replace('/', "__")
}
