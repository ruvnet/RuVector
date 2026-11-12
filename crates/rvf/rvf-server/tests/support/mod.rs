//! Shared harness: fixture cache, in-process server, malicious mirror.
#![allow(dead_code)]

use rvf_runtime::progressive_stream::build::{build_stream_store, BuildParams, StreamDataset};
use rvf_server::stream::{StreamServer, StreamServerConfig};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const KEY: [u8; 32] = [0xA7; 32];

pub struct Fixture {
    pub dir: tempfile::TempDir,
    pub rvf: PathBuf,
    pub keyfile: PathBuf,
    pub ds: StreamDataset,
}

pub fn public_key() -> [u8; 32] {
    ed25519_dalek::SigningKey::from_bytes(&KEY)
        .verifying_key()
        .to_bytes()
}

/// Build (or reuse) a stream-ready store. Stores are cached under
/// `OFER_STREAM_CACHE`, or by default under `CARGO_TARGET_TMPDIR`, keyed by
/// (n, dim, clusters, seed), so only the first run pays the build.
pub fn fixture(n: usize, dim: usize, clusters: usize, seed: u64) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let ds = StreamDataset::clustered(n, dim, clusters, 0.4, seed);
    let cache = std::env::var_os("OFER_STREAM_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ofer-stream-cache"));
    let rvf = {
        let p = cache.join(format!("{n}x{dim}c{clusters}s{seed}.rvf"));
        if !p.exists() {
            std::fs::create_dir_all(&cache).unwrap();
            // Unique temp name: parallel tests may build the same fixture.
            let tmp = cache.join(format!(
                "{n}x{dim}c{clusters}s{seed}.{}.{:?}.tmp",
                std::process::id(),
                std::thread::current().id()
            ));
            build_stream_store(&tmp, &ds, &BuildParams::default(), &[9u8; 32]).unwrap();
            std::fs::rename(&tmp, &p).unwrap();
        }
        p
    };
    let keyfile = dir.path().join("desc.key");
    std::fs::write(
        &keyfile,
        KEY.iter().map(|b| format!("{b:02x}")).collect::<String>(),
    )
    .unwrap();
    Fixture {
        dir,
        rvf,
        keyfile,
        ds,
    }
}

pub struct Running {
    pub http: SocketAddr,
    pub tcp: SocketAddr,
    pub server: Arc<StreamServer>,
    _rt: tokio::runtime::Runtime,
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

pub fn start(f: &Fixture, tweak: impl FnOnce(&mut StreamServerConfig)) -> Running {
    let mut cfg = StreamServerConfig::new(f.rvf.clone(), f.keyfile.clone());
    cfg.chunk_log2 = 16;
    tweak(&mut cfg);
    let server = StreamServer::load(cfg).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let http = listener.local_addr().unwrap();
    let app = rvf_server::stream::router(server.clone());
    rt.spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    let tcp_port = free_port();
    let tcp: SocketAddr = format!("127.0.0.1:{tcp_port}").parse().unwrap();
    let store = rvf_server::open_or_create_store(&rvf_server::ServerConfig {
        data_path: f.dir.path().join("live.rvf"),
        dimension: 4,
        ..Default::default()
    })
    .unwrap();
    let srv = server.clone();
    rt.spawn(async move {
        let _ = rvf_server::tcp::serve_tcp_with_stream(
            &format!("127.0.0.1:{tcp_port}"),
            store,
            Some(srv),
        )
        .await;
    });
    std::thread::sleep(std::time::Duration::from_millis(100));
    Running {
        http,
        tcp,
        server,
        _rt: rt,
    }
}

/// Connect with explicit read and write timeouts, so a wedged server fails a
/// test instead of hanging it.
pub fn connect(addr: SocketAddr) -> TcpStream {
    let s = TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(10)).unwrap();
    s.set_read_timeout(Some(std::time::Duration::from_secs(30)))
        .unwrap();
    s.set_write_timeout(Some(std::time::Duration::from_secs(30)))
        .unwrap();
    s
}

/// Raw HTTP/1.1 request returning (status, headers-lowercased, body bytes read).
pub fn raw_get(
    addr: SocketAddr,
    path: &str,
    range: Option<&str>,
) -> (u16, Vec<(String, String)>, Vec<u8>) {
    let mut s = connect(addr);
    let r = range.map(|r| format!("Range: {r}\r\n")).unwrap_or_default();
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: x\r\n{r}Connection: close\r\n\r\n"
    )
    .unwrap();
    let mut all = Vec::new();
    s.read_to_end(&mut all).unwrap();
    let end = all.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4;
    let head = String::from_utf8_lossy(&all[..end]).to_string();
    let mut lines = head.split("\r\n");
    let status = lines
        .next()
        .unwrap()
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    let hdrs = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    (status, hdrs, all[end..].to_vec())
}

/// How a malicious mirror misbehaves on the HTTP layer.
#[derive(Clone, Copy, Debug)]
pub enum Lie {
    /// Correct headers, possibly mutated body bytes.
    None,
    /// Content-Length larger than what is sent (then closes).
    LengthTooBig,
    /// Content-Length smaller than the real range (silently truncates).
    LengthTooSmall,
    /// Content-Range starts at 0 though the request asked for more.
    RangeStartsAtZero,
    /// Ignores Range and answers 200 with the whole stream.
    IgnoresRange,
    /// Claims a different total in Content-Range.
    WrongTotal,
}

/// A one-thread HTTP mirror serving `stream` (already mutated) with `lie`.
pub fn mirror(stream: Vec<u8>, lie: Lie, cut_first: Option<usize>) -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        let mut conn_no = 0usize;
        for conn in l.incoming() {
            let Ok(mut s) = conn else { continue };
            let mut req = Vec::new();
            let mut tmp = [0u8; 1024];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                match s.read(&mut tmp) {
                    Ok(n) if n > 0 => req.extend_from_slice(&tmp[..n]),
                    _ => break,
                }
            }
            let req = String::from_utf8_lossy(&req).to_string();
            let off: usize = req
                .lines()
                .find_map(|l| l.strip_prefix("Range: bytes="))
                .and_then(|r| r.split('-').next())
                .and_then(|a| a.trim().parse().ok())
                .unwrap_or(0);
            conn_no += 1;
            let total = stream.len();
            let real = &stream[off.min(total)..];
            let (status, cr, cl, body): (&str, String, usize, &[u8]) = match lie {
                Lie::None => (
                    "206 Partial Content",
                    format!("bytes {}-{}/{}", off, total - 1, total),
                    real.len(),
                    real,
                ),
                Lie::LengthTooBig => (
                    "206 Partial Content",
                    format!("bytes {}-{}/{}", off, total - 1, total),
                    real.len() + 5000,
                    real,
                ),
                Lie::LengthTooSmall => {
                    let cut = (real.len() / 2).max(1);
                    (
                        "206 Partial Content",
                        format!("bytes {}-{}/{}", off, off + cut - 1, total),
                        cut,
                        &real[..cut],
                    )
                }
                Lie::RangeStartsAtZero => (
                    "206 Partial Content",
                    format!("bytes 0-{}/{}", total - 1, total),
                    total,
                    &stream[..],
                ),
                Lie::IgnoresRange => ("200 OK", String::new(), total, &stream[..]),
                Lie::WrongTotal => (
                    "206 Partial Content",
                    format!("bytes {}-{}/{}", off, total - 1, total + 7 * conn_no),
                    real.len(),
                    real,
                ),
            };
            let mut head = format!("HTTP/1.1 {status}\r\nContent-Length: {cl}\r\n");
            if !cr.is_empty() {
                head.push_str(&format!("Content-Range: {cr}\r\n"));
            }
            head.push_str("Connection: close\r\n\r\n");
            let _ = s.write_all(head.as_bytes());
            let body = match (cut_first, conn_no) {
                (Some(c), 1) => &body[..c.min(body.len())],
                _ => body,
            };
            let _ = s.write_all(body);
        }
    });
    addr
}

pub fn read_descriptor(addr: SocketAddr) -> Vec<u8> {
    raw_get(addr, "/v1/stream/descriptor", None).2
}

pub fn exists(p: &Path) -> bool {
    p.exists()
}
