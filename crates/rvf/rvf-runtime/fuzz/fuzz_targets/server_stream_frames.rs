#![no_main]
//! Fuzz of the rvf-server stream surface over real sockets: arbitrary bytes as
//! a TCP message stream (incl. 0x05 stream requests), as an HTTP request, and
//! arbitrary (start,end) into the frame plan. A panic hook aborts so a task
//! panic (normally swallowed by tokio) becomes a crash.

use libfuzzer_sys::fuzz_target;
use rvf_runtime::progressive_stream::build::{build_stream_store, BuildParams, StreamDataset};
use rvf_server::stream::{StreamServer, StreamServerConfig};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::OnceLock;
use std::time::Duration;

struct Srv {
    http: SocketAddr,
    tcp: SocketAddr,
    server: std::sync::Arc<StreamServer>,
    _rt: tokio::runtime::Runtime,
    _dir: tempfile::TempDir,
}

fn srv() -> &'static Srv {
    static S: OnceLock<Srv> = OnceLock::new();
    S.get_or_init(|| {
        std::panic::set_hook(Box::new(|i| {
            eprintln!("PANIC: {i}");
            std::process::abort();
        }));
        let dir = tempfile::tempdir().unwrap();
        let rvf = dir.path().join("s.rvf");
        let ds = StreamDataset::clustered(2600, 8, 8, 0.4, 5);
        build_stream_store(&rvf, &ds, &BuildParams { vec_batch: 300, ..Default::default() }, &[0x11; 32]).unwrap();
        let key = dir.path().join("k");
        std::fs::write(&key, "a7".repeat(32)).unwrap();
        let mut cfg = StreamServerConfig::new(rvf, key);
        cfg.chunk_log2 = 12;
        cfg.max_concurrent_streams = 64;
        let server = StreamServer::load(cfg).unwrap();
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        let l = rt.block_on(tokio::net::TcpListener::bind("127.0.0.1:0")).unwrap();
        let http = l.local_addr().unwrap();
        let app = rvf_server::stream::router(server.clone());
        rt.spawn(async move { axum::serve(l, app).await.unwrap(); });
        let tl = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let tcp = tl.local_addr().unwrap();
        drop(tl);
        let store = rvf_server::open_or_create_store(&rvf_server::ServerConfig {
            data_path: dir.path().join("live.rvf"),
            dimension: 4,
            ..Default::default()
        }).unwrap();
        let s2 = server.clone();
        let addr = tcp.to_string();
        rt.spawn(async move { let _ = rvf_server::tcp::serve_tcp_with_stream(&addr, store, Some(s2)).await; });
        std::thread::sleep(Duration::from_millis(200));
        Srv { http, tcp, server, _rt: rt, _dir: dir }
    })
}

fn poke(addr: SocketAddr, bytes: &[u8]) {
    if let Ok(mut s) = TcpStream::connect(addr) {
        let _ = s.set_read_timeout(Some(Duration::from_millis(30)));
        let _ = s.set_write_timeout(Some(Duration::from_millis(100)));
        let _ = s.write_all(bytes);
        let mut b = [0u8; 4096];
        let _ = s.read(&mut b);
    }
}

fuzz_target!(|data: &[u8]| {
    let s = srv();
    if data.len() >= 16 {
        let a = u64::from_le_bytes(data[..8].try_into().unwrap());
        let b = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let (a, b) = (a % (s.server.stream_len() + 8), b % (s.server.stream_len() + 8));
        let _ = s.server.read_range(a, b.min(a + (1 << 20)));
    }
    match data.first().copied().unwrap_or(0) % 3 {
        0 => {
            // framed 0x05 request with fuzzed payload
            let p = &data[1.min(data.len())..data.len().min(40)];
            let mut f = (p.len() as u32).to_be_bytes().to_vec();
            f.extend_from_slice(&[0x05, 0, 0, 1]);
            f.extend_from_slice(p);
            poke(s.tcp, &f);
        }
        1 => poke(s.tcp, &data[1.min(data.len())..]),
        _ => {
            let r = String::from_utf8_lossy(&data[1.min(data.len())..data.len().min(200)]).replace(['\r', '\n'], "");
            let req = format!("GET /v1/stream/body HTTP/1.1\r\nHost: x\r\nRange: {r}\r\nConnection: close\r\n\r\n");
            poke(s.http, req.as_bytes());
        }
    }
});
