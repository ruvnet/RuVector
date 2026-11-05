//! Round 3: slot pinning after idle_timeout / total_deadline.
mod support;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};
use support::*;

fn open_stream(addr: std::net::SocketAddr) -> TcpStream {
    let mut s = connect(addr);
    s.write_all(b"GET /v1/stream/body HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .unwrap();
    s
}

/// Silent clients are now released after idle_timeout (expected to pass).
#[test]
fn f3_oracle_silent_clients_are_released() {
    let f = fixture(60000, 32, 16, 5);
    let r = start(&f, |c| {
        c.max_concurrent_streams = 2;
        c.idle_timeout = Duration::from_secs(2);
    });
    let _a = open_stream(r.http);
    let _b = open_stream(r.http);
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0, 429);
    std::thread::sleep(Duration::from_secs(5));
    assert_eq!(
        raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0,
        206,
        "slots freed"
    );
}

/// F3-05 (fixed): a reader that trickles just fast enough to dodge idle_timeout
/// is cut off by the minimum-rate rule and its slot freed.
#[test]
fn f3_05_slow_reader_is_cut_by_min_rate() {
    let f = fixture(60000, 32, 16, 5);
    let r = start(&f, |c| {
        c.max_concurrent_streams = 2;
        c.max_streams_per_client = 2;
        c.idle_timeout = Duration::from_secs(2);
        c.min_rate_bytes_per_sec = 256 << 10;
        c.rate_grace = Duration::from_secs(2);
    });
    let mut readers = Vec::new();
    for _ in 0..2 {
        let s = open_stream(r.http);
        s.set_read_timeout(Some(Duration::from_millis(50))).unwrap();
        readers.push(s);
    }
    let t0 = Instant::now();
    let mut buf = [0u8; 4096];
    let mut st = 0;
    while t0.elapsed() < Duration::from_secs(14) {
        for s in &mut readers {
            let _ = s.read(&mut buf);
        }
        std::thread::sleep(Duration::from_millis(100));
        if t0.elapsed() > Duration::from_secs(3) {
            st = raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0;
            if st == 206 {
                break;
            }
        }
    }
    assert_eq!(st, 206, "slow readers must lose their slots");
}

/// F3-05 (fixed): one client address cannot take the whole pool.
#[test]
fn f3_05_per_client_cap() {
    let f = fixture(60000, 32, 16, 5);
    let r = start(&f, |c| {
        c.max_concurrent_streams = 8;
        c.max_streams_per_client = 2;
        c.idle_timeout = Duration::from_secs(30);
    });
    let _a = open_stream(r.http);
    let _b = open_stream(r.http);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(
        raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0,
        429,
        "third stream from the same address is refused although the pool has room"
    );
}

/// F4-06 (fixed): addresses in one /24 share a prefix cap, so two source
/// addresses cannot pin the pool; a client from another prefix still gets in.
#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "binds 127.0.0.2/127.0.0.3, which only Linux routes to loopback by default")]
fn f4_06_two_addresses_cannot_pin_the_pool() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpSocket;
    let f = fixture(60000, 32, 16, 5);
    let mut cfg = rvf_server::stream::StreamServerConfig::new(f.rvf.clone(), f.keyfile.clone());
    cfg.chunk_log2 = 16;
    let server = rvf_server::stream::StreamServer::load(cfg).unwrap();
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let l = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
        let port = l.local_addr().unwrap().port();
        let app = rvf_server::stream::router(server.clone());
        tokio::spawn(async move {
            axum::serve(
                l,
                app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
            )
            .await
            .unwrap();
        });
        let target: std::net::SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
        let get = |src: &'static str, range: &'static str| async move {
            let s = TcpSocket::new_v4().unwrap();
            s.bind(format!("{src}:0").parse().unwrap()).unwrap();
            let mut c = s.connect(target).await.unwrap();
            let req = format!(
                "GET /v1/stream/body HTTP/1.1\r\nHost: x\r\n{range}Connection: close\r\n\r\n"
            );
            c.write_all(req.as_bytes()).await.unwrap();
            let mut b = vec![0u8; 64];
            let n = tokio::time::timeout(std::time::Duration::from_secs(20), c.read(&mut b))
                .await
                .expect("read timed out")
                .unwrap();
            (String::from_utf8_lossy(&b[..n.min(20)]).to_string(), c)
        };
        let mut held = Vec::new();
        let mut statuses = Vec::new();
        for src in ["127.0.0.2", "127.0.0.3"] {
            for _ in 0..4 {
                let (st, c) = get(src, "").await;
                statuses.push(st);
                held.push(c);
            }
        }
        let ok = statuses.iter().filter(|s| s.contains(" 200")).count();
        assert!(ok <= 4, "one /24 took {ok} slots: {statuses:?}");
        let (other, _c) = get("127.1.0.9", "Range: bytes=0-9\r\n").await;
        assert!(
            other.contains("206"),
            "a new prefix is locked out: {other:?}"
        );
    });
}

/// F4-06 (fixed): a router mounted without connect info fails closed.
#[test]
fn f4_06_missing_connect_info_fails_closed() {
    let f = fixture(60000, 32, 16, 5);
    let server = {
        let mut cfg = rvf_server::stream::StreamServerConfig::new(f.rvf.clone(), f.keyfile.clone());
        cfg.chunk_log2 = 16;
        rvf_server::stream::StreamServer::load(cfg).unwrap()
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let l = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let addr = l.local_addr().unwrap();
    let app = rvf_server::stream::router(server);
    rt.spawn(async move {
        axum::serve(l, app).await.unwrap();
    });
    assert_eq!(
        raw_get(addr, "/v1/stream/body", Some("bytes=0-9")).0,
        503,
        "no peer address must mean no stream, not no limit"
    );
}
