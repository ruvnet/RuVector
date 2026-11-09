//! Round 2 review: abuse of the stream endpoints.
mod support;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use support::*;

/// F2-05 (fixed): silent clients cannot pin stream slots. A stream whose
/// client stopped reading is aborted after `idle_timeout` and its slot freed;
/// the TCP pool is separate from HTTP; silent TCP sockets are closed.
#[test]
fn f2_05_silent_clients_do_not_pin_stream_slots() {
    let f = fixture(60000, 32, 16, 5);
    let r = start(&f, |c| {
        c.max_concurrent_streams = 2;
        c.idle_timeout = Duration::from_millis(800);
    });
    assert!(
        r.server.stream_len() > 12 << 20,
        "need a body larger than socket buffers"
    );
    let mut held = Vec::new();
    for _ in 0..2 {
        let mut s = connect(r.http);
        s.write_all(b"GET /v1/stream/body HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .unwrap();
        held.push(s); // never read
    }
    std::thread::sleep(Duration::from_millis(300));
    // While HTTP slots are pinned: fail fast (429, not queued) ...
    let t0 = std::time::Instant::now();
    assert_eq!(raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0, 429);
    assert!(
        t0.elapsed() < Duration::from_secs(1),
        "refusal is immediate"
    );
    // ... and TCP 0x05 is a separate pool: it is served.
    let tcp_req = |addr| {
        let mut t = connect(addr);
        let mut p = vec![1u8];
        p.extend_from_slice(&0u64.to_be_bytes());
        p.extend_from_slice(&100u32.to_be_bytes());
        let mut fr = (p.len() as u32).to_be_bytes().to_vec();
        fr.extend_from_slice(&[0x05, 0, 0, 1]);
        fr.extend_from_slice(&p);
        t.write_all(&fr).unwrap();
        let mut h = [0u8; 8];
        t.read_exact(&mut h).unwrap();
        h[4]
    };
    assert_eq!(tcp_req(r.tcp), 0x85, "TCP pool is independent of HTTP");
    // After the idle timeout the abandoned streams release their slots.
    std::thread::sleep(Duration::from_secs(3));
    assert_eq!(raw_get(r.http, "/v1/stream/body", Some("bytes=0-9")).0, 206);
    drop(held);

    // A silent TCP socket is closed by the server after the idle timeout.
    let mut silent = connect(r.tcp);
    silent
        .set_read_timeout(Some(Duration::from_secs(4)))
        .unwrap();
    let mut b = [0u8; 1];
    assert_eq!(
        silent.read(&mut b).unwrap(),
        0,
        "server closed the idle socket"
    );
}

/// Hostile TCP 0x05 frames and hostile HTTP Range headers never kill the
/// server (expected to pass).
#[test]
fn f2_oracle_hostile_inputs_do_not_panic() {
    let f = fixture(2000, 16, 8, 3);
    let r = start(&f, |_| {});
    let mut x = 0x9E3779B97F4A7C15u64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for i in 0..300 {
        let mut t = connect(r.tcp);
        t.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let plen = (rnd() % 20) as usize;
        let mut p: Vec<u8> = (0..plen).map(|_| rnd() as u8).collect();
        if i % 3 == 0 && p.len() == 13 {
            p[0] = (rnd() % 3) as u8;
        }
        let mut fr = (p.len() as u32).to_be_bytes().to_vec();
        fr.extend_from_slice(&[0x05, 1, 2, 3]);
        fr.extend_from_slice(&p);
        let _ = t.write_all(&fr);
        let mut h = [0u8; 8];
        let _ = t.read_exact(&mut h);
    }
    for rg in [
        "bytes=0-18446744073709551615",
        "bytes=18446744073709551615-",
        "bytes=0--1",
        "bytes= 0-1",
        "bytes=0-1 ",
        "bytes=+1-2",
        "bytes=00000000000000000000001-",
        "BYTES=0-1",
    ] {
        let (st, _, _) = raw_get(r.http, "/v1/stream/body", Some(rg));
        eprintln!("{rg:?} -> {st}");
    }
    // TCP offset u64::MAX, max_len u32::MAX
    let mut t = connect(r.tcp);
    let mut p = vec![1u8];
    p.extend_from_slice(&(u64::MAX - 5).to_be_bytes());
    p.extend_from_slice(&u32::MAX.to_be_bytes());
    let mut fr = (p.len() as u32).to_be_bytes().to_vec();
    fr.extend_from_slice(&[0x05, 0, 0, 9]);
    fr.extend_from_slice(&p);
    t.write_all(&fr).unwrap();
    let mut h = [0u8; 8];
    t.read_exact(&mut h).unwrap();
    // server still alive
    assert_eq!(raw_get(r.http, "/v1/stream/descriptor", None).0, 200);
}
