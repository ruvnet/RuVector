//! Blocking transports for `rvf_runtime::progressive_stream::client`:
//! HTTP/1.1 `Range` and the TCP `0x05`/`0x85` pair. std only.
//!
//! The HTTP transport treats the server as untrusted for framing: it
//! requires a status of 206 (or 200 at offset 0), a `Content-Range` that
//! starts exactly where it asked, a `Content-Length` consistent with it and
//! with a stable total, and no chunked encoding. A response that disagrees is
//! an I/O error (the client suspends and retries), never fed to the verifier.
//! Whatever passes still goes through full chunk verification.

use rvf_runtime::progressive_stream::client::{FetchEnd, Transport};
use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

fn lie(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

/// `GET <path>` with `Range: bytes=<offset>-`.
pub struct HttpTransport {
    addr: String,
    path: String,
    timeout: Duration,
    total: Option<u64>,
}

impl HttpTransport {
    /// A transport for `host:port` and the body path (`/v1/stream/body`).
    pub fn new(addr: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            addr: addr.into(),
            path: path.into(),
            timeout: Duration::from_secs(10),
            total: None,
        }
    }
}

struct Head {
    status: u16,
    content_length: Option<u64>,
    content_range: Option<(u64, u64, u64)>,
    chunked: bool,
}

fn read_head(s: &mut TcpStream) -> io::Result<(Head, Vec<u8>)> {
    let mut buf = Vec::with_capacity(1024);
    let mut tmp = [0u8; 1024];
    let end = loop {
        let n = s.read(&mut tmp)?;
        if n == 0 {
            return Err(lie("connection closed in headers"));
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
        if buf.len() > 16 * 1024 {
            return Err(lie("oversized headers"));
        }
    };
    let text = String::from_utf8_lossy(&buf[..end]).to_string();
    let mut lines = text.split("\r\n");
    let status = lines
        .next()
        .and_then(|l| l.split(' ').nth(1))
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| lie("bad status line"))?;
    let mut h = Head {
        status,
        content_length: None,
        content_range: None,
        chunked: false,
    };
    for l in lines {
        let Some((k, v)) = l.split_once(':') else {
            continue;
        };
        let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
        match k.as_str() {
            "content-length" => h.content_length = v.parse().ok(),
            "transfer-encoding" => h.chunked = v.to_ascii_lowercase().contains("chunked"),
            "content-range" => {
                // bytes a-b/total
                let r = v.strip_prefix("bytes ").unwrap_or("");
                if let Some((ab, t)) = r.split_once('/') {
                    if let Some((a, b)) = ab.split_once('-') {
                        if let (Ok(a), Ok(b), Ok(t)) = (a.parse(), b.parse(), t.parse()) {
                            h.content_range = Some((a, b, t));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let rest = buf[end..].to_vec();
    Ok((h, rest))
}

impl Transport for HttpTransport {
    fn fetch(&mut self, offset: u64, sink: &mut dyn FnMut(&[u8]) -> bool) -> io::Result<FetchEnd> {
        let sa = self
            .addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| lie("no address"))?;
        let mut s = TcpStream::connect_timeout(&sa, self.timeout)?;
        s.set_read_timeout(Some(self.timeout))?;
        s.set_write_timeout(Some(self.timeout))?;
        s.set_nodelay(true)?;
        let req = format!(
            "GET {} HTTP/1.1\r\nHost: rvf\r\nRange: bytes={}-\r\nConnection: close\r\n\r\n",
            self.path, offset
        );
        s.write_all(req.as_bytes())?;
        let (h, mut rest) = read_head(&mut s)?;
        if h.chunked {
            return Err(lie("chunked transfer encoding refused"));
        }
        let clen = h
            .content_length
            .ok_or_else(|| lie("missing Content-Length"))?;
        match (h.status, h.content_range) {
            (206, Some((a, b, total))) => {
                if a != offset || b < a || clen != b - a + 1 || b >= total {
                    return Err(lie("Content-Range disagrees with the request"));
                }
                if *self.total.get_or_insert(total) != total {
                    return Err(lie("stream total changed between responses"));
                }
            }
            (200, _) if offset == 0 => {
                if *self.total.get_or_insert(clen) != clen {
                    return Err(lie("stream total changed between responses"));
                }
            }
            _ => return Err(lie("unexpected status for a ranged request")),
        }
        let mut left = clen;
        let mut buf = vec![0u8; 64 * 1024];
        if !rest.is_empty() {
            rest.truncate(left.min(rest.len() as u64) as usize);
            left -= rest.len() as u64;
            if !sink(&rest) {
                return Ok(FetchEnd::Aborted);
            }
        }
        while left > 0 {
            let want = left.min(buf.len() as u64) as usize;
            let n = s.read(&mut buf[..want])?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "short body (Content-Length lie or dropped connection)",
                ));
            }
            left -= n as u64;
            if !sink(&buf[..n]) {
                return Ok(FetchEnd::Aborted);
            }
        }
        Ok(FetchEnd::Eof)
    }
}

/// The TCP `0x05`/`0x85` transport (one connection, request/response).
pub struct TcpTransport {
    addr: String,
    /// Bytes requested per message.
    pub request_len: u32,
    timeout: Duration,
}

impl TcpTransport {
    /// A transport for `host:port`.
    pub fn new(addr: impl Into<String>, request_len: u32) -> Self {
        Self {
            addr: addr.into(),
            request_len,
            timeout: Duration::from_secs(10),
        }
    }

    /// Fetch the signed descriptor (`kind 0`).
    pub fn descriptor(&self) -> io::Result<Vec<u8>> {
        let mut s = self.connect()?;
        let (_, body) = Self::call(&mut s, 0, 0, 0)?;
        Ok(body)
    }

    fn connect(&self) -> io::Result<TcpStream> {
        let sa = self
            .addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| lie("no address"))?;
        let s = TcpStream::connect_timeout(&sa, self.timeout)?;
        s.set_read_timeout(Some(self.timeout))?;
        s.set_write_timeout(Some(self.timeout))?;
        s.set_nodelay(true)?;
        Ok(s)
    }

    fn call(s: &mut TcpStream, kind: u8, offset: u64, max_len: u32) -> io::Result<(u64, Vec<u8>)> {
        let mut p = vec![kind];
        p.extend_from_slice(&offset.to_be_bytes());
        p.extend_from_slice(&max_len.to_be_bytes());
        let mut f = Vec::new();
        f.extend_from_slice(&(p.len() as u32).to_be_bytes());
        f.extend_from_slice(&[0x05, 0, 0, 1]);
        f.extend_from_slice(&p);
        s.write_all(&f)?;
        let mut h = [0u8; 8];
        s.read_exact(&mut h)?;
        let len = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as usize;
        if h[4] != 0x85 || !(8..=32 * 1024 * 1024).contains(&len) {
            return Err(lie("bad stream reply"));
        }
        let mut body = vec![0u8; len];
        s.read_exact(&mut body)?;
        let total = u64::from_be_bytes(body[..8].try_into().expect("8"));
        Ok((total, body.split_off(8)))
    }
}

impl Transport for TcpTransport {
    fn fetch(
        &mut self,
        mut offset: u64,
        sink: &mut dyn FnMut(&[u8]) -> bool,
    ) -> io::Result<FetchEnd> {
        let mut s = self.connect()?;
        loop {
            let (total, bytes) = Self::call(&mut s, 1, offset, self.request_len)?;
            if bytes.is_empty() || offset + bytes.len() as u64 > total {
                return Err(lie("reply outside the stream"));
            }
            offset += bytes.len() as u64;
            if !sink(&bytes) {
                return Ok(FetchEnd::Aborted);
            }
            if offset >= total {
                return Ok(FetchEnd::Eof);
            }
        }
    }
}
