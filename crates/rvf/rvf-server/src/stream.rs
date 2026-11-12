//! Verified streaming delivery (ADR-353): `GET /v1/stream/descriptor`,
//! `GET /v1/stream/body` and the TCP message pair `0x05` / `0x85`.
//!
//! The server holds an existing `.rvf` body in memory, a descriptor generated
//! with `StreamDescriptorBuilder` and signed with a **server-held key read from
//! a configured path** (64 hex characters or 32 raw bytes). The server never
//! generates or stores keys. The body is served as the stage-ordered frame
//! stream with `Range: bytes=start-[end]` for resume.
//!
//! Abuse controls: a semaphore bounds concurrent streams (429 when full);
//! only a single `bytes=a-[b]` range is accepted (multi-range, suffix ranges,
//! malformed or unsatisfiable ranges get 416); one response is capped at
//! `max_range_bytes`; TCP replies are capped at `tcp_max_response`.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{ConnectInfo, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use bytes::Bytes;
use ed25519_dalek::SigningKey;
use rvf_runtime::progressive_stream::build::{descriptor_for, BuildParams};
use rvf_runtime::progressive_stream::plan::StreamPlan;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Bytes of the frame stream handed to the socket per send.
const SEND_PIECE: u64 = 128 << 10;

/// Stream server configuration.
#[derive(Clone, Debug)]
pub struct StreamServerConfig {
    /// The `.rvf` body to serve.
    pub body_path: PathBuf,
    /// Path of the descriptor signing key (hex or raw 32 bytes).
    pub key_path: PathBuf,
    /// Descriptor `chunk_log2`: frame size is `1 << chunk_log2` (`0xFF` = one
    /// frame per segment).
    pub chunk_log2: u8,
    /// Descriptor epoch.
    pub epoch: u64,
    /// Concurrent body streams allowed.
    pub max_concurrent_streams: usize,
    /// Largest response for one range request, bytes.
    pub max_range_bytes: u64,
    /// Largest TCP stream reply payload, bytes.
    pub tcp_max_response: u32,
    /// A stream that cannot hand over one buffered piece for this long (a
    /// client that stopped reading) is aborted and its slot freed; also the
    /// idle timeout for TCP stream connections.
    pub idle_timeout: Duration,
    /// Hard wall-clock limit for one body response.
    pub total_deadline: Duration,
    /// Minimum average delivery rate once `rate_grace` has passed, measured per
    /// `rate_grace` window. A reader below it is cut off and its slot freed.
    pub min_rate_bytes_per_sec: u64,
    /// Window (and start-up allowance) for the minimum-rate rule.
    pub rate_grace: Duration,
    /// Per-client policy; see [`ClientCap`]. Default `PerAddress`.
    pub client_cap: ClientCap,
    /// Streams (HTTP) one client address may hold; TCP connections allow 4x.
    pub max_streams_per_client: usize,
    /// Streams one network prefix (IPv4 /24, IPv6 /64) may hold. Clients behind
    /// one NAT or proxy share a prefix and therefore share this cap.
    pub max_streams_per_prefix: usize,
    /// The last `reserve_slots` of the pool are only given to a prefix that holds
    /// no stream yet, so established clients cannot take every slot.
    pub reserve_slots: usize,
    /// Most streams that may run below `8 x min_rate_bytes_per_sec` at once
    /// (measured per rate window); a stream that would exceed it is cut. Keeps
    /// at least the rest of the pool for fast clients.
    pub max_slow_streams: usize,
}

impl StreamServerConfig {
    /// Defaults: 64 KiB frames, epoch 1, 8 streams, 256 MiB per range, 1 MiB
    /// per TCP reply.
    pub fn new(body_path: PathBuf, key_path: PathBuf) -> Self {
        Self {
            body_path,
            key_path,
            chunk_log2: 16,
            epoch: 1,
            max_concurrent_streams: 8,
            max_range_bytes: 256 << 20,
            tcp_max_response: 1 << 20,
            idle_timeout: Duration::from_secs(30),
            total_deadline: Duration::from_secs(600),
            min_rate_bytes_per_sec: 32 << 10,
            rate_grace: Duration::from_secs(10),
            client_cap: ClientCap::PerAddress,
            max_streams_per_client: 2,
            max_streams_per_prefix: 4,
            reserve_slots: 2,
            max_slow_streams: 4,
        }
    }
}

/// Per-client-address slot accounting, keyed by address and by network prefix
/// (IPv4 /24, IPv6 /64; IPv4-mapped IPv6 is folded into IPv4), so rotating
/// addresses inside one prefix does not multiply a client's share.
#[derive(Default)]
pub struct ClientTable(Mutex<ClientCounts>);

#[derive(Default)]
struct ClientCounts {
    addr: HashMap<IpAddr, usize>,
    prefix: HashMap<[u8; 9], usize>,
}

/// One slot held by a client address; released on drop.
pub struct ClientSlot {
    table: Arc<ClientTable>,
    ip: IpAddr,
    prefix: [u8; 9],
}

fn canonical(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(IpAddr::V6(v6), IpAddr::V4),
        v4 => v4,
    }
}

fn prefix_key(ip: IpAddr) -> [u8; 9] {
    let mut k = [0u8; 9];
    match ip {
        IpAddr::V4(a) => {
            k[0] = 4;
            k[1..4].copy_from_slice(&a.octets()[..3]);
        }
        IpAddr::V6(a) => {
            k[0] = 6;
            k[1..9].copy_from_slice(&a.octets()[..8]);
        }
    }
    k
}

impl ClientTable {
    /// Take a slot unless `ip` or its prefix is at its cap. Returns the slot and
    /// how many slots the prefix held before.
    fn acquire(
        self: &Arc<Self>,
        ip: IpAddr,
        addr_cap: usize,
        prefix_cap: usize,
    ) -> Option<(ClientSlot, usize)> {
        let ip = canonical(ip);
        let prefix = prefix_key(ip);
        let mut m = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let na = m.addr.get(&ip).copied().unwrap_or(0);
        let np = m.prefix.get(&prefix).copied().unwrap_or(0);
        if na >= addr_cap.max(1) || np >= prefix_cap.max(1) {
            return None;
        }
        *m.addr.entry(ip).or_insert(0) += 1;
        *m.prefix.entry(prefix).or_insert(0) += 1;
        Some((
            ClientSlot {
                table: self.clone(),
                ip,
                prefix,
            },
            np,
        ))
    }
}

impl Drop for ClientSlot {
    fn drop(&mut self) {
        let mut m = self.table.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(n) = m.addr.get_mut(&self.ip) {
            *n -= 1;
            if *n == 0 {
                m.addr.remove(&self.ip);
            }
        }
        if let Some(n) = m.prefix.get_mut(&self.prefix) {
            *n -= 1;
            if *n == 0 {
                m.prefix.remove(&self.prefix);
            }
        }
    }
}

/// Everything one HTTP stream holds: pool permit, per-client slot, and (while
/// it is classed as slow) a slot in the slow-stream budget.
pub struct StreamSlot {
    _permit: OwnedSemaphorePermit,
    _client: Option<ClientSlot>,
}

/// Per-client policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClientCap {
    /// Enforce the per-address, per-prefix and reserve rules. A request whose
    /// peer address is unknown (the router was mounted without
    /// `into_make_service_with_connect_info::<SocketAddr>()`) is REFUSED with
    /// 503, so a missing connect-info layer cannot silently disable the limits.
    PerAddress,
    /// Explicit opt-out: no per-client accounting (single trusted client, or
    /// accounting done by a fronting proxy).
    Disabled,
}

/// Loaded stream server state.
pub struct StreamServer {
    body: Arc<Vec<u8>>,
    plan: Arc<StreamPlan>,
    permits: Arc<Semaphore>,
    tcp_permits: Arc<Semaphore>,
    http_clients: Arc<ClientTable>,
    tcp_clients: Arc<ClientTable>,
    slow: Mutex<usize>,
    cfg: StreamServerConfig,
}

fn read_key(path: &std::path::Path) -> Result<SigningKey, String> {
    let raw = std::fs::read(path).map_err(|e| format!("key file {}: {e}", path.display()))?;
    let bytes: [u8; 32] = if raw.len() == 32 {
        raw.try_into().expect("32")
    } else {
        let t = String::from_utf8_lossy(&raw);
        let t = t.trim();
        if t.len() != 64 || !t.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("key file must hold 32 raw bytes or 64 hex characters".into());
        }
        let mut out = [0u8; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u8::from_str_radix(&t[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())?;
        }
        out
    };
    Ok(SigningKey::from_bytes(&bytes))
}

impl StreamServer {
    /// Read the body and key, build the staged descriptor and frame plan.
    pub fn load(cfg: StreamServerConfig) -> Result<Arc<Self>, String> {
        let key = read_key(&cfg.key_path)?;
        let p = BuildParams {
            chunk_log2: cfg.chunk_log2,
            epoch: cfg.epoch,
            ..Default::default()
        };
        // The artifact's file_id: the signed root page's FileIdentity.
        let tail = std::fs::read(&cfg.body_path).map_err(|e| format!("body: {e}"))?;
        let mut file_id = [0u8; 16];
        if tail.len() > 4096 && tail[tail.len() - 4096..][..4] == *b"0MVR" {
            file_id.copy_from_slice(&tail[tail.len() - 4096 + 0xF00..][..16]);
        }
        let (body, bundle) = descriptor_for(&cfg.body_path, file_id, key, &p)
            .map_err(|e| format!("descriptor: {e}"))?;
        let plan = StreamPlan::new(bundle, &body);
        Ok(Arc::new(Self {
            permits: Arc::new(Semaphore::new(cfg.max_concurrent_streams.max(1))),
            tcp_permits: Arc::new(Semaphore::new(cfg.max_concurrent_streams.max(1))),
            http_clients: Arc::default(),
            tcp_clients: Arc::default(),
            slow: Mutex::new(0),
            body: Arc::new(body),
            plan: Arc::new(plan),
            cfg,
        }))
    }

    /// The signed descriptor bytes.
    pub fn descriptor(&self) -> &[u8] {
        &self.plan.bundle().descriptor
    }

    /// Total bytes of the frame stream.
    pub fn stream_len(&self) -> u64 {
        self.plan.total_len()
    }

    /// Configuration.
    pub fn config(&self) -> &StreamServerConfig {
        &self.cfg
    }

    /// Copy `[start, end)` of the frame stream.
    pub fn read_range(&self, start: u64, end: u64) -> Vec<u8> {
        let mut out = Vec::with_capacity((end.saturating_sub(start)).min(1 << 24) as usize);
        self.plan.write_range(&self.body, start, end, &mut |b| {
            out.extend_from_slice(b);
            true
        });
        out
    }

    /// A permit from the TCP pool (separate from HTTP, so one transport's
    /// silent clients cannot starve the other).
    pub fn try_tcp_permit(&self) -> Option<OwnedSemaphorePermit> {
        self.tcp_permits.clone().try_acquire_owned().ok()
    }

    /// Take one HTTP stream slot (held until dropped): a pool permit and, under
    /// `ClientCap::PerAddress`, a per-client slot. With `PerAddress` an unknown
    /// peer is refused (fail closed).
    pub fn try_permit(&self, peer: Option<IpAddr>) -> Result<StreamSlot, StreamRefusal> {
        let pool = self.cfg.max_concurrent_streams.max(1);
        let client = match (self.cfg.client_cap, peer) {
            (ClientCap::Disabled, _) => None,
            (ClientCap::PerAddress, None) => return Err(StreamRefusal::PeerUnknown),
            (ClientCap::PerAddress, Some(ip)) => {
                let (slot, held_by_prefix) = self
                    .http_clients
                    .acquire(
                        ip,
                        self.cfg.max_streams_per_client,
                        self.cfg.max_streams_per_prefix,
                    )
                    .ok_or(StreamRefusal::Busy)?;
                // The reserve: when only `reserve_slots` are free, a prefix that
                // already holds a stream does not get another.
                let free = self.permits.available_permits();
                if free <= self.cfg.reserve_slots.min(pool.saturating_sub(1)) && held_by_prefix > 0
                {
                    return Err(StreamRefusal::Busy);
                }
                Some(slot)
            }
        };
        let permit = self
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| StreamRefusal::Busy)?;
        Ok(StreamSlot {
            _permit: permit,
            _client: client,
        })
    }

    /// Per-client TCP connection slot (stream-enabled servers only).
    pub fn try_tcp_client(&self, peer: IpAddr) -> Option<ClientSlot> {
        if self.cfg.client_cap == ClientCap::Disabled {
            return None;
        }
        self.tcp_clients
            .acquire(
                peer,
                self.cfg.max_streams_per_client.saturating_mul(4),
                self.cfg.max_streams_per_prefix.saturating_mul(4),
            )
            .map(|(s, _)| s)
    }

    /// Reserve one place in the slow-stream budget; `None` when it is full.
    fn try_slow(self: &Arc<Self>) -> Option<SlowToken> {
        let mut n = self.slow.lock().unwrap_or_else(|e| e.into_inner());
        if *n >= self.cfg.max_slow_streams {
            return None;
        }
        *n += 1;
        Some(SlowToken(self.clone()))
    }
}

/// Place in the slow-stream budget, returned on drop.
struct SlowToken(Arc<StreamServer>);

impl Drop for SlowToken {
    fn drop(&mut self) {
        let mut n = self.0.slow.lock().unwrap_or_else(|e| e.into_inner());
        *n = n.saturating_sub(1);
    }
}

/// Why a stream slot was not granted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum StreamRefusal {
    /// Pool, per-address, per-prefix or reserve limit.
    Busy,
    /// `ClientCap::PerAddress` without a peer address (connect-info missing).
    PeerUnknown,
}

/// Router for the stream endpoints.
pub fn router(server: Arc<StreamServer>) -> Router {
    Router::new()
        .route("/v1/stream/descriptor", get(descriptor))
        .route("/v1/stream/body", get(body))
        .with_state(server)
}

async fn descriptor(State(s): State<Arc<StreamServer>>) -> Response {
    let mut h = HeaderMap::new();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    h.insert(
        "x-rvf-descriptor-sha256",
        HeaderValue::from_str(&hex(&s.plan.bundle().descriptor_hash)).expect("hex"),
    );
    (StatusCode::OK, h, s.descriptor().to_vec()).into_response()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Parse `bytes=a-` / `bytes=a-b`. Anything else is unsatisfiable.
fn parse_range(v: &str, total: u64) -> Result<(u64, u64), ()> {
    let spec = v.trim().strip_prefix("bytes=").ok_or(())?;
    if spec.contains(',') {
        return Err(()); // multi-range
    }
    let (a, b) = spec.split_once('-').ok_or(())?;
    if a.is_empty() {
        return Err(()); // suffix range
    }
    let start: u64 = a.parse().map_err(|_| ())?;
    let last = if b.is_empty() {
        total.saturating_sub(1)
    } else {
        b.parse::<u64>()
            .map_err(|_| ())?
            .min(total.saturating_sub(1))
    };
    if start >= total || last < start {
        return Err(());
    }
    Ok((start, last + 1))
}

async fn body(
    State(s): State<Arc<StreamServer>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    let total = s.stream_len();
    let (start, mut end, partial) = match headers.get(header::RANGE) {
        None => (0, total, false),
        Some(v) => match v.to_str().ok().and_then(|v| parse_range(v, total).ok()) {
            Some((a, z)) => (a, z, true),
            None => {
                let mut h = HeaderMap::new();
                h.insert(
                    header::CONTENT_RANGE,
                    HeaderValue::from_str(&format!("bytes */{total}")).expect("hdr"),
                );
                return (StatusCode::RANGE_NOT_SATISFIABLE, h).into_response();
            }
        },
    };
    if partial {
        end = end.min(start + s.cfg.max_range_bytes);
    }
    let permit = match s.try_permit(peer.map(|c| c.0.ip())) {
        Ok(p) => p,
        Err(StreamRefusal::Busy) => {
            return (StatusCode::TOO_MANY_REQUESTS, "too many concurrent streams").into_response();
        }
        Err(StreamRefusal::PeerUnknown) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "peer address unavailable: serve with connect info or set ClientCap::Disabled",
            )
                .into_response();
        }
    };
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::convert::Infallible>>(4);
    let srv = s.clone();
    // The permit belongs to the producer task: it is released when the
    // response is finished, when the client goes away, or when the client stops
    // reading for `idle_timeout` / the response exceeds `total_deadline`.
    tokio::spawn(async move {
        let _permit = permit;
        let started = Instant::now();
        let (idle, deadline) = (srv.cfg.idle_timeout, srv.cfg.total_deadline);
        let (win, floor) = (srv.cfg.rate_grace, srv.cfg.min_rate_bytes_per_sec);
        let (mut win_start, mut win_bytes) = (started, 0u64);
        let mut slow: Option<SlowToken> = None;
        let mut at = start;
        while at < end {
            if started.elapsed() > deadline {
                return;
            }
            let to = (at + SEND_PIECE).min(end);
            let mut piece = Vec::with_capacity((to - at) as usize);
            srv.plan.write_range(&srv.body, at, to, &mut |b| {
                piece.extend_from_slice(b);
                true
            });
            // `send_timeout` parks the task (no polling) until the client
            // reads, and gives up after `idle`.
            if tx.send_timeout(Ok(Bytes::from(piece)), idle).await.is_err() {
                return;
            }
            // Minimum-rate rule: over each `win` window the client must take
            // at least `floor` bytes/s, else the slot is freed.
            win_bytes += to - at;
            let el = win_start.elapsed();
            if el >= win {
                if (win_bytes as u128) * 1_000 < (floor as u128) * el.as_millis() {
                    return;
                }
                // Below 8x the floor the stream is "slow" and draws on the slow budget;
                // when that is full it is cut, leaving the rest of the pool to
                // clients that read quickly.
                if (win_bytes as u128) * 1_000 < (floor as u128) * 8 * el.as_millis() {
                    if slow.is_none() {
                        slow = srv.try_slow();
                        if slow.is_none() {
                            return;
                        }
                    }
                } else {
                    slow = None;
                }
                // Below 8x the floor the stream is "slow" and draws on the slow
                // budget; when that is full it is cut, leaving the rest of the
                // pool to clients that read quickly.
                if (win_bytes as u128) * 1_000 < (floor as u128) * 8 * el.as_millis() {
                    if slow.is_none() {
                        slow = srv.try_slow();
                        if slow.is_none() {
                            return;
                        }
                    }
                } else {
                    slow = None;
                }
                win_start = Instant::now();
                win_bytes = 0;
            }
            at = to;
        }
    });
    let stream =
        futures_util::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|v| (v, rx)) });
    let mut h = HeaderMap::new();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    h.insert(header::ACCEPT_RANGES, HeaderValue::from_static("bytes"));
    h.insert(
        header::CONTENT_LENGTH,
        HeaderValue::from_str(&(end - start).to_string()).expect("len"),
    );
    let status = if partial {
        h.insert(
            header::CONTENT_RANGE,
            HeaderValue::from_str(&format!("bytes {}-{}/{}", start, end - 1, total)).expect("hdr"),
        );
        StatusCode::PARTIAL_CONTENT
    } else {
        StatusCode::OK
    };
    (status, h, Body::from_stream(stream)).into_response()
}

#[cfg(test)]
mod tests {
    use super::parse_range;

    #[test]
    fn range_parser_rejects_abuse() {
        assert_eq!(parse_range("bytes=0-", 100), Ok((0, 100)));
        assert_eq!(parse_range("bytes=10-19", 100), Ok((10, 20)));
        assert_eq!(parse_range("bytes=90-5000", 100), Ok((90, 100)));
        for bad in [
            "bytes=0-1,5-6",
            "bytes=-5",
            "bytes=100-",
            "bytes=20-10",
            "bytes=a-b",
            "items=0-1",
            "bytes=0-1-2",
            "bytes=18446744073709551616-",
        ] {
            assert!(parse_range(bad, 100).is_err(), "{bad}");
        }
    }
}
