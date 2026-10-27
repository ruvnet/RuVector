//! Resumable stream client over a [`Transport`].
//!
//! The client owns a [`StreamIngest`]. It fetches the frame stream from the
//! ingest's current offset, feeds it, and on a refused frame drops the
//! connection and re-fetches from the start of that frame (a per-offset
//! retry budget bounds this; the verifier's own fault budget bounds how many
//! refusals the stream tolerates in total). A transport error or early EOF
//! suspends the stream and resumes it at chunk granularity.
//!
//! Optional frame log: every accepted frame is appended to a file, which is
//! the client's spill. [`recover`] replays a log through a fresh verifier, so
//! nothing read back from disk is trusted: it is re-hashed like fresh input.

use super::ingest::StreamIngest;
use rvf_types::ErrorCode;
use rvf_wire::stream::{StreamPolicy, StreamState};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

/// How a fetch ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FetchEnd {
    /// The server finished the response.
    Eof,
    /// The sink asked to stop.
    Aborted,
}

/// A source of the frame stream.
pub trait Transport {
    /// Deliver the stream bytes from `offset` to `sink` until it returns
    /// `false`, the stream ends, or the transport fails.
    fn fetch(&mut self, offset: u64, sink: &mut dyn FnMut(&[u8]) -> bool) -> io::Result<FetchEnd>;
}

/// Client limits.
#[derive(Clone, Copy, Debug)]
pub struct ClientConfig {
    /// Connections that make no progress before giving up (`Suspended`).
    pub max_stalls: u32,
    /// Refusals of the frame at one offset before giving up.
    pub frame_retries: u32,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            max_stalls: 4,
            frame_retries: 5,
        }
    }
}

/// Counters for benchmarks and diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClientStats {
    /// Bytes handed to the sink by the transport.
    pub bytes_fetched: u64,
    /// Fetch calls.
    pub connections: u32,
    /// Refused frames.
    pub refusals: u32,
    /// Transport errors / early EOFs.
    pub drops: u32,
}

/// Why `run` returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunEnd {
    /// Every segment verified and the body identity confirmed.
    Complete,
    /// Every segment verified, but chunks arrived out of file order (stage
    /// order), so `rvf_identity` is unconfirmed until
    /// `StreamIngest::finish_identity` succeeds. Answers stay unsealed.
    CompleteUnverified,
    /// Terminal failure; nothing is published.
    Poisoned(ErrorCode),
    /// Out of retries; resumable by calling `run` again.
    Suspended,
}

impl RunEnd {
    /// True for `Complete` and `CompleteUnverified` (all chunks verified).
    pub fn all_verified(self) -> bool {
        matches!(self, RunEnd::Complete | RunEnd::CompleteUnverified)
    }
}

/// Fetch-verify-publish loop.
pub struct StreamClient<'p, T: Transport> {
    /// The consumer (query it between `run` steps or from the progress hook).
    pub ingest: StreamIngest<'p>,
    transport: T,
    cfg: ClientConfig,
    /// Counters.
    pub stats: ClientStats,
    log: Option<File>,
    last_refused: Option<(u64, u32)>,
}

impl<'p, T: Transport> StreamClient<'p, T> {
    /// A client for a fresh stream.
    pub fn new(policy: StreamPolicy<'p>, transport: T, cfg: ClientConfig) -> Self {
        Self::with_ingest(StreamIngest::new(policy), transport, cfg)
    }

    /// A client continuing an existing consumer (e.g. after [`recover`]).
    pub fn with_ingest(ingest: StreamIngest<'p>, transport: T, cfg: ClientConfig) -> Self {
        Self {
            ingest,
            transport,
            cfg,
            stats: ClientStats::default(),
            log: None,
            last_refused: None,
        }
    }

    /// Append accepted frames to `path` (created/truncated to the consumer's
    /// current offset).
    pub fn with_log(mut self, path: &Path) -> io::Result<Self> {
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        // The log must be exactly the consumer's stream prefix. Cut a longer
        // log back to it and append from there; a shorter one cannot be
        // extended (the missing bytes are not ours to invent).
        let want = self.ingest.offset();
        let have = f.metadata()?.len();
        if have < want {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "frame log is shorter than the consumer's stream offset",
            ));
        }
        f.set_len(want)?;
        f.seek(SeekFrom::Start(want))?;
        self.log = Some(f);
        Ok(self)
    }

    /// Run until `Complete`, a terminal state, or the retry budget is spent.
    /// `progress` is called after every fetched slice has been consumed.
    pub fn run(&mut self, progress: &mut dyn FnMut(&StreamIngest<'p>)) -> RunEnd {
        let mut stalls = 0u32;
        loop {
            match self.ingest.state() {
                StreamState::Complete => return RunEnd::Complete,
                StreamState::CompleteUnverified => return RunEnd::CompleteUnverified,
                StreamState::Poisoned(e) | StreamState::Rejected(e) => return RunEnd::Poisoned(e),
                StreamState::Suspended => {
                    self.ingest.resume();
                }
                _ => {}
            }
            let start = self.ingest.offset();
            self.stats.connections += 1;
            let mut refused: Option<(u64, ErrorCode)> = None;
            let res = {
                let (ingest, stats, log) = (&mut self.ingest, &mut self.stats, &mut self.log);
                self.transport.fetch(start, &mut |bytes| {
                    stats.bytes_fetched += bytes.len() as u64;
                    let data0 = ingest.offset();
                    let out = ingest.feed(bytes);
                    if let Some(f) = log.as_mut() {
                        let upto = match out.refused {
                            Some(_) => out.resume_offset,
                            None => data0 + bytes.len() as u64,
                        };
                        if upto > data0 {
                            let _ = f.write_all(&bytes[..(upto - data0) as usize]);
                        } else if out.refused.is_some() {
                            // set_len does not move the cursor: seek back too, or the next
                            // write lands past a hole of zeros and recover() stops there.
                            let _ = f.set_len(upto);
                            let _ = f.seek(SeekFrom::Start(upto));
                        }
                    }
                    if let Some(code) = out.refused {
                        refused = Some((out.resume_offset, code));
                        return false;
                    }
                    progress(ingest);
                    true
                })
            };
            if res.is_err() {
                self.stats.drops += 1;
            }
            if let Some((off, _)) = refused {
                self.stats.refusals += 1;
                let n = match self.last_refused {
                    Some((o, n)) if o == off => n + 1,
                    _ => 1,
                };
                self.last_refused = Some((off, n));
                if n > self.cfg.frame_retries {
                    self.ingest.end_of_input();
                    return RunEnd::Suspended;
                }
            }
            if matches!(
                self.ingest.state(),
                StreamState::Poisoned(_)
                    | StreamState::Rejected(_)
                    | StreamState::Complete
                    | StreamState::CompleteUnverified
            ) {
                continue;
            }
            if self.ingest.offset() > start {
                stalls = 0;
            } else {
                stalls += 1;
                if stalls > self.cfg.max_stalls {
                    self.ingest.end_of_input();
                    return RunEnd::Suspended;
                }
            }
            if refused.is_none() {
                // Clean EOF or drop before Complete: suspend, then resume.
                self.ingest.end_of_input();
            }
        }
    }
}

/// Rebuild a consumer from a frame log by replaying it through a fresh
/// verifier. Returns the consumer and the number of bytes re-hashed. A log
/// that stops mid-frame, or contains a refused frame, is cut there.
pub fn recover<'p>(policy: StreamPolicy<'p>, log: &Path) -> io::Result<(StreamIngest<'p>, u64)> {
    let mut ingest = StreamIngest::new(policy);
    let mut f = File::open(log)?;
    let mut buf = vec![0u8; 64 * 1024];
    let mut replayed = 0u64;
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        replayed += n as u64;
        let out = ingest.feed(&buf[..n]);
        if out.refused.is_some() {
            break;
        }
    }
    if matches!(ingest.state(), StreamState::Receiving) {
        ingest.end_of_input();
    }
    Ok((ingest, replayed))
}

/// Serves a byte vector (the whole frame stream) from any offset.
pub struct MemTransport {
    /// The stream bytes.
    pub bytes: Vec<u8>,
    /// Slice size handed to the sink.
    pub slice: usize,
    /// Stop (as a dropped connection) after this many bytes per fetch.
    pub cut_after: Option<usize>,
}

impl MemTransport {
    /// A transport over `bytes`.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            slice: 16 * 1024,
            cut_after: None,
        }
    }
}

impl Transport for MemTransport {
    fn fetch(&mut self, offset: u64, sink: &mut dyn FnMut(&[u8]) -> bool) -> io::Result<FetchEnd> {
        let mut at = (offset as usize).min(self.bytes.len());
        let stop = self
            .cut_after
            .map_or(self.bytes.len(), |c| (at + c).min(self.bytes.len()));
        while at < stop {
            let z = (at + self.slice).min(stop);
            if !sink(&self.bytes[at..z]) {
                return Ok(FetchEnd::Aborted);
            }
            at = z;
        }
        if stop < self.bytes.len() {
            return Err(io::Error::new(io::ErrorKind::ConnectionReset, "cut"));
        }
        Ok(FetchEnd::Eof)
    }
}

/// Link-rate limiter: delays delivery so the long-run rate is `mbps`.
pub struct Throttled<T> {
    /// The wrapped transport.
    pub inner: T,
    /// Megabits per second (decimal).
    pub mbps: f64,
}

impl<T: Transport> Transport for Throttled<T> {
    fn fetch(&mut self, offset: u64, sink: &mut dyn FnMut(&[u8]) -> bool) -> io::Result<FetchEnd> {
        let rate = self.mbps * 1e6 / 8.0; // bytes per second
        let t0 = Instant::now();
        let mut sent = 0u64;
        self.inner.fetch(offset, &mut |b| {
            sent += b.len() as u64;
            let due = Duration::from_secs_f64(sent as f64 / rate);
            if let Some(wait) = due.checked_sub(t0.elapsed()) {
                std::thread::sleep(wait);
            }
            sink(b)
        })
    }
}
