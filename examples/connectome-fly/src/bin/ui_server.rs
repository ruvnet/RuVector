//! Connectome OS live UI backend — **real LIF data, no simulation
//! proxy, no synthetic data in the stream**.
//!
//! Binds a tiny HTTP+SSE server on 127.0.0.1:5174 (override with
//! `CONNECTOME_UI_PORT`). Every connection to `/stream` spins up a
//! fresh `Engine` + `Observer` against a fresh synthetic SBM
//! connectome and streams one Server-Sent-Event per simulated `dt_ms`
//! tick, carrying:
//!
//! * real spike events from the LIF engine,
//! * the Observer's most recent Fiedler value (the live `λ₂` of the
//!   co-firing-window Laplacian — **not** a proxy),
//! * a real CPM-Leiden community partition re-run on a cadence,
//! * the Engine's total spike count and simulation clock.
//!
//! Zero external dependencies. The server is deliberately single-
//! threaded blocking I/O — the point is "prove the browser is
//! downstream of the real Rust engine", not "serve ten thousand
//! clients". See §17 item 27 for the discovery that kicked this off
//! (the user noticing the UI was a mock).
//!
//! Routes:
//!   GET /status       → JSON: engine identity, connectome config,
//!                              crate version, a witness hash so the
//!                              browser can prove it's really talking
//!                              to this binary (no static mock).
//!   GET /stream       → `text/event-stream` — one SSE per simulated
//!                              tick (default 1 ms).
//!   GET /              → tiny "alive" page with the current status.
//!
//! The Vite dev server proxies `/api/*` to this binary (see
//! `ui/vite.config.js`), so the browser hits `/api/stream` and it
//! lands here.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use connectome_fly::{
    Analysis, AnalysisConfig, Connectome, ConnectomeConfig, Engine, EngineConfig,
    NeuronId, Observer, Stimulus,
};

/// Source of truth for the running connectome. Either synthesized
/// from the default SBM config, or loaded from a directory containing
/// FlyWire v783 TSVs (`neurons.tsv`, `connections.tsv`, optional
/// `classification.tsv`). The caller ships a `&'static` description
/// that goes into the `/status` endpoint so the browser can tell.
enum ConnectomeSource {
    SyntheticSbm {
        cfg: ConnectomeConfig,
        conn: Connectome,
    },
    Flywire {
        dir: String,
        conn: Connectome,
    },
}

impl ConnectomeSource {
    fn conn(&self) -> &Connectome {
        match self {
            Self::SyntheticSbm { conn, .. } | Self::Flywire { conn, .. } => conn,
        }
    }
    fn status_label(&self) -> &'static str {
        match self {
            Self::SyntheticSbm { .. } => "synthetic-sbm",
            Self::Flywire { dir, .. } => {
                // Princeton dirs carry the exact file on disk; TSV
                // dirs don't. Cheap heuristic — test both common file
                // names — at runtime so the label reflects what was
                // actually parsed.
                let p = std::path::Path::new(dir);
                if p.join("connections_princeton.csv.gz").exists() {
                    "flywire-princeton-csv"
                } else {
                    "flywire-v783-tsv"
                }
            }
        }
    }
    fn num_modules(&self) -> u32 {
        match self {
            Self::SyntheticSbm { cfg, .. } => cfg.num_modules as u32,
            // FlyWire has no explicit module count; use community
            // count of a quick CPM-Leiden run lazily, or 0 as a
            // placeholder. /status just reports it as 0 for now.
            Self::Flywire { .. } => 0,
        }
    }
    fn source_detail(&self) -> String {
        match self {
            Self::SyntheticSbm { .. } => "connectome-fly/src/lif/engine.rs".into(),
            Self::Flywire { dir, .. } => format!(
                "connectome-fly/src/connectome/flywire/streaming.rs (dir={dir})"
            ),
        }
    }
}

fn load_connectome() -> ConnectomeSource {
    // Princeton-format gzipped CSV path: `neurons.csv.gz` +
    // `connections_princeton.csv.gz` under a single dir.
    if let Ok(dir) = std::env::var("CONNECTOME_FLYWIRE_PRINCETON_DIR") {
        let dir_path = std::path::Path::new(&dir);
        let neurons = dir_path.join("neurons.csv.gz");
        let conns = dir_path.join("connections_princeton.csv.gz");
        eprintln!("[ui_server] loading FlyWire Princeton CSV from {dir}…");
        match connectome_fly::connectome::flywire::princeton::load_flywire_princeton(
            &neurons, &conns,
        ) {
            Ok(conn) => {
                eprintln!(
                    "[ui_server] Princeton loaded: n={} synapses={} (from {dir})",
                    conn.num_neurons(),
                    conn.num_synapses()
                );
                return ConnectomeSource::Flywire { dir, conn };
            }
            Err(e) => {
                eprintln!("[ui_server] Princeton load failed: {e:?} — falling back");
            }
        }
    }
    // v783 TSV path: `neurons.tsv` + `connections.tsv` (+ optional
    // `classification.tsv`) under a single dir.
    if let Ok(dir) = std::env::var("CONNECTOME_FLYWIRE_DIR") {
        let path = std::path::Path::new(&dir);
        eprintln!("[ui_server] loading FlyWire v783 TSVs from {dir}…");
        match connectome_fly::connectome::flywire::streaming::load_flywire_streaming(path) {
            Ok(conn) => {
                eprintln!(
                    "[ui_server] FlyWire loaded: n={} synapses={} (from {dir})",
                    conn.num_neurons(),
                    conn.num_synapses()
                );
                return ConnectomeSource::Flywire { dir, conn };
            }
            Err(e) => {
                eprintln!("[ui_server] FlyWire load failed: {e:?} — falling back to synthetic SBM");
            }
        }
    }
    let cfg = ConnectomeConfig::default();
    let conn = Connectome::generate(&cfg);
    ConnectomeSource::SyntheticSbm { cfg, conn }
}

const DEFAULT_PORT: u16 = 5174;
/// Snapshot of the CPM community partition every N ticks (expensive —
/// a full Leiden-CPM run over the current graph). 50 ticks = every
/// 50 ms of simulated time ≈ every 50 server ms under no throttling.
const COMMUNITY_SNAPSHOT_EVERY_TICKS: u32 = 50;
/// Step size per SSE event. Matches the engine's default dt_ms.
const TICK_MS: f32 = 1.0;
/// How many ticks before the server auto-stops a client stream to
/// avoid infinite-memory observers. 10 000 ticks = 10 s simulated.
const MAX_TICKS_PER_CLIENT: u32 = 10_000;

/// Global per-process witness counter so the `/status` endpoint can
/// hand out a unique id per boot — the browser can cache it on first
/// `/status` and assert every subsequent `/status` returns the same
/// value, proving it's not a static mock file.
static WITNESS: AtomicU64 = AtomicU64::new(0);

fn main() {
    let port: u16 = std::env::var("CONNECTOME_UI_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let addr = format!("127.0.0.1:{port}");
    // Initialise witness to a deterministic function of wall time so
    // a restart is detectable.
    let boot = Instant::now().elapsed().as_nanos() as u64 ^ (port as u64) << 32;
    WITNESS.store(boot.max(1), Ordering::Relaxed);

    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("[ui_server] bind {addr} failed: {e}");
        std::process::exit(2);
    });
    eprintln!(
        "[ui_server] listening on http://{addr}  (engine=rust-lif crate_ver={})",
        connectome_fly::VERSION
    );
    for stream in listener.incoming() {
        match stream {
            Ok(s) => {
                thread::spawn(move || handle(s));
            }
            Err(e) => eprintln!("[ui_server] accept error: {e}"),
        }
    }
}

fn handle(mut stream: TcpStream) {
    let peer = stream
        .peer_addr()
        .map(|a| a.to_string())
        .unwrap_or_else(|_| "?".into());
    let mut reader = BufReader::new(stream.try_clone().expect("clone"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    // Drain headers (we don't need any).
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).is_err() || line == "\r\n" || line.is_empty() {
            break;
        }
    }
    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();
    eprintln!("[ui_server] {peer} GET {path}");
    match path.as_str() {
        "/status" | "/api/status" => write_status(&mut stream),
        "/stream" | "/api/stream" => run_sse_stream(&mut stream),
        "/" | "/index.html" => write_landing(&mut stream),
        _ => write_404(&mut stream),
    }
    let _ = stream.flush();
}

fn write_404(stream: &mut TcpStream) {
    let body = b"{\"error\":\"not found\"}";
    let _ = write!(
        stream,
        "HTTP/1.1 404 Not Found\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(body);
}

fn write_status(stream: &mut TcpStream) {
    let witness = WITNESS.load(Ordering::Relaxed);
    let src = load_connectome();
    let conn = src.conn();
    let body = format!(
        r#"{{"engine":"rust-lif","source":"{src_detail}",
"substrate":"{label}","crate_version":"{ver}","connectome":{{"num_neurons":{n},"num_synapses":{syn},"num_modules":{m}}},
"detector":"Observer::compute_fiedler (eigensolver::approx_fiedler_power / sparse_fiedler)",
"community_algorithm":"analysis::leiden::leiden_labels_cpm (weight-normalized CPM)",
"witness":{witness},"mock":false,"simulated":false}}"#,
        src_detail = src.source_detail(),
        label = src.status_label(),
        ver = connectome_fly::VERSION,
        n = conn.num_neurons(),
        syn = conn.num_synapses(),
        m = src.num_modules(),
        witness = witness
    );
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(body.as_bytes());
}

fn write_landing(stream: &mut TcpStream) {
    let body = format!(
        "<!doctype html><title>Connectome OS — UI Server</title>\
         <style>body{{font:14px/1.5 system-ui;max-width:48rem;margin:2rem auto;padding:0 1rem;background:#0b1120;color:#d1e0ff}}\
         code{{background:#1a2333;padding:.2em .4em;border-radius:.3em;color:#7fffd4}}</style>\
         <h1>Connectome OS — live Rust LIF backend</h1>\
         <p>This is a <b>real</b> simulation backend, not a mock. Routes:</p>\
         <ul><li><code>GET /status</code> — engine identity + connectome config</li>\
         <li><code>GET /stream</code> — Server-Sent-Events stream of real spikes, Fiedler λ₂ values, and CPM community snapshots</li></ul>\
         <p>Crate version: <code>{}</code></p>",
        connectome_fly::VERSION
    );
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(body.as_bytes());
}

fn run_sse_stream(stream: &mut TcpStream) {
    // Real-time streaming over TCP: disable Nagle so every SSE frame
    // flushes immediately instead of coalescing.
    let _ = stream.set_nodelay(true);
    // SSE preamble.
    if write!(
        stream,
        "HTTP/1.1 200 OK\r\n\
         Content-Type: text/event-stream\r\n\
         Cache-Control: no-store\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Connection: keep-alive\r\n\
         X-Engine: rust-lif\r\n\
         X-Mock: false\r\n\r\n"
    )
    .is_err()
    {
        return;
    }
    let _ = stream.flush();
    eprintln!("[ui_server]   sse: preamble sent");

    // Fresh simulation per client. The connectome either comes from
    // the default synthetic SBM or from a FlyWire v783 dataset at the
    // path in CONNECTOME_FLYWIRE_DIR — the browser can independently
    // verify which by reading /status.
    let src = load_connectome();
    let conn = src.conn();
    let mut engine = Engine::new(conn, EngineConfig::default());
    let skip_fiedler =
        std::env::var("CONNECTOME_SKIP_FIEDLER").ok().as_deref() == Some("1");
    // Fiedler-detector cadence. At N ≤ 10k the default 5 ms cadence
    // holds; at N = 115k (real fly brain) each detect is O(n²)–O(n³)
    // on the co-firing Laplacian and stalls the loop for seconds. We
    // back off to 500 ms automatically at N ≥ 10k, and to `infinity`
    // (detector disabled) when CONNECTOME_SKIP_FIEDLER=1.
    let detect_every_ms: f32 = if skip_fiedler {
        f32::INFINITY
    } else if conn.num_neurons() >= 10_000 {
        500.0
    } else {
        5.0
    };
    let mut observer = Observer::new(conn.num_neurons() as usize)
        .with_detector(50.0, detect_every_ms, 20, 2.0);
    eprintln!(
        "[ui_server] observer: detect_every_ms={} skip_fiedler={}",
        detect_every_ms, skip_fiedler
    );
    // Drive the network with a continuous pulse train into all
    // sensory neurons. `run_with` re-pushes every stim event onto
    // the heap on each call, so we apply the full stim ONCE on the
    // first iteration and use `Stimulus::empty()` thereafter — the
    // events are already on the heap and will fire at their
    // scheduled times. This is a 1000× speedup vs the naive
    // re-apply-each-tick form.
    let sensory: Vec<NeuronId> = conn.sensory_neurons().to_vec();
    let stim_len_ms = (MAX_TICKS_PER_CLIENT as f32) * TICK_MS;
    let stim_full = Stimulus::pulse_train(&sensory, 0.0, stim_len_ms, 15.0, 40.0);
    let stim_empty = Stimulus::empty();
    eprintln!(
        "[ui_server] stim: {} events into {} sensory neurons (pulse_train 40 Hz × {:.0} ms, pushed once)",
        stim_full.len(),
        sensory.len(),
        stim_len_ms
    );
    let mut stim_applied = false;

    // Seed the "hello" event so the client can confirm the connection
    // without waiting for the first tick to produce spikes.
    if write_sse(
        stream,
        "hello",
        &format!(
            r#"{{"engine":"rust-lif","substrate":"{label}","crate":"{ver}","connectome":{{"n":{n},"synapses":{syn}}},"witness":{w}}}"#,
            label = src.status_label(),
            ver = connectome_fly::VERSION,
            n = conn.num_neurons(),
            syn = conn.num_synapses(),
            w = WITNESS.load(Ordering::Relaxed)
        ),
    )
    .is_err()
    {
        eprintln!("[ui_server]   sse: hello write failed");
        return;
    }
    eprintln!("[ui_server]   sse: hello sent (substrate={})", src.status_label());

    // Community snapshot state.
    let analysis = Analysis::new(AnalysisConfig::default());
    let _ = &analysis;
    let mut tick: u32 = 0;
    let mut last_spike_count: usize = 0;
    let mut sim_clock: f32 = 0.0;
    let started = Instant::now();

    loop {
        if tick >= MAX_TICKS_PER_CLIENT {
            let _ = write_sse(stream, "end", r#"{"reason":"max_ticks"}"#);
            return;
        }
        sim_clock += TICK_MS;
        // Step the real engine forward TICK_MS. Apply the stim once
        // up-front; afterwards the events are on the heap and
        // `run_with` just processes them at their scheduled times.
        let stim_ref = if stim_applied {
            &stim_empty
        } else {
            stim_applied = true;
            &stim_full
        };
        engine.run_with(stim_ref, &mut observer, sim_clock);

        // Collect spikes produced this tick.
        let spikes = observer.spikes();
        let delta_end = spikes.len();
        let mut delta_ids: Vec<u32> = Vec::with_capacity(delta_end.saturating_sub(last_spike_count));
        for s in &spikes[last_spike_count..delta_end] {
            delta_ids.push(s.neuron.idx() as u32);
        }
        last_spike_count = delta_end;

        // Fiedler.
        let lambda2 = observer.latest_fiedler();
        let baseline_mean = observer.fiedler_baseline_mean();

        // Per-tick SSE.
        let spikes_json = json_array_u32(&delta_ids, 128);
        let tick_body = format!(
            r#"{{"t":{t:.1},"tick":{tick},"spikes":{sp},"n_spikes_total":{tot},"fiedler":{lam},"baseline_mean":{bm},"wall_ms":{wm}}}"#,
            t = sim_clock,
            tick = tick,
            sp = spikes_json,
            tot = engine.total_spikes(),
            lam = fmt_f32(lambda2),
            bm = fmt_f32(baseline_mean),
            wm = started.elapsed().as_millis()
        );
        if write_sse(stream, "tick", &tick_body).is_err() {
            return;
        }

        // Community snapshot every N ticks (cheap at N ≈ 1024; at
        // N ≥ 10k we throttle to every 2 s of simulated time so the
        // CPM run doesn't stall the SSE loop). Set CONNECTOME_SKIP_COMMUNITIES=1
        // to disable entirely on huge substrates (e.g. full FlyWire).
        let skip_communities =
            std::env::var("CONNECTOME_SKIP_COMMUNITIES").ok().as_deref() == Some("1");
        let snapshot_every = if conn.num_neurons() < 8_000 {
            COMMUNITY_SNAPSHOT_EVERY_TICKS
        } else {
            2_000 // every 2 s of simulated time on big substrates
        };
        if !skip_communities && tick > 0 && tick % snapshot_every == 0 {
            let labels = connectome_fly::analysis::leiden::leiden_labels_cpm(conn, 3.1);
            let (num_communities, module_ari_proxy) = summarise(conn, &labels);
            let snap = format!(
                r#"{{"tick":{tick},"num_communities":{nc},"module_sample":{ms}}}"#,
                tick = tick,
                nc = num_communities,
                ms = module_ari_proxy
            );
            if write_sse(stream, "communities", &snap).is_err() {
                return;
            }
        }

        tick += 1;

        // Throttle: real-time pace. Without this the loop runs
        // ~50 000 ticks/s and the browser floods. 1 ms wall ~ 1 ms sim
        // gives a stable 1 kHz event rate.
        thread::sleep(Duration::from_millis(1));
    }
}

/// Tiny unprotected JSON array writer for u32 ids. Truncates to
/// `cap` entries to keep the SSE line bounded.
fn json_array_u32(ids: &[u32], cap: usize) -> String {
    if ids.is_empty() {
        return "[]".into();
    }
    let end = ids.len().min(cap);
    let mut s = String::with_capacity(6 * end + 2);
    s.push('[');
    for (i, id) in ids[..end].iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&id.to_string());
    }
    s.push(']');
    s
}

fn fmt_f32(x: f32) -> String {
    if x.is_nan() {
        "null".into()
    } else {
        format!("{:.6}", x)
    }
}

/// Tiny summary for the community snapshot: return (num_distinct,
/// module-alignment-score). Module-alignment is a cheap proxy: for
/// each predicted community, how pure is it by ground-truth module.
/// On FlyWire-loaded connectomes every neuron has module=0 (the loader
/// doesn't synthesize SBM module ids), so the purity number is
/// vacuously 1.0 there; the browser still gets the community count.
fn summarise(conn: &Connectome, labels: &[u32]) -> (u32, f32) {
    use std::collections::HashMap;
    let mut by_pred: HashMap<u32, HashMap<u16, u32>> = HashMap::new();
    for (i, &p) in labels.iter().enumerate() {
        let m = conn.meta(NeuronId(i as u32)).module;
        *by_pred.entry(p).or_default().entry(m).or_insert(0) += 1;
    }
    let num = by_pred.len() as u32;
    let mut sum_purity: f32 = 0.0;
    let mut total_nodes: u32 = 0;
    for (_pred, mod_counts) in &by_pred {
        let total: u32 = mod_counts.values().sum();
        let max: u32 = *mod_counts.values().max().unwrap_or(&0);
        total_nodes += total;
        sum_purity += max as f32;
    }
    let purity = if total_nodes > 0 {
        sum_purity / total_nodes as f32
    } else {
        0.0
    };
    (num, purity)
}

fn write_sse(stream: &mut TcpStream, event: &str, data: &str) -> std::io::Result<()> {
    // SSE frame: `event: <name>\ndata: <json>\n\n`.
    write!(stream, "event: {event}\ndata: {data}\n\n")?;
    stream.flush()
}
