//! `ruview-vitals-worker` — per-Pi WiFi-CSI vital signs worker
//! (ADR-183 Tier 1).
//!
//! Boot order:
//! 1. Parse [`Config`] from env.
//! 2. Build shared [`WorkerState`].
//! 3. Spawn the brain POST loop (every `brain_post_interval`).
//! 4. Spawn the gRPC `Vitals` service on `grpc_listen`.
//! 5. Spawn a counters heartbeat (once per minute).
//! 6. Run the UDP ingest hot loop on `udp_listen`: parse, pipeline
//!    step, fan out via the shared broadcast channel.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use tokio::net::UdpSocket;
use tracing_subscriber::EnvFilter;

use ruview_vitals_worker::{
    brain, grpc,
    pipeline::{now_us, VitalsPipeline},
    state::WorkerState,
    Adr018Frame, Adr018Header, Config, Result, VERSION,
};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<()> {
    init_tracing();
    let cfg = Config::from_env()?;
    let (state, _initial_rx) = WorkerState::new(cfg);

    tracing::info!(
        version = VERSION,
        node = %state.config.node_name,
        udp = %state.config.udp_listen,
        grpc = %state.config.grpc_listen,
        brain = %state.config.brain_url,
        window_frames = state.config.window_frames,
        relay_targets = state.config.relay_targets.len(),
        "ruview-vitals-worker starting"
    );

    // gRPC server.
    {
        let s = Arc::clone(&state);
        tokio::spawn(async move {
            if let Err(e) = grpc::serve(s).await {
                tracing::error!(error = %e, "gRPC server exited");
            }
        });
    }

    // Brain POST loop.
    match brain::BrainClient::new(
        state.config.brain_url.clone(),
        state.config.node_name.clone(),
    ) {
        Ok(client) => {
            let s = Arc::clone(&state);
            let interval = state.config.brain_post_interval;
            tokio::spawn(brain::run_brain_loop(client, s, interval));
        }
        Err(e) => {
            tracing::error!(error = %e, "brain client init failed; vitals will not be POSTed");
        }
    }

    // Heartbeat — counters tracer once a minute.
    {
        let s = Arc::clone(&state);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(60));
            tick.tick().await;
            loop {
                tick.tick().await;
                let snap = s.stats.snapshot();
                tracing::info!(
                    packets_received = snap.packets_received,
                    packets_dropped = snap.packets_dropped,
                    packets_relayed = snap.packets_relayed,
                    readings_emitted = snap.readings_emitted,
                    brain_posts_ok = snap.brain_posts_ok,
                    brain_posts_failed = snap.brain_posts_failed,
                    uptime_seconds = s.uptime_seconds(),
                    "vitals-worker heartbeat"
                );
            }
        });
    }

    // UDP relay fan-out (ADR-183 Tier 2 iter 9). When configured,
    // each received datagram is forwarded unchanged to one or more
    // targets — typically `cognitum-v0:5005` from worker Pis so the
    // fusion master sees CSI from every room.
    let relay_tx = if state.config.relay_targets.is_empty() {
        None
    } else {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<u8>>(2048);
        let targets = state.config.relay_targets.clone();
        tokio::spawn(async move {
            let socket = match UdpSocket::bind("0.0.0.0:0").await {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "relay socket bind failed");
                    return;
                }
            };
            tracing::info!(targets = ?targets, "UDP relay fan-out up");
            while let Some(buf) = rx.recv().await {
                for t in &targets {
                    if let Err(e) = socket.send_to(&buf, t).await {
                        tracing::warn!(error = %e, target = %t, "relay send failed");
                    }
                }
            }
        });
        Some(tx)
    };

    // UDP ingest hot loop.
    let socket = UdpSocket::bind(state.config.udp_listen).await?;
    tracing::info!(addr = %socket.local_addr()?, "UDP listener up");

    let mut pipeline = VitalsPipeline::esp32_default();
    let verbose = state.config.verbose;
    let mut buf = vec![0u8; 65_536];
    loop {
        let (len, peer) = match socket.recv_from(&mut buf).await {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "UDP recv_from failed");
                continue;
            }
        };
        state.stats.packets_received.fetch_add(1, Ordering::Relaxed);

        let datagram = &buf[..len];

        // Relay first — any parse decision is independent of fan-out
        // (the v6 feature-state frames we drop locally are still
        // useful upstream at v0). `try_send` keeps this lock-free
        // under burst.
        if let Some(tx) = &relay_tx {
            if tx.try_send(datagram.to_vec()).is_ok() {
                state.stats.packets_relayed.fetch_add(1, Ordering::Relaxed);
            }
        }

        match Adr018Frame::parse(datagram) {
            Some(frame) => {
                if verbose {
                    log_frame(&peer, &frame.header, len);
                }
                state.stats.windows_processed.fetch_add(1, Ordering::Relaxed);
                let ts = now_us();
                if let Some(step) = pipeline.step(&frame, ts) {
                    state.record(step.reading).await;
                }
            }
            None => {
                state.stats.packets_dropped.fetch_add(1, Ordering::Relaxed);
                if let Some(hdr) = Adr018Header::parse(datagram) {
                    tracing::warn!(
                        peer = %peer,
                        len,
                        node_id = hdr.node_id,
                        n_subcarriers = hdr.n_subcarriers,
                        n_antennas = hdr.n_antennas,
                        "drop: payload too short"
                    );
                } else {
                    tracing::warn!(peer = %peer, len, "drop: not an ADR-018 frame");
                }
            }
        }
    }
}

fn log_frame(peer: &std::net::SocketAddr, hdr: &Adr018Header, len: usize) {
    tracing::debug!(
        peer = %peer,
        len,
        magic = format_args!("0x{:08x}", hdr.magic),
        node_id = hdr.node_id,
        antennas = hdr.n_antennas,
        subcarriers = hdr.n_subcarriers,
        channel = hdr.channel,
        rssi_dbm = hdr.rssi,
        noise_dbm = hdr.noise_floor,
        ts_us = hdr.timestamp_us,
        "ADR-018 frame"
    );
}

fn init_tracing() {
    let filter = EnvFilter::try_from_env("RUVIEW_VITALS_LOG")
        .or_else(|_| EnvFilter::try_new("info,ruview_vitals_worker=info"))
        .expect("default tracing filter");
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_writer(std::io::stderr)
        .init();
}
