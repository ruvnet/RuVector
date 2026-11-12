//! Binary entrypoint for the RVF streaming server.

use clap::Parser;
use std::path::PathBuf;

use rvf_server::ServerConfig;

#[derive(Parser)]
#[command(
    name = "rvf-server",
    about = "RuVector Format TCP/HTTP streaming server"
)]
struct Cli {
    /// HTTP listen address. Anything but loopback requires a bearer token in
    /// the RVF_SERVER_TOKEN environment variable.
    #[arg(long, default_value = "127.0.0.1")]
    host: std::net::IpAddr,

    /// HTTP listen port
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// TCP streaming listen port
    #[arg(long, default_value_t = 9090)]
    tcp_port: u16,

    /// Path to the RVF data directory / file
    #[arg(long, default_value = "data.rvf")]
    data_dir: PathBuf,

    /// Vector dimension (used when creating a new store)
    #[arg(long, default_value_t = 128)]
    dimension: u16,

    /// Serve this .rvf as a verified stream (requires --stream-key)
    #[arg(long, requires = "stream_key")]
    stream_body: Option<PathBuf>,

    /// Path of the descriptor signing key (64 hex chars or 32 raw bytes);
    /// supplied by the operator, never generated or stored by the server
    #[arg(long)]
    stream_key: Option<PathBuf>,

    /// Stream frame size as log2 bytes (12..=30, or 255 for one per segment)
    #[arg(long, default_value_t = 16)]
    stream_chunk_log2: u8,

    /// Descriptor epoch
    #[arg(long, default_value_t = 1)]
    stream_epoch: u64,

    /// Concurrent body streams allowed
    #[arg(long, default_value_t = 8)]
    stream_max_streams: usize,

    /// Largest response for one Range request, in bytes
    #[arg(long, default_value_t = 268_435_456)]
    stream_max_range: u64,

    /// Minimum client read rate (bytes/s) enforced per rate window
    #[arg(long, default_value_t = 32_768)]
    stream_min_rate: u64,

    /// Streams one client address may hold at once
    #[arg(long, default_value_t = 2)]
    stream_max_per_client: usize,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let config = ServerConfig {
        http_port: cli.port,
        tcp_port: cli.tcp_port,
        data_path: cli.data_dir,
        dimension: cli.dimension,
        host: cli.host,
        token: rvf_server::auth::AuthToken::from_env(),
    };

    let stream = match (cli.stream_body, cli.stream_key) {
        (Some(body), Some(key)) => {
            let mut c = rvf_server::stream::StreamServerConfig::new(body, key);
            c.chunk_log2 = cli.stream_chunk_log2;
            c.epoch = cli.stream_epoch;
            c.max_concurrent_streams = cli.stream_max_streams;
            c.max_range_bytes = cli.stream_max_range;
            c.min_rate_bytes_per_sec = cli.stream_min_rate;
            c.max_streams_per_client = cli.stream_max_per_client;
            Some(c)
        }
        _ => None,
    };
    if let Err(e) = rvf_server::run_with_stream(config, stream).await {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
}
