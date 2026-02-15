//! QR Cognitive Seed — "A World Inside a World"
//!
//! Demonstrates building, parsing, and bootstrapping a QR seed:
//!
//! 1. Build an RVQS payload with microkernel, hosts, layers, and signature
//! 2. Verify it fits in a single QR code (≤2,953 bytes)
//! 3. Parse the seed back and extract the download manifest
//! 4. Simulate progressive bootstrap from seed → full intelligence
//!
//! Run: cargo run --example qr_seed_bootstrap -p rvf-runtime

use rvf_runtime::qr_seed::{
    BootstrapProgress, DownloadManifest, ParsedSeed, SeedBuilder, make_host_entry,
};
use rvf_types::qr_seed::*;

fn main() {
    println!("=== QR Cognitive Seed: A World Inside a World ===\n");

    // --- Phase 0: Build the seed ---
    println!("[Phase 0] Building RVQS seed...");

    // Simulated compressed WASM microkernel (2.1 KB Brotli).
    let microkernel = vec![0xCA; 2100];

    // Primary CDN host.
    let primary_host = make_host_entry(
        "https://cdn.ruvector.ai/rvf/brain-v1.rvf",
        0,  // highest priority
        1,  // region: US-East
        [0xAA; 16], // host key hash
    )
    .expect("primary host");

    // Fallback host.
    let fallback_host = make_host_entry(
        "https://mirror.ruvector.ai/rvf/brain-v1.rvf",
        1,  // lower priority
        2,  // region: EU-West
        [0xBB; 16],
    )
    .expect("fallback host");

    // Progressive layers.
    let layers = vec![
        LayerEntry {
            layer_id: layer_id::LEVEL0,
            priority: 0,
            offset: 0,
            size: 4_096,
            content_hash: [0x11; 16],
            required: 1,
            _pad: 0,
        },
        LayerEntry {
            layer_id: layer_id::HOT_CACHE,
            priority: 1,
            offset: 4_096,
            size: 51_200,
            content_hash: [0x22; 16],
            required: 1,
            _pad: 0,
        },
        LayerEntry {
            layer_id: layer_id::HNSW_LAYER_A,
            priority: 2,
            offset: 55_296,
            size: 204_800,
            content_hash: [0x33; 16],
            required: 0,
            _pad: 0,
        },
        LayerEntry {
            layer_id: layer_id::QUANT_DICT,
            priority: 3,
            offset: 260_096,
            size: 102_400,
            content_hash: [0x44; 16],
            required: 0,
            _pad: 0,
        },
        LayerEntry {
            layer_id: layer_id::HNSW_LAYER_B,
            priority: 4,
            offset: 362_496,
            size: 512_000,
            content_hash: [0x55; 16],
            required: 0,
            _pad: 0,
        },
    ];

    // Ed25519 signature (64 bytes).
    let signature = vec![0xEE; 64];

    let mut builder = SeedBuilder::new(
        [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08], // file_id
        384,       // dimension
        100_000,   // total vectors
    )
    .with_microkernel(microkernel)
    .add_host(primary_host)
    .add_host(fallback_host)
    .with_signature(0, signature) // 0 = Ed25519
    .with_content_hash([0xAB; 8]);

    builder.base_dtype = 1;  // F16
    builder.profile_id = 2;  // Hot profile
    builder.content_hash_full = Some([0xDD; 32]);
    builder.total_file_size = Some(874_496);
    builder.stream_upgrade = true;

    for layer in &layers {
        builder = builder.add_layer(*layer);
    }

    let (payload, header) = builder.build().expect("seed build");

    println!("  Seed magic:    0x{:08X} (\"RVQS\")", header.seed_magic);
    println!("  Version:       {}", header.seed_version);
    println!("  Flags:         0x{:04X}", header.flags);
    println!("  File ID:       {:02X?}", header.file_id);
    println!("  Vectors:       {}", header.total_vector_count);
    println!("  Dimension:     {}", header.dimension);
    println!("  Microkernel:   {} bytes (compressed)", header.microkernel_size);
    println!("  Manifest:      {} bytes", header.download_manifest_size);
    println!("  Signature:     {} bytes (Ed25519)", header.sig_length);
    println!("  Total size:    {} bytes", header.total_seed_size);
    println!("  QR capacity:   {} bytes", QR_MAX_BYTES);
    println!(
        "  Fits in QR:    {} ({} bytes headroom)",
        header.fits_in_qr(),
        QR_MAX_BYTES as u32 - header.total_seed_size
    );
    println!();

    // --- Phase 1: Parse and verify ---
    println!("[Phase 1] Parsing seed from binary...");

    let parsed = ParsedSeed::parse(&payload).expect("parse seed");

    println!("  Header valid:  {}", parsed.header.is_valid_magic());
    println!(
        "  Microkernel:   {} ({} bytes)",
        if parsed.microkernel.is_some() { "present" } else { "absent" },
        parsed.microkernel.map(|m| m.len()).unwrap_or(0)
    );
    println!(
        "  Manifest:      {} ({} bytes)",
        if parsed.manifest_bytes.is_some() { "present" } else { "absent" },
        parsed.manifest_bytes.map(|m| m.len()).unwrap_or(0)
    );
    println!(
        "  Signature:     {} ({} bytes)",
        if parsed.signature.is_some() { "present" } else { "absent" },
        parsed.signature.map(|s| s.len()).unwrap_or(0)
    );

    // Extract signed payload for verification.
    if let Some(signed) = parsed.signed_payload(&payload) {
        println!("  Signed bytes:  {} (verify against signature)", signed.len());
    }
    println!();

    // --- Phase 2: Parse download manifest ---
    println!("[Phase 2] Parsing download manifest...");

    let manifest = parsed.parse_manifest().expect("parse manifest");

    println!("  Hosts: {}", manifest.hosts.len());
    for (i, host) in manifest.hosts.iter().enumerate() {
        let label = if i == 0 { "Primary" } else { "Fallback" };
        println!(
            "    [{label}] {} (priority={}, region={})",
            host.url_str().unwrap_or("<invalid>"),
            host.priority,
            host.region
        );
    }

    println!("  Content hash:  {:?}", manifest.content_hash.map(|h| hex_short(&h)));
    println!("  Total size:    {} bytes", manifest.total_file_size.unwrap_or(0));
    println!("  Layers: {}", manifest.layers.len());
    for layer in &manifest.layers {
        let name = layer_name(layer.layer_id);
        println!(
            "    [{:>2}] {:<20} offset={:<8} size={:<8} required={} hash={}",
            layer.priority,
            name,
            layer.offset,
            layer.size,
            if layer.required == 1 { "yes" } else { "no " },
            hex_short(&layer.content_hash)
        );
    }
    println!();

    // --- Phase 3: Simulate progressive bootstrap ---
    println!("[Phase 3] Simulating progressive bootstrap...\n");

    let mut progress = BootstrapProgress::new(&manifest);

    println!("  Boot phase: {} | query_ready: {} | recall: {:.0}%",
        phase_name(progress.phase), progress.query_ready, progress.estimated_recall * 100.0);

    for layer in &manifest.layers {
        progress.record_layer(layer);
        println!(
            "  Downloaded {:<20} | phase: {} | query_ready: {} | recall: {:.0}% | progress: {:.1}%",
            layer_name(layer.layer_id),
            phase_name(progress.phase),
            progress.query_ready,
            progress.estimated_recall * 100.0,
            progress.progress_fraction() * 100.0
        );
    }

    println!("\n=== Seed bootstrapped to full intelligence ===");
    println!("  The AI that lived in printed ink now spans {} bytes.", manifest.total_file_size.unwrap_or(0));
}

fn phase_name(phase: u8) -> &'static str {
    match phase {
        0 => "Parse  ",
        1 => "Stream ",
        2 => "Full   ",
        _ => "Unknown",
    }
}

fn layer_name(id: u8) -> &'static str {
    match id {
        0 => "Level 0 manifest",
        1 => "Hot cache",
        2 => "HNSW Layer A",
        3 => "Quant dictionaries",
        4 => "HNSW Layer B",
        5 => "Full vectors",
        6 => "HNSW Layer C",
        _ => "Unknown",
    }
}

fn hex_short(bytes: &[u8]) -> String {
    bytes
        .iter()
        .take(4)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join("")
        + ".."
}
