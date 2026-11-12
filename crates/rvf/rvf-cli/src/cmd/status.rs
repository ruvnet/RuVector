//! `rvf status` -- Show store status.

use clap::Args;
#[derive(Args)]
pub struct StatusArgs {
    /// Path to the RVF store
    path: String,
    /// Output as JSON
    #[arg(long)]
    json: bool,
    /// Open the last committed snapshot even if the file ends inside a segment
    #[arg(long)]
    recover: bool,
}

pub fn run(args: StatusArgs) -> Result<(), Box<dyn std::error::Error>> {
    let store = crate::cmd::open_readonly_checked(&args.path, args.recover)?;
    crate::cmd::warn_on_metadata_recovery(&store, &args.path);
    let status = store.status();

    if args.json {
        crate::output::print_json(&serde_json::json!({
            "total_vectors": status.total_vectors,
            "total_segments": status.total_segments,
            "file_size": status.file_size,
            "epoch": status.current_epoch,
            "profile_id": status.profile_id,
            "dead_space_ratio": status.dead_space_ratio,
            "read_only": status.read_only,
        }));
    } else {
        println!("RVF Store: {}", args.path);
        crate::output::print_kv("Vectors:", &status.total_vectors.to_string());
        crate::output::print_kv("Segments:", &status.total_segments.to_string());
        crate::output::print_kv("File size:", &format!("{} bytes", status.file_size));
        crate::output::print_kv("Epoch:", &status.current_epoch.to_string());
        crate::output::print_kv("Profile:", &status.profile_id.to_string());
        crate::output::print_kv(
            "Dead space:",
            &format!("{:.1}%", status.dead_space_ratio * 100.0),
        );
    }
    Ok(())
}
