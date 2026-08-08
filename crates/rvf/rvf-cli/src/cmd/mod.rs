pub mod compact;
pub mod create;
pub mod delete;
pub mod derive;
pub mod embed_ebpf;
pub mod embed_kernel;
pub mod filter;
pub mod freeze;
pub mod ingest;
pub mod inspect;
pub mod launch;
pub mod query;
pub mod rebuild_refcounts;
pub mod serve;
pub mod status;
pub mod verify_attestation;
pub mod verify_witness;

/// Convert an RvfError into a boxed std::error::Error.
///
/// RvfError implements Display but not std::error::Error (it is no_std),
/// so we wrap it in a std::io::Error for CLI error propagation.
pub fn map_rvf_err(e: rvf_types::RvfError) -> Box<dyn std::error::Error> {
    Box::new(std::io::Error::other(format!("{e}")))
}

/// Warn on stderr when a store opened with a damaged metadata chain.
///
/// Opening such an artifact succeeds and serves the longest replayable prefix
/// (ADR-280 §5), so without this the loss is invisible at the command line.
/// Every command that opens a store should call this immediately afterwards.
pub fn warn_on_metadata_recovery(store: &rvf_runtime::RvfStore, path: &str) {
    let recovery = store.metadata_recovery();
    if recovery.dropped_generations == 0 && recovery.dropped_records == 0 {
        return;
    }
    eprintln!(
        "warning: {path}: metadata is damaged -- recovered generation {}, \
         dropped {} generation(s) and {} record(s). \
         Run `rvf compact {path}` to rewrite the file around the recovered state.",
        recovery.generation, recovery.dropped_generations, recovery.dropped_records
    );
}
