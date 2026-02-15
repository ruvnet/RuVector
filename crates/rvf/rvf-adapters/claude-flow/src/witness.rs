//! Witness (audit trail) handling for Claude-Flow RVF adapter.
//!
//! Provides tamper-evident audit trails via WITNESS_SEG with hash-chained
//! entries using SHAKE-256. Records agent actions for compliance and
//! reproducibility.

use std::io::Write;

use rvf_crypto::shake256_256;
use rvf_crypto::witness::{create_witness_chain, verify_witness_chain, WitnessEntry};

use crate::config::ClaudeFlowConfig;

/// Witness types specific to Claude-Flow operations.
pub mod types {
    /// Agent memory ingestion.
    pub const INGEST_MEMORY: u8 = 0x20;
    /// Agent memory search/query.
    pub const SEARCH_MEMORY: u8 = 0x21;
    /// Agent action (generic).
    pub const AGENT_ACTION: u8 = 0x22;
    /// Session start/end.
    pub const SESSION_EVENT: u8 = 0x23;
}

/// Writer for WITNESS_SEG audit trails.
pub struct WitnessWriter {
    config: ClaudeFlowConfig,
    entries: Vec<WitnessEntry>,
    last_hash: [u8; 32],
}

impl WitnessWriter {
    /// Create a new witness writer.
    pub fn new(config: ClaudeFlowConfig) -> Self {
        Self {
            config,
            entries: Vec::new(),
            last_hash: [0u8; 32],
        }
    }

    /// Create/open the underlying witness file.
    pub fn create(config: &ClaudeFlowConfig) -> Result<Self, WitnessError> {
        config
            .ensure_dirs()
            .map_err(|e| WitnessError::Io(e.to_string()))?;
        Ok(Self::new(config.clone()))
    }

    /// Open an existing witness file and load its chain.
    pub fn open(config: &ClaudeFlowConfig) -> Result<Self, WitnessError> {
        let path = config.witness_path();
        if !path.exists() {
            return Self::create(config);
        }
        let data = std::fs::read(&path).map_err(|e| WitnessError::Io(e.to_string()))?;
        let entries = verify_witness_chain(&data).map_err(WitnessError::Chain)?;
        let last_hash = entries
            .last()
            .map(|e| {
                let encoded = encode_entry(e);
                shake256_256(&encoded)
            })
            .unwrap_or([0u8; 32]);
        Ok(Self {
            config: config.clone(),
            entries,
            last_hash,
        })
    }

    /// Record an action with optional arguments.
    pub fn record_action(
        &mut self,
        action: &str,
        args: &[&str],
    ) -> Result<(), WitnessError> {
        self.record_with_type(types::AGENT_ACTION, action, args)
    }

    /// Record a memory ingestion event.
    pub fn record_ingest(
        &mut self,
        key: &str,
        value: &str,
        namespace: Option<&str>,
    ) -> Result<(), WitnessError> {
        let payload = format!("{}|{}|{}", key, value, namespace.unwrap_or(""));
        self.record_with_type(types::INGEST_MEMORY, "ingest_memory", &[&payload])
    }

    /// Record a memory search event.
    pub fn record_search(&mut self, query_k: usize) -> Result<(), WitnessError> {
        let payload = query_k.to_string();
        self.record_with_type(types::SEARCH_MEMORY, "search_memory", &[&payload])
    }

    /// Record a session event.
    pub fn record_session(&mut self, event: &str) -> Result<(), WitnessError> {
        self.record_with_type(types::SESSION_EVENT, event, &[])
    }

    /// Internal helper to record with a specific witness type.
    fn record_with_type(
        &mut self,
        witness_type: u8,
        action: &str,
        args: &[&str],
    ) -> Result<(), WitnessError> {
        let action_payload = if args.is_empty() {
            action.to_string()
        } else {
            format!("{}:{}", action, args.join(","))
        };
        let action_hash = shake256_256(action_payload.as_bytes());
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let entry = WitnessEntry {
            prev_hash: self.last_hash,
            action_hash,
            timestamp_ns,
            witness_type,
        };
        self.last_hash = shake256_256(&encode_entry(&entry));
        self.entries.push(entry);
        self.flush()?;
        Ok(())
    }

    /// Flush the in-memory chain to disk.
    pub fn flush(&mut self) -> Result<(), WitnessError> {
        let chain_bytes = create_witness_chain(&self.entries);
        let path = self.config.witness_path();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| WitnessError::Io(e.to_string()))?;
        file.write_all(&chain_bytes)
            .map_err(|e| WitnessError::Io(e.to_string()))?;
        file.sync_all()
            .map_err(|e| WitnessError::Io(e.to_string()))?;
        Ok(())
    }

    /// Verify the integrity of the loaded chain.
    pub fn verify(&self) -> Result<(), WitnessError> {
        let chain_bytes = create_witness_chain(&self.entries);
        verify_witness_chain(&chain_bytes)
            .map(|_| ())
            .map_err(WitnessError::Chain)
    }

    /// Iterate over entries.
    pub fn entries(&self) -> impl Iterator<Item = &WitnessEntry> {
        self.entries.iter()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Errors from witness operations.
#[derive(Debug)]
pub enum WitnessError {
    /// I/O error.
    Io(String),
    /// Witness chain verification error.
    Chain(rvf_types::RvfError),
}

impl std::fmt::Display for WitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(msg) => write!(f, "witness I/O error: {msg}"),
            Self::Chain(e) => write!(f, "witness chain error: {e}"),
        }
    }
}

impl std::error::Error for WitnessError {}

/// Encode a WitnessEntry into 73 bytes.
fn encode_entry(entry: &WitnessEntry) -> [u8; 73] {
    let mut buf = [0u8; 73];
    buf[0..32].copy_from_slice(&entry.prev_hash);
    buf[32..64].copy_from_slice(&entry.action_hash);
    buf[64..72].copy_from_slice(&entry.timestamp_ns.to_le_bytes());
    buf[72] = entry.witness_type;
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_config(dir: &TempDir) -> ClaudeFlowConfig {
        ClaudeFlowConfig::new(dir.path(), "test-agent")
    }

    #[test]
    fn create_and_record() {
        let dir = TempDir::new().unwrap();
        let config = make_test_config(&dir);
        let mut writer = WitnessWriter::create(&config).unwrap();
        assert!(writer.is_empty());

        writer.record_action("test-action", &[]).unwrap();
        assert_eq!(writer.len(), 1);
        assert!(writer.verify().is_ok());
    }

    #[test]
    fn record_and_reload() {
        let dir = TempDir::new().unwrap();
        let config = make_test_config(&dir);

        {
            let mut w = WitnessWriter::create(&config).unwrap();
            w.record_action("alpha", &[]).unwrap();
            w.record_action("beta", &["arg1"]).unwrap();
            w.flush().unwrap();
        }

        let w2 = WitnessWriter::open(&config).unwrap();
        assert_eq!(w2.len(), 2);
        assert!(w2.verify().is_ok());
        let entries: Vec<_> = w2.entries().collect();
        assert_eq!(entries[0].witness_type, types::AGENT_ACTION);
        assert_eq!(entries[1].witness_type, types::AGENT_ACTION);
    }

    #[test]
    fn ingest_and_search_events() {
        let dir = TempDir::new().unwrap();
        let config = make_test_config(&dir);
        let mut w = WitnessWriter::create(&config).unwrap();

        w.record_ingest("key1", "value1", Some("ns")).unwrap();
        w.record_search(5).unwrap();
        w.record_session("start").unwrap();

        assert_eq!(w.len(), 3);
        let witness_types: Vec<_> = w.entries().map(|e| e.witness_type).collect();
        assert_eq!(
            witness_types,
            [types::INGEST_MEMORY, types::SEARCH_MEMORY, types::SESSION_EVENT]
        );
    }

    #[test]
    fn tampering_detected() {
        let dir = TempDir::new().unwrap();
        let config = make_test_config(&dir);
        let mut w = WitnessWriter::create(&config).unwrap();
        w.record_action("a", &[]).unwrap();
        w.record_action("b", &[]).unwrap();
        w.flush().unwrap();

        // Corrupt the file
        let path = config.witness_path();
        let mut data = std::fs::read(&path).unwrap();
        data[73] ^= 0xFF; // flip a byte in the second entry
        std::fs::write(&path, &data).unwrap();

        // open() validates the chain, so corruption is caught on load
        let result = WitnessWriter::open(&config);
        assert!(result.is_err() || result.unwrap().verify().is_err());
    }
}
