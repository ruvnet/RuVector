//! Task execution system with sandboxing and verification

use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use rand::rngs::OsRng;
use sha2::{Sha256, Digest};

/// Task types supported by the network
#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum TaskType {
    /// Vector search in HNSW index
    VectorSearch,
    /// Vector insertion
    VectorInsert,
    /// Generate embeddings
    Embedding,
    /// Semantic task-to-agent matching
    SemanticMatch,
    /// Neural network inference
    NeuralInference,
    /// AES encryption/decryption
    Encryption,
    /// Data compression
    Compression,
    /// Custom WASM module (requires verification)
    CustomWasm,
}

/// Task priority levels
#[wasm_bindgen]
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
}

/// A task submitted to the network
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Task {
    pub id: String,
    pub task_type: TaskType,
    pub encrypted_payload: Vec<u8>,
    pub payload_hash: [u8; 32],
    pub submitter_id: String,
    pub submitter_pubkey: Vec<u8>,
    pub priority: TaskPriority,
    pub base_reward: u64,
    pub max_credits: u64,
    pub redundancy: u8,
    pub created_at: u64,
    pub expires_at: u64,
    pub signature: Vec<u8>,
}

/// Result of task execution
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TaskResult {
    pub task_id: String,
    pub encrypted_result: Vec<u8>,
    pub result_hash: [u8; 32],
    pub worker_id: String,
    pub execution_time_ms: u64,
    pub signature: Vec<u8>,
    pub proof: ExecutionProof,
}

/// Proof of correct execution
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct ExecutionProof {
    /// Hash of input + output (for spot-checking)
    pub io_hash: [u8; 32],
    /// Intermediate state hashes (for verification)
    pub checkpoints: Vec<[u8; 32]>,
    /// Random challenge response (if spot-check)
    pub challenge_response: Option<Vec<u8>>,
}

/// Sandboxed task executor
#[wasm_bindgen]
pub struct WasmTaskExecutor {
    /// Maximum memory for task execution
    max_memory: usize,
    /// Maximum execution time in ms
    max_time_ms: u64,
    /// Encryption key for task payloads
    task_key: Option<Vec<u8>>,
}

#[wasm_bindgen]
impl WasmTaskExecutor {
    /// Create a new task executor
    #[wasm_bindgen(constructor)]
    pub fn new(max_memory: usize) -> Result<WasmTaskExecutor, JsValue> {
        Ok(WasmTaskExecutor {
            max_memory,
            max_time_ms: 30_000, // 30 seconds default
            task_key: None,
        })
    }

    /// Set encryption key for payload decryption
    #[wasm_bindgen(js_name = setTaskKey)]
    pub fn set_task_key(&mut self, key: &[u8]) -> Result<(), JsValue> {
        if key.len() != 32 {
            return Err(JsValue::from_str("Key must be 32 bytes"));
        }
        self.task_key = Some(key.to_vec());
        Ok(())
    }
}

// Non-wasm methods (internal use)
impl WasmTaskExecutor {
    /// Execute a task with full sandboxing
    pub async fn execute(&self, task: &Task) -> Result<TaskResult, JsValue> {
        // Validate task hasn't expired
        let now = js_sys::Date::now() as u64;
        if now > task.expires_at {
            return Err(JsValue::from_str("Task has expired"));
        }

        // Decrypt payload
        let payload = self.decrypt_payload(&task.encrypted_payload)?;

        // Verify payload hash
        let mut hasher = Sha256::new();
        hasher.update(&payload);
        let hash: [u8; 32] = hasher.finalize().into();
        if hash != task.payload_hash {
            return Err(JsValue::from_str("Payload hash mismatch - tampering detected"));
        }

        // Execute based on task type (with timeout)
        let start = js_sys::Date::now() as u64;
        let result = match task.task_type {
            TaskType::VectorSearch => self.execute_vector_search(&payload).await?,
            TaskType::VectorInsert => self.execute_vector_insert(&payload).await?,
            TaskType::Embedding => self.execute_embedding(&payload).await?,
            TaskType::SemanticMatch => self.execute_semantic_match(&payload).await?,
            TaskType::Encryption => self.execute_encryption(&payload).await?,
            TaskType::Compression => self.execute_compression(&payload).await?,
            TaskType::NeuralInference => self.execute_neural(&payload).await?,
            TaskType::CustomWasm => {
                return Err(JsValue::from_str("Custom WASM requires explicit verification"));
            }
        };
        let execution_time = (js_sys::Date::now() as u64) - start;

        // Create execution proof
        let mut io_hasher = Sha256::new();
        io_hasher.update(&payload);
        io_hasher.update(&result);
        let io_hash: [u8; 32] = io_hasher.finalize().into();

        // Encrypt result
        let encrypted_result = self.encrypt_payload(&result, &task.submitter_pubkey)?;

        // Hash result
        let mut result_hasher = Sha256::new();
        result_hasher.update(&result);
        let result_hash: [u8; 32] = result_hasher.finalize().into();

        Ok(TaskResult {
            task_id: task.id.clone(),
            encrypted_result,
            result_hash,
            worker_id: String::new(), // Set by caller
            execution_time_ms: execution_time,
            signature: Vec::new(), // Set by caller
            proof: ExecutionProof {
                io_hash,
                checkpoints: Vec::new(),
                challenge_response: None,
            },
        })
    }

    /// Decrypt task payload
    fn decrypt_payload(&self, encrypted: &[u8]) -> Result<Vec<u8>, JsValue> {
        let key = self.task_key.as_ref()
            .ok_or_else(|| JsValue::from_str("No task key set"))?;

        if encrypted.len() < 12 {
            return Err(JsValue::from_str("Invalid encrypted payload"));
        }

        let (nonce_bytes, ciphertext) = encrypted.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let key_array: [u8; 32] = key.clone().try_into()
            .map_err(|_| JsValue::from_str("Invalid key length"))?;
        let cipher = Aes256Gcm::new_from_slice(&key_array)
            .map_err(|_| JsValue::from_str("Failed to create cipher"))?;

        cipher.decrypt(nonce, ciphertext)
            .map_err(|_| JsValue::from_str("Decryption failed - invalid key or tampered data"))
    }

    /// Encrypt result for submitter
    fn encrypt_payload(&self, plaintext: &[u8], _recipient_pubkey: &[u8]) -> Result<Vec<u8>, JsValue> {
        // For now, use symmetric encryption (would use ECDH in production)
        let key = self.task_key.as_ref()
            .ok_or_else(|| JsValue::from_str("No task key set"))?;

        let key_array: [u8; 32] = key.clone().try_into()
            .map_err(|_| JsValue::from_str("Invalid key length"))?;
        let cipher = Aes256Gcm::new_from_slice(&key_array)
            .map_err(|_| JsValue::from_str("Failed to create cipher"))?;

        // Generate random nonce
        let mut nonce_bytes = [0u8; 12];
        getrandom::getrandom(&mut nonce_bytes)
            .map_err(|_| JsValue::from_str("Failed to generate nonce"))?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher.encrypt(nonce, plaintext)
            .map_err(|_| JsValue::from_str("Encryption failed"))?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend(ciphertext);
        Ok(result)
    }

    // Task executors (stubs - would integrate with actual WASM modules)

    async fn execute_vector_search(&self, _payload: &[u8]) -> Result<Vec<u8>, JsValue> {
        // Would call WasmHnswIndex.search()
        Ok(vec![])
    }

    async fn execute_vector_insert(&self, _payload: &[u8]) -> Result<Vec<u8>, JsValue> {
        Ok(vec![])
    }

    async fn execute_embedding(&self, _payload: &[u8]) -> Result<Vec<u8>, JsValue> {
        Ok(vec![])
    }

    async fn execute_semantic_match(&self, _payload: &[u8]) -> Result<Vec<u8>, JsValue> {
        Ok(vec![])
    }

    async fn execute_encryption(&self, _payload: &[u8]) -> Result<Vec<u8>, JsValue> {
        Ok(vec![])
    }

    async fn execute_compression(&self, _payload: &[u8]) -> Result<Vec<u8>, JsValue> {
        Ok(vec![])
    }

    async fn execute_neural(&self, _payload: &[u8]) -> Result<Vec<u8>, JsValue> {
        Ok(vec![])
    }
}

/// Task queue for P2P distribution
#[wasm_bindgen]
pub struct WasmTaskQueue {
    pending: Vec<Task>,
    claimed: std::collections::HashMap<String, String>, // task_id -> worker_id
}

impl WasmTaskQueue {
    pub fn new() -> Result<WasmTaskQueue, JsValue> {
        Ok(WasmTaskQueue {
            pending: Vec::new(),
            claimed: std::collections::HashMap::new(),
        })
    }

    /// Create a task for submission
    pub fn create_task(
        &self,
        task_type: &str,
        payload: &[u8],
        max_credits: u64,
        identity: &crate::identity::WasmNodeIdentity,
    ) -> Result<Task, JsValue> {
        let task_type = match task_type {
            "vectors" | "vector_search" => TaskType::VectorSearch,
            "vector_insert" => TaskType::VectorInsert,
            "embeddings" | "embedding" => TaskType::Embedding,
            "semantic" | "semantic_match" => TaskType::SemanticMatch,
            "neural" | "neural_inference" => TaskType::NeuralInference,
            "encryption" => TaskType::Encryption,
            "compression" => TaskType::Compression,
            _ => return Err(JsValue::from_str("Unknown task type")),
        };

        // Hash payload
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let payload_hash: [u8; 32] = hasher.finalize().into();

        let now = js_sys::Date::now() as u64;

        let task = Task {
            id: Uuid::new_v4().to_string(),
            task_type,
            encrypted_payload: Vec::new(), // Set after encryption
            payload_hash,
            submitter_id: identity.node_id(),
            submitter_pubkey: identity.public_key_bytes(),
            priority: TaskPriority::Normal,
            base_reward: calculate_base_reward(task_type, payload.len()),
            max_credits,
            redundancy: 3,
            created_at: now,
            expires_at: now + 60_000, // 1 minute default
            signature: Vec::new(), // Set after signing
        };

        Ok(task)
    }

    /// Submit task to network
    pub async fn submit(&mut self, task: Task) -> Result<SubmitResult, JsValue> {
        self.pending.push(task.clone());

        Ok(SubmitResult {
            task_id: task.id,
            cost: task.base_reward,
        })
    }

    /// Claim next available task
    pub async fn claim_next(
        &mut self,
        identity: &crate::identity::WasmNodeIdentity,
    ) -> Result<Option<Task>, JsValue> {
        // Find unclaimed task
        for task in &self.pending {
            if !self.claimed.contains_key(&task.id) {
                self.claimed.insert(task.id.clone(), identity.node_id());
                return Ok(Some(task.clone()));
            }
        }
        Ok(None)
    }

    /// Complete a task
    pub async fn complete(
        &mut self,
        task_id: String,
        _result: TaskResult,
        _identity: &crate::identity::WasmNodeIdentity,
    ) -> Result<(), JsValue> {
        // Remove from pending
        self.pending.retain(|t| t.id != task_id);
        self.claimed.remove(&task_id);
        Ok(())
    }

    /// Disconnect from network
    pub fn disconnect(&self) -> Result<(), JsValue> {
        Ok(())
    }
}

pub struct SubmitResult {
    pub task_id: String,
    pub cost: u64,
}

impl From<SubmitResult> for JsValue {
    fn from(result: SubmitResult) -> Self {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"taskId".into(), &result.task_id.into()).unwrap();
        js_sys::Reflect::set(&obj, &"cost".into(), &result.cost.into()).unwrap();
        obj.into()
    }
}

/// Calculate base reward based on task type and size
fn calculate_base_reward(task_type: TaskType, payload_size: usize) -> u64 {
    match task_type {
        TaskType::VectorSearch => 1 + (payload_size / 10000) as u64,
        TaskType::VectorInsert => 1 + (payload_size / 20000) as u64,
        TaskType::Embedding => 5 + (payload_size / 1000) as u64,
        TaskType::SemanticMatch => 1,
        TaskType::NeuralInference => 3 + (payload_size / 5000) as u64,
        TaskType::Encryption => 1 + (payload_size / 100000) as u64,
        TaskType::Compression => 1 + (payload_size / 50000) as u64,
        TaskType::CustomWasm => 10, // Premium for custom code
    }
}
