//! # RVF Swarm Consensus Demo
//!
//! This demo shows how RuVix enables native multi-agent swarm coordination
//! with Byzantine fault-tolerant consensus.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                     SWARM COORDINATOR                        │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
//! │  │ Coordinator │──│  Consensus  │──│  Broadcast  │          │
//! │  │   (Leader)  │  │    Engine   │  │   Channel   │          │
//! │  └─────────────┘  └─────────────┘  └──────┬──────┘          │
//! │                                           │                  │
//! │  ┌────────────────────────────────────────┼────────────────┐ │
//! │  │                 VALIDATORS (BFT)       │                │ │
//! │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │ │
//! │  │  │Validator1│  │Validator2│  │Validator3│  (f=1, n=3)   │ │
//! │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘               │ │
//! │  │       │             │             │                      │ │
//! │  │       └─────────────┼─────────────┘                      │ │
//! │  │                     │ VOTE                               │ │
//! │  └─────────────────────┼────────────────────────────────────┘ │
//! │                        │                                      │
//! │  ┌─────────────────────┼────────────────────────────────────┐ │
//! │  │                 WORKERS                                   │ │
//! │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐          │ │
//! │  │  │Worker 1│  │Worker 2│  │Worker 3│  │Worker 4│          │ │
//! │  │  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘          │ │
//! │  │      └───────────┴───────────┴───────────┘                │ │
//! │  │                     │                                     │ │
//! │  └─────────────────────┼─────────────────────────────────────┘ │
//! │                        ▼                                      │
//! │  ┌──────────────────────────────────────────────────────────┐ │
//! │  │              SHARED VECTOR MEMORY                         │ │
//! │  │   [Proof-gated mutations] [Witness-logged] [Immutable]    │ │
//! │  └──────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Features Demonstrated
//!
//! 1. **Capability-based agent isolation** - Each agent type has specific capabilities
//! 2. **Proof-gated consensus** - Votes require cryptographic proofs
//! 3. **Queue-based IPC** - Zero-copy message passing between agents
//! 4. **Shared vector memory** - Native vector stores for agent coordination
//! 5. **Witness logging** - All consensus decisions are tamper-evident
//!
//! ## Running
//!
//! ```bash
//! # On RuVix kernel (QEMU or real hardware)
//! ruvix run --rvf swarm.rvf.json
//!
//! # Or in hosted mode for testing
//! cargo run --example swarm-consensus
//! ```

use ruvix_nucleus::{Kernel, KernelConfig, Syscall, SyscallResult, ProofTier};
use ruvix_types::{TaskHandle, QueueHandle, VectorKey, CapabilityRights};

/// Message types for swarm communication
#[derive(Debug, Clone)]
pub enum SwarmMessage {
    /// Task proposal from coordinator
    Propose {
        proposal_id: u64,
        task_hash: [u8; 32],
        distribution: Vec<WorkerAssignment>,
    },
    /// Vote from validator
    Vote {
        proposal_id: u64,
        validator_id: u8,
        approve: bool,
        signature: [u8; 64],
    },
    /// Commit notification (consensus reached)
    Commit {
        proposal_id: u64,
        proof: [u8; 32],
    },
    /// Task assignment to worker
    Execute {
        task_id: u64,
        input_vector_key: VectorKey,
        output_vector_key: VectorKey,
    },
    /// Result from worker
    Result {
        task_id: u64,
        worker_id: u8,
        output_vector_key: VectorKey,
        proof: [u8; 32],
    },
}

/// Worker task assignment
#[derive(Debug, Clone)]
pub struct WorkerAssignment {
    pub worker_id: u8,
    pub task_slice: (usize, usize),
}

/// Consensus state for PBFT
#[derive(Debug, Default)]
pub struct ConsensusState {
    pub current_view: u64,
    pub proposal_id: u64,
    pub votes: Vec<(u8, bool, [u8; 64])>,
    pub committed: bool,
}

impl ConsensusState {
    /// Check if quorum is reached (2f+1 for f=1, n=3)
    pub fn has_quorum(&self) -> bool {
        let approvals = self.votes.iter().filter(|(_, approve, _)| *approve).count();
        approvals >= 2 // 2f+1 where f=1
    }
}

/// Coordinator agent - proposes and commits
pub struct Coordinator {
    task: TaskHandle,
    inbox: QueueHandle,
    broadcast: QueueHandle,
    consensus: ConsensusState,
}

impl Coordinator {
    pub fn new(kernel: &mut Kernel) -> Result<Self, &'static str> {
        // Spawn coordinator task with leader capabilities
        let task = kernel.spawn_task(
            CapabilityRights::TASK_SPAWN | CapabilityRights::QUEUE_SEND | CapabilityRights::PROOF_EMIT,
            10, // High priority
        )?;

        // Create communication queues
        let inbox = kernel.create_queue(256)?;
        let broadcast = kernel.create_queue(256)?;

        Ok(Self {
            task,
            inbox,
            broadcast,
            consensus: ConsensusState::default(),
        })
    }

    /// Propose a new task distribution
    pub fn propose(&mut self, kernel: &mut Kernel, task_hash: [u8; 32], workers: &[u8]) -> Result<u64, &'static str> {
        self.consensus.proposal_id += 1;
        let proposal_id = self.consensus.proposal_id;

        // Create distribution plan
        let distribution: Vec<WorkerAssignment> = workers
            .iter()
            .enumerate()
            .map(|(i, &worker_id)| WorkerAssignment {
                worker_id,
                task_slice: (i * 100, (i + 1) * 100),
            })
            .collect();

        // Create proof for proposal
        let proof = kernel.create_proof(task_hash, ProofTier::Reflex, proposal_id as u32)?;

        // Broadcast proposal to validators
        let msg = SwarmMessage::Propose {
            proposal_id,
            task_hash,
            distribution,
        };

        kernel.dispatch(Syscall::QueueSend {
            queue: self.broadcast,
            message: serialize_message(&msg),
            proof: Some(proof),
        })?;

        println!("[Coordinator] Proposed task {} for consensus", proposal_id);
        Ok(proposal_id)
    }

    /// Process votes and commit if quorum reached
    pub fn process_votes(&mut self, kernel: &mut Kernel) -> Result<bool, &'static str> {
        // Receive votes from inbox
        while let SyscallResult::MessageReceived { data, .. } = kernel.dispatch(Syscall::QueueRecv {
            queue: self.inbox,
            timeout_ns: Some(1_000_000), // 1ms timeout
        })? {
            if let Some(SwarmMessage::Vote { proposal_id, validator_id, approve, signature }) =
                deserialize_message(&data)
            {
                if proposal_id == self.consensus.proposal_id {
                    self.consensus.votes.push((validator_id, approve, signature));
                    println!("[Coordinator] Received vote from validator {}: {}", validator_id, approve);
                }
            }
        }

        // Check quorum
        if self.consensus.has_quorum() && !self.consensus.committed {
            self.commit(kernel)?;
            return Ok(true);
        }

        Ok(false)
    }

    fn commit(&mut self, kernel: &mut Kernel) -> Result<(), &'static str> {
        self.consensus.committed = true;

        // Create commit proof
        let commit_hash = [0u8; 32]; // Would be hash of votes
        let proof = kernel.create_proof(commit_hash, ProofTier::Standard, self.consensus.proposal_id as u32)?;

        // Broadcast commit
        let msg = SwarmMessage::Commit {
            proposal_id: self.consensus.proposal_id,
            proof: proof.hash,
        };

        kernel.dispatch(Syscall::QueueSend {
            queue: self.broadcast,
            message: serialize_message(&msg),
            proof: Some(proof),
        })?;

        // Emit attestation record
        kernel.dispatch(Syscall::AttestEmit {
            record_type: "consensus_commit",
            data: &commit_hash,
            proof,
        })?;

        println!("[Coordinator] Consensus reached! Committed proposal {}", self.consensus.proposal_id);
        Ok(())
    }
}

/// Validator agent - votes on proposals
pub struct Validator {
    id: u8,
    task: TaskHandle,
    inbox: QueueHandle,
    coordinator_queue: QueueHandle,
}

impl Validator {
    pub fn new(kernel: &mut Kernel, id: u8, coordinator_queue: QueueHandle) -> Result<Self, &'static str> {
        let task = kernel.spawn_task(
            CapabilityRights::QUEUE_RECV | CapabilityRights::QUEUE_SEND | CapabilityRights::PROOF_EMIT,
            8, // Medium-high priority
        )?;

        let inbox = kernel.create_queue(64)?;

        Ok(Self {
            id,
            task,
            inbox,
            coordinator_queue,
        })
    }

    /// Process incoming proposals and vote
    pub fn process(&mut self, kernel: &mut Kernel) -> Result<(), &'static str> {
        if let SyscallResult::MessageReceived { data, .. } = kernel.dispatch(Syscall::QueueRecv {
            queue: self.inbox,
            timeout_ns: Some(10_000_000), // 10ms timeout
        })? {
            if let Some(SwarmMessage::Propose { proposal_id, task_hash, .. }) = deserialize_message(&data) {
                // Validate the proposal (simplified)
                let valid = self.validate_proposal(&task_hash);

                // Create vote with proof
                let vote_hash = [self.id; 32]; // Simplified
                let proof = kernel.create_proof(vote_hash, ProofTier::Reflex, proposal_id as u32)?;

                let signature = [self.id; 64]; // Would be real Ed25519 signature

                let vote = SwarmMessage::Vote {
                    proposal_id,
                    validator_id: self.id,
                    approve: valid,
                    signature,
                };

                kernel.dispatch(Syscall::QueueSend {
                    queue: self.coordinator_queue,
                    message: serialize_message(&vote),
                    proof: Some(proof),
                })?;

                println!("[Validator {}] Voted {} on proposal {}", self.id, valid, proposal_id);
            }
        }

        Ok(())
    }

    fn validate_proposal(&self, task_hash: &[u8; 32]) -> bool {
        // In real implementation: verify task integrity, check resources, etc.
        task_hash[0] != 0 // Simple validation
    }
}

/// Worker agent - executes tasks
pub struct Worker {
    id: u8,
    task: TaskHandle,
    inbox: QueueHandle,
    result_queue: QueueHandle,
    vector_store: ruvix_nucleus::VectorStoreHandle,
}

impl Worker {
    pub fn new(
        kernel: &mut Kernel,
        id: u8,
        result_queue: QueueHandle,
        vector_store: ruvix_nucleus::VectorStoreHandle,
    ) -> Result<Self, &'static str> {
        let task = kernel.spawn_task(
            CapabilityRights::QUEUE_RECV | CapabilityRights::VECTOR_READ | CapabilityRights::VECTOR_WRITE,
            5, // Normal priority
        )?;

        let inbox = kernel.create_queue(32)?;

        Ok(Self {
            id,
            task,
            inbox,
            result_queue,
            vector_store,
        })
    }

    /// Execute assigned task
    pub fn execute(&mut self, kernel: &mut Kernel) -> Result<(), &'static str> {
        if let SyscallResult::MessageReceived { data, .. } = kernel.dispatch(Syscall::QueueRecv {
            queue: self.inbox,
            timeout_ns: Some(100_000_000), // 100ms timeout
        })? {
            if let Some(SwarmMessage::Execute { task_id, input_vector_key, output_vector_key }) =
                deserialize_message(&data)
            {
                println!("[Worker {}] Executing task {}", self.id, task_id);

                // Read input vector
                let input = kernel.dispatch(Syscall::VectorGet {
                    store: self.vector_store,
                    key: input_vector_key,
                })?;

                // Process (simplified - would be real inference/computation)
                let output = self.process_vector(&input);

                // Create proof for output
                let output_hash = [0u8; 32]; // Would be hash of output
                let proof = kernel.create_proof(output_hash, ProofTier::Standard, task_id as u32)?;

                // Store result vector
                kernel.dispatch(Syscall::VectorPutProved {
                    store: self.vector_store,
                    key: output_vector_key,
                    data: output,
                    proof: proof.clone(),
                })?;

                // Report completion
                let result = SwarmMessage::Result {
                    task_id,
                    worker_id: self.id,
                    output_vector_key,
                    proof: proof.hash,
                };

                kernel.dispatch(Syscall::QueueSend {
                    queue: self.result_queue,
                    message: serialize_message(&result),
                    proof: Some(proof),
                })?;

                println!("[Worker {}] Completed task {}", self.id, task_id);
            }
        }

        Ok(())
    }

    fn process_vector(&self, _input: &SyscallResult) -> Vec<f32> {
        // Simulated processing
        vec![1.0, 2.0, 3.0, 4.0]
    }
}

// Serialization helpers (simplified)
fn serialize_message(_msg: &SwarmMessage) -> Vec<u8> {
    vec![0u8; 64] // Would be real serialization
}

fn deserialize_message(_data: &[u8]) -> Option<SwarmMessage> {
    None // Would be real deserialization
}

/// Main swarm entry point
pub fn swarm_main() {
    println!("=== RuVix Swarm Consensus Demo ===\n");

    // Initialize kernel
    let mut kernel = Kernel::new(KernelConfig {
        max_tasks: 16,
        max_queues: 32,
        max_vector_stores: 4,
        ..Default::default()
    });

    kernel.boot(0, [0u8; 32]).expect("Kernel boot failed");
    println!("[Kernel] Booted successfully\n");

    // Create shared vector store
    let vector_store = kernel.create_vector_store(
        ruvix_nucleus::VectorStoreConfig::new(128, 10000)
    ).expect("Failed to create vector store");

    // Create result collection queue
    let result_queue = kernel.create_queue(1024).expect("Failed to create result queue");

    // Spawn coordinator
    let mut coordinator = Coordinator::new(&mut kernel).expect("Failed to spawn coordinator");
    println!("[Swarm] Coordinator spawned");

    // Spawn validators (3 for BFT with f=1)
    let mut validators: Vec<Validator> = (0..3)
        .map(|i| Validator::new(&mut kernel, i as u8, coordinator.inbox).expect("Failed to spawn validator"))
        .collect();
    println!("[Swarm] {} validators spawned", validators.len());

    // Spawn workers
    let mut workers: Vec<Worker> = (0..4)
        .map(|i| Worker::new(&mut kernel, i as u8, result_queue, vector_store).expect("Failed to spawn worker"))
        .collect();
    println!("[Swarm] {} workers spawned\n", workers.len());

    // Demo: Submit a task for distributed processing
    println!("--- Starting Consensus Round ---\n");

    let task_hash = [1u8; 32]; // Task identifier
    let worker_ids: Vec<u8> = workers.iter().map(|w| w.id).collect();

    // 1. Coordinator proposes
    let proposal_id = coordinator.propose(&mut kernel, task_hash, &worker_ids)
        .expect("Proposal failed");

    // 2. Validators vote
    for validator in &mut validators {
        validator.process(&mut kernel).expect("Validator processing failed");
    }

    // 3. Coordinator collects votes and commits
    let committed = coordinator.process_votes(&mut kernel).expect("Vote processing failed");

    if committed {
        println!("\n--- Consensus Achieved! ---");
        println!("Proposal {} committed with {} validator votes", proposal_id, validators.len());

        // 4. Workers would now execute their assigned tasks
        println!("\n--- Workers Executing ---");
        for worker in &mut workers {
            // In real system, coordinator would dispatch Execute messages
            println!("[Worker {}] Ready to execute", worker.id);
        }
    }

    println!("\n=== Demo Complete ===");
    println!("This demonstrates RuVix's native support for:");
    println!("  - Capability-based agent isolation");
    println!("  - Proof-gated consensus voting");
    println!("  - Zero-copy queue IPC");
    println!("  - Shared vector memory");
    println!("  - Witness-logged decisions");
}

fn main() {
    swarm_main();
}
