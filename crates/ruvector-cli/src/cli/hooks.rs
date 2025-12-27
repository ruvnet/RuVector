//! Hooks CLI - Self-learning intelligence system for Claude Code integration
//!
//! Provides Q-learning based agent routing, error pattern recognition,
//! file sequence prediction, and swarm coordination.

use crate::config::Config;
use anyhow::{Context, Result};
use colored::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Hooks subcommands
#[derive(clap::Subcommand, Debug)]
pub enum HooksCommands {
    /// Initialize hooks in current project
    Init {
        /// Force overwrite existing configuration
        #[arg(long)]
        force: bool,
    },

    /// Install hooks into Claude settings
    Install {
        /// Claude settings directory
        #[arg(long, default_value = ".claude")]
        settings_dir: String,
    },

    /// Show intelligence statistics
    Stats,

    // === Memory Commands ===
    /// Store content in semantic memory
    Remember {
        /// Memory type (edit, command, decision, pattern)
        #[arg(short = 't', long)]
        memory_type: String,

        /// Content to remember
        content: Vec<String>,
    },

    /// Search memory semantically
    Recall {
        /// Search query
        query: Vec<String>,

        /// Number of results
        #[arg(short = 'k', long, default_value = "5")]
        top_k: usize,
    },

    // === Learning Commands ===
    /// Record a learning trajectory
    Learn {
        /// State identifier
        state: String,

        /// Action taken
        action: String,

        /// Reward value (-1.0 to 1.0)
        #[arg(short, long, default_value = "0.0")]
        reward: f32,
    },

    /// Get action suggestion for state
    Suggest {
        /// Current state
        state: String,

        /// Available actions (comma-separated)
        #[arg(short, long)]
        actions: String,
    },

    /// Route task to best agent
    Route {
        /// Task description
        task: Vec<String>,

        /// File being worked on
        #[arg(long)]
        file: Option<String>,

        /// Crate/module context
        #[arg(long)]
        crate_name: Option<String>,

        /// Operation type (edit, review, test)
        #[arg(long, default_value = "edit")]
        operation: String,
    },

    // === Hook Integrations ===
    /// Pre-edit intelligence hook
    PreEdit {
        /// File path
        file: String,
    },

    /// Post-edit learning hook
    PostEdit {
        /// File path
        file: String,

        /// Whether edit succeeded
        #[arg(long)]
        success: bool,
    },

    /// Pre-command intelligence hook
    PreCommand {
        /// Command being run
        command: Vec<String>,
    },

    /// Post-command learning hook
    PostCommand {
        /// Command that ran
        command: Vec<String>,

        /// Whether command succeeded
        #[arg(long)]
        success: bool,

        /// Stderr output (for error learning)
        #[arg(long)]
        stderr: Option<String>,
    },

    // === Session Hooks ===
    /// Session start hook
    SessionStart {
        /// Session ID
        #[arg(long)]
        session_id: Option<String>,
    },

    /// Session end hook
    SessionEnd {
        /// Export metrics
        #[arg(long)]
        export_metrics: bool,
    },

    /// Pre-compact hook
    PreCompact {
        /// Conversation length
        #[arg(long)]
        length: Option<usize>,
    },

    // === V3 Intelligence Features ===
    /// Record error pattern for learning
    RecordError {
        /// Command that failed
        command: String,

        /// Stderr output
        stderr: String,
    },

    /// Get suggested fix for error code
    SuggestFix {
        /// Error code (e.g., E0308, TS2322)
        error_code: String,
    },

    /// Suggest next files to edit
    SuggestNext {
        /// Current file
        file: String,

        /// Number of suggestions
        #[arg(short = 'n', long, default_value = "3")]
        count: usize,
    },

    /// Check if tests should run
    ShouldTest {
        /// File that was edited
        file: String,
    },

    // === Swarm Commands ===
    /// Register agent in swarm
    SwarmRegister {
        /// Agent ID
        agent_id: String,

        /// Agent type
        agent_type: String,

        /// Agent capabilities (comma-separated)
        #[arg(long)]
        capabilities: Option<String>,
    },

    /// Record agent coordination
    SwarmCoordinate {
        /// Source agent
        source: String,

        /// Target agent
        target: String,

        /// Coordination weight
        #[arg(long, default_value = "1.0")]
        weight: f32,
    },

    /// Optimize task distribution
    SwarmOptimize {
        /// Tasks to distribute (comma-separated)
        tasks: String,
    },

    /// Recommend agent for task type
    SwarmRecommend {
        /// Task type
        task_type: String,
    },

    /// Handle agent failure
    SwarmHeal {
        /// Failed agent ID
        agent_id: String,
    },

    /// Show swarm statistics
    SwarmStats,
}

// === Data Structures ===

/// Q-learning pattern entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QPattern {
    pub state: String,
    pub action: String,
    pub q_value: f32,
    pub visits: u32,
    pub last_update: u64,
}

/// Memory entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub memory_type: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub timestamp: u64,
}

/// Learning trajectory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub id: String,
    pub state: String,
    pub action: String,
    pub outcome: String,
    pub reward: f32,
    pub timestamp: u64,
}

/// Error pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    pub code: String,
    pub error_type: String,
    pub message: String,
    pub fixes: Vec<String>,
    pub occurrences: u32,
}

/// File edit sequence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSequence {
    pub from_file: String,
    pub to_file: String,
    pub count: u32,
}

/// Swarm agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmAgent {
    pub id: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub success_rate: f32,
    pub task_count: u32,
    pub status: String,
}

/// Swarm edge (coordination)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmEdge {
    pub source: String,
    pub target: String,
    pub weight: f32,
    pub coordination_count: u32,
}

/// Intelligence storage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntelligenceData {
    pub patterns: HashMap<String, QPattern>,
    pub memories: Vec<MemoryEntry>,
    pub trajectories: Vec<Trajectory>,
    pub errors: HashMap<String, ErrorPattern>,
    pub file_sequences: Vec<FileSequence>,
    pub agents: HashMap<String, SwarmAgent>,
    pub edges: Vec<SwarmEdge>,
    pub stats: IntelligenceStats,
}

/// Intelligence statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IntelligenceStats {
    pub total_patterns: u32,
    pub total_memories: u32,
    pub total_trajectories: u32,
    pub total_errors: u32,
    pub session_count: u32,
    pub last_session: u64,
}

/// Intelligence engine
pub struct Intelligence {
    data: IntelligenceData,
    data_path: PathBuf,
    alpha: f32, // Learning rate
    gamma: f32, // Discount factor
    epsilon: f32, // Exploration rate
}

impl Intelligence {
    /// Create new intelligence engine
    pub fn new(data_path: PathBuf) -> Self {
        let data = Self::load_data(&data_path).unwrap_or_default();
        Self {
            data,
            data_path,
            alpha: 0.1,
            gamma: 0.95,
            epsilon: 0.1,
        }
    }

    /// Load data from file
    fn load_data(path: &Path) -> Result<IntelligenceData> {
        if path.exists() {
            let content = fs::read_to_string(path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(IntelligenceData::default())
        }
    }

    /// Save data to file
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.data_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(&self.data)?;
        fs::write(&self.data_path, content)?;
        Ok(())
    }

    /// Get current timestamp
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Generate simple embedding from text (hash-based for speed)
    fn embed(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0f32; 64];
        for (i, c) in text.chars().enumerate() {
            let idx = (c as usize + i * 7) % 64;
            embedding[idx] += 1.0;
        }
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }
        embedding
    }

    /// Cosine similarity between embeddings
    fn similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    // === Memory Operations ===

    /// Remember content
    pub fn remember(&mut self, memory_type: &str, content: &str, metadata: HashMap<String, String>) -> String {
        let id = format!("mem_{}", Self::now());
        let embedding = self.embed(content);

        self.data.memories.push(MemoryEntry {
            id: id.clone(),
            memory_type: memory_type.to_string(),
            content: content.to_string(),
            embedding,
            metadata,
            timestamp: Self::now(),
        });

        // Limit memory size
        if self.data.memories.len() > 5000 {
            self.data.memories.drain(0..1000);
        }

        self.data.stats.total_memories = self.data.memories.len() as u32;
        id
    }

    /// Recall from memory
    pub fn recall(&self, query: &str, top_k: usize) -> Vec<&MemoryEntry> {
        let query_embed = self.embed(query);

        let mut scored: Vec<_> = self.data.memories
            .iter()
            .map(|m| {
                let score = Self::similarity(&query_embed, &m.embedding);
                (score, m)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(top_k).map(|(_, m)| m).collect()
    }

    // === Q-Learning Operations ===

    /// Get Q-value for state-action pair
    fn get_q(&self, state: &str, action: &str) -> f32 {
        let key = format!("{}|{}", state, action);
        self.data.patterns.get(&key).map(|p| p.q_value).unwrap_or(0.0)
    }

    /// Update Q-value
    fn update_q(&mut self, state: &str, action: &str, reward: f32) {
        let key = format!("{}|{}", state, action);

        let pattern = self.data.patterns.entry(key.clone()).or_insert(QPattern {
            state: state.to_string(),
            action: action.to_string(),
            q_value: 0.0,
            visits: 0,
            last_update: 0,
        });

        // Q-learning update
        pattern.q_value = pattern.q_value + self.alpha * (reward - pattern.q_value);
        pattern.visits += 1;
        pattern.last_update = Self::now();

        self.data.stats.total_patterns = self.data.patterns.len() as u32;
    }

    /// Learn from trajectory
    pub fn learn(&mut self, state: &str, action: &str, outcome: &str, reward: f32) -> String {
        let id = format!("traj_{}", Self::now());

        // Update Q-value
        self.update_q(state, action, reward);

        // Store trajectory
        self.data.trajectories.push(Trajectory {
            id: id.clone(),
            state: state.to_string(),
            action: action.to_string(),
            outcome: outcome.to_string(),
            reward,
            timestamp: Self::now(),
        });

        // Limit trajectories
        if self.data.trajectories.len() > 1000 {
            self.data.trajectories.drain(0..200);
        }

        self.data.stats.total_trajectories = self.data.trajectories.len() as u32;
        id
    }

    /// Suggest best action for state
    pub fn suggest(&self, state: &str, actions: &[String]) -> (String, f32) {
        let mut best_action = actions.first().cloned().unwrap_or_default();
        let mut best_q = f32::MIN;

        for action in actions {
            let q = self.get_q(state, action);
            if q > best_q {
                best_q = q;
                best_action = action.clone();
            }
        }

        let confidence = if best_q > 0.0 { best_q.min(1.0) } else { 0.0 };
        (best_action, confidence)
    }

    /// Route to best agent
    pub fn route(&self, task: &str, file: Option<&str>, crate_name: Option<&str>, operation: &str) -> (String, f32, String) {
        let file_type = file
            .and_then(|f| Path::new(f).extension())
            .and_then(|e| e.to_str())
            .unwrap_or("unknown");

        let state = format!("{}_{}_in_{}", operation, file_type, crate_name.unwrap_or("project"));

        // Agent candidates based on file type
        let agents: Vec<String> = match file_type {
            "rs" => vec!["rust-developer", "coder", "reviewer", "tester"],
            "ts" | "tsx" | "js" | "jsx" => vec!["typescript-developer", "coder", "frontend-dev"],
            "py" => vec!["python-developer", "coder", "ml-developer"],
            "md" => vec!["docs-writer", "coder"],
            "toml" | "json" | "yaml" => vec!["config-specialist", "coder"],
            _ => vec!["coder", "reviewer"],
        }.into_iter().map(String::from).collect();

        let (agent, confidence) = self.suggest(&state, &agents);

        let reason = if confidence > 0.5 {
            "learned from past success".to_string()
        } else if confidence > 0.0 {
            "based on patterns".to_string()
        } else {
            format!("default for {} files", file_type)
        };

        (agent, confidence, reason)
    }

    // === Error Pattern Learning ===

    /// Record error pattern
    pub fn record_error(&mut self, command: &str, stderr: &str) -> Vec<String> {
        let mut recorded = Vec::new();

        // Parse Rust errors
        for line in stderr.lines() {
            if let Some(code) = Self::extract_error_code(line) {
                let key = code.clone();
                let pattern = self.data.errors.entry(key.clone()).or_insert(ErrorPattern {
                    code: code.clone(),
                    error_type: Self::classify_error(&code),
                    message: line.chars().take(200).collect(),
                    fixes: Vec::new(),
                    occurrences: 0,
                });
                pattern.occurrences += 1;
                recorded.push(code);
            }
        }

        self.data.stats.total_errors = self.data.errors.len() as u32;
        recorded
    }

    /// Extract error code from line
    fn extract_error_code(line: &str) -> Option<String> {
        // Rust: error[E0308]
        if let Some(start) = line.find("error[E") {
            let rest = &line[start + 6..];
            if let Some(end) = rest.find(']') {
                return Some(format!("E{}", &rest[1..end]));
            }
        }
        // TypeScript: TS2322
        if let Some(start) = line.find("TS") {
            let rest = &line[start..];
            let code: String = rest.chars().take_while(|c| c.is_alphanumeric()).collect();
            if code.len() >= 5 {
                return Some(code);
            }
        }
        None
    }

    /// Classify error type
    fn classify_error(code: &str) -> String {
        match code {
            c if c.starts_with("E03") => "type-error",
            c if c.starts_with("E04") => "resolution-error",
            c if c.starts_with("E05") => "lifetime-error",
            c if c.starts_with("TS2") => "typescript-type-error",
            _ => "unknown",
        }.to_string()
    }

    /// Suggest fix for error
    pub fn suggest_fix(&self, error_code: &str) -> Option<&ErrorPattern> {
        self.data.errors.get(error_code)
    }

    // === File Sequence Prediction ===

    /// Record file edit
    pub fn record_file_edit(&mut self, file: &str, previous_file: Option<&str>) {
        if let Some(prev) = previous_file {
            let existing = self.data.file_sequences
                .iter_mut()
                .find(|s| s.from_file == prev && s.to_file == file);

            if let Some(seq) = existing {
                seq.count += 1;
            } else {
                self.data.file_sequences.push(FileSequence {
                    from_file: prev.to_string(),
                    to_file: file.to_string(),
                    count: 1,
                });
            }
        }
    }

    /// Suggest next files
    pub fn suggest_next(&self, file: &str, count: usize) -> Vec<(&str, u32)> {
        let mut suggestions: Vec<_> = self.data.file_sequences
            .iter()
            .filter(|s| s.from_file == file)
            .map(|s| (s.to_file.as_str(), s.count))
            .collect();

        suggestions.sort_by(|a, b| b.1.cmp(&a.1));
        suggestions.into_iter().take(count).collect()
    }

    /// Check if tests should run
    pub fn should_test(&self, file: &str) -> (bool, String) {
        let file_type = Path::new(file)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match file_type {
            "rs" => {
                let crate_match = file.contains("crates/");
                if crate_match {
                    let crate_name = file
                        .split("crates/")
                        .nth(1)
                        .and_then(|s| s.split('/').next())
                        .unwrap_or("all");
                    (true, format!("cargo test -p {}", crate_name))
                } else {
                    (true, "cargo test".to_string())
                }
            }
            "ts" | "tsx" | "js" | "jsx" => (true, "npm test".to_string()),
            "py" => (true, "pytest".to_string()),
            _ => (false, String::new()),
        }
    }

    // === Swarm Operations ===

    /// Register agent
    pub fn swarm_register(&mut self, id: &str, agent_type: &str, capabilities: Vec<String>) {
        self.data.agents.insert(id.to_string(), SwarmAgent {
            id: id.to_string(),
            agent_type: agent_type.to_string(),
            capabilities,
            success_rate: 1.0,
            task_count: 0,
            status: "active".to_string(),
        });
    }

    /// Record coordination
    pub fn swarm_coordinate(&mut self, source: &str, target: &str, weight: f32) {
        let existing = self.data.edges
            .iter_mut()
            .find(|e| e.source == source && e.target == target);

        if let Some(edge) = existing {
            edge.weight = (edge.weight + weight) / 2.0;
            edge.coordination_count += 1;
        } else {
            self.data.edges.push(SwarmEdge {
                source: source.to_string(),
                target: target.to_string(),
                weight,
                coordination_count: 1,
            });
        }
    }

    /// Recommend agent for task
    pub fn swarm_recommend(&self, task_type: &str) -> Option<&SwarmAgent> {
        self.data.agents
            .values()
            .filter(|a| a.status == "active" && a.agent_type == task_type)
            .max_by(|a, b| {
                a.success_rate.partial_cmp(&b.success_rate).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Handle agent failure
    pub fn swarm_heal(&mut self, agent_id: &str) -> Option<String> {
        if let Some(agent) = self.data.agents.get_mut(agent_id) {
            agent.status = "failed".to_string();
            agent.success_rate *= 0.8;
        }

        // Find replacement
        let failed_type = self.data.agents.get(agent_id).map(|a| a.agent_type.clone())?;
        self.data.agents
            .values()
            .filter(|a| a.status == "active" && a.agent_type == failed_type && a.id != agent_id)
            .max_by(|a, b| a.success_rate.partial_cmp(&b.success_rate).unwrap_or(std::cmp::Ordering::Equal))
            .map(|a| a.id.clone())
    }

    /// Get swarm stats
    pub fn swarm_stats(&self) -> (usize, usize, f32) {
        let agent_count = self.data.agents.len();
        let edge_count = self.data.edges.len();
        let avg_success = if agent_count > 0 {
            self.data.agents.values().map(|a| a.success_rate).sum::<f32>() / agent_count as f32
        } else {
            0.0
        };
        (agent_count, edge_count, avg_success)
    }

    /// Get full stats
    pub fn stats(&self) -> &IntelligenceStats {
        &self.data.stats
    }

    /// Get pattern count
    pub fn pattern_count(&self) -> usize {
        self.data.patterns.len()
    }

    /// Get memory count
    pub fn memory_count(&self) -> usize {
        self.data.memories.len()
    }
}

// === Command Implementations ===

/// Get intelligence data path
fn get_intelligence_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".ruvector").join("intelligence.json")
}

/// Initialize hooks
pub fn init_hooks(force: bool, _config: &Config) -> Result<()> {
    let claude_dir = PathBuf::from(".claude");
    let settings_path = claude_dir.join("settings.json");

    if settings_path.exists() && !force {
        println!("{}", "Hooks already initialized. Use --force to overwrite.".yellow());
        return Ok(());
    }

    fs::create_dir_all(&claude_dir)?;

    let hooks_config = serde_json::json!({
        "hooks": {
            "PreToolUse": [{
                "matcher": "Edit|Write|MultiEdit",
                "hooks": [{
                    "type": "command",
                    "command": "ruvector hooks pre-edit \"$TOOL_INPUT_FILE_PATH\""
                }]
            }],
            "PostToolUse": [{
                "matcher": "Edit|Write|MultiEdit",
                "hooks": [{
                    "type": "command",
                    "command": "ruvector hooks post-edit \"$TOOL_INPUT_FILE_PATH\" --success=$TOOL_STATUS"
                }]
            }, {
                "matcher": "Bash",
                "hooks": [{
                    "type": "command",
                    "command": "ruvector hooks post-command \"$TOOL_INPUT_COMMAND\" --success=$TOOL_STATUS"
                }]
            }],
            "SessionStart": [{
                "hooks": [{
                    "type": "command",
                    "command": "ruvector hooks session-start"
                }]
            }]
        }
    });

    fs::write(&settings_path, serde_json::to_string_pretty(&hooks_config)?)?;

    println!("{}", "✅ Hooks initialized!".green().bold());
    println!("   Created: {}", settings_path.display());
    println!("\n{}", "Next steps:".bold());
    println!("   1. Restart Claude Code to activate hooks");
    println!("   2. Run 'ruvector hooks stats' to verify");

    Ok(())
}

/// Install hooks
pub fn install_hooks(settings_dir: &str, _config: &Config) -> Result<()> {
    let settings_path = PathBuf::from(settings_dir).join("settings.json");

    if !settings_path.exists() {
        return init_hooks(false, _config);
    }

    let content = fs::read_to_string(&settings_path)?;
    let mut settings: serde_json::Value = serde_json::from_str(&content)?;

    // Add hooks if not present
    if settings.get("hooks").is_none() {
        settings["hooks"] = serde_json::json!({
            "SessionStart": [{
                "hooks": [{
                    "type": "command",
                    "command": "ruvector hooks session-start"
                }]
            }]
        });
        fs::write(&settings_path, serde_json::to_string_pretty(&settings)?)?;
        println!("{}", "✅ Hooks installed!".green().bold());
    } else {
        println!("{}", "Hooks already installed.".yellow());
    }

    Ok(())
}

/// Show stats
pub fn show_stats(_config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());
    let stats = intel.stats();

    println!("{}", "🧠 RuVector Intelligence Stats".bold().cyan());
    println!();
    println!("  {} Q-learning patterns", stats.total_patterns.to_string().green());
    println!("  {} vector memories", stats.total_memories.to_string().green());
    println!("  {} learning trajectories", stats.total_trajectories.to_string().green());
    println!("  {} error patterns", stats.total_errors.to_string().green());
    println!();

    let (agents, edges, avg_success) = intel.swarm_stats();
    println!("{}", "Swarm Status:".bold());
    println!("  {} agents registered", agents.to_string().cyan());
    println!("  {} coordination edges", edges.to_string().cyan());
    if avg_success.is_nan() || avg_success == 0.0 {
        println!("  {}% average success rate", "N/A".cyan());
    } else {
        println!("  {:.0}% average success rate", (avg_success * 100.0).to_string().cyan());
    }

    Ok(())
}

/// Remember content
pub fn remember_content(memory_type: &str, content: &str, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());
    let id = intel.remember(memory_type, content, HashMap::new());
    intel.save()?;

    println!("{}", serde_json::json!({ "success": true, "id": id }));
    Ok(())
}

/// Recall from memory
pub fn recall_content(query: &str, top_k: usize, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());
    let results = intel.recall(query, top_k);

    let output: Vec<_> = results.iter().map(|m| {
        serde_json::json!({
            "type": m.memory_type,
            "content": m.content.chars().take(200).collect::<String>(),
            "timestamp": m.timestamp
        })
    }).collect();

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "query": query,
        "results": output
    }))?);

    Ok(())
}

/// Learn trajectory
pub fn learn_trajectory(state: &str, action: &str, reward: f32, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());
    let id = intel.learn(state, action, "recorded", reward);
    intel.save()?;

    println!("{}", serde_json::json!({
        "success": true,
        "id": id,
        "state": state,
        "action": action,
        "reward": reward
    }));

    Ok(())
}

/// Suggest action
pub fn suggest_action(state: &str, actions_str: &str, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());
    let actions: Vec<String> = actions_str.split(',').map(|s| s.trim().to_string()).collect();
    let (action, confidence) = intel.suggest(state, &actions);

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "state": state,
        "action": action,
        "confidence": confidence,
        "explored": confidence == 0.0
    }))?);

    Ok(())
}

/// Route to agent
pub fn route_task(task: &str, file: Option<&str>, crate_name: Option<&str>, operation: &str, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());
    let (agent, confidence, reason) = intel.route(task, file, crate_name, operation);

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "task": task,
        "recommended": agent,
        "confidence": confidence,
        "reasoning": reason,
        "file": file,
        "crate": crate_name
    }))?);

    Ok(())
}

/// Pre-edit hook
pub fn pre_edit_hook(file: &str, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());

    let file_type = Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown");

    let crate_name = file
        .split("crates/")
        .nth(1)
        .and_then(|s| s.split('/').next());

    let (agent, confidence, reason) = intel.route(
        &format!("edit {}", file),
        Some(file),
        crate_name,
        "edit"
    );

    let similar = intel.recall(&format!("edit {} {}", file_type, crate_name.unwrap_or("")), 3);

    println!("{}", "🧠 Intelligence Analysis:".bold());
    println!("   📁 {}/{}",
        crate_name.unwrap_or("project").cyan(),
        Path::new(file).file_name().unwrap_or_default().to_string_lossy()
    );
    println!("   🤖 Recommended: {} ({:.0}% confidence)",
        agent.green().bold(),
        confidence * 100.0
    );
    if !reason.is_empty() {
        println!("      → {}", reason.dimmed());
    }
    if !similar.is_empty() {
        println!("   📚 {} similar past edits found", similar.len());
    }

    Ok(())
}

/// Post-edit hook
pub fn post_edit_hook(file: &str, success: bool, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());

    let file_type = Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown");

    let crate_name = file
        .split("crates/")
        .nth(1)
        .and_then(|s| s.split('/').next());

    let state = format!("edit_{}_in_{}", file_type, crate_name.unwrap_or("project"));
    let action = if success { "successful-edit" } else { "failed-edit" };
    let reward = if success { 1.0 } else { -0.5 };

    intel.learn(&state, action, if success { "completed" } else { "failed" }, reward);
    intel.remember(
        "edit",
        &format!("{} edit of {} in {}",
            if success { "successful" } else { "failed" },
            file_type,
            crate_name.unwrap_or("project")
        ),
        HashMap::new()
    );

    intel.save()?;

    let icon = if success { "✅" } else { "❌" };
    println!("📊 Learning recorded: {} {}",
        icon,
        Path::new(file).file_name().unwrap_or_default().to_string_lossy()
    );

    // Suggest tests
    let (should_test, test_cmd) = intel.should_test(file);
    if should_test {
        println!("   🧪 Consider: {}", test_cmd.cyan());
    }

    // Suggest next files
    let next = intel.suggest_next(file, 2);
    if !next.is_empty() {
        let files: Vec<_> = next.iter()
            .map(|(f, _)| Path::new(f).file_name().unwrap_or_default().to_string_lossy().to_string())
            .collect();
        println!("   📁 Often edit next: {}", files.join(", ").dimmed());
    }

    Ok(())
}

/// Pre-command hook
pub fn pre_command_hook(command: &str, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());

    let cmd_type = if command.starts_with("cargo") { "cargo" }
        else if command.starts_with("npm") { "npm" }
        else if command.starts_with("git") { "git" }
        else if command.starts_with("wasm-pack") { "wasm" }
        else { "other" };

    let state = format!("{}_in_general", cmd_type);
    let actions = vec!["command-succeeded".to_string(), "command-failed".to_string()];
    let (suggestion, confidence) = intel.suggest(&state, &actions);

    println!("🧠 Command: {}", cmd_type.cyan());
    if confidence > 0.3 {
        println!("   💡 Likely: {}", suggestion);
    }

    Ok(())
}

/// Post-command hook
pub fn post_command_hook(command: &str, success: bool, stderr: Option<&str>, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());

    let cmd_type = if command.starts_with("cargo") { "cargo" }
        else if command.starts_with("npm") { "npm" }
        else if command.starts_with("git") { "git" }
        else if command.starts_with("wasm-pack") { "wasm" }
        else { "other" };

    let state = format!("{}_in_general", cmd_type);
    let action = if success { "command-succeeded" } else { "command-failed" };
    let reward = if success { 1.0 } else { -0.5 };

    intel.learn(&state, action, &command.chars().take(100).collect::<String>(), reward);

    // Record errors if failed
    if !success {
        if let Some(err) = stderr {
            let errors = intel.record_error(command, err);
            if !errors.is_empty() {
                println!("📊 Command ❌ recorded ({} error patterns learned)", errors.len());
                for code in errors.iter().take(2) {
                    if let Some(pattern) = intel.suggest_fix(code) {
                        if !pattern.fixes.is_empty() {
                            println!("   💡 {}: {}", code, pattern.fixes[0]);
                        }
                    }
                }
                intel.save()?;
                return Ok(());
            }
        }
    }

    intel.save()?;

    let icon = if success { "✅" } else { "❌" };
    println!("📊 Command {} recorded", icon);

    Ok(())
}

/// Session start hook
pub fn session_start_hook(_session_id: Option<&str>, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());
    intel.data.stats.session_count += 1;
    intel.data.stats.last_session = Intelligence::now();
    intel.save()?;

    println!("{}", "🧠 RuVector Intelligence Layer Active".bold().cyan());
    println!();
    println!("⚡ Intelligence guides: agent routing, error fixes, file sequences");

    Ok(())
}

/// Session end hook
pub fn session_end_hook(export_metrics: bool, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());

    if export_metrics {
        let stats = intel.stats();
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "patterns": stats.total_patterns,
            "memories": stats.total_memories,
            "trajectories": stats.total_trajectories,
            "errors": stats.total_errors,
            "sessions": stats.session_count
        }))?);
    }

    println!("{}", "📊 Session ended. Learning data saved.".green());

    Ok(())
}

/// Pre-compact hook
pub fn pre_compact_hook(length: Option<usize>, _config: &Config) -> Result<()> {
    println!("🗜️ Pre-compact: conversation length = {}", length.unwrap_or(0));
    Ok(())
}

/// Record error
pub fn record_error_cmd(command: &str, stderr: &str, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());
    let errors = intel.record_error(command, stderr);
    intel.save()?;

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "recorded": errors.len(),
        "errors": errors
    }))?);

    Ok(())
}

/// Suggest fix
pub fn suggest_fix_cmd(error_code: &str, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());

    if let Some(pattern) = intel.suggest_fix(error_code) {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "code": pattern.code,
            "type": pattern.error_type,
            "occurrences": pattern.occurrences,
            "fixes": pattern.fixes
        }))?);
    } else {
        println!("{}", serde_json::json!({
            "code": error_code,
            "found": false
        }));
    }

    Ok(())
}

/// Suggest next files
pub fn suggest_next_cmd(file: &str, count: usize, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());
    let suggestions = intel.suggest_next(file, count);

    let output: Vec<_> = suggestions.iter().map(|(f, c)| {
        serde_json::json!({
            "file": f,
            "count": c
        })
    }).collect();

    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}

/// Should test
pub fn should_test_cmd(file: &str, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());
    let (suggest, command) = intel.should_test(file);

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "suggest": suggest,
        "command": command
    }))?);

    Ok(())
}

/// Swarm register
pub fn swarm_register_cmd(agent_id: &str, agent_type: &str, capabilities: Option<&str>, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());
    let caps: Vec<String> = capabilities
        .map(|s| s.split(',').map(|c| c.trim().to_string()).collect())
        .unwrap_or_default();

    intel.swarm_register(agent_id, agent_type, caps);
    intel.save()?;

    println!("{}", serde_json::json!({
        "success": true,
        "agent_id": agent_id,
        "type": agent_type
    }));

    Ok(())
}

/// Swarm coordinate
pub fn swarm_coordinate_cmd(source: &str, target: &str, weight: f32, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());
    intel.swarm_coordinate(source, target, weight);
    intel.save()?;

    println!("{}", serde_json::json!({
        "success": true,
        "source": source,
        "target": target,
        "weight": weight
    }));

    Ok(())
}

/// Swarm optimize
pub fn swarm_optimize_cmd(tasks: &str, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());
    let task_list: Vec<&str> = tasks.split(',').map(|s| s.trim()).collect();

    let assignments: Vec<_> = task_list.iter().map(|task| {
        let (agent, edges, _) = intel.swarm_stats();
        serde_json::json!({
            "task": task,
            "available_agents": agent,
            "coordination_edges": edges
        })
    }).collect();

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "tasks": task_list.len(),
        "assignments": assignments
    }))?);

    Ok(())
}

/// Swarm recommend
pub fn swarm_recommend_cmd(task_type: &str, _config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());

    if let Some(agent) = intel.swarm_recommend(task_type) {
        println!("{}", serde_json::to_string_pretty(&serde_json::json!({
            "task_type": task_type,
            "recommended": agent.id,
            "success_rate": agent.success_rate,
            "capabilities": agent.capabilities
        }))?);
    } else {
        println!("{}", serde_json::json!({
            "task_type": task_type,
            "recommended": null,
            "message": "No matching agent found"
        }));
    }

    Ok(())
}

/// Swarm heal
pub fn swarm_heal_cmd(agent_id: &str, _config: &Config) -> Result<()> {
    let mut intel = Intelligence::new(get_intelligence_path());
    let replacement = intel.swarm_heal(agent_id);
    intel.save()?;

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "failed_agent": agent_id,
        "replacement": replacement,
        "healed": replacement.is_some()
    }))?);

    Ok(())
}

/// Swarm stats
pub fn swarm_stats_cmd(_config: &Config) -> Result<()> {
    let intel = Intelligence::new(get_intelligence_path());
    let (agents, edges, avg_success) = intel.swarm_stats();

    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "agents": agents,
        "edges": edges,
        "average_success_rate": avg_success,
        "topology": "mesh"
    }))?);

    Ok(())
}
