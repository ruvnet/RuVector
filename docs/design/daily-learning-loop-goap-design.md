# Daily Learning and Optimization Loop - GOAP Design Document

## Executive Summary

This document defines a comprehensive Goal-Oriented Action Planning (GOAP) system for RuVector that autonomously discovers, logs, assesses, and shares knowledge with the pi.ruv.io collective brain. The system leverages SPARC methodology phases, rvagent for coordination, Gemini 2.5 Flash for reasoning, and the existing SONA/ReasoningBank infrastructure.

---

## 1. GOAP State Space Definition

### 1.1 World State Model

The GOAP planner operates on a world state represented as a typed struct:

```rust
/// GOAP World State for Daily Learning Loop
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningWorldState {
    // === Knowledge Discovery State ===
    /// Number of codebase patterns discovered this cycle
    pub patterns_discovered: u32,
    /// Number of optimization opportunities identified
    pub optimizations_found: u32,
    /// Last codebase scan timestamp (Unix seconds)
    pub last_scan_timestamp: u64,
    /// Files analyzed in current cycle
    pub files_analyzed: u32,
    /// SONA trajectory buffer fill level (0.0-1.0)
    pub trajectory_buffer_fill: f32,

    // === Quality Assessment State ===
    /// Average quality score of discoveries (0.0-1.0)
    pub avg_discovery_quality: f32,
    /// Novelty score (how different from existing patterns)
    pub novelty_score: f32,
    /// Confidence in discoveries (0.0-1.0)
    pub confidence: f32,

    // === Logging State ===
    /// Discoveries pending logging
    pub pending_logs: u32,
    /// Logged discoveries this cycle
    pub logged_discoveries: u32,
    /// Method attribution complete
    pub method_attribution_complete: bool,

    // === Cloud Submission State ===
    /// Discoveries pending submission to pi.ruv.io
    pub pending_submissions: u32,
    /// Successfully submitted this cycle
    pub submitted_to_cloud: u32,
    /// Cloud connection healthy
    pub cloud_connection_healthy: bool,
    /// Last successful submission timestamp
    pub last_submission_timestamp: u64,

    // === Consolidation State ===
    /// Local patterns updated
    pub local_patterns_updated: bool,
    /// EWC++ consolidation needed
    pub ewc_consolidation_pending: bool,
    /// ReasoningBank pattern count
    pub reasoning_bank_patterns: u32,
    /// Memory pressure (0.0-1.0)
    pub memory_pressure: f32,

    // === Cycle Control ===
    /// Current cycle phase
    pub cycle_phase: CyclePhase,
    /// Cycle start timestamp
    pub cycle_start: u64,
    /// Daily quota remaining
    pub daily_quota_remaining: u32,
    /// Error count this cycle
    pub error_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CyclePhase {
    Idle,
    Discovery,
    Assessment,
    Logging,
    Submission,
    Consolidation,
    Complete,
}
```

### 1.2 Goal States

```rust
/// Goal definitions for GOAP planner
pub struct LearningGoals {
    /// Primary daily goal: complete learning cycle
    pub daily_cycle_complete: GoalSpec,
    /// Discovery goal: find valuable patterns
    pub discovery_target: GoalSpec,
    /// Quality goal: maintain high standards
    pub quality_threshold: GoalSpec,
    /// Sharing goal: contribute to collective brain
    pub sharing_target: GoalSpec,
    /// Consolidation goal: prevent forgetting
    pub consolidation_complete: GoalSpec,
}

impl Default for LearningGoals {
    fn default() -> Self {
        Self {
            daily_cycle_complete: GoalSpec {
                name: "daily_cycle_complete".into(),
                conditions: vec![
                    Condition::Eq("cycle_phase", Value::Enum("Complete")),
                    Condition::Gte("logged_discoveries", Value::U32(5)),
                    Condition::Eq("ewc_consolidation_pending", Value::Bool(false)),
                ],
                priority: 1.0,
            },
            discovery_target: GoalSpec {
                name: "discovery_target".into(),
                conditions: vec![
                    Condition::Gte("patterns_discovered", Value::U32(10)),
                    Condition::Gte("optimizations_found", Value::U32(3)),
                ],
                priority: 0.9,
            },
            quality_threshold: GoalSpec {
                name: "quality_threshold".into(),
                conditions: vec![
                    Condition::Gte("avg_discovery_quality", Value::F32(0.7)),
                    Condition::Gte("novelty_score", Value::F32(0.5)),
                ],
                priority: 0.85,
            },
            sharing_target: GoalSpec {
                name: "sharing_target".into(),
                conditions: vec![
                    Condition::Gte("submitted_to_cloud", Value::U32(5)),
                    Condition::Eq("pending_submissions", Value::U32(0)),
                ],
                priority: 0.8,
            },
            consolidation_complete: GoalSpec {
                name: "consolidation_complete".into(),
                conditions: vec![
                    Condition::Eq("local_patterns_updated", Value::Bool(true)),
                    Condition::Eq("ewc_consolidation_pending", Value::Bool(false)),
                    Condition::Lte("memory_pressure", Value::F32(0.7)),
                ],
                priority: 0.75,
            },
        }
    }
}
```

---

## 2. GOAP Actions

### 2.1 Knowledge Discovery Actions

```rust
/// Action: Scan codebase for patterns
pub struct ScanCodebaseAction {
    pub name: &'static str,
    pub cost: f32,
    pub preconditions: Vec<Condition>,
    pub effects: Vec<Effect>,
}

impl Default for ScanCodebaseAction {
    fn default() -> Self {
        Self {
            name: "scan_codebase",
            cost: 2.0,
            preconditions: vec![
                // At least 1 hour since last scan
                Condition::TimeSince("last_scan_timestamp", Duration::hours(1)),
                // Not in middle of another phase
                Condition::OneOf("cycle_phase", vec!["Idle", "Discovery"]),
                // Have quota remaining
                Condition::Gte("daily_quota_remaining", Value::U32(1)),
            ],
            effects: vec![
                Effect::Set("cycle_phase", Value::Enum("Discovery")),
                Effect::Increment("files_analyzed", 100),
                Effect::IncrementRange("patterns_discovered", 0, 5),
                Effect::IncrementRange("optimizations_found", 0, 2),
                Effect::SetNow("last_scan_timestamp"),
            ],
        }
    }
}

/// Action: Analyze patterns with Gemini
pub struct AnalyzePatternsAction {
    pub name: &'static str,
    pub cost: f32,
    pub tools_used: Vec<&'static str>,
}

impl Default for AnalyzePatternsAction {
    fn default() -> Self {
        Self {
            name: "analyze_patterns_gemini",
            cost: 3.0,
            tools_used: vec![
                "rvagent-backends::gemini::GeminiClient",
                "rvagent-middleware::sona::SonaMiddleware",
                "ruvector-sona::ReasoningBank::find_similar",
            ],
            preconditions: vec![
                Condition::Gte("patterns_discovered", Value::U32(1)),
                Condition::Eq("cycle_phase", Value::Enum("Discovery")),
            ],
            effects: vec![
                Effect::Set("cycle_phase", Value::Enum("Assessment")),
                Effect::ComputeQuality("avg_discovery_quality"),
                Effect::ComputeNovelty("novelty_score"),
                Effect::Set("confidence", Value::F32(0.0)), // computed dynamically
            ],
        }
    }
}

/// Action: Deep codebase analysis
pub struct DeepAnalysisAction {
    pub name: &'static str,
    pub cost: f32,
}

impl Default for DeepAnalysisAction {
    fn default() -> Self {
        Self {
            name: "deep_analysis",
            cost: 5.0,
            preconditions: vec![
                Condition::Lt("patterns_discovered", Value::U32(10)),
                Condition::Eq("cycle_phase", Value::Enum("Discovery")),
                Condition::Gte("daily_quota_remaining", Value::U32(3)),
            ],
            effects: vec![
                Effect::IncrementRange("patterns_discovered", 5, 15),
                Effect::IncrementRange("optimizations_found", 2, 5),
                Effect::Decrement("daily_quota_remaining", 3),
            ],
        }
    }
}
```

### 2.2 Method Logging Actions

```rust
/// Action: Log discovery with method attribution
pub struct LogDiscoveryAction {
    pub name: &'static str,
    pub cost: f32,
}

impl Default for LogDiscoveryAction {
    fn default() -> Self {
        Self {
            name: "log_discovery",
            cost: 1.0,
            preconditions: vec![
                Condition::Gte("patterns_discovered", Value::U32(1)),
                Condition::OneOf("cycle_phase", vec!["Assessment", "Logging"]),
            ],
            effects: vec![
                Effect::Set("cycle_phase", Value::Enum("Logging")),
                Effect::Transfer("patterns_discovered", "pending_logs", 1),
                Effect::Increment("logged_discoveries", 1),
                Effect::Set("method_attribution_complete", Value::Bool(true)),
            ],
        }
    }
}

/// Logged discovery record structure
pub struct DiscoveryLog {
    /// Unique discovery ID
    pub id: Uuid,
    /// Discovery timestamp
    pub timestamp: u64,
    /// What was discovered
    pub discovery_type: DiscoveryType,
    /// Human-readable description
    pub description: String,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
    /// Novelty score (0.0-1.0)
    pub novelty_score: f32,
    /// Tools/methods that led to discovery
    pub methods_used: Vec<MethodAttribution>,
    /// Related file paths
    pub related_files: Vec<String>,
    /// Code snippets if applicable
    pub code_snippets: Vec<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Submission status
    pub submission_status: SubmissionStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DiscoveryType {
    Pattern { name: String, category: String },
    Optimization { target: String, improvement: f32 },
    BugPattern { severity: String },
    Architecture { component: String },
    Security { vulnerability_class: String },
    Performance { metric: String, improvement: f32 },
    Convention { scope: String },
    Solution { problem_domain: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MethodAttribution {
    /// Tool or technique used
    pub tool: String,
    /// Crate/module path
    pub module_path: String,
    /// Specific function/method
    pub function: String,
    /// Parameters used
    pub parameters: HashMap<String, Value>,
    /// Contribution to discovery (0.0-1.0)
    pub contribution_weight: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SubmissionStatus {
    Pending,
    Submitted { submission_id: String },
    Accepted { memory_id: Uuid },
    Rejected { reason: String },
}
```

### 2.3 Quality Assessment Actions

```rust
/// Action: Assess discovery quality
pub struct AssessQualityAction {
    pub name: &'static str,
    pub cost: f32,
}

impl Default for AssessQualityAction {
    fn default() -> Self {
        Self {
            name: "assess_quality",
            cost: 1.5,
            preconditions: vec![
                Condition::Gte("logged_discoveries", Value::U32(1)),
                Condition::Eq("cycle_phase", Value::Enum("Logging")),
            ],
            effects: vec![
                Effect::ComputeFromLogs("avg_discovery_quality"),
                Effect::FilterByQuality("pending_submissions", 0.7),
            ],
        }
    }
}

/// Quality assessment criteria
pub struct QualityAssessment {
    /// Novelty: how different from existing patterns
    pub novelty: f32,
    /// Usefulness: potential impact on development
    pub usefulness: f32,
    /// Clarity: how well-described
    pub clarity: f32,
    /// Correctness: verified accuracy
    pub correctness: f32,
    /// Generalizability: applies beyond specific case
    pub generalizability: f32,

    /// Composite score (weighted average)
    pub composite_score: f32,
}

impl QualityAssessment {
    pub fn compute(discovery: &DiscoveryLog, bank: &ReasoningBank) -> Self {
        // Novelty from similarity search
        let similar = bank.find_similar(&discovery.embedding, 5);
        let max_similarity = similar.first()
            .map(|p| p.similarity(&discovery.embedding))
            .unwrap_or(0.0);
        let novelty = 1.0 - max_similarity;

        // Other scores from Gemini analysis
        // ...

        Self {
            novelty,
            usefulness: 0.0, // computed by Gemini
            clarity: 0.0,
            correctness: 0.0,
            generalizability: 0.0,
            composite_score: novelty * 0.3 + /* ... */,
        }
    }
}
```

### 2.4 pi.ruv.io Submission Actions

```rust
/// Action: Submit discoveries to pi.ruv.io cloud brain
pub struct SubmitToCloudAction {
    pub name: &'static str,
    pub cost: f32,
}

impl Default for SubmitToCloudAction {
    fn default() -> Self {
        Self {
            name: "submit_to_cloud",
            cost: 2.0,
            preconditions: vec![
                Condition::Gte("pending_submissions", Value::U32(1)),
                Condition::Eq("cloud_connection_healthy", Value::Bool(true)),
                Condition::OneOf("cycle_phase", vec!["Logging", "Submission"]),
                Condition::Gte("avg_discovery_quality", Value::F32(0.7)),
            ],
            effects: vec![
                Effect::Set("cycle_phase", Value::Enum("Submission")),
                Effect::Transfer("pending_submissions", "submitted_to_cloud", 1),
                Effect::SetNow("last_submission_timestamp"),
            ],
        }
    }
}

/// Cloud submission request (maps to pi.ruv.io /v1/memories POST)
pub struct CloudSubmissionRequest {
    /// Maps to ShareRequest
    pub category: BrainCategory,
    pub title: String,
    pub content: String,
    pub tags: Vec<String>,
    pub code_snippet: Option<String>,

    /// Method attribution for provenance
    pub methods_used: Vec<MethodAttribution>,

    /// Local discovery ID for tracking
    pub local_discovery_id: Uuid,
}

impl From<DiscoveryLog> for CloudSubmissionRequest {
    fn from(log: DiscoveryLog) -> Self {
        Self {
            category: match &log.discovery_type {
                DiscoveryType::Pattern { category, .. } =>
                    BrainCategory::from_str(category).unwrap_or(BrainCategory::Pattern),
                DiscoveryType::Optimization { .. } => BrainCategory::Performance,
                DiscoveryType::Security { .. } => BrainCategory::Security,
                DiscoveryType::Architecture { .. } => BrainCategory::Architecture,
                DiscoveryType::Solution { .. } => BrainCategory::Solution,
                _ => BrainCategory::Pattern,
            },
            title: log.description.chars().take(200).collect(),
            content: format_discovery_content(&log),
            tags: log.tags,
            code_snippet: log.code_snippets.first().cloned(),
            methods_used: log.methods_used,
            local_discovery_id: log.id,
        }
    }
}

/// MCP brain tools integration
pub struct BrainMcpClient {
    base_url: String,
    http: reqwest::Client,
}

impl BrainMcpClient {
    pub async fn brain_share(&self, request: CloudSubmissionRequest) -> Result<ShareResponse> {
        // POST to pi.ruv.io/v1/memories
        let share_req = ShareRequest {
            category: request.category,
            title: request.title,
            content: request.content,
            tags: Some(request.tags),
            code_snippet: request.code_snippet,
        };

        let response = self.http
            .post(format!("{}/v1/memories", self.base_url))
            .json(&share_req)
            .send()
            .await?;

        Ok(response.json().await?)
    }

    pub async fn brain_status(&self) -> Result<StatusResponse> {
        // GET pi.ruv.io/v1/status
        let response = self.http
            .get(format!("{}/v1/status", self.base_url))
            .send()
            .await?;

        Ok(response.json().await?)
    }

    pub async fn check_connection(&self) -> bool {
        self.brain_status().await.is_ok()
    }
}
```

### 2.5 Learning Consolidation Actions

```rust
/// Action: Update local patterns
pub struct UpdateLocalPatternsAction {
    pub name: &'static str,
    pub cost: f32,
}

impl Default for UpdateLocalPatternsAction {
    fn default() -> Self {
        Self {
            name: "update_local_patterns",
            cost: 1.5,
            preconditions: vec![
                Condition::Gte("logged_discoveries", Value::U32(1)),
                Condition::OneOf("cycle_phase", vec!["Submission", "Consolidation"]),
            ],
            effects: vec![
                Effect::Set("cycle_phase", Value::Enum("Consolidation")),
                Effect::Set("local_patterns_updated", Value::Bool(true)),
                Effect::Increment("reasoning_bank_patterns",
                    DynamicValue::Field("logged_discoveries")),
            ],
        }
    }
}

/// Action: Run EWC++ consolidation to prevent forgetting
pub struct EwcConsolidationAction {
    pub name: &'static str,
    pub cost: f32,
}

impl Default for EwcConsolidationAction {
    fn default() -> Self {
        Self {
            name: "ewc_consolidation",
            cost: 3.0,
            preconditions: vec![
                Condition::Eq("ewc_consolidation_pending", Value::Bool(true)),
                Condition::Eq("cycle_phase", Value::Enum("Consolidation")),
            ],
            effects: vec![
                Effect::Set("ewc_consolidation_pending", Value::Bool(false)),
                Effect::Reduce("memory_pressure", 0.2),
            ],
        }
    }
}

/// Action: Prune low-quality patterns
pub struct PrunePatternsAction {
    pub name: &'static str,
    pub cost: f32,
}

impl Default for PrunePatternsAction {
    fn default() -> Self {
        Self {
            name: "prune_patterns",
            cost: 1.0,
            preconditions: vec![
                Condition::Gte("memory_pressure", Value::F32(0.8)),
                Condition::Eq("cycle_phase", Value::Enum("Consolidation")),
            ],
            effects: vec![
                Effect::Reduce("memory_pressure", 0.3),
                Effect::Reduce("reasoning_bank_patterns",
                    DynamicValue::Percentage(0.1)),
            ],
        }
    }
}

/// Action: Complete daily cycle
pub struct CompleteCycleAction {
    pub name: &'static str,
    pub cost: f32,
}

impl Default for CompleteCycleAction {
    fn default() -> Self {
        Self {
            name: "complete_cycle",
            cost: 0.5,
            preconditions: vec![
                Condition::Eq("local_patterns_updated", Value::Bool(true)),
                Condition::Eq("ewc_consolidation_pending", Value::Bool(false)),
                Condition::Eq("pending_submissions", Value::U32(0)),
                Condition::Eq("cycle_phase", Value::Enum("Consolidation")),
            ],
            effects: vec![
                Effect::Set("cycle_phase", Value::Enum("Complete")),
                Effect::Reset("patterns_discovered"),
                Effect::Reset("optimizations_found"),
                Effect::Reset("logged_discoveries"),
                Effect::Reset("submitted_to_cloud"),
                Effect::Reset("error_count"),
            ],
        }
    }
}
```

---

## 3. Daily Schedule and Triggers

### 3.1 Schedule Configuration

```rust
/// Daily learning loop schedule
pub struct LearningSchedule {
    /// Primary daily run time (e.g., 02:00 UTC)
    pub primary_run_time: NaiveTime,

    /// Secondary opportunistic triggers
    pub opportunistic_triggers: Vec<OpportunisticTrigger>,

    /// Minimum interval between full cycles
    pub min_cycle_interval: Duration,

    /// Maximum cycles per day
    pub max_daily_cycles: u32,

    /// Timezone for schedule
    pub timezone: Tz,
}

impl Default for LearningSchedule {
    fn default() -> Self {
        Self {
            // Run at 2 AM UTC (low traffic period)
            primary_run_time: NaiveTime::from_hms_opt(2, 0, 0).unwrap(),
            opportunistic_triggers: vec![
                OpportunisticTrigger::GitCommit { min_commits: 10 },
                OpportunisticTrigger::IdlePeriod { min_idle_mins: 30 },
                OpportunisticTrigger::TrajectoryBufferFull { threshold: 0.8 },
                OpportunisticTrigger::NewPatternsAvailable { count: 20 },
            ],
            min_cycle_interval: Duration::hours(6),
            max_daily_cycles: 4,
            timezone: Tz::UTC,
        }
    }
}

#[derive(Clone, Debug)]
pub enum OpportunisticTrigger {
    /// Trigger after N git commits
    GitCommit { min_commits: u32 },
    /// Trigger after idle period
    IdlePeriod { min_idle_mins: u32 },
    /// Trigger when trajectory buffer is filling up
    TrajectoryBufferFull { threshold: f32 },
    /// Trigger when SONA has extracted new patterns
    NewPatternsAvailable { count: u32 },
    /// Trigger on explicit user request
    UserRequest,
    /// Trigger from CI/CD pipeline completion
    CiPipelineComplete { success: bool },
}
```

### 3.2 Trigger Detection System

```rust
/// Trigger detection daemon
pub struct TriggerDetector {
    schedule: LearningSchedule,
    last_cycle_time: Option<Instant>,
    cycles_today: u32,
    commit_counter: AtomicU32,
    last_activity: AtomicU64,
}

impl TriggerDetector {
    /// Check if a learning cycle should be triggered
    pub fn should_trigger(&self, state: &LearningWorldState) -> Option<TriggerReason> {
        // Check daily quota
        if self.cycles_today >= self.schedule.max_daily_cycles {
            return None;
        }

        // Check minimum interval
        if let Some(last) = self.last_cycle_time {
            if last.elapsed() < self.schedule.min_cycle_interval {
                return None;
            }
        }

        // Check scheduled time
        let now = Utc::now().with_timezone(&self.schedule.timezone);
        let scheduled = now.date_naive().and_time(self.schedule.primary_run_time);
        if (now.naive_local() - scheduled).abs() < Duration::minutes(5) {
            return Some(TriggerReason::Scheduled);
        }

        // Check opportunistic triggers
        for trigger in &self.schedule.opportunistic_triggers {
            if self.check_opportunistic_trigger(trigger, state) {
                return Some(TriggerReason::Opportunistic(trigger.clone()));
            }
        }

        None
    }

    fn check_opportunistic_trigger(
        &self,
        trigger: &OpportunisticTrigger,
        state: &LearningWorldState,
    ) -> bool {
        match trigger {
            OpportunisticTrigger::GitCommit { min_commits } => {
                self.commit_counter.load(Ordering::Relaxed) >= *min_commits
            }
            OpportunisticTrigger::IdlePeriod { min_idle_mins } => {
                let last = self.last_activity.load(Ordering::Relaxed);
                let idle_secs = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs() - last;
                idle_secs >= (*min_idle_mins as u64) * 60
            }
            OpportunisticTrigger::TrajectoryBufferFull { threshold } => {
                state.trajectory_buffer_fill >= *threshold
            }
            OpportunisticTrigger::NewPatternsAvailable { count } => {
                state.reasoning_bank_patterns >= *count
            }
            OpportunisticTrigger::UserRequest => false, // handled separately
            OpportunisticTrigger::CiPipelineComplete { .. } => false, // webhook
        }
    }
}

#[derive(Clone, Debug)]
pub enum TriggerReason {
    Scheduled,
    Opportunistic(OpportunisticTrigger),
    Manual,
}
```

### 3.3 Background Daemon Integration

```rust
/// Integration with mcp-brain-server background training loop
pub async fn integrate_with_brain_training(
    state: &AppState,
    learning_state: &mut LearningWorldState,
) -> TrainingCycleResult {
    // This runs within the existing 5-minute background loop
    // from mcp-brain-server/src/main.rs lines 22-49

    // 1. Check if learning cycle should run
    let detector = TriggerDetector::default();
    if let Some(reason) = detector.should_trigger(learning_state) {
        tracing::info!("Learning cycle triggered: {:?}", reason);

        // 2. Run GOAP planner
        let planner = GoapPlanner::new();
        let goal = LearningGoals::default().daily_cycle_complete;
        let plan = planner.plan(learning_state, &goal);

        // 3. Execute plan actions
        for action in plan.actions {
            execute_learning_action(action, state, learning_state).await?;
        }
    }

    // 4. Run existing training (SONA + domain evolution)
    crate::routes::run_training_cycle(state)
}
```

---

## 4. Integration Points

### 4.1 Existing System Integration Map

```
+------------------------------------------+
|           Daily Learning Loop            |
+------------------------------------------+
            |           |           |
            v           v           v
+----------+   +----------+   +----------+
| rvagent  |   |   SONA   |   | pi.ruv.io|
| (coord)  |   | Middleware|   | (cloud)  |
+----------+   +----------+   +----------+
     |              |              |
     v              v              v
+-------------------------------------------+
|                                           |
|  rvagent-backends::GeminiClient           |
|  ├── Gemini 2.5 Flash API                 |
|  └── GOOGLE_API_KEY from Google Secrets   |
|                                           |
|  rvagent-middleware::SonaMiddleware       |
|  ├── TrajectoryBuffer (Loop A)            |
|  ├── ReasoningBank (Loop B)               |
|  └── EWC++ (Loop C)                       |
|                                           |
|  mcp-brain-server::routes                 |
|  ├── /v1/memories (brain_share)           |
|  ├── /v1/memories/search (brain_search)   |
|  ├── /v1/status (brain_status)            |
|  └── /v1/train (training endpoint)        |
|                                           |
|  ruvector-sona::ReasoningBank             |
|  ├── K-means++ clustering                 |
|  ├── Pattern extraction                   |
|  └── Similarity search (HNSW)             |
|                                           |
+-------------------------------------------+
```

### 4.2 rvagent Integration

```rust
// File: crates/rvAgent/rvagent-middleware/src/learning_loop.rs

use crate::sona::{SonaMiddleware, SonaMiddlewareConfig};
use rvagent_backends::gemini::GeminiClient;
use rvagent_core::models::resolve_model;

/// Learning loop coordinator using rvagent infrastructure
pub struct LearningLoopCoordinator {
    /// SONA middleware for pattern learning
    sona: Arc<SonaMiddleware>,

    /// Gemini client for reasoning
    gemini: GeminiClient,

    /// GOAP planner
    planner: GoapPlanner,

    /// Current world state
    state: RwLock<LearningWorldState>,

    /// Discovery log storage
    discoveries: RwLock<Vec<DiscoveryLog>>,

    /// Cloud client
    brain_client: BrainMcpClient,
}

impl LearningLoopCoordinator {
    pub async fn new() -> Result<Self> {
        let sona_config = SonaMiddlewareConfig {
            enabled: true,
            hidden_dim: 256,
            embedding_dim: 256,
            micro_lora_rank: 2,
            base_lora_rank: 8,
            trajectory_buffer_capacity: 1024,
            quality_threshold: 0.3,
            background_interval_secs: 3600,
            pattern_clusters: 100,
            ewc_lambda: 2000.0,
            ewc_max_tasks: 10,
            record_trajectories: true,
            enable_pattern_search: true,
            pattern_search_k: 5,
        };

        let gemini_config = resolve_model("google:gemini-2.5-flash-preview-05-20");
        let gemini = GeminiClient::new(gemini_config)?;

        let brain_client = BrainMcpClient::new("https://pi.ruv.io");

        Ok(Self {
            sona: Arc::new(SonaMiddleware::new(sona_config)),
            gemini,
            planner: GoapPlanner::new(),
            state: RwLock::new(LearningWorldState::default()),
            discoveries: RwLock::new(Vec::new()),
            brain_client,
        })
    }

    /// Execute a single learning cycle
    pub async fn run_cycle(&self) -> Result<CycleReport> {
        let mut state = self.state.write();
        state.cycle_phase = CyclePhase::Discovery;
        state.cycle_start = now_secs();

        // Plan and execute
        let goal = LearningGoals::default().daily_cycle_complete;
        let plan = self.planner.plan(&state, &goal);

        let mut report = CycleReport::new();

        for action in plan.actions {
            match self.execute_action(&mut state, action).await {
                Ok(result) => report.add_success(result),
                Err(e) => {
                    state.error_count += 1;
                    report.add_error(e);
                    if state.error_count > 3 {
                        break; // Abort cycle on repeated errors
                    }
                }
            }
        }

        report.finalize(&state);
        Ok(report)
    }
}
```

### 4.3 Gemini Integration via Google Secrets

```rust
// File: crates/rvAgent/rvagent-backends/src/gemini_secrets.rs

use google_cloud_secretmanager::client::SecretManagerService;

/// Load Gemini API key from Google Cloud Secret Manager
pub async fn load_gemini_api_key() -> Result<String> {
    // In Cloud Run, use default credentials
    let client = SecretManagerService::default().await?;

    let secret_name = "projects/ruv-dev/secrets/gemini-api-key/versions/latest";
    let response = client.access_secret_version(secret_name).await?;

    let payload = response.payload
        .ok_or_else(|| anyhow!("Secret payload is empty"))?;

    String::from_utf8(payload.data)
        .map_err(|e| anyhow!("Invalid UTF-8 in secret: {}", e))
}

/// Initialize Gemini client with secret from Google Cloud
pub async fn create_gemini_client_from_secrets() -> Result<GeminiClient> {
    let api_key = load_gemini_api_key().await?;
    std::env::set_var("GOOGLE_API_KEY", &api_key);

    let config = resolve_model("google:gemini-2.5-flash-preview-05-20");
    GeminiClient::new(config)
}
```

### 4.4 MCP Server Tool Integration

```rust
// File: npm/packages/ruvector/bin/mcp-server.js (tools integration)

// Add learning loop tools to existing MCP server
const learningLoopTools = [
    {
        name: 'learning_trigger',
        description: 'Manually trigger a learning cycle',
        inputSchema: {
            type: 'object',
            properties: {
                reason: { type: 'string', description: 'Reason for manual trigger' }
            }
        }
    },
    {
        name: 'learning_status',
        description: 'Get current learning loop status',
        inputSchema: {
            type: 'object',
            properties: {}
        }
    },
    {
        name: 'learning_discoveries',
        description: 'List recent discoveries',
        inputSchema: {
            type: 'object',
            properties: {
                limit: { type: 'number', default: 10 },
                since: { type: 'string', description: 'ISO timestamp' }
            }
        }
    },
    {
        name: 'learning_submit',
        description: 'Submit a specific discovery to pi.ruv.io',
        inputSchema: {
            type: 'object',
            properties: {
                discovery_id: { type: 'string', description: 'UUID of discovery' }
            },
            required: ['discovery_id']
        }
    }
];
```

---

## 5. Implementation Milestones

### Milestone 1: GOAP Planner Core (Week 1)

**Deliverables:**
- `goap_planner.rs` with A* search implementation
- `world_state.rs` with state representation
- `actions.rs` with action definitions
- Unit tests for planner logic

**Success Criteria:**
- [ ] Planner generates valid action sequences for all goal states
- [ ] Action preconditions correctly evaluated
- [ ] Effect application updates state correctly
- [ ] A* search finds optimal (lowest cost) plans
- [ ] Test coverage > 80%

**Technical Implementation:**
```rust
// crates/rvAgent/rvagent-core/src/goap/mod.rs
pub mod planner;
pub mod world_state;
pub mod actions;
pub mod goals;

// A* search for GOAP planning
pub fn plan(initial: &WorldState, goal: &GoalSpec) -> Plan {
    let mut open = BinaryHeap::new();
    let mut came_from: HashMap<WorldState, (WorldState, Action)> = HashMap::new();
    let mut g_score: HashMap<WorldState, f32> = HashMap::new();

    g_score.insert(initial.clone(), 0.0);
    open.push(PlanNode {
        state: initial.clone(),
        f_score: heuristic(initial, goal),
    });

    while let Some(current) = open.pop() {
        if goal.satisfied_by(&current.state) {
            return reconstruct_plan(came_from, &current.state);
        }

        for action in get_applicable_actions(&current.state) {
            let next_state = apply_action(&current.state, &action);
            let tentative_g = g_score[&current.state] + action.cost;

            if tentative_g < *g_score.get(&next_state).unwrap_or(&f32::MAX) {
                came_from.insert(next_state.clone(), (current.state.clone(), action.clone()));
                g_score.insert(next_state.clone(), tentative_g);
                open.push(PlanNode {
                    state: next_state.clone(),
                    f_score: tentative_g + heuristic(&next_state, goal),
                });
            }
        }
    }

    Plan::empty() // No plan found
}
```

### Milestone 2: Discovery Engine (Week 2)

**Deliverables:**
- Codebase scanner integration
- Pattern recognition with SONA
- Gemini-based analysis pipeline
- Discovery logging system

**Success Criteria:**
- [ ] Scanner analyzes 100+ files per cycle
- [ ] Pattern recognition identifies 5+ patterns per scan
- [ ] Gemini analysis provides quality scores
- [ ] Discoveries logged with full method attribution
- [ ] Integration tests pass

**Technical Implementation:**
```rust
// crates/rvAgent/rvagent-middleware/src/discovery.rs

pub struct DiscoveryEngine {
    sona: Arc<SonaMiddleware>,
    gemini: GeminiClient,
    file_scanner: FileScanner,
}

impl DiscoveryEngine {
    pub async fn scan_codebase(&self) -> Vec<DiscoveryCandidate> {
        let mut candidates = Vec::new();

        // 1. Scan files for patterns
        let files = self.file_scanner.scan_directory("./").await?;

        for file in files {
            // 2. Generate embeddings
            let embedding = self.sona.generate_embedding(&file.content);

            // 3. Check similarity to known patterns
            let similar = self.sona.find_patterns(&file.content);

            // 4. If novel, analyze with Gemini
            if similar.is_empty() || similar[0].1 < 0.8 {
                let analysis = self.gemini_analyze(&file).await?;

                if analysis.is_discovery {
                    candidates.push(DiscoveryCandidate {
                        file_path: file.path,
                        embedding,
                        analysis,
                        methods_used: vec![
                            MethodAttribution::sona_pattern_search(),
                            MethodAttribution::gemini_analysis(),
                        ],
                    });
                }
            }
        }

        candidates
    }
}
```

### Milestone 3: Quality Assessment (Week 3)

**Deliverables:**
- Quality scoring system
- Novelty detection via HNSW
- Confidence estimation
- Quality-based filtering

**Success Criteria:**
- [ ] Quality scores in [0.0, 1.0] range
- [ ] Novelty detection uses HNSW similarity search
- [ ] Confidence correlates with actual discovery value
- [ ] Filtering removes low-quality discoveries (< 0.7)
- [ ] Assessment latency < 100ms per discovery

**Technical Implementation:**
```rust
// crates/rvAgent/rvagent-middleware/src/quality.rs

pub struct QualityAssessor {
    reasoning_bank: Arc<RwLock<ReasoningBank>>,
    gemini: GeminiClient,
}

impl QualityAssessor {
    pub async fn assess(&self, discovery: &DiscoveryCandidate) -> QualityAssessment {
        // 1. Novelty from HNSW similarity search
        let bank = self.reasoning_bank.read();
        let similar = bank.find_similar(&discovery.embedding, 5);
        let novelty = 1.0 - similar.first()
            .map(|p| p.similarity(&discovery.embedding))
            .unwrap_or(0.0);

        // 2. Other metrics from Gemini
        let prompt = format!(
            "Assess this code discovery for usefulness, clarity, correctness, and generalizability.\n\
             Discovery: {}\n\
             Respond with JSON: {{\"usefulness\": 0.0-1.0, \"clarity\": 0.0-1.0, ...}}",
            discovery.description
        );

        let response = self.gemini.complete(&[Message::human(&prompt)]).await?;
        let scores: GeminiScores = serde_json::from_str(&response.content())?;

        QualityAssessment {
            novelty,
            usefulness: scores.usefulness,
            clarity: scores.clarity,
            correctness: scores.correctness,
            generalizability: scores.generalizability,
            composite_score: novelty * 0.3
                + scores.usefulness * 0.25
                + scores.clarity * 0.15
                + scores.correctness * 0.2
                + scores.generalizability * 0.1,
        }
    }
}
```

### Milestone 4: Cloud Submission Pipeline (Week 4)

**Deliverables:**
- pi.ruv.io API client
- Submission queue with retry logic
- Response handling and status tracking
- Google Secrets integration

**Success Criteria:**
- [ ] Successful submissions to pi.ruv.io /v1/memories
- [ ] Retry logic handles transient failures
- [ ] Submission status tracked per discovery
- [ ] API key loaded from Google Secret Manager
- [ ] End-to-end submission test passes

**Technical Implementation:**
```rust
// crates/rvAgent/rvagent-backends/src/brain_client.rs

pub struct BrainSubmissionPipeline {
    client: BrainMcpClient,
    queue: RwLock<VecDeque<QueuedSubmission>>,
    status_tracker: DashMap<Uuid, SubmissionStatus>,
}

impl BrainSubmissionPipeline {
    pub async fn submit(&self, discovery: &DiscoveryLog) -> Result<Uuid> {
        let request = CloudSubmissionRequest::from(discovery.clone());

        // Add to queue
        self.queue.write().push_back(QueuedSubmission {
            request: request.clone(),
            attempts: 0,
            last_attempt: None,
        });

        self.status_tracker.insert(
            request.local_discovery_id,
            SubmissionStatus::Pending
        );

        Ok(request.local_discovery_id)
    }

    pub async fn process_queue(&self) -> usize {
        let mut processed = 0;

        while let Some(mut submission) = self.dequeue() {
            submission.attempts += 1;
            submission.last_attempt = Some(Instant::now());

            match self.client.brain_share(submission.request.clone()).await {
                Ok(response) => {
                    self.status_tracker.insert(
                        submission.request.local_discovery_id,
                        SubmissionStatus::Accepted {
                            memory_id: response.memory_id
                        }
                    );
                    processed += 1;
                }
                Err(e) if submission.attempts < 3 => {
                    // Requeue with exponential backoff
                    tokio::time::sleep(Duration::from_secs(2u64.pow(submission.attempts))).await;
                    self.queue.write().push_back(submission);
                }
                Err(e) => {
                    self.status_tracker.insert(
                        submission.request.local_discovery_id,
                        SubmissionStatus::Rejected {
                            reason: e.to_string()
                        }
                    );
                }
            }
        }

        processed
    }
}
```

### Milestone 5: Consolidation System (Week 5)

**Deliverables:**
- Local pattern update pipeline
- EWC++ consolidation integration
- Memory pressure management
- Pattern pruning system

**Success Criteria:**
- [ ] Patterns stored in ReasoningBank after each cycle
- [ ] EWC++ prevents catastrophic forgetting
- [ ] Memory pressure stays below 80%
- [ ] Low-quality patterns pruned automatically
- [ ] Consolidation completes in < 30 seconds

**Technical Implementation:**
```rust
// crates/rvAgent/rvagent-middleware/src/consolidation.rs

pub struct ConsolidationEngine {
    sona: Arc<SonaMiddleware>,
    reasoning_bank: Arc<RwLock<ReasoningBank>>,
    ewc: Arc<RwLock<EwcPlusPlus>>,
}

impl ConsolidationEngine {
    pub fn consolidate(&self, discoveries: &[DiscoveryLog]) -> ConsolidationReport {
        let mut report = ConsolidationReport::new();

        // 1. Add discoveries to ReasoningBank
        {
            let mut bank = self.reasoning_bank.write();
            for discovery in discoveries {
                let trajectory = discovery_to_trajectory(discovery);
                bank.add_trajectory(&trajectory);
            }

            // Extract patterns
            let patterns = bank.extract_patterns();
            report.patterns_extracted = patterns.len();
        }

        // 2. Run EWC++ consolidation
        {
            let mut ewc = self.ewc.write();
            let bank = self.reasoning_bank.read();

            // Check for task boundary
            let gradients = compute_pattern_gradients(&bank);
            if ewc.detect_task_boundary(&gradients) {
                ewc.start_new_task();
                report.new_task_started = true;
            }

            // Update Fisher information
            ewc.update_fisher(&gradients);

            // Consolidate if too many tasks
            if ewc.task_count() > 5 {
                ewc.consolidate_all_tasks();
                report.tasks_consolidated = true;
            }
        }

        // 3. Prune low-quality patterns
        {
            let mut bank = self.reasoning_bank.write();
            let before = bank.pattern_count();
            bank.prune_patterns(0.3, 0, 86400);
            bank.consolidate(0.95);
            report.patterns_pruned = before - bank.pattern_count();
        }

        // 4. Check memory pressure
        report.memory_pressure = self.calculate_memory_pressure();

        report
    }
}
```

### Milestone 6: Scheduler & Integration (Week 6)

**Deliverables:**
- Trigger detection system
- Schedule configuration
- Integration with mcp-brain-server
- CLI and MCP tools

**Success Criteria:**
- [ ] Scheduled runs execute at configured times
- [ ] Opportunistic triggers fire correctly
- [ ] Integration with existing training loop
- [ ] CLI commands work: `npx ruvector learning status`
- [ ] MCP tools accessible via brain_* pattern

**Technical Implementation:**
```rust
// crates/mcp-brain-server/src/learning_integration.rs

/// Integrate learning loop into brain server background task
pub async fn start_learning_daemon(state: AppState) {
    let coordinator = LearningLoopCoordinator::new().await
        .expect("Failed to initialize learning coordinator");

    let detector = TriggerDetector::default();

    // Check every minute
    let interval = Duration::from_secs(60);

    loop {
        tokio::time::sleep(interval).await;

        let learning_state = coordinator.state.read();

        if let Some(reason) = detector.should_trigger(&learning_state) {
            drop(learning_state);

            tracing::info!("Learning cycle triggered: {:?}", reason);

            match coordinator.run_cycle().await {
                Ok(report) => {
                    tracing::info!(
                        "Learning cycle complete: {} discoveries, {} submitted",
                        report.total_discoveries,
                        report.successful_submissions
                    );
                }
                Err(e) => {
                    tracing::error!("Learning cycle failed: {}", e);
                }
            }
        }
    }
}
```

---

## 6. Success Metrics

### 6.1 Quantitative Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Discoveries per cycle | >= 5 | Count of logged discoveries |
| Quality score average | >= 0.7 | Mean of composite scores |
| Submission success rate | >= 90% | Accepted / Total submitted |
| Cycle completion rate | >= 95% | Complete / Started |
| Memory pressure | <= 70% | Measured after consolidation |
| Cycle duration | <= 10 min | Time from start to complete |
| Pattern novelty | >= 50% | Truly new vs similar to existing |

### 6.2 Qualitative Success Criteria

1. **Knowledge Quality**: Discoveries are genuinely useful and actionable
2. **Method Attribution**: Clear provenance for all discoveries
3. **Cloud Integration**: Seamless submission to pi.ruv.io
4. **Learning Retention**: EWC++ prevents forgetting of valuable patterns
5. **Autonomy**: Runs without human intervention for days
6. **Observability**: Full logging and status monitoring

### 6.3 Monitoring Dashboard

```rust
/// Metrics exposed for monitoring
pub struct LearningMetrics {
    // Prometheus-compatible metrics
    pub cycles_total: Counter,
    pub cycles_successful: Counter,
    pub discoveries_total: Counter,
    pub discoveries_quality: Histogram,
    pub submissions_total: Counter,
    pub submissions_accepted: Counter,
    pub consolidation_duration_seconds: Histogram,
    pub memory_pressure_ratio: Gauge,
    pub patterns_count: Gauge,
}
```

---

## 7. Risk Mitigation

### 7.1 Identified Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limits | Medium | High | Exponential backoff, quota tracking |
| Memory exhaustion | Low | High | Pressure monitoring, automatic pruning |
| Gemini API costs | Medium | Medium | Daily quota, quality gating |
| Network failures | Low | Medium | Retry logic, offline queue |
| Catastrophic forgetting | Low | High | EWC++ consolidation |
| Low-quality submissions | Medium | Medium | Quality threshold gating |

### 7.2 Fallback Strategies

1. **Offline Mode**: Queue submissions when cloud unavailable
2. **Degraded Mode**: Skip Gemini analysis, use heuristics only
3. **Emergency Stop**: Manual cycle abort via MCP tool
4. **Rollback**: Restore patterns from last known good state

---

## 8. Future Extensions

### 8.1 Phase 2 Enhancements

- Multi-codebase scanning (cross-project learning)
- Federated learning between RuVector instances
- Active learning queries to user
- Automated code improvement suggestions

### 8.2 Phase 3 Vision

- Self-modifying optimization strategies
- Emergent pattern synthesis
- Cross-modal learning (code + docs + tests)
- Distributed swarm learning

---

## Appendix A: Action Summary Table

| Action | Cost | Phase | Preconditions | Effects |
|--------|------|-------|---------------|---------|
| scan_codebase | 2.0 | Discovery | idle/discovery, quota >= 1 | files_analyzed++, patterns++ |
| analyze_patterns_gemini | 3.0 | Discovery | patterns >= 1 | quality computed, novelty computed |
| deep_analysis | 5.0 | Discovery | patterns < 10, quota >= 3 | patterns += 5-15 |
| log_discovery | 1.0 | Logging | patterns >= 1 | logged++, pending_logs-- |
| assess_quality | 1.5 | Logging | logged >= 1 | quality computed, filtered |
| submit_to_cloud | 2.0 | Submission | pending >= 1, cloud healthy | submitted++, pending-- |
| update_local_patterns | 1.5 | Consolidation | logged >= 1 | patterns updated |
| ewc_consolidation | 3.0 | Consolidation | ewc pending | ewc complete, pressure-- |
| prune_patterns | 1.0 | Consolidation | pressure >= 0.8 | pressure--, patterns-- |
| complete_cycle | 0.5 | Consolidation | all done | phase = Complete |

---

## Appendix B: Configuration Schema

```yaml
# daily-learning-config.yaml
schedule:
  primary_run_time: "02:00"
  timezone: "UTC"
  min_cycle_interval_hours: 6
  max_daily_cycles: 4

triggers:
  git_commit_threshold: 10
  idle_period_minutes: 30
  trajectory_buffer_threshold: 0.8
  new_patterns_threshold: 20

quality:
  minimum_score: 0.7
  novelty_weight: 0.3
  usefulness_weight: 0.25
  clarity_weight: 0.15
  correctness_weight: 0.2
  generalizability_weight: 0.1

cloud:
  base_url: "https://pi.ruv.io"
  retry_attempts: 3
  retry_backoff_base_ms: 500

consolidation:
  ewc_lambda: 2000.0
  ewc_max_tasks: 10
  prune_quality_threshold: 0.3
  prune_max_age_seconds: 86400
  consolidation_similarity: 0.95
  memory_pressure_threshold: 0.8

gemini:
  model: "gemini-2.5-flash-preview-05-20"
  max_tokens: 4096
  temperature: 0.3
  secret_name: "projects/ruv-dev/secrets/gemini-api-key/versions/latest"
```

---

*Document Version: 1.0.0*
*Last Updated: 2026-03-16*
*Author: RuVector GOAP Planning Agent*
