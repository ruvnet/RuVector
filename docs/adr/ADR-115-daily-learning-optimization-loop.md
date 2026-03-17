# ADR-115: Daily Learning and Optimization Loop with GOAP Reasoning

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | Proposed                                        |
| **Date**    | 2026-03-16                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-110 (Neural-Symbolic), ADR-107 (rvAgent)    |
| **Depends** | ADR-107, ADR-110, ADR-104, ADR-112              |
| **Crates**  | `rvagent-learning` (new), `mcp-brain-server`, `rvagent-mcp` |

## 1. Context

The π.ruv.io shared brain system and RuVector ecosystem need a systematic approach to:
1. **Continuously discover** new knowledge, patterns, and optimizations
2. **Log discoveries** with full provenance (tools used, methods applied)
3. **Feed knowledge** to the cloud brain for collective intelligence
4. **Learn autonomously** from successful patterns using GOAP-style reasoning

### Current State

- **π.ruv.io** (Cloud Run): Stores shared knowledge with SONA learning, witness chains
- **rvagent**: Native Rust agent framework with MCP server (ADR-112)
- **SONA**: Self-Optimizing Neural Architecture for pattern learning
- **ReasoningBank**: HNSW-indexed pattern storage (150x-12,500x faster search)
- **Internal Voice**: Neural-symbolic integration for reasoning transparency (ADR-110)

### Gap

No automated daily learning loop exists to:
- Systematically explore and discover new patterns
- Log the discovery methodology for reproducibility
- Submit discoveries to the collective brain
- Use intelligent planning (GOAP) for goal-directed discovery

---

## 2. Decision

Implement a **Daily Learning and Optimization Loop** using Goal-Oriented Action Planning (GOAP) with Gemini 2.5 Flash for reasoning.

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY LEARNING LOOP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   DISCOVER  │────▶│   ASSESS    │────▶│    LOG      │        │
│  │  (Explore)  │     │  (Quality)  │     │ (Provenance)│        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐        │
│  │   SUBMIT    │◀────│ CONSOLIDATE │◀────│   PLAN      │        │
│  │ (π.ruv.io)  │     │   (SONA)    │     │   (GOAP)    │        │
│  └─────────────┘     └─────────────┘     └─────────────┘        │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                    GOAP PLANNER (Gemini 2.5 Flash)               │
│  World State │ Goals │ Actions │ Preconditions │ Effects        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 GOAP State Space

```rust
/// World state for GOAP planner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningWorldState {
    // Discovery state
    pub patterns_discovered: usize,
    pub patterns_pending_assessment: usize,
    pub patterns_pending_submission: usize,

    // Quality metrics
    pub novelty_threshold: f64,
    pub quality_threshold: f64,

    // Resource state
    pub api_quota_remaining: usize,
    pub memory_utilization: f64,
    pub last_submission_time: DateTime<Utc>,

    // Learning state
    pub sona_patterns_count: usize,
    pub reasoning_bank_entries: usize,
    pub consolidation_due: bool,

    // Connection state
    pub pi_ruv_io_connected: bool,
    pub gemini_available: bool,
}
```

### 2.3 GOAP Goals

```rust
/// Goals for the learning loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningGoal {
    /// Discover N new patterns
    DiscoverPatterns { target_count: usize },

    /// Submit discoveries to π.ruv.io
    SubmitToCloudBrain { min_quality: f64 },

    /// Consolidate learned patterns (prevent forgetting)
    ConsolidateLearning,

    /// Optimize specific domain
    OptimizeDomain { domain: String },

    /// Complete daily learning cycle
    CompleteDailyCycle,
}
```

### 2.4 GOAP Actions

| Action | Preconditions | Effects | Cost |
|--------|---------------|---------|------|
| `ScanCodebase` | `!scanning`, `memory_util < 0.8` | `+patterns_discovered` | 10 |
| `AnalyzePatterns` | `patterns_pending > 0` | `+assessed_patterns` | 5 |
| `ComputeNovelty` | `assessed_pattern` | `novelty_score` | 3 |
| `LogDiscovery` | `novelty > threshold` | `+logged_discovery` | 2 |
| `SubmitToPi` | `logged_discovery`, `connected` | `-pending`, `+submitted` | 8 |
| `ConsolidateSona` | `consolidation_due` | `!consolidation_due` | 15 |
| `ReasonWithGemini` | `gemini_available`, `quota > 0` | `+reasoning_result` | 20 |
| `RefreshConnection` | `!connected` | `connected` | 5 |

```rust
/// GOAP action definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoapAction {
    pub name: String,
    pub preconditions: HashMap<String, StateValue>,
    pub effects: HashMap<String, StateValue>,
    pub cost: f64,
    pub executor: ActionExecutor,
}

#[derive(Debug, Clone)]
pub enum ActionExecutor {
    /// Rust function
    Native(fn(&mut LearningWorldState) -> Result<ActionResult>),

    /// rvagent tool call
    RvAgent { tool: String, params: Value },

    /// MCP tool call
    Mcp { server: String, tool: String, params: Value },

    /// Gemini reasoning
    Gemini { prompt_template: String },
}
```

### 2.5 Discovery Logging System

```rust
/// Discovery log entry with full provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryLog {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,

    // What was discovered
    pub category: DiscoveryCategory,
    pub title: String,
    pub content: String,
    pub embedding: Vec<f32>,

    // How it was discovered
    pub method: DiscoveryMethod,
    pub tools_used: Vec<ToolUsage>,
    pub reasoning_chain: Vec<ReasoningStep>,

    // Quality assessment
    pub novelty_score: f64,
    pub quality_score: f64,
    pub confidence: f64,

    // Submission status
    pub submitted_to_pi: bool,
    pub pi_memory_id: Option<Uuid>,
    pub witness_chain: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiscoveryCategory {
    Pattern,       // Code pattern
    Optimization,  // Performance optimization
    Architecture,  // Architectural insight
    Security,      // Security consideration
    Convention,    // Best practice
    Tooling,       // Tool/workflow improvement
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUsage {
    pub tool_name: String,
    pub tool_type: ToolType,
    pub parameters: Value,
    pub duration_ms: u64,
    pub success: bool,
    pub output_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolType {
    RuVectorCrate { crate_name: String },
    RvAgentTool { tool_id: String },
    McpTool { server: String, tool: String },
    ExternalApi { api_name: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub step_number: usize,
    pub thought_type: ThoughtType,
    pub content: String,
    pub evidence: Vec<Uuid>,
    pub confidence: f64,
}
```

### 2.6 π.ruv.io Integration

```rust
/// Client for submitting to π.ruv.io cloud brain
pub struct PiRuvIoClient {
    endpoint: String,
    auth_token: String,
    contributor_id: String,
}

impl PiRuvIoClient {
    /// Create client from Google Secrets
    pub async fn from_google_secrets(project: &str) -> Result<Self> {
        let secret_manager = SecretManager::new(project).await?;
        let auth_token = secret_manager.get_secret("PI_RUVIO_AUTH_TOKEN").await?;
        let contributor_id = secret_manager.get_secret("PI_CONTRIBUTOR_ID").await?;

        Ok(Self {
            endpoint: "https://π.ruv.io/v1".to_string(),
            auth_token,
            contributor_id,
        })
    }

    /// Submit discovery to shared brain
    pub async fn submit(&self, discovery: &DiscoveryLog) -> Result<BrainShareResult> {
        let payload = BrainSharePayload {
            category: discovery.category.to_string(),
            title: discovery.title.clone(),
            content: discovery.content.clone(),
            tags: self.extract_tags(discovery),
            code_snippet: discovery.extract_code_snippet(),
        };

        let response = self.client
            .post(&format!("{}/brain/share", self.endpoint))
            .bearer_auth(&self.auth_token)
            .json(&payload)
            .send()
            .await?;

        Ok(response.json().await?)
    }

    /// Search for related knowledge
    pub async fn search(&self, query: &str) -> Result<Vec<BrainMemory>> {
        let response = self.client
            .post(&format!("{}/brain/search", self.endpoint))
            .bearer_auth(&self.auth_token)
            .json(&json!({ "query": query }))
            .send()
            .await?;

        Ok(response.json().await?)
    }
}
```

### 2.7 Gemini 2.5 Flash Integration (GOAP Reasoning)

```rust
/// Gemini 2.5 Flash client for GOAP reasoning
pub struct GeminiGoapReasoner {
    api_key: String,
    model: String,
}

impl GeminiGoapReasoner {
    pub async fn from_google_secrets(project: &str) -> Result<Self> {
        let secret_manager = SecretManager::new(project).await?;
        let api_key = secret_manager.get_secret("GEMINI_API_KEY").await?;

        Ok(Self {
            api_key,
            model: "gemini-2.5-flash".to_string(),
        })
    }

    /// Generate GOAP plan for given goal
    pub async fn plan(&self, state: &LearningWorldState, goal: &LearningGoal) -> Result<GoapPlan> {
        let prompt = format!(r#"
You are a GOAP (Goal-Oriented Action Planning) reasoner for the RuVector learning system.

## Current World State
{state_json}

## Goal
{goal_json}

## Available Actions
{actions_json}

Generate an optimal action sequence to achieve the goal. Consider:
1. Action preconditions must be satisfied
2. Minimize total cost
3. Handle failures gracefully
4. Prefer parallel actions when independent

Output JSON:
{{
  "plan": [
    {{"action": "action_name", "params": {{}}, "parallel_with": []}}
  ],
  "estimated_cost": 0.0,
  "reasoning": "explanation"
}}
"#,
            state_json = serde_json::to_string_pretty(state)?,
            goal_json = serde_json::to_string_pretty(goal)?,
            actions_json = self.format_actions(),
        );

        let response = self.call_gemini(&prompt).await?;
        self.parse_plan(&response)
    }

    /// Assess discovery quality and novelty
    pub async fn assess_discovery(&self, discovery: &DiscoveryLog) -> Result<QualityAssessment> {
        let prompt = format!(r#"
Assess the quality and novelty of this discovery:

## Discovery
Title: {title}
Category: {category}
Content: {content}

## Assessment Criteria
1. Novelty (0-1): How new/unique is this knowledge?
2. Quality (0-1): How well-formulated and accurate?
3. Utility (0-1): How useful for other developers?
4. Confidence (0-1): How confident in the assessment?

Output JSON:
{{
  "novelty": 0.0,
  "quality": 0.0,
  "utility": 0.0,
  "confidence": 0.0,
  "reasoning": "explanation",
  "suggested_tags": []
}}
"#,
            title = discovery.title,
            category = discovery.category.to_string(),
            content = discovery.content,
        );

        let response = self.call_gemini(&prompt).await?;
        self.parse_assessment(&response)
    }
}
```

### 2.8 Daily Schedule and Triggers

```rust
/// Daily learning loop scheduler
pub struct DailyLearningScheduler {
    config: SchedulerConfig,
    goap_planner: GoapPlanner,
    gemini: GeminiGoapReasoner,
    pi_client: PiRuvIoClient,
    discovery_log: DiscoveryLogStore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Cron expression for daily run (default: "0 2 * * *" = 2 AM)
    pub cron_schedule: String,

    /// Maximum discoveries per cycle
    pub max_discoveries_per_cycle: usize,

    /// Minimum novelty to submit
    pub novelty_threshold: f64,

    /// Enable continuous mode (run on code changes)
    pub continuous_mode: bool,

    /// Domains to explore
    pub exploration_domains: Vec<String>,
}

impl DailyLearningScheduler {
    /// Run the daily learning cycle
    pub async fn run_cycle(&mut self) -> Result<CycleResult> {
        let start_time = Instant::now();
        let mut state = self.get_current_state().await?;

        // 1. Plan the discovery session
        let goal = LearningGoal::CompleteDailyCycle;
        let plan = self.gemini.plan(&state, &goal).await?;

        tracing::info!(?plan, "GOAP plan generated");

        // 2. Execute plan actions
        let mut discoveries = Vec::new();
        for action in plan.actions {
            let result = self.execute_action(&mut state, &action).await?;

            if let Some(discovery) = result.discovery {
                // 3. Log the discovery with provenance
                self.discovery_log.store(&discovery).await?;
                discoveries.push(discovery);
            }
        }

        // 4. Assess and submit high-quality discoveries
        for discovery in &mut discoveries {
            let assessment = self.gemini.assess_discovery(discovery).await?;
            discovery.novelty_score = assessment.novelty;
            discovery.quality_score = assessment.quality;

            if assessment.novelty > self.config.novelty_threshold {
                let result = self.pi_client.submit(discovery).await?;
                discovery.submitted_to_pi = true;
                discovery.pi_memory_id = Some(result.memory_id);
            }
        }

        // 5. Consolidate learning (SONA)
        self.consolidate_learning(&discoveries).await?;

        Ok(CycleResult {
            duration: start_time.elapsed(),
            discoveries_found: discoveries.len(),
            discoveries_submitted: discoveries.iter().filter(|d| d.submitted_to_pi).count(),
            state_after: state,
        })
    }

    /// Trigger on code change (continuous mode)
    pub async fn on_code_change(&mut self, changed_files: &[PathBuf]) -> Result<()> {
        if !self.config.continuous_mode {
            return Ok(());
        }

        let goal = LearningGoal::DiscoverPatterns { target_count: 1 };
        let mut state = self.get_current_state().await?;
        state.exploration_focus = Some(changed_files.to_vec());

        let plan = self.gemini.plan(&state, &goal).await?;
        self.execute_plan(&mut state, &plan).await
    }
}
```

### 2.9 Crate Structure

```
crates/rvAgent/rvagent-learning/
  Cargo.toml
  src/
    lib.rs              # Public API
    goap/
      mod.rs            # GOAP planner
      state.rs          # World state
      actions.rs        # Action definitions
      planner.rs        # A* planning
    discovery/
      mod.rs            # Discovery system
      scanner.rs        # Codebase scanner
      analyzer.rs       # Pattern analyzer
      logger.rs         # Discovery logging
    integration/
      mod.rs            # External integrations
      pi_ruvio.rs       # π.ruv.io client
      gemini.rs         # Gemini 2.5 Flash
      secrets.rs        # Google Secrets Manager
    scheduler/
      mod.rs            # Scheduling system
      cron.rs           # Cron-based triggers
      continuous.rs     # File-change triggers
    consolidation/
      mod.rs            # Learning consolidation
      sona.rs           # SONA integration
      reasoning_bank.rs # Pattern storage
  tests/
    goap_tests.rs
    discovery_tests.rs
    integration_tests.rs
```

---

## 3. Google Cloud Configuration

### 3.1 Google Secrets Required

| Secret Name | Description |
|-------------|-------------|
| `GEMINI_API_KEY` | Gemini 2.5 Flash API key |
| `PI_RUVIO_AUTH_TOKEN` | Authentication for π.ruv.io |
| `PI_CONTRIBUTOR_ID` | Contributor identifier |

### 3.2 Secret Access

```bash
# Create secrets
gcloud secrets create GEMINI_API_KEY --project=ruv-dev
gcloud secrets create PI_RUVIO_AUTH_TOKEN --project=ruv-dev
gcloud secrets create PI_CONTRIBUTOR_ID --project=ruv-dev

# Add secret versions
echo -n "your-gemini-key" | gcloud secrets versions add GEMINI_API_KEY --data-file=-
echo -n "your-auth-token" | gcloud secrets versions add PI_RUVIO_AUTH_TOKEN --data-file=-
echo -n "your-contributor-id" | gcloud secrets versions add PI_CONTRIBUTOR_ID --data-file=-

# Grant access to Cloud Run service
gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
  --member="serviceAccount:ruvbrain@ruv-dev.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

---

## 4. Implementation Milestones

| Phase | Component | Effort | Success Criteria |
|-------|-----------|--------|------------------|
| 1 | GOAP State & Actions | 8 hrs | State machine compiles, actions defined |
| 2 | Discovery Scanner | 12 hrs | Scans codebase, extracts patterns |
| 3 | Gemini Integration | 6 hrs | GOAP plans generated, quality assessed |
| 4 | π.ruv.io Client | 4 hrs | Discoveries submitted successfully |
| 5 | Discovery Logging | 6 hrs | Full provenance captured in logs |
| 6 | Scheduler | 4 hrs | Daily cron + continuous triggers |
| 7 | SONA Consolidation | 6 hrs | Patterns stored, no forgetting |
| 8 | Integration Tests | 8 hrs | E2E cycle passes |

**Total: ~54 hours**

---

## 5. Consequences

### Positive
- **Autonomous learning**: System discovers and learns without manual intervention
- **Full provenance**: Every discovery has traceable methodology
- **Collective intelligence**: Knowledge shared with π.ruv.io community
- **Intelligent planning**: GOAP ensures goal-directed, efficient discovery
- **Cost-effective reasoning**: Gemini 2.5 Flash provides fast, cheap inference

### Negative
- **API costs**: Gemini calls incur per-token costs
- **Complexity**: GOAP planner adds architectural complexity
- **Latency**: Cloud calls add network latency
- **Secret management**: Requires Google Cloud setup

### Neutral
- Daily schedule configurable
- Continuous mode optional
- Novelty threshold tunable

---

## 6. References

- ADR-107: rvAgent Native Swarm WASM
- ADR-110: Neural-Symbolic Internal Voice
- ADR-112: rvAgent MCP Server
- [GOAP for Game AI](https://alumni.media.mit.edu/~jorkin/goap.html)
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Google Secret Manager](https://cloud.google.com/secret-manager)
