//! GitHub Gist publisher for autonomous brain discoveries.
//!
//! Only publishes **truly novel** findings — requires:
//! - Minimum novelty score (new inferences, not just restated knowledge)
//! - Minimum evidence threshold (enough observations to be credible)
//! - Strange loop quality gate (meta-cognitive self-assessment)
//! - Rate limited to 1 gist per hour
//!
//! Each gist includes formal verification links, witness chain hashes,
//! and links back to π.ruv.io for independent verification.

use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

// ── Novelty thresholds ──
// Tuned aggressively: only publish genuinely novel, high-confidence findings.
// Previous thresholds (2/100/0.008) allowed floods of "weak co-occurrence" noise.
// These gates should yield ~1 gist per day at most when substantive new data arrives.
/// Minimum new inferences: must derive non-trivial forward-chained claims
const MIN_NEW_INFERENCES: usize = 5;
/// Minimum evidence observations — need enough data for statistical significance
const MIN_EVIDENCE: usize = 500;
/// Minimum strange loop quality score — higher = more self-aware reasoning
const MIN_STRANGE_LOOP_SCORE: f32 = 0.05;
/// Minimum propositions extracted in this cycle
const MIN_PROPOSITIONS: usize = 10;
/// Minimum SONA patterns — 0 means SONA isn't required (it needs trajectory data)
const MIN_SONA_PATTERNS: usize = 0;
/// Minimum Pareto front growth — evolution must have found new solutions
const MIN_PARETO_GROWTH: usize = 2;
/// Minimum confidence for ANY inference to be included in a discovery
const MIN_INFERENCE_CONFIDENCE: f64 = 0.60;

/// A discovery worthy of publishing.
///
/// The key distinction: `findings` should contain the actual discovered knowledge
/// (e.g., "category X relates_to category Y with 0.92 confidence"), not process
/// metrics (e.g., "10 propositions extracted"). The gist template formats
/// findings as the intellectual contribution, not a status report.
#[derive(Debug, Clone, Serialize)]
pub struct Discovery {
    pub title: String,
    pub category: String,
    pub abstract_text: String,
    /// The actual discovered knowledge — propositions, inferences, cross-domain connections.
    /// Each entry should be a substantive claim, not a metric.
    pub findings: Vec<String>,
    /// Methodology narrative
    pub methodology: Vec<String>,
    pub evidence_count: usize,
    pub confidence: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// IDs of supporting memories (for verification links)
    pub witness_memory_ids: Vec<String>,
    /// Witness chain hashes from RVF containers
    pub witness_hashes: Vec<String>,
    /// Strange loop self-assessment score
    pub strange_loop_score: f32,
    /// Number of new symbolic inferences (novelty signal)
    pub new_inferences: usize,
    /// Number of propositions extracted
    pub propositions_extracted: usize,
    /// SONA patterns that contributed
    pub sona_patterns: usize,
    /// Pareto front growth (new solutions found)
    pub pareto_growth: usize,
    /// Whether curiosity-driven exploration found this
    pub curiosity_triggered: bool,
    /// Self-reflection narrative from internal voice
    pub self_reflection: String,
    /// The actual propositions discovered (subject, predicate, object, confidence)
    pub propositions: Vec<(String, String, String, f64)>,
    /// The actual inferences derived
    pub inferences: Vec<String>,
}

impl Discovery {
    /// Filter out weak/generic inferences, keeping only substantive ones.
    /// Returns the strong inferences that survive the quality gate.
    pub fn strong_inferences(&self) -> Vec<&str> {
        self.inferences.iter()
            .filter(|inf| {
                // Reject generic "weak co-occurrence" noise
                let lower = inf.to_lowercase();
                if lower.contains("shows weak co-occurrence") {
                    return false;
                }
                // Reject inferences with generic cluster IDs as subjects
                if lower.starts_with("cluster_") {
                    return false;
                }
                // Require minimum confidence (parse from explanation string)
                if let Some(pct_start) = lower.find("confidence: ") {
                    let rest = &lower[pct_start + 12..];
                    if let Some(pct_end) = rest.find('%') {
                        if let Ok(pct) = rest[..pct_end].parse::<f64>() {
                            return pct >= MIN_INFERENCE_CONFIDENCE * 100.0;
                        }
                    }
                }
                // If we can't parse confidence, keep it only if it has substance
                !lower.contains("weak") && inf.len() > 30
            })
            .map(|s| s.as_str())
            .collect()
    }

    /// Filter propositions to only those with confidence >= threshold.
    pub fn strong_propositions(&self) -> Vec<&(String, String, String, f64)> {
        self.propositions.iter()
            .filter(|(subj, pred, _obj, conf)| {
                // Skip generic cluster labels
                if subj.starts_with("cluster_") { return false; }
                // Skip "co_occurs_with" at low confidence
                if pred == "co_occurs_with" && *conf < 0.55 { return false; }
                *conf >= MIN_INFERENCE_CONFIDENCE
            })
            .collect()
    }

    /// Check if this discovery meets the novelty bar for publishing.
    pub fn is_publishable(&self) -> bool {
        let strong = self.strong_inferences();
        let strong_props = self.strong_propositions();

        self.new_inferences >= MIN_NEW_INFERENCES
            && self.evidence_count >= MIN_EVIDENCE
            && self.strange_loop_score >= MIN_STRANGE_LOOP_SCORE
            && self.propositions_extracted >= MIN_PROPOSITIONS
            && self.pareto_growth >= MIN_PARETO_GROWTH
            && strong.len() >= 2  // Must have at least 2 non-trivial inferences
            && strong_props.len() >= 3  // Must have at least 3 substantive propositions
    }

    /// Explain why a discovery was or wasn't published.
    pub fn novelty_report(&self) -> String {
        let checks: Vec<(&str, bool, String)> = vec![
            ("inferences", self.new_inferences >= MIN_NEW_INFERENCES,
             format!("{}/{}", self.new_inferences, MIN_NEW_INFERENCES)),
            ("evidence", self.evidence_count >= MIN_EVIDENCE,
             format!("{}/{}", self.evidence_count, MIN_EVIDENCE)),
            ("strange_loop", self.strange_loop_score >= MIN_STRANGE_LOOP_SCORE,
             format!("{:.4}/{:.4}", self.strange_loop_score, MIN_STRANGE_LOOP_SCORE)),
            ("propositions", self.propositions_extracted >= MIN_PROPOSITIONS,
             format!("{}/{}", self.propositions_extracted, MIN_PROPOSITIONS)),
            ("pareto_growth", self.pareto_growth >= MIN_PARETO_GROWTH,
             format!("{}/{}", self.pareto_growth, MIN_PARETO_GROWTH)),
            ("has_inferences", !self.inferences.is_empty(),
             format!("{} items", self.inferences.len())),
        ];

        let failed: Vec<String> = checks.iter()
            .filter(|(_, ok, _)| !ok)
            .map(|(name, _, val)| format!("{} {}", name, val))
            .collect();

        if failed.is_empty() {
            "NOVEL: all thresholds met".to_string()
        } else {
            format!("NOT NOVEL: {}", failed.join(", "))
        }
    }
}

/// Response from GitHub Gist API
#[derive(Debug, Deserialize)]
struct GistResponse {
    html_url: String,
    #[allow(dead_code)]
    id: String,
}

/// Gist publisher with rate limiting and novelty gating
pub struct GistPublisher {
    token: String,
    last_publish: Mutex<Option<Instant>>,
    min_interval: Duration,
    published_count: Mutex<u64>,
    /// Titles of previously published discoveries (dedup within session)
    published_titles: Mutex<Vec<String>>,
}

impl GistPublisher {
    /// Create from env var GITHUB_GIST_PAT; returns None if not set.
    pub fn from_env() -> Option<Self> {
        let token = std::env::var("GITHUB_GIST_PAT").ok()?;
        if token.is_empty() {
            return None;
        }
        Some(Self {
            token,
            last_publish: Mutex::new(None),
            min_interval: Duration::from_secs(86400), // 24 hour minimum between gists
            published_count: Mutex::new(0),
            published_titles: Mutex::new(Vec::new()),
        })
    }

    /// Check if we can publish (rate limit + content dedup)
    pub fn can_publish(&self, discovery: &Discovery) -> bool {
        // Rate limit
        let last = self.last_publish.lock();
        if let Some(t) = *last {
            if t.elapsed() < self.min_interval {
                return false;
            }
        }
        // Content dedup: don't publish if core category + dominant inference already published
        let titles = self.published_titles.lock();
        let key = format!("{}:{}", discovery.category,
            discovery.strong_inferences().first().unwrap_or(&""));
        !titles.iter().any(|t| t == &key || t == &discovery.title)
    }

    pub fn published_count(&self) -> u64 {
        *self.published_count.lock()
    }

    /// Attempt to publish a discovery. Returns:
    /// - Ok(Some(url)) if published
    /// - Ok(None) if not novel enough or rate limited
    /// - Err if API failed
    pub async fn try_publish(&self, discovery: &Discovery) -> Result<Option<String>, String> {
        if !discovery.is_publishable() {
            tracing::debug!(
                "Discovery not publishable: {}",
                discovery.novelty_report()
            );
            return Ok(None);
        }
        if !self.can_publish(discovery) {
            tracing::debug!("Gist publish rate limited or duplicate content");
            return Ok(None);
        }

        // Only include strong inferences and propositions in the gist
        let strong_inferences = discovery.strong_inferences();
        let strong_propositions = discovery.strong_propositions();

        if strong_inferences.len() < 2 {
            tracing::debug!("Discovery has {} strong inferences (need 2+), skipping", strong_inferences.len());
            return Ok(None);
        }

        let filename = format!(
            "pi-brain-discovery-{}.md",
            discovery.timestamp.format("%Y%m%d-%H%M%S")
        );

        // Use Gemini with Google Grounding to do deep research on the discovery
        // topics, then produce a substantive article with real-world context
        let raw_content = format_academic_gist(discovery);
        let content = match research_and_write_with_gemini(discovery, &strong_inferences, &strong_propositions).await {
            Ok(polished) => {
                tracing::info!("Gemini deep research produced {} chars", polished.len());
                polished
            }
            Err(e) => {
                tracing::warn!("Gemini deep research failed ({}), using raw content", e);
                raw_content
            }
        };

        let body = serde_json::json!({
            "description": format!("π Brain Discovery: {}", discovery.title),
            "public": true,
            "files": {
                filename: {
                    "content": content
                }
            }
        });

        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.github.com/gists")
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "pi-brain/0.1")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("HTTP error: {}", e))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!(
                "GitHub API {}: {}",
                status,
                &text[..text.len().min(200)]
            ));
        }

        let gist: GistResponse = resp
            .json()
            .await
            .map_err(|e| format!("Parse error: {}", e))?;

        *self.last_publish.lock() = Some(Instant::now());
        *self.published_count.lock() += 1;
        {
            let mut titles = self.published_titles.lock();
            titles.push(discovery.title.clone());
            // Also store the content dedup key
            let key = format!("{}:{}", discovery.category,
                discovery.strong_inferences().first().unwrap_or(&""));
            titles.push(key);
        }

        tracing::info!(
            "Published discovery gist: {} -> {} (novelty: {})",
            discovery.title,
            gist.html_url,
            discovery.novelty_report()
        );

        Ok(Some(gist.html_url))
    }
}

/// Format a discovery as an academic-style markdown document with verification.
/// Format a discovery as an academic-style document focused on the actual
/// discovered knowledge, not pipeline metrics.
fn format_academic_gist(d: &Discovery) -> String {
    // Format propositions as a knowledge table
    let propositions_md = if d.propositions.is_empty() {
        String::new()
    } else {
        let rows: Vec<String> = d.propositions.iter().map(|(s, p, o, c)| {
            format!("| {} | {} | {} | {:.2} |", s, p, o, c)
        }).collect();
        format!(
            "| Subject | Relation | Object | Confidence |\n\
             |---------|----------|--------|------------|\n\
             {}\n",
            rows.join("\n")
        )
    };

    // Format inferences as numbered claims
    let inferences_md = d.inferences.iter().enumerate()
        .map(|(i, inf)| format!("{}. {}", i + 1, inf))
        .collect::<Vec<_>>()
        .join("\n");

    // Format findings (the high-level insights)
    let findings_md = d.findings.iter().enumerate()
        .map(|(i, f)| format!("{}. {}", i + 1, f))
        .collect::<Vec<_>>()
        .join("\n");

    // Witness links
    let witness_md = d.witness_memory_ids.iter().take(5)
        .zip(d.witness_hashes.iter().take(5).chain(std::iter::repeat(&String::new())))
        .map(|(id, hash)| {
            let short = &id[..id.len().min(8)];
            if hash.is_empty() {
                format!("| [`{}`](https://pi.ruv.io/v1/memories/{}) | — |", short, id)
            } else {
                format!("| [`{}`](https://pi.ruv.io/v1/memories/{}) | `{}` |", short, id, &hash[..hash.len().min(16)])
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"# {title}

> **Domain:** {category} · **Generated:** {timestamp} by [π Brain](https://pi.ruv.io)
> Autonomous discovery from {evidence} knowledge observations

---

## Abstract

{abstract_text}

## Discovered Knowledge

### Novel Inferences

The following claims were derived by forward-chaining symbolic reasoning
over extracted propositions. These are **new knowledge** not present in
any single input observation — they emerge from combining evidence across
multiple sources:

{inferences}

### Extracted Propositions

Symbolic knowledge extracted from {evidence} observations across {n_clusters}
domain clusters:

{propositions}

## Cross-Domain Insights

{findings}

## Internal Deliberation

The brain's Internal Voice reflected on this learning cycle:

> {self_reflection}

## Verification

### Witness Chain

| Memory | Hash |
|--------|------|
{witnesses}

### Reproduce

```bash
curl -H "Authorization: Bearer KEY" "https://pi.ruv.io/v1/propositions"
curl -H "Authorization: Bearer KEY" "https://pi.ruv.io/v1/memories/search?q={category}&limit=10"
curl -H "Authorization: Bearer KEY" "https://pi.ruv.io/v1/cognitive/status"
```

---

*Autonomously generated by [π Brain](https://pi.ruv.io). No human authored this content.
{evidence} observations · {n_inferences} inferences · {n_props} propositions · strange loop {sl:.4}*
"#,
        title = d.title,
        category = d.category,
        timestamp = d.timestamp.format("%Y-%m-%d %H:%M UTC"),
        evidence = d.evidence_count,
        abstract_text = d.abstract_text,
        inferences = if inferences_md.is_empty() { "No novel inferences this cycle.".to_string() } else { inferences_md },
        n_clusters = d.propositions.len().max(1),
        propositions = if propositions_md.is_empty() { "No propositions extracted.".to_string() } else { propositions_md },
        findings = if findings_md.is_empty() { "No cross-domain insights this cycle.".to_string() } else { findings_md },
        self_reflection = d.self_reflection,
        witnesses = if witness_md.is_empty() { "| — | — |".to_string() } else { witness_md },
        n_inferences = d.new_inferences,
        n_props = d.propositions_extracted,
        sl = d.strange_loop_score,
    )
}

/// Use Gemini with Google Grounding to conduct deep research on discovery topics,
/// then produce a substantive article with real-world context, recent papers,
/// and specific domain knowledge — not just cluster co-occurrence summaries.
async fn research_and_write_with_gemini(
    discovery: &Discovery,
    strong_inferences: &[&str],
    strong_propositions: &[&(String, String, String, f64)],
) -> Result<String, String> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| "GEMINI_API_KEY not set".to_string())?;
    let model = std::env::var("GEMINI_MODEL")
        .unwrap_or_else(|_| "gemini-2.5-flash".to_string());

    // Build summaries from STRONG signals only (filtered out weak co-occurrences)
    let inferences_summary = strong_inferences.iter()
        .take(8)
        .map(|i| format!("- {}", i))
        .collect::<Vec<_>>()
        .join("\n");

    let propositions_summary = strong_propositions.iter()
        .take(10)
        .map(|(s, p, o, c)| format!("- {} {} {} (confidence: {:.0}%)", s, p, o, c * 100.0))
        .collect::<Vec<_>>()
        .join("\n");

    let findings_summary = discovery.findings.iter()
        .filter(|f| !f.to_lowercase().contains("weak co-occurrence"))
        .take(5)
        .map(|f| format!("- {}", f))
        .collect::<Vec<_>>()
        .join("\n");

    // Extract the key domain topics for grounding research
    let topics: Vec<&str> = strong_propositions.iter()
        .flat_map(|(s, _p, o, _c)| vec![s.as_str(), o.as_str()])
        .filter(|t| !t.starts_with("cluster_") && !t.is_empty())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .take(5)
        .collect();

    let prompt = format!(
r#"You are a research scientist at the π Brain autonomous AI knowledge system (pi.ruv.io).

The π Brain has identified the following substantive cross-domain connections. Your job is to:

1. **Use Google Search grounding** to find REAL recent papers, news, or data that validate or contextualize these connections
2. Write a deep research article that connects the brain's autonomous findings to real-world knowledge
3. Provide genuinely novel analysis — not just "X co-occurs with Y"

## Brain's Filtered Findings (only high-confidence signals)

**Strong inferences (>60% confidence):**
{inferences}

**Strong propositions:**
{propositions}

**Cross-domain insights:**
{findings}

**Domain topics to research:** {topics}

## Research Instructions

Use Google Search to find:
- Recent academic papers (2024-2026) related to these domain intersections
- Real-world events or data that support or contradict these findings
- Novel connections that the brain may have missed
- Quantitative data points (statistics, benchmarks, metrics)

## Article Structure

Write the article as:

### Title
A specific, compelling title about the actual discovery — NOT generic like "Preliminary Co-occurrence of X with Y"

### Summary
2-3 sentences explaining what was found and why it matters to a general audience

### Deep Analysis
For each significant finding:
- What the brain detected (the raw signal)
- What Google Search reveals about this connection in the real world
- Why this matters (practical implications)
- Confidence assessment with honest limitations

### Real-World Context
Cite specific recent papers, events, or datasets that ground these findings. Include URLs where possible.

### Methodology
Brief explanation of how the π Brain works: embedding-based clustering, cosine similarity, symbolic forward-chaining, and confidence-gated language

### Limitations
Be brutally honest about what this does NOT prove

### Verification
- Dashboard: https://pi.ruv.io
- API: https://pi.ruv.io/v1/status
- Propositions: https://pi.ruv.io/v1/propositions
- Witness hashes: {witnesses}

**Stats:** {evidence} observations, {n_inferences} strong inferences, {n_props} propositions

## Rules
- NEVER pad with generic text. Every paragraph must contain specific, verifiable claims.
- If grounding search returns nothing relevant, say so — don't fabricate.
- Use real paper titles, author names, publication venues. If unsure, say "reportedly" or "according to search results".
- NO "weak co-occurrence" language — that's been filtered out. Focus on the strong signals.
- Keep under 2500 words. Quality over quantity.
- Output ONLY the markdown article.

Write the article now:"#,
        inferences = if inferences_summary.is_empty() { "No strong inferences survived filtering.".to_string() } else { inferences_summary },
        propositions = if propositions_summary.is_empty() { "No strong propositions survived filtering.".to_string() } else { propositions_summary },
        findings = if findings_summary.is_empty() { "No non-trivial findings.".to_string() } else { findings_summary },
        topics = topics.join(", "),
        evidence = discovery.evidence_count,
        n_inferences = strong_inferences.len(),
        n_props = strong_propositions.len(),
        witnesses = discovery.witness_hashes.iter().take(3)
            .map(|h| format!("`{}`", h))
            .collect::<Vec<_>>()
            .join(", "),
    );

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let grounding = std::env::var("GEMINI_GROUNDING")
        .unwrap_or_else(|_| "true".to_string()) == "true";

    let client = reqwest::Client::new();

    // ── Pass 1: Grounded research on the topics ──
    // Ask Gemini to research the domain topics using Google Search, returning
    // structured findings we can feed back to the brain.
    let research_prompt = format!(
        "Research these topics using Google Search and return a structured summary \
         of the most relevant recent findings (2024-2026):\n\
         Topics: {topics}\n\
         Context: An autonomous AI knowledge system detected associations between these domains.\n\n\
         For each topic, provide:\n\
         1. Most relevant recent paper or article (title, authors, date, URL if available)\n\
         2. Key quantitative finding or statistic\n\
         3. How it relates to the other topics\n\n\
         Be concise. Return ONLY factual findings, no filler. Max 800 words.",
        topics = topics.join(", ")
    );

    let pass1_result = call_gemini(&client, &url, &research_prompt, grounding, 4096, 0.2).await;
    let grounded_research = match pass1_result {
        Ok(text) => {
            tracing::info!("Pass 1 (grounded research): {} chars", text.len());
            text
        }
        Err(e) => {
            tracing::warn!("Pass 1 grounding failed: {}", e);
            String::new()
        }
    };

    // ── Pass 2: Brain-guided search via pi.ruv.io ──
    // Search the brain's memory for additional context related to the grounded findings.
    let brain_context = if !topics.is_empty() {
        let brain_url = std::env::var("BRAIN_URL")
            .unwrap_or_else(|_| "https://pi.ruv.io".to_string());
        let brain_key = std::env::var("BRAIN_SYSTEM_KEY")
            .or_else(|_| std::env::var("brain-api-key"))
            .unwrap_or_default();

        let mut brain_memories = Vec::new();
        for topic in &topics {
            let search_url = format!(
                "{}/v1/memories/search?q={}&limit=3",
                brain_url, topic.replace(' ', "%20")
            );
            if let Ok(resp) = client.get(&search_url)
                .header("Authorization", format!("Bearer {}", brain_key))
                .send().await
            {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(results) = json.get("results").and_then(|r| r.as_array()) {
                        for mem in results.iter().take(2) {
                            if let (Some(title), Some(content)) = (
                                mem.get("title").and_then(|t| t.as_str()),
                                mem.get("content").and_then(|c| c.as_str()),
                            ) {
                                brain_memories.push(format!(
                                    "- **{}**: {}", title, &content[..content.len().min(200)]
                                ));
                            }
                        }
                    }
                }
            }
        }
        if brain_memories.is_empty() {
            String::new()
        } else {
            format!("\n## Brain Memory Context\n\n{}", brain_memories.join("\n"))
        }
    } else {
        String::new()
    };

    // ── Pass 3: Final synthesis — combine brain signals + grounded research ──
    let synthesis_prompt = format!(
        "{original_prompt}\n\n\
         ## Additional Context from Research\n\n\
         ### Google Search Grounded Findings\n\n\
         {grounded}\n\n\
         ### π Brain Memory Search Results\n\n\
         {brain}\n\n\
         IMPORTANT: Synthesize ALL of the above — the brain's autonomous findings, \
         the grounded research, and the brain memory context — into a single cohesive \
         article. The grounded research provides real-world validation; the brain \
         memories provide internal context. Together they should produce genuinely \
         novel analysis that neither source could produce alone.\n\n\
         Write the final article now:",
        original_prompt = prompt,
        grounded = if grounded_research.is_empty() { "No grounded findings available.".to_string() } else { grounded_research },
        brain = if brain_context.is_empty() { "No additional brain memories found.".to_string() } else { brain_context },
    );

    let final_text = call_gemini(&client, &url, &synthesis_prompt, grounding, 8192, 0.3).await?;

    // Append verification footer
    let footer = format!(
        "\n\n---\n\n\
         *This article was autonomously generated by the [π Brain](https://pi.ruv.io) \
         using a 3-pass research loop: (1) Google-grounded topic research, \
         (2) brain memory search for internal context, (3) Gemini synthesis. \
         Based on {} observations. No human authored or curated the findings.*\n\n\
         **Live Dashboard:** [π.ruv.io](https://pi.ruv.io) · \
         **API:** [/v1/status](https://pi.ruv.io/v1/status) · \
         **Verify:** [/v1/propositions](https://pi.ruv.io/v1/propositions)\n",
        discovery.evidence_count
    );

    Ok(format!("{}{}", final_text.trim(), footer))
}

/// Call Gemini API with optional grounding.
async fn call_gemini(
    client: &reqwest::Client,
    url: &str,
    prompt: &str,
    grounding: bool,
    max_tokens: u32,
    temperature: f32,
) -> Result<String, String> {
    let mut body = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature
        }
    });

    if grounding {
        body["tools"] = serde_json::json!([{"google_search": {}}]);
    }

    let resp = client
        .post(url)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Gemini HTTP error: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Gemini API {}: {}", status, &text[..text.len().min(200)]));
    }

    let json: serde_json::Value = resp.json().await
        .map_err(|e| format!("Gemini parse error: {}", e))?;

    json.get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.get(0))
        .and_then(|p| p.get("text"))
        .and_then(|t| t.as_str())
        .map(|s| s.to_string())
        .ok_or("No text in Gemini response".to_string())
}
