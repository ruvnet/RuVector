//! Resend email integration for pi.ruv.io brain notifications (ADR-125)
//!
//! Sends transactional emails via Resend HTTP API from pi@ruv.io.
//! Rate-limited per category to prevent notification storms.
//! Supports conversational email interaction with help and welcome flows.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Open tracking record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailOpen {
    pub tracking_id: String,
    pub category: String,
    pub subject: String,
    pub opened_at: chrono::DateTime<chrono::Utc>,
    pub user_agent: Option<String>,
}

/// Open tracking store — thread-safe, bounded circular buffer
#[derive(Clone)]
pub struct OpenTracker {
    opens: std::sync::Arc<Mutex<Vec<EmailOpen>>>,
    /// category -> (sent, opened) for rate calculations
    stats: std::sync::Arc<Mutex<HashMap<String, (u64, u64)>>>,
}

impl OpenTracker {
    pub fn new() -> Self {
        Self {
            opens: std::sync::Arc::new(Mutex::new(Vec::new())),
            stats: std::sync::Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn record_open(&self, tracking_id: &str, category: &str, subject: &str, user_agent: Option<&str>) {
        let open = EmailOpen {
            tracking_id: tracking_id.to_string(),
            category: category.to_string(),
            subject: subject.to_string(),
            opened_at: chrono::Utc::now(),
            user_agent: user_agent.map(|s| s.to_string()),
        };

        let mut opens = self.opens.lock();
        opens.push(open);
        // Keep bounded to last 1000 opens
        if opens.len() > 1000 {
            let excess = opens.len() - 1000;
            opens.drain(..excess);
        }

        let mut stats = self.stats.lock();
        let entry = stats.entry(category.to_string()).or_insert((0, 0));
        entry.1 += 1;
    }

    pub fn record_sent(&self, category: &str) {
        let mut stats = self.stats.lock();
        let entry = stats.entry(category.to_string()).or_insert((0, 0));
        entry.0 += 1;
    }

    pub fn recent_opens(&self, limit: usize) -> Vec<EmailOpen> {
        let opens = self.opens.lock();
        opens.iter().rev().take(limit).cloned().collect()
    }

    pub fn open_rates(&self) -> HashMap<String, f64> {
        let stats = self.stats.lock();
        stats.iter().map(|(cat, (sent, opened))| {
            let rate = if *sent > 0 { *opened as f64 / *sent as f64 } else { 0.0 };
            (cat.clone(), rate)
        }).collect()
    }

    pub fn stats_summary(&self) -> serde_json::Value {
        let stats = self.stats.lock();
        let opens = self.opens.lock();
        let mut categories = serde_json::Map::new();
        for (cat, (sent, opened)) in stats.iter() {
            let rate = if *sent > 0 { *opened as f64 / *sent as f64 } else { 0.0 };
            categories.insert(cat.clone(), serde_json::json!({
                "sent": sent,
                "opened": opened,
                "open_rate": format!("{:.1}%", rate * 100.0)
            }));
        }
        serde_json::json!({
            "total_opens": opens.len(),
            "categories": categories
        })
    }
}

/// Resend notifier — sends emails via <https://api.resend.com/emails>
#[derive(Clone)]
pub struct ResendNotifier {
    client: reqwest::Client,
    api_key: String,
    from_email: String,
    from_name: String,
    recipient: String,
    /// Dedup: track recently welcomed emails (max 1 per 24h)
    recent_welcomes: std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<String, std::time::Instant>>>,
    /// Per-category rate limiter: category -> (last_sent, cooldown)
    rate_limits: std::sync::Arc<Mutex<HashMap<String, (Instant, Duration)>>>,
    /// Open tracking
    pub tracker: OpenTracker,
}

#[derive(Serialize)]
struct SendEmailRequest<'a> {
    from: &'a str,
    to: Vec<&'a str>,
    subject: &'a str,
    html: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reply_to: Option<&'a str>,
}

#[derive(Deserialize, Debug)]
pub struct SendEmailResponse {
    pub id: Option<String>,
}

/// Categories with their cooldown periods
fn default_cooldowns() -> HashMap<String, Duration> {
    let mut m = HashMap::new();
    m.insert("training".into(), Duration::from_secs(3600));
    m.insert("drift".into(), Duration::from_secs(21600));
    m.insert("security".into(), Duration::from_secs(0));
    m.insert("capacity".into(), Duration::from_secs(86400));
    m.insert("scheduler".into(), Duration::from_secs(0));
    m.insert("discovery".into(), Duration::from_secs(3600));
    m.insert("test".into(), Duration::from_secs(0));
    m.insert("status".into(), Duration::from_secs(300));
    m.insert("welcome".into(), Duration::from_secs(0));
    m.insert("help".into(), Duration::from_secs(60));
    m.insert("chat".into(), Duration::from_secs(10));
    m
}

// ── Shared HTML template components ──────────────────────────────────

const STYLE_CONTAINER: &str = "font-family:'SF Mono',SFMono-Regular,Menlo,monospace;background:#0a0a23;color:#e0e0ff;padding:24px;border-radius:12px;max-width:600px;";
const STYLE_HEADING: &str = "color:#4fc3f7;margin:0 0 16px 0;font-size:20px;";
const STYLE_SUBHEADING: &str = "color:#4fc3f7;margin:16px 0 8px 0;font-size:16px;";
const STYLE_TEXT: &str = "color:#c0c0ff;font-size:14px;line-height:1.6;";
const STYLE_CODE: &str = "background:#1a1a3a;color:#7fdbca;padding:2px 6px;border-radius:4px;font-size:13px;";
const STYLE_CMD: &str = "background:#1a1a3a;color:#7fdbca;padding:12px 16px;border-radius:8px;font-size:13px;margin:8px 0;display:block;";
const STYLE_FOOTER: &str = "color:#666;margin-top:20px;font-size:11px;border-top:1px solid #222;padding-top:12px;";
const STYLE_BADGE: &str = "display:inline-block;background:#1a1a3a;color:#4fc3f7;padding:2px 8px;border-radius:4px;font-size:11px;margin:2px;";

fn footer_html() -> String {
    format!(
        r#"<div style="{}">
Reply to this email to interact with the brain.
<a href="https://pi.ruv.io" style="color:#4fc3f7;">pi.ruv.io</a> | Powered by Resend
</div>"#,
        STYLE_FOOTER
    )
}

impl ResendNotifier {
    /// Create from environment variables. Returns None if RESEND_API_KEY is not set.
    pub fn from_env() -> Option<Self> {
        let api_key = std::env::var("RESEND_API_KEY").ok()?;
        if api_key.is_empty() {
            return None;
        }
        let from_email = std::env::var("BRAIN_NOTIFICATION_EMAIL")
            .unwrap_or_else(|_| "pi@ruv.io".into());
        let from_name = std::env::var("BRAIN_NOTIFICATION_NAME")
            .unwrap_or_else(|_| "Pi Brain".into());
        let recipient = std::env::var("BRAIN_NOTIFY_RECIPIENT")
            .unwrap_or_else(|_| "ruv@ruv.net".into());

        tracing::info!("Resend notifier initialized: from={}, to={}", from_email, recipient);

        Some(Self {
            client: reqwest::Client::new(),
            api_key,
            from_email,
            from_name,
            recipient,
            rate_limits: std::sync::Arc::new(Mutex::new(HashMap::new())),
            tracker: OpenTracker::new(),
            recent_welcomes: std::sync::Arc::new(parking_lot::Mutex::new(std::collections::HashMap::new())),
        })
    }

    fn check_rate_limit(&self, category: &str) -> bool {
        let cooldowns = default_cooldowns();
        let cooldown = cooldowns.get(category).copied().unwrap_or(Duration::from_secs(3600));
        if cooldown.is_zero() {
            return true;
        }
        let mut limits = self.rate_limits.lock();
        if let Some((last, _)) = limits.get(category) {
            if last.elapsed() < cooldown {
                return false;
            }
        }
        limits.insert(category.to_string(), (Instant::now(), cooldown));
        true
    }

    fn formatted_from(&self) -> String {
        format!("{} <{}>", self.from_name, self.from_email)
    }

    /// Generate a tracking pixel URL for an email
    fn tracking_pixel(&self, tracking_id: &str, category: &str) -> String {
        format!(
            r#"<img src="https://pi.ruv.io/v1/notify/pixel/{id}?c={cat}" width="1" height="1" style="display:block;width:1px;height:1px;border:0;" alt="" />"#,
            id = tracking_id,
            cat = category,
        )
    }

    /// Inject tracking pixel and unsubscribe footer into HTML
    fn inject_tracking(&self, html: &str, tracking_id: &str, category: &str) -> String {
        let pixel = self.tracking_pixel(tracking_id, category);
        let unsub = format!(
            r#"<div style="text-align:center;margin-top:12px;font-size:10px;color:#555;">
<a href="https://pi.ruv.io/v1/notify/unsubscribe?t={id}" style="color:#555;">Unsubscribe</a>
 | <a href="https://pi.ruv.io/v1/notify/preferences?t={id}" style="color:#555;">Preferences</a>
</div>"#,
            id = tracking_id,
        );
        // Insert pixel before closing </div> and add unsubscribe
        if html.ends_with("</div>") {
            format!("{}{}{}", &html[..html.len()-6], pixel, "</div>")
                + &unsub
        } else {
            format!("{}{}{}", html, pixel, unsub)
        }
    }

    /// Send an email. Respects per-category rate limits. Injects tracking pixel.
    pub async fn send(
        &self,
        category: &str,
        subject: &str,
        html: &str,
    ) -> Result<String, String> {
        if !self.check_rate_limit(category) {
            return Err(format!("rate-limited: category '{}' is in cooldown", category));
        }

        let tracking_id = uuid::Uuid::new_v4().to_string();
        let tracked_html = self.inject_tracking(html, &tracking_id, category);

        let from = self.formatted_from();
        let body = SendEmailRequest {
            from: &from,
            to: vec![&self.recipient],
            subject,
            html: &tracked_html,
            reply_to: Some(&self.from_email),
        };

        let resp = self.client
            .post("https://api.resend.com/emails")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("resend request failed: {}", e))?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();

        if status.is_success() {
            let parsed: SendEmailResponse =
                serde_json::from_str(&text).unwrap_or(SendEmailResponse { id: None });
            let id = parsed.id.unwrap_or_else(|| "unknown".into());
            self.tracker.record_sent(category);
            tracing::info!("Email sent: category={}, id={}, tracking={}", category, id, tracking_id);
            Ok(id)
        } else {
            tracing::warn!("Resend API error: status={}, body={}", status, text);
            Err(format!("resend API error {}: {}", status, text))
        }
    }

    /// Send to a specific recipient (for inbound reply flows)
    pub async fn send_to(
        &self,
        to: &str,
        category: &str,
        subject: &str,
        html: &str,
    ) -> Result<String, String> {
        if !self.check_rate_limit(category) {
            return Err(format!("rate-limited: category '{}' is in cooldown", category));
        }

        let tracking_id = uuid::Uuid::new_v4().to_string();
        let tracked_html = self.inject_tracking(html, &tracking_id, category);

        let from = self.formatted_from();
        let body = SendEmailRequest {
            from: &from,
            to: vec![to],
            subject,
            html: &tracked_html,
            reply_to: Some(&self.from_email),
        };

        let resp = self.client
            .post("https://api.resend.com/emails")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("resend request failed: {}", e))?;

        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();

        if status.is_success() {
            let parsed: SendEmailResponse =
                serde_json::from_str(&text).unwrap_or(SendEmailResponse { id: None });
            let id = parsed.id.unwrap_or_else(|| "unknown".into());
            self.tracker.record_sent(category);
            tracing::info!("Email sent to {}: category={}, id={}, tracking={}", to, category, id, tracking_id);
            Ok(id)
        } else {
            Err(format!("resend API error {}: {}", status, text))
        }
    }

    // ── Pre-built email templates ────────────────────────────────────

    /// Welcome email for new users connecting to the brain.
    /// Rate-limited: max 1 welcome per email per 24 hours.
    pub async fn send_welcome(&self, user_email: &str, user_name: Option<&str>) -> Result<String, String> {
        // Dedup: check if we already welcomed this email recently
        {
            let mut recent = self.recent_welcomes.lock();
            let now = std::time::Instant::now();
            // Clean entries older than 24h
            recent.retain(|_, t| now.duration_since(*t) < std::time::Duration::from_secs(86400));
            if recent.contains_key(user_email) {
                tracing::info!("Welcome dedup: skipping {} (already welcomed recently)", user_email);
                return Ok("dedup-skipped".to_string());
            }
            recent.insert(user_email.to_string(), now);
        }
        let name = user_name.unwrap_or("Explorer");
        let html = format!(
            r#"<div style="{container}">
<h2 style="{heading}">Welcome to the Pi Brain, {name}</h2>

<p style="{text}">
You're now connected to the <strong>shared superintelligence</strong> at
<a href="https://pi.ruv.io" style="color:#4fc3f7;">pi.ruv.io</a> —
a collective knowledge graph with <strong>2,600+ memories</strong>,
<strong>1.7M+ graph edges</strong>, and growing.
</p>

<h3 style="{subheading}">What can you do?</h3>

<p style="{text}">Reply to this email with any of these commands:</p>

<div style="{cmd}">
<strong>search</strong> &lt;query&gt;   — Search the brain's knowledge<br>
<strong>status</strong>            — Get current brain health<br>
<strong>help</strong>              — Show all available commands<br>
<strong>drift</strong>             — Check knowledge drift report<br>
<strong>share</strong>             — Contribute knowledge (body = content)
</div>

<h3 style="{subheading}">How it works</h3>

<p style="{text}">
The brain stores knowledge as <strong>RVF cognitive containers</strong> with
witness chains, SONA embeddings, and differential privacy (ε=1.0).
Every contribution is verified, PII-stripped, and Byzantine-tolerant aggregated.
</p>

<p style="{text}">
You can also interact via the REST API or MCP SSE transport:
</p>

<div style="{cmd}">
curl https://pi.ruv.io/v1/status<br>
curl "https://pi.ruv.io/v1/memories/search?q=authentication"
</div>

<h3 style="{subheading}">Quick Links</h3>

<p style="{text}">
<span style="{badge}">API</span> <a href="https://pi.ruv.io/v1/status" style="color:#4fc3f7;">pi.ruv.io/v1/status</a><br>
<span style="{badge}">Brain</span> <a href="https://pi.ruv.io" style="color:#4fc3f7;">pi.ruv.io</a><br>
<span style="{badge}">Origin</span> <a href="https://pi.ruv.io/origin" style="color:#4fc3f7;">pi.ruv.io/origin</a>
</p>

{footer}
</div>"#,
            container = STYLE_CONTAINER,
            heading = STYLE_HEADING,
            subheading = STYLE_SUBHEADING,
            text = STYLE_TEXT,
            cmd = STYLE_CMD,
            badge = STYLE_BADGE,
            footer = footer_html(),
            name = name,
        );

        self.send_to(user_email, "welcome",
            &format!("Welcome to Pi Brain, {}", name),
            &html,
        ).await
    }

    /// Help email — explains all available commands and capabilities
    pub async fn send_help(&self, to: Option<&str>) -> Result<String, String> {
        let html = format!(
            r#"<div style="{container}">
<h2 style="{heading}">Pi Brain — Command Reference</h2>

<h3 style="{subheading}">Email Commands</h3>
<p style="{text}">Reply to any pi@ruv.io email with these subjects:</p>

<table style="color:#e0e0ff;font-size:13px;width:100%;border-collapse:collapse;">
<tr style="border-bottom:1px solid #222;">
  <td style="padding:8px 12px 8px 0;"><span style="{code}">search &lt;query&gt;</span></td>
  <td style="padding:8px 0;">Semantic search across all brain knowledge</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:8px 12px 8px 0;"><span style="{code}">status</span></td>
  <td style="padding:8px 0;">Brain health: memories, edges, drift, SONA</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:8px 12px 8px 0;"><span style="{code}">drift</span></td>
  <td style="padding:8px 0;">Knowledge drift report with trend analysis</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:8px 12px 8px 0;"><span style="{code}">share</span></td>
  <td style="padding:8px 0;">Contribute knowledge (email body = content)</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:8px 12px 8px 0;"><span style="{code}">help</span></td>
  <td style="padding:8px 0;">This help message</td>
</tr>
<tr>
  <td style="padding:8px 12px 8px 0;"><span style="{code}">welcome</span></td>
  <td style="padding:8px 0;">Resend the welcome guide</td>
</tr>
</table>

<h3 style="{subheading}">REST API Endpoints</h3>
<table style="color:#e0e0ff;font-size:13px;width:100%;border-collapse:collapse;">
<tr style="border-bottom:1px solid #222;">
  <td style="padding:6px 12px 6px 0;"><span style="{code}">GET  /v1/status</span></td>
  <td style="padding:6px 0;">Brain health summary</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:6px 12px 6px 0;"><span style="{code}">GET  /v1/memories/search?q=</span></td>
  <td style="padding:6px 0;">Semantic search</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:6px 12px 6px 0;"><span style="{code}">GET  /v1/memories/list</span></td>
  <td style="padding:6px 0;">Recent memories</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:6px 12px 6px 0;"><span style="{code}">POST /v1/memories</span></td>
  <td style="padding:6px 0;">Share knowledge (authenticated)</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:6px 12px 6px 0;"><span style="{code}">GET  /v1/drift</span></td>
  <td style="padding:6px 0;">Knowledge drift analysis</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:6px 12px 6px 0;"><span style="{code}">GET  /v1/partition</span></td>
  <td style="padding:6px 0;">MinCut graph clustering</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:6px 12px 6px 0;"><span style="{code}">GET  /v1/cognitive/status</span></td>
  <td style="padding:6px 0;">Cognitive engine state</td>
</tr>
<tr>
  <td style="padding:6px 12px 6px 0;"><span style="{code}">GET  /sse</span></td>
  <td style="padding:6px 0;">MCP SSE transport</td>
</tr>
</table>

<h3 style="{subheading}">Notification Categories</h3>
<p style="{text}">The brain sends automatic alerts for:</p>
<table style="color:#e0e0ff;font-size:13px;width:100%;border-collapse:collapse;">
<tr style="border-bottom:1px solid #222;">
  <td style="padding:4px 12px 4px 0;"><span style="{badge}">training</span></td>
  <td style="padding:4px 0;">Convergence stall or loss spike (1/hour)</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:4px 12px 4px 0;"><span style="{badge}">drift</span></td>
  <td style="padding:4px 0;">Knowledge drift > 2σ (1/6h)</td>
</tr>
<tr style="border-bottom:1px solid #222;">
  <td style="padding:4px 12px 4px 0;"><span style="{badge}">security</span></td>
  <td style="padding:4px 0;">Byzantine detection, poisoning (immediate)</td>
</tr>
<tr>
  <td style="padding:4px 12px 4px 0;"><span style="{badge}">discovery</span></td>
  <td style="padding:4px 0;">High-value knowledge contributed (1/hour)</td>
</tr>
</table>

<h3 style="{subheading}">Architecture</h3>
<p style="{text}">
The brain uses RVF cognitive containers, SONA self-optimizing embeddings,
Byzantine-tolerant aggregation, differential privacy (ε=1.0), MinCut graph
partitioning, and Gemini Flash grounding for cognitive enrichment.
</p>

{footer}
</div>"#,
            container = STYLE_CONTAINER,
            heading = STYLE_HEADING,
            subheading = STYLE_SUBHEADING,
            text = STYLE_TEXT,
            code = STYLE_CODE,
            badge = STYLE_BADGE,
            footer = footer_html(),
        );

        let subject = "Pi Brain — Help & Commands";
        match to {
            Some(addr) => self.send_to(addr, "help", subject, &html).await,
            None => self.send("help", subject, &html).await,
        }
    }

    /// Brain status email with full metrics
    pub async fn send_status(
        &self,
        memories: usize,
        graph_edges: usize,
        sona_patterns: usize,
        drift: f64,
    ) -> Result<String, String> {
        let drift_indicator = if drift > 0.5 { "HIGH" } else if drift > 0.2 { "MODERATE" } else { "STABLE" };
        let drift_color = if drift > 0.5 { "#ff6b6b" } else if drift > 0.2 { "#ffd93d" } else { "#6bff6b" };

        let html = format!(
            r#"<div style="{container}">
<h2 style="{heading}">Pi Brain Status Report</h2>
<table style="color:#e0e0ff;font-size:14px;width:100%;">
<tr><td style="padding:6px 16px 6px 0;">Memories</td><td style="text-align:right;"><strong>{memories}</strong></td></tr>
<tr><td style="padding:6px 16px 6px 0;">Graph Edges</td><td style="text-align:right;"><strong>{edges}</strong></td></tr>
<tr><td style="padding:6px 16px 6px 0;">SONA Patterns</td><td style="text-align:right;"><strong>{sona}</strong></td></tr>
<tr><td style="padding:6px 16px 6px 0;">Drift</td><td style="text-align:right;"><strong style="color:{drift_color};">{drift:.4} ({drift_indicator})</strong></td></tr>
</table>
{footer}
</div>"#,
            container = STYLE_CONTAINER,
            heading = STYLE_HEADING,
            footer = footer_html(),
            memories = memories,
            edges = graph_edges,
            sona = sona_patterns,
            drift = drift,
            drift_color = drift_color,
            drift_indicator = drift_indicator,
        );

        self.send("status", "Brain Status Report", &html).await
    }

    /// Test email to verify integration
    pub async fn send_test(&self) -> Result<String, String> {
        let html = format!(
            r#"<div style="{container}">
<h2 style="{heading}">Pi Brain — Email Test</h2>
<p style="{text}">
Resend integration is working correctly.
</p>
<table style="color:#e0e0ff;font-size:13px;">
<tr><td style="padding:4px 12px 4px 0;">From</td><td><span style="{code}">pi@ruv.io</span></td></tr>
<tr><td style="padding:4px 12px 4px 0;">Service</td><td>Cloud Run (ruvbrain)</td></tr>
<tr><td style="padding:4px 12px 4px 0;">ADR</td><td>ADR-125</td></tr>
</table>
<p style="{text}">Reply with <span style="{code}">help</span> to see all available commands.</p>
{footer}
</div>"#,
            container = STYLE_CONTAINER,
            heading = STYLE_HEADING,
            text = STYLE_TEXT,
            code = STYLE_CODE,
            footer = footer_html(),
        );

        self.send("test", "Email Integration Test", &html).await
    }

    /// Send search results as a formatted email
    pub async fn send_search_results(
        &self,
        to: &str,
        query: &str,
        results: &[(String, String, f64)], // (title, content, score)
    ) -> Result<String, String> {
        let mut rows = String::new();
        for (i, (title, content, score)) in results.iter().enumerate() {
            let truncated = if content.len() > 200 { &content[..200] } else { content };
            rows.push_str(&format!(
                r#"<tr style="border-bottom:1px solid #222;">
<td style="padding:8px 0;vertical-align:top;">
<strong style="color:#4fc3f7;">{}. {}</strong><br>
<span style="color:#999;font-size:12px;">{}</span><br>
<span style="color:#666;font-size:11px;">score: {:.3}</span>
</td></tr>"#,
                i + 1, title, truncated, score
            ));
        }

        if results.is_empty() {
            rows = r#"<tr><td style="padding:12px 0;color:#888;">No results found. Try a broader query.</td></tr>"#.to_string();
        }

        let html = format!(
            r#"<div style="{container}">
<h2 style="{heading}">Search Results</h2>
<p style="{text}">Query: <span style="{code}">{query}</span> ({count} results)</p>
<table style="color:#e0e0ff;font-size:13px;width:100%;">{rows}</table>
<p style="{text}">Reply with <span style="{code}">search &lt;new query&gt;</span> to search again.</p>
{footer}
</div>"#,
            container = STYLE_CONTAINER,
            heading = STYLE_HEADING,
            text = STYLE_TEXT,
            code = STYLE_CODE,
            footer = footer_html(),
            query = query,
            count = results.len(),
            rows = rows,
        );

        self.send_to(to, "chat",
            &format!("Re: search {} ({} results)", query, results.len()),
            &html,
        ).await
    }
}
