# ADR-151: Kalshi Integration via RuVector Neural Trader

## Status

Proposed

## Date

2026-04-20

## Context

[Kalshi](https://kalshi.com) is a CFTC-regulated US event-contract exchange exposing a signed REST + WebSocket API at `https://trading-api.kalshi.com/trade-api/v2`. Contracts are binary (YES/NO) with prices in cents (0–100), settling to $1 on resolution. This is a natural target for the **RuVector Neural Trader** stack, which already models probabilistic event markets in `examples/neural-trader/specialized/prediction-markets.js` (Polymarket/Kalshi/PredictIt) but lacks a live Rust execution path against the real Kalshi API.

Existing building blocks in the ruvector workspace we will reuse verbatim:

| Crate / Module | Role | Reuse |
|---|---|---|
| `crates/neural-trader-core` | Canonical `MarketEvent`, `EventType`, `Side`, ingest traits | Normalize Kalshi ticker/trade/orderbook into `MarketEvent` |
| `crates/neural-trader-coherence` | Coherence / regime detection across markets | Cross-market arbitrage and regime-gated sizing |
| `crates/neural-trader-replay` | Deterministic replay for backtests | Replay captured Kalshi streams against strategies |
| `crates/ruvector-attention` / `ruvector-attn-mincut` | Attention + MinCut for market graph clustering | Group correlated Kalshi series (e.g., all Fed-rate strikes) |
| `crates/ruvector-cnn` | Temporal convnet over order-book imbalance | Short-horizon tick prediction on Kalshi trades feed |
| `crates/ruvllm` / RuvLtra | Local inference for news/event summarization → probability prior | Provide `modelProbability` input analogous to the JS example |
| `crates/mcp-brain-server` + pi.ruv.io brain | Persistent knowledge store for contracts, resolutions, past edges | Long-memory of market behavior and strategy outcomes |

Kalshi authentication is **RSA-PSS-SHA256** signed requests, *not* HMAC. Each request requires:

```
KALSHI-ACCESS-KEY:       <API key UUID>
KALSHI-ACCESS-TIMESTAMP: <unix ms>
KALSHI-ACCESS-SIGNATURE: base64( RSA-PSS-SHA256( PEM_PRIVATE_KEY, timestamp + method + path ) )
```

Because of this, secrets must be held server-side in GCS and never serialized into logs, traces, or the brain.

### Credential state (already provisioned)

Created in Google Cloud project `ruv-dev` (ADR-150 context):

| Secret | Source | Access |
|---|---|---|
| `KALSHI_API_KEY` | UUID, 16 bytes | `gcloud secrets versions access latest --secret=KALSHI_API_KEY --project=ruv-dev` |
| `KALSHI_PRIVATE_KEY_PEM` | RSA private key (1679 bytes) | same pattern |
| `KALSHI_API_URL` | `https://trading-api.kalshi.com/trade-api/v2` | same pattern |

Local development copy is at `/home/ruvultra/projects/ruvector/.kalshi/kalshi.pem`. Both `.kalshi/` and `*.pem` are already in `.gitignore`; no key material is ever committed.

## Decision

Introduce a new Rust crate **`ruvector-kalshi`** that (a) authenticates against the live Kalshi API using RSA-PSS-SHA256, (b) normalizes Kalshi events into `neural_trader_core::MarketEvent`, and (c) exposes a strategy runtime that composes attention/mincut/cnn signals with a ruvllm-supplied prior to quote and execute on binary contracts — with pi.ruv.io brain as the long-term memory.

No Python. No JS shelling. All three layers (auth, stream, strategy) are pure Rust; WASM exposure is optional via a thin `ruvector-kalshi-wasm` sibling for dashboards.

### Workspace layout

```
crates/
├── ruvector-kalshi/              # new — auth, REST, WS, secrets
│   ├── src/
│   │   ├── lib.rs
│   │   ├── auth.rs               # RSA-PSS signer
│   │   ├── rest.rs               # GET/POST wrappers
│   │   ├── ws.rs                 # tokio-tungstenite feed
│   │   ├── models.rs             # Kalshi DTOs (Market, Event, Order, Fill)
│   │   ├── normalize.rs          # Kalshi → neural_trader_core::MarketEvent
│   │   └── secrets.rs            # gcloud + local-PEM loader
│   └── Cargo.toml
├── ruvector-kalshi-wasm/         # optional — browser-safe subset (read-only)
└── neural-trader-strategies/     # new — reusable strategy trait + Kalshi strats
    └── src/
        ├── lib.rs                # Strategy trait, PositionSizer, RiskGate
        ├── expected_value.rs     # EV+Kelly fractional sizing
        ├── coherence_arb.rs      # cross-market arb via neural-trader-coherence
        └── attn_scalper.rs       # attention+cnn short-horizon scalper

examples/
└── kalshi/                       # new — runnable demos
    ├── list-markets.rs
    ├── stream-orderbook.rs
    ├── paper-trade.rs
    └── live-trade.rs             # gated behind KALSHI_ENABLE_LIVE=1
```

### Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     ruvector-kalshi (Rust)                          │
│                                                                     │
│  secrets::load() ──► gcloud secrets versions access (cached 5min)   │
│       │                                                             │
│       ▼                                                             │
│  auth::Signer  { api_key, rsa_private_key }                         │
│       │ sign(ts, method, path) → KALSHI-ACCESS-SIGNATURE            │
│       ▼                                                             │
│  rest::Client  ──► GET  /markets, /portfolio, /events               │
│                    POST /orders                                     │
│  ws::Feed      ──► wss: ticker/orderbook/trade/fill                 │
│       │                                                             │
│       ▼                                                             │
│  normalize::to_market_event()  ──► neural_trader_core::MarketEvent  │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                   neural-trader-strategies                          │
│                                                                     │
│  ┌─────────────┐   ┌────────────────┐   ┌──────────────┐            │
│  │ EV + Kelly  │   │ Coherence Arb  │   │ Attn Scalper │            │
│  │ (prior from │   │ (mincut groups │   │ (cnn over    │            │
│  │  ruvllm)    │◄──┤  via attention)│◄──┤  imbalance)  │            │
│  └─────┬───────┘   └───────┬────────┘   └──────┬───────┘            │
│        │                   │                   │                    │
│        └──► RiskGate ◄─────┴───────────────────┘                    │
│             (position cap, daily-loss kill, vol regime)             │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Execution + Memory                              │
│                                                                     │
│  Paper mode → neural-trader-replay ledger                           │
│  Live mode  → ruvector-kalshi::rest::post_order                     │
│                                                                     │
│  pi.ruv.io brain: store contract outcomes, strategy P&L, market     │
│  resolutions as MarketEvent-derived memories (category: pattern).   │
└────────────────────────────────────────────────────────────────────┘
```

### Authentication detail

```rust
// crates/ruvector-kalshi/src/auth.rs (sketch)
use rsa::{RsaPrivateKey, pkcs1::DecodeRsaPrivateKey,
          pss::SigningKey, signature::RandomizedSigner};
use sha2::Sha256;
use base64::{engine::general_purpose::STANDARD, Engine};

pub struct Signer {
    api_key: String,
    signing_key: SigningKey<Sha256>,
}

impl Signer {
    pub fn sign(&self, ts_ms: u64, method: &str, path: &str) -> (String, String, String) {
        let msg = format!("{ts_ms}{method}{path}");
        let mut rng = rand::thread_rng();
        let sig = self.signing_key.sign_with_rng(&mut rng, msg.as_bytes());
        let sig_b64 = STANDARD.encode(sig.to_bytes());
        (self.api_key.clone(), ts_ms.to_string(), sig_b64)
    }
}
```

Dependencies: `rsa = "0.9"`, `sha2 = "0.10"`, `base64 = "0.22"`, `reqwest = { version = "0.12", features = ["json", "rustls-tls"] }`, `tokio-tungstenite`, `tokio`.

### Secret loading (no hardcoding, no .env commits)

```rust
// crates/ruvector-kalshi/src/secrets.rs (sketch)
pub enum SecretSource { Gcloud, LocalPem(PathBuf), Env }

pub async fn load() -> Result<Credentials> {
    match env::var("KALSHI_SECRET_SOURCE").as_deref() {
        Ok("local") => load_local(),                           // dev only
        _           => load_gcloud("ruv-dev").await,           // default
    }
}
```

`load_gcloud` shells out to `gcloud secrets versions access latest --secret=...` with a 5-minute in-process cache. Production deploys will use Workload Identity on Cloud Run so no gcloud CLI is needed — Secret Manager SDK direct read.

### Risk gate (cheap, mandatory)

The `RiskGate` wraps every strategy and enforces, before any `post_order`:

1. **Max single-position notional**: default 10% of cash.
2. **Daily loss kill**: hard stop at −3% of starting balance (configurable).
3. **Concentration cap**: ≤ 40% across one neural-trader-coherence cluster.
4. **Min edge**: ≥ 300 bps over mid after fees (Kalshi takes 2%).
5. **Live gate**: `KALSHI_ENABLE_LIVE=1` required; absent → paper ledger only.

## Consequences

### Positive

- First production path from ruvector neural capabilities to a regulated live venue.
- Reuses existing Rust crates — no duplication, no Python sidecars.
- Kalshi events land in the canonical `MarketEvent` shape so they immediately flow through existing coherence, replay, and attention modules.
- Secrets are centrally managed in GCS; no key material in the repo, logs, or the brain.
- pi.ruv.io accumulates a long-memory of event-market behavior, feeding back into future strategies (ADR-149/150 synergy).

### Negative / risks

- **Regulatory**: Kalshi is US-only; strategy code must remain paper-only unless the operator has verified a Kalshi account. Live trading is gated behind an env flag and manual credential load.
- **Key custody**: RSA private key in GCS means any party with `ruv-dev` Secret Accessor can trade. Mitigation: restrict IAM to a single service account; rotate PEM quarterly; consider KMS-wrapped storage later.
- **Latency**: gcloud CLI shell-out is ~200 ms — acceptable at startup, unacceptable per-request. We cache and never re-read inside the hot path.
- **Rate limits**: Kalshi enforces per-endpoint limits. The `rest::Client` must include a token-bucket limiter. Replay tests catch 429 regressions.
- **Model drift**: ruvllm priors can degrade silently. The `neural-trader-replay` suite must run nightly in CI with the previous week's captured feed and compare P&L against a baseline.

### Neutral

- Adds ~2 crates to the workspace — consistent with existing `neural-trader-*` pattern.
- Optional wasm sibling is not blocking; it can come later when a dashboard needs read-only market access.

## Implementation Plan

**Phase 1 — Auth & REST (week 1)**
- `ruvector-kalshi` crate skeleton
- `auth::Signer` + round-trip test against Kalshi `/exchange/status`
- `rest::Client` with GET `/markets`, `/events`, `/portfolio`
- `examples/kalshi/list-markets.rs`

**Phase 2 — Stream & normalize (week 2)**
- `ws::Feed` subscribing to ticker, trade, orderbook channels
- `normalize::to_market_event` with unit tests covering all Kalshi event kinds
- `examples/kalshi/stream-orderbook.rs` dumping `MarketEvent` JSONL

**Phase 3 — Strategy runtime & paper trade (week 3)**
- `neural-trader-strategies` crate with `Strategy` trait + `RiskGate`
- `ExpectedValueKelly` strategy wired to ruvllm prior
- `CoherenceArb` strategy wired to `neural-trader-coherence`
- `AttentionScalper` wired to `ruvector-cnn`
- `examples/kalshi/paper-trade.rs` replays captured feed → ledger

**Phase 4 — Live & memory (week 4)**
- `rest::post_order`, cancel, amend
- Brain integration: every resolution and P&L event → `brain_share`
- `examples/kalshi/live-trade.rs` behind `KALSHI_ENABLE_LIVE=1`
- Nightly replay benchmark in CI

## Testing Strategy

- **Unit**: auth signature matches Kalshi reference vectors; normalize round-trips against fixture feeds.
- **Integration**: hit Kalshi `/exchange/status` in a gated CI job (needs a sandbox API key).
- **Replay**: `neural-trader-replay` replays captured `.jsonl` against each strategy; asserts P&L envelope.
- **Chaos**: drop WS connection, 429 storms, expired timestamps — all must fail closed (no orders).
- **Security**: `aidefence_scan` over any fields we send to the brain to strip accidental PII.

## Alternatives Considered

1. **Extend the JS example directly** — rejected; CLAUDE.md mandates Rust for all ruOS components, and we lose access to the Rust neural-trader crates.
2. **Generic "exchange" crate with Kalshi as a driver** — deferred; premature abstraction. Ship Kalshi first, extract the trait once a second venue (Polymarket) is scoped.
3. **Store PEM in the repo encrypted with age/sops** — rejected; GCS Secret Manager is the existing pattern (ADR-150, `CRATES_API_KEY`, etc.) and plays well with Cloud Run Workload Identity.

## References

- ADR-084: Neural Trader foundation (referenced by `neural-trader-core::lib.rs`)
- ADR-149: Brain performance optimizations
- ADR-150: π Brain + RuvLtra via Tailscale
- Kalshi API docs: https://trading-api.readme.io/reference/getting-started
- Existing example: `examples/neural-trader/specialized/prediction-markets.js`
