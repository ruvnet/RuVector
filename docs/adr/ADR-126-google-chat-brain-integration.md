# ADR-126: Google Chat Bot for Pi Brain Interaction

**Status**: Proposed
**Date**: 2026-03-24
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud), ADR-064 (Pi Brain Infrastructure), ADR-125 (Resend Email Integration)

## 1. Context

The pi.ruv.io brain currently supports interaction via REST API, MCP SSE transport, and email (ADR-125). Adding Google Chat (Spaces) as an interaction channel brings the brain directly into team workflows — developers can search knowledge, check status, and receive discovery alerts without leaving their chat client.

Google Workspace Marketplace apps can be distributed to organizations, making the brain accessible to entire teams via a simple install. The Chat API supports both direct messages and Space-based collaboration where multiple team members interact with the brain simultaneously.

## 2. Decision

Deploy a Google Chat app ("Pi Brain") that routes messages to the `POST /v1/chat/google` endpoint on the brain server. The app uses HTTP endpoint mode (not Pub/Sub) for simplicity and lowest latency.

## 3. Architecture

### 3.1 Components

| Component | Details |
|-----------|---------|
| **Chat App** | Google Chat HTTP endpoint app |
| **Service Account** | `pi-brain-chat@ruv-dev.iam.gserviceaccount.com` |
| **Webhook URL** | `https://pi.ruv.io/v1/chat/google` |
| **Project** | `ruv-dev` (us-central1) |
| **Response Format** | Cards V2 (rich cards with headers, sections, key-value widgets) |

### 3.2 Message Flow

```
User (Google Chat) → Google Chat API → POST https://pi.ruv.io/v1/chat/google
                                                    ↓
                                           Brain Server (axum)
                                                    ↓
                                      Parse command → Execute → Format Card
                                                    ↓
                                           JSON response → Chat API → User
```

### 3.3 Supported Commands

| Command | Description | Response |
|---------|-------------|----------|
| `search <query>` | Semantic search across brain knowledge | Ranked results card |
| `status` | Brain health metrics | Key-value card |
| `drift` | Knowledge drift analysis | Drift report card |
| `recent` | Latest discoveries | Recent memories card |
| `help` | Command reference | Help card |
| *(free text)* | Auto-search fallback | Search results or help |

### 3.4 Event Types

| Event | Handler |
|-------|---------|
| `ADDED_TO_SPACE` | Welcome card with stats + commands |
| `REMOVED_FROM_SPACE` | No-op (silent) |
| `MESSAGE` | Parse command, execute, respond with card |

### 3.5 Response Format

All responses use Google Chat Cards V2 with:
- **Header**: Pi Brain logo, title, subtitle
- **Sections**: Text paragraphs or key-value decorated text
- **Links**: pi.ruv.io, API status, origin story

## 4. Google Workspace Marketplace Configuration

### 4.1 App Configuration (via Chat API)

```yaml
name: "Pi Brain"
description: "Shared superintelligence — search knowledge, check drift, get discoveries"
avatar_url: "https://pi.ruv.io/og-image.svg"
endpoint_url: "https://pi.ruv.io/v1/chat/google"
authentication: "HTTP_ENDPOINT"
visibility: "EXTERNAL" (or INTERNAL for org-only)
slash_commands:
  - /search: Semantic search
  - /status: Brain health
  - /drift: Knowledge drift
  - /recent: Latest discoveries
```

### 4.2 Required OAuth Scopes

- `https://www.googleapis.com/auth/chat.bot` — Bot messaging

### 4.3 Marketplace Listing

| Field | Value |
|-------|-------|
| App Name | Pi Brain |
| Category | Productivity / Developer Tools |
| Description | Shared superintelligence for your team. Search 2,600+ knowledge memories, monitor brain health, and get daily discovery digests — all from Google Chat. |
| Support URL | https://pi.ruv.io |
| Privacy Policy | https://pi.ruv.io/origin |
| Icon | https://pi.ruv.io/og-image.svg |

## 5. Deployment Steps

1. [x] Enable Chat API on `ruv-dev` project
2. [x] Create service account `pi-brain-chat`
3. [x] Implement `POST /v1/chat/google` handler
4. [ ] Configure Chat app in Google Cloud Console → APIs → Google Chat API → Configuration
5. [ ] Set HTTP endpoint to `https://pi.ruv.io/v1/chat/google`
6. [ ] Add slash commands
7. [ ] Test in direct message
8. [ ] Publish to Google Workspace Marketplace (optional)

## 6. Security

- Google Chat verifies the bot's identity via the service account
- The `/v1/chat/google` endpoint validates the Bearer token from Google Chat
- No user credentials are stored — the brain uses pseudonymous contributor IDs
- All brain content is already PII-stripped (ε=1.0 differential privacy)

## 7. Cost

| Item | Estimate |
|------|----------|
| Google Chat API | Free (included with Workspace) |
| Cloud Run | No additional cost (existing service) |
| Service Account | Free |

## 8. Success Criteria

- [x] Chat API enabled on ruv-dev
- [x] Service account created
- [x] Handler implemented with Cards V2 responses
- [ ] Chat app configured and testable in DM
- [ ] Slash commands registered
- [ ] Search returns relevant results within 2s
- [ ] Status/drift/recent commands return accurate data
