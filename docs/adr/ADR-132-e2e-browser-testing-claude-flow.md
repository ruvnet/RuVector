# ADR-132: E2E Browser Testing with @claude-flow/browser

## Status

Proposed

## Date

2026-04-02

## Context

The `ui/ruvocal` dashboard (SvelteKit + Svelte 5) has unit and SSR tests via Vitest but lacks end-to-end browser tests that validate real user flows. The `@claude-flow/browser` skill provides AI-optimized browser automation via Playwright, enabling agents to navigate, interact, screenshot, and assert against live UI — making it ideal for E2E testing orchestrated by claude-flow swarms.

### Current Test Gap

| Layer | Coverage | Tool |
|-------|----------|------|
| Unit (client) | `*.svelte.test.ts` | Vitest + Playwright env |
| SSR | `*.ssr.test.ts` | Vitest + Node env |
| Server | `*.test.ts` / `*.spec.ts` | Vitest + Node env |
| **E2E (browser)** | **None** | **Proposed: @claude-flow/browser** |

### Key UI Routes to Cover

| Route | Purpose | Priority |
|-------|---------|----------|
| `/login` | Authentication flow | P0 |
| `/conversation/[id]` | Core chat + streaming | P0 |
| `/settings` | User preferences | P1 |
| `/admin/stats` | Admin dashboard stats | P1 |
| `/metrics` | System metrics view | P1 |
| `/models` | Model selection | P2 |
| `/r/[id]` | Shared conversation view | P2 |

## Decision

Adopt `@claude-flow/browser` as the E2E testing framework for `ui/ruvocal`, integrated with claude-flow swarm orchestration for parallel test execution.

### Architecture

```
┌─────────────────────────────────────┐
│  claude-flow swarm (hierarchical)   │
│  ┌───────────┐  ┌───────────┐      │
│  │ test-agent│  │ test-agent│ ...   │
│  │ (auth)    │  │ (chat)    │       │
│  └─────┬─────┘  └─────┬─────┘      │
│        │               │            │
│  ┌─────▼───────────────▼─────┐      │
│  │   @claude-flow/browser    │      │
│  │   (Playwright engine)     │      │
│  └─────────────┬─────────────┘      │
│                │                    │
│  ┌─────────────▼─────────────┐      │
│  │   SvelteKit dev server    │      │
│  │   localhost:5173          │      │
│  └───────────────────────────┘      │
└─────────────────────────────────────┘
```

### @claude-flow/browser Tool Reference

The browser skill exposes these MCP tools for E2E automation:

| Tool | Purpose | E2E Use |
|------|---------|---------|
| `browser_open` | Navigate to URL | Load pages under test |
| `browser_click` | Click elements | Interact with buttons, links |
| `browser_fill` | Fill form inputs | Login forms, settings, chat input |
| `browser_type` | Type text | Chat messages, search queries |
| `browser_press` | Press keys | Enter to send, Escape to close |
| `browser_snapshot` | AI-optimized DOM snapshot | Assert page state |
| `browser_screenshot` | Visual capture | Visual regression testing |
| `browser_get-text` | Extract text content | Verify rendered output |
| `browser_get-title` | Get page title | Route validation |
| `browser_get-url` | Get current URL | Navigation assertions |
| `browser_wait` | Wait for condition | Loading states, streaming |
| `browser_eval` | Run JS in page | Custom assertions, state checks |
| `browser_select` | Select dropdown option | Model selection, settings |
| `browser_scroll` | Scroll viewport | Long conversation history |
| `browser_hover` | Hover elements | Tooltip verification |
| `browser_check/uncheck` | Toggle checkboxes | Settings toggles |
| `browser_back/forward` | Navigation history | Back/forward flow |
| `browser_reload` | Reload page | State persistence checks |
| `browser_close` | Close browser | Cleanup |
| `browser_session-list` | List active sessions | Multi-tab testing |

### E2E Test Patterns

#### Pattern 1: Authentication Flow

```
1. browser_open → http://localhost:5173/login
2. browser_snapshot → verify login form rendered
3. browser_fill → username/password fields
4. browser_click → submit button
5. browser_wait → redirect to /conversation
6. browser_get-url → assert URL changed
7. browser_snapshot → verify authenticated state
```

#### Pattern 2: Chat Conversation

```
1. browser_open → http://localhost:5173/conversation/[id]
2. browser_snapshot → verify chat UI loaded
3. browser_fill → message input
4. browser_press → Enter
5. browser_wait → streaming response appears
6. browser_get-text → verify assistant response
7. browser_screenshot → capture conversation state
```

#### Pattern 3: Settings Management

```
1. browser_open → http://localhost:5173/settings
2. browser_snapshot → verify settings page
3. browser_select → change model preference
4. browser_check → toggle feature flag
5. browser_click → save button
6. browser_reload → verify persistence
7. browser_snapshot → assert settings retained
```

#### Pattern 4: Admin Dashboard

```
1. browser_open → http://localhost:5173/admin/stats
2. browser_wait → stats data loaded
3. browser_snapshot → verify dashboard components
4. browser_get-text → extract metric values
5. browser_eval → assert metric ranges
6. browser_screenshot → visual baseline
```

### Swarm-Based Parallel Execution

```bash
# Initialize test swarm
npx @claude-flow/cli@latest swarm init \
  --topology hierarchical \
  --max-agents 6 \
  --strategy specialized

# Spawn parallel test agents
# Agent 1: Auth tests
# Agent 2: Chat flow tests
# Agent 3: Settings tests
# Agent 4: Admin dashboard tests
# Agent 5: Model selection tests
# Agent 6: Shared conversation tests
```

Each agent uses `@claude-flow/browser` independently with isolated browser sessions, enabling full parallel execution.

### Test File Organization

```
tests/
└── e2e/
    ├── auth.e2e.ts           # Login/logout flows
    ├── conversation.e2e.ts   # Chat and streaming
    ├── settings.e2e.ts       # User preferences
    ├── admin.e2e.ts          # Admin dashboard
    ├── models.e2e.ts         # Model selection
    ├── shared.e2e.ts         # Shared conversation views
    ├── fixtures/
    │   ├── test-users.ts     # Test credentials
    │   └── test-data.ts      # Seed data
    └── helpers/
        ├── browser.ts        # Browser helper wrappers
        └── assertions.ts     # Custom assertion utilities
```

### CI Integration

E2E tests run as a GitHub Actions workflow:

1. Start SvelteKit dev server (`npm run dev`)
2. Initialize claude-flow swarm
3. Spawn browser test agents in parallel
4. Collect results and screenshots
5. Fail pipeline on assertion failures
6. Archive screenshots as artifacts

## Consequences

### Positive

- Real browser coverage for all critical user flows
- Parallel execution via swarm reduces total test time
- AI-optimized snapshots enable intelligent assertions (not just CSS selectors)
- Visual regression detection via screenshots
- Reuses existing claude-flow infrastructure

### Negative

- Browser tests are inherently slower than unit tests
- Requires running dev server during CI
- Playwright dependency adds ~100MB to CI image
- Flaky test risk with streaming/async UI states

### Mitigations

- Use `browser_wait` with explicit conditions to reduce flakiness
- Run E2E only on PR merges to main (not every push)
- Implement retry logic for network-dependent tests
- Use `browser_eval` for deterministic state checks over visual assertions

## References

- [claude-flow browser skill](/browser)
- [SvelteKit testing docs](https://kit.svelte.dev/docs/testing)
- [Playwright documentation](https://playwright.dev/)
- [ADR-089: CNN Browser Demo](./ADR-089-cnn-browser-demo.md)
