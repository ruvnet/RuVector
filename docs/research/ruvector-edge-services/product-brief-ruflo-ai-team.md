# Product brief: RuFlo AI Team and MCP Launch Doctor

Status: input to ADR-351 (downstream products that consume ruvector edge services)
Source: product research supplied by rUv, 2026-09-29. Marketplace figures were
observed by the author in the live Claude directory and are not independently
re-verified here.

## Thesis

The strongest opportunity is not another generic connector. It is a packaged
"AI team" product built on RuFlo Federation, Cognitum, and ruOS.

> Give Claude a team, a shared control room, and a computer when the work requires one.

## Marketplace observations (author-reported)

- The directory showed 3,841 connectors. Discovery is dominated by systems of
  record: Google Drive, Gmail, Calendar, Microsoft 365, Notion, Figma, Slack,
  Atlassian, HubSpot, Asana and Linear.
- Popular plugins include Notion, Canva, Linear, Carta and Anthropic's
  vertical workflow bundles. Trending: financial-advisor platforms, legal
  research, Amazon selling, jobs and consumer services.
- Plugins (skills + MCP services + interactive MCP Apps) are now the main
  third-party extension format:
  https://claude.com/blog/build-plugins-for-claude
- MCP Apps render dashboards, forms and live interfaces inline:
  https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/
- Search gaps (semantic matching, so counts are directional, medium confidence):
  "AI chief of staff" 0; "business operating system" 7; "agent cost
  tracking" 10; "fleet" 24; "agent evaluation" 26 (Langfuse, DeepEval,
  MLflow, Coval); "multi-agent team" 33, few providing a visible,
  general-purpose team.

## Flagship: RuFlo AI Team

A user describes an outcome. The plugin creates a small specialist team,
assigns work, prevents duplicate effort, tracks budget and risk, and shows
progress in an interactive control room. It sells an outcome, not
"multi-agent infrastructure".

| Layer | Provides |
|---|---|
| RuFlo | agents, claims, messaging, workflows, signed evidence |
| Cognitum | durable memory, evaluation, cost accounting, policy, security events |
| ruOS | isolated desktop for browser/GUI work |
| MCP App | live Kanban, agent activity, approvals, evidence, spend |
| Agent Skills | research, release preparation, security review, business ops |

Initial experience:

1. "Research this market and produce a launch plan."
2. Claude proposes a three-agent team and a declared budget.
3. User approves.
4. Inline control room shows assignments, progress, cost and evidence.
5. Agents request explicit approval before external writes or desktop actions.
6. The run produces a signed, shareable result page and a reusable template
   (the viral loop: recipients inspect evidence, fork, start their own team).

## Opportunity ranking

| Product | Unique | Fast to ship | Viral | Review fit | Recommendation |
|---|---:|---:|---:|---:|---|
| RuFlo AI Team | 5 | 4 | 5 | 4 | Flagship |
| MCP Launch Doctor | 5 | 5 | 4 | 5 | Fastest launch |
| Agent Flight Recorder | 4 | 4 | 4 | 4 | Enterprise add-on |
| Business Control Room | 4 | 3 | 5 | 3 | Expand later |
| Cognitum Fleet Doctor | 3 | 4 | 2 | 4 | Strong vertical |

## Fast wedge: MCP Launch Doctor

> Turn an API or MCP repository into a secure, directory-ready Claude plugin.

- Audit tool names, descriptions and annotations.
- Validate OAuth and RFC 9728 discovery.
- Detect secret-bearing schemas and prompt-injection risks.
- Generate review tests, listing copy, icons, legal checklists, submission bundles.
- Produce a shareable readiness scorecard.

Builds largely on existing assets: `chatgpt-mcp-studio-site`, the security
harness, submission skills, and the RuFlo review work. No clear marketplace
leader exists for end-to-end MCP directory validation, and the directory
policy (accurate behavior, helpful errors, frugal outputs, OAuth, explicit
annotations) makes it timely:
https://support.claude.com/en/articles/13145358-anthropic-software-directory-policy

## Freemium model (RuFlo AI Team)

Free: one team at a time, up to three agents, one metered ruOS machine, a
monthly allowance of agent/model/desktop units, public or private signed
result pages, no card required.

Paid: larger teams and concurrent runs, persistent Cognitum memory, scheduled
operations, private templates and longer retention, higher budgets,
evaluations, policy gates, audit exports.

Charge against unified "work units" (model tokens, machine time, storage,
external API cost). Set the free allowance from measured p50/p95 run cost,
not an arbitrary number.

## Approval-safe design

- External writes are approval-gated.
- Never transfer money or financial assets.
- No advertising or sponsored results.
- Narrow, factual tools; first-party RuFlo/Cognitum/ruOS APIs.
- Fence third-party and agent-generated content as untrusted data.
- Never read Claude memory or unrelated conversation data.
- Explicit boolean tool annotations and accurate OAuth challenges.
- Public sharing is opt-in with a redaction preview.
- The free machine is isolated, disposable and resource-capped.

## Recommended sequence

1. Package the existing RuFlo connector with 4-6 team-workflow skills.
2. Add a cross-platform MCP App control-room UI.
3. Add Cognitum cost, evaluation and security receipts.
4. Attach the metered free ruOS desktop only when a task needs a computer.
5. Add signed, redacted, forkable result pages.
6. Beta-test three workflows: market research, release preparation, security audit.
7. Submit as "RuFlo AI Team", not as infrastructure.

Evidence quality: marketplace composition and policies high confidence;
search-gap counts medium; virality is a product hypothesis.

## How ruvector edge services support this (to be reflected in ADR-351)

| Product need | Edge service |
|---|---|
| Accurate OAuth + RFC 9728 challenges for every MCP surface | `ruvector-edge-auth` (resource-bound tokens, AS + PR metadata) |
| Launch Doctor's OAuth/9728 validator | reuses the edge-auth conformance tests as a checker |
| Durable team memory, recall, evidence search | vector / quantized collections, one DO per tenant collection |
| Signed, shareable, forkable result pages | RVF export with witness chain, served read-only per tenant |
| Work-unit metering and free-tier caps | per-tenant quota and usage counters in the edge core |
| Control-room MCP App | MCP endpoint behind edge-auth, tenant-scoped |
