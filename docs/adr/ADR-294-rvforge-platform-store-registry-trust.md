# ADR-294: RVForge Platform — Agent Store, Registry, and Trust System

- **Status**: Accepted
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — implementation in progress: crates/rvforge-registry landed (content addressing, release rules, trust-raise enforcement, non-destructive revocation, transparency log — 67 tests); publisher CLI verbs in flight; Store web UI and hosted registry pending.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-283, ADR-284, ADR-286, ADR-288, ADR-290
- **Tags**: rvforge, marketplace, registry, trust, licensing, revocation, enterprise

## Context

ADR-283 established how one canonical `.rvf` becomes signed installers for
Windows, macOS, Linux, and RVM. That pipeline solves packaging. It does not
solve distribution. A publisher who can produce a signed `.dmg` still has no
way for a user to discover the agent, understand what it will do to their
machine, pay for it, receive updates without silently gaining new
permissions, or learn that the build they installed last month has since been
revoked. Packaging is the first half of shipping; discovery, trust,
licensing, updates, and governance are the half that makes it a product.

RVForge is closer to Steam plus npm plus an enterprise application catalog
than to the Apple App Store. It distributes governed intelligence — code,
model, memory, policy, runtime, evaluation, identity, lineage, and witness —
rather than distributing applications. That framing matters because it sets
the reach: the model works on Windows, macOS, Linux, browsers with
restrictions, and native RVM systems, but it cannot fully replace Apple's
iPhone and iPad App Store, because Apple's review guideline 2.5.2 restricts
downloaded executable functionality
(https://developer.apple.com/app-store/review/guidelines/).

The primary failure mode is not a technical one. It is becoming another
untrusted agent marketplace filled with unverifiable wrappers and exaggerated
claims. Every credible defense already exists as an optional badge somewhere
in the industry, and optional badges have not worked. The fix is to make
capability disclosure, reproducible evaluations, publisher identity, and
witnessed execution **mandatory** — properties of every listing rather than
achievements a publisher may opt into.

## Decision

Build **RVForge** as an independent agentic application store, runtime,
package registry, and trust system layered on RVF and RVM.

```text
Developers publish intelligence
Users install agents
RVM quarantines execution
RVF preserves identity and state
RVForge governs trust, licensing, and updates
```

### 1. Five products

1. **RVForge Store** — public marketplace for discovering, purchasing, and
   installing agents.
2. **RVForge Reader** — desktop application that installs and runs RVFs
   through WASM, operating system isolation, or RVM.
3. **RVForge Publisher** — web console and npm CLI for building, testing,
   signing, and publishing RVFs.
4. **RVForge Registry** — content-addressed package registry holding
   releases, manifests, signatures, evaluations, and provenance.
5. **RVForge Enterprise** — private agent stores with organizational
   approval, policy enforcement, deployment, and audit controls.

### 2. Distribution model: install once, distribute RVFs

A user installs RVForge one time through a signed `.exe`, `.dmg`, `.deb`,
`.rpm`, or `.AppImage`. From then on, agents are distributed directly as
signed `.rvf` files.

```text
Install RVForge → Browse agents → Review capabilities →
Install signed RVF → Run inside quarantine → Store encrypted state locally
```

RVForge does not require a new operating system installer for every agent.
The per-agent installer pipeline of ADR-283 remains available and is the
right tool for branded consumer and partner applications — it is no longer
the only path to a user.

### 3. Core marketplace objects

```text
Publisher · Organization · RVF Package · Release · Capability Manifest
Runtime Profile · Model Manifest · Evaluation Report · Security Report
License · Entitlement · Installation · State Capsule · Update
Witness Receipt · Revocation
```

Every release is immutable. A new version creates a new signed release linked
to its predecessor, so the release history of a package is an append-only
chain rather than a mutable pointer. This is what makes "the build you
installed" a stable, auditable referent for revocation and forensics.

### 4. Publisher CLI

The publisher CLI package is **`@ruvector/rvforge`**, Node.js 20 or later.

```bash
npx @ruvector/rvforge init
npx @ruvector/rvforge pack agent.rvf
npx @ruvector/rvforge test agent.rvf
npx @ruvector/rvforge publish agent.rvf
```

**Naming supersession.** Earlier drafts of ADR-283 §4 named the CLI package
`@ruvector/forge`. That name is superseded by `@ruvector/rvforge`, which is
the single publisher entry point for both the installer pipeline and the
marketplace. This ADR records the change so the drift between the two
documents is explicit rather than discovered at implementation time.

`init` collects agent name; description and category; icon and screenshots;
pricing model; support information; privacy policy; runtime requirements; and
publisher identity.

`pack` validates:

```text
RVF structure · Publisher signature · Executable segments
Model provenance · Capability policy · Runtime compatibility
Memory requirements · External services · Software inventory
License compatibility
```

`test` runs clean installation; deterministic evaluations; capability
denials; network monitoring; filesystem escape attempts; resource exhaustion;
malformed inputs; state checkpoint and recovery; update and rollback; and
witness verification.

`publish` shows the publisher the exact profile the store will display:

```text
Validation passed
Security profile: restricted
Evaluation score: 94 percent
Supported runtimes: WASM and RVM
Supported systems: Windows, macOS, Linux
Requested capabilities: selected files
Network access: none
Ready to publish
```

### 5. Store, library, and runtime surfaces

Store home sections: Featured Agents · Verified Publishers · Runs Entirely
Locally · Enterprise Ready · Spatial Intelligence · Developer Tools ·
Healthcare · Audio Intelligence · Recently Updated · Free and Open Source.
Search filters cover category, price, publisher, local or cloud model,
runtime, operating system, capability level, open source, enterprise
approved, offline support, and evaluation score.

Every listing displays agent name, publisher identity, version, price,
purpose, screenshots, runtime requirements, model location, data handling,
capabilities, evaluation results, security findings, software inventory,
release history, and user reviews. Primary actions are Install · Try in
Temporary Session · Review Capabilities · View Source · Purchase · Add to
Organization.

Library states are Installed · Running · Paused · Updates · Quarantined ·
Organization Managed · Archived, and each installed agent exposes Open ·
Pause · Terminate · Clone · Reset · Export State · Import State · Review
Activity · Change Permissions · Verify · Uninstall.

While an agent runs, RVForge displays current task, runtime type, model
activity, CPU usage, memory usage, network connections, filesystem access,
tool calls, recent actions, witness status, and estimated execution cost.
Emergency controls stay visible at all times: Pause · Terminate · Disconnect
Network · Revoke Capabilities · Rollback State.

### 6. Capability contract as the installation UX

Before installation, RVForge presents the exact capability contract in both
directions — what the agent requests and what it provably cannot do:

```text
This agent requests:
  Selected document access · 512 MB memory · Local model execution
  Encrypted persistent state

This agent cannot:
  Access the internet · Read other folders · Use the microphone
  Run background processes · Contact external model providers
```

Actions are Install · Customize Permissions · Cancel. **Broad permission
descriptions such as "access your computer" are prohibited.** A capability
that cannot be described concretely cannot be requested.

Updates carry a semantic permission difference:

```text
Version 1.3 changes:
  Adds PDF processing
  Requests access to selected folders
  Introduces optional OpenAI connectivity
  Changes memory schema from 2 to 3
```

Users must approve any capability expansion. Updates containing only code
fixes within the existing contract may follow organizational update policy.
Rollback remains available until state migration makes rollback unsafe, and
that condition must be declared before installation — never discovered after
an update has already migrated the user's state (ADR-288).

### 7. Trust levels are evidence, not endorsement

1. **Published** — identity verified and package structurally valid.
2. **Tested** — automated runtime, security, and evaluation tests passed.
3. **Reviewed** — human security and capability review completed.
4. **Enterprise Approved** — approved by the customer's security or
   governance team.

Each level names the evidence that was gathered. No level asserts that
software is universally safe, and store copy must not imply otherwise.

### 8. Review pipeline

```text
Upload → Signature verification → Static inspection →
Malware and dependency scanning → Quarantined execution →
Capability testing → Behavioral evaluation → Publisher review →
Publish or reject
```

Manual review is **required** when an agent requests any of: unrestricted
filesystem access; arbitrary network access; process creation; native code;
credentials; background execution; financial transactions; health decisions;
physical device control; inter-agent delegation.

### 9. Security model

1. The publisher signs the RVF.
2. RVForge verifies and countersigns the release record.
3. RVM verifies both signatures before execution (ADR-284, ADR-290).
4. Every capability defaults to denied (ADR-286).
5. Every privileged operation passes through `rvm-security`.
6. Runtime actions produce witness records.
7. The registry maintains a public transparency log.
8. Compromised packages can be revoked.
9. Installed packages can be quarantined without deleting user state.
10. Enterprise administrators can override public store availability.

**RVForge must never silently revoke or delete locally owned RVFs.**
Revocation blocks execution by policy while preserving export and forensic
access. A user whose agent is revoked keeps their state capsule, keeps the
ability to export it, and keeps enough of the artifact to investigate what
happened. Remote deletion of a user's local property is not a security
control available to this platform.

### 10. Commercial model

```text
Free · Open source · One time purchase · Subscription · Per user
Per device · Per organization · Per execution · Usage metered
Private enterprise license · Partner appliance license
```

A reasonable initial marketplace fee is **10 percent**, excluding model and
compute costs. Inference may come from an embedded local model, publisher
supplied inference, Cognitum Meta LLM, customer supplied model credentials,
or enterprise private inference. All external inference costs must be
disclosed before execution.

### 11. Enterprise

Administrators can approve or deny agents; create private catalogs; set
capability ceilings; require local inference; restrict network domains;
control updates; assign licenses; deploy agents; revoke capabilities; inspect
witnesses; export audit evidence; and set jurisdiction rules.

**Organizational policy always overrides publisher-requested permissions.** A
publisher may request a capability; only the organization can grant it inside
that organization's fleet.

### 12. Scope

**MVP (P15)**: Windows, macOS, and Linux Reader; public RVF registry; npm
publisher CLI; publisher identity and signing; search and agent listings;
free agent installation; capability cards; WASM quarantine; RVM integration;
updates and rollback; witness viewer; revocation; private organization
catalogs.

**Deferred**: paid applications; revenue sharing; mobile support; advanced
evaluations.

**Effort**: a functional marketplace MVP is roughly twelve to sixteen weeks
with five engineers. A production commercial store with payments, enterprise
controls, moderation, and independent security validation is closer to six
months.

## Acceptance criteria

The platform acceptance test is one end-to-end pass:

1. A publisher uploads one signed RVF.
2. Automated review detects its exact capabilities.
3. A user installs it without developer tools.
4. RVM denies undeclared access.
5. The agent runs offline.
6. An update requests fresh permission.
7. Every build, installation, and privileged action verifies through the
   public witness record.

## Consequences

### Positive

- Install-once/distribute-RVFs collapses the per-agent packaging cost to
  zero, which is what makes a long tail of small agents economically viable.
- Mandatory capability disclosure gives the store a comparison axis that
  conventional app stores cannot offer: two agents can be ranked by what they
  are unable to do.
- Immutable predecessor-linked releases make revocation precise. A specific
  release is blocked; the publisher's other releases and the user's state are
  untouched.
- Countersigning splits trust between publisher and platform, so neither a
  compromised publisher key nor a compromised registry alone is sufficient to
  ship an executing package.
- Enterprise catalogs reuse the same registry and trust machinery rather than
  forking a second distribution path.

### Negative

- Mandatory review is a throughput ceiling. Ten trigger conditions route a
  meaningful fraction of interesting agents into human review, and that
  queue is a permanent staffed cost, not a launch expense.
- Reproducible evaluations are only as good as their harnesses; a published
  evaluation score invites gaming and will need adversarial maintenance.
- The no-silent-deletion rule means a compromised package can remain on disk,
  exportable, after revocation. We accept the forensic-artifact risk to avoid
  holding a remote-delete capability over users' own files.
- Apple's 2.5.2 restriction structurally excludes iPhone and iPad from the
  install-once model, so mobile reach requires a different product, not a
  port.
- A 10 percent fee funds review and infrastructure but is a competitive
  disadvantage against registries that charge nothing because they verify
  nothing.

## Alternatives Considered

- **Distribute only through existing app stores.** Rejected: guideline 2.5.2
  forbids the downloaded-executable model outright on iOS, and every other
  store's review process is opaque to us and cannot express a capability
  contract or a witness record. We would inherit their trust vocabulary and
  lose ours.
- **Reuse the npm registry for RVF packages.** Rejected: npm has no notion of
  capability manifests, evaluation reports, entitlements, revocation that
  blocks execution, or countersigning, and its mutable-by-default publishing
  model conflicts with immutable predecessor-linked releases. We use npm to
  distribute the publisher CLI, not the agents.
- **Publisher signature only, no platform countersignature.** Rejected:
  a single signature makes a stolen publisher key sufficient to ship
  executable code to every installed base. Countersigning is also what lets
  revocation be enforced at execution time rather than merely advertised.
- **Trust badges as optional publisher opt-in.** Rejected explicitly — this
  is the industry default and it is the mechanism by which agent marketplaces
  fill with unverifiable wrappers. Disclosure, evaluation, identity, and
  witnessing are mandatory or they are decoration.
- **Allow broad permission descriptions for convenience.** Rejected: "access
  your computer" is the exact phrasing that trained a generation of users to
  click through consent dialogs, and it would void the comparison axis in the
  first Positive consequence above.

## Implementation Surfaces

```text
@ruvector/rvforge     npm publisher CLI (supersedes @ruvector/forge)
rvforge registry      Content-addressed release, manifest, and provenance store
rvforge store         Public marketplace web application
rvforge reader        Desktop installer and runtime host (ADR-289)
rvforge review        Automated review pipeline and manual review queue
rvforge enterprise    Private catalogs, policy engine, audit export
```
