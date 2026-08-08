# ADR-295: RVForge Agent Dock — Persistent Security and Control Surface

- **Status**: Implemented
- **Date**: 2026-08-03
- **Updated**: 2026-08-03 — first implementation target shipped: dock in crates/rvforge-reader (typed AgentProvidedStatus/SystemOwnedStatus boundary, 8-state machine, roster, event thresholds, pill+expanded UI; 90 tests). Other platform placements (tray/menubar/Live Activity/appliance) tracked as follow-ups.
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-285, ADR-286, ADR-289, ADR-294
- **Tags**: rvforge, agent-dock, ux, trust, security, control-surface

## Context

ADR-294 established RVForge as a store, registry, and trust system, and it
specified what the Reader shows *while an agent runs*: current task, runtime
type, model activity, resource usage, network connections, filesystem access,
tool calls, witness status, cost, and always-visible emergency controls. That
surface exists inside the RVForge application. Persistent agents do not.

An agent installed through RVForge is expected to keep running after the user
switches to a document, a browser, or a terminal. At that moment every control
ADR-294 specified is behind a window the user is not looking at. Conventional
agent interfaces resolve this by hiding autonomous activity in the background
and surfacing it only when the agent chooses to speak — which is precisely the
arrangement that makes a running agent unaccountable. If the only evidence that
an agent is executing is a message the agent decided to send, then the absence
of a message is not evidence that it stopped.

The dock is therefore **a security and control surface, not a decorative chat
widget**. It is a continuous trust indicator, and its job is to make the
following three claims independently checkable at all times, from any
application, without the agent's cooperation:

1. *Something is running* — and here is what and how much of it.
2. *Here is what it is permitted to do* — sourced from the capability policy,
   not from the agent's own description of itself.
3. *You can stop it right now* — through a control the agent cannot reach.

The threat this addresses is a compromised or adversarial agent spoofing its
own status. Three concrete shapes: an agent that renders a convincing "Stopped"
state while continuing to execute; an agent that renders a fake approval dialog
to harvest a click that grants a capability the user never intended; and an
agent that renders a trust badge or a "network disabled" indicator it does not
actually possess. Each of these is trivially available to any agent that can
draw its own status UI. None of them is available to an agent whose status UI is
drawn by RVForge from RVForge's own state.

## Decision

Ship an **RVForge Agent Dock**: a horizontal pill, persistent across
applications, that displays and controls running agents. Its chrome is owned by
RVForge; agents contribute content into it but never render it.

### 1. Collapsed state

The collapsed pill carries eight elements:

```text
Agent icon · Agent name · Current task · Progress · Runtime status ·
Pause · Terminate · Expand
```

Rendered:

```text
Cognitum Analyst   Reviewing 42 documents   68%   Pause   Stop
```

This is the default resting form. It answers "what is running and what is it
doing" without any interaction, and it carries the stop control inline, so
terminating an agent never requires first opening something.

### 2. Expanded state

Expanding the pill shows ten elements:

1. Current objective.
2. Recent actions.
3. Pending approvals.
4. Model and token usage.
5. CPU and memory consumption.
6. Network activity.
7. Capability grants.
8. Witness status.
9. Estimated cost.
10. Text or voice instruction field.

Items 4 through 8 are the accountability core: they are measured by RVForge and
RVM rather than reported by the agent, so they remain trustworthy when the agent
is not. Item 10 is the only place in the dock where the user addresses the agent
directly.

### 3. Agent states

Eight states, each with an unmistakable color and icon — unmistakable meaning
distinguishable at pill size, at a glance, and without reading the label:

```text
Idle · Running · Waiting for approval · Paused · Capability denied ·
Error · Quarantined · Completed
```

`Capability denied` and `Quarantined` are deliberately first-class states rather
than error subtypes. A denied capability (ADR-286) and a quarantined package
(ADR-294 §9) are both normal, expected outcomes of the trust system working, and
collapsing them into a generic error would train users to dismiss them.

### 4. One-action pause and terminate

**The user must always be able to pause or terminate an agent with one action.**
This holds in every state, in both collapsed and expanded form, and on every
platform placement. Pause and Terminate are never behind an expand, a menu, a
confirmation chain, or a state in which the agent could make them unavailable.

### 5. Multiple agents

Do not place every agent in the bar. A swarm of fifteen agents rendered as
fifteen pills is an unusable bar and an unreadable trust surface. Show:

```text
Active agent · Two secondary agent icons · Additional agent count ·
Aggregate resource usage · Approval count
```

Selecting the count opens the full swarm view. The aggregate resource usage and
the approval count are the two figures that must stay visible regardless of how
many agents exist, because they are the ones that reveal a fleet doing more work
or asking for more consent than the user expects.

### 6. Platform placement

The dock is one contract with six host presentations:

1. **macOS** — menu bar item, or a floating notch-style dock.
2. **Windows** — system tray, plus an optional floating dock.
3. **Linux** — panel item, or a floating dock.
4. **Browser** — a toolbar inside RVForge.
5. **Mobile** — Live Activity carrying status and approvals.
6. **RVM appliance** — dashboard showing active coherence domains.

Each placement must satisfy the same acceptance test in §Acceptance criteria. A
placement that cannot carry one-action terminate is not a valid placement.

### 7. Trust boundary — chrome versus content

**The dock chrome is controlled by RVForge, never by the agent.** Agents supply
only two things: task text and progress. They can never alter:

```text
Pause button · Trust badge · Network indicator · Permission state
```

Everything in that list is rendered from RVForge and RVM state — capability
policy, measured runtime telemetry, witness records — and is unreachable from
agent-supplied content. An agent that wishes to appear paused must actually be
paused, because the state indicator is a projection of the runtime's state, not
a field the agent populates.

**System messages and agent-generated content must be visually distinct**, by a
distinction the agent cannot imitate: agent content is confined to regions that
are visibly marked as agent-authored, and system chrome uses presentation
affordances unavailable inside those regions. Without this, a malicious agent
displays a fake approval prompt, or claims it has stopped while it is still
executing, and the user has no way to tell the difference.

This separation is what makes the three claims in §Context checkable. It is the
load-bearing rule of this ADR; every other element here is a presentation
detail by comparison.

### 8. Key capability card

Expanding the agent icon immediately shows seven lines:

```text
Verified publisher · RVM or WASM isolated · Local model ·
Network disabled · Selected folder access · Encrypted memory ·
Witness chain valid
```

This is the running-agent counterpart to the pre-installation capability
contract of ADR-294 §6: the same facts, restated as present-tense properties of
a live instance rather than as requests. It is chrome, not content — each line
is derived from the capability policy and the witness chain, so "Network
disabled" means the runtime is denying network access, not that the agent
asserts it does not use it.

### 9. Noise control

The largest UX risk is excessive noise from perpetual agents. A dock that
interrupts constantly gets dismissed, and a dismissed trust surface provides no
trust. The fix is event thresholds. Surface only:

```text
Approvals · Policy violations · Cost limits · Failures ·
Meaningful milestones
```

Everything else accrues silently into the expanded state's recent actions, where
a user who wants detail can find it and a user who does not is not taxed for it.
Routine progress is a number in the pill, never an interruption.

## Acceptance criteria

The dock acceptance test is a single timed task:

**From any application, the user can identify the active agent, understand what
it is doing, inspect its permissions, and terminate it within five seconds and
two interactions.**

"From any application" means the test is run without the RVForge window
focused, and without opening it. Each of the six platform placements in §6 must
pass this test independently. Because inspecting permissions and terminating
must both fit inside two interactions, permissions cannot sit behind a
navigation chain and terminate cannot sit behind a confirmation chain.

## Consequences

### Positive

- A running agent becomes continuously accountable rather than accountable only
  while its window is focused. The absence of activity in the dock is
  meaningful, which is exactly what the absence of an agent's own messages is
  not.
- Chrome/content separation makes status spoofing structurally unavailable
  rather than policy-forbidden. There is no agent-reachable path to the pause
  button, trust badge, network indicator, or permission state.
- The capability card gives the pre-install capability contract a live
  counterpart, so a user can check at any moment that the contract they
  approved is the contract being enforced.
- One-action terminate on every surface means the fastest available response to
  suspected misbehavior is also the safest one.
- Event thresholds keep the surface credible; a dock that only speaks for
  approvals, violations, cost limits, and failures is a dock users keep enabled.

### Negative

- Six platform placements are six implementations of the same contract, each
  with its own OS affordances and constraints. Per-OS surface area is the
  dominant ongoing cost of this decision, and divergence between placements is
  the most likely source of a placement that quietly fails the acceptance test.
- Strict chrome/content separation constrains agent UI richness. An agent cannot
  render custom controls, custom approval flows, or branded status presentation
  in the dock, and some legitimately useful interaction patterns are lost with
  the spoofing risk.
- Mobile Live Activity and the RVM appliance dashboard are governed by host
  frameworks whose update cadence, size limits, and interaction models we do not
  control, so those two placements will lag the desktop ones in fidelity.
- Aggregating a swarm into one active agent plus two icons plus a count means
  the dock deliberately hides individual agents. The aggregate figures must be
  right, because they are all the user sees until they open the swarm view.
- A persistent always-on-top surface competes for scarce screen real estate with
  the user's actual work, and users will attempt to hide it. Any hide affordance
  must not become a way to run agents invisibly.

## Alternatives Considered

- **Notification center only.** Rejected: notifications are transient and
  historical. They can report that something happened; they cannot show that
  something *is happening*, and they carry no live control. A user who missed
  the notification has no surface that still says an agent is running, and
  terminate-from-notification is not available once the notification is gone.
- **In-app status only — no dock.** Rejected: it fails the acceptance test by
  construction. Persistent agents run while the user is in other applications,
  which is exactly when status and control are unavailable. This is the status
  quo the dock exists to replace.
- **Agent-rendered dock UI.** Rejected for spoofing risk. If an agent renders
  its own status surface, then "Paused", "Network disabled", "Verified
  publisher", and an approval prompt are all things the agent can draw whether
  or not they are true. Every guarantee in this ADR reduces to the agent's own
  honesty, which is the property we cannot assume. Agents supply task text and
  progress; RVForge draws everything that carries a security meaning.

## Implementation note

The first implementation target is the Tauri Reader (`crates/rvforge-reader`),
as a dock window plus the runtime status screen. The chrome/content separation
of §7 maps directly onto Tauri's process structure: dock chrome — pill layout,
state indicator, pause and terminate controls, trust badge, network indicator,
capability card — is Rust-owned window chrome, driven by RVM state through
`rvm-ffi` (ADR-289 §4). Agent-supplied task text and progress are webview
content, confined to the regions §7 marks as agent-authored.

Terminate in the dock is `rvm_terminate`; pause is `rvm_suspend`; the witness
status line and the capability card read from the witness chain and capability
policy rather than from any agent-provided value. The dock therefore adds no
security-critical logic of its own — matching ADR-289's rule that the desktop
layer never reimplements what RVM enforces.

## Implementation Surfaces

```text
rvforge-reader     Tauri dock window (Rust chrome) + runtime status screen
rvm-ffi            rvm_suspend / rvm_terminate / rvm_export_witness for dock controls
rvm-policy         capability grants rendered in the capability card
rvm-witness        witness status line and witness-chain validity indicator
dock placements    macOS menu bar · Windows tray · Linux panel · browser toolbar ·
                   mobile Live Activity · RVM appliance dashboard
```
