# rvAgent Security Documentation

This document describes the threat model, security defaults, and all 13 security controls implemented in rvAgent.

## Threat Model

rvAgent operates in an environment where:

1. **LLM outputs are untrusted** -- the model may be influenced by indirect prompt injection via file contents, grep results, or command output
2. **Filesystem content is untrusted** -- AGENTS.md, SKILL.md, and user files may contain malicious content
3. **Subagent results are untrusted** -- child agents may produce oversized, malformed, or injection-bearing output
4. **Network endpoints are untrusted** -- ACP server requests may be unauthenticated or malicious
5. **Unicode content may be weaponized** -- BiDi overrides, zero-width characters, and homoglyphs can mislead both humans and models

The security model assumes that the agent framework itself is trusted but all external inputs (LLM responses, file contents, user input, network requests) must be validated at system boundaries.

## Security Defaults

All security features are enabled by default. No configuration is required for baseline protection:

| Default | Value | Effect |
|---|---|---|
| `virtual_mode` | `true` | Filesystem operations confined to working directory |
| `sensitive_env_patterns` | 10 patterns | Env vars matching SECRET, KEY, TOKEN, etc. stripped before child processes |
| `trust_agents_md` | `false` | AGENTS.md files require explicit trust |
| `max_response_length` | 100 KB | SubAgent responses truncated beyond this limit |
| Grep mode | Literal (fixed-string) | Prevents ReDoS from regex patterns |
| Skill name validation | ASCII-only | Rejects Unicode confusable characters |
| Tool result wrapping | Enabled | All tool outputs wrapped in `<tool_output>` blocks |

## Security Controls

### C1: Atomic Path Resolution (CRITICAL)

**Threat:** TOCTOU symlink race conditions where a path resolves safely at check time but is swapped to a symlink before file open.

**Control:** Two-phase resolution:

1. Open file with `O_NOFOLLOW` to reject symlinks
2. Post-open verification via `/proc/self/fd/N` to confirm the real path is within `cwd`

Additionally, `virtual_mode` defaults to `true`, confining all filesystem operations within the configured working directory. Ripgrep invocations include `--no-follow` to prevent symlink traversal during search.

**Configuration:**

```rust
SecurityPolicy {
    virtual_mode: true, // default
    ..Default::default()
}
```

### C2: Shell Execution Hardening (CRITICAL)

**Threat:** Shell injection, credential leakage via environment, and command template injection.

**Controls:**

1. **Environment sanitization** -- before spawning child processes, all env vars matching these patterns are stripped:
   - `SECRET`, `KEY`, `TOKEN`, `PASSWORD`, `CREDENTIAL`
   - `AWS_*`, `AZURE_*`, `GCP_*`
   - `DATABASE_URL`, `PRIVATE`

2. **Optional command allowlist** -- when configured, only explicitly listed commands may be executed

3. **Witness chain logging** -- every `execute()` call is recorded with a SHAKE-256 hash of the command for audit

4. **`env_clear()` + explicit safe env** -- child processes do not inherit the full parent environment

**Configuration:**

```rust
SecurityPolicy {
    command_allowlist: vec!["cargo".into(), "npm".into(), "git".into()],
    sensitive_env_patterns: vec!["SECRET".into(), "KEY".into(), /* ... */],
    ..Default::default()
}
```

### C3: Tool Result Sanitization (CRITICAL)

**Threat:** Indirect prompt injection where tool outputs (file contents, grep results, command output) contain instructions that manipulate the LLM.

**Control:** `ToolResultSanitizerMiddleware` wraps all tool result messages in clearly delimited blocks:

```
<tool_output tool="read_file" id="call_abc123">
[actual tool output here]
</tool_output>
```

This provides defense-in-depth by making tool output boundaries unambiguous to the model.

### C4: AGENTS.md / SKILL.md Trust Verification (CRITICAL)

**Threat:** Untrusted AGENTS.md or SKILL.md files injecting malicious instructions into the system prompt.

**Controls:**

1. **Hash verification** -- trusted sources can provide a signed manifest; files are verified against it before loading
2. **`trust_agents_md` flag** -- defaults to `false`; must be explicitly enabled
3. **Size limits** -- YAML frontmatter capped at 4KB, skill files capped at 1MB
4. **YAML bomb protection** -- explicit recursion depth and anchor expansion limits in `serde_yaml` parsing

### C5: Sandbox Path Restriction (CRITICAL)

**Threat:** Sandbox implementations allowing filesystem access outside their designated root.

**Control:** The `SandboxBackend` trait requires implementations to declare `sandbox_root() -> &Path`. All file operations must be confined to this root. This is an implementation contract -- concrete sandbox providers (Modal, Runloop, Daytona) must enforce isolation on their end.

### C6: ACP Server Authentication (HIGH)

**Threat:** Unauthenticated access to the ACP server allowing arbitrary agent invocation.

**Controls:**

1. **API key authentication** -- `Authorization: Bearer <key>` header required on all endpoints
2. **Rate limiting** -- configurable, default 60 requests/minute
3. **Request body size limit** -- default 1MB, prevents resource exhaustion
4. **TLS enforcement** -- required for non-localhost connections

The ACP server returns structured error responses (`ErrorResponse`) with appropriate HTTP status codes (401, 413, 429).

### C7: Unicode Security (HIGH)

**Threat:** BiDi override attacks that reverse displayed text, zero-width characters that hide content, and homoglyph attacks using visually similar characters from different scripts.

**Controls (full parity with Python `unicode_security.py`):**

1. **BiDi detection** -- detects U+202A-U+202E (directional embeddings/overrides) and U+2066-U+2069 (isolate controls)
2. **Zero-width detection** -- detects U+200B-U+200F, U+2060, U+FEFF
3. **Script confusable detection** -- identifies Cyrillic, Greek, and Armenian characters that are visual lookalikes for Latin (e.g., Cyrillic 'A' U+0410 vs Latin 'A')
4. **Mixed-script URL checking** -- detects URLs with domains containing characters from multiple scripts
5. **Stripping function** -- `strip_dangerous_unicode()` removes all dangerous codepoints while preserving safe Unicode (accented characters, CJK, etc.)

### C8: SubAgent Result Validation (HIGH)

**Threat:** Runaway subagents producing oversized responses, or subagent outputs containing prompt injection patterns.

**Controls:**

1. **Maximum response length** -- configurable via `SecurityPolicy.max_response_length`, default 100KB
2. **Control character stripping** -- removes known prompt injection patterns from subagent output
3. **Tool call rate limiting** -- detects runaway behavior (excessive tool calls within a single subagent run)

### C9: Session Encryption at Rest (MEDIUM)

**Threat:** Session data containing conversation history, file contents, and potentially sensitive information stored in plaintext.

**Controls:**

1. **AES-256-GCM encryption** -- session checkpoints encrypted before writing to disk
2. **Unpredictable filenames** -- UUIDs used for conversation history offload files
3. **Restrictive permissions** -- files created with 0600 (owner read/write only)
4. **PII stripping** -- optional pattern-based PII removal before persistence

### C10: Skill Name ASCII Restriction (MEDIUM)

**Threat:** Unicode confusable attacks where a skill named with Cyrillic characters (e.g., "deploy" using Cyrillic 'е' and 'р') is mistaken for a legitimate skill.

**Control:** `validate_ascii_identifier()` requires skill names to:
- Start with an ASCII lowercase letter
- Contain only ASCII lowercase letters, digits, hyphens, and underscores
- Explicitly rejects `c.is_alphabetic()` in favor of `c.is_ascii_lowercase()` to prevent non-Latin alphabetic characters

### C11: CompositeBackend Path Re-Validation (MEDIUM)

**Threat:** Path traversal after prefix stripping in `CompositeBackend`, where a path like `/workspace/../etc/passwd` becomes `../etc/passwd` after stripping the `/workspace` prefix.

**Control:** After prefix stripping, the resulting path is re-validated:
- Rejects paths containing `..` components
- Rejects paths starting with `~`
- Returns `FileOperationError::InvalidPath` on violation

### C12: Tool Call ID Validation (MEDIUM)

**Threat:** Injection via tool call IDs containing special characters or excessive length.

**Control:** Tool call IDs are validated to:
- Maximum 128 characters
- ASCII alphanumeric characters, hyphens, and underscores only

### C13: Grep Literal Mode Enforcement (MEDIUM)

**Threat:** ReDoS (Regular Expression Denial of Service) when user-controlled patterns are passed to grep.

**Control:** Grep defaults to literal/fixed-string mode (equivalent to `rg -F`). The `StateBackend` uses `line.contains(pattern)` for string matching. The `FilesystemBackend` uses `grep-searcher` with literal matching enabled. If regex mode is needed, the `regex` crate's built-in backtracking limits provide protection.

## Configuration Reference

All security settings are configured via `SecurityPolicy` in `RvAgentConfig`:

```rust
pub struct SecurityPolicy {
    /// Confine filesystem to working directory (default: true)
    pub virtual_mode: bool,

    /// Optional shell command allowlist (default: empty = all allowed)
    pub command_allowlist: Vec<String>,

    /// Env var patterns stripped before child processes
    pub sensitive_env_patterns: Vec<String>,

    /// Max subagent response length in bytes (default: 102400)
    pub max_response_length: usize,

    /// Trust AGENTS.md files in working directory (default: false)
    pub trust_agents_md: bool,
}
```

Resource budgets provide additional governance:

```rust
pub struct ResourceBudget {
    pub max_time_secs: u32,          // default: 300
    pub max_tokens: u64,             // default: 200_000
    pub max_cost_microdollars: u64,  // default: 5_000_000 ($5)
    pub max_tool_calls: u32,         // default: 500
    pub max_external_writes: u32,    // default: 100
}
```
