# Claude Code CLI: Context and Session Management

## Context Window Management

### Token Budgets

Key token constants found in the source:

| Constant | Value | Purpose |
|----------|-------|---------|
| `maxTokens` | 64000 | Default max output tokens |
| `maxTokens` | 40000 | Alternative budget |
| `maxTokens` | 16000 | Constrained budget |
| `maxOutputTokensOverride` | 4096 | Override for specific cases |
| `budget_tokens` | 10000 | Thinking budget |
| `max_tokens` | 1000/256/512 | Various API call limits |

### Auto-Compaction

When the conversation context approaches the model's window limit,
Claude Code automatically compacts the history.

**Compaction Types**:
- `Compact20260112` -- Current compaction algorithm
- `Compact20260112Edit` -- Edit-aware compaction
- `CompactOnPromptTooLong` -- Triggered when prompt exceeds limit

**Compaction Flow**:
```
Token count check
     |
     v
CompactThreshold exceeded?
     |  No -> Continue
     |  Yes
     v
CompactMessage generated (summary of conversation)
     |
     v
CompactHooks fired
     |
     v
Message history replaced with compact summary
     |
     v
CompactTracking updated
```

**Configuration**:
- `autoCompactWindow` -- Context window size before compacting
- `DISABLE_AUTO_COMPACT` / `DISABLE_COMPACT` -- Disable compaction
- `ENABLE_CLAUDE_CODE_SM_COMPACT` / `DISABLE_CLAUDE_CODE_SM_COMPACT` --
  Small model compaction
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW` -- Custom window size
- `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` -- Compact percentage override
- `CLAUDE_CODE_DISABLE_PRECOMPACT_SKIP` -- Disable skip optimization
- `CLAUDE_AFTER_LAST_COMPACT` -- State after last compaction
- `/compact` slash command -- Manual trigger

**Compaction discovers tools**:
`CompactDiscoveredTools` -- Tracks which tools were used, so the
compacted summary preserves tool context.

### 1M Context Support

`CLAUDE_CODE_DISABLE_1M_CONTEXT` -- Can disable extended 1M token context
(normally enabled for models that support it).

## Session Persistence

### Session Storage

Sessions are persisted to disk at:
```
~/.claude/projects/<project-hash>/
```

**Key functions**:
- `persistSession` -- Save session state to disk
- `resumeSession` / `resumeSessionAt` -- Restore session
- `sessionFile` -- Session data file path

### Session Resume

Multiple resume modes:
- `-c, --continue` -- Resume most recent session in current directory
- `-r, --resume [id]` -- Resume by session ID
- `--session-id <uuid>` -- Use specific session UUID
- `--fork-session` -- Fork from existing session (new ID)
- `--from-pr [pr]` -- Resume session linked to a PR
- `/resume` slash command -- Interactive picker

### Session Teleport

The `/teleport` command can transfer session context across environments:
- `processMessagesForTeleportResume` -- Handles teleport resume
- Useful for moving between CLI and VS Code, or between machines

### Session Configuration

- `CLAUDE_CODE_REMOTE_SESSION_ID` -- Remote session identifier
- `TEST_ENABLE_SESSION_PERSISTENCE` -- Test mode persistence
- `--no-session-persistence` -- Disable persistence (print mode)
- `CLAUDE_CODE_RESUME_INTERRUPTED_TURN` -- Resume interrupted turns
- `CLAUDE_CODE_RESUME_THRESHOLD_MINUTES` -- Time threshold for resume
- `CLAUDE_CODE_RESUME_TOKEN_THRESHOLD` -- Token threshold for resume

## CLAUDE.md System

### Auto-Discovery

Claude Code discovers CLAUDE.md files by walking up the directory tree:

```
./CLAUDE.md                     <- Project root
../CLAUDE.md                    <- Parent directory
~/.claude/CLAUDE.md             <- User-level global
.claude/settings.json "claudeMdExcludes"  <- Exclusions
```

Additional directories via:
- `--add-dir` CLI flag
- `CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD` env var
- `CLAUDE_CODE_DISABLE_CLAUDE_MDS` -- Disable entirely

### Memory System

Auto-memory creates and manages persistent knowledge:

```
~/.claude/projects/<project>/memory/MEMORY.md
```

- `autoMemoryEnabled` -- Enable/disable per project
- `autoMemoryDirectory` -- Custom storage directory
- `CLAUDE_CODE_DISABLE_AUTO_MEMORY` -- Disable entirely
- `/memory` slash command -- View/edit
- `CLAUDE_COWORK_MEMORY_PATH_OVERRIDE` -- Override path
- `CLAUDE_COWORK_MEMORY_EXTRA_GUIDELINES` -- Extra memory guidelines
- `autoDreamEnabled` -- Background memory consolidation

## File Checkpointing

Claude Code can checkpoint file states for recovery:

- `CheckpointingEnabled` -- Feature flag
- `CLAUDE_CODE_DISABLE_FILE_CHECKPOINTING` -- Disable
- `CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING` -- SDK mode
- `/rewind` slash command -- Revert to checkpoint

Checkpointing tracks:
- File content before/after edits
- `checkpoint_count` -- Number of checkpoints
- `checkpoint_age_entries` -- Age tracking

## Prompt Caching

Claude Code uses Anthropic's prompt caching to reduce costs:

- `cache_control` / `cacheControl` -- Cache control headers
- `promptCacheSharingEnabled` -- Share cache across sessions
- `promptCacheReadTokens` / `promptCacheWriteTokens` -- Token tracking
- `DISABLE_PROMPT_CACHING` -- Disable globally
- `DISABLE_PROMPT_CACHING_HAIKU/SONNET/OPUS` -- Per-model disable
- `ENABLE_PROMPT_CACHING_1H_BEDROCK` -- 1-hour cache on Bedrock
- `PromptCache1hAllowlist` -- Models eligible for 1h cache

## Idle Detection

- `CLAUDE_CODE_IDLE_THRESHOLD_MINUTES` -- Idle timeout
- `CLAUDE_CODE_IDLE_TOKEN_THRESHOLD` -- Token threshold for idle
- `CLAUDE_CODE_STALL_TIMEOUT_MS_FOR_TESTING` -- Stall detection

## Background Tasks

- `CLAUDE_AUTO_BACKGROUND_TASKS` -- Auto background task creation
- `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` -- Disable
- `MessageBackground` -- Background message handling
- Tasks can run while user is idle
