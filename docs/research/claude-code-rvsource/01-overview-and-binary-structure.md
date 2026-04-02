# Claude Code CLI: Overview and Binary Structure

## Version Analyzed

- **Version**: 2.1.90
- **Runtime**: Bun 1.3.11 (compiled as Single Executable Application)
- **Binary Format**: ELF 64-bit LSB executable, x86-64, dynamically linked
- **Binary Size**: ~229 MB (uncompressed), ~51 MB (.zst compressed)

## Installation Locations

```
/home/<user>/.local/bin/claude           -> symlink to active version
/home/<user>/.local/share/claude/versions/2.1.90   -> native binary (Bun SEA)
/home/<user>/.vscode-remote/extensions/anthropic.claude-code-2.1.90-linux-x64/
  extension.js       -> VS Code extension entry (1.8 MB, minified)
  webview/index.js   -> React-based UI (4.8 MB, minified)
  package.json       -> VS Code extension manifest
  claude-code-settings.schema.json  -> 76-property settings schema
  resources/
    native-binary/claude   -> same binary, for VS Code
    walkthrough/           -> onboarding assets
```

## Binary Architecture

The Claude Code CLI is a **Bun Single Executable Application (SEA)**.

### How It Works

1. **Bun Runtime**: The binary embeds the full Bun runtime (v1.3.11)
   - V8 engine compiled in (not Node.js -- Bun uses JavaScriptCore, but the
     binary symbols show V8, indicating a hybrid or Node.js compatibility layer)
   - Native HTTP client, filesystem, and child_process support

2. **Embedded JS Bundles**: The binary contains multiple bundled JS regions:
   - **Bun framework code** (~2.1 MB at offset ~1.8M): React SSR, HMR, Tailwind
   - **Application code** (~12.8 MB at offset ~107M-120M): The actual Claude Code
     application with all tools, agent loop, MCP client, permissions, etc.

3. **Compression**: Ships with a `.zst` (Zstandard) compressed copy alongside
   the full binary for efficient distribution

### Binary Sections (Key Offsets)

| Region | Offset | Content |
|--------|--------|---------|
| ELF headers + V8 | 0 - 1.8M | Native code, V8 engine |
| Bun framework bundle | ~1.8M - 4.0M | Bun's built-in React, HMR, Tailwind |
| V8 snapshot data | ~4M - 107M | V8 heap snapshot, compiled bytecode |
| Application JS | ~107M - 120M | Claude Code application source |
| Binary data | 120M+ | Resources, compressed assets |

## Version Management

```
~/.local/share/claude/versions/
  2.1.86   (228 MB)
  2.1.87   (228 MB)
  2.1.90   (229 MB)  <- current
```

The CLI supports auto-updates with two channels:
- `latest` (default): Bleeding edge
- `stable`: Tested releases

Version selection controlled by `autoUpdatesChannel` setting and
`DISABLE_AUTOUPDATER` env var.

## Dual Interface

Claude Code operates as both:

1. **CLI Tool** (`claude` binary): Terminal-based interactive REPL and
   non-interactive `--print` mode
2. **VS Code Extension**: WebView-based UI communicating with the same
   core via the `extension.js` bridge

The extension.js acts as a bridge between VS Code's extension host and
the native binary, translating between VS Code's API and Claude Code's
internal protocols.

## Package Identity

```json
{
  "name": "claude-code",
  "version": "2.1.90",
  "displayName": "Claude Code for VS Code",
  "publisher": "Anthropic"
}
```

Zero npm dependencies in the VS Code extension -- everything is bundled.
