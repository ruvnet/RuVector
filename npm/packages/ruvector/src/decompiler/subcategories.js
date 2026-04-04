/**
 * subcategories.js - Fine-grained module classification keywords.
 *
 * Each key is a hierarchical module path (e.g. 'tools/bash').
 * Keywords can be plain strings (exact match) or contain '.*' for regex.
 * Used by module-splitter.js to classify statements into ~30-40 modules
 * instead of the original ~9 broad categories.
 */

'use strict';

// ── Fine-grained module classification ─────────────────────────────────────
const SUBCATEGORIES = {
  // ── tools/* ────────────────────────────────────────────────────────────
  'tools/bash': [
    'BashTool', 'child_process', 'execSync', 'spawnSync', 'spawn(',
    'shell.*command', 'shellArgs', 'commandLine', 'bashCommand',
    'killProcess', 'processExit', 'childProcess',
  ],
  'tools/read': [
    'FileReadTool', 'ReadTool', 'readFile', 'readFileSync',
    'FileRead', 'fileContents', 'readContent',
  ],
  'tools/edit': [
    'FileEditTool', 'EditTool', 'old_string', 'new_string',
    'applyEdit', 'textEdit', 'replaceInFile', 'editContent',
  ],
  'tools/write': [
    'FileWriteTool', 'WriteTool', 'writeFile', 'writeFileSync',
    'createFile', 'FileWrite', 'writeContent',
  ],
  'tools/glob': [
    'GlobTool', 'glob(', 'globSync', 'minimatch', 'picomatch',
    'ListFilesTool', 'filePattern', 'globPattern',
  ],
  'tools/grep': [
    'GrepTool', 'ripgrep', 'SearchTool', 'searchPattern',
    'contentSearch', 'grepResult', 'matchLine',
  ],
  'tools/agent': [
    'AgentTool', 'AgentOutputTool', 'subagent', 'spawnAgent',
    'agentTask', 'taskResult', 'delegateTask',
  ],
  'tools/web-fetch': [
    'WebFetch', 'httpGet', 'fetchUrl', 'urlFetch',
    'webRequest', 'httpRequest',
  ],
  'tools/web-search': [
    'WebSearch', 'searchResults', 'webQuery',
    'searchEngine', 'searchWeb',
  ],
  'tools/notebook': [
    'NotebookEdit', 'notebook', 'jupyter', 'ipynb',
    'cellOutput', 'notebookCell',
  ],
  'tools/mcp-dispatch': [
    'ToolUse', 'ToolResult',
    'toolDefinition', 'toolSchema', 'inputSchema',
    'toolChoice', 'toolRunner', 'dispatchTool',
  ],
  'tools/todo': [
    'TodoWrite', 'TodoRead', 'todoList', 'todoItem',
  ],

  // ── core/* ─────────────────────────────────────────────────────────────
  'core/agent-loop': [
    'agentLoop', 'mainLoop', 'querySource', 'toolUseContext',
    'systemPrompt', 'conversationTurn', 'assistantMessage',
    'userMessage', 'messageHistory', 'handleToolUse',
    'processMessage', 'runLoop', 'loopIteration',
  ],
  'core/streaming': [
    'content_block_delta', 'message_start', 'message_stop',
    'message_delta', 'content_block_start', 'content_block_stop',
    'text_delta', 'input_json_delta', 'StreamEvent',
    'onStream', 'streamHandler', 'stream_event',
    'streamResponse', 'streamingMode',
  ],
  'core/context-manager': [
    'tengu_compact', 'microcompact', 'auto_compact',
    'compact_boundary', 'preCompactTokenCount',
    'postCompactTokenCount', 'compaction',
    'tokenCount', 'contextWindow', 'maxTokens',
    'promptCache', 'cacheControl', 'truncat',
    'contextOverflow', 'compactMessages',
  ],
  'core/session': [
    'sessionId', 'conversationId', 'sessionState',
    'persistSession', 'checkpoint', 'resume.*session',
    'restore.*session', 'turnCount', 'sessionHistory',
    'saveSession', 'loadSession',
  ],
  'core/error-handler': [
    'ErrorHandler', 'errorBoundary', 'handleError',
    'retryWith', 'isRetryable', 'overloaded',
    'rateLimited', 'backoff', 'retryAfter',
    'APIError', 'NetworkError',
  ],

  // ── permissions/* ──────────────────────────────────────────────────────
  'permissions/checker': [
    'canUseTool', 'Permission', 'permission',
    'allowedTools', 'permissionMode', 'isAllowed',
    'checkPermission', 'grantPermission', 'allowList',
    'denyList', 'alwaysAllowRules', 'denyWrite',
    'permissionCheck', 'allowRule', 'denyRule',
  ],
  'permissions/sandbox': [
    'sandbox', 'bubblewrap', 'seatbelt', 'firejail',
    'containerize', 'isolat', 'sandboxMode',
    'seccomp', 'landlock', 'pledg',
  ],
  'permissions/rules': [
    'permissionRule', 'ruleSet', 'matchRule',
    'pathRule', 'toolRule', 'readOnlyRule',
    'globRule', 'regexRule',
  ],

  // ── auth/* ─────────────────────────────────────────────────────────────
  'auth/oauth': [
    'OAuth', 'PKCE', 'authorization_code', 'token.*endpoint',
    'refresh.*token', 'authorizationUrl', 'codeVerifier',
    'codeChallenge', 'oauthFlow', 'oauthCallback',
  ],
  'auth/api-key': [
    'x-api-key', 'ANTHROPIC_API_KEY', 'apiKeyHelper',
    'apiKey.*valid', 'loadApiKey',
    'keyring',
  ],
  'auth/bedrock': [
    'Bedrock', 'BedrockRuntime', 'aws.*region',
    'awsProfile', 'sigv4', 'awsCredentials',
  ],
  'auth/vertex': [
    'Vertex', 'vertex.*ai', 'google.*cloud',
    'googleAuth', 'serviceAccount', 'vertexProject',
  ],

  // ── mcp/* ──────────────────────────────────────────────────────────────
  'mcp/client': [
    'McpClient', 'mcp.*connect', 'mcp.*initialize',
    'mcpConnection', 'mcp_client', 'connectMcp',
  ],
  'mcp/transport': [
    'StdioTransport', 'SseTransport', 'StreamableHttp',
    'McpTransport', 'transport.*type', 'transportLayer',
    'stdio.*transport', 'websocket.*transport',
  ],
  'mcp/protocol': [
    'jsonrpc', 'tools/list', 'tools/call',
    'resources/list', 'prompts/list', 'McpError',
    'mcp__', 'McpServer', 'mcp_server',
    'callTool', 'listTools',
  ],
  'mcp/servers': [
    'mcpServers', 'serverConfig', 'serverList',
    'registeredServers', 'spawnServer', 'serverProcess',
  ],

  // ── config/* ───────────────────────────────────────────────────────────
  'config/settings': [
    'settings.*json', 'loadSettings', 'saveSettings',
    'userSettings', 'Settings', 'configuration',
    'loadConfig', 'parseConfig',
  ],
  'config/env-vars': [
    'CLAUDE_CODE_', 'ANTHROPIC_',
    'envVar', 'dotenv', 'loadEnv',
  ],
  'config/models': [
    'modelId', 'modelName', 'model.*select',
    'mainLoopModel', 'availableModels', 'modelOverrides',
    'modelPreference', 'defaultModel',
  ],
  'config/feature-flags': [
    'featureFlag', 'isEnabled', 'flagValue',
    'experimentId', 'feature.*gate', 'rollout',
    'featureEnabled', 'featureConfig',
  ],

  // ── telemetry/* ────────────────────────────────────────────────────────
  'telemetry/otel': [
    'opentelemetry', 'OTEL_', 'TraceProvider',
    'SpanProcessor', 'tracing', 'span',
    'tracer', 'otelExporter',
  ],
  'telemetry/datadog': [
    'datadog', 'DD_', 'ddTrace', 'datadogExporter',
  ],
  'telemetry/events': [
    'tengu_', 'trackEvent', 'analytics',
    'Telemetry', 'sentry',
    'eventEmit', 'emitEvent', 'telemetryEvent',
  ],
  'telemetry/cost': [
    'cost', 'tokenUsage', 'inputTokens', 'outputTokens',
    'cacheRead', 'cacheCreation', 'pricing',
    'costTracker', 'usageMetrics',
  ],
  'telemetry/perfetto': [
    'perfetto', 'perfTrace', 'traceBegin',
    'traceEnd', 'traceCounter',
  ],

  // ── ui/* ────────────────────────────────────────────────────────────────
  'ui/slash-commands': [
    'slashCommand', 'registerCommand', 'commandHandler',
    'parseCommand', '/help', '/clear', '/compact',
    '/bug', '/init', '/login', '/logout',
    '/doctor', '/config', '/cost', '/memory',
  ],
  'ui/ink-components': [
    'useInput', 'useFocus', 'useApp', 'useStdin', 'useStdout',
    'inkRenderer', 'InkProvider', 'measureElement',
  ],
  'ui/keybindings': [
    'keybinding', 'keyHandler', 'hotkey',
    'onKeyPress', 'keyMap', 'shortcut',
  ],
  'ui/terminal': [
    'ansiColor', 'chalk', 'stripAnsi',
    'cursorMove', 'clearLine', 'terminalWidth',
    'isTerminal', 'ttyColumns',
  ],

  // ── model-provider/* ───────────────────────────────────────────────────
  'model-provider/anthropic': [
    'anthropic', 'Anthropic', 'claude-', 'claude_',
    'messagesCreate', 'AnthropicClient',
  ],
  'model-provider/openai': [
    'openai', 'OpenAI', 'chatCompletion',
    'gpt-', 'openAiClient',
  ],
  'model-provider/router': [
    'provider', 'routeModel', 'selectProvider',
    'providerConfig', 'modelRouter',
  ],

  // ── git/* ──────────────────────────────────────────────────────────────
  'git/operations': [
    'gitDiff', 'gitStatus', 'gitLog', 'gitCommit',
    'gitAdd', 'gitBranch', 'gitCheckout',
    'isGitRepo', 'getGitRoot', 'gitStash',
  ],

  // ── filesystem/* ───────────────────────────────────────────────────────
  'filesystem/operations': [
    'readdirSync', 'mkdirSync', 'statSync', 'lstatSync',
    'renameSync', 'unlinkSync', 'copyFileSync',
    'existsSync', 'realpathSync', 'accessSync',
    'fs.readdir', 'fs.mkdir', 'fs.stat', 'fs.lstat',
  ],

  // ── network/* ──────────────────────────────────────────────────────────
  'network/http': [
    'http.*request', 'https.*request', 'fetch(',
    'axios', 'got(', 'requestOptions',
    'responseBody', 'statusCode',
  ],
};

// ── String-literal patterns for minified code ─────────────────────────────
// Minified bundles mangle identifiers but preserve string literals.
// These patterns match quoted strings commonly found in each domain.
// Each pattern is matched against the raw code (not just identifiers).
const STRING_PATTERNS = {
  'tools/bash': ['"bash"', '"shell"', '"command"', '"child_process"', '"spawn"', '"BashTool"'],
  'tools/read': ['"FileReadTool"', '"ReadFileTool"', '"cat "', '"readFile"'],
  'tools/edit': ['"FileEditTool"', '"old_string"', '"new_string"', '"EditFileTool"'],
  'tools/write': ['"FileWriteTool"', '"WriteFileTool"', '"createFile"'],
  'tools/glob': ['"GlobTool"', '"ListFilesTool"', '"glob"', '"minimatch"'],
  'tools/grep': ['"GrepTool"', '"ripgrep"', '"rg "', '"SearchTool"'],
  'tools/agent': ['"AgentTool"', '"Task"', '"subagent"'],
  'tools/web-fetch': ['"WebFetchTool"', '"url_fetch"'],
  'tools/web-search': ['"WebSearchTool"', '"web_search"'],
  'tools/notebook': ['"NotebookEditTool"', '"ipynb"', '"jupyter"'],
  'tools/mcp-dispatch': ['"inputSchema"', '"toolSchema"', '"toolDefinition"'],
  'tools/todo': ['"TodoWriteTool"', '"TodoReadTool"'],
  'core/agent-loop': ['"assistant"', '"user"', '"system"', '"systemPrompt"', '"messageHistory"'],
  'core/streaming': [
    '"content_block_delta"', '"message_start"', '"message_stop"',
    '"message_delta"', '"content_block_start"', '"content_block_stop"',
    '"text_delta"', '"input_json_delta"', '"stream_event"',
  ],
  'core/context-manager': [
    '"tengu_compact"', '"auto_compact"', '"compact"',
    '"contextWindow"', '"maxTokens"', '"cacheControl"',
  ],
  'core/session': ['"sessionId"', '"conversationId"', '"checkpoint"', '"resume"'],
  'core/error-handler': ['"overloaded"', '"rate_limit"', '"retryAfter"', '"APIError"'],
  'permissions/checker': [
    '"canUseTool"', '"permission"', '"allowedTools"',
    '"permissionMode"', '"alwaysAllow"',
  ],
  'permissions/sandbox': ['"sandbox"', '"bubblewrap"', '"seatbelt"', '"firejail"'],
  'auth/oauth': ['"OAuth"', '"PKCE"', '"authorization_code"', '"refresh_token"', '"code_verifier"'],
  'auth/api-key': ['"x-api-key"', '"ANTHROPIC_API_KEY"', '"apiKeyHelper"'],
  'auth/bedrock': ['"bedrock"', '"BedrockRuntime"', '"aws-region"'],
  'auth/vertex': ['"vertex"', '"vertexai"', '"google-cloud"'],
  'mcp/client': ['"McpClient"', '"mcp_client"'],
  'mcp/transport': ['"stdio"', '"sse"', '"streamable-http"', '"StdioTransport"'],
  'mcp/protocol': ['"jsonrpc"', '"tools/list"', '"tools/call"', '"resources/list"', '"mcp__"'],
  'mcp/servers': ['"mcpServers"', '"serverConfig"'],
  'config/settings': ['"settings.json"', '"userSettings"', '".claude"'],
  'config/env-vars': ['"CLAUDE_CODE_"', '"ANTHROPIC_"', '"CLAUDE_CONFIG"', '"CLAUDE_SKIP"'],
  'config/models': ['"modelId"', '"claude-sonnet"', '"claude-opus"', '"claude-haiku"'],
  'config/feature-flags': ['"featureFlag"', '"experiment"', '"rollout"'],
  'telemetry/otel': ['"opentelemetry"', '"OTEL_"', '"TraceProvider"'],
  'telemetry/datadog': ['"datadog"', '"DD_TRACE"'],
  'telemetry/events': ['"tengu_"', '"trackEvent"', '"analytics"', '"telemetryEvent"'],
  'telemetry/cost': ['"inputTokens"', '"outputTokens"', '"cacheRead"', '"cacheCreation"'],
  'ui/slash-commands': ['"/help"', '"/clear"', '"/compact"', '"/bug"', '"/init"', '"/doctor"'],
  'ui/ink-components': ['"useInput"', '"useFocus"', '"useApp"', '"inkRenderer"'],
  'ui/keybindings': ['"keybinding"', '"shortcut"', '"hotkey"'],
  'ui/terminal': ['"chalk"', '"stripAnsi"', '"ansiColor"'],
  'model-provider/anthropic': ['"anthropic"', '"claude-"', '"Anthropic"', '"messages"'],
  'model-provider/openai': ['"openai"', '"gpt-"', '"chatCompletion"'],
  'git/operations': ['"git diff"', '"git status"', '"git log"', '"git commit"'],
  'network/http': ['"Content-Type"', '"application/json"', '"Authorization"'],
};

// ── Legacy MODULE_KEYWORDS alias ───────────────────────────────────────────
// Maps old broad categories for backward compat.
const MODULE_KEYWORDS = {
  'tool-dispatch': SUBCATEGORIES['tools/mcp-dispatch'],
  'permission-system': SUBCATEGORIES['permissions/checker'],
  'mcp-client': SUBCATEGORIES['mcp/protocol'],
  'streaming-handler': SUBCATEGORIES['core/streaming'],
  'context-manager': SUBCATEGORIES['core/context-manager'],
  'agent-loop': SUBCATEGORIES['core/agent-loop'],
  'commands': SUBCATEGORIES['ui/slash-commands'],
  'telemetry': SUBCATEGORIES['telemetry/events'],
  'config': SUBCATEGORIES['config/settings'],
  'session': SUBCATEGORIES['core/session'],
  'model-provider': SUBCATEGORIES['model-provider/anthropic'],
};

module.exports = { SUBCATEGORIES, MODULE_KEYWORDS, STRING_PATTERNS };
