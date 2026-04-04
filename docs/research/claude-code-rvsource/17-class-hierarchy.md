# 17 - Class Hierarchy

Class definitions, inheritance trees, and type system extracted from `cli.js` (v2.0.62).

## Class Statistics

| Metric | Count |
|--------|-------|
| Total classes | 1,557 |
| Classes with inheritance | 956 |
| Distinct base classes | ~30 |
| Error subclasses | ~60 |

## Major Base Class Hierarchies

### 1. Error Hierarchy (`extends Error`)

60+ custom error classes. Key identified errors:

```
Error
├── Az       (generic app error)
├── BTA      (unknown)
├── D21      (unknown)
├── E70      (unknown)
├── E8       (base for BJA - tool errors)
├── EE0      (unknown)
├── GI       (API/network error base - see below)
├── Ha       (unknown)
├── JA0      (unknown)
├── KU       (unknown)
├── LyA      (unknown)
├── Mh       (unknown)
├── NE       (unknown)
├── NNA      (unknown)
├── Oh       (unknown)
├── OY0      (unknown)
├── PH0      (unknown)
├── RY0      (unknown)
├── TYA      (unknown)
├── U81      (unknown)
├── V4       (unknown)
├── VU       (unknown)
├── VX1      (unknown)
├── Va1      (unknown)
├── WF2      (unknown)
├── YH1      (unknown)
├── YV       (unknown)
├── bS       (unknown)
├── dI2      (unknown)
├── do1      (unknown)
├── el       (unknown)
├── fWA      (unknown)
├── gY       (unknown)
├── j22      (unknown)
├── lU       (unknown)
├── mB1      (unknown)
├── mn1      (unknown)
├── oB       (unknown)
├── pYA      (unknown)
├── q71      (unknown)
├── qAB      (unknown)
├── qi1      (unknown)
├── uDA      (unknown)
├── uY       (unknown)
├── vaA      (unknown)
├── vS       (unknown)
├── yl2      (unknown)
└── zQ1      (unknown)
```

### 2. GI Hierarchy (API Error Classes)

`GI extends Error` serves as the base for API and network related errors:

```
GI (extends Error) - "API Error Base"
├── CAB
├── DAB
├── EAB
├── Fx1
├── HAB
├── IAB
├── JAB
├── KAB
├── LAB
├── MAB
├── NAB
├── OAB
├── UAB
├── VAB
├── WAB
├── XAB
├── YAB
├── wAB
└── zAB
```

Evidence: `GI` is used in contexts involving HTTP status codes, retry logic, and API responses.
The string `"ModelErrorException"` (`yhB` class) extends `uU` and is associated with stop reasons.

### 3. L6 Hierarchy (Tool/Component Classes)

`L6` is likely the base tool class or a React component base:

```
L6 - "Tool/Component Base"
├── A8A      ├── IT       ├── W8A
├── AHA      ├── J8A      ├── X8A
├── B8A      ├── KT       ├── XT
├── BbA      ├── Q8A      ├── Y8A
├── Bo       ├── QbA      ├── Yo
├── EX       ├── Zb       ├── Z8A
├── G8A      ├── Zo       ├── cd
├── Go       ├── dd       ├── e4A
├── I8A      ├── eDA      ├── md
│            ├── mS       ├── oDA
│            ├── qw       ├── rDA
│            ├── sDA      ├── t4A
│            ├── tDA      ├── uS
│            └── ud
```

~35 subclasses. Given 19 built-in tools plus slash commands plus internal components,
this maps closely to the tool registry.

### 4. V9 Hierarchy (UI Component Base)

```
V9 - "UI Component"
├── CX
├── DHA
├── FHA
├── IHA
├── Io
├── KHA
├── VHA
├── WHA
├── Wo
└── wC
```

10 subclasses, likely representing distinct UI panels or widgets in the Ink terminal UI.

### 5. RX Hierarchy (Command/Handler Base)

```
RX - "Command Handler"
├── CE
├── CqA
├── EqA
├── Mq
├── XAA
├── dZA
├── jl
├── jqA
├── qqA
├── uZA
└── wqA
```

11 subclasses. Matches roughly with the number of major command categories (slash commands
that render React components like /config, /context, /cost, etc.).

### 6. oJ Hierarchy (Middleware/Processor Chain)

```
oJ - "Processor/Middleware"
├── C_2    ├── I_2    ├── O_2    ├── U_2
├── D_2    ├── J_2    ├── q_2    ├── V_2
├── E_2    ├── K_2    ├── R_2    ├── W_2
├── F_2    ├── L_2    ├── Ra0    ├── w_2
├── H_2    ├── M_2    ├── T_2    ├── z_2
│          └── N_2
```

~20 subclasses forming a processing chain, likely request/response middleware.

### 7. c90 / MIA / LIA Hierarchies (Specialized Bases)

```
c90 - "State Manager?"
├── Az2
├── Bz2
└── Qz2

MIA - "Input Schema?"
├── CK2
├── EK2
└── zK2

LIA - "Output Schema?"
├── DK2
├── FK2
├── HK2
└── VK2
```

## Key Named Classes (from string evidence)

| String Name | Context | Purpose |
|-------------|---------|---------|
| `"AgentBaseInternalState"` | State management | Internal state for agent instances |
| `"ModelErrorException"` | Error handling | API model error (extends `uU`) |
| `"McpError"` | MCP protocol | MCP communication errors |
| `"FileReadStream"` | File I/O | Streaming file reader |
| `"FileWriteStream"` | File I/O | Streaming file writer |
| `"GlobalSession"` | Session management | Global session state singleton |
| `"GlobalPreferences"` | Settings | User preferences singleton |
| `"GlobalKeyMap"` | Input handling | Keyboard shortcut mappings |
| `"GlobalRef"` | React state | Global reference container |

## Tool Registry Type (reconstructed from `XF0` class)

```typescript
class ToolRegistry {  // XF0 in minified
  toolDefinitions: Tool[];
  canUseTool: PermissionChecker;
  tools: Tool[];
  toolUseContext: ToolUseContext;
  hasErrored: boolean;
  progressAvailableResolve: Function;

  constructor(tools: Tool[], canUseTool: PermissionChecker, context: ToolUseContext) {
    this.toolDefinitions = tools;
    this.canUseTool = canUseTool;
    this.toolUseContext = context;
  }
}
```

## AppState Type (reconstructed from `Ll()`)

```typescript
interface AppState {
  settings: Settings;
  backgroundTasks: Record<string, unknown>;
  verbose: boolean;
  mainLoopModel: string | null;
  mainLoopModelForSession: string | null;
  statusLineText: string | undefined;
  showExpandedTodos: boolean;
  toolPermissionContext: ToolPermissionContext;
  agent: string | undefined;
  agentDefinitions: {
    activeAgents: AgentDefinition[];
    allAgents: AgentDefinition[];
  };
  fileHistory: {
    snapshots: Snapshot[];
    trackedFiles: Set<string>;
  };
  mcp: {
    clients: McpClient[];
    tools: Tool[];
    commands: Command[];
    resources: Record<string, Resource[]>;
  };
  plugins: {
    enabled: Plugin[];
    disabled: Plugin[];
    commands: Command[];
    agents: AgentDefinition[];
    errors: Error[];
    installationStatus: InstallationStatus;
  };
  todos: Record<string, Todo>;
  notifications: { current: Notification | null; queue: Notification[] };
  elicitation: { queue: unknown[] };
  thinkingEnabled: boolean;
  feedbackSurvey: { timeLastShown: number | null; submitCountAtLastAppearance: number | null };
  sessionHooks: Record<string, unknown>;
  inbox: { messages: unknown[] };
  promptSuggestion: { text: string | null; shownAt: number };
  queuedCommands: Command[];
  gitDiff: { stats: unknown | null; hunks: Map<string, unknown>; lastUpdated: number };
}
```

## Model Names (from `yGA` / `kGA` arrays)

```typescript
const modelNames = ["sonnet", "opus", "haiku", "sonnet[1m]"];  // yGA
const modelNamesWithInherit = [...modelNames, "inherit"];        // kGA
```

## Agent Definition Schema (from Zod validation)

```typescript
const agentDefinitionSchema = z.object({
  prompt: z.string().min(1, "Prompt cannot be empty"),
  model: z.enum(kGA).optional(),
  permissionMode: z.enum(ET).optional(),
  // ET = ["acceptEdits", "bypassPermissions", "default", "dontAsk", "plan"]
});
```
