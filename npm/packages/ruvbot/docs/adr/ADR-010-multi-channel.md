# ADR-010: Multi-Channel Integration

## Status
Accepted

## Date
2026-01-27

## Context

Clawdbot supports multiple messaging channels:
- Slack, Discord, Telegram, Signal, WhatsApp, Line, iMessage
- Web, CLI, API interfaces

RuvBot must match and exceed with:
- All Clawdbot channels
- Multi-tenant channel isolation
- Unified message handling

## Decision

### Channel Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RuvBot Channel Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  Channel Adapters (8+)                                           │
│    ├─ SlackAdapter       : @slack/bolt                          │
│    ├─ DiscordAdapter     : discord.js                           │
│    ├─ TelegramAdapter    : telegraf                             │
│    ├─ SignalAdapter      : signal-client                        │
│    ├─ WhatsAppAdapter    : baileys                              │
│    ├─ LineAdapter        : @line/bot-sdk                        │
│    ├─ WebAdapter         : WebSocket + REST                     │
│    └─ CLIAdapter         : readline + terminal                  │
├─────────────────────────────────────────────────────────────────┤
│  Message Normalization                                           │
│    └─ Unified Message format                                    │
│    └─ Attachment handling                                       │
│    └─ Thread/reply context                                      │
├─────────────────────────────────────────────────────────────────┤
│  Multi-Tenant Isolation                                          │
│    └─ Channel credentials per tenant                            │
│    └─ Namespace isolation                                       │
│    └─ Rate limiting per tenant                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Unified Message Interface

```typescript
interface UnifiedMessage {
  id: string;
  channelId: string;
  channelType: ChannelType;
  tenantId: string;
  userId: string;
  content: string;
  attachments?: Attachment[];
  threadId?: string;
  replyTo?: string;
  timestamp: Date;
  metadata: Record<string, unknown>;
}

type ChannelType =
  | 'slack' | 'discord' | 'telegram'
  | 'signal' | 'whatsapp' | 'line'
  | 'imessage' | 'web' | 'api' | 'cli';
```

### Channel Registry

```typescript
interface ChannelRegistry {
  register(adapter: ChannelAdapter): void;
  get(type: ChannelType): ChannelAdapter | undefined;
  start(): Promise<void>;
  stop(): Promise<void>;
  broadcast(message: string, filter?: ChannelFilter): Promise<void>;
}
```

## Consequences

### Positive
- Unified message handling across all channels
- Multi-tenant channel isolation
- Easy to add new channels

### Negative
- Complexity of maintaining multiple integrations
- Different channel capabilities

### RuvBot Advantages over Clawdbot
- Multi-tenant channel credentials
- Channel-specific rate limiting
- Cross-channel message routing
