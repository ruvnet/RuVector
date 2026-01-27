/**
 * RuvBot - Self-learning AI Assistant with RuVector Backend
 *
 * Main entry point for the RuvBot framework.
 * Combines Clawdbot-style personal AI with RuVector's WASM vector operations.
 */

import { EventEmitter } from 'eventemitter3';
import pino from 'pino';
import { v4 as uuidv4 } from 'uuid';

import { ConfigManager, type BotConfig } from './core/BotConfig.js';
import { BotStateManager, type BotStatus } from './core/BotState.js';
import type {
  Agent,
  AgentConfig,
  Session,
  Message,
  BotEvent,
  BotEventType,
  Result,
  ok,
  err,
} from './core/types.js';
import { RuvBotError, ConfigurationError, InitializationError } from './core/errors.js';

type BotState = BotStatus;

// ============================================================================
// Types
// ============================================================================

export interface RuvBotOptions {
  config?: Partial<BotConfig>;
  configPath?: string;
  autoStart?: boolean;
}

export interface RuvBotEvents {
  ready: () => void;
  shutdown: () => void;
  error: (error: Error) => void;
  message: (message: Message, session: Session) => void;
  'agent:spawn': (agent: Agent) => void;
  'agent:stop': (agentId: string) => void;
  'session:create': (session: Session) => void;
  'session:end': (sessionId: string) => void;
  'memory:store': (entryId: string) => void;
  'skill:invoke': (skillName: string, params: Record<string, unknown>) => void;
}

// ============================================================================
// RuvBot Main Class
// ============================================================================

export class RuvBot extends EventEmitter<RuvBotEvents> {
  private readonly id: string;
  private readonly configManager: ConfigManager;
  private readonly stateManager: BotStateManager;
  private readonly logger: pino.Logger;

  private agents: Map<string, Agent> = new Map();
  private sessions: Map<string, Session> = new Map();
  private isRunning: boolean = false;
  private startTime?: Date;

  constructor(options: RuvBotOptions = {}) {
    super();

    this.id = uuidv4();

    // Initialize configuration
    if (options.config) {
      this.configManager = new ConfigManager(options.config);
    } else {
      this.configManager = ConfigManager.fromEnv();
    }

    // Validate configuration
    const validation = this.configManager.validate();
    if (!validation.valid) {
      throw new ConfigurationError(
        `Invalid configuration: ${validation.errors.join(', ')}`
      );
    }

    // Initialize logger
    const config = this.configManager.getConfig();
    this.logger = pino({
      level: config.logging.level,
      transport: config.logging.pretty
        ? { target: 'pino-pretty', options: { colorize: true } }
        : undefined,
    });

    // Initialize state manager
    this.stateManager = new BotStateManager();

    this.logger.info({ botId: this.id }, 'RuvBot instance created');

    // Auto-start if requested
    if (options.autoStart) {
      this.start().catch((error) => {
        this.logger.error({ error }, 'Auto-start failed');
        this.emit('error', error);
      });
    }
  }

  // ==========================================================================
  // Lifecycle Methods
  // ==========================================================================

  /**
   * Start the bot and all configured services
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      this.logger.warn('RuvBot is already running');
      return;
    }

    this.logger.info('Starting RuvBot...');
    this.stateManager.setStatus('starting');

    try {
      const config = this.configManager.getConfig();

      // Initialize core services
      await this.initializeServices();

      // Start integrations
      await this.startIntegrations(config);

      // Start API server if enabled
      if (config.api.enabled) {
        await this.startApiServer(config);
      }

      // Mark as running
      this.isRunning = true;
      this.startTime = new Date();
      this.stateManager.setStatus('running');

      this.logger.info(
        { botId: this.id, name: config.name },
        'RuvBot started successfully'
      );
      this.emit('ready');
    } catch (error) {
      this.stateManager.setStatus('error');
      this.logger.error({ error }, 'Failed to start RuvBot');
      throw new InitializationError(
        `Failed to start RuvBot: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Stop the bot and cleanup resources
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      this.logger.warn('RuvBot is not running');
      return;
    }

    this.logger.info('Stopping RuvBot...');
    this.stateManager.setStatus('stopping');

    try {
      // Stop all agents
      for (const [agentId] of this.agents) {
        await this.stopAgent(agentId);
      }

      // End all sessions
      for (const [sessionId] of this.sessions) {
        await this.endSession(sessionId);
      }

      // Stop integrations
      await this.stopIntegrations();

      // Stop API server
      await this.stopApiServer();

      this.isRunning = false;
      this.stateManager.setStatus('stopped');

      this.logger.info('RuvBot stopped successfully');
      this.emit('shutdown');
    } catch (error) {
      this.stateManager.setStatus('error');
      this.logger.error({ error }, 'Error during shutdown');
      throw error;
    }
  }

  // ==========================================================================
  // Agent Management
  // ==========================================================================

  /**
   * Spawn a new agent with the given configuration
   */
  async spawnAgent(config: AgentConfig): Promise<Agent> {
    const agentId = config.id || uuidv4();

    if (this.agents.has(agentId)) {
      throw new RuvBotError(`Agent with ID ${agentId} already exists`, 'AGENT_EXISTS');
    }

    const agent: Agent = {
      id: agentId,
      name: config.name,
      config,
      status: 'idle',
      createdAt: new Date(),
      lastActiveAt: new Date(),
    };

    this.agents.set(agentId, agent);
    this.logger.info({ agentId, name: config.name }, 'Agent spawned');
    this.emit('agent:spawn', agent);

    return agent;
  }

  /**
   * Stop an agent by ID
   */
  async stopAgent(agentId: string): Promise<void> {
    const agent = this.agents.get(agentId);
    if (!agent) {
      throw new RuvBotError(`Agent with ID ${agentId} not found`, 'AGENT_NOT_FOUND');
    }

    // End all sessions for this agent
    for (const [sessionId, session] of this.sessions) {
      if (session.agentId === agentId) {
        await this.endSession(sessionId);
      }
    }

    this.agents.delete(agentId);
    this.logger.info({ agentId }, 'Agent stopped');
    this.emit('agent:stop', agentId);
  }

  /**
   * Get an agent by ID
   */
  getAgent(agentId: string): Agent | undefined {
    return this.agents.get(agentId);
  }

  /**
   * List all active agents
   */
  listAgents(): Agent[] {
    return Array.from(this.agents.values());
  }

  // ==========================================================================
  // Session Management
  // ==========================================================================

  /**
   * Create a new session for an agent
   */
  async createSession(
    agentId: string,
    options: {
      userId?: string;
      channelId?: string;
      platform?: Session['platform'];
      metadata?: Record<string, unknown>;
    } = {}
  ): Promise<Session> {
    const agent = this.agents.get(agentId);
    if (!agent) {
      throw new RuvBotError(`Agent with ID ${agentId} not found`, 'AGENT_NOT_FOUND');
    }

    const sessionId = uuidv4();
    const config = this.configManager.getConfig();

    const session: Session = {
      id: sessionId,
      agentId,
      userId: options.userId,
      channelId: options.channelId,
      platform: options.platform || 'api',
      messages: [],
      context: {
        topics: [],
        entities: [],
      },
      metadata: options.metadata || {},
      createdAt: new Date(),
      updatedAt: new Date(),
      expiresAt: new Date(Date.now() + config.session.defaultTTL),
    };

    this.sessions.set(sessionId, session);
    this.logger.info({ sessionId, agentId }, 'Session created');
    this.emit('session:create', session);

    return session;
  }

  /**
   * End a session by ID
   */
  async endSession(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new RuvBotError(`Session with ID ${sessionId} not found`, 'SESSION_NOT_FOUND');
    }

    this.sessions.delete(sessionId);
    this.logger.info({ sessionId }, 'Session ended');
    this.emit('session:end', sessionId);
  }

  /**
   * Get a session by ID
   */
  getSession(sessionId: string): Session | undefined {
    return this.sessions.get(sessionId);
  }

  /**
   * List all active sessions
   */
  listSessions(): Session[] {
    return Array.from(this.sessions.values());
  }

  // ==========================================================================
  // Message Handling
  // ==========================================================================

  /**
   * Send a message to an agent in a session
   */
  async chat(
    sessionId: string,
    content: string,
    options: {
      userId?: string;
      attachments?: Message['attachments'];
      metadata?: Message['metadata'];
    } = {}
  ): Promise<Message> {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new RuvBotError(`Session with ID ${sessionId} not found`, 'SESSION_NOT_FOUND');
    }

    const agent = this.agents.get(session.agentId);
    if (!agent) {
      throw new RuvBotError(`Agent with ID ${session.agentId} not found`, 'AGENT_NOT_FOUND');
    }

    // Create user message
    const userMessage: Message = {
      id: uuidv4(),
      sessionId,
      role: 'user',
      content,
      attachments: options.attachments,
      metadata: options.metadata,
      createdAt: new Date(),
    };

    // Add to session
    session.messages.push(userMessage);
    session.updatedAt = new Date();

    // Update agent status
    agent.status = 'processing';
    agent.lastActiveAt = new Date();

    this.logger.debug({ sessionId, messageId: userMessage.id }, 'User message received');
    this.emit('message', userMessage, session);

    try {
      // Generate response (placeholder for LLM integration)
      const responseContent = await this.generateResponse(session, agent, content);

      // Create assistant message
      const assistantMessage: Message = {
        id: uuidv4(),
        sessionId,
        role: 'assistant',
        content: responseContent,
        createdAt: new Date(),
      };

      session.messages.push(assistantMessage);
      session.updatedAt = new Date();
      agent.status = 'idle';

      this.logger.debug(
        { sessionId, messageId: assistantMessage.id },
        'Assistant response generated'
      );
      this.emit('message', assistantMessage, session);

      return assistantMessage;
    } catch (error) {
      agent.status = 'error';
      throw error;
    }
  }

  // ==========================================================================
  // Status & Info
  // ==========================================================================

  /**
   * Get the current bot status
   */
  getStatus(): {
    id: string;
    name: string;
    state: BotState;
    isRunning: boolean;
    uptime?: number;
    agents: number;
    sessions: number;
  } {
    const config = this.configManager.getConfig();

    return {
      id: this.id,
      name: config.name,
      state: this.stateManager.getStatus(),
      isRunning: this.isRunning,
      uptime: this.startTime
        ? Date.now() - this.startTime.getTime()
        : undefined,
      agents: this.agents.size,
      sessions: this.sessions.size,
    };
  }

  /**
   * Get the current configuration
   */
  getConfig(): Readonly<BotConfig> {
    return this.configManager.getConfig();
  }

  // ==========================================================================
  // Private Methods
  // ==========================================================================

  private async initializeServices(): Promise<void> {
    this.logger.debug('Initializing core services...');
    // TODO: Initialize memory manager, skill registry, etc.
  }

  private async startIntegrations(config: BotConfig): Promise<void> {
    this.logger.debug('Starting integrations...');

    if (config.slack.enabled) {
      this.logger.info('Slack integration enabled');
      // TODO: Initialize Slack adapter
    }

    if (config.discord.enabled) {
      this.logger.info('Discord integration enabled');
      // TODO: Initialize Discord adapter
    }

    if (config.webhook.enabled) {
      this.logger.info('Webhook integration enabled');
      // TODO: Initialize webhook handler
    }
  }

  private async stopIntegrations(): Promise<void> {
    this.logger.debug('Stopping integrations...');
    // TODO: Stop all integration adapters
  }

  private async startApiServer(config: BotConfig): Promise<void> {
    this.logger.info(
      { port: config.api.port, host: config.api.host },
      'Starting API server...'
    );
    // TODO: Initialize Fastify server
  }

  private async stopApiServer(): Promise<void> {
    this.logger.debug('Stopping API server...');
    // TODO: Stop Fastify server
  }

  private async generateResponse(
    session: Session,
    agent: Agent,
    userMessage: string
  ): Promise<string> {
    // Placeholder response generation
    // TODO: Integrate with LLM orchestrator
    return `[RuvBot] Received: "${userMessage}"`;
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new RuvBot instance
 */
export function createRuvBot(options?: RuvBotOptions): RuvBot {
  return new RuvBot(options);
}

/**
 * Create a RuvBot instance from environment variables
 */
export function createRuvBotFromEnv(): RuvBot {
  return new RuvBot();
}

export default RuvBot;
