#!/usr/bin/env node

/**
 * RuVector MCP Server
 *
 * Model Context Protocol server for RuVector hooks
 * Provides self-learning intelligence tools for Claude Code
 *
 * Usage:
 *   npx ruvector mcp start
 *   claude mcp add ruvector npx ruvector mcp start
 */

const { Server } = require('@modelcontextprotocol/sdk/server/index.js');
const { StdioServerTransport } = require('@modelcontextprotocol/sdk/server/stdio.js');
const {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} = require('@modelcontextprotocol/sdk/types.js');
const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');

// Try to load the full IntelligenceEngine
let IntelligenceEngine = null;
let engineAvailable = false;

try {
  const core = require('../dist/core/intelligence-engine.js');
  IntelligenceEngine = core.IntelligenceEngine || core.default;
  engineAvailable = true;
} catch (e) {
  // IntelligenceEngine not available
}

// Intelligence class with full RuVector stack support
class Intelligence {
  constructor() {
    this.intelPath = this.getIntelPath();
    this.data = this.load();
    this.engine = null;

    // Initialize full engine if available
    if (engineAvailable && IntelligenceEngine) {
      try {
        this.engine = new IntelligenceEngine({
          embeddingDim: 256,
          maxMemories: 100000,
          enableSona: true,
          enableAttention: true,
        });
        // Import existing data
        if (this.data) {
          this.engine.import(this.convertLegacyData(this.data), true);
        }
      } catch (e) {
        this.engine = null;
      }
    }
  }

  convertLegacyData(data) {
    const converted = { memories: [], routingPatterns: {}, errorPatterns: {}, coEditPatterns: {} };
    if (data.memories) {
      converted.memories = data.memories.map(m => ({
        id: m.id || `mem-${Date.now()}`,
        content: m.content,
        type: m.type || 'general',
        embedding: m.embedding || [],
        created: m.created || new Date().toISOString(),
        accessed: 0,
      }));
    }
    if (data.patterns) {
      for (const [key, value] of Object.entries(data.patterns)) {
        const [state, action] = key.split('|');
        if (state && action) {
          if (!converted.routingPatterns[state]) converted.routingPatterns[state] = {};
          converted.routingPatterns[state][action] = value.q_value || value || 0.5;
        }
      }
    }
    return converted;
  }

  getIntelPath() {
    const projectPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    const homePath = path.join(require('os').homedir(), '.ruvector', 'intelligence.json');
    if (fs.existsSync(path.dirname(projectPath))) return projectPath;
    if (fs.existsSync(path.join(process.cwd(), '.claude'))) return projectPath;
    if (fs.existsSync(homePath)) return homePath;
    return projectPath;
  }

  load() {
    try {
      if (fs.existsSync(this.intelPath)) {
        return JSON.parse(fs.readFileSync(this.intelPath, 'utf-8'));
      }
    } catch {}
    return { patterns: {}, memories: [], trajectories: [], errors: {}, agents: {}, edges: [] };
  }

  save() {
    const dir = path.dirname(this.intelPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    // Export engine data if available
    if (this.engine) {
      try {
        const engineData = this.engine.export();
        this.data.engineStats = engineData.stats;
      } catch {}
    }

    fs.writeFileSync(this.intelPath, JSON.stringify(this.data, null, 2));
  }

  stats() {
    const baseStats = {
      total_patterns: Object.keys(this.data.patterns || {}).length,
      total_memories: (this.data.memories || []).length,
      total_trajectories: (this.data.trajectories || []).length,
      total_errors: Object.keys(this.data.errors || {}).length
    };

    if (this.engine) {
      try {
        const engineStats = this.engine.getStats();
        return {
          ...baseStats,
          engineEnabled: true,
          sonaEnabled: engineStats.sonaEnabled,
          attentionEnabled: engineStats.attentionEnabled,
          embeddingDim: engineStats.memoryDimensions,
          totalMemories: engineStats.totalMemories,
          totalEpisodes: engineStats.totalEpisodes,
          trajectoriesRecorded: engineStats.trajectoriesRecorded,
          patternsLearned: engineStats.patternsLearned,
          microLoraUpdates: engineStats.microLoraUpdates,
          ewcConsolidations: engineStats.ewcConsolidations,
        };
      } catch {}
    }

    return { ...baseStats, engineEnabled: false };
  }

  embed(text) {
    if (this.engine) {
      try {
        return this.engine.embed(text);
      } catch {}
    }
    // Fallback: 64-dim hash
    const embedding = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
      const idx = (text.charCodeAt(i) + i * 7) % 64;
      embedding[idx] += 1.0;
    }
    const norm = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
    if (norm > 0) for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
    return embedding;
  }

  similarity(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
    const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
    return normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
  }

  async remember(content, type = 'general') {
    // Use engine if available (VectorDB storage)
    if (this.engine) {
      try {
        const entry = await this.engine.remember(content, type);
        // Also store in legacy format
        this.data.memories = this.data.memories || [];
        this.data.memories.push({ content, type, created: new Date().toISOString(), embedding: entry.embedding });
        this.save();
        return { stored: true, total: this.data.memories.length, engineStored: true };
      } catch {}
    }

    // Fallback
    this.data.memories = this.data.memories || [];
    this.data.memories.push({ content, type, created: new Date().toISOString(), embedding: this.embed(content) });
    this.save();
    return { stored: true, total: this.data.memories.length };
  }

  async recall(query, topK = 5) {
    // Use engine if available (HNSW search - 150x faster)
    if (this.engine) {
      try {
        const results = await this.engine.recall(query, topK);
        return results.map(r => ({
          content: r.content,
          type: r.type,
          score: r.score || 0,
          created: r.created,
          engineResult: true
        }));
      } catch {}
    }

    // Fallback: brute-force
    const queryEmbed = this.embed(query);
    const scored = (this.data.memories || []).map((m, i) => ({
      ...m,
      index: i,
      score: this.similarity(queryEmbed, m.embedding)
    }));
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
  }

  async route(task, file = null) {
    // Use engine if available (SONA-enhanced routing)
    if (this.engine) {
      try {
        const result = await this.engine.route(task, file);
        return {
          agent: result.agent,
          confidence: result.confidence,
          reason: result.reason,
          alternates: result.alternates,
          sonaPatterns: result.patterns?.length || 0,
          engineRouted: true
        };
      } catch {}
    }

    // Fallback
    const ext = file ? path.extname(file) : '';
    const state = `edit:${ext || 'unknown'}`;
    const actions = this.data.patterns[state] || {};

    const defaults = {
      '.rs': 'rust-developer',
      '.ts': 'typescript-developer',
      '.tsx': 'react-developer',
      '.js': 'javascript-developer',
      '.jsx': 'react-developer',
      '.py': 'python-developer',
      '.go': 'go-developer',
      '.sql': 'database-specialist',
      '.md': 'documentation-specialist'
    };

    let bestAgent = defaults[ext] || 'coder';
    let bestScore = 0.5;

    for (const [agent, score] of Object.entries(actions)) {
      if (score > bestScore) {
        bestAgent = agent;
        bestScore = score;
      }
    }

    return {
      agent: bestAgent,
      confidence: Math.min(bestScore, 1.0),
      reason: Object.keys(actions).length > 0 ? 'learned from patterns' : 'default mapping'
    };
  }

  getCapabilities() {
    if (!this.engine) {
      return { engine: false, vectorDb: false, sona: false, attention: false, embeddingDim: 64 };
    }
    try {
      const stats = this.engine.getStats();
      return {
        engine: true,
        vectorDb: true,
        sona: stats.sonaEnabled,
        attention: stats.attentionEnabled,
        embeddingDim: stats.memoryDimensions,
      };
    } catch {
      return { engine: true, vectorDb: false, sona: false, attention: false, embeddingDim: 256 };
    }
  }
}

// Create MCP server
const server = new Server(
  {
    name: 'ruvector',
    version: '0.1.53',
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

const intel = new Intelligence();

// Define tools
const TOOLS = [
  {
    name: 'hooks_stats',
    description: 'Get RuVector intelligence statistics including learned patterns, memories, and trajectories',
    inputSchema: {
      type: 'object',
      properties: {},
      required: []
    }
  },
  {
    name: 'hooks_route',
    description: 'Route a task to the best agent based on learned patterns',
    inputSchema: {
      type: 'object',
      properties: {
        task: { type: 'string', description: 'Task description' },
        file: { type: 'string', description: 'File path (optional)' }
      },
      required: ['task']
    }
  },
  {
    name: 'hooks_remember',
    description: 'Store context in vector memory for later recall',
    inputSchema: {
      type: 'object',
      properties: {
        content: { type: 'string', description: 'Content to remember' },
        type: { type: 'string', description: 'Memory type (project, code, decision, context)', default: 'general' }
      },
      required: ['content']
    }
  },
  {
    name: 'hooks_recall',
    description: 'Search vector memory for relevant context',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Search query' },
        top_k: { type: 'number', description: 'Number of results', default: 5 }
      },
      required: ['query']
    }
  },
  {
    name: 'hooks_init',
    description: 'Initialize RuVector hooks in the current project',
    inputSchema: {
      type: 'object',
      properties: {
        pretrain: { type: 'boolean', description: 'Run pretrain after init', default: false },
        build_agents: { type: 'string', description: 'Focus for agent generation (quality, speed, security, testing, fullstack)' },
        force: { type: 'boolean', description: 'Force overwrite existing settings', default: false }
      },
      required: []
    }
  },
  {
    name: 'hooks_pretrain',
    description: 'Pretrain intelligence by analyzing the repository structure and git history',
    inputSchema: {
      type: 'object',
      properties: {
        depth: { type: 'number', description: 'Git history depth to analyze', default: 100 },
        skip_git: { type: 'boolean', description: 'Skip git history analysis', default: false },
        verbose: { type: 'boolean', description: 'Show detailed progress', default: false }
      },
      required: []
    }
  },
  {
    name: 'hooks_build_agents',
    description: 'Generate optimized agent configurations based on repository analysis',
    inputSchema: {
      type: 'object',
      properties: {
        focus: {
          type: 'string',
          description: 'Focus type for agent generation',
          enum: ['quality', 'speed', 'security', 'testing', 'fullstack'],
          default: 'quality'
        },
        include_prompts: { type: 'boolean', description: 'Include system prompts in agent configs', default: true }
      },
      required: []
    }
  },
  {
    name: 'hooks_verify',
    description: 'Verify that hooks are configured correctly',
    inputSchema: {
      type: 'object',
      properties: {},
      required: []
    }
  },
  {
    name: 'hooks_doctor',
    description: 'Diagnose and optionally fix setup issues',
    inputSchema: {
      type: 'object',
      properties: {
        fix: { type: 'boolean', description: 'Automatically fix issues', default: false }
      },
      required: []
    }
  },
  {
    name: 'hooks_export',
    description: 'Export intelligence data for backup',
    inputSchema: {
      type: 'object',
      properties: {
        include_all: { type: 'boolean', description: 'Include all data (patterns, memories, trajectories)', default: false }
      },
      required: []
    }
  },
  {
    name: 'hooks_capabilities',
    description: 'Get RuVector engine capabilities (VectorDB, SONA, Attention)',
    inputSchema: {
      type: 'object',
      properties: {},
      required: []
    }
  },
  {
    name: 'hooks_import',
    description: 'Import intelligence data from backup file',
    inputSchema: {
      type: 'object',
      properties: {
        data: { type: 'object', description: 'Exported data object to import' },
        merge: { type: 'boolean', description: 'Merge with existing data', default: true }
      },
      required: ['data']
    }
  },
  {
    name: 'hooks_swarm_recommend',
    description: 'Get agent recommendation for a task type using learned patterns',
    inputSchema: {
      type: 'object',
      properties: {
        task_type: { type: 'string', description: 'Type of task (research, code, test, review, debug, etc.)' },
        file: { type: 'string', description: 'Optional file path for context' }
      },
      required: ['task_type']
    }
  },
  {
    name: 'hooks_suggest_context',
    description: 'Get relevant context suggestions for the current task',
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Current task or query' },
        top_k: { type: 'number', description: 'Number of suggestions', default: 5 }
      },
      required: []
    }
  },
  {
    name: 'hooks_trajectory_begin',
    description: 'Begin tracking a new execution trajectory',
    inputSchema: {
      type: 'object',
      properties: {
        context: { type: 'string', description: 'Task or operation context' },
        agent: { type: 'string', description: 'Agent performing the task' }
      },
      required: ['context']
    }
  },
  {
    name: 'hooks_trajectory_step',
    description: 'Add a step to the current trajectory',
    inputSchema: {
      type: 'object',
      properties: {
        action: { type: 'string', description: 'Action taken' },
        result: { type: 'string', description: 'Result of action' },
        reward: { type: 'number', description: 'Reward signal (0-1)', default: 0.5 }
      },
      required: ['action']
    }
  },
  {
    name: 'hooks_trajectory_end',
    description: 'End the current trajectory with a quality score',
    inputSchema: {
      type: 'object',
      properties: {
        success: { type: 'boolean', description: 'Whether the task succeeded' },
        quality: { type: 'number', description: 'Quality score (0-1)', default: 0.5 }
      },
      required: []
    }
  },
  {
    name: 'hooks_coedit_record',
    description: 'Record co-edit pattern (files edited together)',
    inputSchema: {
      type: 'object',
      properties: {
        primary_file: { type: 'string', description: 'Primary file being edited' },
        related_files: { type: 'array', items: { type: 'string' }, description: 'Related files edited together' }
      },
      required: ['primary_file', 'related_files']
    }
  },
  {
    name: 'hooks_coedit_suggest',
    description: 'Get suggested related files based on co-edit patterns',
    inputSchema: {
      type: 'object',
      properties: {
        file: { type: 'string', description: 'Current file' },
        top_k: { type: 'number', description: 'Number of suggestions', default: 5 }
      },
      required: ['file']
    }
  },
  {
    name: 'hooks_error_record',
    description: 'Record an error and its fix for learning',
    inputSchema: {
      type: 'object',
      properties: {
        error: { type: 'string', description: 'Error message or code' },
        fix: { type: 'string', description: 'Fix that resolved the error' },
        file: { type: 'string', description: 'File where error occurred' }
      },
      required: ['error', 'fix']
    }
  },
  {
    name: 'hooks_error_suggest',
    description: 'Get suggested fixes for an error based on learned patterns',
    inputSchema: {
      type: 'object',
      properties: {
        error: { type: 'string', description: 'Error message or code' }
      },
      required: ['error']
    }
  },
  {
    name: 'hooks_force_learn',
    description: 'Force an immediate learning cycle',
    inputSchema: {
      type: 'object',
      properties: {},
      required: []
    }
  }
];

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: TOOLS };
});

// Call tool handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'hooks_stats': {
        const stats = intel.stats();
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              stats,
              intel_path: intel.intelPath
            }, null, 2)
          }]
        };
      }

      case 'hooks_route': {
        const result = await intel.route(args.task, args.file);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              task: args.task,
              file: args.file,
              ...result
            }, null, 2)
          }]
        };
      }

      case 'hooks_remember': {
        const result = await intel.remember(args.content, args.type || 'general');
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              ...result
            }, null, 2)
          }]
        };
      }

      case 'hooks_recall': {
        const results = await intel.recall(args.query, args.top_k || 5);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              query: args.query,
              results: results.map(r => ({
                content: r.content,
                type: r.type,
                score: typeof r.score === 'number' ? r.score.toFixed(3) : r.score,
                created: r.created,
                engineResult: r.engineResult || false
              }))
            }, null, 2)
          }]
        };
      }

      case 'hooks_init': {
        let cmd = 'npx ruvector hooks init';
        if (args.force) cmd += ' --force';
        if (args.pretrain) cmd += ' --pretrain';
        if (args.build_agents) cmd += ` --build-agents ${args.build_agents}`;

        try {
          const output = execSync(cmd, { encoding: 'utf-8', timeout: 60000 });
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: true, output }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_pretrain': {
        let cmd = 'npx ruvector hooks pretrain';
        if (args.depth) cmd += ` --depth ${args.depth}`;
        if (args.skip_git) cmd += ' --skip-git';
        if (args.verbose) cmd += ' --verbose';

        try {
          const output = execSync(cmd, { encoding: 'utf-8', timeout: 120000 });
          // Reload intelligence after pretrain
          intel.data = intel.load();
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({
                success: true,
                output,
                new_stats: intel.stats()
              }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_build_agents': {
        let cmd = 'npx ruvector hooks build-agents';
        if (args.focus) cmd += ` --focus ${args.focus}`;
        if (args.include_prompts) cmd += ' --include-prompts';

        try {
          const output = execSync(cmd, { encoding: 'utf-8', timeout: 30000 });
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: true, output }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_verify': {
        try {
          const output = execSync('npx ruvector hooks verify', { encoding: 'utf-8', timeout: 15000 });
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: true, output }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message, output: e.stdout }, null, 2)
            }]
          };
        }
      }

      case 'hooks_doctor': {
        let cmd = 'npx ruvector hooks doctor';
        if (args.fix) cmd += ' --fix';

        try {
          const output = execSync(cmd, { encoding: 'utf-8', timeout: 15000 });
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: true, output }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_export': {
        const exportData = {
          version: '2.0',
          exported_at: new Date().toISOString(),
          patterns: intel.data.patterns || {},
          memories: args.include_all ? (intel.data.memories || []) : [],
          trajectories: args.include_all ? (intel.data.trajectories || []) : [],
          errors: intel.data.errors || {},
          stats: intel.stats(),
          capabilities: intel.getCapabilities()
        };
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, data: exportData }, null, 2)
          }]
        };
      }

      case 'hooks_capabilities': {
        const capabilities = intel.getCapabilities();
        const stats = intel.stats();
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              capabilities,
              features: {
                vectorDb: capabilities.vectorDb ? 'HNSW indexing (150x faster search)' : 'Brute-force fallback',
                sona: capabilities.sona ? 'Micro-LoRA + Base-LoRA + EWC++' : 'Q-learning fallback',
                attention: capabilities.attention ? 'Self-attention embeddings' : 'Hash embeddings',
                embeddingDim: capabilities.embeddingDim,
              },
              stats: {
                totalMemories: stats.totalMemories || stats.total_memories,
                trajectoriesRecorded: stats.trajectoriesRecorded || 0,
                patternsLearned: stats.patternsLearned || stats.total_patterns,
                microLoraUpdates: stats.microLoraUpdates || 0,
                ewcConsolidations: stats.ewcConsolidations || 0,
              }
            }, null, 2)
          }]
        };
      }

      case 'hooks_import': {
        try {
          const data = args.data;
          const merge = args.merge !== false;

          if (data.patterns) {
            if (merge) {
              Object.assign(intel.data.patterns, data.patterns);
            } else {
              intel.data.patterns = data.patterns;
            }
          }
          if (data.memories) {
            if (merge) {
              intel.data.memories = [...(intel.data.memories || []), ...data.memories];
            } else {
              intel.data.memories = data.memories;
            }
          }
          if (data.errors) {
            if (merge) {
              Object.assign(intel.data.errors, data.errors);
            } else {
              intel.data.errors = data.errors;
            }
          }
          intel.save();

          return {
            content: [{
              type: 'text',
              text: JSON.stringify({
                success: true,
                message: `Imported ${Object.keys(data.patterns || {}).length} patterns, ${(data.memories || []).length} memories`,
                merge
              }, null, 2)
            }]
          };
        } catch (e) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: e.message }, null, 2)
            }]
          };
        }
      }

      case 'hooks_swarm_recommend': {
        const taskType = args.task_type || '';
        const file = args.file || '';

        // Map task types to recommended agents
        const taskAgentMap = {
          research: ['researcher', 'analyst', 'explorer'],
          code: ['coder', 'backend-dev', 'sparc-coder'],
          test: ['tester', 'tdd-london-swarm', 'production-validator'],
          review: ['reviewer', 'code-analyzer', 'analyst'],
          debug: ['coder', 'tester', 'analyst'],
          refactor: ['code-analyzer', 'reviewer', 'architect'],
          document: ['documenter', 'api-docs', 'researcher'],
          security: ['security-manager', 'reviewer', 'code-analyzer'],
          performance: ['perf-analyzer', 'performance-benchmarker', 'optimizer'],
          architecture: ['system-architect', 'architect', 'planner']
        };

        // Get learned route if file provided
        let learnedAgent = null;
        if (file) {
          const route = await intel.route({ task: taskType, file });
          learnedAgent = route?.agent;
        }

        const recommendations = taskAgentMap[taskType.toLowerCase()] || ['coder', 'researcher', 'analyst'];

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              task_type: taskType,
              recommendations,
              learned_agent: learnedAgent,
              suggested: learnedAgent || recommendations[0]
            }, null, 2)
          }]
        };
      }

      case 'hooks_suggest_context': {
        const query = args.query || '';
        const topK = args.top_k || 5;

        // Get relevant memories
        const memories = await intel.recall(query, topK);

        // Get recent patterns
        const recentPatterns = Object.entries(intel.data.patterns || {})
          .slice(0, topK)
          .map(([state, actions]) => ({ state, topAction: Object.keys(actions)[0] }));

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              query,
              memories: memories.map(m => ({ content: m.content, type: m.type, score: m.score })),
              patterns: recentPatterns
            }, null, 2)
          }]
        };
      }

      case 'hooks_trajectory_begin': {
        const context = args.context;
        const agent = args.agent || 'unknown';

        // Store trajectory start in intel
        if (!intel.data.activeTrajectories) intel.data.activeTrajectories = {};
        const trajId = `traj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        intel.data.activeTrajectories[trajId] = {
          id: trajId,
          context,
          agent,
          steps: [],
          startTime: Date.now()
        };

        // Also use engine if available
        if (intel.engine) {
          try {
            intel.engine.beginTrajectory(context);
          } catch (e) { /* fallback to manual */ }
        }

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, trajectory_id: trajId, context, agent }, null, 2)
          }]
        };
      }

      case 'hooks_trajectory_step': {
        const action = args.action;
        const result = args.result || '';
        const reward = args.reward || 0.5;

        // Add to most recent trajectory
        const trajectories = intel.data.activeTrajectories || {};
        const trajIds = Object.keys(trajectories);
        if (trajIds.length === 0) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: 'No active trajectory. Call hooks_trajectory_begin first.' }, null, 2)
            }]
          };
        }

        const latestTrajId = trajIds[trajIds.length - 1];
        trajectories[latestTrajId].steps.push({ action, result, reward, time: Date.now() });

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, trajectory_id: latestTrajId, step: trajectories[latestTrajId].steps.length }, null, 2)
          }]
        };
      }

      case 'hooks_trajectory_end': {
        const success = args.success !== false;
        const quality = args.quality || (success ? 0.8 : 0.2);

        const trajectories = intel.data.activeTrajectories || {};
        const trajIds = Object.keys(trajectories);
        if (trajIds.length === 0) {
          return {
            content: [{
              type: 'text',
              text: JSON.stringify({ success: false, error: 'No active trajectory.' }, null, 2)
            }]
          };
        }

        const latestTrajId = trajIds[trajIds.length - 1];
        const traj = trajectories[latestTrajId];
        traj.endTime = Date.now();
        traj.quality = quality;
        traj.success = success;

        // Move to completed trajectories
        if (!intel.data.trajectories) intel.data.trajectories = [];
        intel.data.trajectories.push(traj);
        delete trajectories[latestTrajId];

        // Learn from trajectory
        if (intel.engine && traj.steps.length > 0) {
          try {
            intel.engine.endTrajectory(latestTrajId, quality);
          } catch (e) { /* fallback */ }
        }

        intel.save();

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              success: true,
              trajectory_id: latestTrajId,
              steps: traj.steps.length,
              duration_ms: traj.endTime - traj.startTime,
              quality
            }, null, 2)
          }]
        };
      }

      case 'hooks_coedit_record': {
        const primaryFile = args.primary_file;
        const relatedFiles = args.related_files || [];

        if (!intel.data.coEditPatterns) intel.data.coEditPatterns = {};
        if (!intel.data.coEditPatterns[primaryFile]) intel.data.coEditPatterns[primaryFile] = {};

        for (const related of relatedFiles) {
          intel.data.coEditPatterns[primaryFile][related] = (intel.data.coEditPatterns[primaryFile][related] || 0) + 1;
        }

        // Use engine if available
        if (intel.engine) {
          try {
            for (const related of relatedFiles) {
              intel.engine.recordCoEdit(primaryFile, related);
            }
          } catch (e) { /* fallback */ }
        }

        intel.save();

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, primary_file: primaryFile, related_count: relatedFiles.length }, null, 2)
          }]
        };
      }

      case 'hooks_coedit_suggest': {
        const file = args.file;
        const topK = args.top_k || 5;

        let suggestions = [];

        // Try engine first
        if (intel.engine) {
          try {
            suggestions = intel.engine.getLikelyNextFiles(file, topK);
          } catch (e) { /* fallback */ }
        }

        // Fallback to data
        if (suggestions.length === 0 && intel.data.coEditPatterns && intel.data.coEditPatterns[file]) {
          suggestions = Object.entries(intel.data.coEditPatterns[file])
            .sort((a, b) => b[1] - a[1])
            .slice(0, topK)
            .map(([f, count]) => ({ file: f, count, confidence: count / 10 }));
        }

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, file, suggestions }, null, 2)
          }]
        };
      }

      case 'hooks_error_record': {
        const error = args.error;
        const fix = args.fix;
        const file = args.file || '';

        if (!intel.data.errors) intel.data.errors = {};
        if (!intel.data.errors[error]) intel.data.errors[error] = [];
        intel.data.errors[error].push({ fix, file, recorded: Date.now() });

        // Use engine if available
        if (intel.engine) {
          try {
            intel.engine.recordErrorFix(error, fix);
          } catch (e) { /* fallback */ }
        }

        intel.save();

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, error: error.substring(0, 50), fixes_recorded: intel.data.errors[error].length }, null, 2)
          }]
        };
      }

      case 'hooks_error_suggest': {
        const error = args.error;

        let suggestions = [];

        // Try engine first
        if (intel.engine) {
          try {
            suggestions = intel.engine.getSuggestedFixes(error);
          } catch (e) { /* fallback */ }
        }

        // Fallback to data
        if (suggestions.length === 0 && intel.data.errors) {
          // Find similar errors
          for (const [errKey, fixes] of Object.entries(intel.data.errors)) {
            if (error.includes(errKey) || errKey.includes(error)) {
              suggestions.push(...fixes.map(f => f.fix));
            }
          }
        }

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, error: error.substring(0, 50), suggestions: [...new Set(suggestions)].slice(0, 5) }, null, 2)
          }]
        };
      }

      case 'hooks_force_learn': {
        let result = 'Learning triggered';

        if (intel.engine) {
          try {
            // Run forceLearn on engine
            const learnResult = intel.engine.forceLearn();
            result = learnResult || 'Engine learning complete';

            // Also tick for regular updates
            intel.engine.tick();
          } catch (e) {
            result = `Learning: ${e.message}`;
          }
        }

        // Save any updates
        intel.save();

        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: true, result, stats: intel.stats() }, null, 2)
          }]
        };
      }

      default:
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ success: false, error: `Unknown tool: ${name}` }, null, 2)
          }],
          isError: true
        };
    }
  } catch (error) {
    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ success: false, error: error.message }, null, 2)
      }],
      isError: true
    };
  }
});

// Resources - expose intelligence data
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: 'ruvector://intelligence/stats',
        name: 'Intelligence Stats',
        description: 'Current RuVector intelligence statistics',
        mimeType: 'application/json'
      },
      {
        uri: 'ruvector://intelligence/patterns',
        name: 'Learned Patterns',
        description: 'Q-learning patterns for agent routing',
        mimeType: 'application/json'
      },
      {
        uri: 'ruvector://intelligence/memories',
        name: 'Vector Memories',
        description: 'Stored context memories',
        mimeType: 'application/json'
      }
    ]
  };
});

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  switch (uri) {
    case 'ruvector://intelligence/stats':
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(intel.stats(), null, 2)
        }]
      };

    case 'ruvector://intelligence/patterns':
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify(intel.data.patterns || {}, null, 2)
        }]
      };

    case 'ruvector://intelligence/memories':
      return {
        contents: [{
          uri,
          mimeType: 'application/json',
          text: JSON.stringify((intel.data.memories || []).map(m => ({
            content: m.content,
            type: m.type,
            created: m.created
          })), null, 2)
        }]
      };

    default:
      throw new Error(`Unknown resource: ${uri}`);
  }
});

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('RuVector MCP server running on stdio');
}

main().catch(console.error);
