/**
 * NovitaProvider - Novita OpenAI-Compatible LLM Integration
 *
 * Uses the official OpenAI SDK against Novita's OpenAI-compatible endpoint.
 */

import OpenAI from 'openai';

import type {
  LLMProvider,
  Message,
  CompletionOptions,
  StreamOptions,
  Completion,
  Token,
  ModelInfo,
  Tool,
  ToolCall,
} from './index.js';

export interface NovitaConfig {
  apiKey: string;
  baseUrl?: string;
  model?: string;
  maxRetries?: number;
  timeout?: number;
}

export type NovitaModel =
  | 'deepseek/deepseek-v3.2'
  | 'minimax-minimax-m2.5'
  | 'zai-org-glm-5'
  | string;

type NovitaMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
};

const MODEL_INFO: Record<string, ModelInfo> = {
  'deepseek/deepseek-v3.2': {
    id: 'deepseek/deepseek-v3.2',
    name: 'DeepSeek V3.2',
    maxTokens: 8192,
    contextWindow: 64000,
  },
  'minimax-minimax-m2.5': {
    id: 'minimax-minimax-m2.5',
    name: 'MiniMax M2.5',
    maxTokens: 8192,
    contextWindow: 128000,
  },
  'zai-org-glm-5': {
    id: 'zai-org-glm-5',
    name: 'GLM-5',
    maxTokens: 8192,
    contextWindow: 128000,
  },
};

export class NovitaProvider implements LLMProvider {
  private readonly config: Required<NovitaConfig>;
  private readonly client: OpenAI;
  private readonly model: NovitaModel;

  constructor(config: NovitaConfig) {
    this.config = {
      apiKey: config.apiKey,
      baseUrl: config.baseUrl ?? 'https://api.novita.ai/openai',
      model: config.model ?? 'deepseek/deepseek-v3.2',
      maxRetries: config.maxRetries ?? 3,
      timeout: config.timeout ?? 120000,
    };
    this.model = this.config.model;
    this.client = new OpenAI({
      apiKey: this.config.apiKey,
      baseURL: this.config.baseUrl,
      maxRetries: this.config.maxRetries,
      timeout: this.config.timeout,
    });
  }

  async complete(messages: Message[], options?: CompletionOptions): Promise<Completion> {
    const modelInfo = this.getModel();
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: this.convertMessages(messages),
      max_tokens: options?.maxTokens ?? modelInfo.maxTokens,
      temperature: options?.temperature ?? 0.7,
      top_p: options?.topP,
      stop: options?.stopSequences,
      tools: options?.tools ? this.convertTools(options.tools) : undefined,
    });

    const choice = response.choices[0];
    const finishReason = this.mapFinishReason(choice?.finish_reason);
    const toolCalls = (choice?.message.tool_calls ?? []).map((toolCall) => ({
      id: toolCall.id,
      name: toolCall.function.name,
      input: JSON.parse(toolCall.function.arguments || '{}'),
    }));

    return {
      content: choice?.message.content ?? '',
      finishReason,
      usage: {
        inputTokens: response.usage?.prompt_tokens ?? 0,
        outputTokens: response.usage?.completion_tokens ?? 0,
      },
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    };
  }

  async *stream(messages: Message[], options?: StreamOptions): AsyncGenerator<Token, Completion, void> {
    const modelInfo = this.getModel();
    const stream = await this.client.chat.completions.create({
      model: this.model,
      messages: this.convertMessages(messages),
      max_tokens: options?.maxTokens ?? modelInfo.maxTokens,
      temperature: options?.temperature ?? 0.7,
      top_p: options?.topP,
      stop: options?.stopSequences,
      tools: options?.tools ? this.convertTools(options.tools) : undefined,
      stream: true,
    });

    let fullContent = '';
    let inputTokens = 0;
    let outputTokens = 0;
    let finishReason: Completion['finishReason'] = 'stop';
    const toolCalls: ToolCall[] = [];
    const pendingToolCalls: Map<number, { id: string; name: string; arguments: string }> = new Map();

    for await (const chunk of stream) {
      const choice = chunk.choices[0];
      if (!choice) continue;

      const token = choice.delta.content;
      if (token) {
        fullContent += token;
        options?.onToken?.(token);
        yield { type: 'text', text: token };
      }

      if (choice.delta.tool_calls) {
        for (const tc of choice.delta.tool_calls) {
          const idx = tc.index ?? 0;
          if (!pendingToolCalls.has(idx)) {
            pendingToolCalls.set(idx, { id: '', name: '', arguments: '' });
          }
          const pending = pendingToolCalls.get(idx)!;
          if (tc.id) pending.id = tc.id;
          if (tc.function?.name) pending.name = tc.function.name;
          if (tc.function?.arguments) pending.arguments += tc.function.arguments;
        }
      }

      if (choice.finish_reason) {
        finishReason = this.mapFinishReason(choice.finish_reason);
      }

      if (chunk.usage) {
        inputTokens = chunk.usage.prompt_tokens ?? inputTokens;
        outputTokens = chunk.usage.completion_tokens ?? outputTokens;
      }
    }

    for (const pending of pendingToolCalls.values()) {
      if (!pending.id || !pending.name) continue;
      try {
        const input = JSON.parse(pending.arguments || '{}');
        const toolUse = { id: pending.id, name: pending.name, input };
        toolCalls.push(toolUse);
        yield { type: 'tool_use', toolUse };
      } catch {
        // Ignore invalid function argument payloads.
      }
    }

    return {
      content: fullContent,
      finishReason,
      usage: { inputTokens, outputTokens },
      toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
    };
  }

  async countTokens(text: string): Promise<number> {
    return Math.ceil(text.length / 4);
  }

  getModel(): ModelInfo {
    return MODEL_INFO[this.model] ?? {
      id: this.model,
      name: this.model,
      maxTokens: 4096,
      contextWindow: 32000,
    };
  }

  async isHealthy(): Promise<boolean> {
    try {
      await this.client.chat.completions.create({
        model: this.model,
        messages: [{ role: 'user', content: 'ping' }],
        max_tokens: 1,
      });
      return true;
    } catch {
      return false;
    }
  }

  private mapFinishReason(reason: string | null | undefined): Completion['finishReason'] {
    if (reason === 'length') return 'length';
    if (reason === 'tool_calls') return 'tool_use';
    return 'stop';
  }

  private convertMessages(messages: Message[]): NovitaMessage[] {
    return messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
    }));
  }

  private convertTools(tools: Tool[]): Array<{
    type: 'function';
    function: { name: string; description: string; parameters: Record<string, unknown> };
  }> {
    return tools.map((tool) => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters,
      },
    }));
  }
}

export function createNovitaProvider(config: NovitaConfig): NovitaProvider {
  return new NovitaProvider(config);
}

export default NovitaProvider;
