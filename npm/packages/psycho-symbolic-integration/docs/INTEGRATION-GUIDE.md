# 🔧 Psycho-Symbolic Integration Guide

## Table of Contents
1. [Installation](#installation)
2. [Architecture Overview](#architecture-overview)
3. [Integration Patterns](#integration-patterns)
4. [API Reference](#api-reference)
5. [Performance Tuning](#performance-tuning)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
- Node.js >= 18.0.0
- npm >= 9.0.0

### Basic Installation

```bash
# Install the integration package
npm install psycho-symbolic-integration

# Core dependencies (required)
npm install psycho-symbolic-reasoner @ruvector/agentic-synth

# Optional: Vector database
npm install ruvector
```

### Verify Installation

```bash
# Check versions
npm list psycho-symbolic-reasoner
npm list @ruvector/agentic-synth
npm list ruvector
```

---

## Architecture Overview

### Component Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                 Application Layer                             │
├──────────────────────────────────────────────────────────────┤
│        IntegratedPsychoSymbolicSystem API                    │
├───────────────┬─────────────────┬──────────────────────────┤
│               │                 │                            │
│  Psycho-      │  Agentic-Synth  │  Ruvector                 │
│  Symbolic     │  Adapter        │  Adapter                  │
│  Reasoner     │                 │  (Optional)               │
│               │                 │                            │
├───────────────┼─────────────────┼──────────────────────────┤
│               │                 │                            │
│ Core Engine:  │ Features:       │ Features:                 │
│ - WASM/Rust   │ - Preference    │ - Vector search           │
│ - 0.3ms query │   guidance      │ - Embeddings              │
│ - Graph       │ - Sentiment     │ - Semantic cache          │
│ - Planning    │   validation    │ - Hybrid queries          │
│ - Sentiment   │ - Quality       │                            │
│ - Preferences │   scoring       │                            │
│               │                 │                            │
└───────────────┴─────────────────┴──────────────────────────┘
```

### Data Flow

```
User Input
    │
    ├─── Analyze Text ───────────────────►  Psycho-Symbolic Reasoner
    │                                       │
    │                                       ├─ Sentiment (0.4ms)
    │                                       ├─ Preferences (0.6ms)
    │                                       └─ Emotional context
    │
    ├─── Generate Data ──────────────────►  Agentic-Synth + Adapter
    │                                       │
    │                                       ├─ Apply preferences
    │                                       ├─ Sentiment guidance
    │                                       ├─ Validate quality
    │                                       └─ Return enhanced data
    │
    └─── Query Knowledge ────────────────►  Hybrid Reasoning
                                            │
                                            ├─ Symbolic query (1.2ms)
                                            ├─ Vector search (10ms)
                                            └─ Combined results
```

---

## Integration Patterns

### Pattern 1: Sentiment-Guided Generation

**Use Case**: Generate content with specific emotional tone

```typescript
import { quickStart } from 'psycho-symbolic-integration';

const system = await quickStart();

const result = await system.generateIntelligently('structured', {
  count: 100,
  schema: {
    message: 'string',
    tone: 'string'
  }
}, {
  targetSentiment: {
    score: 0.8,      // Positive sentiment
    emotion: 'joy'   // Primary emotion
  },
  qualityThreshold: 0.9
});

console.log(`Generated ${result.data.length} messages`);
console.log(`Sentiment match: ${result.psychoMetrics.sentimentMatch * 100}%`);
```

**Performance**: 2-5 seconds for 100 records

### Pattern 2: Preference-Aware Data

**Use Case**: Generate data aligned with user preferences

```typescript
const userPreferences = [
  "I prefer concise, actionable content",
  "I like data-driven insights",
  "I value simplicity over complexity"
];

const result = await system.generateIntelligently('structured', {
  count: 50,
  schema: contentSchema
}, {
  userPreferences,
  contextualFactors: {
    environment: 'business',
    constraints: ['length <= 200 words']
  }
});

console.log(`Preference alignment: ${result.psychoMetrics.preferenceAlignment}`);
```

**Performance**: 1-3 seconds for 50 records

### Pattern 3: Hybrid Reasoning

**Use Case**: Combine symbolic logic with semantic search

```typescript
// Load knowledge base
await system.loadKnowledgeBase({
  nodes: [ /* your entities */ ],
  edges: [ /* relationships */ ]
});

// Hybrid query: 60% symbolic, 40% vector
const results = await system.intelligentQuery(
  'Find stress management techniques for busy professionals',
  {
    symbolicWeight: 0.6,  // Logical rules
    vectorWeight: 0.4,    // Semantic similarity
    maxResults: 10
  }
);

// Results include combined scoring
results.forEach(r => {
  console.log(`${r.nodes[0].id}:`);
  console.log(`  Symbolic match: ${r.reasoning.symbolicMatch}`);
  console.log(`  Semantic match: ${r.reasoning.semanticMatch}`);
  console.log(`  Combined score: ${r.reasoning.combinedScore}`);
});
```

**Performance**: 10-50ms depending on graph size

### Pattern 4: Goal-Oriented Planning

**Use Case**: Plan complex data generation strategies

```typescript
const plan = await system.planDataGeneration(
  'Generate 10,000 training examples for sentiment model',
  {
    targetQuality: 0.95,
    balancedSentiment: true,
    diverseEmotions: ['joy', 'sadness', 'anger', 'fear', 'surprise'],
    maxCostPerRecord: 0.001
  }
);

// Execute plan step by step
for (const step of plan.steps) {
  console.log(`Executing: ${step.action}`);
  // ... execute step
}
```

**Performance**: Planning takes 2-5ms, execution varies

### Pattern 5: Batch Processing

**Use Case**: Process large datasets efficiently

```typescript
const batchSize = 100;
const totalRecords = 10000;
const results = [];

for (let i = 0; i < totalRecords; i += batchSize) {
  const batch = await system.generateIntelligently('structured', {
    count: batchSize,
    schema: mySchema
  }, psychoConfig);

  results.push(...batch.data);

  // Store in vector DB for semantic search
  if (system.ruvectorAdapter?.isAvailable()) {
    await system.ruvectorAdapter.storeKnowledgeGraph({
      nodes: batch.data.map((d, idx) => ({
        id: `record_${i + idx}`,
        type: 'generated',
        properties: d
      })),
      edges: []
    });
  }

  console.log(`Processed ${i + batchSize}/${totalRecords}`);
}
```

**Performance**: ~2 seconds per 100 records

---

## API Reference

### IntegratedPsychoSymbolicSystem

#### Constructor

```typescript
new IntegratedPsychoSymbolicSystem(config?: IntegratedSystemConfig)
```

**Config Options**:
```typescript
interface IntegratedSystemConfig {
  reasoner?: {
    enableGraphReasoning?: boolean;
    enableAffectExtraction?: boolean;
    enablePlanning?: boolean;
    logLevel?: 'debug' | 'info' | 'warn' | 'error';
  };

  synth?: {
    provider?: 'gemini' | 'openrouter';
    apiKey?: string;
    model?: string;
    cache?: {
      enabled?: boolean;
      maxSize?: number;
    };
  };

  vector?: {
    dbPath?: string;
    collectionName?: string;
    dimensions?: number;
    enableSemanticCache?: boolean;
  };
}
```

#### Methods

**initialize()**
```typescript
await system.initialize(): Promise<void>
```
Initialize all components. Must be called before other operations.

**generateIntelligently()**
```typescript
await system.generateIntelligently(
  type: 'timeseries' | 'events' | 'structured',
  baseOptions: any,
  psychoConfig?: PsychoGuidedGenerationConfig
): Promise<GenerationResult>
```

**intelligentQuery()**
```typescript
await system.intelligentQuery(
  query: string,
  options?: {
    symbolicWeight?: number;
    vectorWeight?: number;
    maxResults?: number;
  }
): Promise<SemanticQueryResult[]>
```

**analyzeText()**
```typescript
await system.analyzeText(text: string): Promise<{
  sentiment: SentimentResult;
  preferences: PreferencesResult;
}>
```

**loadKnowledgeBase()**
```typescript
await system.loadKnowledgeBase(knowledgeBase: {
  nodes: Node[];
  edges: Edge[];
}): Promise<void>
```

**planDataGeneration()**
```typescript
await system.planDataGeneration(
  goal: string,
  constraints: any
): Promise<Plan>
```

### Quick Start Functions

```typescript
// Fast initialization with defaults
const system = await quickStart(apiKey?: string): Promise<IntegratedPsychoSymbolicSystem>

// Manual creation
const system = createIntegratedSystem(config: IntegratedSystemConfig): IntegratedPsychoSymbolicSystem
```

---

## Performance Tuning

### Optimize for Speed

```typescript
const system = new IntegratedPsychoSymbolicSystem({
  reasoner: {
    enableGraphReasoning: true,
    enableAffectExtraction: false,  // Disable if not needed
    enablePlanning: false,           // Disable if not needed
    logLevel: 'error'                // Reduce logging overhead
  },

  synth: {
    cache: {
      enabled: true,
      maxSize: 10000  // Larger cache for better hit rate
    }
  },

  vector: {
    enableSemanticCache: true  // Cache embeddings
  }
});
```

**Expected Performance**:
- Sentiment analysis: 0.3-0.5ms
- Graph query: 1-2ms
- Data generation: 1-3s per 100 records
- Hybrid query: 5-20ms

### Optimize for Quality

```typescript
const result = await system.generateIntelligently('structured', {
  count: 100,
  schema: mySchema
}, {
  targetSentiment: { score: 0.8, emotion: 'positive' },
  qualityThreshold: 0.95,  // High quality bar
  userPreferences: detailedPreferences,
  contextualFactors: {
    emotionalState: 'focused',
    environment: 'professional',
    constraints: [
      'quality >= 0.95',
      'coherence >= 0.9',
      'relevance >= 0.85'
    ]
  }
});
```

**Expected Quality**:
- Preference alignment: 85-95%
- Sentiment match: 80-90%
- Overall quality: 90-95%

### Memory Management

```typescript
// Clear caches periodically
if (system.ruvectorAdapter) {
  system.ruvectorAdapter.clearCache();
}

system.synthAdapter.clearHistory();

// Monitor memory usage
const insights = system.getSystemInsights();
console.log(insights.adapters.vectorCache);
```

---

## Best Practices

### 1. API Key Management

```typescript
// ✅ Good: Use environment variables
const system = await quickStart(process.env.GEMINI_API_KEY);

// ❌ Bad: Hardcode API keys
const system = await quickStart('your-api-key-here');
```

### 2. Error Handling

```typescript
try {
  const result = await system.generateIntelligently(...);
} catch (error) {
  if (error.message.includes('API key')) {
    console.error('API key not configured');
  } else if (error.message.includes('quota')) {
    console.error('API quota exceeded, implement backoff');
  } else {
    console.error('Generation failed:', error);
  }
}
```

### 3. Batch Operations

```typescript
// ✅ Good: Process in batches
for (let i = 0; i < 10000; i += 100) {
  await generateBatch(100);
}

// ❌ Bad: Generate all at once
await generate(10000);  // May timeout or exhaust memory
```

### 4. Cache Strategy

```typescript
// Enable caching for repeated queries
const system = new IntegratedPsychoSymbolicSystem({
  synth: {
    cache: { enabled: true, maxSize: 1000 }
  },
  vector: {
    enableSemanticCache: true
  }
});
```

### 5. Cleanup

```typescript
// Always cleanup on exit
process.on('SIGINT', async () => {
  await system.shutdown();
  process.exit(0);
});
```

---

## Troubleshooting

### Common Issues

#### "Ruvector not available"
**Cause**: Optional peer dependency not installed

**Solution**:
```bash
npm install ruvector
```

Or disable vector features:
```typescript
// System will work without vector DB
const system = new IntegratedPsychoSymbolicSystem({
  // Don't specify vector config
});
```

#### "API key not configured"
**Cause**: Missing or invalid API key

**Solution**:
```bash
export GEMINI_API_KEY="your-key-here"
```

#### "Generation quality too low"
**Cause**: Insufficient guidance or low quality threshold

**Solution**:
```typescript
const result = await system.generateIntelligently('structured', options, {
  qualityThreshold: 0.7,  // Lower threshold
  userPreferences: [      // Add more guidance
    'specific preference 1',
    'specific preference 2'
  ]
});
```

#### "Slow hybrid queries"
**Cause**: Large knowledge graph or inefficient weights

**Solution**:
```typescript
// Increase symbolic weight for faster queries
const results = await system.intelligentQuery(query, {
  symbolicWeight: 0.8,  // More symbolic, less vector
  vectorWeight: 0.2,
  maxResults: 5         // Reduce result count
});
```

### Debug Mode

```typescript
const system = new IntegratedPsychoSymbolicSystem({
  reasoner: {
    logLevel: 'debug'  // Enable detailed logging
  }
});

// Check system status
console.log(system.getSystemInsights());
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/ruvnet/ruvector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/ruvector/discussions)
- **Documentation**: [Main Docs](https://github.com/ruvnet/ruvector)

---

**Ready to build intelligent AI systems?** 🚀

Check out the [examples](../examples/) directory for complete working examples!
