---
name: "QE Code Intelligence"
description: "Knowledge graph-based code understanding with semantic search and 80% token reduction through intelligent context retrieval."
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qe-code-intelligence.yaml

---

# QE Code Intelligence

## Purpose

Guide the use of v3's code intelligence capabilities including knowledge graph construction, semantic code search, dependency mapping, and context-aware code understanding with significant token reduction.

## Activation

- When understanding unfamiliar code
- When searching for code semantically
- When analyzing dependencies
- When building code knowledge graphs
- When reducing context for AI operations

## Quick Start

```bash
# Index codebase into knowledge graph
aqe kg index --source src/ --incremental

# Semantic code search
aqe kg search "authentication middleware" --limit 10

# Query dependencies
aqe kg deps --file src/services/UserService.ts --depth 3

# Get intelligent context
aqe kg context --query "how does payment processing work"
```

## Agent Workflow

```typescript
// Build knowledge graph
Task("Index codebase", `
  Build knowledge graph for the project:
  - Parse all TypeScript files in src/
  - Extract entities (classes, functions, types)
  - Map relationships (imports, calls, inheritance)
  - Generate embeddings for semantic search
  Store in AgentDB vector database.
`, "qe-knowledge-graph")

// Semantic search
Task("Find relevant code", `
  Search for code related to "user authentication flow":
  - Use semantic similarity (not just keyword)
  - Include related functions and types
  - Rank by relevance score
  - Return with minimal context (80% token reduction)
`, "qe-semantic-searcher")
```

## Knowledge Graph Operations

### 1. Codebase Indexing

```typescript
await knowledgeGraph.index({
  source: 'src/**/*.ts',
  extraction: {
    entities: ['class', 'function', 'interface', 'type', 'variable'],
    relationships: ['imports', 'calls', 'extends', 'implements', 'uses'],
    metadata: ['jsdoc', 'complexity', 'lines']
  },
  embeddings: {
    model: 'code-embedding',
    dimensions: 384,
    normalize: true
  },
  incremental: true  // Only index changed files
});
```

### 2. Semantic Search

```typescript
await semanticSearcher.search({
  query: 'payment processing with stripe',
  options: {
    similarity: 'cosine',
    threshold: 0.7,
    limit: 20,
    includeContext: true
  },
  filters: {
    fileTypes: ['.ts', '.tsx'],
    excludePaths: ['node_modules', 'dist']
  }
});
```

### 3. Dependency Analysis

```typescript
await dependencyMapper.analyze({
  entry: 'src/services/OrderService.ts',
  depth: 3,
  direction: 'both',  // imports and importedBy
  output: {
    graph: true,
    metrics: {
      afferentCoupling: true,
      efferentCoupling: true,
      instability: true
    }
  }
});
```

## Token Reduction Strategy

```typescript
// Get context with 80% token reduction
const context = await codeIntelligence.getOptimizedContext({
  query: 'implement user registration',
  budget: 4000,  // max tokens
  strategy: {
    relevanceRanking: true,
    summarization: true,
    codeCompression: true,
    deduplication: true
  },
  include: {
    signatures: true,
    implementations: 'relevant-only',
    comments: 'essential',
    examples: 'top-3'
  }
});
```

## Knowledge Graph Schema

```typescript
interface KnowledgeGraph {
  entities: {
    id: string;
    type: 'class' | 'function' | 'interface' | 'type' | 'file';
    name: string;
    file: string;
    line: number;
    embedding: number[];
    metadata: Record<string, any>;
  }[];
  relationships: {
    source: string;
    target: string;
    type: 'imports' | 'calls' | 'extends' | 'implements' | 'uses';
    weight: number;
  }[];
  indexes: {
    byName: Map<string, string[]>;
    byFile: Map<string, string[]>;
    byType: Map<string, string[]>;
  };
}
```

## Search Results

```typescript
interface SearchResult {
  entity: {
    name: string;
    type: string;
    file: string;
    line: number;
  };
  relevance: number;
  snippet: string;
  context: {
    before: string[];
    after: string[];
    related: string[];
  };
  explanation: string;
}
```

## CLI Examples

```bash
# Full reindex
aqe kg index --source src/ --force

# Search with filters
aqe kg search "database connection" --type function --file "*.service.ts"

# Show entity details
aqe kg show --entity UserService --relations

# Export graph
aqe kg export --format dot --output codebase.dot

# Statistics
aqe kg stats
```

## Coordination

**Primary Agents**: qe-knowledge-graph, qe-semantic-searcher, qe-dependency-mapper
**Coordinator**: qe-code-intelligence-coordinator
**Related Skills**: qe-test-generation, qe-defect-intelligence
