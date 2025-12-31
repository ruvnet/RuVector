# Neural Trader Integration Examples

Comprehensive examples demonstrating the integration of the [neural-trader](https://www.npmjs.com/package/neural-trader) ecosystem with the RuVector platform.

## Overview

This directory contains examples showcasing all 20+ `@neural-trader` packages integrated with RuVector's high-performance HNSW vector database for pattern matching, signal storage, and neural network operations.

## Package Ecosystem

| Package | Version | Description |
|---------|---------|-------------|
| `neural-trader` | 2.7.1 | Core engine with native HNSW, SIMD, 178 NAPI functions |
| `@neural-trader/core` | 2.0.0 | Ultra-low latency Rust + Node.js bindings |
| `@neural-trader/strategies` | 2.6.0 | Strategy management and backtesting |
| `@neural-trader/execution` | 2.6.0 | Trade execution and order management |
| `@neural-trader/mcp` | 2.1.0 | MCP server with 87+ trading tools |
| `@neural-trader/risk` | 2.6.0 | VaR, stress testing, risk metrics |
| `@neural-trader/portfolio` | 2.6.0 | Portfolio optimization (Markowitz, Risk Parity) |
| `@neural-trader/neural` | 2.6.0 | Neural network training and prediction |
| `@neural-trader/brokers` | 2.1.1 | Alpaca, Interactive Brokers integration |
| `@neural-trader/backtesting` | 2.6.0 | Historical simulation engine |
| `@neural-trader/market-data` | 2.1.1 | Real-time and historical data providers |
| `@neural-trader/features` | 2.1.2 | 150+ technical indicators |
| `@neural-trader/backend` | 2.2.1 | High-performance Rust backend |
| `@neural-trader/predictor` | 0.1.0 | Conformal prediction with intervals |
| `@neural-trader/agentic-accounting-rust-core` | 0.1.1 | FIFO/LIFO/HIFO crypto tax calculations |
| `@neural-trader/sports-betting` | 2.1.1 | Arbitrage, Kelly sizing, odds analysis |
| `@neural-trader/prediction-markets` | 2.1.1 | Polymarket, Kalshi integration |
| `@neural-trader/news-trading` | 2.1.1 | Sentiment analysis, event-driven trading |
| `@neural-trader/mcp-protocol` | 2.0.0 | JSON-RPC 2.0 protocol types |
| `@neural-trader/benchoptimizer` | 2.1.1 | Performance benchmarking suite |

## Installation

```bash
cd examples/neural-trader
npm install
```

## Examples

### Core Integration
```bash
# Basic integration with RuVector
npm run core:basic

# HNSW vector search for pattern matching
npm run core:hnsw

# Technical indicators (150+ available)
npm run core:features
```

### Strategy & Backtesting
```bash
# Full strategy backtest with walk-forward optimization
npm run strategies:backtest
```

### Portfolio Management
```bash
# Portfolio optimization (Markowitz, Risk Parity, Black-Litterman)
npm run portfolio:optimize
```

### Neural Networks
```bash
# LSTM training for price prediction
npm run neural:train
```

### Risk Management
```bash
# VaR, CVaR, stress testing, risk limits
npm run risk:metrics
```

### MCP Integration
```bash
# Model Context Protocol server demo
npm run mcp:server
```

### Accounting
```bash
# Crypto tax calculations with FIFO/LIFO/HIFO
npm run accounting:crypto-tax
```

### Specialized Markets
```bash
# Sports betting: arbitrage, Kelly criterion
npm run specialized:sports

# Prediction markets: Polymarket, expected value
npm run specialized:prediction

# News trading: sentiment analysis, event-driven
npm run specialized:news
```

### Full Platform
```bash
# Complete platform integration demo
npm run full:platform
```

### Advanced Examples
```bash
# Production broker integration with Alpaca
npm run advanced:broker

# Order book microstructure analysis (VPIN, Kyle's Lambda)
npm run advanced:microstructure

# Conformal prediction with guaranteed intervals
npm run advanced:conformal
```

### Exotic Examples
```bash
# Multi-agent swarm trading coordination
npm run exotic:swarm

# Graph neural network correlation analysis
npm run exotic:gnn

# Transformer attention-based regime detection
npm run exotic:attention

# Deep Q-Learning reinforcement learning agent
npm run exotic:rl

# Quantum-inspired portfolio optimization (QAOA)
npm run exotic:quantum

# Hyperbolic Poincaré disk market embeddings
npm run exotic:hyperbolic

# Cross-exchange atomic arbitrage with MEV protection
npm run exotic:arbitrage
```

## Directory Structure

```
examples/neural-trader/
├── package.json           # Dependencies for all examples
├── README.md              # This file
├── core/                  # Core integration examples
│   ├── basic-integration.js
│   ├── hnsw-vector-search.js
│   └── technical-indicators.js
├── strategies/            # Strategy examples
│   └── backtesting.js
├── portfolio/             # Portfolio optimization
│   └── optimization.js
├── neural/                # Neural network examples
│   └── training.js
├── risk/                  # Risk management
│   └── risk-metrics.js
├── mcp/                   # MCP server integration
│   └── mcp-server.js
├── accounting/            # Accounting & tax
│   └── crypto-tax.js
├── specialized/           # Specialized markets
│   ├── sports-betting.js
│   ├── prediction-markets.js
│   └── news-trading.js
├── advanced/              # Production-grade implementations
│   ├── live-broker-alpaca.js
│   ├── order-book-microstructure.js
│   └── conformal-prediction.js
├── exotic/                # Cutting-edge techniques
│   ├── multi-agent-swarm.js
│   ├── gnn-correlation-network.js
│   ├── attention-regime-detection.js
│   ├── reinforcement-learning-agent.js
│   ├── quantum-portfolio-optimization.js
│   ├── hyperbolic-embeddings.js
│   └── atomic-arbitrage.js
└── full-integration/      # Complete platform
    └── platform.js
```

## RuVector Integration Points

These examples demonstrate how to leverage RuVector with neural-trader:

1. **Pattern Storage**: Store historical trading patterns as vectors for similarity search
2. **Signal Caching**: Cache trading signals with vector embeddings for quick retrieval
3. **Model Weights**: Store neural network checkpoints for versioning
4. **News Embeddings**: Index news articles with sentiment embeddings
5. **Trade Decision Logging**: Log decisions with vector search for analysis

## Advanced & Exotic Techniques

### Advanced (Production-Grade)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `live-broker-alpaca.js` | Production broker integration | Smart order routing, pre-trade risk checks, slicing algorithms |
| `order-book-microstructure.js` | Market microstructure analysis | VPIN, Kyle's Lambda, spread decomposition, hidden liquidity |
| `conformal-prediction.js` | Guaranteed prediction intervals | Distribution-free coverage, adaptive conformal inference |

### Exotic (Cutting-Edge)

| Example | Description | Key Concepts |
|---------|-------------|--------------|
| `multi-agent-swarm.js` | Distributed trading intelligence | Consensus mechanisms, pheromone signals, emergent behavior |
| `gnn-correlation-network.js` | Graph neural network analysis | Correlation networks, centrality measures, spectral analysis |
| `attention-regime-detection.js` | Transformer-based regimes | Multi-head attention, positional encoding, regime classification |
| `reinforcement-learning-agent.js` | DQN trading agent | Experience replay, epsilon-greedy, target networks |
| `quantum-portfolio-optimization.js` | QAOA & quantum annealing | QUBO formulation, simulated quantum circuits, cardinality constraints |
| `hyperbolic-embeddings.js` | Poincaré disk embeddings | Hyperbolic geometry, hierarchical structure, Möbius operations |
| `atomic-arbitrage.js` | Cross-exchange arbitrage | Flash loans, MEV protection, atomic execution |

## Performance

- **HNSW Search**: < 1ms for 1M+ vectors
- **Insert Throughput**: 45,000+ vectors/second
- **SIMD Acceleration**: 150x faster distance calculations
- **Native Rust Bindings**: Sub-millisecond latency

## MCP Tools (87+)

The MCP server exposes tools for:
- Market Data (8 tools): `getQuote`, `getHistoricalData`, `streamPrices`, etc.
- Trading (8 tools): `placeOrder`, `cancelOrder`, `getPositions`, etc.
- Analysis (8 tools): `calculateIndicator`, `runBacktest`, `detectPatterns`, etc.
- Risk (8 tools): `calculateVaR`, `runStressTest`, `checkRiskLimits`, etc.
- Portfolio (8 tools): `optimizePortfolio`, `rebalance`, `getPerformance`, etc.
- Neural (8 tools): `trainModel`, `predict`, `evaluateModel`, etc.

## Claude Code Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "neural-trader": {
      "command": "npx",
      "args": ["@neural-trader/mcp", "start"],
      "env": {
        "ALPACA_API_KEY": "your-api-key",
        "ALPACA_SECRET_KEY": "your-secret-key"
      }
    }
  }
}
```

## Resources

- [Neural Trader GitHub](https://github.com/ruvnet/neural-trader)
- [RuVector GitHub](https://github.com/ruvnet/ruvector)
- [NPM Packages](https://www.npmjs.com/search?q=%40neural-trader)

## License

MIT OR Apache-2.0
