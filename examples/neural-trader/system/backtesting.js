/**
 * Backtesting Framework
 *
 * Historical simulation with comprehensive performance metrics:
 * - Sharpe Ratio, Sortino Ratio
 * - Maximum Drawdown, Calmar Ratio
 * - Win Rate, Profit Factor
 * - Value at Risk (VaR), Expected Shortfall
 * - Rolling statistics and regime analysis
 */

import { TradingPipeline, createTradingPipeline } from './trading-pipeline.js';

// Backtesting Configuration
const backtestConfig = {
  // Simulation settings
  simulation: {
    initialCapital: 100000,
    startDate: null,      // Use all available data if null
    endDate: null,
    rebalanceFrequency: 'daily',  // daily, weekly, monthly
    warmupPeriod: 50      // Days for indicator warmup
  },

  // Execution assumptions
  execution: {
    slippage: 0.001,      // 0.1%
    commission: 0.001,    // 0.1%
    marketImpact: 0.0005, // 0.05% for large orders
    fillRate: 1.0         // 100% fill rate assumed
  },

  // Risk-free rate for Sharpe calculation
  riskFreeRate: 0.05,     // 5% annual

  // Benchmark
  benchmark: 'buyAndHold'  // buyAndHold, equalWeight, or custom
};

/**
 * Performance Metrics Calculator
 */
class PerformanceMetrics {
  constructor(riskFreeRate = 0.05) {
    this.riskFreeRate = riskFreeRate;
    this.dailyRiskFreeRate = Math.pow(1 + riskFreeRate, 1/252) - 1;
  }

  // Calculate all metrics from equity curve
  calculate(equityCurve, benchmark = null) {
    if (equityCurve.length < 2) {
      return this.emptyMetrics();
    }

    const returns = this.calculateReturns(equityCurve);
    const benchmarkReturns = benchmark ? this.calculateReturns(benchmark) : null;

    return {
      // Return metrics
      totalReturn: this.totalReturn(equityCurve),
      annualizedReturn: this.annualizedReturn(returns),
      cagr: this.cagr(equityCurve),

      // Risk metrics
      volatility: this.volatility(returns),
      annualizedVolatility: this.annualizedVolatility(returns),
      maxDrawdown: this.maxDrawdown(equityCurve),
      averageDrawdown: this.averageDrawdown(equityCurve),
      drawdownDuration: this.drawdownDuration(equityCurve),

      // Risk-adjusted metrics
      sharpeRatio: this.sharpeRatio(returns),
      sortinoRatio: this.sortinoRatio(returns),
      calmarRatio: this.calmarRatio(equityCurve, returns),
      informationRatio: benchmarkReturns ? this.informationRatio(returns, benchmarkReturns) : null,

      // Trade metrics
      winRate: this.winRate(returns),
      profitFactor: this.profitFactor(returns),
      averageWin: this.averageWin(returns),
      averageLoss: this.averageLoss(returns),
      payoffRatio: this.payoffRatio(returns),
      expectancy: this.expectancy(returns),

      // Tail risk metrics
      var95: this.valueAtRisk(returns, 0.95),
      var99: this.valueAtRisk(returns, 0.99),
      cvar95: this.conditionalVaR(returns, 0.95),
      skewness: this.skewness(returns),
      kurtosis: this.kurtosis(returns),

      // Additional metrics
      tradingDays: returns.length,
      bestDay: Math.max(...returns),
      worstDay: Math.min(...returns),
      positiveMonths: this.positiveMonths(returns),

      // Raw data
      returns,
      equityCurve
    };
  }

  calculateReturns(equityCurve) {
    const returns = [];
    for (let i = 1; i < equityCurve.length; i++) {
      returns.push((equityCurve[i] - equityCurve[i-1]) / equityCurve[i-1]);
    }
    return returns;
  }

  totalReturn(equityCurve) {
    return (equityCurve[equityCurve.length - 1] - equityCurve[0]) / equityCurve[0];
  }

  annualizedReturn(returns) {
    const totalReturn = returns.reduce((a, b) => (1 + a) * (1 + b), 1) - 1;
    const years = returns.length / 252;
    return Math.pow(1 + totalReturn, 1 / years) - 1;
  }

  cagr(equityCurve) {
    const years = (equityCurve.length - 1) / 252;
    return Math.pow(equityCurve[equityCurve.length - 1] / equityCurve[0], 1 / years) - 1;
  }

  volatility(returns) {
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }

  annualizedVolatility(returns) {
    return this.volatility(returns) * Math.sqrt(252);
  }

  maxDrawdown(equityCurve) {
    let maxDrawdown = 0;
    let peak = equityCurve[0];

    for (const value of equityCurve) {
      if (value > peak) peak = value;
      const drawdown = (peak - value) / peak;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }

    return maxDrawdown;
  }

  averageDrawdown(equityCurve) {
    const drawdowns = [];
    let peak = equityCurve[0];

    for (const value of equityCurve) {
      if (value > peak) peak = value;
      drawdowns.push((peak - value) / peak);
    }

    return drawdowns.reduce((a, b) => a + b, 0) / drawdowns.length;
  }

  drawdownDuration(equityCurve) {
    let maxDuration = 0;
    let currentDuration = 0;
    let peak = equityCurve[0];

    for (const value of equityCurve) {
      if (value >= peak) {
        peak = value;
        currentDuration = 0;
      } else {
        currentDuration++;
        if (currentDuration > maxDuration) maxDuration = currentDuration;
      }
    }

    return maxDuration;
  }

  sharpeRatio(returns) {
    const excessReturns = returns.map(r => r - this.dailyRiskFreeRate);
    const meanExcess = excessReturns.reduce((a, b) => a + b, 0) / excessReturns.length;
    const vol = this.volatility(excessReturns);
    return vol > 0 ? (meanExcess / vol) * Math.sqrt(252) : 0;
  }

  sortinoRatio(returns) {
    const excessReturns = returns.map(r => r - this.dailyRiskFreeRate);
    const meanExcess = excessReturns.reduce((a, b) => a + b, 0) / excessReturns.length;

    // Downside deviation
    const negativeReturns = excessReturns.filter(r => r < 0);
    if (negativeReturns.length === 0) return Infinity;

    const downsideVariance = negativeReturns.reduce((a, b) => a + b * b, 0) / returns.length;
    const downsideDeviation = Math.sqrt(downsideVariance);

    return downsideDeviation > 0 ? (meanExcess / downsideDeviation) * Math.sqrt(252) : 0;
  }

  calmarRatio(equityCurve, returns) {
    const annReturn = this.annualizedReturn(returns);
    const maxDD = this.maxDrawdown(equityCurve);
    return maxDD > 0 ? annReturn / maxDD : 0;
  }

  informationRatio(returns, benchmarkReturns) {
    const trackingError = [];
    const minLen = Math.min(returns.length, benchmarkReturns.length);

    for (let i = 0; i < minLen; i++) {
      trackingError.push(returns[i] - benchmarkReturns[i]);
    }

    const meanTE = trackingError.reduce((a, b) => a + b, 0) / trackingError.length;
    const teVol = this.volatility(trackingError);

    return teVol > 0 ? (meanTE / teVol) * Math.sqrt(252) : 0;
  }

  winRate(returns) {
    const wins = returns.filter(r => r > 0).length;
    return returns.length > 0 ? wins / returns.length : 0;
  }

  profitFactor(returns) {
    const grossProfit = returns.filter(r => r > 0).reduce((a, b) => a + b, 0);
    const grossLoss = Math.abs(returns.filter(r => r < 0).reduce((a, b) => a + b, 0));
    return grossLoss > 0 ? grossProfit / grossLoss : Infinity;
  }

  averageWin(returns) {
    const wins = returns.filter(r => r > 0);
    return wins.length > 0 ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
  }

  averageLoss(returns) {
    const losses = returns.filter(r => r < 0);
    return losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / losses.length : 0;
  }

  payoffRatio(returns) {
    const avgWin = this.averageWin(returns);
    const avgLoss = Math.abs(this.averageLoss(returns));
    return avgLoss > 0 ? avgWin / avgLoss : Infinity;
  }

  expectancy(returns) {
    const winRate = this.winRate(returns);
    const avgWin = this.averageWin(returns);
    const avgLoss = Math.abs(this.averageLoss(returns));
    return winRate * avgWin - (1 - winRate) * avgLoss;
  }

  valueAtRisk(returns, confidence = 0.95) {
    const sorted = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidence) * sorted.length);
    return -sorted[index];
  }

  conditionalVaR(returns, confidence = 0.95) {
    const sorted = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidence) * sorted.length);
    const tailReturns = sorted.slice(0, index + 1);
    return tailReturns.length > 0 ? -tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length : 0;
  }

  skewness(returns) {
    const n = returns.length;
    const mean = returns.reduce((a, b) => a + b, 0) / n;
    const m2 = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
    const m3 = returns.reduce((a, b) => a + Math.pow(b - mean, 3), 0) / n;
    const std = Math.sqrt(m2);
    return std > 0 ? m3 / Math.pow(std, 3) : 0;
  }

  kurtosis(returns) {
    const n = returns.length;
    const mean = returns.reduce((a, b) => a + b, 0) / n;
    const m2 = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / n;
    const m4 = returns.reduce((a, b) => a + Math.pow(b - mean, 4), 0) / n;
    const std = Math.sqrt(m2);
    return std > 0 ? m4 / Math.pow(std, 4) - 3 : 0;  // Excess kurtosis
  }

  positiveMonths(returns) {
    // Group by 21-day periods (approximate months)
    const monthlyReturns = [];
    for (let i = 0; i < returns.length; i += 21) {
      const monthReturn = returns.slice(i, i + 21).reduce((a, b) => (1 + a) * (1 + b) - 1, 0);
      monthlyReturns.push(monthReturn);
    }
    const positive = monthlyReturns.filter(r => r > 0).length;
    return monthlyReturns.length > 0 ? positive / monthlyReturns.length : 0;
  }

  emptyMetrics() {
    return {
      totalReturn: 0, annualizedReturn: 0, cagr: 0,
      volatility: 0, annualizedVolatility: 0, maxDrawdown: 0,
      sharpeRatio: 0, sortinoRatio: 0, calmarRatio: 0,
      winRate: 0, profitFactor: 0, expectancy: 0,
      var95: 0, var99: 0, cvar95: 0,
      tradingDays: 0, returns: [], equityCurve: []
    };
  }
}

/**
 * Backtest Engine
 */
class BacktestEngine {
  constructor(config = backtestConfig) {
    this.config = config;
    this.metricsCalculator = new PerformanceMetrics(config.riskFreeRate);
    this.pipeline = createTradingPipeline();
  }

  // Run backtest on historical data
  async run(historicalData, options = {}) {
    const {
      symbols = ['DEFAULT'],
      newsData = [],
      riskManager = null
    } = options;

    const results = {
      equityCurve: [this.config.simulation.initialCapital],
      benchmarkCurve: [this.config.simulation.initialCapital],
      trades: [],
      dailyReturns: [],
      positions: [],
      signals: []
    };

    // Initialize portfolio
    let portfolio = {
      equity: this.config.simulation.initialCapital,
      cash: this.config.simulation.initialCapital,
      positions: {},
      assets: symbols
    };

    // Skip warmup period
    const startIndex = this.config.simulation.warmupPeriod;
    const prices = {};

    // Process each day
    for (let i = startIndex; i < historicalData.length; i++) {
      const dayData = historicalData[i];
      const currentPrice = dayData.close || dayData.price || 100;

      // Update prices
      for (const symbol of symbols) {
        prices[symbol] = currentPrice;
      }

      // Get historical window for pipeline
      const windowStart = Math.max(0, i - 100);
      const marketWindow = historicalData.slice(windowStart, i + 1);

      // Get news for this day (simplified - would filter by date in production)
      const dayNews = newsData.filter((n, idx) => idx < 3);

      // Execute pipeline
      const context = {
        marketData: marketWindow,
        newsData: dayNews,
        symbols,
        portfolio,
        prices,
        riskManager
      };

      try {
        const pipelineResult = await this.pipeline.execute(context);

        // Store signals
        if (pipelineResult.signals) {
          results.signals.push({
            day: i,
            signals: pipelineResult.signals
          });
        }

        // Execute orders
        if (pipelineResult.orders && pipelineResult.orders.length > 0) {
          for (const order of pipelineResult.orders) {
            const trade = this.executeTrade(order, portfolio, prices);
            if (trade) {
              results.trades.push({ day: i, ...trade });
            }
          }
        }
      } catch (error) {
        // Pipeline error - skip this day
        console.warn(`Day ${i} pipeline error:`, error.message);
      }

      // Update portfolio value
      portfolio.equity = portfolio.cash;
      for (const [symbol, qty] of Object.entries(portfolio.positions)) {
        portfolio.equity += qty * (prices[symbol] || 0);
      }

      results.equityCurve.push(portfolio.equity);
      results.positions.push({ ...portfolio.positions });

      // Update benchmark (buy and hold)
      const benchmarkReturn = i > startIndex
        ? (currentPrice / historicalData[i - 1].close) - 1
        : 0;
      const lastBenchmark = results.benchmarkCurve[results.benchmarkCurve.length - 1];
      results.benchmarkCurve.push(lastBenchmark * (1 + benchmarkReturn));

      // Daily return
      if (results.equityCurve.length >= 2) {
        const prev = results.equityCurve[results.equityCurve.length - 2];
        const curr = results.equityCurve[results.equityCurve.length - 1];
        results.dailyReturns.push((curr - prev) / prev);
      }
    }

    // Calculate performance metrics
    results.metrics = this.metricsCalculator.calculate(
      results.equityCurve,
      results.benchmarkCurve
    );

    results.benchmarkMetrics = this.metricsCalculator.calculate(
      results.benchmarkCurve
    );

    // Trade statistics
    results.tradeStats = this.calculateTradeStats(results.trades);

    return results;
  }

  // Execute a trade
  executeTrade(order, portfolio, prices) {
    const price = prices[order.symbol] || order.price;
    const value = order.quantity * price;
    const costs = value * (this.config.execution.slippage + this.config.execution.commission);

    if (order.side === 'buy') {
      if (portfolio.cash < value + costs) {
        return null;  // Insufficient funds
      }
      portfolio.cash -= value + costs;
      portfolio.positions[order.symbol] = (portfolio.positions[order.symbol] || 0) + order.quantity;
    } else {
      const currentQty = portfolio.positions[order.symbol] || 0;
      if (currentQty < order.quantity) {
        return null;  // Insufficient shares
      }
      portfolio.cash += value - costs;
      portfolio.positions[order.symbol] = currentQty - order.quantity;
    }

    return {
      symbol: order.symbol,
      side: order.side,
      quantity: order.quantity,
      price,
      value,
      costs,
      timestamp: Date.now()
    };
  }

  // Calculate trade statistics
  calculateTradeStats(trades) {
    if (trades.length === 0) {
      return { totalTrades: 0, buyTrades: 0, sellTrades: 0, totalVolume: 0, totalCosts: 0 };
    }

    return {
      totalTrades: trades.length,
      buyTrades: trades.filter(t => t.side === 'buy').length,
      sellTrades: trades.filter(t => t.side === 'sell').length,
      totalVolume: trades.reduce((a, t) => a + t.value, 0),
      totalCosts: trades.reduce((a, t) => a + t.costs, 0),
      avgTradeSize: trades.reduce((a, t) => a + t.value, 0) / trades.length
    };
  }

  // Generate backtest report
  generateReport(results) {
    const m = results.metrics;
    const b = results.benchmarkMetrics;
    const t = results.tradeStats;

    return `
══════════════════════════════════════════════════════════════════════
BACKTEST REPORT
══════════════════════════════════════════════════════════════════════

PERFORMANCE SUMMARY
──────────────────────────────────────────────────────────────────────
                        Strategy      Benchmark     Difference
Total Return:           ${(m.totalReturn * 100).toFixed(2)}%        ${(b.totalReturn * 100).toFixed(2)}%         ${((m.totalReturn - b.totalReturn) * 100).toFixed(2)}%
Annualized Return:      ${(m.annualizedReturn * 100).toFixed(2)}%        ${(b.annualizedReturn * 100).toFixed(2)}%         ${((m.annualizedReturn - b.annualizedReturn) * 100).toFixed(2)}%
CAGR:                   ${(m.cagr * 100).toFixed(2)}%        ${(b.cagr * 100).toFixed(2)}%         ${((m.cagr - b.cagr) * 100).toFixed(2)}%

RISK METRICS
──────────────────────────────────────────────────────────────────────
Volatility (Ann.):      ${(m.annualizedVolatility * 100).toFixed(2)}%        ${(b.annualizedVolatility * 100).toFixed(2)}%
Max Drawdown:           ${(m.maxDrawdown * 100).toFixed(2)}%        ${(b.maxDrawdown * 100).toFixed(2)}%
Avg Drawdown:           ${(m.averageDrawdown * 100).toFixed(2)}%
DD Duration (days):     ${m.drawdownDuration}

RISK-ADJUSTED RETURNS
──────────────────────────────────────────────────────────────────────
Sharpe Ratio:           ${m.sharpeRatio.toFixed(2)}           ${b.sharpeRatio.toFixed(2)}
Sortino Ratio:          ${m.sortinoRatio.toFixed(2)}           ${b.sortinoRatio.toFixed(2)}
Calmar Ratio:           ${m.calmarRatio.toFixed(2)}           ${b.calmarRatio.toFixed(2)}
Information Ratio:      ${m.informationRatio?.toFixed(2) || 'N/A'}

TRADE STATISTICS
──────────────────────────────────────────────────────────────────────
Win Rate:               ${(m.winRate * 100).toFixed(1)}%
Profit Factor:          ${m.profitFactor.toFixed(2)}
Avg Win:                ${(m.averageWin * 100).toFixed(2)}%
Avg Loss:               ${(m.averageLoss * 100).toFixed(2)}%
Payoff Ratio:           ${m.payoffRatio.toFixed(2)}
Expectancy:             ${(m.expectancy * 100).toFixed(3)}%

TAIL RISK
──────────────────────────────────────────────────────────────────────
VaR (95%):              ${(m.var95 * 100).toFixed(2)}%
VaR (99%):              ${(m.var99 * 100).toFixed(2)}%
CVaR (95%):             ${(m.cvar95 * 100).toFixed(2)}%
Skewness:               ${m.skewness.toFixed(2)}
Kurtosis:               ${m.kurtosis.toFixed(2)}

TRADING ACTIVITY
──────────────────────────────────────────────────────────────────────
Total Trades:           ${t.totalTrades}
Buy Trades:             ${t.buyTrades}
Sell Trades:            ${t.sellTrades}
Total Volume:           $${t.totalVolume.toFixed(2)}
Total Costs:            $${t.totalCosts.toFixed(2)}
Avg Trade Size:         $${(t.avgTradeSize || 0).toFixed(2)}

ADDITIONAL METRICS
──────────────────────────────────────────────────────────────────────
Trading Days:           ${m.tradingDays}
Best Day:               ${(m.bestDay * 100).toFixed(2)}%
Worst Day:              ${(m.worstDay * 100).toFixed(2)}%
Positive Months:        ${(m.positiveMonths * 100).toFixed(1)}%

══════════════════════════════════════════════════════════════════════
`;
  }
}

/**
 * Walk-Forward Analysis
 */
class WalkForwardAnalyzer {
  constructor(config = {}) {
    this.trainRatio = config.trainRatio || 0.7;
    this.numFolds = config.numFolds || 5;
    this.engine = new BacktestEngine();
  }

  async analyze(historicalData, options = {}) {
    const foldSize = Math.floor(historicalData.length / this.numFolds);
    const results = [];

    for (let i = 0; i < this.numFolds; i++) {
      const testStart = i * foldSize;
      const testEnd = (i + 1) * foldSize;
      const trainEnd = Math.floor(testStart * this.trainRatio);

      // In-sample (training) period
      const trainData = historicalData.slice(0, trainEnd);

      // Out-of-sample (test) period
      const testData = historicalData.slice(testStart, testEnd);

      // Run backtest on test period
      const foldResult = await this.engine.run(testData, options);

      results.push({
        fold: i + 1,
        trainPeriod: { start: 0, end: trainEnd },
        testPeriod: { start: testStart, end: testEnd },
        metrics: foldResult.metrics
      });
    }

    // Aggregate results
    const avgSharpe = results.reduce((a, r) => a + r.metrics.sharpeRatio, 0) / results.length;
    const avgReturn = results.reduce((a, r) => a + r.metrics.totalReturn, 0) / results.length;

    return {
      folds: results,
      aggregate: {
        avgSharpe,
        avgReturn,
        consistency: this.calculateConsistency(results)
      }
    };
  }

  calculateConsistency(results) {
    const profitableFolds = results.filter(r => r.metrics.totalReturn > 0).length;
    return profitableFolds / results.length;
  }
}

// Exports
export {
  BacktestEngine,
  PerformanceMetrics,
  WalkForwardAnalyzer,
  backtestConfig
};

// Demo if run directly
const isMainModule = import.meta.url === `file://${process.argv[1]}`;
if (isMainModule) {
  console.log('══════════════════════════════════════════════════════════════════════');
  console.log('BACKTESTING FRAMEWORK DEMO');
  console.log('══════════════════════════════════════════════════════════════════════\n');

  // Generate synthetic historical data
  const generateHistoricalData = (days) => {
    const data = [];
    let price = 100;

    for (let i = 0; i < days; i++) {
      const trend = Math.sin(i / 50) * 0.001;  // Cyclical trend
      const noise = (Math.random() - 0.5) * 0.02;  // Random noise
      const change = trend + noise;

      price *= (1 + change);

      data.push({
        date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000),
        open: price * (1 - Math.random() * 0.005),
        high: price * (1 + Math.random() * 0.01),
        low: price * (1 - Math.random() * 0.01),
        close: price,
        volume: 1000000 * (0.5 + Math.random())
      });
    }

    return data;
  };

  const historicalData = generateHistoricalData(500);

  console.log('1. Data Summary:');
  console.log('──────────────────────────────────────────────────────────────────────');
  console.log(`   Days: ${historicalData.length}`);
  console.log(`   Start: ${historicalData[0].date.toISOString().split('T')[0]}`);
  console.log(`   End: ${historicalData[historicalData.length-1].date.toISOString().split('T')[0]}`);
  console.log(`   Start Price: $${historicalData[0].close.toFixed(2)}`);
  console.log(`   End Price: $${historicalData[historicalData.length-1].close.toFixed(2)}`);
  console.log();

  const engine = new BacktestEngine();

  console.log('2. Running Backtest...');
  console.log('──────────────────────────────────────────────────────────────────────');

  engine.run(historicalData, {
    symbols: ['TEST'],
    newsData: [
      { symbol: 'TEST', text: 'Strong growth reported in quarterly earnings', source: 'news' },
      { symbol: 'TEST', text: 'Analyst upgrades stock to buy rating', source: 'analyst' }
    ]
  }).then(results => {
    console.log(engine.generateReport(results));

    console.log('3. Equity Curve Summary:');
    console.log('──────────────────────────────────────────────────────────────────────');
    console.log(`   Initial: $${results.equityCurve[0].toFixed(2)}`);
    console.log(`   Final: $${results.equityCurve[results.equityCurve.length-1].toFixed(2)}`);
    console.log(`   Peak: $${Math.max(...results.equityCurve).toFixed(2)}`);
    console.log(`   Trough: $${Math.min(...results.equityCurve).toFixed(2)}`);

    console.log();
    console.log('══════════════════════════════════════════════════════════════════════');
    console.log('Backtesting demo completed');
    console.log('══════════════════════════════════════════════════════════════════════');
  }).catch(err => {
    console.error('Backtest error:', err);
  });
}
