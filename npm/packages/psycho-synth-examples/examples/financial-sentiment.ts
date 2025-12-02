/**
 * Financial Sentiment & Risk Analysis with Psycho-Symbolic Reasoning
 *
 * Demonstrates:
 * - Market sentiment extraction from news/reports
 * - Investor preference and risk tolerance analysis
 * - Fear/greed emotional indexing
 * - Portfolio personality profiling
 * - Synthetic investor persona generation
 * - Trading psychology insights
 */

import { quickStart } from 'psycho-symbolic-integration';

async function analyzeFinancialSentiment() {
  console.log('💹 Financial Sentiment & Risk Analysis\n');
  console.log('='.repeat(70));

  const system = await quickStart(process.env.GEMINI_API_KEY);

  // ============================================================================
  // PART 1: Market News Sentiment Analysis
  // ============================================================================
  console.log('\n📰 PART 1: Real-Time Market News Sentiment (0.4ms per headline)\n');

  const marketNews = [
    "Markets rally on positive economic data and strong earnings reports",
    "Investors cautious amid rising inflation concerns and uncertainty",
    "Tech stocks plunge as regulatory fears intensify globally",
    "Central bank signals potential interest rate cuts - markets surge",
    "Economic downturn fears trigger widespread market selloff",
    "Record highs reached as investor confidence remains strong",
    "Volatility spikes amid geopolitical tensions and trade disputes",
    "Analysts upgrade forecasts following better than expected GDP growth"
  ];

  const newsAnalysis = [];

  for (let i = 0; i < marketNews.length; i++) {
    const headline = marketNews[i];
    const sentiment = await system.reasoner.extractSentiment(headline);

    newsAnalysis.push({
      headline,
      sentiment: sentiment.score,
      emotion: sentiment.primaryEmotion,
      confidence: sentiment.confidence,
      marketImpact: sentiment.score > 0.5 ? 'bullish' : sentiment.score < -0.5 ? 'bearish' : 'neutral'
    });

    console.log(`📰 News ${i + 1}: "${headline}"`);
    console.log(`   Sentiment: ${sentiment.score.toFixed(2)} (${sentiment.primaryEmotion})`);
    console.log(`   Market impact: ${newsAnalysis[i].marketImpact.toUpperCase()}`);
    console.log(`   Confidence: ${(sentiment.confidence * 100).toFixed(0)}%`);
    console.log('');
  }

  // Calculate market sentiment index
  const avgMarketSentiment = newsAnalysis.reduce((sum, n) => sum + n.sentiment, 0) / newsAnalysis.length;
  const bullishNews = newsAnalysis.filter(n => n.marketImpact === 'bullish').length;
  const bearishNews = newsAnalysis.filter(n => n.marketImpact === 'bearish').length;

  console.log('📊 Market Sentiment Index:');
  console.log(`   Overall sentiment: ${avgMarketSentiment.toFixed(2)} ${avgMarketSentiment > 0 ? '(Optimistic)' : '(Pessimistic)'}`);
  console.log(`   Bullish news: ${bullishNews} (${(bullishNews / newsAnalysis.length * 100).toFixed(0)}%)`);
  console.log(`   Bearish news: ${bearishNews} (${(bearishNews / newsAnalysis.length * 100).toFixed(0)}%)`);

  // ============================================================================
  // PART 2: Investor Psychology Analysis
  // ============================================================================
  console.log('\n\n🧠 PART 2: Investor Preference & Risk Tolerance Analysis\n');

  const investorStatements = [
    "I prefer steady, low-risk investments that preserve capital",
    "I'm willing to take significant risks for higher potential returns",
    "Diversification across multiple asset classes is my priority",
    "I focus on long-term growth and ignore short-term volatility",
    "I get anxious during market downturns and prefer to sell quickly",
    "Value investing and fundamental analysis guide my decisions",
    "I love the excitement of day trading and quick profits"
  ];

  const investorProfiles = [];

  for (let i = 0; i < investorStatements.length; i++) {
    const statement = investorStatements[i];
    const [sentiment, preferences] = await Promise.all([
      system.reasoner.extractSentiment(statement),
      system.reasoner.extractPreferences(statement)
    ]);

    // Calculate risk tolerance
    const riskKeywords = {
      high: ['risks', 'excitement', 'quick', 'trading', 'aggressive'],
      low: ['steady', 'preserve', 'anxious', 'safe', 'conservative']
    };

    const highRiskScore = riskKeywords.high.filter(kw =>
      statement.toLowerCase().includes(kw)
    ).length;

    const lowRiskScore = riskKeywords.low.filter(kw =>
      statement.toLowerCase().includes(kw)
    ).length;

    const riskTolerance = highRiskScore > lowRiskScore ? 'high' :
                         lowRiskScore > highRiskScore ? 'low' : 'medium';

    investorProfiles.push({
      id: `investor_${i + 1}`,
      statement,
      sentiment,
      preferences: preferences.preferences,
      riskTolerance
    });

    console.log(`💼 Investor ${i + 1}:`);
    console.log(`   Statement: "${statement}"`);
    console.log(`   Sentiment: ${sentiment.score.toFixed(2)} (${sentiment.primaryEmotion})`);
    console.log(`   Risk tolerance: ${riskTolerance.toUpperCase()}`);

    if (preferences.preferences.length > 0) {
      console.log(`   Investment preferences:`);
      preferences.preferences.slice(0, 2).forEach((pref: any) => {
        console.log(`     - ${pref.type}: "${pref.subject}" (strength: ${pref.strength.toFixed(2)})`);
      });
    }
    console.log('');
  }

  // ============================================================================
  // PART 3: Fear & Greed Emotional Index
  // ============================================================================
  console.log('\n😱💰 PART 3: Fear & Greed Emotional Index\n');

  // Analyze emotional states from market commentary
  const fearIndicators = newsAnalysis.filter(n =>
    ['fear', 'anxious', 'worried', 'panic'].includes(n.emotion)
  ).length;

  const greedIndicators = newsAnalysis.filter(n =>
    ['excited', 'optimistic', 'confident', 'euphoric'].includes(n.emotion)
  ).length;

  const fearGreedIndex = ((greedIndicators - fearIndicators) / newsAnalysis.length + 1) * 50;

  console.log(`Fear & Greed Index: ${fearGreedIndex.toFixed(0)}/100`);
  console.log(`   Interpretation: ${
    fearGreedIndex > 75 ? 'EXTREME GREED (Caution advised)' :
    fearGreedIndex > 60 ? 'Greed' :
    fearGreedIndex > 40 ? 'Neutral' :
    fearGreedIndex > 25 ? 'Fear' :
    'EXTREME FEAR (Potential opportunity)'
  }`);
  console.log(`   Fear indicators: ${fearIndicators}`);
  console.log(`   Greed indicators: ${greedIndicators}`);

  // ============================================================================
  // PART 4: Generate Synthetic Investor Personas
  // ============================================================================
  console.log('\n\n🎲 PART 4: Generate Synthetic Investor Personas\n');

  console.log('Generating 50 synthetic investor personas for portfolio modeling...\n');

  const syntheticInvestors = await system.generateIntelligently('structured', {
    count: 50,
    schema: {
      investor_id: { type: 'string', required: true },
      age: { type: 'number', min: 25, max: 70, required: true },
      investment_experience: {
        type: 'enum',
        enum: ['beginner', 'intermediate', 'advanced', 'expert'],
        required: true
      },
      risk_tolerance: {
        type: 'enum',
        enum: ['very_conservative', 'conservative', 'moderate', 'aggressive', 'very_aggressive'],
        required: true
      },
      investment_style: {
        type: 'enum',
        enum: ['value', 'growth', 'income', 'index', 'day_trader', 'swing_trader'],
        required: true
      },
      emotional_bias: {
        type: 'enum',
        enum: ['loss_aversion', 'overconfidence', 'herd_mentality', 'confirmation_bias', 'balanced'],
        required: true
      },
      portfolio_size: { type: 'number', min: 10000, max: 5000000, required: true },
      time_horizon: {
        type: 'enum',
        enum: ['short_term', 'medium_term', 'long_term'],
        required: true
      },
      volatility_tolerance: { type: 'number', min: 0, max: 1, required: true },
      panic_sell_probability: { type: 'number', min: 0, max: 1, required: true },
      primary_investment_goals: { type: 'array', required: true }
    }
  }, {
    targetSentiment: {
      score: 0.0, // Neutral - diverse investor psychology
      emotion: 'analytical'
    },
    userPreferences: investorStatements,
    contextualFactors: {
      environment: 'financial_markets',
      constraints: ['portfolio_size >= 10000']
    },
    qualityThreshold: 0.89
  });

  console.log(`✅ Generated ${syntheticInvestors.data.length} synthetic investor personas`);
  console.log(`📊 Generation Quality:`);
  console.log(`   Preference alignment: ${(syntheticInvestors.psychoMetrics.preferenceAlignment * 100).toFixed(1)}%`);
  console.log(`   Quality score: ${(syntheticInvestors.psychoMetrics.qualityScore * 100).toFixed(1)}%`);

  // ============================================================================
  // PART 5: Portfolio Psychology Analysis
  // ============================================================================
  console.log('\n\n📈 PART 5: Portfolio Psychology Distribution\n');

  const psychologyStats = {
    riskTolerance: new Map<string, number>(),
    emotionalBias: new Map<string, number>(),
    investmentStyle: new Map<string, number>(),
    highPanicSell: syntheticInvestors.data.filter((i: any) => i.panic_sell_probability > 0.6).length
  };

  syntheticInvestors.data.forEach((investor: any) => {
    // Risk tolerance
    const riskCount = psychologyStats.riskTolerance.get(investor.risk_tolerance) || 0;
    psychologyStats.riskTolerance.set(investor.risk_tolerance, riskCount + 1);

    // Emotional bias
    const biasCount = psychologyStats.emotionalBias.get(investor.emotional_bias) || 0;
    psychologyStats.emotionalBias.set(investor.emotional_bias, biasCount + 1);

    // Investment style
    const styleCount = psychologyStats.investmentStyle.get(investor.investment_style) || 0;
    psychologyStats.investmentStyle.set(investor.investment_style, styleCount + 1);
  });

  console.log('Risk Tolerance Distribution:');
  Array.from(psychologyStats.riskTolerance.entries())
    .sort((a, b) => b[1] - a[1])
    .forEach(([risk, count]) => {
      const pct = (count / syntheticInvestors.data.length * 100).toFixed(1);
      console.log(`   ${risk}: ${count} (${pct}%)`);
    });

  console.log('\nEmotional Bias Distribution:');
  Array.from(psychologyStats.emotionalBias.entries())
    .sort((a, b) => b[1] - a[1])
    .forEach(([bias, count]) => {
      const pct = (count / syntheticInvestors.data.length * 100).toFixed(1);
      console.log(`   ${bias}: ${count} (${pct}%)`);
    });

  console.log(`\n⚠️  High panic-sell risk investors: ${psychologyStats.highPanicSell} (${(psychologyStats.highPanicSell / syntheticInvestors.data.length * 100).toFixed(1)}%)`);

  // ============================================================================
  // PART 6: Trading Psychology Insights
  // ============================================================================
  console.log('\n\n🎯 PART 6: Trading Psychology Insights\n');

  // Group by emotional bias
  const biasGroups = {
    loss_aversion: syntheticInvestors.data.filter((i: any) => i.emotional_bias === 'loss_aversion'),
    overconfidence: syntheticInvestors.data.filter((i: any) => i.emotional_bias === 'overconfidence'),
    herd_mentality: syntheticInvestors.data.filter((i: any) => i.emotional_bias === 'herd_mentality')
  };

  Object.entries(biasGroups).forEach(([bias, investors]: [string, any]) => {
    if (investors.length === 0) return;

    const avgVolatilityTolerance = investors.reduce((sum: number, i: any) =>
      sum + i.volatility_tolerance, 0) / investors.length;

    const avgPanicSell = investors.reduce((sum: number, i: any) =>
      sum + i.panic_sell_probability, 0) / investors.length;

    console.log(`${bias.toUpperCase()} Investors (${investors.length}):`);
    console.log(`   Avg volatility tolerance: ${(avgVolatilityTolerance * 100).toFixed(0)}%`);
    console.log(`   Avg panic-sell probability: ${(avgPanicSell * 100).toFixed(0)}%`);
    console.log(`   Recommended strategy: ${
      bias === 'loss_aversion' ? 'Conservative portfolio with capital preservation' :
      bias === 'overconfidence' ? 'Risk management and diversification education' :
      'Contrarian indicators and independent analysis'
    }`);
    console.log('');
  });

  // ============================================================================
  // PART 7: Sample Investor Profiles
  // ============================================================================
  console.log('\n📋 PART 7: Sample Investor Psychological Profiles\n');

  syntheticInvestors.data.slice(0, 3).forEach((investor: any, idx: number) => {
    console.log(`Investor ${idx + 1}:`);
    console.log(`   ID: ${investor.investor_id}`);
    console.log(`   Age: ${investor.age}`);
    console.log(`   Experience: ${investor.investment_experience}`);
    console.log(`   Risk tolerance: ${investor.risk_tolerance}`);
    console.log(`   Investment style: ${investor.investment_style}`);
    console.log(`   Emotional bias: ${investor.emotional_bias}`);
    console.log(`   Portfolio size: $${investor.portfolio_size.toLocaleString()}`);
    console.log(`   Time horizon: ${investor.time_horizon}`);
    console.log(`   Volatility tolerance: ${(investor.volatility_tolerance * 100).toFixed(0)}%`);
    console.log(`   Panic-sell risk: ${(investor.panic_sell_probability * 100).toFixed(0)}%`);
    console.log('');
  });

  // ============================================================================
  // PART 8: Market Recommendations
  // ============================================================================
  console.log('\n💡 PART 8: Psychological Market Recommendations\n');

  console.log('Based on sentiment and investor psychology analysis:\n');

  const recommendations = [
    `📊 Market sentiment: ${avgMarketSentiment > 0 ? 'BULLISH' : 'BEARISH'} (${avgMarketSentiment.toFixed(2)})`,
    `😱 Fear & Greed Index: ${fearGreedIndex.toFixed(0)}/100 - ${fearGreedIndex > 70 ? 'Consider profit-taking' : fearGreedIndex < 30 ? 'Potential buying opportunity' : 'Balanced market'}`,
    `⚠️  ${psychologyStats.highPanicSell} investors at high panic-sell risk - volatility ahead`,
    `🎯 Dominant investor bias: ${Array.from(psychologyStats.emotionalBias.entries()).sort((a, b) => b[1] - a[1])[0][0]}`,
    `💼 Most common strategy: ${Array.from(psychologyStats.investmentStyle.entries()).sort((a, b) => b[1] - a[1])[0][0]}`,
    `📈 For conservative investors: Focus on capital preservation given ${bearishNews} bearish signals`,
    `🚀 For aggressive investors: ${bullishNews} bullish signals suggest growth opportunities`
  ];

  recommendations.forEach(rec => console.log(rec));

  console.log('\n✅ Financial Sentiment Analysis Complete!');

  await system.shutdown();
}

// Run the analysis
analyzeFinancialSentiment().catch(console.error);
