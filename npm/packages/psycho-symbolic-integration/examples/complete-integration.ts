/**
 * Complete Integration Example
 *
 * Demonstrates the full power of combining:
 * - Psycho-Symbolic Reasoner (0.3ms symbolic reasoning)
 * - Ruvector (vector database)
 * - Agentic-Synth (AI data generation)
 *
 * This example shows:
 * 1. Loading a knowledge base
 * 2. Hybrid symbolic+vector queries
 * 3. Psychologically-guided data generation
 * 4. Sentiment and preference analysis
 * 5. Goal-oriented planning
 */

import { IntegratedPsychoSymbolicSystem, quickStart } from '../src/index.js';

async function main() {
  console.log('🎯 Integrated Psycho-Symbolic System - Complete Example\n');
  console.log('='.repeat(60));

  // ============================================================================
  // STEP 1: Initialize the system
  // ============================================================================
  console.log('\n📦 Step 1: Initializing integrated system...\n');

  const system = await quickStart(process.env.GEMINI_API_KEY);

  console.log('✅ System initialized with all components');
  console.log(JSON.stringify(system.getSystemInsights(), null, 2));

  // ============================================================================
  // STEP 2: Load knowledge base for reasoning
  // ============================================================================
  console.log('\n📚 Step 2: Loading wellness knowledge base...\n');

  const wellnessKnowledgeBase = {
    nodes: [
      {
        id: 'stress',
        type: 'emotion',
        properties: {
          valence: -0.7,
          arousal: 0.8,
          category: 'negative'
        }
      },
      {
        id: 'anxiety',
        type: 'emotion',
        properties: {
          valence: -0.6,
          arousal: 0.9,
          category: 'negative'
        }
      },
      {
        id: 'meditation',
        type: 'activity',
        properties: {
          duration: 15,
          energy: 'low',
          category: 'mindfulness',
          effectiveness: 0.85
        }
      },
      {
        id: 'exercise',
        type: 'activity',
        properties: {
          duration: 30,
          energy: 'high',
          category: 'physical',
          effectiveness: 0.78
        }
      },
      {
        id: 'deep_breathing',
        type: 'technique',
        properties: {
          duration: 5,
          difficulty: 'easy',
          category: 'mindfulness',
          effectiveness: 0.92
        }
      },
      {
        id: 'journaling',
        type: 'activity',
        properties: {
          duration: 20,
          energy: 'low',
          category: 'cognitive',
          effectiveness: 0.75
        }
      }
    ],
    edges: [
      {
        from: 'meditation',
        to: 'stress',
        relationship: 'reduces',
        weight: 0.85
      },
      {
        from: 'meditation',
        to: 'anxiety',
        relationship: 'reduces',
        weight: 0.80
      },
      {
        from: 'exercise',
        to: 'stress',
        relationship: 'reduces',
        weight: 0.78
      },
      {
        from: 'deep_breathing',
        to: 'stress',
        relationship: 'reduces',
        weight: 0.92
      },
      {
        from: 'deep_breathing',
        to: 'anxiety',
        relationship: 'reduces',
        weight: 0.88
      },
      {
        from: 'journaling',
        to: 'stress',
        relationship: 'reduces',
        weight: 0.75
      }
    ]
  };

  await system.loadKnowledgeBase(wellnessKnowledgeBase);
  console.log('✅ Knowledge base loaded into symbolic and vector stores');

  // ============================================================================
  // STEP 3: Intelligent hybrid queries
  // ============================================================================
  console.log('\n🔍 Step 3: Performing hybrid reasoning queries...\n');

  const queries = [
    'Find quick techniques for reducing anxiety',
    'What activities help with stress management?',
    'Show me mindfulness practices'
  ];

  for (const query of queries) {
    console.log(`Query: "${query}"`);
    const results = await system.intelligentQuery(query, {
      symbolicWeight: 0.6,
      vectorWeight: 0.4,
      maxResults: 3
    });

    console.log(`Found ${results.length} results:`);
    results.forEach((result: any, idx: number) => {
      console.log(`  ${idx + 1}. ${result.nodes[0]?.id || 'unknown'}`);
      console.log(`     Combined score: ${result.reasoning.combinedScore.toFixed(3)}`);
      console.log(`     (symbolic: ${result.reasoning.symbolicMatch.toFixed(2)}, semantic: ${result.reasoning.semanticMatch.toFixed(2)})`);
    });
    console.log('');
  }

  // ============================================================================
  // STEP 4: Analyze text for sentiment and preferences
  // ============================================================================
  console.log('\n😊 Step 4: Analyzing user text for insights...\n');

  const userTexts = [
    "I'm feeling overwhelmed with work and need quick stress relief",
    "I prefer gentle exercises that don't take too much time",
    "Meditation helps me focus, but I struggle to maintain consistency"
  ];

  for (const text of userTexts) {
    console.log(`Text: "${text}"`);
    const analysis = await system.analyzeText(text);

    console.log(`  Sentiment:`);
    console.log(`    Score: ${analysis.sentiment.score.toFixed(2)}`);
    console.log(`    Emotion: ${analysis.sentiment.primaryEmotion}`);
    console.log(`    Confidence: ${(analysis.sentiment.confidence * 100).toFixed(1)}%`);

    if (analysis.preferences.preferences.length > 0) {
      console.log(`  Preferences:`);
      analysis.preferences.preferences.forEach((pref: any, idx: number) => {
        console.log(`    ${idx + 1}. ${pref.type}: "${pref.subject}" (strength: ${pref.strength.toFixed(2)})`);
      });
    }
    console.log('');
  }

  // ============================================================================
  // STEP 5: Plan data generation strategy
  // ============================================================================
  console.log('\n🎯 Step 5: Planning data generation strategy with GOAP...\n');

  const generationGoal = 'Generate 100 wellness activity records optimized for stress reduction';
  const constraints = {
    targetQuality: 0.9,
    maxDuration: 30, // minutes per activity
    preferredCategories: ['mindfulness', 'cognitive']
  };

  const plan = await system.planDataGeneration(generationGoal, constraints);

  console.log(`Goal: ${generationGoal}`);
  console.log(`Plan details:`);
  console.log(`  Steps: ${plan.steps.length}`);
  console.log(`  Estimated time: ${plan.estimatedTime}ms`);
  console.log(`  Estimated quality: ${(plan.estimatedQuality * 100).toFixed(1)}%`);

  if (plan.recommendations.length > 0) {
    console.log(`  Recommendations:`);
    plan.recommendations.forEach((rec: string, idx: number) => {
      console.log(`    ${idx + 1}. ${rec}`);
    });
  }

  // ============================================================================
  // STEP 6: Generate synthetic data with psychological guidance
  // ============================================================================
  console.log('\n🎲 Step 6: Generating psychologically-guided synthetic data...\n');

  const generationResult = await system.generateIntelligently(
    'structured',
    {
      count: 20,
      schema: {
        activity_name: { type: 'string', required: true },
        category: {
          type: 'enum',
          enum: ['mindfulness', 'physical', 'cognitive', 'social'],
          required: true
        },
        duration_minutes: { type: 'number', min: 5, max: 60 },
        difficulty: {
          type: 'enum',
          enum: ['easy', 'medium', 'hard']
        },
        stress_reduction_score: { type: 'number', min: 0, max: 1 },
        description: { type: 'string' }
      }
    },
    {
      targetSentiment: {
        score: 0.7, // Positive sentiment
        emotion: 'calm'
      },
      userPreferences: [
        'I prefer activities that are easy to start',
        'I like quick results',
        'I value mindfulness practices'
      ],
      contextualFactors: {
        emotionalState: 'stressed',
        environment: 'home',
        constraints: ['duration_minutes <= 30', 'difficulty != hard']
      },
      qualityThreshold: 0.8
    }
  );

  console.log(`Generated ${generationResult.data.length} wellness activities`);
  console.log(`\nPsycho-metrics:`);
  console.log(`  Preference alignment: ${(generationResult.psychoMetrics.preferenceAlignment * 100).toFixed(1)}%`);
  console.log(`  Sentiment match: ${(generationResult.psychoMetrics.sentimentMatch * 100).toFixed(1)}%`);
  console.log(`  Contextual fit: ${(generationResult.psychoMetrics.contextualFit * 100).toFixed(1)}%`);
  console.log(`  Quality score: ${(generationResult.psychoMetrics.qualityScore * 100).toFixed(1)}%`);

  if (generationResult.suggestions.length > 0) {
    console.log(`\nGeneration suggestions:`);
    generationResult.suggestions.forEach((suggestion: string, idx: number) => {
      console.log(`  ${idx + 1}. ${suggestion}`);
    });
  }

  console.log(`\nSample generated activities:`);
  generationResult.data.slice(0, 5).forEach((activity: any, idx: number) => {
    console.log(`\n  ${idx + 1}. ${activity.activity_name}`);
    console.log(`     Category: ${activity.category}`);
    console.log(`     Duration: ${activity.duration_minutes} minutes`);
    console.log(`     Difficulty: ${activity.difficulty}`);
    console.log(`     Stress reduction: ${(activity.stress_reduction_score * 100).toFixed(0)}%`);

    if (activity._psychoMetrics) {
      console.log(`     Sentiment: ${activity._psychoMetrics.sentimentScore.toFixed(2)} (${activity._psychoMetrics.emotion})`);
    }
  });

  // ============================================================================
  // STEP 7: System insights and cleanup
  // ============================================================================
  console.log('\n\n📊 Step 7: System insights and performance...\n');

  const insights = system.getSystemInsights();
  console.log('System status:');
  console.log(JSON.stringify(insights, null, 2));

  console.log('\n🧹 Cleaning up...');
  await system.shutdown();

  console.log('\n✨ Example complete!');
  console.log('\n' + '='.repeat(60));
  console.log('\n🎉 Key Takeaways:');
  console.log('  ✅ Sub-millisecond symbolic reasoning');
  console.log('  ✅ Hybrid symbolic + vector queries');
  console.log('  ✅ Psychological analysis of text');
  console.log('  ✅ Goal-oriented planning (GOAP)');
  console.log('  ✅ Sentiment-guided data generation');
  console.log('  ✅ Preference-aware synthetic data');
  console.log('\n💡 This demonstrates the power of combining:');
  console.log('   • Fast symbolic reasoning (psycho-symbolic-reasoner)');
  console.log('   • Semantic vector search (ruvector)');
  console.log('   • AI data generation (agentic-synth)');
  console.log('\n🚀 Ready for production use!');
}

// Run the example
main().catch(console.error);
