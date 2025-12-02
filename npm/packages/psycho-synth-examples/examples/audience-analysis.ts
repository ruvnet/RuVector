/**
 * Audience Analysis with Psycho-Symbolic Reasoning
 *
 * Demonstrates:
 * - Real-time sentiment extraction from audience feedback (0.4ms)
 * - Preference profiling and segmentation
 * - Psychographic clustering
 * - Engagement prediction modeling
 * - Synthetic audience data generation
 */

import { quickStart } from 'psycho-symbolic-integration';

interface AudienceMember {
  id: string;
  feedback: string;
  sentiment?: any;
  preferences?: any[];
  psychographicProfile?: any;
  engagementPrediction?: number;
}

async function analyzeAudience() {
  console.log('🎭 Audience Analysis with Psycho-Symbolic Reasoning\n');
  console.log('='.repeat(70));

  const system = await quickStart(process.env.GEMINI_API_KEY);

  // ============================================================================
  // PART 1: Real Audience Sentiment Analysis (Ultra-Fast)
  // ============================================================================
  console.log('\n📊 PART 1: Real-Time Sentiment Analysis (0.4ms per analysis)\n');

  const audienceFeedback = [
    "This content is engaging but could be more concise",
    "I love the interactive elements! Very innovative approach",
    "The pacing feels rushed, I prefer slower, deeper dives",
    "Not relevant to my interests, seems too technical",
    "Brilliant insights! I'd love to see more practical examples",
    "The presentation style is too formal for my taste",
    "Fascinating topic, but needs better visual aids",
    "This is exactly what I was looking for - actionable advice!"
  ];

  const analyzedAudience: AudienceMember[] = [];

  console.log('Analyzing feedback from 8 audience members...\n');

  for (let i = 0; i < audienceFeedback.length; i++) {
    const feedback = audienceFeedback[i];

    const [sentiment, preferences] = await Promise.all([
      system.reasoner.extractSentiment(feedback),
      system.reasoner.extractPreferences(feedback)
    ]);

    analyzedAudience.push({
      id: `audience_${i + 1}`,
      feedback,
      sentiment,
      preferences: preferences.preferences
    });

    console.log(`👤 Audience Member ${i + 1}:`);
    console.log(`   Feedback: "${feedback}"`);
    console.log(`   Sentiment: ${sentiment.score.toFixed(2)} (${sentiment.primaryEmotion})`);
    console.log(`   Confidence: ${(sentiment.confidence * 100).toFixed(1)}%`);

    if (preferences.preferences.length > 0) {
      console.log(`   Preferences detected: ${preferences.preferences.length}`);
      preferences.preferences.forEach((pref: any) => {
        console.log(`     - ${pref.type}: "${pref.subject}" (strength: ${pref.strength.toFixed(2)})`);
      });
    }
    console.log('');
  }

  // ============================================================================
  // PART 2: Psychographic Profiling
  // ============================================================================
  console.log('\n🧠 PART 2: Psychographic Audience Segmentation\n');

  const segments = {
    enthusiasts: analyzedAudience.filter(a => a.sentiment!.score > 0.5),
    critics: analyzedAudience.filter(a => a.sentiment!.score < -0.2),
    neutrals: analyzedAudience.filter(a =>
      a.sentiment!.score >= -0.2 && a.sentiment!.score <= 0.5
    )
  };

  console.log(`📈 Segment Distribution:`);
  console.log(`   Enthusiasts (positive): ${segments.enthusiasts.length} (${(segments.enthusiasts.length / analyzedAudience.length * 100).toFixed(1)}%)`);
  console.log(`   Critics (negative): ${segments.critics.length} (${(segments.critics.length / analyzedAudience.length * 100).toFixed(1)}%)`);
  console.log(`   Neutrals: ${segments.neutrals.length} (${(segments.neutrals.length / analyzedAudience.length * 100).toFixed(1)}%)`);

  // Extract common preferences per segment
  console.log('\n🎯 Segment Characteristics:\n');

  for (const [segmentName, members] of Object.entries(segments)) {
    if (members.length === 0) continue;

    const allPreferences = members.flatMap(m => m.preferences || []);
    const avgSentiment = members.reduce((sum, m) => sum + m.sentiment!.score, 0) / members.length;

    console.log(`${segmentName.toUpperCase()}:`);
    console.log(`   Average sentiment: ${avgSentiment.toFixed(2)}`);
    console.log(`   Total preferences: ${allPreferences.length}`);

    if (allPreferences.length > 0) {
      const topPrefs = allPreferences
        .sort((a, b) => b.strength - a.strength)
        .slice(0, 3);
      console.log(`   Top preferences:`);
      topPrefs.forEach((pref, idx) => {
        console.log(`     ${idx + 1}. ${pref.type}: "${pref.subject}" (${pref.strength.toFixed(2)})`);
      });
    }
    console.log('');
  }

  // ============================================================================
  // PART 3: Generate Synthetic Audience Personas
  // ============================================================================
  console.log('\n🎲 PART 3: Generate Synthetic Audience Personas\n');

  console.log('Generating 20 synthetic audience personas based on real patterns...\n');

  // Create preference profiles for each segment
  const enthusiastPreferences = [
    "I love innovative and interactive content",
    "Practical examples are very valuable to me",
    "I prefer engaging and actionable insights"
  ];

  const criticPreferences = [
    "I prefer slower, more detailed explanations",
    "Content should be highly relevant to my specific needs",
    "I value traditional presentation styles"
  ];

  const syntheticPersonas = await system.generateIntelligently('structured', {
    count: 20,
    schema: {
      persona_id: { type: 'string', required: true },
      name: { type: 'string', required: true },
      age_group: {
        type: 'enum',
        enum: ['18-24', '25-34', '35-44', '45-54', '55+'],
        required: true
      },
      engagement_level: {
        type: 'enum',
        enum: ['low', 'medium', 'high', 'very_high'],
        required: true
      },
      content_preferences: { type: 'array', required: true },
      learning_style: {
        type: 'enum',
        enum: ['visual', 'auditory', 'kinesthetic', 'reading'],
        required: true
      },
      pain_points: { type: 'array', required: true },
      engagement_prediction: { type: 'number', min: 0, max: 1, required: true }
    }
  }, {
    targetSentiment: {
      score: 0.3, // Mixed audience
      emotion: 'interested'
    },
    userPreferences: [
      ...enthusiastPreferences,
      ...criticPreferences
    ],
    contextualFactors: {
      environment: 'digital_content',
      constraints: ['engagement_prediction >= 0.3']
    },
    qualityThreshold: 0.85
  });

  console.log(`✅ Generated ${syntheticPersonas.data.length} synthetic personas`);
  console.log(`📊 Quality Metrics:`);
  console.log(`   Preference alignment: ${(syntheticPersonas.psychoMetrics.preferenceAlignment * 100).toFixed(1)}%`);
  console.log(`   Sentiment match: ${(syntheticPersonas.psychoMetrics.sentimentMatch * 100).toFixed(1)}%`);
  console.log(`   Overall quality: ${(syntheticPersonas.psychoMetrics.qualityScore * 100).toFixed(1)}%`);

  console.log('\n📋 Sample Personas:\n');

  syntheticPersonas.data.slice(0, 5).forEach((persona: any, idx: number) => {
    console.log(`${idx + 1}. ${persona.name} (${persona.age_group})`);
    console.log(`   Engagement: ${persona.engagement_level}`);
    console.log(`   Learning style: ${persona.learning_style}`);
    console.log(`   Engagement prediction: ${(persona.engagement_prediction * 100).toFixed(0)}%`);
    console.log(`   Top preference: ${persona.content_preferences?.[0] || 'N/A'}`);
    console.log('');
  });

  // ============================================================================
  // PART 4: Predictive Engagement Modeling
  // ============================================================================
  console.log('\n🔮 PART 4: Predictive Engagement Analysis\n');

  // Analyze engagement factors
  const highEngagement = syntheticPersonas.data.filter(
    (p: any) => p.engagement_prediction > 0.7
  );
  const lowEngagement = syntheticPersonas.data.filter(
    (p: any) => p.engagement_prediction < 0.4
  );

  console.log(`High engagement personas: ${highEngagement.length}`);
  console.log(`Low engagement personas: ${lowEngagement.length}`);

  // Extract common characteristics
  if (highEngagement.length > 0) {
    const learningStyles = highEngagement.reduce((acc: any, p: any) => {
      acc[p.learning_style] = (acc[p.learning_style] || 0) + 1;
      return acc;
    }, {});

    console.log('\n✨ High Engagement Characteristics:');
    console.log(`   Dominant learning styles: ${Object.entries(learningStyles)
      .sort(([, a]: any, [, b]: any) => b - a)
      .slice(0, 2)
      .map(([style]) => style)
      .join(', ')}`);
  }

  // ============================================================================
  // PART 5: Actionable Recommendations
  // ============================================================================
  console.log('\n💡 PART 5: AI-Generated Recommendations\n');

  const avgSentiment = analyzedAudience.reduce(
    (sum, a) => sum + a.sentiment!.score, 0
  ) / analyzedAudience.length;

  console.log('📈 Audience Insights Summary:');
  console.log(`   Overall sentiment: ${avgSentiment.toFixed(2)} (${avgSentiment > 0 ? 'Positive' : 'Needs improvement'})`);
  console.log(`   Total audience analyzed: ${analyzedAudience.length} real + ${syntheticPersonas.data.length} synthetic`);
  console.log(`   Dominant emotions: ${
    Array.from(new Set(analyzedAudience.map(a => a.sentiment!.primaryEmotion))).join(', ')
  }`);

  console.log('\n🎯 Recommendations for Content Optimization:');

  const recommendations = [];

  if (segments.critics.length > segments.enthusiasts.length) {
    recommendations.push('• Address negative feedback: content pacing and relevance');
    recommendations.push('• Increase practical examples and actionable insights');
  } else {
    recommendations.push('• Maintain current engagement strategies');
    recommendations.push('• Scale interactive and innovative elements');
  }

  if (analyzedAudience.some(a => a.preferences?.some(p => p.subject.includes('visual')))) {
    recommendations.push('• Enhance visual aids and presentations');
  }

  recommendations.forEach(rec => console.log(rec));

  console.log('\n✨ Analysis Complete!');

  await system.shutdown();
}

// Run the analysis
analyzeAudience().catch(console.error);
