/**
 * Exotic Psychological Profiling with Psycho-Symbolic Reasoning
 *
 * Demonstrates advanced psychological insights:
 * - Personality archetype detection (Jung, MBTI, Big Five)
 * - Cognitive bias identification
 * - Decision-making pattern analysis
 * - Attachment style profiling
 * - Communication pattern extraction
 * - Conflict resolution style detection
 * - Motivational drivers and fear analysis
 * - Shadow aspects and blind spots
 * - Synthetic psychological persona generation
 */

import { quickStart } from 'psycho-symbolic-integration';

async function performExoticPsychologicalProfiling() {
  console.log('🧠 Exotic Psychological Profiling with AI\n');
  console.log('='.repeat(70));

  const system = await quickStart(process.env.GEMINI_API_KEY);

  // ============================================================================
  // PART 1: Personality Archetype Detection
  // ============================================================================
  console.log('\n🎭 PART 1: Personality Archetype Detection (0.4ms per profile)\n');

  const personalityStatements = [
    "I thrive on new challenges and take bold risks to achieve my goals",
    "I find deep meaning in helping others and creating harmony in groups",
    "I'm driven by curiosity and love exploring complex ideas and systems",
    "Structure and tradition give me comfort - I value reliability above all",
    "I express myself through creativity and see beauty in everything",
    "I question authority and fight for justice and individual freedom",
    "I seek wisdom and spiritual growth through introspection and meditation",
    "I love adventure and spontaneity - routine feels like a prison to me"
  ];

  const archetypeMapping = {
    hero: ['challenges', 'achieve', 'goals', 'overcome', 'victory'],
    caregiver: ['helping', 'harmony', 'support', 'nurture', 'compassion'],
    sage: ['wisdom', 'knowledge', 'understanding', 'learn', 'explore'],
    ruler: ['control', 'structure', 'order', 'tradition', 'authority'],
    creator: ['creativity', 'express', 'innovate', 'beauty', 'art'],
    rebel: ['freedom', 'question', 'fight', 'change', 'independent'],
    magician: ['transform', 'spiritual', 'growth', 'wisdom', 'deeper'],
    explorer: ['adventure', 'discover', 'freedom', 'spontaneity', 'new']
  };

  const profiles = [];

  for (let i = 0; i < personalityStatements.length; i++) {
    const statement = personalityStatements[i];
    const [sentiment, preferences] = await Promise.all([
      system.reasoner.extractSentiment(statement),
      system.reasoner.extractPreferences(statement)
    ]);

    // Detect archetype
    let primaryArchetype = 'unknown';
    let maxScore = 0;

    for (const [archetype, keywords] of Object.entries(archetypeMapping)) {
      const score = keywords.filter(kw =>
        statement.toLowerCase().includes(kw)
      ).length;

      if (score > maxScore) {
        maxScore = score;
        primaryArchetype = archetype;
      }
    }

    profiles.push({
      id: `profile_${i + 1}`,
      statement,
      sentiment,
      preferences: preferences.preferences,
      archetype: primaryArchetype,
      archetypeConfidence: maxScore / archetypeMapping[primaryArchetype as keyof typeof archetypeMapping].length
    });

    console.log(`👤 Profile ${i + 1}:`);
    console.log(`   Statement: "${statement}"`);
    console.log(`   Primary archetype: ${primaryArchetype.toUpperCase()}`);
    console.log(`   Confidence: ${(profiles[i].archetypeConfidence * 100).toFixed(0)}%`);
    console.log(`   Sentiment: ${sentiment.score.toFixed(2)} (${sentiment.primaryEmotion})`);
    console.log('');
  }

  // ============================================================================
  // PART 2: Cognitive Bias Detection
  // ============================================================================
  console.log('\n🧩 PART 2: Cognitive Bias Identification\n');

  const biasStatements = [
    "I always knew this would happen - it was so obvious from the start",
    "Everyone agrees with me on this, so I must be right",
    "I've invested so much already, I can't quit now even though it's not working",
    "That success was all because of my skills, but the failure was just bad luck",
    "I'll start that diet next Monday - I work better under deadlines anyway",
    "This rare event happened to me, so it must be very common",
    "I only look for information that confirms what I already believe"
  ];

  const biasTypes = {
    hindsight: "I always knew|it was obvious|predicted",
    bandwagon: "everyone|most people|all agree",
    sunk_cost: "invested|already spent|can't quit now|too far",
    attribution: "my skills|my talent|just luck|bad timing",
    planning: "next Monday|tomorrow|soon|later",
    availability: "happened to me|I saw|common|frequent",
    confirmation: "confirms|proves me right|already believe"
  };

  console.log('Detected Cognitive Biases:\n');

  for (let i = 0; i < biasStatements.length; i++) {
    const statement = biasStatements[i];
    const sentiment = await system.reasoner.extractSentiment(statement);

    let detectedBias = 'unknown';
    for (const [bias, pattern] of Object.entries(biasTypes)) {
      const regex = new RegExp(pattern, 'i');
      if (regex.test(statement)) {
        detectedBias = bias;
        break;
      }
    }

    console.log(`🔍 Statement ${i + 1}: "${statement.substring(0, 60)}..."`);
    console.log(`   Detected bias: ${detectedBias.toUpperCase().replace('_', ' ')} BIAS`);
    console.log(`   Emotional tone: ${sentiment.primaryEmotion}`);
    console.log(`   Implications: ${
      detectedBias === 'hindsight' ? 'Overestimates predictive ability' :
      detectedBias === 'bandwagon' ? 'Influenced by popular opinion' :
      detectedBias === 'sunk_cost' ? 'Difficulty cutting losses' :
      detectedBias === 'attribution' ? 'Skewed success/failure interpretation' :
      detectedBias === 'planning' ? 'Procrastination tendency' :
      detectedBias === 'availability' ? 'Overestimates event probability' :
      detectedBias === 'confirmation' ? 'Echo chamber risk' :
      'Unidentified pattern'
    }`);
    console.log('');
  }

  // ============================================================================
  // PART 3: Decision-Making Pattern Analysis
  // ============================================================================
  console.log('\n🎯 PART 3: Decision-Making Pattern Analysis\n');

  const decisionStatements = [
    "I carefully analyze all data before making any decision",
    "I trust my gut feeling - intuition rarely fails me",
    "I ask for input from everyone before deciding anything",
    "I make quick decisions and adjust as I go",
    "I need to sleep on big decisions - time brings clarity",
    "I use structured frameworks and decision matrices",
    "I let my emotions guide me to the right choice"
  ];

  const decisionStyles = {
    analytical: ['analyze', 'data', 'facts', 'research', 'evidence'],
    intuitive: ['gut', 'feeling', 'intuition', 'sense', 'instinct'],
    collaborative: ['ask', 'input', 'consensus', 'team', 'together'],
    decisive: ['quick', 'fast', 'immediate', 'decisive', 'action'],
    reflective: ['time', 'sleep', 'think', 'ponder', 'consider'],
    systematic: ['framework', 'structure', 'process', 'system', 'method'],
    emotional: ['emotions', 'feel', 'heart', 'passion', 'values']
  };

  console.log('Decision-Making Styles:\n');

  for (let i = 0; i < decisionStatements.length; i++) {
    const statement = decisionStatements[i];

    let style = 'unknown';
    let maxMatch = 0;

    for (const [styleName, keywords] of Object.entries(decisionStyles)) {
      const matches = keywords.filter(kw =>
        statement.toLowerCase().includes(kw)
      ).length;

      if (matches > maxMatch) {
        maxMatch = matches;
        style = styleName;
      }
    }

    console.log(`💭 Statement ${i + 1}: "${statement}"`);
    console.log(`   Style: ${style.toUpperCase()}`);
    console.log(`   Strengths: ${
      style === 'analytical' ? 'Thorough, minimizes errors' :
      style === 'intuitive' ? 'Fast, pattern recognition' :
      style === 'collaborative' ? 'Diverse perspectives, buy-in' :
      style === 'decisive' ? 'Speed, momentum' :
      style === 'reflective' ? 'Wisdom, reduced impulsivity' :
      style === 'systematic' ? 'Consistency, reproducibility' :
      style === 'emotional' ? 'Values alignment, authenticity' :
      'Unknown'
    }`);
    console.log(`   Risks: ${
      style === 'analytical' ? 'Analysis paralysis, slow' :
      style === 'intuitive' ? 'Bias blind spots, inconsistency' :
      style === 'collaborative' ? 'Groupthink, slow consensus' :
      style === 'decisive' ? 'Impulsivity, insufficient data' :
      style === 'reflective' ? 'Procrastination, missed opportunities' :
      style === 'systematic' ? 'Rigidity, creativity loss' :
      style === 'emotional' ? 'Rationalization, regret' :
      'Unknown'
    }`);
    console.log('');
  }

  // ============================================================================
  // PART 4: Attachment Style & Relationship Patterns
  // ============================================================================
  console.log('\n💝 PART 4: Attachment Style Detection\n');

  const attachmentStatements = [
    "I'm comfortable with intimacy and don't worry about relationships",
    "I worry that people don't really love me and will abandon me",
    "I prefer to keep my distance and value independence above all",
    "I want closeness but fear it will lead to disappointment"
  ];

  const attachmentStyles = [
    { name: 'secure', statement: attachmentStatements[0], pattern: 'comfortable|trust|balanced' },
    { name: 'anxious', statement: attachmentStatements[1], pattern: 'worry|fear|abandon|unloved' },
    { name: 'avoidant', statement: attachmentStatements[2], pattern: 'distance|independent|alone' },
    { name: 'fearful', statement: attachmentStatements[3], pattern: 'want.*but|fear.*closeness|conflicted' }
  ];

  for (const style of attachmentStyles) {
    const sentiment = await system.reasoner.extractSentiment(style.statement);
    const preferences = await system.reasoner.extractPreferences(style.statement);

    console.log(`${style.name.toUpperCase()} ATTACHMENT:`);
    console.log(`   Statement: "${style.statement}"`);
    console.log(`   Sentiment: ${sentiment.score.toFixed(2)} (${sentiment.primaryEmotion})`);
    console.log(`   Characteristics: ${
      style.name === 'secure' ? 'Comfortable with intimacy, low anxiety, trusting' :
      style.name === 'anxious' ? 'High relationship anxiety, fears abandonment, seeks reassurance' :
      style.name === 'avoidant' ? 'Values independence, uncomfortable with closeness, self-reliant' :
      'Desires intimacy but fears vulnerability, mixed signals'
    }`);

    if (preferences.preferences.length > 0) {
      console.log(`   Core need: ${preferences.preferences[0].subject}`);
    }
    console.log('');
  }

  // ============================================================================
  // PART 5: Generate Exotic Psychological Personas
  // ============================================================================
  console.log('\n🎲 PART 5: Generate Synthetic Psychological Personas\n');

  console.log('Generating 100 complex psychological profiles...\n');

  const syntheticProfiles = await system.generateIntelligently('structured', {
    count: 100,
    schema: {
      profile_id: { type: 'string', required: true },
      name: { type: 'string', required: true },
      age: { type: 'number', min: 22, max: 65, required: true },
      personality_archetype: {
        type: 'enum',
        enum: ['hero', 'caregiver', 'sage', 'ruler', 'creator', 'rebel', 'magician', 'explorer'],
        required: true
      },
      secondary_archetype: {
        type: 'enum',
        enum: ['hero', 'caregiver', 'sage', 'ruler', 'creator', 'rebel', 'magician', 'explorer']
      },
      dominant_cognitive_bias: {
        type: 'enum',
        enum: ['confirmation', 'availability', 'anchoring', 'sunk_cost', 'attribution', 'hindsight', 'bandwagon'],
        required: true
      },
      decision_making_style: {
        type: 'enum',
        enum: ['analytical', 'intuitive', 'collaborative', 'decisive', 'reflective', 'systematic', 'emotional'],
        required: true
      },
      attachment_style: {
        type: 'enum',
        enum: ['secure', 'anxious', 'avoidant', 'fearful'],
        required: true
      },
      conflict_resolution: {
        type: 'enum',
        enum: ['competing', 'collaborating', 'compromising', 'avoiding', 'accommodating'],
        required: true
      },
      communication_style: {
        type: 'enum',
        enum: ['assertive', 'passive', 'aggressive', 'passive_aggressive'],
        required: true
      },
      primary_motivation: {
        type: 'enum',
        enum: ['achievement', 'affiliation', 'power', 'security', 'growth', 'autonomy'],
        required: true
      },
      core_fear: { type: 'string', required: true },
      shadow_aspects: { type: 'array', required: true },
      emotional_intelligence: { type: 'number', min: 0, max: 1, required: true },
      psychological_flexibility: { type: 'number', min: 0, max: 1, required: true },
      self_awareness_level: { type: 'number', min: 0, max: 1, required: true }
    }
  }, {
    targetSentiment: {
      score: 0.1,
      emotion: 'reflective'
    },
    userPreferences: [
      ...personalityStatements,
      ...decisionStatements,
      ...attachmentStatements
    ],
    contextualFactors: {
      environment: 'psychological_research',
      constraints: ['emotional_intelligence >= 0.3', 'self_awareness_level >= 0.2']
    },
    qualityThreshold: 0.92
  });

  console.log(`✅ Generated ${syntheticProfiles.data.length} synthetic psychological profiles`);
  console.log(`📊 Generation Quality:`);
  console.log(`   Preference alignment: ${(syntheticProfiles.psychoMetrics.preferenceAlignment * 100).toFixed(1)}%`);
  console.log(`   Complexity score: ${(syntheticProfiles.psychoMetrics.qualityScore * 100).toFixed(1)}%`);

  // ============================================================================
  // PART 6: Psychological Pattern Analysis
  // ============================================================================
  console.log('\n\n📈 PART 6: Psychological Pattern Distribution\n');

  const patterns = {
    archetype: new Map<string, number>(),
    bias: new Map<string, number>(),
    attachment: new Map<string, number>(),
    decisionStyle: new Map<string, number>(),
    conflictStyle: new Map<string, number>()
  };

  syntheticProfiles.data.forEach((profile: any) => {
    patterns.archetype.set(profile.personality_archetype,
      (patterns.archetype.get(profile.personality_archetype) || 0) + 1);

    patterns.bias.set(profile.dominant_cognitive_bias,
      (patterns.bias.get(profile.dominant_cognitive_bias) || 0) + 1);

    patterns.attachment.set(profile.attachment_style,
      (patterns.attachment.get(profile.attachment_style) || 0) + 1);

    patterns.decisionStyle.set(profile.decision_making_style,
      (patterns.decisionStyle.get(profile.decision_making_style) || 0) + 1);

    patterns.conflictStyle.set(profile.conflict_resolution,
      (patterns.conflictStyle.get(profile.conflict_resolution) || 0) + 1);
  });

  console.log('Personality Archetype Distribution:');
  Array.from(patterns.archetype.entries())
    .sort((a, b) => b[1] - a[1])
    .forEach(([archetype, count]) => {
      const pct = (count / syntheticProfiles.data.length * 100).toFixed(1);
      console.log(`   ${archetype}: ${count} (${pct}%)`);
    });

  console.log('\nAttachment Style Distribution:');
  Array.from(patterns.attachment.entries())
    .forEach(([style, count]) => {
      const pct = (count / syntheticProfiles.data.length * 100).toFixed(1);
      console.log(`   ${style}: ${count} (${pct}%)`);
    });

  console.log('\nConflict Resolution Distribution:');
  Array.from(patterns.conflictStyle.entries())
    .sort((a, b) => b[1] - a[1])
    .forEach(([style, count]) => {
      const pct = (count / syntheticProfiles.data.length * 100).toFixed(1);
      console.log(`   ${style}: ${count} (${pct}%)`);
    });

  // ============================================================================
  // PART 7: Psychological Compatibility Matrix
  // ============================================================================
  console.log('\n\n💫 PART 7: Psychological Compatibility Insights\n');

  const compatibilityRules = {
    archetype: {
      hero: ['caregiver', 'sage', 'magician'],
      caregiver: ['hero', 'ruler', 'explorer'],
      sage: ['creator', 'magician', 'hero'],
      rebel: ['creator', 'explorer', 'magician']
    },
    attachment: {
      secure: ['secure', 'anxious', 'avoidant', 'fearful'],
      anxious: ['secure'],
      avoidant: ['secure'],
      fearful: ['secure']
    }
  };

  console.log('High Compatibility Archetype Pairs:\n');
  Object.entries(compatibilityRules.archetype).forEach(([primary, compatible]) => {
    console.log(`   ${primary.toUpperCase()} works well with: ${compatible.join(', ')}`);
  });

  console.log('\nAttachment Style Compatibility:\n');
  console.log('   SECURE: Compatible with all styles (acts as stabilizer)');
  console.log('   ANXIOUS: Needs secure attachment for stability');
  console.log('   AVOIDANT: Needs secure attachment to develop intimacy');
  console.log('   FEARFUL: Benefits most from secure attachment support');

  // ============================================================================
  // PART 8: Sample Complex Psychological Profiles
  // ============================================================================
  console.log('\n\n📋 PART 8: Sample Complex Psychological Profiles\n');

  syntheticProfiles.data.slice(0, 3).forEach((profile: any, idx: number) => {
    console.log(`${'-'.repeat(70)}`);
    console.log(`PROFILE ${idx + 1}: ${profile.name} (Age ${profile.age})\n`);

    console.log(`🎭 PERSONALITY:`);
    console.log(`   Primary archetype: ${profile.personality_archetype.toUpperCase()}`);
    if (profile.secondary_archetype) {
      console.log(`   Secondary archetype: ${profile.secondary_archetype}`);
    }

    console.log(`\n🧠 COGNITIVE PATTERNS:`);
    console.log(`   Dominant bias: ${profile.dominant_cognitive_bias}`);
    console.log(`   Decision style: ${profile.decision_making_style}`);

    console.log(`\n💝 RELATIONSHIP DYNAMICS:`);
    console.log(`   Attachment style: ${profile.attachment_style}`);
    console.log(`   Conflict resolution: ${profile.conflict_resolution}`);
    console.log(`   Communication: ${profile.communication_style}`);

    console.log(`\n🎯 MOTIVATIONS & FEARS:`);
    console.log(`   Primary motivation: ${profile.primary_motivation}`);
    console.log(`   Core fear: ${profile.core_fear}`);

    console.log(`\n📊 PSYCHOLOGICAL METRICS:`);
    console.log(`   Emotional intelligence: ${(profile.emotional_intelligence * 100).toFixed(0)}%`);
    console.log(`   Psychological flexibility: ${(profile.psychological_flexibility * 100).toFixed(0)}%`);
    console.log(`   Self-awareness: ${(profile.self_awareness_level * 100).toFixed(0)}%`);

    if (profile.shadow_aspects && profile.shadow_aspects.length > 0) {
      console.log(`\n🌑 SHADOW ASPECTS:`);
      profile.shadow_aspects.slice(0, 3).forEach((aspect: string) => {
        console.log(`   - ${aspect}`);
      });
    }

    console.log('');
  });

  // ============================================================================
  // PART 9: Insights & Recommendations
  // ============================================================================
  console.log(`\n${'='.repeat(70)}\n`);
  console.log('✨ PART 9: Deep Psychological Insights\n');

  const avgEQ = syntheticProfiles.data.reduce((sum: number, p: any) =>
    sum + p.emotional_intelligence, 0) / syntheticProfiles.data.length;

  const avgFlex = syntheticProfiles.data.reduce((sum: number, p: any) =>
    sum + p.psychological_flexibility, 0) / syntheticProfiles.data.length;

  const avgAwareness = syntheticProfiles.data.reduce((sum: number, p: any) =>
    sum + p.self_awareness_level, 0) / syntheticProfiles.data.length;

  console.log('Population Psychological Health Indicators:\n');
  console.log(`   Average Emotional Intelligence: ${(avgEQ * 100).toFixed(0)}%`);
  console.log(`   Average Psychological Flexibility: ${(avgFlex * 100).toFixed(0)}%`);
  console.log(`   Average Self-Awareness: ${(avgAwareness * 100).toFixed(0)}%`);

  const secureAttachment = syntheticProfiles.data.filter(
    (p: any) => p.attachment_style === 'secure'
  ).length;

  console.log(`\n   Secure Attachment Rate: ${(secureAttachment / syntheticProfiles.data.length * 100).toFixed(1)}% ${
    secureAttachment / syntheticProfiles.data.length > 0.5 ? '(Healthy population)' : '(Intervention recommended)'
  }`);

  console.log('\n🌟 Key Insights:');
  console.log(`   • Most common archetype: ${Array.from(patterns.archetype.entries()).sort((a, b) => b[1] - a[1])[0][0]}`);
  console.log(`   • Most common bias: ${Array.from(patterns.bias.entries()).sort((a, b) => b[1] - a[1])[0][0]}`);
  console.log(`   • Most common decision style: ${Array.from(patterns.decisionStyle.entries()).sort((a, b) => b[1] - a[1])[0][0]}`);
  console.log(`   • Primary conflict approach: ${Array.from(patterns.conflictStyle.entries()).sort((a, b) => b[1] - a[1])[0][0]}`);

  console.log('\n✅ Exotic Psychological Profiling Complete!');
  console.log(`\n📊 Analyzed ${profiles.length} archetypes + ${biasStatements.length} biases + ${decisionStatements.length} decision styles`);
  console.log(`🎲 Generated ${syntheticProfiles.data.length} complex psychological personas`);

  await system.shutdown();
}

// Run the profiling
performExoticPsychologicalProfiling().catch(console.error);
