/**
 * Medical Patient Analysis with Psycho-Symbolic Reasoning
 *
 * Demonstrates:
 * - Patient sentiment and emotional state analysis
 * - Treatment preference extraction
 * - Compliance prediction modeling
 * - Pain and symptom severity assessment
 * - Synthetic patient persona generation
 * - Psychosocial factor identification
 *
 * IMPORTANT: For educational and research purposes only
 * Not for clinical diagnosis or treatment decisions
 */

import { quickStart } from 'psycho-symbolic-integration';

async function analyzePatientPsychology() {
  console.log('🏥 Medical Patient Psychological Analysis\n');
  console.log('='.repeat(70));
  console.log('⚠️  EDUCATIONAL USE ONLY - NOT FOR CLINICAL DECISIONS\n');

  const system = await quickStart(process.env.GEMINI_API_KEY);

  // ============================================================================
  // PART 1: Patient Sentiment & Emotional State Analysis
  // ============================================================================
  console.log('\n💬 PART 1: Patient Statement Analysis (0.4ms per statement)\n');

  const patientStatements = [
    "I'm worried about my chronic pain and how it affects my daily life",
    "The treatment is helping but I struggle with the side effects",
    "I feel hopeful about recovery and trust my care team",
    "I'm frustrated with the slow progress and constant appointments",
    "Anxiety about my diagnosis is affecting my sleep and appetite",
    "I prefer natural remedies and am hesitant about medications",
    "The pain is manageable now and I'm feeling more optimistic",
    "I'm overwhelmed by the treatment options and don't know what to choose"
  ];

  const patientAnalysis = [];

  for (let i = 0; i < patientStatements.length; i++) {
    const statement = patientStatements[i];
    const [sentiment, preferences] = await Promise.all([
      system.reasoner.extractSentiment(statement),
      system.reasoner.extractPreferences(statement)
    ]);

    // Extract pain/severity indicators
    const severityKeywords = ['severe', 'intense', 'unbearable', 'chronic', 'constant'];
    const severityScore = severityKeywords.filter(kw =>
      statement.toLowerCase().includes(kw)
    ).length / severityKeywords.length;

    patientAnalysis.push({
      id: `patient_${i + 1}`,
      statement,
      sentiment,
      preferences: preferences.preferences,
      severityScore,
      emotionalState: sentiment.primaryEmotion
    });

    console.log(`👤 Patient ${i + 1}:`);
    console.log(`   Statement: "${statement}"`);
    console.log(`   Emotional state: ${sentiment.primaryEmotion} (sentiment: ${sentiment.score.toFixed(2)})`);
    console.log(`   Confidence: ${(sentiment.confidence * 100).toFixed(0)}%`);
    console.log(`   Severity indicators: ${(severityScore * 100).toFixed(0)}%`);

    if (preferences.preferences.length > 0) {
      console.log(`   Treatment preferences:`);
      preferences.preferences.forEach((pref: any) => {
        console.log(`     - ${pref.type}: "${pref.subject}" (strength: ${pref.strength.toFixed(2)})`);
      });
    }
    console.log('');
  }

  // ============================================================================
  // PART 2: Psychosocial Risk Assessment
  // ============================================================================
  console.log('\n🎯 PART 2: Psychosocial Risk Assessment\n');

  const riskFactors = {
    highAnxiety: patientAnalysis.filter(p => ['anxious', 'worried', 'stressed'].includes(p.emotionalState)),
    depression: patientAnalysis.filter(p => p.sentiment.score < -0.5),
    frustration: patientAnalysis.filter(p => p.emotionalState === 'frustrated'),
    hopeful: patientAnalysis.filter(p => p.sentiment.score > 0.5)
  };

  console.log('Risk Factor Distribution:\n');
  console.log(`   High anxiety: ${riskFactors.highAnxiety.length} patients (${(riskFactors.highAnxiety.length / patientAnalysis.length * 100).toFixed(0)}%)`);
  console.log(`   Depressive indicators: ${riskFactors.depression.length} patients (${(riskFactors.depression.length / patientAnalysis.length * 100).toFixed(0)}%)`);
  console.log(`   Frustration: ${riskFactors.frustration.length} patients (${(riskFactors.frustration.length / patientAnalysis.length * 100).toFixed(0)}%)`);
  console.log(`   Positive outlook: ${riskFactors.hopeful.length} patients (${(riskFactors.hopeful.length / patientAnalysis.length * 100).toFixed(0)}%)`);

  const avgSentiment = patientAnalysis.reduce((sum, p) => sum + p.sentiment.score, 0) / patientAnalysis.length;
  console.log(`\n   Overall patient sentiment: ${avgSentiment.toFixed(2)} ${avgSentiment < 0 ? '(Concerning)' : '(Positive)'}`);

  // ============================================================================
  // PART 3: Treatment Compliance Prediction
  // ============================================================================
  console.log('\n\n💊 PART 3: Treatment Compliance Prediction\n');

  const compliancePredictions = patientAnalysis.map(patient => {
    // Factors affecting compliance:
    // 1. Positive sentiment (+)
    // 2. Trust in treatment (+)
    // 3. Side effect concerns (-)
    // 4. Overwhelmed state (-)

    const sentimentFactor = (patient.sentiment.score + 1) / 2; // 0-1 scale
    const trustIndicators = patient.preferences.filter((p: any) =>
      p.subject.toLowerCase().includes('trust') || p.subject.toLowerCase().includes('help')
    ).length;

    const concernIndicators = patient.statement.match(/but|struggle|hesitant|worried|overwhelmed/gi)?.length || 0;

    const complianceScore = (
      (sentimentFactor * 0.4) +
      (Math.min(trustIndicators / 2, 1) * 0.3) +
      (Math.max(1 - (concernIndicators / 3), 0) * 0.3)
    );

    return {
      ...patient,
      complianceScore,
      complianceRisk: complianceScore < 0.5 ? 'HIGH' : complianceScore < 0.7 ? 'MEDIUM' : 'LOW'
    };
  }).sort((a, b) => a.complianceScore - b.complianceScore);

  console.log('Compliance Risk Assessment:\n');

  const highRisk = compliancePredictions.filter(p => p.complianceRisk === 'HIGH');
  const mediumRisk = compliancePredictions.filter(p => p.complianceRisk === 'MEDIUM');
  const lowRisk = compliancePredictions.filter(p => p.complianceRisk === 'LOW');

  console.log(`   HIGH RISK: ${highRisk.length} patients - require close monitoring`);
  console.log(`   MEDIUM RISK: ${mediumRisk.length} patients - may need support`);
  console.log(`   LOW RISK: ${lowRisk.length} patients - likely compliant`);

  if (highRisk.length > 0) {
    console.log('\n   High-risk patients:');
    highRisk.forEach(p => {
      console.log(`     - Patient ${p.id.split('_')[1]}: ${(p.complianceScore * 100).toFixed(0)}% compliance score`);
      console.log(`       Emotional state: ${p.emotionalState}`);
      console.log(`       Primary concern: ${p.preferences[0]?.subject || 'N/A'}`);
    });
  }

  // ============================================================================
  // PART 4: Generate Synthetic Patient Personas
  // ============================================================================
  console.log('\n\n🎲 PART 4: Generate Synthetic Patient Personas\n');

  console.log('Generating 100 synthetic patient personas for clinical research...\n');

  const syntheticPatients = await system.generateIntelligently('structured', {
    count: 100,
    schema: {
      patient_id: { type: 'string', required: true },
      age: { type: 'number', min: 18, max: 85, required: true },
      condition_category: {
        type: 'enum',
        enum: ['chronic_pain', 'cardiovascular', 'mental_health', 'diabetes', 'respiratory', 'autoimmune'],
        required: true
      },
      severity_level: {
        type: 'enum',
        enum: ['mild', 'moderate', 'severe'],
        required: true
      },
      emotional_state: {
        type: 'enum',
        enum: ['anxious', 'depressed', 'hopeful', 'frustrated', 'accepting', 'overwhelmed'],
        required: true
      },
      support_system: {
        type: 'enum',
        enum: ['strong', 'moderate', 'weak', 'none'],
        required: true
      },
      health_literacy: {
        type: 'enum',
        enum: ['low', 'medium', 'high'],
        required: true
      },
      treatment_adherence: { type: 'number', min: 0, max: 1, required: true },
      coping_mechanisms: { type: 'array', required: true },
      barriers_to_care: { type: 'array', required: true },
      pain_level: { type: 'number', min: 0, max: 10, required: true },
      quality_of_life: { type: 'number', min: 0, max: 1, required: true }
    }
  }, {
    targetSentiment: {
      score: -0.2, // Slightly negative - representing healthcare concerns
      emotion: 'concerned'
    },
    userPreferences: patientStatements,
    contextualFactors: {
      environment: 'healthcare',
      constraints: ['pain_level >= 0', 'quality_of_life >= 0.2']
    },
    qualityThreshold: 0.90
  });

  console.log(`✅ Generated ${syntheticPatients.data.length} synthetic patient personas`);
  console.log(`📊 Generation Quality:`);
  console.log(`   Preference alignment: ${(syntheticPatients.psychoMetrics.preferenceAlignment * 100).toFixed(1)}%`);
  console.log(`   Sentiment match: ${(syntheticPatients.psychoMetrics.sentimentMatch * 100).toFixed(1)}%`);
  console.log(`   Quality score: ${(syntheticPatients.psychoMetrics.qualityScore * 100).toFixed(1)}%`);

  // ============================================================================
  // PART 5: Patient Population Analysis
  // ============================================================================
  console.log('\n\n📈 PART 5: Patient Population Analysis\n');

  const populationStats = {
    byCondition: new Map<string, number>(),
    bySeverity: new Map<string, number>(),
    byEmotionalState: new Map<string, number>(),
    lowAdherence: syntheticPatients.data.filter((p: any) => p.treatment_adherence < 0.5).length,
    highPain: syntheticPatients.data.filter((p: any) => p.pain_level > 7).length,
    lowQoL: syntheticPatients.data.filter((p: any) => p.quality_of_life < 0.4).length
  };

  syntheticPatients.data.forEach((patient: any) => {
    const condCount = populationStats.byCondition.get(patient.condition_category) || 0;
    populationStats.byCondition.set(patient.condition_category, condCount + 1);

    const sevCount = populationStats.bySeverity.get(patient.severity_level) || 0;
    populationStats.bySeverity.set(patient.severity_level, sevCount + 1);

    const emotCount = populationStats.byEmotionalState.get(patient.emotional_state) || 0;
    populationStats.byEmotionalState.set(patient.emotional_state, emotCount + 1);
  });

  console.log('Condition Distribution:');
  Array.from(populationStats.byCondition.entries())
    .sort((a, b) => b[1] - a[1])
    .forEach(([condition, count]) => {
      const pct = (count / syntheticPatients.data.length * 100).toFixed(1);
      console.log(`   ${condition}: ${count} (${pct}%)`);
    });

  console.log('\nSeverity Distribution:');
  Array.from(populationStats.bySeverity.entries())
    .forEach(([severity, count]) => {
      const pct = (count / syntheticPatients.data.length * 100).toFixed(1);
      console.log(`   ${severity}: ${count} (${pct}%)`);
    });

  console.log('\n⚠️  High-Risk Population Indicators:');
  console.log(`   Low treatment adherence: ${populationStats.lowAdherence} (${(populationStats.lowAdherence / syntheticPatients.data.length * 100).toFixed(1)}%)`);
  console.log(`   High pain levels (>7/10): ${populationStats.highPain} (${(populationStats.highPain / syntheticPatients.data.length * 100).toFixed(1)}%)`);
  console.log(`   Low quality of life: ${populationStats.lowQoL} (${(populationStats.lowQoL / syntheticPatients.data.length * 100).toFixed(1)}%)`);

  // ============================================================================
  // PART 6: Intervention Recommendations
  // ============================================================================
  console.log('\n\n💡 PART 6: Patient Care Intervention Recommendations\n');

  // Group high-risk patients by emotional state
  const emotionalStates = Array.from(populationStats.byEmotionalState.entries())
    .sort((a, b) => b[1] - a[1]);

  console.log('Emotional State Distribution & Interventions:\n');

  emotionalStates.forEach(([state, count]) => {
    const patientsInState = syntheticPatients.data.filter((p: any) => p.emotional_state === state);
    const avgAdherence = patientsInState.reduce((sum: number, p: any) =>
      sum + p.treatment_adherence, 0) / patientsInState.length;

    console.log(`${state.toUpperCase()} (${count} patients):`);
    console.log(`   Average adherence: ${(avgAdherence * 100).toFixed(0)}%`);
    console.log(`   Recommended intervention: ${
      state === 'anxious' ? 'Anxiety management, relaxation techniques, clear communication' :
      state === 'depressed' ? 'Mental health support, counseling referral, social services' :
      state === 'frustrated' ? 'Expectation management, progress tracking, education' :
      state === 'overwhelmed' ? 'Simplified care plans, care coordinator, family support' :
      state === 'hopeful' ? 'Reinforce positive outlook, maintain engagement' :
      'Acceptance-focused therapy, support groups'
    }`);
    console.log('');
  });

  // ============================================================================
  // PART 7: Sample Patient Profiles
  // ============================================================================
  console.log('\n📋 PART 7: Sample Patient Profiles\n');

  syntheticPatients.data.slice(0, 3).forEach((patient: any, idx: number) => {
    console.log(`Patient Profile ${idx + 1}:`);
    console.log(`   ID: ${patient.patient_id}`);
    console.log(`   Age: ${patient.age}`);
    console.log(`   Condition: ${patient.condition_category} (${patient.severity_level})`);
    console.log(`   Emotional state: ${patient.emotional_state}`);
    console.log(`   Support system: ${patient.support_system}`);
    console.log(`   Health literacy: ${patient.health_literacy}`);
    console.log(`   Treatment adherence: ${(patient.treatment_adherence * 100).toFixed(0)}%`);
    console.log(`   Pain level: ${patient.pain_level}/10`);
    console.log(`   Quality of life: ${(patient.quality_of_life * 100).toFixed(0)}%`);
    console.log(`   Coping mechanisms: ${patient.coping_mechanisms?.slice(0, 3).join(', ')}`);
    console.log('');
  });

  // ============================================================================
  // PART 8: Clinical Insights Summary
  // ============================================================================
  console.log('\n✨ PART 8: Clinical Insights Summary\n');

  console.log('Key findings from psychosocial analysis:\n');

  const insights = [
    `📊 Analyzed ${patientAnalysis.length} real + ${syntheticPatients.data.length} synthetic patients`,
    `⚠️  ${highRisk.length} patients at high risk for non-compliance`,
    `😟 ${riskFactors.highAnxiety.length} patients showing anxiety symptoms`,
    `🎯 ${populationStats.lowAdherence} patients need adherence support programs`,
    `💊 ${populationStats.highPain} patients require enhanced pain management`,
    `📈 ${riskFactors.hopeful.length} patients showing positive treatment response`,
    `🤝 Recommend psychosocial support for ${state.toUpperCase()} and FRUSTRATED patients`
  ];

  insights.forEach(insight => console.log(insight));

  console.log('\n✅ Medical Patient Analysis Complete!');
  console.log('\n⚠️  Remember: For educational/research use only - not for clinical decisions');

  await system.shutdown();
}

// Run the analysis
analyzePatientPsychology().catch(console.error);
