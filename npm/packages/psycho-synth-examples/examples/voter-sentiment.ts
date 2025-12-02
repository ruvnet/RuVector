/**
 * Voter Sentiment & Preference Analysis with Psycho-Symbolic Reasoning
 *
 * Demonstrates:
 * - Political sentiment extraction (0.4ms per voter)
 * - Issue preference mapping
 * - Voter segmentation by psychographic profile
 * - Swing voter identification
 * - Synthetic voter persona generation for polling
 * - Campaign message optimization
 */

import { quickStart } from 'psycho-symbolic-integration';

interface Voter {
  id: string;
  statement: string;
  sentiment?: any;
  preferences?: any[];
  issuePositions?: Map<string, number>;
  swingVoterScore?: number;
}

async function analyzeVoterSentiment() {
  console.log('🗳️  Voter Sentiment & Preference Analysis\n');
  console.log('='.repeat(70));

  const system = await quickStart(process.env.GEMINI_API_KEY);

  // ============================================================================
  // PART 1: Real Voter Statement Analysis
  // ============================================================================
  console.log('\n📊 PART 1: Analyzing Real Voter Statements (0.4ms each)\n');

  const voterStatements = [
    "I'm concerned about healthcare costs but also value economic growth",
    "Climate change is my top priority - we need immediate action",
    "I support lower taxes and less government regulation",
    "Education reform is critical, especially funding for public schools",
    "We need stronger border security while treating immigrants humanely",
    "I'm worried about inflation and the cost of living",
    "Social justice issues matter most to me - equality for all",
    "I'm fiscally conservative but socially progressive",
    "Small business support and job creation should be the focus",
    "I prefer candidates who are moderate and willing to compromise"
  ];

  const analyzedVoters: Voter[] = [];

  for (let i = 0; i < voterStatements.length; i++) {
    const statement = voterStatements[i];

    const [sentiment, preferences] = await Promise.all([
      system.reasoner.extractSentiment(statement),
      system.reasoner.extractPreferences(statement)
    ]);

    analyzedVoters.push({
      id: `voter_${i + 1}`,
      statement,
      sentiment,
      preferences: preferences.preferences
    });

    console.log(`🗳️  Voter ${i + 1}:`);
    console.log(`   Statement: "${statement}"`);
    console.log(`   Sentiment: ${sentiment.score.toFixed(2)} (${sentiment.primaryEmotion})`);
    console.log(`   Issue preferences: ${preferences.preferences.length}`);

    if (preferences.preferences.length > 0) {
      preferences.preferences.slice(0, 2).forEach((pref: any) => {
        console.log(`     - ${pref.type}: "${pref.subject}" (strength: ${pref.strength.toFixed(2)})`);
      });
    }
    console.log('');
  }

  // ============================================================================
  // PART 2: Issue-Based Voter Segmentation
  // ============================================================================
  console.log('\n🎯 PART 2: Issue-Based Voter Segmentation\n');

  // Extract key issues from preferences
  const issueMap = new Map<string, number>();

  analyzedVoters.forEach(voter => {
    voter.preferences?.forEach(pref => {
      const subject = pref.subject.toLowerCase();
      const count = issueMap.get(subject) || 0;
      issueMap.set(subject, count + pref.strength);
    });
  });

  const topIssues = Array.from(issueMap.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  console.log('📊 Top 5 Voter Issues (by aggregate preference strength):\n');
  topIssues.forEach(([issue, strength], idx) => {
    console.log(`   ${idx + 1}. ${issue.charAt(0).toUpperCase() + issue.slice(1)}: ${strength.toFixed(2)}`);
  });

  // ============================================================================
  // PART 3: Swing Voter Identification
  // ============================================================================
  console.log('\n\n⚖️  PART 3: Swing Voter Identification\n');

  // Calculate swing voter score (voters with mixed/moderate sentiments and preferences)
  const swingVoters = analyzedVoters.map(voter => {
    // Swing indicators:
    // 1. Sentiment close to neutral (-0.3 to 0.3)
    // 2. Multiple competing preferences
    // 3. Use of words like "but", "however", "also"

    const sentimentNeutrality = 1 - Math.abs(voter.sentiment!.score);
    const preferenceDiversity = Math.min(voter.preferences!.length / 3, 1);
    const moderateLanguage = voter.statement.match(/but|however|also|while|although/gi)?.length || 0;

    const swingScore = (
      (sentimentNeutrality * 0.4) +
      (preferenceDiversity * 0.4) +
      (Math.min(moderateLanguage / 2, 1) * 0.2)
    );

    return {
      ...voter,
      swingVoterScore: swingScore
    };
  }).sort((a, b) => b.swingVoterScore! - a.swingVoterScore!);

  console.log('Top 5 Swing Voters (most persuadable):\n');
  swingVoters.slice(0, 5).forEach((voter, idx) => {
    console.log(`${idx + 1}. Voter ${voter.id.split('_')[1]}: ${(voter.swingVoterScore! * 100).toFixed(1)}% swing score`);
    console.log(`   Statement: "${voter.statement.substring(0, 60)}..."`);
    console.log(`   Sentiment: ${voter.sentiment!.score.toFixed(2)} (${voter.sentiment!.primaryEmotion})`);
    console.log('');
  });

  // ============================================================================
  // PART 4: Generate Synthetic Voter Personas
  // ============================================================================
  console.log('\n🎲 PART 4: Generate Synthetic Voter Personas for Polling\n');

  console.log('Generating 50 synthetic voter personas for polling simulation...\n');

  const syntheticVoters = await system.generateIntelligently('structured', {
    count: 50,
    schema: {
      voter_id: { type: 'string', required: true },
      age: { type: 'number', min: 18, max: 85, required: true },
      location_type: {
        type: 'enum',
        enum: ['urban', 'suburban', 'rural'],
        required: true
      },
      education_level: {
        type: 'enum',
        enum: ['high_school', 'some_college', 'bachelors', 'graduate'],
        required: true
      },
      income_bracket: {
        type: 'enum',
        enum: ['low', 'middle', 'upper_middle', 'high'],
        required: true
      },
      primary_issue: {
        type: 'enum',
        enum: ['economy', 'healthcare', 'climate', 'education', 'immigration', 'security'],
        required: true
      },
      political_leaning: {
        type: 'enum',
        enum: ['progressive', 'liberal', 'moderate', 'conservative', 'libertarian'],
        required: true
      },
      engagement_level: {
        type: 'enum',
        enum: ['low', 'medium', 'high', 'very_high'],
        required: true
      },
      swing_voter_probability: { type: 'number', min: 0, max: 1, required: true },
      top_concerns: { type: 'array', required: true },
      media_consumption: { type: 'array', required: true }
    }
  }, {
    targetSentiment: {
      score: 0.0, // Neutral - representing diverse political spectrum
      emotion: 'concerned'
    },
    userPreferences: voterStatements,
    contextualFactors: {
      environment: 'political_polling',
      constraints: ['swing_voter_probability >= 0.1']
    },
    qualityThreshold: 0.88
  });

  console.log(`✅ Generated ${syntheticVoters.data.length} synthetic voter personas`);
  console.log(`📊 Generation Quality:`);
  console.log(`   Preference alignment: ${(syntheticVoters.psychoMetrics.preferenceAlignment * 100).toFixed(1)}%`);
  console.log(`   Sentiment match: ${(syntheticVoters.psychoMetrics.sentimentMatch * 100).toFixed(1)}%`);
  console.log(`   Overall quality: ${(syntheticVoters.psychoMetrics.qualityScore * 100).toFixed(1)}%`);

  // ============================================================================
  // PART 5: Voter Demographics & Segmentation Analysis
  // ============================================================================
  console.log('\n\n📈 PART 5: Synthetic Voter Demographics Analysis\n');

  const demographics = {
    byLeaning: new Map<string, number>(),
    byIssue: new Map<string, number>(),
    byLocation: new Map<string, number>(),
    swingVoters: syntheticVoters.data.filter((v: any) => v.swing_voter_probability > 0.5)
  };

  syntheticVoters.data.forEach((voter: any) => {
    // Political leaning
    const leanCount = demographics.byLeaning.get(voter.political_leaning) || 0;
    demographics.byLeaning.set(voter.political_leaning, leanCount + 1);

    // Primary issue
    const issueCount = demographics.byIssue.get(voter.primary_issue) || 0;
    demographics.byIssue.set(voter.primary_issue, issueCount + 1);

    // Location type
    const locCount = demographics.byLocation.get(voter.location_type) || 0;
    demographics.byLocation.set(voter.location_type, locCount + 1);
  });

  console.log('Political Leaning Distribution:');
  Array.from(demographics.byLeaning.entries())
    .sort((a, b) => b[1] - a[1])
    .forEach(([leaning, count]) => {
      const pct = (count / syntheticVoters.data.length * 100).toFixed(1);
      console.log(`   ${leaning}: ${count} (${pct}%)`);
    });

  console.log('\nPrimary Issue Distribution:');
  Array.from(demographics.byIssue.entries())
    .sort((a, b) => b[1] - a[1])
    .forEach(([issue, count]) => {
      const pct = (count / syntheticVoters.data.length * 100).toFixed(1);
      console.log(`   ${issue}: ${count} (${pct}%)`);
    });

  console.log('\nLocation Type Distribution:');
  Array.from(demographics.byLocation.entries())
    .forEach(([location, count]) => {
      const pct = (count / syntheticVoters.data.length * 100).toFixed(1);
      console.log(`   ${location}: ${count} (${pct}%)`);
    });

  console.log(`\n🎯 Swing Voter Population: ${demographics.swingVoters.length} (${(demographics.swingVoters.length / syntheticVoters.data.length * 100).toFixed(1)}%)`);

  // ============================================================================
  // PART 6: Campaign Message Optimization Insights
  // ============================================================================
  console.log('\n\n💡 PART 6: Campaign Message Optimization Insights\n');

  // Analyze swing voters
  const swingVoterProfiles = demographics.swingVoters.reduce((acc: any, voter: any) => {
    const issue = voter.primary_issue;
    if (!acc[issue]) acc[issue] = [];
    acc[issue].push(voter);
    return acc;
  }, {});

  console.log('🎯 Swing Voter Target Groups:\n');

  Object.entries(swingVoterProfiles).forEach(([issue, voters]: [string, any]) => {
    console.log(`${issue.toUpperCase()} Swing Voters: ${voters.length}`);

    const avgAge = voters.reduce((sum: number, v: any) => sum + v.age, 0) / voters.length;
    const locations = voters.map((v: any) => v.location_type);
    const dominantLocation = locations.sort((a: string, b: string) =>
      locations.filter((v: string) => v === b).length - locations.filter((v: string) => v === a).length
    )[0];

    console.log(`   Average age: ${avgAge.toFixed(0)}`);
    console.log(`   Dominant location: ${dominantLocation}`);
    console.log(`   Recommended messaging: Focus on ${issue} with practical solutions`);
    console.log('');
  });

  // ============================================================================
  // PART 7: Sample Voter Profiles
  // ============================================================================
  console.log('\n📋 PART 7: Sample Synthetic Voter Profiles\n');

  syntheticVoters.data.slice(0, 3).forEach((voter: any, idx: number) => {
    console.log(`Voter Profile ${idx + 1}:`);
    console.log(`   ID: ${voter.voter_id}`);
    console.log(`   Demographics: Age ${voter.age}, ${voter.education_level}, ${voter.income_bracket} income`);
    console.log(`   Location: ${voter.location_type}`);
    console.log(`   Political leaning: ${voter.political_leaning}`);
    console.log(`   Primary issue: ${voter.primary_issue}`);
    console.log(`   Engagement: ${voter.engagement_level}`);
    console.log(`   Swing probability: ${(voter.swing_voter_probability * 100).toFixed(0)}%`);
    console.log(`   Top concerns: ${voter.top_concerns?.slice(0, 3).join(', ')}`);
    console.log('');
  });

  // ============================================================================
  // PART 8: Strategic Recommendations
  // ============================================================================
  console.log('\n🎯 PART 8: Strategic Campaign Recommendations\n');

  console.log('Based on voter sentiment analysis:\n');

  const recommendations = [
    `✓ Target ${demographics.swingVoters.length} identified swing voters with personalized messaging`,
    `✓ Focus on top issue: ${topIssues[0][0]} - high preference strength across demographics`,
    `✓ Develop ${demographics.byLocation.get('suburban') || 0} suburban outreach programs`,
    `✓ Create content addressing ${Array.from(demographics.byIssue.keys()).slice(0, 3).join(', ')}`,
    `✓ Engage ${syntheticVoters.data.filter((v: any) => v.engagement_level === 'low').length} low-engagement voters through digital channels`
  ];

  recommendations.forEach(rec => console.log(rec));

  console.log('\n✨ Voter Analysis Complete!');
  console.log(`\n📊 Summary: Analyzed ${analyzedVoters.length} real voters + ${syntheticVoters.data.length} synthetic voters`);
  console.log(`🎯 Identified ${swingVoters.filter(v => v.swingVoterScore! > 0.6).length} high-probability swing voters`);

  await system.shutdown();
}

// Run the analysis
analyzeVoterSentiment().catch(console.error);
