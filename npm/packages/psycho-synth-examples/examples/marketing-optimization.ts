/**
 * Marketing Campaign Optimization with Psycho-Symbolic Reasoning
 *
 * Demonstrates:
 * - Ad copy sentiment analysis and A/B testing
 * - Customer preference extraction for targeting
 * - Campaign message optimization
 * - Synthetic customer persona generation
 * - ROI prediction based on psychological profiles
 */

import { quickStart } from 'psycho-symbolic-integration';

async function optimizeMarketingCampaigns() {
  console.log('📢 Marketing Campaign Optimization with Psycho-Symbolic AI\n');
  console.log('='.repeat(70));

  const system = await quickStart(process.env.GEMINI_API_KEY);

  // ============================================================================
  // PART 1: A/B Test Ad Copy Sentiment Analysis
  // ============================================================================
  console.log('\n🎯 PART 1: A/B Testing Ad Copy Variants (0.4ms analysis per variant)\n');

  const adVariants = {
    emotional: [
      "Transform your life today - experience the joy of success!",
      "Don't miss out! Join thousands who've already discovered happiness",
      "Feel the excitement - your dream lifestyle awaits!"
    ],
    rational: [
      "Proven results: 85% customer satisfaction in independent studies",
      "Save 30% on average costs with our efficient solution",
      "Data-driven approach delivers measurable outcomes"
    ],
    urgency: [
      "Limited time offer - act now or miss your chance forever",
      "Only 24 hours left to claim your exclusive discount",
      "Last chance: offer expires at midnight tonight"
    ],
    social_proof: [
      "Join over 100,000 satisfied customers worldwide",
      "Trusted by industry leaders and Fortune 500 companies",
      "Rated 4.9/5 stars by verified customers"
    ]
  };

  const variantResults: any = {};

  for (const [type, variants] of Object.entries(adVariants)) {
    console.log(`\n${type.toUpperCase()} AD VARIANTS:`);

    const sentiments = await Promise.all(
      variants.map(text => system.reasoner.extractSentiment(text))
    );

    const avgSentiment = sentiments.reduce((sum, s) => sum + s.score, 0) / sentiments.length;
    const avgConfidence = sentiments.reduce((sum, s) => sum + s.confidence, 0) / sentiments.length;

    variantResults[type] = {
      avgSentiment,
      avgConfidence,
      topEmotion: sentiments[0].primaryEmotion,
      variants
    };

    sentiments.forEach((sentiment, idx) => {
      console.log(`   Variant ${idx + 1}: "${variants[idx].substring(0, 50)}..."`);
      console.log(`   Sentiment: ${sentiment.score.toFixed(2)} (${sentiment.primaryEmotion}, confidence: ${(sentiment.confidence * 100).toFixed(0)}%)`);
    });

    console.log(`   → Average sentiment: ${avgSentiment.toFixed(2)}`);
  }

  // Rank ad types by sentiment
  const rankedAdTypes = Object.entries(variantResults)
    .sort(([, a]: any, [, b]: any) => b.avgSentiment - a.avgSentiment);

  console.log('\n\n📊 AD TYPE PERFORMANCE RANKING:\n');
  rankedAdTypes.forEach(([type, results]: [string, any], idx) => {
    console.log(`${idx + 1}. ${type.toUpperCase()}`);
    console.log(`   Average sentiment: ${results.avgSentiment.toFixed(2)}`);
    console.log(`   Primary emotion: ${results.topEmotion}`);
    console.log(`   Confidence: ${(results.avgConfidence * 100).toFixed(0)}%`);
    console.log('');
  });

  // ============================================================================
  // PART 2: Customer Feedback Analysis
  // ============================================================================
  console.log('\n💬 PART 2: Customer Feedback Preference Extraction\n');

  const customerFeedback = [
    "I love products that are eco-friendly and sustainable",
    "Price is my main concern - I need affordable options",
    "Quality matters most to me, I'm willing to pay more",
    "Fast shipping and excellent customer service are essential",
    "I prefer brands that align with my values and ethics",
    "Convenience and ease of use are what I look for",
    "I want innovative features and cutting-edge technology"
  ];

  const customerProfiles = [];

  for (let i = 0; i < customerFeedback.length; i++) {
    const feedback = customerFeedback[i];
    const preferences = await system.reasoner.extractPreferences(feedback);
    const sentiment = await system.reasoner.extractSentiment(feedback);

    customerProfiles.push({
      id: `customer_${i + 1}`,
      feedback,
      preferences: preferences.preferences,
      sentiment
    });

    console.log(`Customer ${i + 1}: "${feedback}"`);
    if (preferences.preferences.length > 0) {
      preferences.preferences.forEach((pref: any) => {
        console.log(`   → ${pref.type}: "${pref.subject}" (strength: ${pref.strength.toFixed(2)})`);
      });
    }
    console.log('');
  }

  // ============================================================================
  // PART 3: Customer Segmentation
  // ============================================================================
  console.log('\n🎯 PART 3: Psychographic Customer Segmentation\n');

  // Group by dominant preference type
  const preferenceGroups = customerProfiles.reduce((acc: any, customer) => {
    const topPref = customer.preferences[0];
    if (topPref) {
      const key = topPref.subject;
      if (!acc[key]) acc[key] = [];
      acc[key].push(customer);
    }
    return acc;
  }, {});

  console.log('Customer Segments by Preference:\n');
  Object.entries(preferenceGroups).forEach(([preference, customers]: [string, any]) => {
    console.log(`${preference.toUpperCase()} Segment: ${customers.length} customers`);
    const avgSentiment = customers.reduce((sum: number, c: any) => sum + c.sentiment.score, 0) / customers.length;
    console.log(`   Average sentiment: ${avgSentiment.toFixed(2)}`);
    console.log(`   Recommended messaging: Focus on ${preference}-related benefits`);
    console.log('');
  });

  // ============================================================================
  // PART 4: Generate Synthetic Customer Personas
  // ============================================================================
  console.log('\n🎲 PART 4: Generate Synthetic Customer Personas\n');

  console.log('Generating 100 synthetic customer personas for campaign targeting...\n');

  const syntheticCustomers = await system.generateIntelligently('structured', {
    count: 100,
    schema: {
      customer_id: { type: 'string', required: true },
      name: { type: 'string', required: true },
      age: { type: 'number', min: 18, max: 75, required: true },
      segment: {
        type: 'enum',
        enum: ['value_seekers', 'quality_conscious', 'eco_friendly', 'tech_savvy', 'convenience_focused'],
        required: true
      },
      purchase_motivation: {
        type: 'enum',
        enum: ['price', 'quality', 'sustainability', 'innovation', 'convenience', 'status'],
        required: true
      },
      brand_loyalty: {
        type: 'enum',
        enum: ['low', 'medium', 'high'],
        required: true
      },
      ad_response_preference: {
        type: 'enum',
        enum: ['emotional', 'rational', 'urgency', 'social_proof'],
        required: true
      },
      monthly_spend: { type: 'number', min: 50, max: 5000, required: true },
      conversion_probability: { type: 'number', min: 0, max: 1, required: true },
      preferred_channels: { type: 'array', required: true },
      pain_points: { type: 'array', required: true }
    }
  }, {
    targetSentiment: {
      score: 0.5,
      emotion: 'interested'
    },
    userPreferences: customerFeedback,
    contextualFactors: {
      environment: 'e-commerce',
      constraints: ['conversion_probability >= 0.2']
    },
    qualityThreshold: 0.87
  });

  console.log(`✅ Generated ${syntheticCustomers.data.length} synthetic customer personas`);
  console.log(`📊 Generation Metrics:`);
  console.log(`   Preference alignment: ${(syntheticCustomers.psychoMetrics.preferenceAlignment * 100).toFixed(1)}%`);
  console.log(`   Quality score: ${(syntheticCustomers.psychoMetrics.qualityScore * 100).toFixed(1)}%`);

  // ============================================================================
  // PART 5: Campaign Targeting Recommendations
  // ============================================================================
  console.log('\n\n💡 PART 5: Data-Driven Campaign Targeting Recommendations\n');

  // Analyze synthetic customer data
  const segmentDistribution = syntheticCustomers.data.reduce((acc: any, customer: any) => {
    acc[customer.segment] = (acc[customer.segment] || 0) + 1;
    return acc;
  }, {});

  const adPreferenceDistribution = syntheticCustomers.data.reduce((acc: any, customer: any) => {
    acc[customer.ad_response_preference] = (acc[customer.ad_response_preference] || 0) + 1;
    return acc;
  }, {});

  console.log('Target Audience Distribution:\n');
  Object.entries(segmentDistribution)
    .sort(([, a]: any, [, b]: any) => b - a)
    .forEach(([segment, count]: [string, any]) => {
      const pct = (count / syntheticCustomers.data.length * 100).toFixed(1);
      console.log(`   ${segment}: ${count} customers (${pct}%)`);
    });

  console.log('\nBest Ad Type by Audience:\n');
  Object.entries(adPreferenceDistribution)
    .sort(([, a]: any, [, b]: any) => b - a)
    .forEach(([adType, count]: [string, any]) => {
      const pct = (count / syntheticCustomers.data.length * 100).toFixed(1);
      console.log(`   ${adType}: ${count} customers (${pct}%)`);
    });

  // ============================================================================
  // PART 6: ROI Prediction & Budget Allocation
  // ============================================================================
  console.log('\n\n💰 PART 6: ROI Prediction & Budget Allocation Strategy\n');

  const highValueCustomers = syntheticCustomers.data.filter(
    (c: any) => c.monthly_spend > 1000 && c.conversion_probability > 0.6
  );

  const avgConversionProb = syntheticCustomers.data.reduce(
    (sum: number, c: any) => sum + c.conversion_probability, 0
  ) / syntheticCustomers.data.length;

  const totalPotentialRevenue = syntheticCustomers.data.reduce(
    (sum: number, c: any) => sum + (c.monthly_spend * c.conversion_probability), 0
  );

  console.log(`High-Value Target Customers: ${highValueCustomers.length} (${(highValueCustomers.length / syntheticCustomers.data.length * 100).toFixed(1)}%)`);
  console.log(`Average conversion probability: ${(avgConversionProb * 100).toFixed(1)}%`);
  console.log(`Estimated monthly revenue potential: $${totalPotentialRevenue.toFixed(2)}`);

  console.log('\n🎯 Budget Allocation Recommendations:\n');

  // Recommend budget allocation based on segment size and value
  const budgetRecommendations = Object.entries(segmentDistribution)
    .sort(([, a]: any, [, b]: any) => b - a)
    .map(([segment, count]: [string, any]) => {
      const segmentCustomers = syntheticCustomers.data.filter((c: any) => c.segment === segment);
      const avgSpend = segmentCustomers.reduce((sum: number, c: any) => sum + c.monthly_spend, 0) / segmentCustomers.length;
      const avgConv = segmentCustomers.reduce((sum: number, c: any) => sum + c.conversion_probability, 0) / segmentCustomers.length;

      return {
        segment,
        size: count,
        avgSpend,
        avgConv,
        roi: avgSpend * avgConv
      };
    });

  budgetRecommendations.forEach((rec, idx) => {
    console.log(`${idx + 1}. ${rec.segment.toUpperCase()}`);
    console.log(`   Audience size: ${rec.size}`);
    console.log(`   Avg monthly spend: $${rec.avgSpend.toFixed(2)}`);
    console.log(`   Avg conversion: ${(rec.avgConv * 100).toFixed(1)}%`);
    console.log(`   Expected ROI: $${rec.roi.toFixed(2)} per customer`);
    console.log('');
  });

  // ============================================================================
  // PART 7: Sample Customer Profiles for Targeting
  // ============================================================================
  console.log('\n📋 PART 7: Sample High-Value Customer Profiles\n');

  highValueCustomers.slice(0, 3).forEach((customer: any, idx: number) => {
    console.log(`High-Value Customer ${idx + 1}:`);
    console.log(`   ID: ${customer.customer_id}`);
    console.log(`   Segment: ${customer.segment}`);
    console.log(`   Age: ${customer.age}`);
    console.log(`   Purchase motivation: ${customer.purchase_motivation}`);
    console.log(`   Brand loyalty: ${customer.brand_loyalty}`);
    console.log(`   Best ad type: ${customer.ad_response_preference}`);
    console.log(`   Monthly spend: $${customer.monthly_spend}`);
    console.log(`   Conversion probability: ${(customer.conversion_probability * 100).toFixed(0)}%`);
    console.log(`   Preferred channels: ${customer.preferred_channels?.slice(0, 3).join(', ')}`);
    console.log('');
  });

  // ============================================================================
  // PART 8: Final Campaign Strategy
  // ============================================================================
  console.log('\n✨ PART 8: Recommended Campaign Strategy\n');

  console.log('Based on psycho-symbolic analysis:\n');

  const topAdType = rankedAdTypes[0][0];
  const topSegment = budgetRecommendations[0].segment;

  const strategy = [
    `✓ Lead with ${topAdType} ad variants (highest sentiment score)`,
    `✓ Target ${topSegment} segment first (${budgetRecommendations[0].size} customers, highest ROI)`,
    `✓ Focus on ${highValueCustomers.length} high-value customers (conversion prob > 60%)`,
    `✓ Allocate ${((budgetRecommendations[0].size / syntheticCustomers.data.length) * 100).toFixed(0)}% of budget to top segment`,
    `✓ A/B test ${topAdType} vs ${rankedAdTypes[1][0]} variants`,
    `✓ Expected campaign ROI: $${budgetRecommendations[0].roi.toFixed(2)} per customer`,
    `✓ Potential monthly revenue: $${totalPotentialRevenue.toFixed(2)}`
  ];

  strategy.forEach(rec => console.log(rec));

  console.log('\n✅ Marketing Campaign Optimization Complete!');

  await system.shutdown();
}

// Run the optimization
optimizeMarketingCampaigns().catch(console.error);
