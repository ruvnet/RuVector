#!/usr/bin/env node

/**
 * CLI for Psycho-Synth Examples
 *
 * Usage:
 *   npx psycho-synth-examples list
 *   npx psycho-synth-examples run <example-name>
 *   npx psycho-synth-examples run audience --api-key YOUR_KEY
 */

import { program } from 'commander';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const examples = [
  {
    name: 'audience',
    title: '🎭 Audience Analysis',
    description: 'Real-time sentiment extraction, psychographic segmentation, persona generation',
    file: 'audience-analysis.ts'
  },
  {
    name: 'voter',
    title: '🗳️  Voter Sentiment',
    description: 'Political preference mapping, swing voter identification, issue analysis',
    file: 'voter-sentiment.ts'
  },
  {
    name: 'marketing',
    title: '📢 Marketing Optimization',
    description: 'Campaign targeting, A/B testing, ROI prediction, customer segmentation',
    file: 'marketing-optimization.ts'
  },
  {
    name: 'financial',
    title: '💹 Financial Sentiment',
    description: 'Market analysis, investor psychology, Fear & Greed Index, risk assessment',
    file: 'financial-sentiment.ts'
  },
  {
    name: 'medical',
    title: '🏥 Medical Patient Analysis',
    description: 'Patient emotional states, compliance prediction, psychosocial assessment',
    file: 'medical-patient-analysis.ts'
  },
  {
    name: 'psychological',
    title: '🧠 Psychological Profiling',
    description: 'Personality archetypes, cognitive biases, attachment styles, decision patterns',
    file: 'psychological-profiling.ts'
  }
];

program
  .name('psycho-synth-examples')
  .description('Psycho-Symbolic Reasoning Examples - Advanced AI Applications')
  .version('0.1.0');

program
  .command('list')
  .description('List all available examples')
  .action(() => {
    console.log('\n🧠 Available Psycho-Synth Examples:\n');
    console.log('='.repeat(70));

    examples.forEach((example, idx) => {
      console.log(`\n${idx + 1}. ${example.title}`);
      console.log(`   ${example.description}`);
      console.log(`   Run: npx psycho-synth-examples run ${example.name}`);
    });

    console.log('\n' + '='.repeat(70));
    console.log('\n💡 Tip: Set GEMINI_API_KEY environment variable before running\n');
  });

program
  .command('run <example>')
  .description('Run a specific example')
  .option('--api-key <key>', 'Gemini API key')
  .action((exampleName, options) => {
    const example = examples.find(e => e.name === exampleName);

    if (!example) {
      console.error(`\n❌ Unknown example: ${exampleName}`);
      console.log('\n💡 Run "npx psycho-synth-examples list" to see available examples\n');
      process.exit(1);
    }

    // Set API key if provided
    if (options.apiKey) {
      process.env.GEMINI_API_KEY = options.apiKey;
    }

    // Check if API key is set
    if (!process.env.GEMINI_API_KEY) {
      console.error('\n❌ Error: GEMINI_API_KEY environment variable not set');
      console.log('\n💡 Set it with:');
      console.log('   export GEMINI_API_KEY="your-key-here"');
      console.log('   or use --api-key flag\n');
      process.exit(1);
    }

    console.log(`\n🚀 Running: ${example.title}\n`);
    console.log('='.repeat(70));

    const examplePath = join(__dirname, '..', 'examples', example.file);

    // Run with tsx
    const child = spawn('npx', ['tsx', examplePath], {
      stdio: 'inherit',
      env: process.env
    });

    child.on('error', (error) => {
      console.error(`\n❌ Error running example: ${error.message}\n`);
      process.exit(1);
    });

    child.on('exit', (code) => {
      if (code !== 0) {
        console.error(`\n❌ Example exited with code ${code}\n`);
        process.exit(code);
      }
    });
  });

program.parse();
