#!/usr/bin/env node

/**
 * Agentic Synth CLI
 * Production-ready CLI for synthetic data generation
 */

import { Command } from 'commander';
import { AgenticSynth } from '../dist/index.js';
import { readFileSync, writeFileSync, existsSync, unlinkSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const packageVersion = JSON.parse(
  readFileSync(resolve(__dirname, '../package.json'), 'utf8')
).version;

const program = new Command();

// Never echo API keys: CLI output often lands in CI logs.
function redactConfig(config) {
  if (!config || typeof config !== 'object') return config;
  const out = { ...config };
  if (out.apiKey) out.apiKey = '[REDACTED]';
  if (out.apiKeys) out.apiKeys = '[REDACTED]';
  return out;
}

// Helper to load JSON config file
function loadConfig(configPath) {
  try {
    if (!existsSync(configPath)) {
      throw new Error(`Config file not found: ${configPath}`);
    }
    const content = readFileSync(configPath, 'utf8');
    return JSON.parse(content);
  } catch (error) {
    if (error.message.includes('not found')) {
      throw error;
    }
    throw new Error(`Invalid JSON in config file: ${error.message}`);
  }
}

// Helper to load schema file
function loadSchema(schemaPath) {
  try {
    if (!existsSync(schemaPath)) {
      throw new Error(`Schema file not found: ${schemaPath}`);
    }
    const content = readFileSync(schemaPath, 'utf8');
    return JSON.parse(content);
  } catch (error) {
    if (error.message.includes('not found')) {
      throw error;
    }
    throw new Error(`Invalid JSON in schema file: ${error.message}`);
  }
}

program
  .name('agentic-synth')
  .description('AI-powered synthetic data generation for agentic systems')
  .version(packageVersion)
  .addHelpText('after', `
Examples:
  $ agentic-synth generate --count 100 --schema schema.json
  $ agentic-synth init --provider gemini
  $ agentic-synth doctor --verbose

Advanced Examples (via @ruvector/agentic-synth-examples):
  $ npx @ruvector/agentic-synth-examples dspy train --models gemini,claude
  $ npx @ruvector/agentic-synth-examples self-learn --task code-generation
  $ npx @ruvector/agentic-synth-examples list

Learn more:
  https://www.npmjs.com/package/@ruvector/agentic-synth-examples
  https://github.com/ruvnet/ruvector/tree/main/packages/agentic-synth
`);

program
  .command('generate')
  .description('Generate synthetic structured data')
  .option('-c, --count <number>', 'Number of records to generate', '10')
  .option('-s, --schema <path>', 'Path to JSON schema file')
  .option('-o, --output <path>', 'Output file path (JSON format)')
  .option('--seed <value>', 'Random seed for reproducibility')
  .option('-p, --provider <provider>', 'Model provider (gemini, openrouter)', 'gemini')
  .option('-m, --model <model>', 'Model name to use')
  .option('--format <format>', 'Output format (json, csv, array)', 'json')
  .option('--config <path>', 'Path to config file with provider settings')
  .action(async (options) => {
    try {
      // Load configuration
      let config = {
        provider: options.provider,
        model: options.model
      };

      // Load config file if provided
      if (options.config) {
        const fileConfig = loadConfig(resolve(options.config));
        config = { ...config, ...fileConfig };
      }

      // Ensure API key is set
      if (!config.apiKey && !process.env.GEMINI_API_KEY && !process.env.OPENROUTER_API_KEY) {
        console.error('Error: API key not found. Set GEMINI_API_KEY or OPENROUTER_API_KEY environment variable, or provide --config file.');
        process.exit(1);
      }

      // Initialize AgenticSynth
      const synth = new AgenticSynth(config);

      // Load schema if provided
      let schema = undefined;
      if (options.schema) {
        schema = loadSchema(resolve(options.schema));
      }

      // Parse count
      const count = parseInt(options.count, 10);
      if (isNaN(count) || count < 1) {
        throw new Error('Count must be a positive integer');
      }

      // Parse seed
      let seed = options.seed;
      if (seed) {
        const seedNum = parseInt(seed, 10);
        seed = isNaN(seedNum) ? seed : seedNum;
      }

      console.log(`Generating ${count} records...`);
      const startTime = Date.now();

      // Generate data using AgenticSynth
      const result = await synth.generateStructured({
        count,
        schema,
        seed,
        format: options.format
      });

      const duration = Date.now() - startTime;

      // Output results
      if (options.output) {
        const outputPath = resolve(options.output);
        writeFileSync(outputPath, JSON.stringify(result.data, null, 2));
        console.log(`✓ Generated ${result.metadata.count} records to ${outputPath}`);
      } else {
        console.log(JSON.stringify(result.data, null, 2));
      }

      // Display metadata
      console.error(`\nMetadata:`);
      console.error(`  Provider: ${result.metadata.provider}`);
      console.error(`  Model: ${result.metadata.model}`);
      console.error(`  Cached: ${result.metadata.cached}`);
      console.error(`  Duration: ${duration}ms`);
      console.error(`  Generated: ${result.metadata.generatedAt}`);

    } catch (error) {
      console.error('Error:', error.message);
      if (error.stack && process.env.DEBUG) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
      process.exit(1);
    }
  });

program
  .command('config')
  .description('Display or test configuration')
  .option('-f, --file <path>', 'Config file path to load')
  .option('-t, --test', 'Test configuration by initializing AgenticSynth')
  .action(async (options) => {
    try {
      let config = {};

      // Load config file if provided
      if (options.file) {
        config = loadConfig(resolve(options.file));
      }

      // Create instance to validate config
      const synth = new AgenticSynth(config);
      const currentConfig = synth.getConfig();

      console.log('Current Configuration:');
      console.log(JSON.stringify(redactConfig(currentConfig), null, 2));

      if (options.test) {
        console.log('\n✓ Configuration is valid and AgenticSynth initialized successfully');
      }

      // Check for API keys
      console.log('\nEnvironment Variables:');
      console.log(`  GEMINI_API_KEY: ${process.env.GEMINI_API_KEY ? '✓ Set' : '✗ Not set'}`);
      console.log(`  OPENROUTER_API_KEY: ${process.env.OPENROUTER_API_KEY ? '✓ Set' : '✗ Not set'}`);

    } catch (error) {
      console.error('Configuration error:', error.message);
      if (error.stack && process.env.DEBUG) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
      process.exit(1);
    }
  });

program
  .command('validate')
  .description('Validate configuration and dependencies')
  .option('-f, --file <path>', 'Config file path to validate')
  .action(async (options) => {
    try {
      let config = {};

      // Load config file if provided
      if (options.file) {
        config = loadConfig(resolve(options.file));
        console.log('✓ Config file is valid JSON');
      }

      // Validate by creating instance
      const synth = new AgenticSynth(config);
      console.log('✓ Configuration schema is valid');

      // Check provider settings
      const currentConfig = synth.getConfig();
      console.log(`✓ Provider: ${currentConfig.provider}`);
      console.log(`✓ Model: ${currentConfig.model || 'default'}`);
      console.log(`✓ Cache strategy: ${currentConfig.cacheStrategy}`);
      console.log(`✓ Max retries: ${currentConfig.maxRetries}`);
      console.log(`✓ Timeout: ${currentConfig.timeout}ms`);

      // Validate API key
      if (!currentConfig.apiKey && !process.env.GEMINI_API_KEY && !process.env.OPENROUTER_API_KEY) {
        console.warn('⚠ Warning: No API key found. Set GEMINI_API_KEY or OPENROUTER_API_KEY environment variable.');
      } else {
        console.log('✓ API key is configured');
      }

      console.log('\n✓ All validations passed');

    } catch (error) {
      console.error('Validation error:', error.message);
      if (error.stack && process.env.DEBUG) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
      process.exit(1);
    }
  });

program
  .command('init')
  .description('Initialize a new agentic-synth configuration file')
  .option('-f, --force', 'Overwrite existing config file')
  .option('-p, --provider <provider>', 'Model provider (gemini, openrouter)', 'gemini')
  .option('-o, --output <path>', 'Output config file path', '.agentic-synth.json')
  .action(async (options) => {
    try {
      const configPath = resolve(options.output);

      // Check if file exists
      if (existsSync(configPath) && !options.force) {
        console.error(`Error: Config file already exists at ${configPath}`);
        console.error('Use --force to overwrite');
        process.exit(1);
      }

      // Create default configuration
      const defaultConfig = {
        provider: options.provider,
        model: options.provider === 'gemini' ? 'gemini-2.0-flash-exp' : 'anthropic/claude-3.5-sonnet',
        cacheStrategy: 'memory',
        maxRetries: 3,
        timeout: 30000,
        debug: false
      };

      // Write config file
      writeFileSync(configPath, JSON.stringify(defaultConfig, null, 2));
      console.log(`✓ Created configuration file: ${configPath}`);
      console.log('\nNext steps:');
      console.log('1. Set your API key:');
      if (options.provider === 'gemini') {
        console.log('   export GEMINI_API_KEY="your-api-key"');
      } else {
        console.log('   export OPENROUTER_API_KEY="your-api-key"');
      }
      console.log('2. Edit the config file to customize settings');
      console.log('3. Run: agentic-synth doctor');
      console.log('4. Generate data: agentic-synth generate --config .agentic-synth.json');

    } catch (error) {
      console.error('Error creating config:', error.message);
      if (error.stack && process.env.DEBUG) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
      process.exit(1);
    }
  });

program
  .command('doctor')
  .description('Run comprehensive diagnostics on environment and configuration')
  .option('-f, --file <path>', 'Config file path to check')
  .option('-v, --verbose', 'Show detailed diagnostic information')
  .action(async (options) => {
    try {
      console.log('🔍 Running diagnostics...\n');

      let errorCount = 0;
      let warningCount = 0;

      // Check 1: Node.js version
      console.log('1. Node.js Environment:');
      const nodeVersion = process.version;
      const majorVersion = parseInt(nodeVersion.slice(1).split('.')[0]);
      if (majorVersion >= 18) {
        console.log(`   ✓ Node.js ${nodeVersion} (compatible)`);
      } else {
        console.log(`   ✗ Node.js ${nodeVersion} (requires >= 18.0.0)`);
        errorCount++;
      }

      // Check 2: Environment variables
      console.log('\n2. API Keys:');
      const hasGeminiKey = !!process.env.GEMINI_API_KEY;
      const hasOpenRouterKey = !!process.env.OPENROUTER_API_KEY;

      if (hasGeminiKey) {
        console.log('   ✓ GEMINI_API_KEY is set');
        if (options.verbose) {
          console.log(`     Length: ${process.env.GEMINI_API_KEY.length} chars`);
        }
      } else {
        console.log('   ✗ GEMINI_API_KEY not set');
        warningCount++;
      }

      if (hasOpenRouterKey) {
        console.log('   ✓ OPENROUTER_API_KEY is set');
        if (options.verbose) {
          console.log(`     Length: ${process.env.OPENROUTER_API_KEY.length} chars`);
        }
      } else {
        console.log('   ✗ OPENROUTER_API_KEY not set');
        warningCount++;
      }

      if (!hasGeminiKey && !hasOpenRouterKey) {
        console.log('   ⚠ Warning: No API keys configured. At least one is required.');
        errorCount++;
      }

      // Check 3: Configuration file
      console.log('\n3. Configuration:');
      let config = {};
      if (options.file) {
        try {
          config = loadConfig(resolve(options.file));
          console.log(`   ✓ Config file loaded: ${options.file}`);
          if (options.verbose) {
            console.log(`     Content: ${JSON.stringify(redactConfig(config), null, 6)}`);
          }
        } catch (error) {
          console.log(`   ✗ Failed to load config: ${error.message}`);
          errorCount++;
        }
      } else {
        const defaultPaths = ['.agentic-synth.json', 'agentic-synth.json', 'config.json'];
        let found = false;
        for (const path of defaultPaths) {
          if (existsSync(path)) {
            config = loadConfig(path);
            console.log(`   ✓ Auto-detected config: ${path}`);
            found = true;
            break;
          }
        }
        if (!found) {
          console.log('   ⚠ No config file found (using defaults)');
          warningCount++;
        }
      }

      // Check 4: AgenticSynth initialization
      console.log('\n4. Package Initialization:');
      try {
        const synth = new AgenticSynth(config);
        const currentConfig = synth.getConfig();
        console.log('   ✓ AgenticSynth initialized successfully');
        console.log(`   ✓ Provider: ${currentConfig.provider}`);
        console.log(`   ✓ Model: ${currentConfig.model || 'default'}`);
        console.log(`   ✓ Cache: ${currentConfig.cacheStrategy}`);
        console.log(`   ✓ Max retries: ${currentConfig.maxRetries}`);
        console.log(`   ✓ Timeout: ${currentConfig.timeout}ms`);
      } catch (error) {
        console.log(`   ✗ Failed to initialize: ${error.message}`);
        errorCount++;
      }

      // Check 5: Dependencies
      console.log('\n5. Dependencies:');
      try {
        // Check if required packages are available
        const packages = [
          '@google/generative-ai',
          'commander',
          'dotenv',
          'zod'
        ];

        for (const pkg of packages) {
          try {
            await import(pkg);
            console.log(`   ✓ ${pkg}`);
          } catch (err) {
            console.log(`   ✗ ${pkg} not found`);
            errorCount++;
          }
        }
      } catch (error) {
        console.log(`   ✗ Dependency check failed: ${error.message}`);
        errorCount++;
      }

      // Check 6: File system permissions
      console.log('\n6. File System:');
      try {
        const testPath = resolve('.agentic-synth-test.tmp');
        writeFileSync(testPath, 'test');
        readFileSync(testPath);
        // Clean up synchronously: process.exit below would beat an async unlink
        unlinkSync(testPath);
        console.log('   ✓ Read/write permissions OK');
      } catch (error) {
        console.log('   ✗ File system permissions issue');
        errorCount++;
      }

      // Summary
      console.log('\n' + '='.repeat(50));
      if (errorCount === 0 && warningCount === 0) {
        console.log('✓ All checks passed! Your environment is ready.');
      } else {
        if (errorCount > 0) {
          console.log(`✗ Found ${errorCount} error(s)`);
        }
        if (warningCount > 0) {
          console.log(`⚠ Found ${warningCount} warning(s)`);
        }
        console.log('\nRecommendations:');
        if (!hasGeminiKey && !hasOpenRouterKey) {
          console.log('- Set at least one API key (GEMINI_API_KEY or OPENROUTER_API_KEY)');
        }
        if (errorCount > 0) {
          console.log('- Fix errors above before using agentic-synth');
        }
        if (!options.file && warningCount > 0) {
          console.log('- Run: agentic-synth init');
          console.log('- Then: agentic-synth doctor --file .agentic-synth.json');
        }
      }
      console.log('='.repeat(50));

      process.exit(errorCount > 0 ? 1 : 0);

    } catch (error) {
      console.error('Doctor command error:', error.message);
      if (error.stack && process.env.DEBUG) {
        console.error('\nStack trace:');
        console.error(error.stack);
      }
      process.exit(1);
    }
  });

// Error handler for unknown commands
program.on('command:*', function () {
  console.error('Invalid command: %s\nSee --help for a list of available commands.', program.args.join(' '));
  process.exit(1);
});

// Show help if no command provided
if (process.argv.length === 2) {
  program.help();
}

program.parse();
