/**
 * CLI module for RuvBot
 *
 * Provides command-line interface for npx @ruvector/ruvbot
 */

import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import { RuvBot } from '../RuvBot.js';
import { ConfigManager } from '../core/BotConfig.js';

const VERSION = '0.1.0';

export function createCLI(): Command {
  const program = new Command();

  program
    .name('ruvbot')
    .description('Self-learning AI assistant bot with WASM embeddings and vector memory')
    .version(VERSION);

  // Start command
  program
    .command('start')
    .description('Start the RuvBot server')
    .option('-p, --port <port>', 'API server port', '3000')
    .option('-c, --config <path>', 'Path to config file')
    .option('--remote', 'Connect to remote services')
    .option('--debug', 'Enable debug logging')
    .action(async (options) => {
      const spinner = ora('Starting RuvBot...').start();

      try {
        const config = options.config
          ? require(options.config)
          : ConfigManager.fromEnv().getConfig();

        const bot = new RuvBot({
          ...config,
          debug: options.debug,
          api: {
            ...config.api,
            port: parseInt(options.port, 10),
          },
        });

        await bot.start();
        spinner.succeed(chalk.green('RuvBot started successfully'));

        console.log(chalk.cyan(`\n  API Server: http://localhost:${options.port}`));
        console.log(chalk.gray('  Press Ctrl+C to stop\n'));

        // Handle shutdown
        process.on('SIGINT', async () => {
          console.log(chalk.yellow('\nShutting down...'));
          await bot.stop();
          process.exit(0);
        });
      } catch (error) {
        spinner.fail(chalk.red('Failed to start RuvBot'));
        console.error(error);
        process.exit(1);
      }
    });

  // Config command
  program
    .command('config')
    .description('Interactive configuration wizard')
    .action(async () => {
      console.log(chalk.cyan('Configuration wizard coming soon...'));
    });

  // Skills command
  program
    .command('skills')
    .description('Manage bot skills')
    .command('list')
    .description('List available skills')
    .action(() => {
      console.log(chalk.cyan('Built-in skills:'));
      console.log('  - search: Semantic search in memory');
      console.log('  - summarize: Summarize text content');
      console.log('  - code: Code generation and analysis');
      console.log('  - memory: Store and retrieve memories');
    });

  // Status command
  program
    .command('status')
    .description('Show bot status and health')
    .option('-w, --watch', 'Watch mode')
    .action(async (options) => {
      console.log(chalk.cyan('Status check coming soon...'));
    });

  // Init command
  program
    .command('init')
    .description('Initialize RuvBot in current directory')
    .option('-y, --yes', 'Skip prompts with defaults')
    .action(async (options) => {
      const spinner = ora('Initializing RuvBot...').start();

      try {
        // Create config file
        const config = {
          name: 'my-ruvbot',
          port: 3000,
          storage: { type: 'sqlite', path: './data/ruvbot.db' },
          memory: { dimensions: 384, maxVectors: 100000 },
          skills: { enabled: ['search', 'summarize', 'code', 'memory'] },
        };

        // Write config file
        const fs = await import('fs/promises');
        await fs.writeFile('ruvbot.config.json', JSON.stringify(config, null, 2));
        await fs.mkdir('data', { recursive: true });
        await fs.mkdir('skills', { recursive: true });

        spinner.succeed(chalk.green('RuvBot initialized'));
        console.log(chalk.gray('\nCreated:'));
        console.log('  - ruvbot.config.json');
        console.log('  - data/');
        console.log('  - skills/');
        console.log(chalk.cyan('\nRun `ruvbot start` to start the bot'));
      } catch (error) {
        spinner.fail(chalk.red('Failed to initialize'));
        console.error(error);
        process.exit(1);
      }
    });

  return program;
}

// Main entry point
export async function main(): Promise<void> {
  const program = createCLI();
  await program.parseAsync(process.argv);
}

// Run if called directly (works in both ESM and CJS)
// The bin entry point will call main() directly
export default main;
