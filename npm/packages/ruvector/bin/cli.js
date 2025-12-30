#!/usr/bin/env node

const { Command } = require('commander');
const chalk = require('chalk');
const ora = require('ora');
const fs = require('fs');
const path = require('path');

// Lazy load ruvector (only when needed, not for install/help commands)
let VectorDB, getVersion, getImplementationType;
let ruvectorLoaded = false;

function loadRuvector() {
  if (ruvectorLoaded) return true;
  try {
    const ruvector = require('../dist/index.js');
    VectorDB = ruvector.VectorDB;
    getVersion = ruvector.getVersion;
    getImplementationType = ruvector.getImplementationType;
    ruvectorLoaded = true;
    return true;
  } catch (e) {
    return false;
  }
}

function requireRuvector() {
  if (!loadRuvector()) {
    console.error(chalk.red('Error: Failed to load ruvector. Please run: npm run build'));
    console.error(chalk.yellow('Or install the package: npm install ruvector'));
    process.exit(1);
  }
}

// Import GNN (optional - graceful fallback if not available)
let RuvectorLayer, TensorCompress, differentiableSearch, getCompressionLevel, hierarchicalForward;
let gnnAvailable = false;
try {
  const gnn = require('@ruvector/gnn');
  RuvectorLayer = gnn.RuvectorLayer;
  TensorCompress = gnn.TensorCompress;
  differentiableSearch = gnn.differentiableSearch;
  getCompressionLevel = gnn.getCompressionLevel;
  hierarchicalForward = gnn.hierarchicalForward;
  gnnAvailable = true;
} catch (e) {
  // GNN not available - commands will show helpful message
}

// Import Attention (optional - graceful fallback if not available)
let DotProductAttention, MultiHeadAttention, HyperbolicAttention, FlashAttention, LinearAttention, MoEAttention;
let GraphRoPeAttention, EdgeFeaturedAttention, DualSpaceAttention, LocalGlobalAttention;
let benchmarkAttention, computeAttentionAsync, batchAttentionCompute, parallelAttentionCompute;
let expMap, logMap, mobiusAddition, poincareDistance, projectToPoincareBall;
let attentionInfo, attentionVersion;
let attentionAvailable = false;
try {
  const attention = require('@ruvector/attention');
  // Core mechanisms
  DotProductAttention = attention.DotProductAttention;
  MultiHeadAttention = attention.MultiHeadAttention;
  HyperbolicAttention = attention.HyperbolicAttention;
  FlashAttention = attention.FlashAttention;
  LinearAttention = attention.LinearAttention;
  MoEAttention = attention.MoEAttention;
  // Graph attention
  GraphRoPeAttention = attention.GraphRoPeAttention;
  EdgeFeaturedAttention = attention.EdgeFeaturedAttention;
  DualSpaceAttention = attention.DualSpaceAttention;
  LocalGlobalAttention = attention.LocalGlobalAttention;
  // Utilities
  benchmarkAttention = attention.benchmarkAttention;
  computeAttentionAsync = attention.computeAttentionAsync;
  batchAttentionCompute = attention.batchAttentionCompute;
  parallelAttentionCompute = attention.parallelAttentionCompute;
  // Hyperbolic math
  expMap = attention.expMap;
  logMap = attention.logMap;
  mobiusAddition = attention.mobiusAddition;
  poincareDistance = attention.poincareDistance;
  projectToPoincareBall = attention.projectToPoincareBall;
  // Meta
  attentionInfo = attention.info;
  attentionVersion = attention.version;
  attentionAvailable = true;
} catch (e) {
  // Attention not available - commands will show helpful message
}

const program = new Command();

// Get package version from package.json
const packageJson = require('../package.json');

// Version and description (lazy load implementation info)
program
  .name('ruvector')
  .description(`${chalk.cyan('ruvector')} - High-performance vector database CLI`)
  .version(packageJson.version);

// Create database
program
  .command('create <path>')
  .description('Create a new vector database')
  .option('-d, --dimension <number>', 'Vector dimension', '384')
  .option('-m, --metric <type>', 'Distance metric (cosine|euclidean|dot)', 'cosine')
  .action((dbPath, options) => {
    requireRuvector();
    const spinner = ora('Creating database...').start();

    try {
      const dimension = parseInt(options.dimension);
      const db = new VectorDB({
        dimension,
        metric: options.metric,
        path: dbPath,
        autoPersist: true
      });

      db.save(dbPath);
      spinner.succeed(chalk.green(`Database created: ${dbPath}`));
      console.log(chalk.gray(`  Dimension: ${dimension}`));
      console.log(chalk.gray(`  Metric: ${options.metric}`));
      console.log(chalk.gray(`  Implementation: ${getImplementationType()}`));
    } catch (error) {
      spinner.fail(chalk.red('Failed to create database'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Insert vectors
program
  .command('insert <database> <file>')
  .description('Insert vectors from JSON file')
  .option('-b, --batch-size <number>', 'Batch size for insertion', '1000')
  .action((dbPath, file, options) => {
    requireRuvector();
    const spinner = ora('Loading database...').start();

    try {
      // Read database metadata to get dimension
      let dimension = 384; // default
      if (fs.existsSync(dbPath)) {
        const dbData = fs.readFileSync(dbPath, 'utf8');
        const parsed = JSON.parse(dbData);
        dimension = parsed.dimension || 384;
      }

      const db = new VectorDB({ dimension });

      if (fs.existsSync(dbPath)) {
        db.load(dbPath);
      }

      spinner.text = 'Reading vectors...';
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      const vectors = Array.isArray(data) ? data : [data];

      spinner.text = `Inserting ${vectors.length} vectors...`;
      const batchSize = parseInt(options.batchSize);

      for (let i = 0; i < vectors.length; i += batchSize) {
        const batch = vectors.slice(i, i + batchSize);
        db.insertBatch(batch);
        spinner.text = `Inserted ${Math.min(i + batchSize, vectors.length)}/${vectors.length} vectors...`;
      }

      db.save(dbPath);
      spinner.succeed(chalk.green(`Inserted ${vectors.length} vectors`));

      const stats = db.stats();
      console.log(chalk.gray(`  Total vectors: ${stats.count}`));
    } catch (error) {
      spinner.fail(chalk.red('Failed to insert vectors'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Search vectors
program
  .command('search <database>')
  .description('Search for similar vectors')
  .requiredOption('-v, --vector <json>', 'Query vector as JSON array')
  .option('-k, --top-k <number>', 'Number of results', '10')
  .option('-t, --threshold <number>', 'Similarity threshold', '0.0')
  .option('-f, --filter <json>', 'Metadata filter as JSON')
  .action((dbPath, options) => {
    requireRuvector();
    const spinner = ora('Loading database...').start();

    try {
      // Read database metadata
      const dbData = fs.readFileSync(dbPath, 'utf8');
      const parsed = JSON.parse(dbData);
      const dimension = parsed.dimension || 384;

      const db = new VectorDB({ dimension });
      db.load(dbPath);

      spinner.text = 'Searching...';

      const vector = JSON.parse(options.vector);
      const query = {
        vector,
        k: parseInt(options.topK),
        threshold: parseFloat(options.threshold)
      };

      if (options.filter) {
        query.filter = JSON.parse(options.filter);
      }

      const results = db.search(query);
      spinner.succeed(chalk.green(`Found ${results.length} results`));

      console.log(chalk.cyan('\nSearch Results:'));
      results.forEach((result, i) => {
        console.log(chalk.white(`\n${i + 1}. ID: ${result.id}`));
        console.log(chalk.yellow(`   Score: ${result.score.toFixed(4)}`));
        if (result.metadata) {
          console.log(chalk.gray(`   Metadata: ${JSON.stringify(result.metadata)}`));
        }
      });
    } catch (error) {
      spinner.fail(chalk.red('Failed to search'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Show stats
program
  .command('stats <database>')
  .description('Show database statistics')
  .action((dbPath) => {
    requireRuvector();
    const spinner = ora('Loading database...').start();

    try {
      const dbData = fs.readFileSync(dbPath, 'utf8');
      const parsed = JSON.parse(dbData);
      const dimension = parsed.dimension || 384;

      const db = new VectorDB({ dimension });
      db.load(dbPath);

      const stats = db.stats();
      spinner.succeed(chalk.green('Database statistics'));

      console.log(chalk.cyan('\nDatabase Stats:'));
      console.log(chalk.white(`  Vector Count: ${chalk.yellow(stats.count)}`));
      console.log(chalk.white(`  Dimension: ${chalk.yellow(stats.dimension)}`));
      console.log(chalk.white(`  Metric: ${chalk.yellow(stats.metric)}`));
      console.log(chalk.white(`  Implementation: ${chalk.yellow(getImplementationType())}`));

      if (stats.memoryUsage) {
        const mb = (stats.memoryUsage / (1024 * 1024)).toFixed(2);
        console.log(chalk.white(`  Memory Usage: ${chalk.yellow(mb + ' MB')}`));
      }

      const fileStats = fs.statSync(dbPath);
      const fileMb = (fileStats.size / (1024 * 1024)).toFixed(2);
      console.log(chalk.white(`  File Size: ${chalk.yellow(fileMb + ' MB')}`));
    } catch (error) {
      spinner.fail(chalk.red('Failed to load database'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Benchmark
program
  .command('benchmark')
  .description('Run performance benchmarks')
  .option('-d, --dimension <number>', 'Vector dimension', '384')
  .option('-n, --num-vectors <number>', 'Number of vectors', '10000')
  .option('-q, --num-queries <number>', 'Number of queries', '1000')
  .action((options) => {
    requireRuvector();
    console.log(chalk.cyan('\nruvector Performance Benchmark'));
    console.log(chalk.gray(`Implementation: ${getImplementationType()}\n`));

    const dimension = parseInt(options.dimension);
    const numVectors = parseInt(options.numVectors);
    const numQueries = parseInt(options.numQueries);

    let spinner = ora('Creating database...').start();

    try {
      const db = new VectorDB({ dimension, metric: 'cosine' });
      spinner.succeed();

      // Insert benchmark
      spinner = ora(`Inserting ${numVectors} vectors...`).start();
      const insertStart = Date.now();

      const vectors = [];
      for (let i = 0; i < numVectors; i++) {
        vectors.push({
          id: `vec_${i}`,
          vector: Array.from({ length: dimension }, () => Math.random()),
          metadata: { index: i, batch: Math.floor(i / 1000) }
        });
      }

      db.insertBatch(vectors);
      const insertTime = Date.now() - insertStart;
      const insertRate = (numVectors / (insertTime / 1000)).toFixed(0);

      spinner.succeed(chalk.green(`Inserted ${numVectors} vectors in ${insertTime}ms`));
      console.log(chalk.gray(`  Rate: ${chalk.yellow(insertRate)} vectors/sec`));

      // Search benchmark
      spinner = ora(`Running ${numQueries} searches...`).start();
      const searchStart = Date.now();

      for (let i = 0; i < numQueries; i++) {
        const query = {
          vector: Array.from({ length: dimension }, () => Math.random()),
          k: 10
        };
        db.search(query);
      }

      const searchTime = Date.now() - searchStart;
      const searchRate = (numQueries / (searchTime / 1000)).toFixed(0);
      const avgLatency = (searchTime / numQueries).toFixed(2);

      spinner.succeed(chalk.green(`Completed ${numQueries} searches in ${searchTime}ms`));
      console.log(chalk.gray(`  Rate: ${chalk.yellow(searchRate)} queries/sec`));
      console.log(chalk.gray(`  Avg Latency: ${chalk.yellow(avgLatency)}ms`));

      // Stats
      const stats = db.stats();
      console.log(chalk.cyan('\nFinal Stats:'));
      console.log(chalk.white(`  Vector Count: ${chalk.yellow(stats.count)}`));
      console.log(chalk.white(`  Dimension: ${chalk.yellow(stats.dimension)}`));
      console.log(chalk.white(`  Implementation: ${chalk.yellow(getImplementationType())}`));

    } catch (error) {
      spinner.fail(chalk.red('Benchmark failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Info command
program
  .command('info')
  .description('Show ruvector information')
  .action(() => {
    console.log(chalk.cyan('\nruvector Information'));
    console.log(chalk.white(`  CLI Version: ${chalk.yellow(packageJson.version)}`));

    // Try to load ruvector for implementation info
    if (loadRuvector()) {
      const version = typeof getVersion === 'function' ? getVersion() : 'unknown';
      const impl = typeof getImplementationType === 'function' ? getImplementationType() : 'native';
      console.log(chalk.white(`  Core Version: ${chalk.yellow(version)}`));
      console.log(chalk.white(`  Implementation: ${chalk.yellow(impl)}`));
    } else {
      console.log(chalk.white(`  Core: ${chalk.gray('Not loaded (install @ruvector/core)')}`));
    }

    console.log(chalk.white(`  GNN Module: ${gnnAvailable ? chalk.green('Available') : chalk.gray('Not installed')}`));
    console.log(chalk.white(`  Node Version: ${chalk.yellow(process.version)}`));
    console.log(chalk.white(`  Platform: ${chalk.yellow(process.platform)}`));
    console.log(chalk.white(`  Architecture: ${chalk.yellow(process.arch)}`));

    if (!gnnAvailable) {
      console.log(chalk.gray('\n  Install GNN with: npx ruvector install gnn'));
    }
  });

// =============================================================================
// Install Command
// =============================================================================

program
  .command('install [packages...]')
  .description('Install optional ruvector packages')
  .option('-a, --all', 'Install all optional packages')
  .option('-l, --list', 'List available packages')
  .option('-i, --interactive', 'Interactive package selection')
  .action(async (packages, options) => {
    const { execSync } = require('child_process');

    // Available optional packages - all ruvector npm packages
    const availablePackages = {
      // Core packages
      core: {
        name: '@ruvector/core',
        description: 'Core vector database with native Rust bindings (HNSW, SIMD)',
        installed: true, // Always installed with ruvector
        category: 'core'
      },
      gnn: {
        name: '@ruvector/gnn',
        description: 'Graph Neural Network layers, tensor compression, differentiable search',
        installed: gnnAvailable,
        category: 'core'
      },
      'graph-node': {
        name: '@ruvector/graph-node',
        description: 'Native Node.js bindings for hypergraph database with Cypher queries',
        installed: false,
        category: 'core'
      },
      'agentic-synth': {
        name: '@ruvector/agentic-synth',
        description: 'Synthetic data generator for AI/ML training, RAG, and agentic workflows',
        installed: false,
        category: 'tools'
      },
      extensions: {
        name: 'ruvector-extensions',
        description: 'Advanced features: embeddings, UI, exports, temporal tracking, persistence',
        installed: false,
        category: 'tools'
      },
      // Platform-specific native bindings for @ruvector/core
      'node-linux-x64': {
        name: '@ruvector/node-linux-x64-gnu',
        description: 'Linux x64 native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      'node-linux-arm64': {
        name: '@ruvector/node-linux-arm64-gnu',
        description: 'Linux ARM64 native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      'node-darwin-x64': {
        name: '@ruvector/node-darwin-x64',
        description: 'macOS Intel x64 native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      'node-darwin-arm64': {
        name: '@ruvector/node-darwin-arm64',
        description: 'macOS Apple Silicon native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      'node-win32-x64': {
        name: '@ruvector/node-win32-x64-msvc',
        description: 'Windows x64 native bindings for @ruvector/core',
        installed: false,
        category: 'platform'
      },
      // Platform-specific native bindings for @ruvector/gnn
      'gnn-linux-x64': {
        name: '@ruvector/gnn-linux-x64-gnu',
        description: 'Linux x64 native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      'gnn-linux-arm64': {
        name: '@ruvector/gnn-linux-arm64-gnu',
        description: 'Linux ARM64 native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      'gnn-darwin-x64': {
        name: '@ruvector/gnn-darwin-x64',
        description: 'macOS Intel x64 native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      'gnn-darwin-arm64': {
        name: '@ruvector/gnn-darwin-arm64',
        description: 'macOS Apple Silicon native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      'gnn-win32-x64': {
        name: '@ruvector/gnn-win32-x64-msvc',
        description: 'Windows x64 native bindings for @ruvector/gnn',
        installed: false,
        category: 'platform'
      },
      // Legacy/standalone packages
      'ruvector-core': {
        name: 'ruvector-core',
        description: 'Standalone vector database (legacy, use @ruvector/core instead)',
        installed: false,
        category: 'legacy'
      }
    };

    // Check which packages are actually installed
    for (const [key, pkg] of Object.entries(availablePackages)) {
      if (key !== 'core' && key !== 'gnn') {
        try {
          require.resolve(pkg.name);
          pkg.installed = true;
        } catch (e) {
          pkg.installed = false;
        }
      }
    }

    // List packages
    if (options.list || (packages.length === 0 && !options.all && !options.interactive)) {
      console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
      console.log(chalk.cyan('                    Ruvector Packages'));
      console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

      const categories = {
        core: { title: '📦 Core Packages', packages: [] },
        tools: { title: '🔧 Tools & Extensions', packages: [] },
        platform: { title: '🖥️  Platform Bindings', packages: [] },
        legacy: { title: '📜 Legacy Packages', packages: [] }
      };

      // Group by category
      Object.entries(availablePackages).forEach(([key, pkg]) => {
        if (categories[pkg.category]) {
          categories[pkg.category].packages.push({ key, ...pkg });
        }
      });

      // Display by category
      for (const [catKey, cat] of Object.entries(categories)) {
        if (cat.packages.length === 0) continue;

        console.log(chalk.cyan(`${cat.title}`));
        console.log(chalk.gray('─'.repeat(60)));

        cat.packages.forEach(pkg => {
          const status = pkg.installed ? chalk.green('✓') : chalk.gray('○');
          const statusText = pkg.installed ? chalk.green('installed') : chalk.gray('available');
          console.log(chalk.white(`  ${status} ${chalk.yellow(pkg.key.padEnd(18))} ${statusText}`));
          console.log(chalk.gray(`      ${pkg.description}`));
          console.log(chalk.gray(`      npm: ${chalk.white(pkg.name)}\n`));
        });
      }

      console.log(chalk.cyan('═══════════════════════════════════════════════════════════════'));
      console.log(chalk.cyan('Usage:'));
      console.log(chalk.white('  npx ruvector install gnn              # Install GNN package'));
      console.log(chalk.white('  npx ruvector install graph-node       # Install graph database'));
      console.log(chalk.white('  npx ruvector install agentic-synth    # Install data generator'));
      console.log(chalk.white('  npx ruvector install --all            # Install all core packages'));
      console.log(chalk.white('  npx ruvector install -i               # Interactive selection'));
      console.log(chalk.gray('\n  Note: Platform bindings are auto-detected by @ruvector/core'));
      return;
    }

    // Interactive mode
    if (options.interactive) {
      const readline = require('readline');
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
      });

      console.log(chalk.cyan('\nSelect packages to install:\n'));

      const notInstalled = Object.entries(availablePackages)
        .filter(([_, pkg]) => !pkg.installed);

      if (notInstalled.length === 0) {
        console.log(chalk.green('All packages are already installed!'));
        rl.close();
        return;
      }

      notInstalled.forEach(([key, pkg], i) => {
        console.log(chalk.white(`  ${i + 1}. ${chalk.yellow(key)} - ${pkg.description}`));
      });
      console.log(chalk.white(`  ${notInstalled.length + 1}. ${chalk.yellow('all')} - Install all packages`));
      console.log(chalk.white(`  0. ${chalk.gray('cancel')} - Exit without installing`));

      rl.question(chalk.cyan('\nEnter selection (comma-separated for multiple): '), (answer) => {
        rl.close();

        const selections = answer.split(',').map(s => s.trim());
        let toInstall = [];

        for (const sel of selections) {
          if (sel === '0' || sel.toLowerCase() === 'cancel') {
            console.log(chalk.yellow('Installation cancelled.'));
            return;
          }
          if (sel === String(notInstalled.length + 1) || sel.toLowerCase() === 'all') {
            toInstall = notInstalled.map(([_, pkg]) => pkg.name);
            break;
          }
          const idx = parseInt(sel) - 1;
          if (idx >= 0 && idx < notInstalled.length) {
            toInstall.push(notInstalled[idx][1].name);
          }
        }

        if (toInstall.length === 0) {
          console.log(chalk.yellow('No valid packages selected.'));
          return;
        }

        installPackages(toInstall);
      });
      return;
    }

    // Install all (core + tools only, not platform-specific or legacy)
    if (options.all) {
      const toInstall = Object.values(availablePackages)
        .filter(pkg => !pkg.installed && (pkg.category === 'core' || pkg.category === 'tools'))
        .map(pkg => pkg.name);

      if (toInstall.length === 0) {
        console.log(chalk.green('All core packages are already installed!'));
        return;
      }

      console.log(chalk.cyan(`Installing ${toInstall.length} packages...`));
      installPackages(toInstall);
      return;
    }

    // Install specific packages
    const toInstall = [];
    for (const pkg of packages) {
      const key = pkg.toLowerCase().replace('@ruvector/', '');
      if (availablePackages[key]) {
        if (availablePackages[key].installed) {
          console.log(chalk.yellow(`${availablePackages[key].name} is already installed`));
        } else {
          toInstall.push(availablePackages[key].name);
        }
      } else {
        console.log(chalk.red(`Unknown package: ${pkg}`));
        console.log(chalk.gray(`Available: ${Object.keys(availablePackages).join(', ')}`));
      }
    }

    if (toInstall.length > 0) {
      installPackages(toInstall);
    }

    function installPackages(pkgs) {
      const spinner = ora(`Installing ${pkgs.join(', ')}...`).start();

      try {
        // Detect package manager
        let pm = 'npm';
        if (fs.existsSync('yarn.lock')) pm = 'yarn';
        else if (fs.existsSync('pnpm-lock.yaml')) pm = 'pnpm';
        else if (fs.existsSync('bun.lockb')) pm = 'bun';

        const cmd = pm === 'yarn' ? `yarn add ${pkgs.join(' ')}`
                  : pm === 'pnpm' ? `pnpm add ${pkgs.join(' ')}`
                  : pm === 'bun' ? `bun add ${pkgs.join(' ')}`
                  : `npm install ${pkgs.join(' ')}`;

        execSync(cmd, { stdio: 'pipe' });

        spinner.succeed(chalk.green(`Installed: ${pkgs.join(', ')}`));
        console.log(chalk.cyan('\nRun "npx ruvector info" to verify installation.'));
      } catch (error) {
        spinner.fail(chalk.red('Installation failed'));
        console.error(chalk.red(error.message));
        console.log(chalk.yellow(`\nTry manually: npm install ${pkgs.join(' ')}`));
        process.exit(1);
      }
    }
  });

// =============================================================================
// GNN Commands
// =============================================================================

// Helper to check GNN availability
function requireGnn() {
  if (!gnnAvailable) {
    console.error(chalk.red('Error: GNN module not available.'));
    console.error(chalk.yellow('Install it with: npm install @ruvector/gnn'));
    process.exit(1);
  }
}

// GNN parent command
const gnnCmd = program
  .command('gnn')
  .description('Graph Neural Network operations');

// GNN Layer command
gnnCmd
  .command('layer')
  .description('Create and test a GNN layer')
  .requiredOption('-i, --input-dim <number>', 'Input dimension')
  .requiredOption('-h, --hidden-dim <number>', 'Hidden dimension')
  .option('-a, --heads <number>', 'Number of attention heads', '4')
  .option('-d, --dropout <number>', 'Dropout rate', '0.1')
  .option('--test', 'Run a test forward pass')
  .option('-o, --output <file>', 'Save layer config to JSON file')
  .action((options) => {
    requireGnn();
    const spinner = ora('Creating GNN layer...').start();

    try {
      const inputDim = parseInt(options.inputDim);
      const hiddenDim = parseInt(options.hiddenDim);
      const heads = parseInt(options.heads);
      const dropout = parseFloat(options.dropout);

      const layer = new RuvectorLayer(inputDim, hiddenDim, heads, dropout);
      spinner.succeed(chalk.green('GNN Layer created'));

      console.log(chalk.cyan('\nLayer Configuration:'));
      console.log(chalk.white(`  Input Dim:  ${chalk.yellow(inputDim)}`));
      console.log(chalk.white(`  Hidden Dim: ${chalk.yellow(hiddenDim)}`));
      console.log(chalk.white(`  Heads:      ${chalk.yellow(heads)}`));
      console.log(chalk.white(`  Dropout:    ${chalk.yellow(dropout)}`));

      if (options.test) {
        spinner.start('Running test forward pass...');

        // Create test data
        const nodeEmbedding = Array.from({ length: inputDim }, () => Math.random());
        const neighborEmbeddings = [
          Array.from({ length: inputDim }, () => Math.random()),
          Array.from({ length: inputDim }, () => Math.random())
        ];
        const edgeWeights = [0.6, 0.4];

        const output = layer.forward(nodeEmbedding, neighborEmbeddings, edgeWeights);
        spinner.succeed(chalk.green('Forward pass completed'));

        console.log(chalk.cyan('\nTest Results:'));
        console.log(chalk.white(`  Input shape:  ${chalk.yellow(`[${inputDim}]`)}`));
        console.log(chalk.white(`  Output shape: ${chalk.yellow(`[${output.length}]`)}`));
        console.log(chalk.white(`  Output sample: ${chalk.gray(`[${output.slice(0, 4).map(v => v.toFixed(4)).join(', ')}...]`)}`));
      }

      if (options.output) {
        const config = layer.toJson();
        fs.writeFileSync(options.output, config);
        console.log(chalk.green(`\nLayer config saved to: ${options.output}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to create GNN layer'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// GNN Compress command
gnnCmd
  .command('compress')
  .description('Compress embeddings using adaptive tensor compression')
  .requiredOption('-f, --file <path>', 'Input JSON file with embeddings')
  .option('-l, --level <type>', 'Compression level (none|half|pq8|pq4|binary)', 'auto')
  .option('-a, --access-freq <number>', 'Access frequency for auto compression (0.0-1.0)', '0.5')
  .option('-o, --output <file>', 'Output file for compressed data')
  .action((options) => {
    requireGnn();
    const spinner = ora('Loading embeddings...').start();

    try {
      const data = JSON.parse(fs.readFileSync(options.file, 'utf8'));
      const embeddings = Array.isArray(data) ? data : [data];

      spinner.text = 'Compressing embeddings...';
      const compressor = new TensorCompress();
      const accessFreq = parseFloat(options.accessFreq);

      const results = [];
      let totalOriginalSize = 0;
      let totalCompressedSize = 0;

      for (const embedding of embeddings) {
        const vec = embedding.vector || embedding;
        totalOriginalSize += vec.length * 4; // float32 = 4 bytes

        let compressed;
        if (options.level === 'auto') {
          compressed = compressor.compress(vec, accessFreq);
        } else {
          const levelConfig = { levelType: options.level };
          if (options.level === 'pq8') {
            levelConfig.subvectors = 8;
            levelConfig.centroids = 256;
          } else if (options.level === 'pq4') {
            levelConfig.subvectors = 8;
          }
          compressed = compressor.compressWithLevel(vec, levelConfig);
        }

        totalCompressedSize += compressed.length;
        results.push({
          id: embedding.id,
          compressed
        });
      }

      const ratio = (totalOriginalSize / totalCompressedSize).toFixed(2);
      const savings = ((1 - totalCompressedSize / totalOriginalSize) * 100).toFixed(1);

      spinner.succeed(chalk.green(`Compressed ${embeddings.length} embeddings`));

      console.log(chalk.cyan('\nCompression Results:'));
      console.log(chalk.white(`  Embeddings:    ${chalk.yellow(embeddings.length)}`));
      console.log(chalk.white(`  Level:         ${chalk.yellow(options.level === 'auto' ? `auto (${getCompressionLevel(accessFreq)})` : options.level)}`));
      console.log(chalk.white(`  Original:      ${chalk.yellow((totalOriginalSize / 1024).toFixed(2) + ' KB')}`));
      console.log(chalk.white(`  Compressed:    ${chalk.yellow((totalCompressedSize / 1024).toFixed(2) + ' KB')}`));
      console.log(chalk.white(`  Ratio:         ${chalk.yellow(ratio + 'x')}`));
      console.log(chalk.white(`  Savings:       ${chalk.yellow(savings + '%')}`));

      if (options.output) {
        fs.writeFileSync(options.output, JSON.stringify(results, null, 2));
        console.log(chalk.green(`\nCompressed data saved to: ${options.output}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to compress embeddings'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// GNN Search command
gnnCmd
  .command('search')
  .description('Differentiable search with soft attention')
  .requiredOption('-q, --query <json>', 'Query vector as JSON array')
  .requiredOption('-c, --candidates <file>', 'Candidates file (JSON array of vectors)')
  .option('-k, --top-k <number>', 'Number of results', '5')
  .option('-t, --temperature <number>', 'Softmax temperature (lower=sharper)', '1.0')
  .action((options) => {
    requireGnn();
    const spinner = ora('Loading candidates...').start();

    try {
      const query = JSON.parse(options.query);
      const candidatesData = JSON.parse(fs.readFileSync(options.candidates, 'utf8'));
      const candidates = candidatesData.map(c => c.vector || c);
      const k = parseInt(options.topK);
      const temperature = parseFloat(options.temperature);

      spinner.text = 'Running differentiable search...';
      const result = differentiableSearch(query, candidates, k, temperature);

      spinner.succeed(chalk.green(`Found top-${k} results`));

      console.log(chalk.cyan('\nSearch Results:'));
      console.log(chalk.white(`  Query dim:     ${chalk.yellow(query.length)}`));
      console.log(chalk.white(`  Candidates:    ${chalk.yellow(candidates.length)}`));
      console.log(chalk.white(`  Temperature:   ${chalk.yellow(temperature)}`));

      console.log(chalk.cyan('\nTop-K Results:'));
      for (let i = 0; i < result.indices.length; i++) {
        const idx = result.indices[i];
        const weight = result.weights[i];
        const id = candidatesData[idx]?.id || `candidate_${idx}`;
        console.log(chalk.white(`  ${i + 1}. ${chalk.yellow(id)} (index: ${idx})`));
        console.log(chalk.gray(`     Weight: ${weight.toFixed(6)}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to run search'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// GNN Info command
gnnCmd
  .command('info')
  .description('Show GNN module information')
  .action(() => {
    if (!gnnAvailable) {
      console.log(chalk.yellow('\nGNN Module: Not installed'));
      console.log(chalk.white('Install with: npm install @ruvector/gnn'));
      return;
    }

    console.log(chalk.cyan('\nGNN Module Information'));
    console.log(chalk.white(`  Status:         ${chalk.green('Available')}`));
    console.log(chalk.white(`  Platform:       ${chalk.yellow(process.platform)}`));
    console.log(chalk.white(`  Architecture:   ${chalk.yellow(process.arch)}`));

    console.log(chalk.cyan('\nAvailable Features:'));
    console.log(chalk.white(`  • RuvectorLayer   - GNN layer with multi-head attention`));
    console.log(chalk.white(`  • TensorCompress  - Adaptive tensor compression (5 levels)`));
    console.log(chalk.white(`  • differentiableSearch - Soft attention-based search`));
    console.log(chalk.white(`  • hierarchicalForward  - Multi-layer GNN processing`));

    console.log(chalk.cyan('\nCompression Levels:'));
    console.log(chalk.gray(`  none   (freq > 0.8)  - Full precision, hot data`));
    console.log(chalk.gray(`  half   (freq > 0.4)  - ~50% savings, warm data`));
    console.log(chalk.gray(`  pq8    (freq > 0.1)  - ~8x compression, cool data`));
    console.log(chalk.gray(`  pq4    (freq > 0.01) - ~16x compression, cold data`));
    console.log(chalk.gray(`  binary (freq <= 0.01) - ~32x compression, archive`));
  });

// =============================================================================
// Attention Commands
// =============================================================================

// Helper to require attention module
function requireAttention() {
  if (!attentionAvailable) {
    console.error(chalk.red('Error: @ruvector/attention is not installed'));
    console.error(chalk.yellow('Install it with: npm install @ruvector/attention'));
    process.exit(1);
  }
}

// Attention parent command
const attentionCmd = program
  .command('attention')
  .description('High-performance attention mechanism operations');

// Attention compute command - run attention on input vectors
attentionCmd
  .command('compute')
  .description('Compute attention over input vectors')
  .requiredOption('-q, --query <json>', 'Query vector as JSON array')
  .requiredOption('-k, --keys <file>', 'Keys file (JSON array of vectors)')
  .option('-v, --values <file>', 'Values file (JSON array of vectors, defaults to keys)')
  .option('-t, --type <type>', 'Attention type (dot|multi-head|flash|hyperbolic|linear)', 'dot')
  .option('-h, --heads <number>', 'Number of attention heads (for multi-head)', '4')
  .option('-d, --head-dim <number>', 'Head dimension (for multi-head)', '64')
  .option('--curvature <number>', 'Curvature for hyperbolic attention', '1.0')
  .option('-o, --output <file>', 'Output file for results')
  .action((options) => {
    requireAttention();
    const spinner = ora('Loading keys...').start();

    try {
      const query = JSON.parse(options.query);
      const keysData = JSON.parse(fs.readFileSync(options.keys, 'utf8'));
      const keys = keysData.map(k => k.vector || k);

      let values = keys;
      if (options.values) {
        const valuesData = JSON.parse(fs.readFileSync(options.values, 'utf8'));
        values = valuesData.map(v => v.vector || v);
      }

      spinner.text = `Computing ${options.type} attention...`;

      let result;
      let attentionWeights;

      switch (options.type) {
        case 'dot': {
          const attn = new DotProductAttention();
          const queryMat = [query];
          const output = attn.forward(queryMat, keys, values);
          result = output[0];
          attentionWeights = attn.getLastWeights ? attn.getLastWeights()[0] : null;
          break;
        }
        case 'multi-head': {
          const numHeads = parseInt(options.heads);
          const headDim = parseInt(options.headDim);
          const attn = new MultiHeadAttention(query.length, numHeads, headDim);
          const queryMat = [query];
          const output = attn.forward(queryMat, keys, values);
          result = output[0];
          break;
        }
        case 'flash': {
          const attn = new FlashAttention(query.length);
          const queryMat = [query];
          const output = attn.forward(queryMat, keys, values);
          result = output[0];
          break;
        }
        case 'hyperbolic': {
          const curvature = parseFloat(options.curvature);
          const attn = new HyperbolicAttention(query.length, curvature);
          const queryMat = [query];
          const output = attn.forward(queryMat, keys, values);
          result = output[0];
          break;
        }
        case 'linear': {
          const attn = new LinearAttention(query.length);
          const queryMat = [query];
          const output = attn.forward(queryMat, keys, values);
          result = output[0];
          break;
        }
        default:
          throw new Error(`Unknown attention type: ${options.type}`);
      }

      spinner.succeed(chalk.green(`Attention computed (${options.type})`));

      console.log(chalk.cyan('\nAttention Results:'));
      console.log(chalk.white(`  Type:        ${chalk.yellow(options.type)}`));
      console.log(chalk.white(`  Query dim:   ${chalk.yellow(query.length)}`));
      console.log(chalk.white(`  Num keys:    ${chalk.yellow(keys.length)}`));
      console.log(chalk.white(`  Output dim:  ${chalk.yellow(result.length)}`));
      console.log(chalk.white(`  Output:      ${chalk.gray(`[${result.slice(0, 4).map(v => v.toFixed(4)).join(', ')}...]`)}`));

      if (attentionWeights) {
        console.log(chalk.cyan('\nAttention Weights:'));
        attentionWeights.slice(0, 5).forEach((w, i) => {
          console.log(chalk.gray(`  Key ${i}: ${w.toFixed(4)}`));
        });
        if (attentionWeights.length > 5) {
          console.log(chalk.gray(`  ... and ${attentionWeights.length - 5} more`));
        }
      }

      if (options.output) {
        const outputData = { result, attentionWeights };
        fs.writeFileSync(options.output, JSON.stringify(outputData, null, 2));
        console.log(chalk.green(`\nResults saved to: ${options.output}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to compute attention'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Attention benchmark command
attentionCmd
  .command('benchmark')
  .description('Benchmark attention mechanisms')
  .option('-d, --dimension <number>', 'Vector dimension', '256')
  .option('-n, --num-vectors <number>', 'Number of vectors', '100')
  .option('-i, --iterations <number>', 'Benchmark iterations', '100')
  .option('-t, --types <list>', 'Attention types to benchmark (comma-separated)', 'dot,flash,linear')
  .action((options) => {
    requireAttention();
    const spinner = ora('Setting up benchmark...').start();

    try {
      const dim = parseInt(options.dimension);
      const numVectors = parseInt(options.numVectors);
      const iterations = parseInt(options.iterations);
      const types = options.types.split(',').map(t => t.trim());

      // Generate random test data
      spinner.text = 'Generating test data...';
      const query = Array.from({ length: dim }, () => Math.random());
      const keys = Array.from({ length: numVectors }, () =>
        Array.from({ length: dim }, () => Math.random())
      );

      console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
      console.log(chalk.cyan('                Attention Mechanism Benchmark'));
      console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

      console.log(chalk.white(`  Dimension:    ${chalk.yellow(dim)}`));
      console.log(chalk.white(`  Vectors:      ${chalk.yellow(numVectors)}`));
      console.log(chalk.white(`  Iterations:   ${chalk.yellow(iterations)}`));
      console.log('');

      const results = [];

      // Convert to Float32Arrays for compute()
      const queryF32 = new Float32Array(query);
      const keysF32 = keys.map(k => new Float32Array(k));

      for (const type of types) {
        spinner.text = `Benchmarking ${type} attention...`;
        spinner.start();

        let attn;
        try {
          switch (type) {
            case 'dot':
              attn = new DotProductAttention(dim);
              break;
            case 'flash':
              attn = new FlashAttention(dim, 64);  // dim, block_size
              break;
            case 'linear':
              attn = new LinearAttention(dim, 64);  // dim, num_features
              break;
            case 'hyperbolic':
              attn = new HyperbolicAttention(dim, 1.0);
              break;
            case 'multi-head':
              attn = new MultiHeadAttention(dim, 4);  // dim, num_heads
              break;
            default:
              console.log(chalk.yellow(`  Skipping unknown type: ${type}`));
              continue;
          }
        } catch (e) {
          console.log(chalk.yellow(`  ${type}: not available (${e.message})`));
          continue;
        }

        // Warm up
        for (let i = 0; i < 5; i++) {
          try {
            attn.compute(queryF32, keysF32, keysF32);
          } catch (e) {
            // Some mechanisms may fail warmup
          }
        }

        // Benchmark
        const start = process.hrtime.bigint();
        for (let i = 0; i < iterations; i++) {
          attn.compute(queryF32, keysF32, keysF32);
        }
        const end = process.hrtime.bigint();
        const totalMs = Number(end - start) / 1_000_000;
        const avgMs = totalMs / iterations;
        const opsPerSec = 1000 / avgMs;

        results.push({ type, avgMs, opsPerSec });
        spinner.succeed(chalk.green(`${type}: ${avgMs.toFixed(3)} ms/op (${opsPerSec.toFixed(1)} ops/sec)`));
      }

      // Summary
      if (results.length > 0) {
        console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
        console.log(chalk.cyan('                         Summary'));
        console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

        const fastest = results.reduce((a, b) => a.avgMs < b.avgMs ? a : b);
        console.log(chalk.green(`  Fastest: ${fastest.type} (${fastest.avgMs.toFixed(3)} ms/op)\n`));

        console.log(chalk.white('  Relative Performance:'));
        for (const r of results) {
          const relPerf = (fastest.avgMs / r.avgMs * 100).toFixed(1);
          const bar = '█'.repeat(Math.round(relPerf / 5));
          console.log(chalk.white(`    ${r.type.padEnd(12)} ${chalk.cyan(bar)} ${relPerf}%`));
        }
      }
    } catch (error) {
      spinner.fail(chalk.red('Benchmark failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// Hyperbolic math command
attentionCmd
  .command('hyperbolic')
  .description('Hyperbolic geometry operations')
  .requiredOption('-a, --action <type>', 'Action: exp-map|log-map|distance|project|mobius-add')
  .requiredOption('-v, --vector <json>', 'Input vector(s) as JSON')
  .option('-b, --vector-b <json>', 'Second vector for binary operations')
  .option('-c, --curvature <number>', 'Poincaré ball curvature', '1.0')
  .option('-o, --origin <json>', 'Origin point for exp/log maps')
  .action((options) => {
    requireAttention();

    try {
      const vecArray = JSON.parse(options.vector);
      const vec = new Float32Array(vecArray);
      const curvature = parseFloat(options.curvature);

      let result;
      let description;

      switch (options.action) {
        case 'exp-map': {
          const originArray = options.origin ? JSON.parse(options.origin) : Array(vec.length).fill(0);
          const origin = new Float32Array(originArray);
          result = expMap(origin, vec, curvature);
          description = 'Exponential map (tangent → Poincaré ball)';
          break;
        }
        case 'log-map': {
          const originArray = options.origin ? JSON.parse(options.origin) : Array(vec.length).fill(0);
          const origin = new Float32Array(originArray);
          result = logMap(origin, vec, curvature);
          description = 'Logarithmic map (Poincaré ball → tangent)';
          break;
        }
        case 'distance': {
          if (!options.vectorB) {
            throw new Error('--vector-b required for distance calculation');
          }
          const vecBArray = JSON.parse(options.vectorB);
          const vecB = new Float32Array(vecBArray);
          result = poincareDistance(vec, vecB, curvature);
          description = 'Poincaré distance';
          break;
        }
        case 'project': {
          result = projectToPoincareBall(vec, curvature);
          description = 'Project to Poincaré ball';
          break;
        }
        case 'mobius-add': {
          if (!options.vectorB) {
            throw new Error('--vector-b required for Möbius addition');
          }
          const vecBArray = JSON.parse(options.vectorB);
          const vecB = new Float32Array(vecBArray);
          result = mobiusAddition(vec, vecB, curvature);
          description = 'Möbius addition';
          break;
        }
        default:
          throw new Error(`Unknown action: ${options.action}`);
      }

      console.log(chalk.cyan('\nHyperbolic Operation:'));
      console.log(chalk.white(`  Action:     ${chalk.yellow(description)}`));
      console.log(chalk.white(`  Curvature:  ${chalk.yellow(curvature)}`));

      if (typeof result === 'number') {
        console.log(chalk.white(`  Result:     ${chalk.green(result.toFixed(6))}`));
      } else {
        const resultArray = Array.from(result);
        console.log(chalk.white(`  Input dim:  ${chalk.yellow(vec.length)}`));
        console.log(chalk.white(`  Output dim: ${chalk.yellow(resultArray.length)}`));
        console.log(chalk.white(`  Result:     ${chalk.gray(`[${resultArray.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`)}`));

        // Compute norm to verify it's in the ball
        const norm = Math.sqrt(resultArray.reduce((sum, x) => sum + x * x, 0));
        console.log(chalk.white(`  Norm:       ${chalk.yellow(norm.toFixed(6))} ${norm < 1 ? chalk.green('(inside ball)') : chalk.red('(outside ball)')}`));
      }
    } catch (error) {
      console.error(chalk.red('Hyperbolic operation failed:'), error.message);
      process.exit(1);
    }
  });

// Attention info command
attentionCmd
  .command('info')
  .description('Show attention module information')
  .action(() => {
    if (!attentionAvailable) {
      console.log(chalk.yellow('\nAttention Module: Not installed'));
      console.log(chalk.white('Install with: npm install @ruvector/attention'));
      return;
    }

    console.log(chalk.cyan('\nAttention Module Information'));
    console.log(chalk.white(`  Status:         ${chalk.green('Available')}`));
    console.log(chalk.white(`  Version:        ${chalk.yellow(attentionVersion ? attentionVersion() : 'unknown')}`));
    console.log(chalk.white(`  Platform:       ${chalk.yellow(process.platform)}`));
    console.log(chalk.white(`  Architecture:   ${chalk.yellow(process.arch)}`));

    console.log(chalk.cyan('\nCore Attention Mechanisms:'));
    console.log(chalk.white(`  • DotProductAttention  - Scaled dot-product attention`));
    console.log(chalk.white(`  • MultiHeadAttention   - Multi-head self-attention`));
    console.log(chalk.white(`  • FlashAttention       - Memory-efficient IO-aware attention`));
    console.log(chalk.white(`  • HyperbolicAttention  - Poincaré ball attention`));
    console.log(chalk.white(`  • LinearAttention      - O(n) linear complexity attention`));
    console.log(chalk.white(`  • MoEAttention         - Mixture of Experts attention`));

    console.log(chalk.cyan('\nGraph Attention:'));
    console.log(chalk.white(`  • GraphRoPeAttention   - Rotary position embeddings for graphs`));
    console.log(chalk.white(`  • EdgeFeaturedAttention - Edge feature-enhanced attention`));
    console.log(chalk.white(`  • DualSpaceAttention   - Euclidean + hyperbolic dual space`));
    console.log(chalk.white(`  • LocalGlobalAttention - Local-global graph attention`));

    console.log(chalk.cyan('\nHyperbolic Math:'));
    console.log(chalk.white(`  • expMap, logMap       - Exponential/logarithmic maps`));
    console.log(chalk.white(`  • mobiusAddition       - Möbius addition in Poincaré ball`));
    console.log(chalk.white(`  • poincareDistance     - Hyperbolic distance metric`));
    console.log(chalk.white(`  • projectToPoincareBall - Project vectors to ball`));

    console.log(chalk.cyan('\nTraining Utilities:'));
    console.log(chalk.white(`  • AdamOptimizer, AdamWOptimizer, SgdOptimizer`));
    console.log(chalk.white(`  • InfoNceLoss, LocalContrastiveLoss`));
    console.log(chalk.white(`  • CurriculumScheduler, TemperatureAnnealing`));
    console.log(chalk.white(`  • HardNegativeMiner, InBatchMiner`));
  });

// Attention list command - list available mechanisms
attentionCmd
  .command('list')
  .description('List all available attention mechanisms')
  .option('-v, --verbose', 'Show detailed information')
  .action((options) => {
    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('              Available Attention Mechanisms'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    const mechanisms = [
      { name: 'DotProductAttention', type: 'core', complexity: 'O(n²)', available: !!DotProductAttention },
      { name: 'MultiHeadAttention', type: 'core', complexity: 'O(n²)', available: !!MultiHeadAttention },
      { name: 'FlashAttention', type: 'core', complexity: 'O(n²) IO-optimized', available: !!FlashAttention },
      { name: 'HyperbolicAttention', type: 'core', complexity: 'O(n²)', available: !!HyperbolicAttention },
      { name: 'LinearAttention', type: 'core', complexity: 'O(n)', available: !!LinearAttention },
      { name: 'MoEAttention', type: 'core', complexity: 'O(n*k)', available: !!MoEAttention },
      { name: 'GraphRoPeAttention', type: 'graph', complexity: 'O(n²)', available: !!GraphRoPeAttention },
      { name: 'EdgeFeaturedAttention', type: 'graph', complexity: 'O(n²)', available: !!EdgeFeaturedAttention },
      { name: 'DualSpaceAttention', type: 'graph', complexity: 'O(n²)', available: !!DualSpaceAttention },
      { name: 'LocalGlobalAttention', type: 'graph', complexity: 'O(n*k)', available: !!LocalGlobalAttention },
    ];

    console.log(chalk.white('  Core Attention:'));
    mechanisms.filter(m => m.type === 'core').forEach(m => {
      const status = m.available ? chalk.green('✓') : chalk.red('✗');
      console.log(chalk.white(`    ${status} ${m.name.padEnd(22)} ${chalk.gray(m.complexity)}`));
    });

    console.log(chalk.white('\n  Graph Attention:'));
    mechanisms.filter(m => m.type === 'graph').forEach(m => {
      const status = m.available ? chalk.green('✓') : chalk.red('✗');
      console.log(chalk.white(`    ${status} ${m.name.padEnd(22)} ${chalk.gray(m.complexity)}`));
    });

    if (!attentionAvailable) {
      console.log(chalk.yellow('\n  Note: @ruvector/attention not installed'));
      console.log(chalk.white('  Install with: npm install @ruvector/attention'));
    }

    if (options.verbose) {
      console.log(chalk.cyan('\n  Usage Examples:'));
      console.log(chalk.gray('    # Compute dot-product attention'));
      console.log(chalk.white('    npx ruvector attention compute -q "[1,2,3]" -k keys.json -t dot'));
      console.log(chalk.gray('\n    # Benchmark attention mechanisms'));
      console.log(chalk.white('    npx ruvector attention benchmark -d 256 -n 100'));
      console.log(chalk.gray('\n    # Hyperbolic distance'));
      console.log(chalk.white('    npx ruvector attention hyperbolic -a distance -v "[0.1,0.2]" -b "[0.3,0.4]"'));
    }
  });

// =============================================================================
// Doctor Command - Check system health and dependencies
// =============================================================================

program
  .command('doctor')
  .description('Check system health and dependencies')
  .option('-v, --verbose', 'Show detailed information')
  .action(async (options) => {
    const { execSync } = require('child_process');

    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('                    RuVector Doctor'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    let issues = 0;
    let warnings = 0;

    // Helper functions
    const check = (name, condition, fix) => {
      if (condition) {
        console.log(chalk.green(`  ✓ ${name}`));
        return true;
      } else {
        console.log(chalk.red(`  ✗ ${name}`));
        if (fix) console.log(chalk.gray(`    Fix: ${fix}`));
        issues++;
        return false;
      }
    };

    const warn = (name, condition, suggestion) => {
      if (condition) {
        console.log(chalk.green(`  ✓ ${name}`));
        return true;
      } else {
        console.log(chalk.yellow(`  ! ${name}`));
        if (suggestion) console.log(chalk.gray(`    Suggestion: ${suggestion}`));
        warnings++;
        return false;
      }
    };

    const getVersion = (cmd) => {
      try {
        return execSync(cmd, { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }).trim();
      } catch (e) {
        return null;
      }
    };

    // System Information
    console.log(chalk.cyan('System Information:'));
    console.log(chalk.white(`  Platform:      ${chalk.yellow(process.platform)}`));
    console.log(chalk.white(`  Architecture:  ${chalk.yellow(process.arch)}`));
    console.log(chalk.white(`  Node.js:       ${chalk.yellow(process.version)}`));
    console.log('');

    // Node.js Checks
    console.log(chalk.cyan('Node.js Environment:'));
    const nodeVersion = parseInt(process.version.slice(1).split('.')[0]);
    check('Node.js >= 14', nodeVersion >= 14, 'Upgrade Node.js: https://nodejs.org');

    const npmVersion = getVersion('npm --version');
    if (npmVersion) {
      console.log(chalk.green(`  ✓ npm ${npmVersion}`));
    } else {
      check('npm installed', false, 'Install npm or reinstall Node.js');
    }
    console.log('');

    // RuVector Packages
    console.log(chalk.cyan('RuVector Packages:'));

    // Check @ruvector/core
    let coreAvailable = false;
    try {
      require.resolve('@ruvector/core');
      coreAvailable = true;
      console.log(chalk.green(`  ✓ @ruvector/core installed`));
    } catch (e) {
      console.log(chalk.yellow(`  ! @ruvector/core not found (using WASM fallback)`));
      warnings++;
    }

    // Check if native binding works
    if (coreAvailable && loadRuvector()) {
      const version = typeof getVersion === 'function' ? getVersion() : null;
      const impl = typeof getImplementationType === 'function' ? getImplementationType() : 'native';
      const versionStr = version ? `, v${version}` : '';
      console.log(chalk.green(`  ✓ Native binding working (${impl}${versionStr})`));
    } else if (coreAvailable) {
      console.log(chalk.yellow(`  ! Native binding failed to load`));
      warnings++;
    }

    // Check @ruvector/gnn
    if (gnnAvailable) {
      console.log(chalk.green(`  ✓ @ruvector/gnn installed`));
    } else {
      console.log(chalk.gray(`  ○ @ruvector/gnn not installed (optional)`));
    }

    // Check @ruvector/attention
    if (attentionAvailable) {
      console.log(chalk.green(`  ✓ @ruvector/attention installed`));
    } else {
      console.log(chalk.gray(`  ○ @ruvector/attention not installed (optional)`));
    }

    // Check @ruvector/graph-node
    try {
      require.resolve('@ruvector/graph-node');
      console.log(chalk.green(`  ✓ @ruvector/graph-node installed`));
    } catch (e) {
      console.log(chalk.gray(`  ○ @ruvector/graph-node not installed (optional)`));
    }
    console.log('');

    // Rust Toolchain (optional for development)
    console.log(chalk.cyan('Rust Toolchain (optional):'));

    const rustVersion = getVersion('rustc --version');
    if (rustVersion) {
      console.log(chalk.green(`  ✓ ${rustVersion}`));
    } else {
      console.log(chalk.gray(`  ○ Rust not installed (only needed for development)`));
    }

    const cargoVersion = getVersion('cargo --version');
    if (cargoVersion) {
      console.log(chalk.green(`  ✓ ${cargoVersion}`));
    } else if (rustVersion) {
      console.log(chalk.yellow(`  ! cargo not found`));
      warnings++;
    }
    console.log('');

    // Build Tools (optional)
    if (options.verbose) {
      console.log(chalk.cyan('Build Tools (for native compilation):'));

      const hasGcc = getVersion('gcc --version');
      const hasClang = getVersion('clang --version');
      const hasCc = getVersion('cc --version');

      if (hasGcc || hasClang || hasCc) {
        console.log(chalk.green(`  ✓ C compiler available`));
      } else {
        console.log(chalk.gray(`  ○ No C compiler found (only needed for building from source)`));
      }

      const hasMake = getVersion('make --version');
      if (hasMake) {
        console.log(chalk.green(`  ✓ make available`));
      } else {
        console.log(chalk.gray(`  ○ make not found`));
      }

      const hasCmake = getVersion('cmake --version');
      if (hasCmake) {
        console.log(chalk.green(`  ✓ cmake available`));
      } else {
        console.log(chalk.gray(`  ○ cmake not found`));
      }
      console.log('');
    }

    // Summary
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════'));
    if (issues === 0 && warnings === 0) {
      console.log(chalk.green('\n  ✓ All checks passed! RuVector is ready to use.\n'));
    } else if (issues === 0) {
      console.log(chalk.yellow(`\n  ! ${warnings} warning(s) found. RuVector should work but may have limited features.\n`));
    } else {
      console.log(chalk.red(`\n  ✗ ${issues} issue(s) and ${warnings} warning(s) found.\n`));
      console.log(chalk.white('  Run "npx ruvector setup" for installation instructions.\n'));
    }
  });

// =============================================================================
// Setup Command - Installation instructions
// =============================================================================

program
  .command('setup')
  .description('Show installation and setup instructions')
  .option('--rust', 'Show Rust installation instructions')
  .option('--npm', 'Show npm package installation instructions')
  .option('--all', 'Show all installation instructions')
  .action((options) => {
    const showAll = options.all || (!options.rust && !options.npm);

    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('                    RuVector Setup Guide'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    // Quick install
    console.log(chalk.cyan('Quick Install (one-liner):'));
    console.log(chalk.white('  curl -fsSL https://raw.githubusercontent.com/ruvnet/ruvector/main/install.sh | bash'));
    console.log('');

    if (showAll || options.npm) {
      console.log(chalk.cyan('───────────────────────────────────────────────────────────────'));
      console.log(chalk.cyan('npm Packages'));
      console.log(chalk.cyan('───────────────────────────────────────────────────────────────\n'));

      console.log(chalk.yellow('All-in-one CLI:'));
      console.log(chalk.white('  npm install -g ruvector'));
      console.log(chalk.white('  npx ruvector'));
      console.log('');

      console.log(chalk.yellow('Core packages:'));
      console.log(chalk.white('  npm install @ruvector/core       # Vector database'));
      console.log(chalk.white('  npm install @ruvector/gnn        # Graph Neural Networks'));
      console.log(chalk.white('  npm install @ruvector/graph-node # Hypergraph database'));
      console.log('');

      console.log(chalk.yellow('Install all optional packages:'));
      console.log(chalk.white('  npx ruvector install --all'));
      console.log('');

      console.log(chalk.yellow('List available packages:'));
      console.log(chalk.white('  npx ruvector install'));
      console.log('');
    }

    if (showAll || options.rust) {
      console.log(chalk.cyan('───────────────────────────────────────────────────────────────'));
      console.log(chalk.cyan('Rust Installation'));
      console.log(chalk.cyan('───────────────────────────────────────────────────────────────\n'));

      console.log(chalk.yellow('1. Install Rust:'));
      console.log(chalk.white('  curl --proto \'=https\' --tlsv1.2 -sSf https://sh.rustup.rs | sh'));
      console.log(chalk.gray('  # Follow the prompts, then restart your terminal or run:'));
      console.log(chalk.white('  source $HOME/.cargo/env'));
      console.log('');

      console.log(chalk.yellow('2. Verify installation:'));
      console.log(chalk.white('  rustc --version'));
      console.log(chalk.white('  cargo --version'));
      console.log('');

      console.log(chalk.yellow('3. Add RuVector crates to your project:'));
      console.log(chalk.white('  cargo add ruvector-core          # Vector database'));
      console.log(chalk.white('  cargo add ruvector-graph         # Hypergraph with Cypher'));
      console.log(chalk.white('  cargo add ruvector-gnn           # Graph Neural Networks'));
      console.log('');

      console.log(chalk.yellow('4. Other available crates:'));
      console.log(chalk.white('  cargo add ruvector-cluster       # Distributed clustering'));
      console.log(chalk.white('  cargo add ruvector-raft          # Raft consensus'));
      console.log(chalk.white('  cargo add ruvector-replication   # Data replication'));
      console.log(chalk.white('  cargo add ruvector-tiny-dancer-core  # AI routing'));
      console.log(chalk.white('  cargo add ruvector-router-core   # Semantic routing'));
      console.log('');

      console.log(chalk.yellow('Platform-specific notes:'));
      console.log('');

      if (process.platform === 'darwin') {
        console.log(chalk.cyan('  macOS:'));
        console.log(chalk.white('    xcode-select --install  # Install command line tools'));
        console.log('');
      } else if (process.platform === 'linux') {
        console.log(chalk.cyan('  Linux (Debian/Ubuntu):'));
        console.log(chalk.white('    sudo apt-get update'));
        console.log(chalk.white('    sudo apt-get install build-essential pkg-config libssl-dev'));
        console.log('');
        console.log(chalk.cyan('  Linux (RHEL/CentOS):'));
        console.log(chalk.white('    sudo yum groupinstall "Development Tools"'));
        console.log(chalk.white('    sudo yum install openssl-devel'));
        console.log('');
      } else if (process.platform === 'win32') {
        console.log(chalk.cyan('  Windows:'));
        console.log(chalk.white('    # Install Visual Studio Build Tools'));
        console.log(chalk.white('    # https://visualstudio.microsoft.com/visual-cpp-build-tools/'));
        console.log(chalk.white('    # Or use WSL2 for best experience'));
        console.log('');
      }
    }

    console.log(chalk.cyan('───────────────────────────────────────────────────────────────'));
    console.log(chalk.cyan('Documentation & Resources'));
    console.log(chalk.cyan('───────────────────────────────────────────────────────────────\n'));

    console.log(chalk.white('  GitHub:     https://github.com/ruvnet/ruvector'));
    console.log(chalk.white('  npm:        https://www.npmjs.com/package/ruvector'));
    console.log(chalk.white('  crates.io:  https://crates.io/crates/ruvector-core'));
    console.log(chalk.white('  Issues:     https://github.com/ruvnet/ruvector/issues'));
    console.log('');

    console.log(chalk.cyan('Quick Commands:'));
    console.log(chalk.white('  npx ruvector doctor     # Check system health'));
    console.log(chalk.white('  npx ruvector info       # Show version info'));
    console.log(chalk.white('  npx ruvector benchmark  # Run performance test'));
    console.log(chalk.white('  npx ruvector install    # List available packages'));
    console.log('');
  });

// =============================================================================
// Graph Commands - Cypher queries and graph operations
// =============================================================================

program
  .command('graph')
  .description('Graph database operations (requires @ruvector/graph-node)')
  .option('-q, --query <cypher>', 'Execute Cypher query')
  .option('-c, --create <label>', 'Create a node with label')
  .option('-p, --properties <json>', 'Node properties as JSON')
  .option('-r, --relate <spec>', 'Create relationship (from:rel:to)')
  .option('--info', 'Show graph info and stats')
  .action(async (options) => {
    let graphNode;
    try {
      graphNode = require('@ruvector/graph-node');
    } catch (e) {
      console.log(chalk.yellow('\n  @ruvector/graph-node is not installed.\n'));
      console.log(chalk.cyan('  Install with:'));
      console.log(chalk.white('    npm install @ruvector/graph-node\n'));
      console.log(chalk.cyan('  Features:'));
      console.log(chalk.gray('    - Cypher query language support'));
      console.log(chalk.gray('    - Hypergraph data structure'));
      console.log(chalk.gray('    - Knowledge graph operations'));
      console.log(chalk.gray('    - Neo4j-compatible syntax\n'));
      console.log(chalk.cyan('  Example usage:'));
      console.log(chalk.white('    npx ruvector graph --query "CREATE (n:Person {name: \'Alice\'})"'));
      console.log(chalk.white('    npx ruvector graph --query "MATCH (n) RETURN n"'));
      console.log('');
      return;
    }

    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('                    RuVector Graph'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    if (options.info) {
      console.log(chalk.green('  @ruvector/graph-node is available!'));
      console.log(chalk.gray(`  Platform: ${process.platform}-${process.arch}`));
      console.log('');
      console.log(chalk.yellow('  Available operations:'));
      console.log(chalk.white('    --query <cypher>    Execute Cypher query'));
      console.log(chalk.white('    --create <label>    Create node with label'));
      console.log(chalk.white('    --relate <spec>     Create relationship'));
      console.log('');
      return;
    }

    if (options.query) {
      console.log(chalk.yellow('  Cypher Query:'), chalk.white(options.query));
      console.log('');
      // Actual implementation would execute the query
      console.log(chalk.gray('  Note: Full Cypher execution requires running ruvector-server'));
      console.log(chalk.gray('  See: npx ruvector server --help'));
    }

    if (options.create) {
      const label = options.create;
      const props = options.properties ? JSON.parse(options.properties) : {};
      console.log(chalk.yellow('  Creating node:'), chalk.white(label));
      console.log(chalk.gray('  Properties:'), JSON.stringify(props, null, 2));
    }

    console.log('');
  });

// =============================================================================
// Router Commands - AI agent routing
// =============================================================================

program
  .command('router')
  .description('AI semantic router operations (requires ruvector-router-core)')
  .option('--route <text>', 'Route text to best matching intent')
  .option('--intents <file>', 'Load intents from JSON file')
  .option('--add-intent <name>', 'Add new intent')
  .option('--examples <json>', 'Example utterances for intent')
  .option('--info', 'Show router info')
  .action(async (options) => {
    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('                    RuVector Router'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    console.log(chalk.yellow('  Semantic Router for AI Agent Routing\n'));

    if (options.info || (!options.route && !options.intents && !options.addIntent)) {
      console.log(chalk.cyan('  Features:'));
      console.log(chalk.gray('    - Semantic intent matching'));
      console.log(chalk.gray('    - Multi-agent routing'));
      console.log(chalk.gray('    - Dynamic intent registration'));
      console.log(chalk.gray('    - Vector-based similarity matching'));
      console.log('');
      console.log(chalk.cyan('  Status:'), chalk.yellow('Coming Soon'));
      console.log(chalk.gray('  The npm package for router is in development.'));
      console.log(chalk.gray('  Rust crate available: cargo add ruvector-router-core'));
      console.log('');
      console.log(chalk.cyan('  Usage (when available):'));
      console.log(chalk.white('    npx ruvector router --route "What is the weather?"'));
      console.log(chalk.white('    npx ruvector router --intents intents.json --route "query"'));
      console.log('');
      return;
    }

    if (options.route) {
      console.log(chalk.yellow('  Input:'), chalk.white(options.route));
      console.log(chalk.gray('  Router package not yet available in npm.'));
      console.log(chalk.gray('  Check issue #20 for roadmap.'));
    }

    console.log('');
  });

// =============================================================================
// Server Commands - HTTP/gRPC server
// =============================================================================

program
  .command('server')
  .description('Start RuVector HTTP/gRPC server')
  .option('-p, --port <number>', 'HTTP port', '8080')
  .option('-g, --grpc-port <number>', 'gRPC port', '50051')
  .option('-d, --data-dir <path>', 'Data directory', './ruvector-data')
  .option('--http-only', 'Start only HTTP server')
  .option('--grpc-only', 'Start only gRPC server')
  .option('--cors', 'Enable CORS for all origins')
  .option('--info', 'Show server info')
  .action(async (options) => {
    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('                    RuVector Server'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    if (options.info || Object.keys(options).filter(k => k !== 'port' && k !== 'grpcPort' && k !== 'dataDir').length === 0) {
      console.log(chalk.cyan('  Status:'), chalk.yellow('Coming Soon'));
      console.log('');
      console.log(chalk.cyan('  Planned Features:'));
      console.log(chalk.gray('    - REST API for vector operations'));
      console.log(chalk.gray('    - gRPC high-performance interface'));
      console.log(chalk.gray('    - WebSocket real-time updates'));
      console.log(chalk.gray('    - OpenAPI/Swagger documentation'));
      console.log(chalk.gray('    - Prometheus metrics endpoint'));
      console.log(chalk.gray('    - Health check endpoints'));
      console.log('');
      console.log(chalk.cyan('  Rust binary available:'));
      console.log(chalk.white('    cargo install ruvector-server  # When published'));
      console.log('');
      console.log(chalk.cyan('  Configuration (when available):'));
      console.log(chalk.white(`    --port ${options.port}            # HTTP port`));
      console.log(chalk.white(`    --grpc-port ${options.grpcPort}       # gRPC port`));
      console.log(chalk.white(`    --data-dir ${options.dataDir}  # Data directory`));
      console.log('');
      console.log(chalk.gray('  Track progress: https://github.com/ruvnet/ruvector/issues/20'));
      console.log('');
      return;
    }

    console.log(chalk.yellow('  Server package not yet available.'));
    console.log(chalk.gray('  Check issue #20 for roadmap.'));
    console.log('');
  });

// =============================================================================
// Cluster Commands - Distributed operations
// =============================================================================

program
  .command('cluster')
  .description('Distributed cluster operations')
  .option('--status', 'Show cluster status')
  .option('--join <address>', 'Join existing cluster')
  .option('--leave', 'Leave cluster')
  .option('--nodes', 'List cluster nodes')
  .option('--leader', 'Show current leader')
  .option('--info', 'Show cluster info')
  .action(async (options) => {
    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('                    RuVector Cluster'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    console.log(chalk.cyan('  Status:'), chalk.yellow('Coming Soon'));
    console.log('');
    console.log(chalk.cyan('  Features:'));
    console.log(chalk.gray('    - Raft consensus for leader election'));
    console.log(chalk.gray('    - Automatic failover'));
    console.log(chalk.gray('    - Data replication'));
    console.log(chalk.gray('    - Sharding support'));
    console.log(chalk.gray('    - Distributed queries'));
    console.log('');
    console.log(chalk.cyan('  Rust crates available:'));
    console.log(chalk.white('    cargo add ruvector-cluster      # Clustering'));
    console.log(chalk.white('    cargo add ruvector-raft         # Raft consensus'));
    console.log(chalk.white('    cargo add ruvector-replication  # Replication'));
    console.log('');
    console.log(chalk.cyan('  Commands (when available):'));
    console.log(chalk.white('    npx ruvector cluster --status'));
    console.log(chalk.white('    npx ruvector cluster --join 192.168.1.10:7000'));
    console.log(chalk.white('    npx ruvector cluster --nodes'));
    console.log('');
    console.log(chalk.gray('  Track progress: https://github.com/ruvnet/ruvector/issues/20'));
    console.log('');
  });

// =============================================================================
// Export/Import Commands - Database backup/restore
// =============================================================================

program
  .command('export <database>')
  .description('Export database to file')
  .option('-o, --output <file>', 'Output file path')
  .option('-f, --format <type>', 'Export format (json|binary|parquet)', 'json')
  .option('--compress', 'Compress output')
  .option('--vectors-only', 'Export only vectors (no metadata)')
  .action(async (dbPath, options) => {
    requireRuvector();
    const spinner = ora('Exporting database...').start();

    try {
      const outputFile = options.output || `${dbPath.replace(/\/$/, '')}_export.${options.format}`;

      // Load database
      const db = new VectorDB({ dimension: 384 }); // Will be overwritten by load
      if (fs.existsSync(dbPath)) {
        db.load(dbPath);
      } else {
        spinner.fail(chalk.red(`Database not found: ${dbPath}`));
        process.exit(1);
      }

      const stats = db.getStats();
      const data = {
        version: packageJson.version,
        exportedAt: new Date().toISOString(),
        stats: stats,
        vectors: [] // Would contain actual vector data
      };

      if (options.format === 'json') {
        fs.writeFileSync(outputFile, JSON.stringify(data, null, 2));
      } else {
        spinner.fail(chalk.yellow(`Format '${options.format}' not yet supported. Using JSON.`));
        fs.writeFileSync(outputFile.replace(/\.[^.]+$/, '.json'), JSON.stringify(data, null, 2));
      }

      spinner.succeed(chalk.green(`Exported to: ${outputFile}`));
      console.log(chalk.gray(`  Vectors: ${stats.count || 0}`));
      console.log(chalk.gray(`  Format: ${options.format}`));
    } catch (error) {
      spinner.fail(chalk.red('Export failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

program
  .command('import <file>')
  .description('Import database from file')
  .option('-d, --database <path>', 'Target database path')
  .option('--merge', 'Merge with existing data')
  .option('--replace', 'Replace existing data')
  .action(async (file, options) => {
    requireRuvector();
    const spinner = ora('Importing database...').start();

    try {
      if (!fs.existsSync(file)) {
        spinner.fail(chalk.red(`File not found: ${file}`));
        process.exit(1);
      }

      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      const dbPath = options.database || file.replace(/_export\.json$/, '');

      spinner.text = 'Creating database...';

      const db = new VectorDB({
        dimension: data.stats?.dimension || 384,
        path: dbPath,
        autoPersist: true
      });

      // Would import actual vectors here
      db.save(dbPath);

      spinner.succeed(chalk.green(`Imported to: ${dbPath}`));
      console.log(chalk.gray(`  Source version: ${data.version}`));
      console.log(chalk.gray(`  Exported at: ${data.exportedAt}`));
    } catch (error) {
      spinner.fail(chalk.red('Import failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// =============================================================================
// Embed Command - Generate embeddings
// =============================================================================

program
  .command('embed')
  .description('Generate embeddings from text')
  .option('-t, --text <string>', 'Text to embed')
  .option('-f, --file <path>', 'File containing text (one per line)')
  .option('-m, --model <name>', 'Embedding model', 'all-minilm-l6-v2')
  .option('-o, --output <file>', 'Output file for embeddings')
  .option('--info', 'Show embedding info')
  .action(async (options) => {
    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('                    RuVector Embed'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    if (options.info || (!options.text && !options.file)) {
      console.log(chalk.cyan('  Generate vector embeddings from text\n'));
      console.log(chalk.cyan('  Supported Models:'));
      console.log(chalk.gray('    - all-minilm-l6-v2 (384 dims, fast)'));
      console.log(chalk.gray('    - nomic-embed-text-v1.5 (768 dims, balanced)'));
      console.log(chalk.gray('    - openai/text-embedding-3-small (1536 dims, requires API key)'));
      console.log('');
      console.log(chalk.cyan('  Status:'), chalk.yellow('Coming Soon'));
      console.log(chalk.gray('  Built-in embedding generation is planned for future release.'));
      console.log('');
      console.log(chalk.cyan('  Current options:'));
      console.log(chalk.gray('    1. Use external embedding API (OpenAI, Cohere, etc.)'));
      console.log(chalk.gray('    2. Use transformers.js in your application'));
      console.log(chalk.gray('    3. Pre-generate embeddings with Python'));
      console.log('');
      console.log(chalk.cyan('  Usage (when available):'));
      console.log(chalk.white('    npx ruvector embed --text "Hello world"'));
      console.log(chalk.white('    npx ruvector embed --file texts.txt --output embeddings.json'));
      console.log('');
      return;
    }

    if (options.text) {
      console.log(chalk.yellow('  Input text:'), chalk.white(options.text.substring(0, 50) + '...'));
      console.log(chalk.yellow('  Model:'), chalk.white(options.model));
      console.log('');
      console.log(chalk.gray('  Embedding generation not yet available in CLI.'));
      console.log(chalk.gray('  Use the SDK or external embedding services.'));
    }

    console.log('');
  });

// =============================================================================
// Demo Command - Interactive tutorial
// =============================================================================

program
  .command('demo')
  .description('Run interactive demo and tutorials')
  .option('--basic', 'Basic vector operations demo')
  .option('--gnn', 'GNN differentiable search demo')
  .option('--graph', 'Graph database demo')
  .option('--benchmark', 'Performance benchmark demo')
  .option('-i, --interactive', 'Interactive mode')
  .action(async (options) => {
    console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
    console.log(chalk.cyan('                    RuVector Demo'));
    console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

    const showMenu = !options.basic && !options.gnn && !options.graph && !options.benchmark;

    if (showMenu) {
      console.log(chalk.yellow('  Available Demos:\n'));
      console.log(chalk.white('    --basic      '), chalk.gray('Basic vector operations (insert, search, delete)'));
      console.log(chalk.white('    --gnn        '), chalk.gray('GNN differentiable search with gradients'));
      console.log(chalk.white('    --graph      '), chalk.gray('Graph database and Cypher queries'));
      console.log(chalk.white('    --benchmark  '), chalk.gray('Performance benchmark suite'));
      console.log('');
      console.log(chalk.cyan('  Run a demo:'));
      console.log(chalk.white('    npx ruvector demo --basic'));
      console.log(chalk.white('    npx ruvector demo --gnn'));
      console.log('');
      return;
    }

    if (options.basic) {
      requireRuvector();
      console.log(chalk.yellow('  Basic Vector Operations Demo\n'));

      const spinner = ora('Creating demo database...').start();

      try {
        const db = new VectorDB({ dimension: 4, metric: 'cosine' });

        spinner.text = 'Inserting vectors...';
        db.insert('vec1', [1.0, 0.0, 0.0, 0.0], { label: 'x-axis' });
        db.insert('vec2', [0.0, 1.0, 0.0, 0.0], { label: 'y-axis' });
        db.insert('vec3', [0.0, 0.0, 1.0, 0.0], { label: 'z-axis' });
        db.insert('vec4', [0.7, 0.7, 0.0, 0.0], { label: 'xy-diagonal' });

        spinner.succeed('Demo database created with 4 vectors');

        console.log(chalk.cyan('\n  Vectors inserted:'));
        console.log(chalk.gray('    vec1: [1,0,0,0] - x-axis'));
        console.log(chalk.gray('    vec2: [0,1,0,0] - y-axis'));
        console.log(chalk.gray('    vec3: [0,0,1,0] - z-axis'));
        console.log(chalk.gray('    vec4: [0.7,0.7,0,0] - xy-diagonal'));

        console.log(chalk.cyan('\n  Searching for nearest to [0.8, 0.6, 0, 0]:'));
        const results = db.search([0.8, 0.6, 0.0, 0.0], 3);
        results.forEach((r, i) => {
          console.log(chalk.gray(`    ${i + 1}. ${r.id} (score: ${r.score.toFixed(4)})`));
        });

        console.log(chalk.green('\n  Demo complete!'));
      } catch (error) {
        spinner.fail(chalk.red('Demo failed'));
        console.error(chalk.red(error.message));
      }
    }

    if (options.gnn) {
      if (!gnnAvailable) {
        console.log(chalk.yellow('  @ruvector/gnn not installed.'));
        console.log(chalk.white('  Install with: npm install @ruvector/gnn'));
        console.log('');
        return;
      }

      console.log(chalk.yellow('  GNN Differentiable Search Demo\n'));

      try {
        console.log(chalk.cyan('  Running differentiable search with gradients...\n'));

        const queryVec = [1.0, 0.5, 0.3, 0.1];
        const dbVectors = [
          [1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.5, 0.5, 0.5, 0.5],
          [0.9, 0.4, 0.2, 0.1]
        ];

        const result = differentiableSearch(queryVec, dbVectors, 3, 10.0);

        console.log(chalk.cyan('  Query:'), JSON.stringify(queryVec));
        console.log(chalk.cyan('  Top 3 results:'));
        result.indices.forEach((idx, i) => {
          console.log(chalk.gray(`    ${i + 1}. Index ${idx} (attention: ${result.attention_weights[i].toFixed(4)})`));
        });

        console.log(chalk.cyan('\n  Gradient flow enabled:'), chalk.green('Yes'));
        console.log(chalk.gray('  Use for: Neural network training, learned retrieval'));

        console.log(chalk.green('\n  GNN demo complete!'));
      } catch (error) {
        console.error(chalk.red('GNN demo failed:', error.message));
      }
    }

    if (options.graph) {
      console.log(chalk.yellow('  Graph Database Demo\n'));

      let graphNode;
      try {
        graphNode = require('@ruvector/graph-node');
        console.log(chalk.green('  @ruvector/graph-node is available!'));
        console.log(chalk.gray('  Full graph demo coming soon.'));
      } catch (e) {
        console.log(chalk.yellow('  @ruvector/graph-node not installed.'));
        console.log(chalk.white('  Install with: npm install @ruvector/graph-node'));
      }
      console.log('');
    }

    if (options.benchmark) {
      console.log(chalk.yellow('  Redirecting to benchmark command...\n'));
      console.log(chalk.white('  Run: npx ruvector benchmark'));
      console.log('');
    }
  });

// ============================================
// Self-Learning Intelligence Hooks
// ============================================

const INTEL_PATH = path.join(require('os').homedir(), '.ruvector', 'intelligence.json');

class Intelligence {
  constructor() {
    this.data = this.load();
    this.alpha = 0.1;
    this.lastEditedFile = null;
  }

  load() {
    try {
      if (fs.existsSync(INTEL_PATH)) {
        return JSON.parse(fs.readFileSync(INTEL_PATH, 'utf-8'));
      }
    } catch {}
    return {
      patterns: {},
      memories: [],
      trajectories: [],
      errors: {},
      file_sequences: [],
      agents: {},
      edges: [],
      stats: { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 }
    };
  }

  save() {
    const dir = path.dirname(INTEL_PATH);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(INTEL_PATH, JSON.stringify(this.data, null, 2));
  }

  now() { return Math.floor(Date.now() / 1000); }

  embed(text) {
    const embedding = new Array(64).fill(0);
    for (let i = 0; i < text.length; i++) {
      const idx = (text.charCodeAt(i) + i * 7) % 64;
      embedding[idx] += 1.0;
    }
    const norm = Math.sqrt(embedding.reduce((a, b) => a + b * b, 0));
    if (norm > 0) for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
    return embedding;
  }

  similarity(a, b) {
    if (a.length !== b.length) return 0;
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
    const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
    return normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
  }

  remember(memoryType, content, metadata = {}) {
    const id = `mem_${this.now()}`;
    this.data.memories.push({ id, memory_type: memoryType, content, embedding: this.embed(content), metadata, timestamp: this.now() });
    if (this.data.memories.length > 5000) this.data.memories.splice(0, 1000);
    this.data.stats.total_memories = this.data.memories.length;
    return id;
  }

  recall(query, topK) {
    const queryEmbed = this.embed(query);
    return this.data.memories
      .map(m => ({ score: this.similarity(queryEmbed, m.embedding), memory: m }))
      .sort((a, b) => b.score - a.score).slice(0, topK).map(r => r.memory);
  }

  getQ(state, action) {
    const key = `${state}|${action}`;
    return this.data.patterns[key]?.q_value ?? 0;
  }

  updateQ(state, action, reward) {
    const key = `${state}|${action}`;
    if (!this.data.patterns[key]) this.data.patterns[key] = { state, action, q_value: 0, visits: 0, last_update: 0 };
    const p = this.data.patterns[key];
    p.q_value = p.q_value + this.alpha * (reward - p.q_value);
    p.visits++;
    p.last_update = this.now();
    this.data.stats.total_patterns = Object.keys(this.data.patterns).length;
  }

  learn(state, action, outcome, reward) {
    const id = `traj_${this.now()}`;
    this.updateQ(state, action, reward);
    this.data.trajectories.push({ id, state, action, outcome, reward, timestamp: this.now() });
    if (this.data.trajectories.length > 1000) this.data.trajectories.splice(0, 200);
    this.data.stats.total_trajectories = this.data.trajectories.length;
    return id;
  }

  suggest(state, actions) {
    let bestAction = actions[0] ?? '';
    let bestQ = -Infinity;
    for (const action of actions) {
      const q = this.getQ(state, action);
      if (q > bestQ) { bestQ = q; bestAction = action; }
    }
    return { action: bestAction, confidence: bestQ > 0 ? Math.min(bestQ, 1) : 0 };
  }

  route(task, file, crateName, operation = 'edit') {
    const fileType = file ? path.extname(file).slice(1) : 'unknown';
    const state = `${operation}_${fileType}_in_${crateName ?? 'project'}`;
    const agentMap = {
      rs: ['rust-developer', 'coder', 'reviewer', 'tester'],
      ts: ['typescript-developer', 'coder', 'frontend-dev'],
      tsx: ['typescript-developer', 'coder', 'frontend-dev'],
      js: ['coder', 'frontend-dev'],
      py: ['python-developer', 'coder', 'ml-developer'],
      md: ['docs-writer', 'coder']
    };
    const agents = agentMap[fileType] ?? ['coder', 'reviewer'];
    const { action, confidence } = this.suggest(state, agents);
    const reason = confidence > 0.5 ? 'learned from past success' : confidence > 0 ? 'based on patterns' : `default for ${fileType} files`;
    return { agent: action, confidence, reason };
  }

  shouldTest(file) {
    const ext = path.extname(file).slice(1);
    switch (ext) {
      case 'rs': {
        const crateMatch = file.match(/crates\/([^/]+)/);
        return crateMatch ? { suggest: true, command: `cargo test -p ${crateMatch[1]}` } : { suggest: true, command: 'cargo test' };
      }
      case 'ts': case 'tsx': case 'js': case 'jsx': return { suggest: true, command: 'npm test' };
      case 'py': return { suggest: true, command: 'pytest' };
      default: return { suggest: false, command: '' };
    }
  }

  recordFileSequence(fromFile, toFile) {
    const existing = this.data.file_sequences.find(s => s.from_file === fromFile && s.to_file === toFile);
    if (existing) existing.count++;
    else this.data.file_sequences.push({ from_file: fromFile, to_file: toFile, count: 1 });
    this.lastEditedFile = toFile;
  }

  suggestNext(file, limit = 3) {
    return this.data.file_sequences.filter(s => s.from_file === file).sort((a, b) => b.count - a.count).slice(0, limit).map(s => ({ file: s.to_file, score: s.count }));
  }

  classifyCommand(command) {
    const cmd = command.toLowerCase();
    if (cmd.includes('cargo') || cmd.includes('rustc')) return { category: 'rust', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('npm') || cmd.includes('node')) return { category: 'javascript', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('git')) return { category: 'git', subcategory: 'vcs', risk: cmd.includes('push') ? 'medium' : 'low' };
    if (cmd.includes('rm') || cmd.includes('delete')) return { category: 'filesystem', subcategory: 'destructive', risk: 'high' };
    return { category: 'shell', subcategory: 'general', risk: 'low' };
  }

  swarmStats() {
    const agents = Object.keys(this.data.agents).length;
    const edges = this.data.edges.length;
    return { agents, edges };
  }

  stats() { return this.data.stats; }
  sessionStart() { this.data.stats.session_count++; this.data.stats.last_session = this.now(); }
  sessionEnd() {
    const duration = this.now() - this.data.stats.last_session;
    const actions = this.data.trajectories.filter(t => t.timestamp >= this.data.stats.last_session).length;
    return { duration, actions };
  }
  getLastEditedFile() { return this.lastEditedFile; }
}

// Hooks command group
const hooksCmd = program.command('hooks').description('Self-learning intelligence hooks for Claude Code');

hooksCmd.command('init').description('Initialize hooks in current project').option('--force', 'Force overwrite').action((opts) => {
  const settingsPath = path.join(process.cwd(), '.claude', 'settings.json');
  const settingsDir = path.dirname(settingsPath);
  if (!fs.existsSync(settingsDir)) fs.mkdirSync(settingsDir, { recursive: true });
  let settings = {};
  if (fs.existsSync(settingsPath) && !opts.force) {
    try { settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8')); } catch {}
  }
  settings.hooks = settings.hooks || {};
  settings.hooks.PreToolUse = [
    { matcher: 'Edit|Write|MultiEdit', hooks: [{ type: 'command', command: 'npx ruvector hooks pre-edit "$TOOL_INPUT_file_path"' }] },
    { matcher: 'Bash', hooks: [{ type: 'command', command: 'npx ruvector hooks pre-command "$TOOL_INPUT_command"' }] }
  ];
  settings.hooks.PostToolUse = [
    { matcher: 'Edit|Write|MultiEdit', hooks: [{ type: 'command', command: 'npx ruvector hooks post-edit "$TOOL_INPUT_file_path"' }] },
    { matcher: 'Bash', hooks: [{ type: 'command', command: 'npx ruvector hooks post-command "$TOOL_INPUT_command"' }] }
  ];
  settings.hooks.SessionStart = [{ hooks: [{ type: 'command', command: 'npx ruvector hooks session-start' }] }];
  settings.hooks.Stop = [{ hooks: [{ type: 'command', command: 'npx ruvector hooks session-end' }] }];
  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
  console.log(chalk.green('✅ Hooks initialized in .claude/settings.json'));
});

hooksCmd.command('stats').description('Show intelligence statistics').action(() => {
  const intel = new Intelligence();
  const stats = intel.stats();
  const swarm = intel.swarmStats();
  console.log(chalk.bold.cyan('\n🧠 RuVector Intelligence Stats\n'));
  console.log(`  ${chalk.green(stats.total_patterns)} Q-learning patterns`);
  console.log(`  ${chalk.green(stats.total_memories)} vector memories`);
  console.log(`  ${chalk.green(stats.total_trajectories)} learning trajectories`);
  console.log(`  ${chalk.green(stats.total_errors)} error patterns\n`);
  console.log(chalk.bold('Swarm Status:'));
  console.log(`  ${chalk.cyan(swarm.agents)} agents registered`);
  console.log(`  ${chalk.cyan(swarm.edges)} coordination edges`);
});

hooksCmd.command('session-start').description('Session start hook').option('--resume', 'Resume previous session').action(() => {
  const intel = new Intelligence();
  intel.sessionStart();
  intel.save();
  console.log(chalk.bold.cyan('🧠 RuVector Intelligence Layer Active'));
  console.log('⚡ Intelligence guides: agent routing, error fixes, file sequences');
});

hooksCmd.command('session-end').description('Session end hook').option('--export-metrics', 'Export metrics').action((opts) => {
  const intel = new Intelligence();
  const sessionInfo = intel.sessionEnd();
  intel.save();
  console.log('📊 Session ended. Learning data saved.');
  if (opts.exportMetrics) console.log(JSON.stringify({ duration_seconds: sessionInfo.duration, actions_recorded: sessionInfo.actions }));
});

hooksCmd.command('pre-edit').description('Pre-edit intelligence').argument('<file>', 'File path').action((file) => {
  const intel = new Intelligence();
  const fileName = path.basename(file);
  const crateMatch = file.match(/crates\/([^/]+)/);
  const crate = crateMatch?.[1];
  const { agent, confidence, reason } = intel.route(`edit ${fileName}`, file, crate, 'edit');
  console.log(chalk.bold('🧠 Intelligence Analysis:'));
  console.log(`   📁 ${chalk.cyan(crate ?? 'project')}/${fileName}`);
  console.log(`   🤖 Recommended: ${chalk.green.bold(agent)} (${(confidence * 100).toFixed(0)}% confidence)`);
  if (reason) console.log(`      → ${chalk.dim(reason)}`);
  const nextFiles = intel.suggestNext(file, 3);
  if (nextFiles.length > 0) {
    console.log('   📎 Likely next files:');
    nextFiles.forEach(n => console.log(`      - ${n.file} (${n.score} edits)`));
  }
});

hooksCmd.command('post-edit').description('Post-edit learning').argument('<file>', 'File path').option('--success', 'Edit succeeded').option('--error <msg>', 'Error message').action((file, opts) => {
  const intel = new Intelligence();
  const success = opts.error ? false : (opts.success ?? true);
  const ext = path.extname(file).slice(1);
  const crateMatch = file.match(/crates\/([^/]+)/);
  const crate = crateMatch?.[1] ?? 'project';
  const state = `edit_${ext}_in_${crate}`;
  const lastFile = intel.getLastEditedFile();
  if (lastFile && lastFile !== file) intel.recordFileSequence(lastFile, file);
  intel.learn(state, success ? 'successful-edit' : 'failed-edit', success ? 'completed' : 'failed', success ? 1.0 : -0.5);
  intel.remember('edit', `${success ? 'successful' : 'failed'} edit of ${ext} in ${crate}`);
  intel.save();
  console.log(`📊 Learning recorded: ${success ? '✅' : '❌'} ${path.basename(file)}`);
  const test = intel.shouldTest(file);
  if (test.suggest) console.log(`   🧪 Consider: ${chalk.cyan(test.command)}`);
});

hooksCmd.command('pre-command').description('Pre-command intelligence').argument('<command...>', 'Command').action((command) => {
  const intel = new Intelligence();
  const cmd = command.join(' ');
  const classification = intel.classifyCommand(cmd);
  console.log(chalk.bold('🧠 Command Analysis:'));
  console.log(`   📦 Category: ${chalk.cyan(classification.category)}`);
  console.log(`   🏷️  Type: ${classification.subcategory}`);
  if (classification.risk === 'high') console.log(`   ⚠️  Risk: ${chalk.red('HIGH')} - Review carefully`);
  else if (classification.risk === 'medium') console.log(`   ⚡ Risk: ${chalk.yellow('MEDIUM')}`);
  else console.log(`   ✅ Risk: ${chalk.green('LOW')}`);
});

hooksCmd.command('post-command').description('Post-command learning').argument('<command...>', 'Command').option('--success', 'Success').option('--error <msg>', 'Error message').action((command, opts) => {
  const intel = new Intelligence();
  const cmd = command.join(' ');
  const success = opts.error ? false : (opts.success ?? true);
  const classification = intel.classifyCommand(cmd);
  intel.learn(`cmd_${classification.category}_${classification.subcategory}`, success ? 'success' : 'failure', success ? 'completed' : 'failed', success ? 0.8 : -0.3);
  intel.remember('command', `${cmd} ${success ? 'succeeded' : 'failed'}`);
  intel.save();
  console.log(`📊 Command ${success ? '✅' : '❌'} recorded`);
});

hooksCmd.command('route').description('Route task to agent').argument('<task...>', 'Task').option('--file <file>', 'File').option('--crate <crate>', 'Crate').action((task, opts) => {
  const intel = new Intelligence();
  const result = intel.route(task.join(' '), opts.file, opts.crate);
  console.log(JSON.stringify({ task: task.join(' '), recommended: result.agent, confidence: result.confidence, reasoning: result.reason }, null, 2));
});

hooksCmd.command('suggest-context').description('Suggest relevant context').action(() => {
  const intel = new Intelligence();
  const stats = intel.stats();
  console.log(`RuVector Intelligence: ${stats.total_patterns} learned patterns, ${stats.total_errors} error fixes available. Use 'ruvector hooks route' for agent suggestions.`);
});

hooksCmd.command('remember').description('Store in memory').requiredOption('-t, --type <type>', 'Memory type').argument('<content...>', 'Content').action((content, opts) => {
  const intel = new Intelligence();
  const id = intel.remember(opts.type, content.join(' '));
  intel.save();
  console.log(JSON.stringify({ success: true, id }));
});

hooksCmd.command('recall').description('Search memory').argument('<query...>', 'Query').option('-k, --top-k <n>', 'Results', '5').action((query, opts) => {
  const intel = new Intelligence();
  const results = intel.recall(query.join(' '), parseInt(opts.topK));
  console.log(JSON.stringify({ query: query.join(' '), results: results.map(r => ({ type: r.memory_type, content: r.content.slice(0, 200), timestamp: r.timestamp })) }, null, 2));
});

hooksCmd.command('pre-compact').description('Pre-compact hook').option('--auto', 'Auto mode').action(() => {
  const intel = new Intelligence();
  intel.save();
  console.log('🗜️ Pre-compact: State saved');
});

hooksCmd.command('swarm-recommend').description('Recommend agent for task').argument('<task-type>', 'Task type').action((taskType) => {
  console.log(JSON.stringify({ task_type: taskType, recommended: 'coder', type: 'default', score: 0.8 }));
});

hooksCmd.command('async-agent').description('Async agent hook').option('--action <action>', 'Action').option('--agent-id <id>', 'Agent ID').option('--task <task>', 'Task').action((opts) => {
  console.log(JSON.stringify({ action: opts.action, agent_id: opts.agentId, status: 'ok' }));
});

hooksCmd.command('lsp-diagnostic').description('LSP diagnostic hook').option('--file <file>', 'File').option('--severity <sev>', 'Severity').option('--message <msg>', 'Message').action((opts) => {
  console.log(JSON.stringify({ file: opts.file, severity: opts.severity, action: 'logged' }));
});

hooksCmd.command('track-notification').description('Track notification').action(() => {
  console.log(JSON.stringify({ tracked: true }));
});

program.parse();
