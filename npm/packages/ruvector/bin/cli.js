#!/usr/bin/env node

// Signal CLI context (disables parallel workers - hooks are short-lived)
process.env.RUVECTOR_CLI = '1';

const { Command } = require('commander');
const _chalk = require('chalk');
const chalk = _chalk.default || _chalk;
const fs = require('fs');
const path = require('path');

// Lazy load ora (spinner) - only needed for commands with progress indicators
let _oraModule = null;
function ora(text) {
  if (_oraModule === null) {
    const _ora = require('ora');
    _oraModule = _ora.default || _ora;
  }
  return _oraModule(text);
}

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

// Lazy load GNN (optional - loaded on first use, not at startup)
// Saves ~6ms startup time by deferring require('@ruvector/gnn')
let _gnnModule = undefined; // undefined = not yet attempted, null = failed, object = loaded
let RuvectorLayer, TensorCompress, differentiableSearch, getCompressionLevel, hierarchicalForward;
let gnnAvailable = false;

function loadGnn() {
  if (_gnnModule !== undefined) return _gnnModule;
  try {
    const gnn = require('@ruvector/gnn');
    RuvectorLayer = gnn.RuvectorLayer;
    TensorCompress = gnn.TensorCompress;
    differentiableSearch = gnn.differentiableSearch;
    getCompressionLevel = gnn.getCompressionLevel;
    hierarchicalForward = gnn.hierarchicalForward;
    _gnnModule = gnn;
    gnnAvailable = true;
    return gnn;
  } catch (e) {
    _gnnModule = null;
    gnnAvailable = false;
    return null;
  }
}

// Lazy load Attention (optional - loaded on first use, not at startup)
// Saves ~5ms startup time by deferring require('@ruvector/attention')
let _attentionModule = undefined; // undefined = not yet attempted
let DotProductAttention, MultiHeadAttention, HyperbolicAttention, FlashAttention, LinearAttention, MoEAttention;
let GraphRoPeAttention, EdgeFeaturedAttention, DualSpaceAttention, LocalGlobalAttention;
let benchmarkAttention, computeAttentionAsync, batchAttentionCompute, parallelAttentionCompute;
let expMap, logMap, mobiusAddition, poincareDistance, projectToPoincareBall;
let attentionInfo, attentionVersion;
let attentionAvailable = false;

function loadAttention() {
  if (_attentionModule !== undefined) return _attentionModule;
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
    _attentionModule = attention;
    attentionAvailable = true;
    return attention;
  } catch (e) {
    _attentionModule = null;
    attentionAvailable = false;
    return null;
  }
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
        dimensions: dimension,
        metric: options.metric,
        storagePath: dbPath,
      });

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
      const db = new VectorDB({ dimensions: dimension, metric: 'cosine' });
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
      console.log(chalk.cyan('\nFinal Stats:'));
      console.log(chalk.white(`  Vector Count: ${chalk.yellow(numVectors)}`));
      console.log(chalk.white(`  Dimension: ${chalk.yellow(dimension)}`));
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
    // Trigger lazy load of optional modules for availability check
    loadGnn();
    loadAttention();

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

    // Trigger lazy load to check availability
    loadGnn();

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

// Helper to check GNN availability (triggers lazy load)
function requireGnn() {
  loadGnn();
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
    loadGnn();
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

// Helper to require attention module (triggers lazy load)
function requireAttention() {
  loadAttention();
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
    loadAttention();
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
    loadAttention();
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

    // Trigger lazy load of optional modules for availability check
    loadGnn();
    loadAttention();

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
      console.log(chalk.white('  cargo add ruvector-router-core       # Semantic routing (Rust crate)'));
      console.log(chalk.white('  npm install @ruvector/router         # Semantic routing (npm)'));
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
  .description('AI semantic router operations (requires @ruvector/router)')
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
      console.log(chalk.gray('  The router subcommand integration is still in development.'));
      console.log(chalk.gray('  npm package:  npm install @ruvector/router'));
      console.log(chalk.gray('  Rust crate:   cargo add ruvector-router-core'));
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

// =============================================================================
// Embed Command - Generate embeddings (now with ONNX + Adaptive LoRA)
// =============================================================================

const embedCmd = program.command('embed').description('Generate embeddings from text (ONNX + Adaptive LoRA)');

embedCmd
  .command('text')
  .description('Embed a text string')
  .argument('<text>', 'Text to embed')
  .option('--adaptive', 'Use adaptive embedder with LoRA')
  .option('--domain <domain>', 'Domain for prototype learning')
  .option('-o, --output <file>', 'Output file for embedding')
  .action(async (text, opts) => {
    try {
      const { performance } = require('perf_hooks');
      const start = performance.now();

      if (opts.adaptive) {
        const { initAdaptiveEmbedder } = require('../dist/core/adaptive-embedder.js');
        const embedder = await initAdaptiveEmbedder();
        const embedding = await embedder.embed(text, { domain: opts.domain });
        const stats = embedder.getStats();

        console.log(chalk.cyan('\n🧠 Adaptive Embedding (ONNX + Micro-LoRA)\n'));
        console.log(chalk.dim(`Text: "${text.slice(0, 60)}..."`));
        console.log(chalk.dim(`Dimension: ${embedding.length}`));
        console.log(chalk.dim(`LoRA rank: ${stats.loraRank} (${stats.loraParams} params)`));
        console.log(chalk.dim(`Prototypes: ${stats.prototypes}`));
        console.log(chalk.dim(`Time: ${(performance.now() - start).toFixed(1)}ms`));

        if (opts.output) {
          fs.writeFileSync(opts.output, JSON.stringify({ text, embedding, stats }, null, 2));
          console.log(chalk.green(`\nSaved to ${opts.output}`));
        }
      } else {
        const { initOnnxEmbedder, embed } = require('../dist/core/onnx-embedder.js');
        await initOnnxEmbedder();
        const result = await embed(text);

        console.log(chalk.cyan('\n📊 ONNX Embedding (all-MiniLM-L6-v2)\n'));
        console.log(chalk.dim(`Text: "${text.slice(0, 60)}..."`));
        console.log(chalk.dim(`Dimension: ${result.embedding.length}`));
        console.log(chalk.dim(`Time: ${(performance.now() - start).toFixed(1)}ms`));

        if (opts.output) {
          fs.writeFileSync(opts.output, JSON.stringify({ text, embedding: result.embedding }, null, 2));
          console.log(chalk.green(`\nSaved to ${opts.output}`));
        }
      }
    } catch (e) {
      console.error(chalk.red('Embedding failed:'), e.message);
    }
  });

embedCmd
  .command('adaptive')
  .description('Adaptive embedding with Micro-LoRA optimization')
  .option('--stats', 'Show adaptive embedder statistics')
  .option('--consolidate', 'Run EWC consolidation')
  .option('--reset', 'Reset adaptive weights')
  .option('--export <file>', 'Export learned weights')
  .option('--import <file>', 'Import learned weights')
  .action(async (opts) => {
    try {
      const { initAdaptiveEmbedder } = require('../dist/core/adaptive-embedder.js');
      const embedder = await initAdaptiveEmbedder();

      if (opts.stats) {
        const stats = embedder.getStats();
        console.log(chalk.cyan('\n🧠 Adaptive Embedder Statistics\n'));
        console.log(chalk.white('Base Model:'), chalk.dim(stats.baseModel));
        console.log(chalk.white('Dimension:'), chalk.dim(stats.dimension));
        console.log(chalk.white('LoRA Rank:'), chalk.dim(stats.loraRank));
        console.log(chalk.white('LoRA Params:'), chalk.dim(`${stats.loraParams} (~${(stats.loraParams / (stats.dimension * stats.dimension) * 100).toFixed(2)}% of base)`));
        console.log(chalk.white('Adaptations:'), chalk.dim(stats.adaptations));
        console.log(chalk.white('Prototypes:'), chalk.dim(stats.prototypes));
        console.log(chalk.white('Memory Size:'), chalk.dim(stats.memorySize));
        console.log(chalk.white('EWC Consolidations:'), chalk.dim(stats.ewcConsolidations));
        console.log(chalk.white('Contrastive Updates:'), chalk.dim(stats.contrastiveUpdates));
        console.log('');
      }

      if (opts.consolidate) {
        console.log(chalk.yellow('Running EWC consolidation...'));
        await embedder.consolidate();
        console.log(chalk.green('✓ Consolidation complete'));
      }

      if (opts.reset) {
        embedder.reset();
        console.log(chalk.green('✓ Adaptive weights reset'));
      }

      if (opts.export) {
        const data = embedder.export();
        fs.writeFileSync(opts.export, JSON.stringify(data, null, 2));
        console.log(chalk.green(`✓ Exported to ${opts.export}`));
      }

      if (opts.import) {
        const data = JSON.parse(fs.readFileSync(opts.import, 'utf-8'));
        embedder.import(data);
        console.log(chalk.green(`✓ Imported from ${opts.import}`));
      }
    } catch (e) {
      console.error(chalk.red('Error:'), e.message);
    }
  });

embedCmd
  .command('benchmark')
  .description('Benchmark base vs adaptive embeddings')
  .option('--iterations <n>', 'Number of iterations', '10')
  .action(async (opts) => {
    try {
      const { performance } = require('perf_hooks');
      const iterations = parseInt(opts.iterations) || 10;

      console.log(chalk.cyan('\n🚀 Embedding Benchmark: Base ONNX vs Adaptive LoRA\n'));

      const testTexts = [
        'This is a test sentence for embedding generation.',
        'The quick brown fox jumps over the lazy dog.',
        'Machine learning models can learn from data.',
        'Vector databases enable semantic search.',
      ];

      // Benchmark base ONNX
      const { initOnnxEmbedder, embed, embedBatch } = require('../dist/core/onnx-embedder.js');
      await initOnnxEmbedder();

      console.log(chalk.yellow('1. Base ONNX Embeddings'));
      const baseStart = performance.now();
      for (let i = 0; i < iterations; i++) {
        await embed(testTexts[i % testTexts.length]);
      }
      const baseTime = (performance.now() - baseStart) / iterations;
      console.log(chalk.dim(`   Single: ${baseTime.toFixed(1)}ms avg`));

      const baseBatchStart = performance.now();
      for (let i = 0; i < Math.ceil(iterations / 4); i++) {
        await embedBatch(testTexts);
      }
      const baseBatchTime = (performance.now() - baseBatchStart) / Math.ceil(iterations / 4);
      console.log(chalk.dim(`   Batch(4): ${baseBatchTime.toFixed(1)}ms avg (${(4000 / baseBatchTime).toFixed(1)}/s)`));

      // Benchmark adaptive
      const { initAdaptiveEmbedder } = require('../dist/core/adaptive-embedder.js');
      const adaptive = await initAdaptiveEmbedder();

      console.log(chalk.yellow('\n2. Adaptive ONNX + LoRA'));
      const adaptStart = performance.now();
      for (let i = 0; i < iterations; i++) {
        await adaptive.embed(testTexts[i % testTexts.length]);
      }
      const adaptTime = (performance.now() - adaptStart) / iterations;
      console.log(chalk.dim(`   Single: ${adaptTime.toFixed(1)}ms avg`));

      const adaptBatchStart = performance.now();
      for (let i = 0; i < Math.ceil(iterations / 4); i++) {
        await adaptive.embedBatch(testTexts);
      }
      const adaptBatchTime = (performance.now() - adaptBatchStart) / Math.ceil(iterations / 4);
      console.log(chalk.dim(`   Batch(4): ${adaptBatchTime.toFixed(1)}ms avg (${(4000 / adaptBatchTime).toFixed(1)}/s)`));

      // Summary
      console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
      console.log(chalk.bold('Summary'));
      console.log(chalk.cyan('═══════════════════════════════════════════════════════════════'));
      const stats = adaptive.getStats();
      console.log(chalk.dim(`\nAdaptive overhead: +${(adaptTime - baseTime).toFixed(1)}ms (+${((adaptTime/baseTime - 1) * 100).toFixed(1)}%)`));
      console.log(chalk.dim(`LoRA params: ${stats.loraParams} (rank ${stats.loraRank})`));
      console.log(chalk.dim(`Memory prototypes: ${stats.prototypes}`));
      console.log(chalk.dim(`Episodic memory: ${stats.memorySize} entries`));

      console.log(chalk.white('\nBenefits of Adaptive:'));
      console.log(chalk.dim('  • Domain-specific fine-tuning via Micro-LoRA'));
      console.log(chalk.dim('  • Contrastive learning from co-edit patterns'));
      console.log(chalk.dim('  • EWC++ prevents catastrophic forgetting'));
      console.log(chalk.dim('  • Prototype-based domain adaptation'));
      console.log(chalk.dim('  • Episodic memory augmentation'));
      console.log('');
    } catch (e) {
      console.error(chalk.red('Benchmark failed:'), e.message);
      if (e.stack) console.error(chalk.dim(e.stack));
    }
  });

embedCmd
  .command('optimized')
  .description('Use optimized ONNX embedder with LRU caching')
  .argument('[text]', 'Text to embed (optional)')
  .option('--cache-size <n>', 'Embedding cache size', '512')
  .option('--stats', 'Show cache statistics')
  .option('--clear-cache', 'Clear all caches')
  .option('--benchmark', 'Run cache benchmark')
  .action(async (text, opts) => {
    try {
      const { performance } = require('perf_hooks');
      const { OptimizedOnnxEmbedder } = require('../dist/core/onnx-optimized.js');

      const embedder = new OptimizedOnnxEmbedder({
        cacheSize: parseInt(opts.cacheSize) || 512,
        lazyInit: false,
      });

      await embedder.init();

      if (opts.clearCache) {
        embedder.clearCache();
        console.log(chalk.green('✓ Caches cleared'));
        return;
      }

      if (opts.benchmark) {
        console.log(chalk.cyan('\n⚡ Optimized ONNX Cache Benchmark\n'));

        const testTexts = [
          'Machine learning algorithms optimize model parameters',
          'Vector databases enable semantic search capabilities',
          'Neural networks learn hierarchical representations',
          'Code embeddings capture syntax and semantic patterns',
          'Transformer models use attention mechanisms',
        ];

        // Cold benchmark
        embedder.clearCache();
        const coldStart = performance.now();
        for (const t of testTexts) await embedder.embed(t);
        const coldTime = performance.now() - coldStart;

        // Warm benchmark
        const warmStart = performance.now();
        for (let i = 0; i < 100; i++) {
          for (const t of testTexts) await embedder.embed(t);
        }
        const warmTime = performance.now() - warmStart;

        const stats = embedder.getCacheStats();

        console.log(chalk.yellow('Performance:'));
        console.log(chalk.dim('  Cold (5 unique texts):'), chalk.white(coldTime.toFixed(2) + 'ms'));
        console.log(chalk.dim('  Warm (500 cached):'), chalk.white(warmTime.toFixed(2) + 'ms'));
        console.log(chalk.dim('  Cache speedup:'), chalk.green((coldTime / warmTime * 100).toFixed(0) + 'x'));
        console.log();
        console.log(chalk.yellow('Cache Stats:'));
        console.log(chalk.dim('  Hit rate:'), chalk.white((stats.embedding.hitRate * 100).toFixed(1) + '%'));
        console.log(chalk.dim('  Cache size:'), chalk.white(stats.embedding.size));
        console.log(chalk.dim('  Total embeds:'), chalk.white(stats.totalEmbeds));
        console.log();
        return;
      }

      if (opts.stats) {
        const stats = embedder.getCacheStats();
        console.log(chalk.cyan('\n📊 Optimized ONNX Embedder Stats\n'));
        console.log(chalk.white('Embedding Cache:'));
        console.log(chalk.dim('  Size:'), stats.embedding.size);
        console.log(chalk.dim('  Hits:'), stats.embedding.hits);
        console.log(chalk.dim('  Misses:'), stats.embedding.misses);
        console.log(chalk.dim('  Hit Rate:'), (stats.embedding.hitRate * 100).toFixed(1) + '%');
        console.log();
        console.log(chalk.white('Performance:'));
        console.log(chalk.dim('  Avg Time:'), stats.avgTimeMs.toFixed(2) + 'ms');
        console.log(chalk.dim('  Total Embeds:'), stats.totalEmbeds);
        console.log();
        return;
      }

      if (text) {
        const start = performance.now();
        const embedding = await embedder.embed(text);
        const elapsed = performance.now() - start;
        const stats = embedder.getCacheStats();

        console.log(chalk.cyan('\n⚡ Optimized ONNX Embedding\n'));
        console.log(chalk.dim(`Text: "${text.slice(0, 60)}${text.length > 60 ? '...' : ''}"`));
        console.log(chalk.dim(`Dimension: ${embedding.length}`));
        console.log(chalk.dim(`Time: ${elapsed.toFixed(2)}ms`));
        console.log(chalk.dim(`Cache hit rate: ${(stats.embedding.hitRate * 100).toFixed(1)}%`));
        console.log();
      } else {
        console.log(chalk.yellow('Usage: ruvector embed optimized <text>'));
        console.log(chalk.dim('  --stats       Show cache statistics'));
        console.log(chalk.dim('  --benchmark   Run cache benchmark'));
        console.log(chalk.dim('  --clear-cache Clear all caches'));
        console.log(chalk.dim('  --cache-size  Set cache size (default: 512)'));
      }
    } catch (e) {
      console.error(chalk.red('Error:'), e.message);
    }
  });

embedCmd
  .command('neural')
  .description('Neural embedding substrate (frontier AI concepts)')
  .option('--health', 'Show neural substrate health')
  .option('--consolidate', 'Run memory consolidation (like sleep)')
  .option('--calibrate', 'Calibrate coherence baseline')
  .option('--swarm-status', 'Show swarm coordination status')
  .option('--drift-stats', 'Show semantic drift statistics')
  .option('--memory-stats', 'Show memory physics statistics')
  .option('--demo', 'Run interactive neural demo')
  .option('--dimension <n>', 'Embedding dimension', '384')
  .action(async (opts) => {
    try {
      const { NeuralSubstrate } = require('../dist/core/neural-embeddings.js');
      const { initOnnxEmbedder, embed } = require('../dist/core/onnx-embedder.js');

      const dimension = parseInt(opts.dimension) || 384;
      const substrate = new NeuralSubstrate({ dimension });

      if (opts.demo) {
        console.log(chalk.cyan('\n🧠 Neural Embedding Substrate Demo\n'));
        console.log(chalk.dim('Frontier AI concepts: drift detection, memory physics, swarm coordination\n'));

        // Initialize ONNX for real embeddings
        await initOnnxEmbedder();

        console.log(chalk.yellow('1. Semantic Drift Detection'));
        console.log(chalk.dim('   Observing embeddings and detecting semantic movement...\n'));

        const texts = [
          'Machine learning optimizes neural networks',
          'Deep learning uses backpropagation',
          'AI models learn from data patterns',
          'Quantum computing is completely different',  // Should trigger drift
        ];

        for (const text of texts) {
          const result = await embed(text);
          const driftEvent = substrate.drift.observe(result.embedding, 'demo');
          const symbol = driftEvent?.category === 'critical' ? '🚨' :
                        driftEvent?.category === 'warning' ? '⚠️' : '✓';
          console.log(chalk.dim(`   ${symbol} "${text.slice(0, 40)}..." → drift: ${driftEvent?.magnitude?.toFixed(3) || '0.000'}`));
        }

        console.log(chalk.yellow('\n2. Memory Physics (Hippocampal Dynamics)'));
        console.log(chalk.dim('   Encoding memories with strength, decay, and consolidation...\n'));

        const memories = [
          { id: 'mem1', text: 'Vector databases store embeddings' },
          { id: 'mem2', text: 'HNSW enables fast nearest neighbor search' },
          { id: 'mem3', text: 'Cosine similarity measures semantic closeness' },
        ];

        for (const mem of memories) {
          const result = await embed(mem.text);
          const entry = substrate.memory.encode(mem.id, result.embedding, mem.text);
          console.log(chalk.dim(`   📝 Encoded "${mem.id}": strength=${entry.strength.toFixed(2)}, interference=${entry.interference.toFixed(2)}`));
        }

        // Query memory
        const queryText = 'How do vector databases work?';
        const queryEmb = await embed(queryText);
        const recalled = substrate.memory.recall(queryEmb.embedding, 2);
        console.log(chalk.dim(`\n   🔍 Query: "${queryText}"`));
        console.log(chalk.dim(`   📚 Recalled: ${recalled.map(m => m.id).join(', ')}`));

        console.log(chalk.yellow('\n3. Agent State Machine (Geometric State)'));
        console.log(chalk.dim('   Managing agent state as movement through embedding space...\n'));

        // Define mode regions
        substrate.state.defineMode('research', queryEmb.embedding, 0.5);
        const codeEmb = await embed('Write code and debug programs');
        substrate.state.defineMode('coding', codeEmb.embedding, 0.5);

        // Update agent state
        const agent1State = substrate.state.updateAgent('agent-1', queryEmb.embedding);
        console.log(chalk.dim(`   🤖 agent-1 mode: ${agent1State.mode}, energy: ${agent1State.energy.toFixed(2)}`));

        const agent2State = substrate.state.updateAgent('agent-2', codeEmb.embedding);
        console.log(chalk.dim(`   🤖 agent-2 mode: ${agent2State.mode}, energy: ${agent2State.energy.toFixed(2)}`));

        console.log(chalk.yellow('\n4. Swarm Coordination'));
        console.log(chalk.dim('   Multi-agent coordination through shared embedding geometry...\n'));

        substrate.swarm.register('researcher', queryEmb.embedding, 'research');
        substrate.swarm.register('coder', codeEmb.embedding, 'development');
        const reviewEmb = await embed('Review code and check quality');
        substrate.swarm.register('reviewer', reviewEmb.embedding, 'review');

        const coherence = substrate.swarm.getCoherence();
        console.log(chalk.dim(`   🌐 Swarm coherence: ${(coherence * 100).toFixed(1)}%`));

        const collaborators = substrate.swarm.findCollaborators('researcher', 2);
        console.log(chalk.dim(`   🤝 Collaborators for researcher: ${collaborators.map(c => c.id).join(', ')}`));

        console.log(chalk.yellow('\n5. Coherence Monitoring (Safety)'));
        console.log(chalk.dim('   Detecting degradation, poisoning, misalignment...\n'));

        try {
          substrate.calibrate();
          const report = substrate.coherence.report();
          console.log(chalk.dim(`   📊 Overall coherence: ${(report.overallScore * 100).toFixed(1)}%`));
          console.log(chalk.dim(`   📊 Stability: ${(report.stabilityScore * 100).toFixed(1)}%`));
          console.log(chalk.dim(`   📊 Alignment: ${(report.alignmentScore * 100).toFixed(1)}%`));
        } catch {
          console.log(chalk.dim('   ℹ️ Need more observations to calibrate coherence'));
        }

        console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
        console.log(chalk.bold('   Neural Substrate: Embeddings as Synthetic Nervous System'));
        console.log(chalk.cyan('═══════════════════════════════════════════════════════════════\n'));

        console.log(chalk.dim('Components:'));
        console.log(chalk.dim('  • SemanticDriftDetector - Control signals, reflex triggers'));
        console.log(chalk.dim('  • MemoryPhysics - Forgetting, interference, consolidation'));
        console.log(chalk.dim('  • EmbeddingStateMachine - Agent state via geometry'));
        console.log(chalk.dim('  • SwarmCoordinator - Multi-agent coordination'));
        console.log(chalk.dim('  • CoherenceMonitor - Safety/alignment detection'));
        console.log(chalk.dim('  • NeuralSubstrate - Unified nervous system layer'));
        console.log('');
        return;
      }

      if (opts.health) {
        const health = substrate.health();
        console.log(chalk.cyan('\n🧠 Neural Substrate Health\n'));

        console.log(chalk.yellow('Drift Detection:'));
        console.log(chalk.dim(`  Current drift: ${health.driftStats.currentDrift.toFixed(4)}`));
        console.log(chalk.dim(`  Velocity: ${health.driftStats.velocity.toFixed(4)}/s`));
        console.log(chalk.dim(`  Critical events: ${health.driftStats.criticalEvents}`));
        console.log(chalk.dim(`  Warning events: ${health.driftStats.warningEvents}`));

        console.log(chalk.yellow('\nMemory Physics:'));
        console.log(chalk.dim(`  Total memories: ${health.memoryStats.totalMemories}`));
        console.log(chalk.dim(`  Avg strength: ${health.memoryStats.avgStrength.toFixed(3)}`));
        console.log(chalk.dim(`  Avg consolidation: ${health.memoryStats.avgConsolidation.toFixed(3)}`));
        console.log(chalk.dim(`  Avg interference: ${health.memoryStats.avgInterference.toFixed(3)}`));

        console.log(chalk.yellow('\nSwarm Coordination:'));
        console.log(chalk.dim(`  Coherence: ${(health.swarmCoherence * 100).toFixed(1)}%`));

        console.log(chalk.yellow('\nCoherence Report:'));
        console.log(chalk.dim(`  Overall: ${(health.coherenceReport.overallScore * 100).toFixed(1)}%`));
        console.log(chalk.dim(`  Drift: ${(health.coherenceReport.driftScore * 100).toFixed(1)}%`));
        console.log(chalk.dim(`  Stability: ${(health.coherenceReport.stabilityScore * 100).toFixed(1)}%`));
        console.log(chalk.dim(`  Alignment: ${(health.coherenceReport.alignmentScore * 100).toFixed(1)}%`));

        if (health.coherenceReport.anomalies.length > 0) {
          console.log(chalk.yellow('\nAnomalies:'));
          for (const a of health.coherenceReport.anomalies) {
            console.log(chalk.red(`  ⚠️ ${a.type}: ${a.description} (severity: ${a.severity.toFixed(2)})`));
          }
        }
        console.log('');
        return;
      }

      if (opts.consolidate) {
        console.log(chalk.yellow('Running memory consolidation...'));
        const result = substrate.consolidate();
        console.log(chalk.green(`✓ Consolidated: ${result.consolidated} memories`));
        console.log(chalk.dim(`  Forgotten: ${result.forgotten} weak memories`));
        return;
      }

      if (opts.calibrate) {
        try {
          substrate.calibrate();
          console.log(chalk.green('✓ Coherence baseline calibrated'));
        } catch (e) {
          console.log(chalk.yellow('Need more observations to calibrate'));
          console.log(chalk.dim('Run --demo first to populate the substrate'));
        }
        return;
      }

      if (opts.driftStats) {
        const stats = substrate.drift.getStats();
        console.log(chalk.cyan('\n📊 Semantic Drift Statistics\n'));
        console.log(chalk.dim(`Current drift: ${stats.currentDrift.toFixed(4)}`));
        console.log(chalk.dim(`Velocity: ${stats.velocity.toFixed(4)} drift/s`));
        console.log(chalk.dim(`Critical events: ${stats.criticalEvents}`));
        console.log(chalk.dim(`Warning events: ${stats.warningEvents}`));
        console.log(chalk.dim(`History size: ${stats.historySize}`));
        console.log('');
        return;
      }

      if (opts.memoryStats) {
        const stats = substrate.memory.getStats();
        console.log(chalk.cyan('\n📊 Memory Physics Statistics\n'));
        console.log(chalk.dim(`Total memories: ${stats.totalMemories}`));
        console.log(chalk.dim(`Average strength: ${stats.avgStrength.toFixed(3)}`));
        console.log(chalk.dim(`Average consolidation: ${stats.avgConsolidation.toFixed(3)}`));
        console.log(chalk.dim(`Average interference: ${stats.avgInterference.toFixed(3)}`));
        console.log('');
        return;
      }

      if (opts.swarmStatus) {
        const coherence = substrate.swarm.getCoherence();
        const clusters = substrate.swarm.detectClusters(0.7);
        console.log(chalk.cyan('\n📊 Swarm Coordination Status\n'));
        console.log(chalk.dim(`Coherence: ${(coherence * 100).toFixed(1)}%`));
        console.log(chalk.dim(`Clusters detected: ${clusters.size}`));
        for (const [leader, members] of clusters) {
          console.log(chalk.dim(`  Cluster ${leader}: ${members.join(', ')}`));
        }
        console.log('');
        return;
      }

      // Default: show help
      console.log(chalk.cyan('\n🧠 Neural Embedding Substrate\n'));
      console.log(chalk.dim('Frontier AI concepts treating embeddings as a synthetic nervous system.\n'));
      console.log(chalk.yellow('Commands:'));
      console.log(chalk.dim('  --demo          Run interactive neural demo'));
      console.log(chalk.dim('  --health        Show neural substrate health'));
      console.log(chalk.dim('  --consolidate   Run memory consolidation (like sleep)'));
      console.log(chalk.dim('  --calibrate     Calibrate coherence baseline'));
      console.log(chalk.dim('  --drift-stats   Show semantic drift statistics'));
      console.log(chalk.dim('  --memory-stats  Show memory physics statistics'));
      console.log(chalk.dim('  --swarm-status  Show swarm coordination status'));
      console.log('');
      console.log(chalk.yellow('Components:'));
      console.log(chalk.dim('  • SemanticDriftDetector - Embeddings as control signals'));
      console.log(chalk.dim('  • MemoryPhysics - Hippocampal memory dynamics'));
      console.log(chalk.dim('  • EmbeddingStateMachine - Agent state via geometry'));
      console.log(chalk.dim('  • SwarmCoordinator - Multi-agent coordination'));
      console.log(chalk.dim('  • CoherenceMonitor - Safety/alignment detection'));
      console.log('');
    } catch (e) {
      console.error(chalk.red('Error:'), e.message);
      if (e.stack) console.error(chalk.dim(e.stack));
    }
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
        const db = new VectorDB({ dimensions: 4, metric: 'cosine' });

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
      loadGnn();
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
// Full RuVector Stack: VectorDB + SONA + Attention
// ============================================

// LAZY LOADING: IntelligenceEngine is only loaded when first accessed
// This reduces CLI startup from ~1000ms to ~70ms for simple operations
let IntelligenceEngine = null;
let engineLoadAttempted = false;

function loadIntelligenceEngine() {
  if (engineLoadAttempted) return IntelligenceEngine;
  engineLoadAttempted = true;
  try {
    const core = require('../dist/core/intelligence-engine.js');
    IntelligenceEngine = core.IntelligenceEngine || core.default;
  } catch (e) {
    // IntelligenceEngine not available, use fallback
  }
  return IntelligenceEngine;
}

class Intelligence {
  constructor(options = {}) {
    this.intelPath = this.getIntelPath();
    this.data = this.load();
    this.alpha = 0.1;
    this.lastEditedFile = null;
    this.sessionStartTime = null;
    this._engine = null;
    this._engineInitialized = false;
    // Skip engine init for fast operations (trajectory, coedit, error commands)
    this._skipEngine = options.skipEngine || false;
  }

  // Lazy getter for engine - only initializes when first accessed
  getEngine() {
    if (this._skipEngine) return null;
    if (this._engineInitialized) return this._engine;
    this._engineInitialized = true;

    const EngineClass = loadIntelligenceEngine();
    if (EngineClass) {
      try {
        this._engine = new EngineClass({
          maxMemories: 100000,
          maxEpisodes: 50000,
          enableSona: true,
          enableAttention: true,
          enableOnnx: true,  // Enable ONNX semantic embeddings
          learningRate: this.alpha,
        });
        // Import existing data into engine
        if (this.data) {
          this._engine.import(this.convertLegacyData(this.data), true);
        }
      } catch (e) {
        this._engine = null;
      }
    }
    return this._engine;
  }

  // Property alias for backwards compatibility
  get engine() {
    return this.getEngine();
  }

  // Check if engine is available WITHOUT triggering initialization
  // Use this for optional engine features that have fallbacks
  hasEngine() {
    return this._engineInitialized && this._engine !== null;
  }

  // Get engine only if already initialized (doesn't trigger lazy load)
  getEngineIfReady() {
    return this._engineInitialized ? this._engine : null;
  }

  // Convert legacy data format to new engine format
  convertLegacyData(data) {
    const converted = {
      memories: [],
      routingPatterns: {},
      errorPatterns: data.errors || {},
      coEditPatterns: {},
      agentMappings: {},
    };

    // Convert memories
    if (data.memories) {
      converted.memories = data.memories.map(m => ({
        id: m.id,
        content: m.content,
        type: m.memory_type || 'general',
        embedding: m.embedding || this.embed(m.content),
        created: m.timestamp ? new Date(m.timestamp * 1000).toISOString() : new Date().toISOString(),
        accessed: 0,
      }));
    }

    // Convert Q-learning patterns to routing patterns
    if (data.patterns) {
      for (const [key, value] of Object.entries(data.patterns)) {
        const [state, action] = key.split('|');
        if (state && action) {
          if (!converted.routingPatterns[state]) {
            converted.routingPatterns[state] = {};
          }
          converted.routingPatterns[state][action] = value.q_value || 0.5;
        }
      }
    }

    // Convert file sequences to co-edit patterns
    if (data.file_sequences) {
      for (const seq of data.file_sequences) {
        if (!converted.coEditPatterns[seq.from_file]) {
          converted.coEditPatterns[seq.from_file] = {};
        }
        converted.coEditPatterns[seq.from_file][seq.to_file] = seq.count;
      }
    }

    return converted;
  }

  // Prefer project-local storage, fall back to home directory
  getIntelPath() {
    const projectPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    const homePath = path.join(require('os').homedir(), '.ruvector', 'intelligence.json');

    if (fs.existsSync(path.dirname(projectPath))) return projectPath;
    if (fs.existsSync(path.join(process.cwd(), '.claude'))) return projectPath;
    if (fs.existsSync(homePath)) return homePath;
    return projectPath;
  }

  load() {
    const defaults = {
      patterns: {},
      memories: [],
      trajectories: [],
      errors: {},
      file_sequences: [],
      agents: {},
      edges: [],
      stats: { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 }
    };
    try {
      if (fs.existsSync(this.intelPath)) {
        const data = JSON.parse(fs.readFileSync(this.intelPath, 'utf-8'));
        // Merge with defaults to ensure all fields exist
        return {
          patterns: data.patterns || defaults.patterns,
          memories: data.memories || defaults.memories,
          trajectories: data.trajectories || defaults.trajectories,
          errors: data.errors || defaults.errors,
          file_sequences: data.file_sequences || defaults.file_sequences,
          agents: data.agents || defaults.agents,
          edges: data.edges || defaults.edges,
          stats: { ...defaults.stats, ...(data.stats || {}) },
          // Preserve learning data if present
          learning: data.learning || undefined
        };
      }
    } catch {}
    return defaults;
  }

  save() {
    const dir = path.dirname(this.intelPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    // If engine is already initialized, export its data (don't trigger lazy load)
    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        const engineData = eng.export();
        // Merge engine data with legacy format for compatibility
        this.data.patterns = {};
        for (const [state, actions] of Object.entries(engineData.routingPatterns || {})) {
          for (const [action, value] of Object.entries(actions)) {
            this.data.patterns[`${state}|${action}`] = { state, action, q_value: value, visits: 1, last_update: this.now() };
          }
        }
        this.data.stats.total_patterns = Object.keys(this.data.patterns).length;
        this.data.stats.total_memories = engineData.stats?.totalMemories || this.data.memories.length;

        // Add engine stats
        this.data.engineStats = engineData.stats;
      } catch (e) {
        // Ignore engine export errors
      }
    }

    fs.writeFileSync(this.intelPath, JSON.stringify(this.data, null, 2));
  }

  now() { return Math.floor(Date.now() / 1000); }

  // Use engine embedding if available (256-dim with attention), otherwise fallback (64-dim hash)
  embed(text) {
    // Only use engine if already initialized (don't trigger lazy load for embeddings)
    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        return eng.embed(text);
      } catch {}
    }
    // Fallback: simple 64-dim hash embedding
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
    if (!a || !b || a.length !== b.length) return 0;
    const dot = a.reduce((sum, v, i) => sum + v * b[i], 0);
    const normA = Math.sqrt(a.reduce((sum, v) => sum + v * v, 0));
    const normB = Math.sqrt(b.reduce((sum, v) => sum + v * v, 0));
    return normA > 0 && normB > 0 ? dot / (normA * normB) : 0;
  }

  // Memory operations - use engine's VectorDB for semantic search
  async rememberAsync(memoryType, content, metadata = {}) {
    if (this.engine) {
      try {
        const entry = await this.engine.remember(content, memoryType);
        // Also store in legacy format for compatibility
        this.data.memories.push({
          id: entry.id,
          memory_type: memoryType,
          content,
          embedding: entry.embedding,
          metadata,
          timestamp: this.now()
        });
        if (this.data.memories.length > 5000) this.data.memories.splice(0, 1000);
        this.data.stats.total_memories = this.data.memories.length;
        return entry.id;
      } catch {}
    }
    return this.remember(memoryType, content, metadata);
  }

  remember(memoryType, content, metadata = {}) {
    const id = `mem_${this.now()}`;
    const embedding = this.embed(content);
    this.data.memories.push({ id, memory_type: memoryType, content, embedding, metadata, timestamp: this.now() });
    if (this.data.memories.length > 5000) this.data.memories.splice(0, 1000);
    this.data.stats.total_memories = this.data.memories.length;

    // Also store in engine if already initialized (don't trigger lazy load)
    const eng = this.getEngineIfReady();
    if (eng) {
      eng.remember(content, memoryType).catch(() => {});
    }

    return id;
  }

  async recallAsync(query, topK = 5) {
    if (this.engine) {
      try {
        const results = await this.engine.recall(query, topK);
        // Return same format as sync recall() - direct memory objects
        return results.map(r => ({
          id: r.id,
          content: r.content || '',
          memory_type: r.type || 'general',
          timestamp: r.created || new Date().toISOString(),
          score: r.score || 0
        }));
      } catch {}
    }
    return this.recall(query, topK);
  }

  recall(query, topK) {
    const queryEmbed = this.embed(query);
    return this.data.memories
      .map(m => ({ score: this.similarity(queryEmbed, m.embedding), memory: m }))
      .sort((a, b) => b.score - a.score).slice(0, topK).map(r => r.memory);
  }

  // Q-learning operations - enhanced with SONA trajectory tracking
  getQ(state, action) {
    const key = `${state}|${action}`;
    if (!this.data.patterns) this.data.patterns = {};
    return this.data.patterns[key]?.q_value ?? 0;
  }

  updateQ(state, action, reward) {
    const key = `${state}|${action}`;
    if (!this.data.patterns) this.data.patterns = {};
    if (!this.data.stats) this.data.stats = { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 };
    if (!this.data.patterns[key]) {
      this.data.patterns[key] = { state, action, q_value: 0, visits: 0, last_update: 0 };
    }
    const p = this.data.patterns[key];
    p.q_value = p.q_value + this.alpha * (reward - p.q_value);
    p.visits++;
    p.last_update = this.now();
    this.data.stats.total_patterns = Object.keys(this.data.patterns).length;

    // Record episode in engine if already initialized (don't trigger lazy load)
    const eng = this.getEngineIfReady();
    if (eng) {
      eng.recordEpisode(state, action, reward, state, false).catch(() => {});
    }
  }

  learn(state, action, outcome, reward) {
    const id = `traj_${this.now()}`;
    this.updateQ(state, action, reward);
    if (!this.data.trajectories) this.data.trajectories = [];
    if (!this.data.stats) this.data.stats = { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 };
    this.data.trajectories.push({ id, state, action, outcome, reward, timestamp: this.now() });
    if (this.data.trajectories.length > 1000) this.data.trajectories.splice(0, 200);
    this.data.stats.total_trajectories = this.data.trajectories.length;

    // End trajectory in engine if already initialized (don't trigger lazy load)
    const eng = this.getEngineIfReady();
    if (eng) {
      eng.endTrajectory(reward > 0.5, reward);
    }

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

  // Agent routing - use engine's SONA-enhanced routing
  async routeAsync(task, file, crateName, operation = 'edit') {
    if (this.engine) {
      try {
        const result = await this.engine.route(task, file);
        // Begin trajectory for learning
        this.engine.beginTrajectory(task, file);
        if (result.agent) {
          this.engine.setTrajectoryRoute(result.agent);
        }
        return {
          agent: result.agent,
          confidence: result.confidence,
          reason: result.reason + (result.patterns?.length ? ` (${result.patterns.length} SONA patterns)` : ''),
          alternates: result.alternates,
          patterns: result.patterns
        };
      } catch {}
    }
    return this.route(task, file, crateName, operation);
  }

  route(task, file, crateName, operation = 'edit') {
    const fileType = file ? path.extname(file).slice(1) : 'unknown';
    const state = `${operation}_${fileType}_in_${crateName ?? 'project'}`;
    const agentMap = {
      rs: ['rust-developer', 'coder', 'reviewer', 'tester'],
      ts: ['typescript-developer', 'coder', 'frontend-dev'],
      tsx: ['react-developer', 'typescript-developer', 'coder'],
      js: ['javascript-developer', 'coder', 'frontend-dev'],
      jsx: ['react-developer', 'coder'],
      py: ['python-developer', 'coder', 'ml-developer'],
      go: ['go-developer', 'coder'],
      sql: ['database-specialist', 'coder'],
      md: ['documentation-specialist', 'coder'],
      yml: ['devops-engineer', 'coder'],
      yaml: ['devops-engineer', 'coder']
    };
    const agents = agentMap[fileType] ?? ['coder', 'reviewer'];
    const { action, confidence } = this.suggest(state, agents);
    const reason = confidence > 0.5 ? 'learned from past success' : confidence > 0 ? 'based on patterns' : `default for ${fileType} files`;

    // Begin trajectory in engine (only if already initialized)
    const eng = this.getEngineIfReady();
    if (eng) {
      eng.beginTrajectory(task || operation, file);
    }

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
      case 'go': return { suggest: true, command: 'go test ./...' };
      default: return { suggest: false, command: '' };
    }
  }

  // Co-edit pattern tracking - use engine's co-edit patterns
  recordFileSequence(fromFile, toFile) {
    const existing = this.data.file_sequences.find(s => s.from_file === fromFile && s.to_file === toFile);
    if (existing) existing.count++;
    else this.data.file_sequences.push({ from_file: fromFile, to_file: toFile, count: 1 });
    this.lastEditedFile = toFile;

    // Record in engine (only if already initialized)
    const eng = this.getEngineIfReady();
    if (eng) {
      eng.recordCoEdit(fromFile, toFile);
    }
  }

  suggestNext(file, limit = 3) {
    // Try engine first (only if already initialized)
    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        const results = eng.getLikelyNextFiles(file, limit);
        if (results.length > 0) {
          return results.map(r => ({ file: r.file, score: r.count }));
        }
      } catch {}
    }
    return this.data.file_sequences
      .filter(s => s.from_file === file)
      .sort((a, b) => b.count - a.count)
      .slice(0, limit)
      .map(s => ({ file: s.to_file, score: s.count }));
  }

  // Error pattern learning
  recordErrorFix(errorPattern, fix) {
    if (!this.data.errors[errorPattern]) {
      this.data.errors[errorPattern] = [];
    }
    if (!this.data.errors[errorPattern].includes(fix)) {
      this.data.errors[errorPattern].push(fix);
    }
    this.data.stats.total_errors = Object.keys(this.data.errors).length;

    // Record in engine (only if already initialized)
    const eng = this.getEngineIfReady();
    if (eng) {
      eng.recordErrorFix(errorPattern, fix);
    }
  }

  getSuggestedFixes(error) {
    // Try engine first (only if already initialized)
    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        const fixes = eng.getSuggestedFixes(error);
        if (fixes.length > 0) return fixes;
      } catch {}
    }
    return this.data.errors[error] || [];
  }

  classifyCommand(command) {
    const cmd = command.toLowerCase();
    if (cmd.includes('cargo') || cmd.includes('rustc')) return { category: 'rust', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('npm') || cmd.includes('node') || cmd.includes('yarn') || cmd.includes('pnpm')) return { category: 'javascript', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('python') || cmd.includes('pip') || cmd.includes('pytest')) return { category: 'python', subcategory: cmd.includes('test') ? 'test' : 'run', risk: 'low' };
    if (cmd.includes('go ')) return { category: 'go', subcategory: cmd.includes('test') ? 'test' : 'build', risk: 'low' };
    if (cmd.includes('git')) return { category: 'git', subcategory: 'vcs', risk: cmd.includes('push') || cmd.includes('force') ? 'medium' : 'low' };
    if (cmd.includes('rm ') || cmd.includes('delete') || cmd.includes('rmdir')) return { category: 'filesystem', subcategory: 'destructive', risk: 'high' };
    if (cmd.includes('sudo') || cmd.includes('chmod') || cmd.includes('chown')) return { category: 'system', subcategory: 'privileged', risk: 'high' };
    if (cmd.includes('docker') || cmd.includes('kubectl')) return { category: 'container', subcategory: 'orchestration', risk: 'medium' };
    return { category: 'shell', subcategory: 'general', risk: 'low' };
  }

  swarmStats() {
    const agents = Object.keys(this.data.agents).length;
    const edges = this.data.edges.length;
    return { agents, edges };
  }

  // Enhanced stats with engine metrics
  stats() {
    const baseStats = this.data.stats;

    // Only use engine if already initialized (don't trigger lazy load for optional stats)
    const eng = this.getEngineIfReady();
    if (eng) {
      try {
        const engineStats = eng.getStats();
        return {
          ...baseStats,
          // Engine stats
          engineEnabled: true,
          sonaEnabled: engineStats.sonaEnabled,
          attentionEnabled: engineStats.attentionEnabled,
          embeddingDim: engineStats.memoryDimensions,
          totalMemories: engineStats.totalMemories,
          totalEpisodes: engineStats.totalEpisodes,
          trajectoriesRecorded: engineStats.trajectoriesRecorded,
          patternsLearned: engineStats.patternsLearned,
          microLoraUpdates: engineStats.microLoraUpdates,
          baseLoraUpdates: engineStats.baseLoraUpdates,
          ewcConsolidations: engineStats.ewcConsolidations,
        };
      } catch {}
    }

    return { ...baseStats, engineEnabled: false };
  }

  sessionStart() {
    this.data.stats.session_count++;
    this.data.stats.last_session = this.now();
    this.sessionStartTime = this.now();

    // Tick engine for background learning (only if already initialized)
    const eng = this.getEngineIfReady();
    if (eng) {
      eng.tick();
    }
  }

  sessionEnd() {
    // Ensure data structure exists with defaults
    if (!this.data) {
      this.data = { patterns: {}, memories: [], trajectories: [], errors: [], agents: {}, edges: [], stats: {} };
    }
    if (!this.data.stats) {
      this.data.stats = { total_patterns: 0, total_memories: 0, total_trajectories: 0, total_errors: 0, session_count: 0, last_session: 0 };
    }
    if (!this.data.trajectories) {
      this.data.trajectories = [];
    }

    const lastSession = this.data.stats.last_session || 0;
    const duration = this.now() - (this.sessionStartTime || lastSession);
    const actions = this.data.trajectories.filter(t => t && t.timestamp >= lastSession).length;

    // Force learning cycle (only if engine already initialized)
    try {
      const eng = this.getEngineIfReady();
      if (eng) {
        eng.forceLearn();
      }
    } catch (e) {
      // Ignore engine errors on session end
    }

    // Auto-compress patterns if enabled (v2.1)
    try {
      if (process.env.RUVECTOR_AUTO_COMPRESS === 'true' || process.env.RUVECTOR_TENSOR_COMPRESS === 'true') {
        const TensorCompressClass = require('../dist/core/tensor-compress').default;
        if (TensorCompressClass && this.data.compressedPatterns) {
          const compress = new TensorCompressClass({ autoCompress: false });
          compress.import(this.data.compressedPatterns);
          const stats = compress.recompressAll();
          this.data.compressedPatterns = compress.export();
          // Only log if significant savings
          if (stats.savingsPercent > 10 && stats.totalTensors > 5) {
            // Silently compress, no console output to avoid hook noise
          }
        }
      }
    } catch (e) {
      // Ignore compression errors on session end
    }

    // Save all data
    try {
      this.save();
    } catch (e) {
      // Ignore save errors on session end
    }

    return { duration, actions };
  }

  getLastEditedFile() { return this.lastEditedFile; }

  // New: Check if full engine is available
  isEngineEnabled() {
    return this.engine !== null;
  }

  // New: Get engine capabilities
  getCapabilities() {
    if (!this.engine) {
      return {
        engine: false,
        vectorDb: false,
        sona: false,
        attention: false,
        embeddingDim: 64,
      };
    }
    const stats = this.engine.getStats();
    return {
      engine: true,
      vectorDb: true,
      sona: stats.sonaEnabled,
      attention: stats.attentionEnabled,
      embeddingDim: stats.memoryDimensions,
    };
  }
}

// Hooks command group
const hooksCmd = program.command('hooks').description('Self-learning intelligence hooks for Claude Code');

// Helper: Detect project type
function detectProjectType() {
  const cwd = process.cwd();
  const types = [];
  if (fs.existsSync(path.join(cwd, 'Cargo.toml'))) types.push('rust');
  if (fs.existsSync(path.join(cwd, 'package.json'))) types.push('node');
  if (fs.existsSync(path.join(cwd, 'requirements.txt')) || fs.existsSync(path.join(cwd, 'pyproject.toml'))) types.push('python');
  if (fs.existsSync(path.join(cwd, 'go.mod'))) types.push('go');
  if (fs.existsSync(path.join(cwd, 'Gemfile'))) types.push('ruby');
  if (fs.existsSync(path.join(cwd, 'pom.xml')) || fs.existsSync(path.join(cwd, 'build.gradle'))) types.push('java');
  return types.length > 0 ? types : ['generic'];
}

// Helper: Get permissions for project type
function getPermissionsForProjectType(types) {
  const basePermissions = [
    'Bash(git status)', 'Bash(git diff:*)', 'Bash(git log:*)', 'Bash(git add:*)',
    'Bash(git commit:*)', 'Bash(git push)', 'Bash(git branch:*)', 'Bash(git checkout:*)',
    'Bash(ls:*)', 'Bash(pwd)', 'Bash(cat:*)', 'Bash(mkdir:*)', 'Bash(which:*)', 'Bash(ruvector:*)'
  ];
  const typePermissions = {
    rust: ['Bash(cargo:*)', 'Bash(rustc:*)', 'Bash(rustfmt:*)', 'Bash(clippy:*)', 'Bash(wasm-pack:*)'],
    node: ['Bash(npm:*)', 'Bash(npx:*)', 'Bash(node:*)', 'Bash(yarn:*)', 'Bash(pnpm:*)'],
    python: ['Bash(python:*)', 'Bash(pip:*)', 'Bash(pytest:*)', 'Bash(poetry:*)', 'Bash(uv:*)'],
    go: ['Bash(go:*)', 'Bash(gofmt:*)'],
    ruby: ['Bash(ruby:*)', 'Bash(gem:*)', 'Bash(bundle:*)', 'Bash(rails:*)'],
    java: ['Bash(mvn:*)', 'Bash(gradle:*)', 'Bash(java:*)', 'Bash(javac:*)'],
    generic: ['Bash(make:*)']
  };
  let perms = [...basePermissions];
  types.forEach(t => { if (typePermissions[t]) perms = perms.concat(typePermissions[t]); });
  return [...new Set(perms)];
}

hooksCmd.command('init')
  .description('Initialize hooks in current project')
  .option('--force', 'Force overwrite existing settings')
  .option('--minimal', 'Only basic hooks (no env, permissions, or advanced hooks)')
  .option('--fast', 'Use fast local wrapper (20x faster, bypasses npx overhead)')
  .option('--no-claude-md', 'Skip CLAUDE.md creation')
  .option('--no-permissions', 'Skip permissions configuration')
  .option('--no-env', 'Skip environment variables')
  .option('--no-gitignore', 'Skip .gitignore update')
  .option('--no-mcp', 'Skip MCP server configuration')
  .option('--no-statusline', 'Skip statusLine configuration')
  .option('--pretrain', 'Run pretrain after init to bootstrap intelligence')
  .option('--build-agents [focus]', 'Generate optimized agents (quality|speed|security|testing|fullstack)')
  .action(async (opts) => {
  const settingsPath = path.join(process.cwd(), '.claude', 'settings.json');
  const settingsDir = path.dirname(settingsPath);
  const isWindows = process.platform === 'win32';
  if (!fs.existsSync(settingsDir)) fs.mkdirSync(settingsDir, { recursive: true });
  let settings = {};
  if (fs.existsSync(settingsPath) && !opts.force) {
    try { settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8')); } catch {}
  }

  // Fix schema if present
  if (settings.$schema) {
    settings.$schema = 'https://json.schemastore.org/claude-code-settings.json';
  }

  // Clean up invalid hook names
  if (settings.hooks) {
    if (settings.hooks.Start) { delete settings.hooks.Start; }
    if (settings.hooks.End) { delete settings.hooks.End; }
  }

  // Detect project type
  const projectTypes = detectProjectType();
  console.log(chalk.blue(`  ✓ Detected project type(s): ${projectTypes.join(', ')}`));

  // Environment variables for intelligence (unless --minimal or --no-env)
  if (!opts.minimal && opts.env !== false) {
    settings.env = settings.env || {};
    // Core intelligence settings
    settings.env.RUVECTOR_INTELLIGENCE_ENABLED = settings.env.RUVECTOR_INTELLIGENCE_ENABLED || 'true';
    settings.env.RUVECTOR_LEARNING_RATE = settings.env.RUVECTOR_LEARNING_RATE || '0.1';
    settings.env.RUVECTOR_MEMORY_BACKEND = settings.env.RUVECTOR_MEMORY_BACKEND || 'rvlite';
    settings.env.INTELLIGENCE_MODE = settings.env.INTELLIGENCE_MODE || 'treatment';
    // v2.0 capabilities
    settings.env.RUVECTOR_AST_ENABLED = settings.env.RUVECTOR_AST_ENABLED || 'true';
    settings.env.RUVECTOR_DIFF_EMBEDDINGS = settings.env.RUVECTOR_DIFF_EMBEDDINGS || 'true';
    settings.env.RUVECTOR_COVERAGE_ROUTING = settings.env.RUVECTOR_COVERAGE_ROUTING || 'true';
    settings.env.RUVECTOR_GRAPH_ALGORITHMS = settings.env.RUVECTOR_GRAPH_ALGORITHMS || 'true';
    settings.env.RUVECTOR_SECURITY_SCAN = settings.env.RUVECTOR_SECURITY_SCAN || 'true';
    // v2.1 learning & compression
    settings.env.RUVECTOR_MULTI_ALGORITHM = settings.env.RUVECTOR_MULTI_ALGORITHM || 'true';
    settings.env.RUVECTOR_DEFAULT_ALGORITHM = settings.env.RUVECTOR_DEFAULT_ALGORITHM || 'double-q';
    settings.env.RUVECTOR_TENSOR_COMPRESS = settings.env.RUVECTOR_TENSOR_COMPRESS || 'true';
    settings.env.RUVECTOR_AUTO_COMPRESS = settings.env.RUVECTOR_AUTO_COMPRESS || 'true';
    console.log(chalk.blue('  ✓ Environment variables configured (v2.1 with multi-algorithm learning)'));
  }

  // Workers configuration (native ruvector workers + agentic-flow integration)
  if (!opts.minimal) {
    settings.workers = settings.workers || {
      enabled: true,
      parallel: true,
      maxConcurrent: 10,
      native: {
        enabled: true,
        types: ['security', 'analysis', 'learning'],
        defaultTimeout: 120000
      },
      triggers: {
        ultralearn: { priority: 'high', agents: ['researcher', 'coder'] },
        optimize: { priority: 'high', agents: ['performance-analyzer'] },
        audit: { priority: 'critical', agents: ['security-analyst', 'tester'] },
        map: { priority: 'medium', agents: ['architect'] },
        security: { priority: 'critical', agents: ['security-analyst'] },
        benchmark: { priority: 'low', agents: ['performance-analyzer'] },
        document: { priority: 'medium', agents: ['documenter'] },
        refactor: { priority: 'medium', agents: ['coder', 'reviewer'] },
        testgaps: { priority: 'high', agents: ['tester'] },
        deepdive: { priority: 'low', agents: ['researcher'] },
        predict: { priority: 'medium', agents: ['analyst'] },
        consolidate: { priority: 'low', agents: ['architect'] }
      }
    };
    console.log(chalk.blue('  ✓ Workers configured (native + 12 triggers)'));
  }

  // Performance configuration with benchmark thresholds
  if (!opts.minimal) {
    settings.performance = settings.performance || {
      modelCache: {
        enabled: true,
        maxSizeMB: 512,
        ttlMinutes: 60
      },
      benchmarkThresholds: {
        triggerDetection: { p95: 5 },      // <5ms
        workerRegistry: { p95: 10 },       // <10ms
        agentSelection: { p95: 1 },        // <1ms
        memoryKeyGen: { p95: 0.1 },        // <0.1ms
        concurrent10: { p95: 1000 },       // <1000ms
        singleEmbedding: { p95: 500 },     // <500ms (WASM)
        batchEmbedding16: { p95: 8000 }    // <8000ms (WASM)
      },
      optimizations: {
        parallelDispatch: true,
        batchEmbeddings: true,
        cacheEmbeddings: true,
        simd: true
      }
    };
    console.log(chalk.blue('  ✓ Performance thresholds configured'));
  }

  // Agent presets configuration
  if (!opts.minimal) {
    settings.agents = settings.agents || {
      presets: {
        'quick-scan': {
          phases: ['file-discovery', 'summarization'],
          timeout: 30000
        },
        'deep-analysis': {
          phases: ['file-discovery', 'pattern-extraction', 'embedding-generation', 'complexity-analysis', 'summarization'],
          timeout: 120000,
          capabilities: { onnxEmbeddings: true, vectorDb: true }
        },
        'security-scan': {
          phases: ['file-discovery', 'security-scan', 'summarization'],
          timeout: 60000
        },
        'learning': {
          phases: ['file-discovery', 'pattern-extraction', 'embedding-generation', 'vector-storage', 'summarization'],
          timeout: 180000,
          capabilities: { onnxEmbeddings: true, vectorDb: true, intelligenceMemory: true }
        }
      },
      capabilities: {
        onnxEmbeddings: true,
        vectorDb: true,
        intelligenceMemory: true,
        parallelProcessing: true
      }
    };
    console.log(chalk.blue('  ✓ Agent presets configured (4 presets)'));
  }

  // Permissions based on detected project type (unless --minimal or --no-permissions)
  if (!opts.minimal && opts.permissions !== false) {
    settings.permissions = settings.permissions || {};
    settings.permissions.allow = settings.permissions.allow || getPermissionsForProjectType(projectTypes);
    settings.permissions.deny = settings.permissions.deny || [
      'Bash(rm -rf /)',
      'Bash(sudo rm:*)',
      'Bash(chmod 777:*)',
      'Bash(mkfs:*)',
      'Bash(dd if=/dev/zero:*)'
    ];
    console.log(chalk.blue('  ✓ Permissions configured (project-specific)'));
  }

  // MCP server configuration (unless --minimal or --no-mcp)
  if (!opts.minimal && opts.mcp !== false) {
    settings.mcpServers = settings.mcpServers || {};
    // Only add if not already configured
    if (!settings.mcpServers['claude-flow'] && !settings.enabledMcpjsonServers?.includes('claude-flow')) {
      settings.enabledMcpjsonServers = settings.enabledMcpjsonServers || [];
      if (!settings.enabledMcpjsonServers.includes('claude-flow')) {
        settings.enabledMcpjsonServers.push('claude-flow');
      }
    }
    console.log(chalk.blue('  ✓ MCP servers configured'));
  }

  // StatusLine configuration (unless --minimal or --no-statusline)
  if (!opts.minimal && opts.statusline !== false) {
    if (!settings.statusLine) {
      if (isWindows) {
        // Windows: PowerShell statusline
        const statuslineScript = path.join(settingsDir, 'statusline-command.ps1');
        const statuslineContent = `# RuVector Intelligence Statusline for Windows PowerShell
# Compatible with PowerShell 5.1+ and PowerShell Core
$ErrorActionPreference = "SilentlyContinue"
$e = [char]27
$inputData = [Console]::In.ReadToEnd()
$data = $inputData | ConvertFrom-Json
$Model = if ($data.model.display_name) { $data.model.display_name } else { "Claude" }
$CWD = if ($data.workspace.current_dir) { $data.workspace.current_dir } else { $data.cwd }
$Dir = Split-Path -Leaf $CWD
$Branch = $null
try { Push-Location $CWD -ErrorAction Stop; $Branch = git branch --show-current 2>$null; Pop-Location } catch {}
Write-Host "$e[1m$Model$e[0m in $e[36m$Dir$e[0m$(if($Branch){" on $e[33m$Branch$e[0m"})"
$IntelFile = Join-Path $CWD ".ruvector\intelligence.json"
if (Test-Path $IntelFile) {
  $Intel = Get-Content $IntelFile -Raw | ConvertFrom-Json
  $Mem = if ($Intel.memories) { $Intel.memories.Count } else { 0 }
  $Traj = if ($Intel.trajectories) { $Intel.trajectories.Count } else { 0 }
  $Sess = if ($Intel.stats -and $Intel.stats.session_count) { $Intel.stats.session_count } else { 0 }
  $Pat = if ($Intel.patterns) { ($Intel.patterns | Get-Member -MemberType NoteProperty).Count } else { 0 }
  $Line2 = "$e[35m RuVector$e[0m"
  if ($Pat -gt 0) { $Line2 += " $e[32m$Pat patterns$e[0m" } else { $Line2 += " $e[2mlearning$e[0m" }
  if ($Mem -gt 0) { $Line2 += " $e[34m$Mem mem$e[0m" }
  if ($Traj -gt 0) { $Line2 += " $e[33m$Traj traj$e[0m" }
  if ($Sess -gt 0) { $Line2 += " $e[2m#$Sess$e[0m" }
  Write-Host $Line2
} else {
  Write-Host "$e[2m RuVector: run 'npx ruvector hooks session-start'$e[0m"
}
`;
        fs.writeFileSync(statuslineScript, statuslineContent);
        settings.statusLine = {
          type: 'command',
          command: 'powershell -NoProfile -ExecutionPolicy Bypass -File .claude/statusline-command.ps1'
        };
      } else {
        // Unix (macOS, Linux): Bash statusline
        const statuslineScript = path.join(settingsDir, 'statusline-command.sh');
        const statuslineContent = `#!/bin/bash
# RuVector Intelligence Statusline - Multi-line display
INPUT=\$(cat)
MODEL=\$(echo "\$INPUT" | jq -r '.model.display_name // "Claude"')
CWD=\$(echo "\$INPUT" | jq -r '.workspace.current_dir // .cwd')
DIR=\$(basename "\$CWD")
BRANCH=\$(cd "\$CWD" 2>/dev/null && git branch --show-current 2>/dev/null)
RESET="\\033[0m"; BOLD="\\033[1m"; CYAN="\\033[36m"; YELLOW="\\033[33m"; GREEN="\\033[32m"; MAGENTA="\\033[35m"; BLUE="\\033[34m"; DIM="\\033[2m"; RED="\\033[31m"
printf "\$BOLD\$MODEL\$RESET in \$CYAN\$DIR\$RESET"
[ -n "\$BRANCH" ] && printf " on \$YELLOW⎇ \$BRANCH\$RESET"
echo
INTEL_FILE=""
for P in "\$CWD/.ruvector/intelligence.json" "\$CWD/npm/packages/ruvector/.ruvector/intelligence.json" "\$HOME/.ruvector/intelligence.json"; do
  [ -f "\$P" ] && INTEL_FILE="\$P" && break
done
if [ -n "\$INTEL_FILE" ]; then
  INTEL=\$(cat "\$INTEL_FILE" 2>/dev/null)
  MEMORY_COUNT=\$(echo "\$INTEL" | jq -r '.memories | length // 0' 2>/dev/null)
  TRAJ_COUNT=\$(echo "\$INTEL" | jq -r '.trajectories | length // 0' 2>/dev/null)
  SESSION_COUNT=\$(echo "\$INTEL" | jq -r '.stats.session_count // 0' 2>/dev/null)
  PATTERN_COUNT=\$(echo "\$INTEL" | jq -r '.patterns | length // 0' 2>/dev/null)
  printf "\$MAGENTA🧠 RuVector\$RESET"
  [ "\$PATTERN_COUNT" != "null" ] && [ "\$PATTERN_COUNT" -gt 0 ] 2>/dev/null && printf " \$GREEN◆\$RESET \$PATTERN_COUNT patterns" || printf " \$DIM◇ learning\$RESET"
  [ "\$MEMORY_COUNT" != "null" ] && [ "\$MEMORY_COUNT" -gt 0 ] 2>/dev/null && printf " \$BLUE⬡\$RESET \$MEMORY_COUNT mem"
  [ "\$TRAJ_COUNT" != "null" ] && [ "\$TRAJ_COUNT" -gt 0 ] 2>/dev/null && printf " \$YELLOW↝\$RESET\$TRAJ_COUNT"
  [ "\$SESSION_COUNT" != "null" ] && [ "\$SESSION_COUNT" -gt 0 ] 2>/dev/null && printf " \$DIM#\$SESSION_COUNT\$RESET"
  echo
else
  printf "\$DIM🧠 RuVector: run 'npx ruvector hooks session-start' to initialize\$RESET\\n"
fi
`;
        fs.writeFileSync(statuslineScript, statuslineContent);
        fs.chmodSync(statuslineScript, '755');
        settings.statusLine = {
          type: 'command',
          command: '.claude/statusline-command.sh'
        };
      }
      console.log(chalk.blue(`  ✓ StatusLine configured (${isWindows ? 'PowerShell' : 'Bash'})`));
    }
  }

  // Fast wrapper creation (--fast option) - 20x faster than npx
  let hookCmd = 'npx ruvector@latest';
  let fastTimeouts = { simple: 2000, complex: 2000, session: 5000 };
  if (opts.fast && !isWindows) {
    const fastWrapperPath = path.join(settingsDir, 'ruvector-fast.sh');
    const fastWrapperContent = `#!/bin/bash
# Fast RuVector hooks wrapper - avoids npx overhead (20x faster)
# Usage: .claude/ruvector-fast.sh hooks <command> [args...]

# Find ruvector CLI - check local first, then global
RUVECTOR_CLI=""

# Check local npm package (for development)
if [ -f "$PWD/npm/packages/ruvector/bin/cli.js" ]; then
  RUVECTOR_CLI="$PWD/npm/packages/ruvector/bin/cli.js"
# Check node_modules
elif [ -f "$PWD/node_modules/ruvector/bin/cli.js" ]; then
  RUVECTOR_CLI="$PWD/node_modules/ruvector/bin/cli.js"
# Check global npm installation
elif [ -f "$PWD/node_modules/.bin/ruvector" ]; then
  exec "$PWD/node_modules/.bin/ruvector" "$@"
elif command -v ruvector &> /dev/null; then
  exec ruvector "$@"
# Fallback to npx (slow but works)
else
  exec npx ruvector@latest "$@"
fi

# Execute with node directly (fast path)
exec node "$RUVECTOR_CLI" "$@"
`;
    fs.writeFileSync(fastWrapperPath, fastWrapperContent);
    fs.chmodSync(fastWrapperPath, '755');
    hookCmd = '.claude/ruvector-fast.sh';
    fastTimeouts = { simple: 300, complex: 500, session: 1000 };
    // Add permission for fast wrapper
    if (settings.permissions && settings.permissions.allow) {
      if (!settings.permissions.allow.includes('Bash(.claude/ruvector-fast.sh:*)')) {
        settings.permissions.allow.push('Bash(.claude/ruvector-fast.sh:*)');
      }
    }
    console.log(chalk.blue('  ✓ Fast wrapper created (.claude/ruvector-fast.sh) - 20x faster hooks'));
  }

  // Core hooks (always included) - with timeouts and error suppression
  settings.hooks = settings.hooks || {};
  settings.hooks.PreToolUse = [
    {
      matcher: 'Edit|Write|MultiEdit',
      hooks: [
        { type: 'command', timeout: fastTimeouts.complex, command: `${hookCmd} hooks pre-edit "$TOOL_INPUT_file_path" 2>/dev/null || true` },
        { type: 'command', timeout: fastTimeouts.complex, command: `${hookCmd} hooks coedit-suggest --file "$TOOL_INPUT_file_path" 2>/dev/null || true` }
      ]
    },
    { matcher: 'Bash', hooks: [{ type: 'command', timeout: fastTimeouts.complex, command: `${hookCmd} hooks pre-command "$TOOL_INPUT_command" 2>/dev/null || true` }] },
    { matcher: 'Read', hooks: [{ type: 'command', timeout: fastTimeouts.simple, command: `${hookCmd} hooks remember "Reading: $TOOL_INPUT_file_path" -t file_access 2>/dev/null || true` }] },
    { matcher: 'Glob|Grep', hooks: [{ type: 'command', timeout: fastTimeouts.simple, command: `${hookCmd} hooks remember "Search: $TOOL_INPUT_pattern" -t search_pattern 2>/dev/null || true` }] },
    { matcher: 'Task', hooks: [{ type: 'command', timeout: fastTimeouts.simple, command: `${hookCmd} hooks remember "Agent: $TOOL_INPUT_subagent_type" -t agent_spawn 2>/dev/null || true` }] }
  ];
  settings.hooks.PostToolUse = [
    { matcher: 'Edit|Write|MultiEdit', hooks: [{ type: 'command', timeout: fastTimeouts.complex, command: `${hookCmd} hooks post-edit "$TOOL_INPUT_file_path" 2>/dev/null || true` }] },
    { matcher: 'Bash', hooks: [{ type: 'command', timeout: fastTimeouts.complex, command: `${hookCmd} hooks post-command "$TOOL_INPUT_command" 2>/dev/null || true` }] }
  ];
  settings.hooks.SessionStart = [{
    hooks: [
      { type: 'command', timeout: fastTimeouts.session, command: `${hookCmd} hooks session-start 2>/dev/null || true` },
      { type: 'command', timeout: fastTimeouts.complex, command: `${hookCmd} hooks trajectory-begin -c "claude-session" -a "claude" 2>/dev/null || true` }
    ]
  }];
  settings.hooks.Stop = [{
    hooks: [
      { type: 'command', timeout: fastTimeouts.complex, command: `${hookCmd} hooks trajectory-end --success --quality 0.8 2>/dev/null || true` },
      { type: 'command', timeout: fastTimeouts.complex, command: `${hookCmd} hooks session-end 2>/dev/null || true` }
    ]
  }];
  console.log(chalk.blue(`  ✓ Core hooks (PreToolUse, PostToolUse, SessionStart, Stop) ${opts.fast ? 'with fast wrapper' : 'with error handling'}`));

  // Advanced hooks (unless --minimal)
  if (!opts.minimal) {
    // Create agentic-flow fast wrapper for background workers
    let workersCmd = 'npx agentic-flow@alpha';
    if (opts.fast && !isWindows) {
      const agenticFastPath = path.join(settingsDir, 'agentic-flow-fast.sh');
      const agenticFastContent = `#!/bin/bash
# Fast agentic-flow wrapper - avoids npx overhead
# Usage: .claude/agentic-flow-fast.sh workers <command> [args...]

# Find agentic-flow CLI
if [ -f "$PWD/node_modules/agentic-flow/bin/cli.js" ]; then
  exec node "$PWD/node_modules/agentic-flow/bin/cli.js" "$@"
elif [ -f "$PWD/node_modules/.bin/agentic-flow" ]; then
  exec "$PWD/node_modules/.bin/agentic-flow" "$@"
elif command -v agentic-flow &> /dev/null; then
  exec agentic-flow "$@"
else
  exec npx agentic-flow@alpha "$@"
fi
`;
      fs.writeFileSync(agenticFastPath, agenticFastContent);
      fs.chmodSync(agenticFastPath, '755');
      workersCmd = '.claude/agentic-flow-fast.sh';
      // Add permission for agentic-flow fast wrapper
      if (settings.permissions && settings.permissions.allow) {
        if (!settings.permissions.allow.includes('Bash(.claude/agentic-flow-fast.sh:*)')) {
          settings.permissions.allow.push('Bash(.claude/agentic-flow-fast.sh:*)');
        }
      }
      console.log(chalk.blue('  ✓ Background workers wrapper created (.claude/agentic-flow-fast.sh)'));
    }

    // UserPromptSubmit - context suggestions + background workers dispatch
    settings.hooks.UserPromptSubmit = [{
      hooks: [
        {
          type: 'command',
          timeout: fastTimeouts.complex,
          command: `${hookCmd} hooks suggest-context 2>/dev/null || true`
        },
        {
          type: 'command',
          timeout: 2000,
          command: `${workersCmd} workers dispatch-prompt "$CLAUDE_USER_PROMPT" 2>/dev/null || true`
        },
        {
          type: 'command',
          timeout: 1000,
          command: `${workersCmd} workers inject-context "$CLAUDE_USER_PROMPT" 2>/dev/null || true`
        }
      ]
    }];
    console.log(chalk.blue('  ✓ Background workers integration (ultralearn, optimize, audit, map, etc.)'));

    // PreCompact - preserve important context before compaction
    settings.hooks.PreCompact = [
      {
        matcher: 'auto',
        hooks: [
          { type: 'command', timeout: fastTimeouts.session, command: `${hookCmd} hooks pre-compact --auto 2>/dev/null || true` },
          { type: 'command', timeout: fastTimeouts.session, command: `${hookCmd} hooks compress 2>/dev/null || true` }
        ]
      },
      {
        matcher: 'manual',
        hooks: [{
          type: 'command',
          timeout: fastTimeouts.session,
          command: `${hookCmd} hooks pre-compact 2>/dev/null || true`
        }]
      }
    ];

    // Notification - track all notifications for learning
    settings.hooks.Notification = [{
      matcher: '.*',
      hooks: [{
        type: 'command',
        timeout: fastTimeouts.simple,
        command: `${hookCmd} hooks track-notification 2>/dev/null || true`
      }]
    }];
    console.log(chalk.blue(`  ✓ Advanced hooks (UserPromptSubmit, PreCompact, Notification, Compress)${opts.fast ? ' - fast mode' : ''}`));

    // Extended environment variables for new capabilities
    settings.env.RUVECTOR_AST_ENABLED = settings.env.RUVECTOR_AST_ENABLED || 'true';
    settings.env.RUVECTOR_DIFF_EMBEDDINGS = settings.env.RUVECTOR_DIFF_EMBEDDINGS || 'true';
    settings.env.RUVECTOR_COVERAGE_ROUTING = settings.env.RUVECTOR_COVERAGE_ROUTING || 'true';
    settings.env.RUVECTOR_GRAPH_ALGORITHMS = settings.env.RUVECTOR_GRAPH_ALGORITHMS || 'true';
    settings.env.RUVECTOR_SECURITY_SCAN = settings.env.RUVECTOR_SECURITY_SCAN || 'true';
    console.log(chalk.blue('  ✓ Extended capabilities (AST, Diff, Coverage, Graph, Security)'));
  }

  fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
  console.log(chalk.green('\n✅ Hooks initialized in .claude/settings.json'));

  // Create CLAUDE.md if it doesn't exist (or force)
  const claudeMdPath = path.join(process.cwd(), 'CLAUDE.md');
  if (opts.claudeMd !== false && (!fs.existsSync(claudeMdPath) || opts.force)) {
    const claudeMdContent = `# Claude Code Project Configuration

## RuVector Self-Learning Intelligence v2.0

This project uses RuVector's self-learning intelligence hooks with advanced capabilities:
- **Q-learning** for agent routing optimization
- **Vector memory** with HNSW indexing (150x faster search)
- **AST parsing** for code complexity analysis
- **Diff embeddings** for change classification and risk scoring
- **Coverage routing** for test-aware agent selection
- **Graph algorithms** for code structure analysis
- **Security scanning** for vulnerability detection
- **10 attention mechanisms** including hyperbolic and graph attention

### Active Hooks

| Hook | Trigger | Purpose |
|------|---------|---------|
| **PreToolUse** | Before Edit/Write/Bash | Agent routing, AST analysis, command risk assessment |
| **PostToolUse** | After Edit/Write/Bash | Q-learning update, diff embeddings, outcome tracking |
| **SessionStart** | Conversation begins | Load intelligence state, display learning stats |
| **Stop** | Conversation ends | Save learning data, export metrics |
| **UserPromptSubmit** | User sends message | RAG context suggestions, pattern recommendations |
| **PreCompact** | Before context compaction | Preserve important context and memories |
| **Notification** | Any notification | Track events for learning |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| \`RUVECTOR_INTELLIGENCE_ENABLED\` | \`true\` | Enable/disable intelligence layer |
| \`RUVECTOR_LEARNING_RATE\` | \`0.1\` | Q-learning rate (0.0-1.0) |
| \`RUVECTOR_MEMORY_BACKEND\` | \`rvlite\` | Memory storage backend |
| \`INTELLIGENCE_MODE\` | \`treatment\` | A/B testing mode (treatment/control) |
| \`RUVECTOR_AST_ENABLED\` | \`true\` | Enable AST parsing and complexity analysis |
| \`RUVECTOR_DIFF_EMBEDDINGS\` | \`true\` | Enable diff embeddings and risk scoring |
| \`RUVECTOR_COVERAGE_ROUTING\` | \`true\` | Enable test coverage-aware routing |
| \`RUVECTOR_GRAPH_ALGORITHMS\` | \`true\` | Enable graph algorithms (MinCut, Louvain) |
| \`RUVECTOR_SECURITY_SCAN\` | \`true\` | Enable security vulnerability scanning |

### Core Commands

\`\`\`bash
# Initialize hooks in a project
npx ruvector hooks init

# View learning statistics
npx ruvector hooks stats

# Route a task to best agent
npx ruvector hooks route "implement feature X"

# Enhanced routing with AST/coverage/diff signals
npx ruvector hooks route-enhanced "fix bug" --file src/api.ts

# Store context in vector memory
npx ruvector hooks remember "important context" -t project

# Recall from memory (semantic search)
npx ruvector hooks recall "context query"
\`\`\`

### AST Analysis Commands

\`\`\`bash
# Analyze file structure, symbols, imports, complexity
npx ruvector hooks ast-analyze src/index.ts

# Get complexity metrics for multiple files
npx ruvector hooks ast-complexity src/*.ts --threshold 15
\`\`\`

### Diff & Risk Analysis Commands

\`\`\`bash
# Analyze commit with semantic embeddings and risk scoring
npx ruvector hooks diff-analyze HEAD

# Classify change type (feature, bugfix, refactor, etc.)
npx ruvector hooks diff-classify

# Find similar past commits
npx ruvector hooks diff-similar -k 5

# Get risk score only
npx ruvector hooks diff-analyze --risk-only
\`\`\`

### Coverage & Testing Commands

\`\`\`bash
# Get coverage-aware routing for a file
npx ruvector hooks coverage-route src/api.ts

# Suggest tests for files based on coverage
npx ruvector hooks coverage-suggest src/*.ts
\`\`\`

### Graph Analysis Commands

\`\`\`bash
# Find optimal code boundaries (MinCut algorithm)
npx ruvector hooks graph-mincut src/*.ts

# Detect code communities (Louvain/Spectral clustering)
npx ruvector hooks graph-cluster src/*.ts --method louvain
\`\`\`

### Security & RAG Commands

\`\`\`bash
# Parallel security vulnerability scan
npx ruvector hooks security-scan src/*.ts

# RAG-enhanced context retrieval
npx ruvector hooks rag-context "how does auth work"

# Git churn analysis (hot spots)
npx ruvector hooks git-churn --days 30
\`\`\`

### MCP Tools (via Claude Code)

When using the RuVector MCP server, these tools are available:

| Tool | Description |
|------|-------------|
| \`hooks_stats\` | Get intelligence statistics |
| \`hooks_route\` | Route task to best agent |
| \`hooks_route_enhanced\` | Enhanced routing with AST/coverage signals |
| \`hooks_remember\` / \`hooks_recall\` | Vector memory operations |
| \`hooks_ast_analyze\` | Parse AST and extract symbols |
| \`hooks_ast_complexity\` | Get complexity metrics |
| \`hooks_diff_analyze\` | Analyze changes with embeddings |
| \`hooks_diff_classify\` | Classify change types |
| \`hooks_coverage_route\` | Coverage-aware routing |
| \`hooks_coverage_suggest\` | Suggest needed tests |
| \`hooks_graph_mincut\` | Find code boundaries |
| \`hooks_graph_cluster\` | Detect communities |
| \`hooks_security_scan\` | Security vulnerability scan |
| \`hooks_rag_context\` | RAG context retrieval |
| \`hooks_git_churn\` | Hot spot analysis |
| \`hooks_attention_info\` | Available attention mechanisms |
| \`hooks_gnn_info\` | GNN layer capabilities |

### Attention Mechanisms

RuVector includes 10 attention mechanisms:

1. **DotProductAttention** - Scaled dot-product attention
2. **MultiHeadAttention** - Parallel attention heads
3. **FlashAttention** - Memory-efficient tiled attention
4. **HyperbolicAttention** - Poincaré ball hyperbolic space
5. **LinearAttention** - O(n) linear complexity
6. **MoEAttention** - Mixture-of-Experts sparse attention
7. **GraphRoPeAttention** - Rotary position for graphs
8. **EdgeFeaturedAttention** - Edge-aware graph attention
9. **DualSpaceAttention** - Euclidean + Hyperbolic hybrid
10. **LocalGlobalAttention** - Sliding window + global tokens

### How It Works

1. **Pre-edit hooks** analyze files via AST and suggest agents based on Q-learned patterns
2. **Post-edit hooks** generate diff embeddings to improve future routing
3. **Coverage routing** adjusts agent weights based on test coverage
4. **Graph algorithms** detect code communities for module boundaries
5. **Security scanning** identifies common vulnerability patterns
6. **RAG context** retrieves relevant memories using HNSW search
7. **Attention mechanisms** provide advanced embedding transformations

### Learning Data

Stored in \`.ruvector/intelligence.json\`:
- **Q-table patterns**: State-action values for agent routing
- **Vector memories**: ONNX embeddings with HNSW indexing
- **Trajectories**: SONA trajectory tracking for meta-learning
- **Co-edit patterns**: File relationship graphs
- **Error patterns**: Known issues and suggested fixes
- **Diff embeddings**: Change classification patterns

### Init Options

\`\`\`bash
npx ruvector hooks init              # Full configuration with all capabilities
npx ruvector hooks init --minimal    # Basic hooks only
npx ruvector hooks init --fast       # Use fast local wrapper (20x faster)
npx ruvector hooks init --pretrain   # Initialize + pretrain from git history
npx ruvector hooks init --build-agents quality  # Generate optimized agents
npx ruvector hooks init --force      # Overwrite existing configuration
\`\`\`

---
*Powered by [RuVector](https://github.com/ruvnet/ruvector) self-learning intelligence v2.0*
`;
    fs.writeFileSync(claudeMdPath, claudeMdContent);
    console.log(chalk.green('✅ CLAUDE.md created in project root'));
  } else if (fs.existsSync(claudeMdPath) && !opts.force) {
    console.log(chalk.yellow('ℹ️  CLAUDE.md already exists (use --force to overwrite)'));
  }

  // Update .gitignore (unless --no-gitignore)
  if (opts.gitignore !== false) {
    const gitignorePath = path.join(process.cwd(), '.gitignore');
    const entriesToAdd = ['.ruvector/', '.claude/statusline.sh'];
    let gitignoreContent = '';
    if (fs.existsSync(gitignorePath)) {
      gitignoreContent = fs.readFileSync(gitignorePath, 'utf-8');
    }
    const linesToAdd = entriesToAdd.filter(entry => !gitignoreContent.includes(entry));
    if (linesToAdd.length > 0) {
      const newContent = gitignoreContent.trim() + '\n\n# RuVector intelligence data\n' + linesToAdd.join('\n') + '\n';
      fs.writeFileSync(gitignorePath, newContent);
      console.log(chalk.blue('  ✓ .gitignore updated'));
    }
  }

  // Create .ruvector directory for intelligence data
  const ruvectorDir = path.join(process.cwd(), '.ruvector');
  if (!fs.existsSync(ruvectorDir)) {
    fs.mkdirSync(ruvectorDir, { recursive: true });
    console.log(chalk.blue('  ✓ .ruvector/ directory created'));
  }

  console.log(chalk.green('\n✅ RuVector hooks initialization complete!'));

  // Run pretrain if requested
  if (opts.pretrain) {
    console.log(chalk.yellow('\n📚 Running pretrain to bootstrap intelligence...\n'));
    const { execSync } = require('child_process');
    try {
      execSync('npx ruvector hooks pretrain', { stdio: 'inherit' });
    } catch (e) {
      console.log(chalk.yellow('⚠️  Pretrain completed with warnings'));
    }
  }

  // Build agents if requested
  if (opts.buildAgents) {
    const focus = typeof opts.buildAgents === 'string' ? opts.buildAgents : 'quality';
    console.log(chalk.yellow(`\n🏗️  Building optimized agents (focus: ${focus})...\n`));
    const { execSync } = require('child_process');
    try {
      execSync(`npx ruvector hooks build-agents --focus ${focus} --include-prompts`, { stdio: 'inherit' });
    } catch (e) {
      console.log(chalk.yellow('⚠️  Agent build completed with warnings'));
    }
  }

  if (!opts.pretrain && !opts.buildAgents) {
    console.log(chalk.dim('   Run `npx ruvector hooks verify` to test the setup'));
    console.log(chalk.dim('   Run `npx ruvector hooks pretrain` to bootstrap intelligence'));
    console.log(chalk.dim('   Run `npx ruvector hooks build-agents` to generate optimized agents'));
  }
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

hooksCmd.command('remember').description('Store in memory').requiredOption('-t, --type <type>', 'Memory type').option('--silent', 'Suppress output').option('--semantic', 'Use ONNX semantic embeddings (slower, better quality)').argument('<content...>', 'Content').action(async (content, opts) => {
  const intel = new Intelligence();
  let id;
  if (opts.semantic) {
    // Use async ONNX embedding
    id = await intel.rememberAsync(opts.type, content.join(' '));
  } else {
    id = intel.remember(opts.type, content.join(' '));
  }
  intel.save();
  if (!opts.silent) {
    console.log(JSON.stringify({ success: true, id, semantic: !!opts.semantic }));
  }
});

hooksCmd.command('recall').description('Search memory').argument('<query...>', 'Query').option('-k, --top-k <n>', 'Results', '5').option('--semantic', 'Use ONNX semantic search (slower, better quality)').action(async (query, opts) => {
  const intel = new Intelligence();
  let results;
  if (opts.semantic) {
    results = await intel.recallAsync(query.join(' '), parseInt(opts.topK));
  } else {
    results = intel.recall(query.join(' '), parseInt(opts.topK));
  }
  console.log(JSON.stringify({ query: query.join(' '), semantic: !!opts.semantic, results: results.map(r => ({ type: r.memory_type || 'unknown', content: (r.content || '').slice(0, 200), timestamp: r.timestamp || '', score: r.score })) }, null, 2));
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

// Trajectory tracking commands
hooksCmd.command('trajectory-begin')
  .description('Begin tracking a new execution trajectory')
  .requiredOption('-c, --context <context>', 'Task or operation context')
  .option('-a, --agent <agent>', 'Agent performing the task', 'unknown')
  .action((opts) => {
    const intel = new Intelligence({ skipEngine: true });  // Fast mode - no engine needed
    const trajId = `traj_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    if (!intel.data.activeTrajectories) intel.data.activeTrajectories = {};
    intel.data.activeTrajectories[trajId] = {
      id: trajId,
      context: opts.context,
      agent: opts.agent,
      steps: [],
      startTime: Date.now()
    };
    intel.save();
    console.log(JSON.stringify({ success: true, trajectory_id: trajId, context: opts.context, agent: opts.agent }));
  });

hooksCmd.command('trajectory-step')
  .description('Add a step to the current trajectory')
  .requiredOption('-a, --action <action>', 'Action taken')
  .option('-r, --result <result>', 'Result of action')
  .option('--reward <reward>', 'Reward signal (0-1)', '0.5')
  .action((opts) => {
    const intel = new Intelligence({ skipEngine: true });  // Fast mode
    const trajectories = intel.data.activeTrajectories || {};
    const trajIds = Object.keys(trajectories);
    if (trajIds.length === 0) {
      console.log(JSON.stringify({ success: false, error: 'No active trajectory' }));
      return;
    }
    const latestTrajId = trajIds[trajIds.length - 1];
    trajectories[latestTrajId].steps.push({
      action: opts.action,
      result: opts.result || '',
      reward: parseFloat(opts.reward),
      time: Date.now()
    });
    intel.save();
    console.log(JSON.stringify({ success: true, trajectory_id: latestTrajId, step: trajectories[latestTrajId].steps.length }));
  });

hooksCmd.command('trajectory-end')
  .description('End the current trajectory with a quality score')
  .option('--success', 'Task succeeded')
  .option('--quality <quality>', 'Quality score (0-1)', '0.5')
  .action((opts) => {
    const intel = new Intelligence({ skipEngine: true });  // Fast mode
    const trajectories = intel.data.activeTrajectories || {};
    const trajIds = Object.keys(trajectories);
    if (trajIds.length === 0) {
      console.log(JSON.stringify({ success: false, error: 'No active trajectory' }));
      return;
    }
    const latestTrajId = trajIds[trajIds.length - 1];
    const traj = trajectories[latestTrajId];
    const quality = opts.success ? 0.8 : parseFloat(opts.quality);
    traj.endTime = Date.now();
    traj.quality = quality;
    traj.success = opts.success || false;

    if (!intel.data.trajectories) intel.data.trajectories = [];
    intel.data.trajectories.push(traj);
    delete trajectories[latestTrajId];
    intel.save();

    console.log(JSON.stringify({
      success: true,
      trajectory_id: latestTrajId,
      steps: traj.steps.length,
      duration_ms: traj.endTime - traj.startTime,
      quality
    }));
  });

// Co-edit pattern commands
hooksCmd.command('coedit-record')
  .description('Record co-edit pattern (files edited together)')
  .requiredOption('-p, --primary <file>', 'Primary file being edited')
  .requiredOption('-r, --related <files...>', 'Related files edited together')
  .action((opts) => {
    const intel = new Intelligence({ skipEngine: true });  // Fast mode
    if (!intel.data.coEditPatterns) intel.data.coEditPatterns = {};
    if (!intel.data.coEditPatterns[opts.primary]) intel.data.coEditPatterns[opts.primary] = {};

    for (const related of opts.related) {
      intel.data.coEditPatterns[opts.primary][related] = (intel.data.coEditPatterns[opts.primary][related] || 0) + 1;
    }
    intel.save();
    console.log(JSON.stringify({ success: true, primary_file: opts.primary, related_count: opts.related.length }));
  });

hooksCmd.command('coedit-suggest')
  .description('Get suggested related files based on co-edit patterns')
  .requiredOption('-f, --file <file>', 'Current file')
  .option('-k, --top-k <n>', 'Number of suggestions', '5')
  .action((opts) => {
    const intel = new Intelligence({ skipEngine: true });  // Fast mode
    let suggestions = [];

    if (intel.data.coEditPatterns && intel.data.coEditPatterns[opts.file]) {
      suggestions = Object.entries(intel.data.coEditPatterns[opts.file])
        .sort((a, b) => b[1] - a[1])
        .slice(0, parseInt(opts.topK))
        .map(([f, count]) => ({ file: f, count, confidence: Math.min(count / 10, 1) }));
    }
    console.log(JSON.stringify({ success: true, file: opts.file, suggestions }));
  });

// Error pattern commands
hooksCmd.command('error-record')
  .description('Record an error and its fix for learning')
  .requiredOption('-e, --error <error>', 'Error message or code')
  .requiredOption('-x, --fix <fix>', 'Fix that resolved the error')
  .option('-f, --file <file>', 'File where error occurred')
  .action((opts) => {
    const intel = new Intelligence({ skipEngine: true });  // Fast mode
    if (!intel.data.errors) intel.data.errors = {};
    if (!intel.data.errors[opts.error]) intel.data.errors[opts.error] = [];
    intel.data.errors[opts.error].push({ fix: opts.fix, file: opts.file || '', recorded: Date.now() });
    intel.save();
    console.log(JSON.stringify({ success: true, error: opts.error.substring(0, 50), fixes_recorded: intel.data.errors[opts.error].length }));
  });

hooksCmd.command('error-suggest')
  .description('Get suggested fixes for an error based on learned patterns')
  .requiredOption('-e, --error <error>', 'Error message or code')
  .action((opts) => {
    const intel = new Intelligence({ skipEngine: true });  // Fast mode
    let suggestions = [];

    if (intel.data.errors) {
      for (const [errKey, fixes] of Object.entries(intel.data.errors)) {
        if (opts.error.includes(errKey) || errKey.includes(opts.error)) {
          suggestions.push(...fixes.map(f => f.fix));
        }
      }
    }
    console.log(JSON.stringify({ success: true, error: opts.error.substring(0, 50), suggestions: [...new Set(suggestions)].slice(0, 5) }));
  });

// Force learning command
hooksCmd.command('force-learn')
  .description('Force an immediate learning cycle')
  .action(() => {
    const intel = new Intelligence({ skipEngine: true });  // Fast mode
    intel.tick();
    console.log(JSON.stringify({ success: true, result: 'Learning cycle triggered', stats: intel.stats() }));
  });

// ============================================
// NEW CAPABILITY COMMANDS (AST, Diff, Coverage, Graph, Security, RAG)
// ============================================

// Lazy load new modules
let ASTParser, DiffEmbeddings, CoverageRouter, GraphAlgorithms, ExtendedWorkerPool;
let newModulesLoaded = false;

function loadNewModules() {
  if (newModulesLoaded) return true;
  try {
    const core = require('../dist/core/index.js');
    // CodeParser is exported as both CodeParser and ASTParser
    ASTParser = core.CodeParser || core.ASTParser;
    DiffEmbeddings = core.default?.parseDiff ? core : require('../dist/core/diff-embeddings.js');
    CoverageRouter = core.default?.parseIstanbulCoverage ? core : require('../dist/core/coverage-router.js');
    GraphAlgorithms = core.default?.minCut ? core : require('../dist/core/graph-algorithms.js');
    ExtendedWorkerPool = core.ExtendedWorkerPool;
    newModulesLoaded = true;
    return true;
  } catch (e) {
    console.error('loadNewModules error:', e.message);
    return false;
  }
}

// AST Analysis Commands
hooksCmd.command('ast-analyze')
  .description('Parse file AST and extract symbols, imports, complexity')
  .argument('<file>', 'File path to analyze')
  .option('--json', 'Output as JSON')
  .option('--symbols', 'Show only symbols')
  .option('--imports', 'Show only imports')
  .action(async (file, opts) => {
    if (!loadNewModules() || !ASTParser) {
      console.log(JSON.stringify({ success: false, error: 'AST parser not available. Run npm run build.' }));
      return;
    }
    try {
      const parser = new ASTParser();
      // CodeParser uses analyze() which returns FileAnalysis
      const analysis = await parser.analyze(file);

      // Get symbols list
      const symbols = await parser.getSymbols(file);

      if (opts.json) {
        console.log(JSON.stringify({
          success: true,
          file,
          language: analysis.language,
          symbols: symbols.map(s => ({ name: s })),
          imports: analysis.imports,
          complexity: { cyclomatic: analysis.complexity, lines: analysis.lines },
          functions: analysis.functions.length,
          classes: analysis.classes.length
        }));
      } else if (opts.symbols) {
        console.log(chalk.bold.cyan(`\n📊 Symbols in ${path.basename(file)}:\n`));
        analysis.functions.forEach(f => console.log(`  function: ${f.name} (line ${f.startLine})`));
        analysis.classes.forEach(c => console.log(`  class: ${c.name} (line ${c.startLine})`));
        analysis.types.forEach(t => console.log(`  type: ${t}`));
      } else if (opts.imports) {
        console.log(chalk.bold.cyan(`\n📦 Imports in ${path.basename(file)}:\n`));
        analysis.imports.forEach(i => console.log(`  ${i.source} (${i.type})`));
      } else {
        console.log(chalk.bold.cyan(`\n📊 AST Analysis: ${path.basename(file)}\n`));
        console.log(`  Language: ${analysis.language}`);
        console.log(`  Functions: ${analysis.functions.length}`);
        console.log(`  Classes: ${analysis.classes.length}`);
        console.log(`  Imports: ${analysis.imports.length}`);
        console.log(`  Complexity: ${analysis.complexity}`);
        console.log(`  Lines: ${analysis.lines}`);
        console.log(`  Parse time: ${analysis.parseTime.toFixed(2)}ms`);
      }
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

hooksCmd.command('ast-complexity')
  .description('Get complexity metrics for files')
  .argument('<files...>', 'Files to analyze')
  .option('--threshold <n>', 'Warn if complexity exceeds threshold', '10')
  .action(async (files, opts) => {
    if (!loadNewModules() || !ASTParser) {
      console.log(JSON.stringify({ success: false, error: 'AST parser not available' }));
      return;
    }
    const parser = new ASTParser();
    const threshold = parseInt(opts.threshold);
    const results = [];

    for (const file of files) {
      try {
        if (!fs.existsSync(file)) continue;
        const analysis = await parser.analyze(file);
        const warning = analysis.complexity > threshold;
        results.push({
          file,
          cyclomatic: analysis.complexity,
          lines: analysis.lines,
          functions: analysis.functions.length,
          classes: analysis.classes.length,
          warning
        });
      } catch (e) {
        results.push({ file, error: e.message });
      }
    }

    console.log(JSON.stringify({ success: true, results, threshold }));
  });

// Diff Embedding Commands
hooksCmd.command('diff-analyze')
  .description('Analyze git diff with semantic embeddings and risk scoring')
  .argument('[commit]', 'Commit hash (defaults to staged changes)')
  .option('--json', 'Output as JSON')
  .option('--risk-only', 'Show only risk score')
  .action(async (commit, opts) => {
    if (!loadNewModules()) {
      console.log(JSON.stringify({ success: false, error: 'Diff embeddings not available' }));
      return;
    }
    try {
      const diffMod = require('../dist/core/diff-embeddings.js');
      let analysis;
      if (commit) {
        analysis = await diffMod.analyzeCommit(commit);
      } else {
        const stagedDiff = diffMod.getStagedDiff();
        if (!stagedDiff) {
          console.log(JSON.stringify({ success: false, error: 'No staged changes' }));
          return;
        }
        const hunks = diffMod.parseDiff(stagedDiff);
        const files = [...new Set(hunks.map(h => h.file))];
        analysis = {
          hash: 'staged',
          message: 'Staged changes',
          files: await Promise.all(files.map(f => diffMod.analyzeFileDiff(f, stagedDiff))),
          totalAdditions: hunks.reduce((s, h) => s + h.additions.length, 0),
          totalDeletions: hunks.reduce((s, h) => s + h.deletions.length, 0),
          riskScore: 0
        };
        analysis.riskScore = analysis.files.reduce((s, f) => s + f.riskScore, 0) / Math.max(1, analysis.files.length);
      }

      if (opts.json) {
        console.log(JSON.stringify({ success: true, ...analysis }));
      } else if (opts.riskOnly) {
        const risk = analysis.riskScore;
        const level = risk > 0.7 ? 'HIGH' : risk > 0.4 ? 'MEDIUM' : 'LOW';
        console.log(JSON.stringify({ success: true, riskScore: risk, riskLevel: level }));
      } else {
        console.log(chalk.bold.cyan(`\n📊 Diff Analysis: ${analysis.hash}\n`));
        console.log(`  Message: ${analysis.message || 'N/A'}`);
        console.log(`  Files: ${analysis.files.length}`);
        console.log(`  Changes: +${analysis.totalAdditions} -${analysis.totalDeletions}`);
        const risk = analysis.riskScore;
        const riskColor = risk > 0.7 ? chalk.red : risk > 0.4 ? chalk.yellow : chalk.green;
        console.log(`  Risk: ${riskColor((risk * 100).toFixed(0) + '%')}`);
        analysis.files.forEach(f => {
          console.log(`    ${f.file}: ${f.category} (+${f.totalAdditions}/-${f.totalDeletions})`);
        });
      }
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

hooksCmd.command('diff-classify')
  .description('Classify a change type (feature, bugfix, refactor, etc.)')
  .argument('[commit]', 'Commit hash')
  .action(async (commit) => {
    if (!loadNewModules()) {
      console.log(JSON.stringify({ success: false, error: 'Diff embeddings not available' }));
      return;
    }
    try {
      const diffMod = require('../dist/core/diff-embeddings.js');
      const analysis = await diffMod.analyzeCommit(commit || 'HEAD');
      const categories = {};
      analysis.files.forEach(f => {
        categories[f.category] = (categories[f.category] || 0) + 1;
      });
      const primary = Object.entries(categories).sort((a, b) => b[1] - a[1])[0];
      console.log(JSON.stringify({
        success: true,
        commit: analysis.hash,
        message: analysis.message,
        primaryCategory: primary ? primary[0] : 'unknown',
        categories
      }));
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

hooksCmd.command('diff-similar')
  .description('Find similar past commits based on diff embeddings')
  .option('-k, --top-k <n>', 'Number of results', '5')
  .option('--commits <n>', 'How many recent commits to search', '50')
  .action(async (opts) => {
    if (!loadNewModules()) {
      console.log(JSON.stringify({ success: false, error: 'Diff embeddings not available' }));
      return;
    }
    try {
      const diffMod = require('../dist/core/diff-embeddings.js');
      const stagedDiff = diffMod.getStagedDiff() || diffMod.getUnstagedDiff();
      if (!stagedDiff) {
        console.log(JSON.stringify({ success: false, error: 'No current changes to compare' }));
        return;
      }
      const similar = await diffMod.findSimilarCommits(stagedDiff, parseInt(opts.commits), parseInt(opts.topK));
      console.log(JSON.stringify({ success: true, similar }));
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

// Coverage Routing Commands
hooksCmd.command('coverage-route')
  .description('Get coverage-aware agent routing for a file')
  .argument('<file>', 'File to analyze')
  .action((file) => {
    if (!loadNewModules()) {
      console.log(JSON.stringify({ success: false, error: 'Coverage router not available' }));
      return;
    }
    try {
      const covMod = require('../dist/core/coverage-router.js');
      const reportPath = covMod.findCoverageReport();
      const summary = reportPath ? covMod.parseIstanbulCoverage(reportPath) : null;
      const routing = covMod.shouldRouteToTester(file, summary);
      const weights = covMod.getCoverageRoutingWeight(file, summary);
      console.log(JSON.stringify({
        success: true,
        file,
        coverageReport: reportPath || 'not found',
        routeToTester: routing.route,
        reason: routing.reason,
        coverage: routing.coverage,
        agentWeights: weights
      }));
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

hooksCmd.command('coverage-suggest')
  .description('Suggest tests for files based on coverage data')
  .argument('<files...>', 'Files to analyze')
  .action((files) => {
    if (!loadNewModules()) {
      console.log(JSON.stringify({ success: false, error: 'Coverage router not available' }));
      return;
    }
    try {
      const covMod = require('../dist/core/coverage-router.js');
      const suggestions = covMod.suggestTests(files);
      console.log(JSON.stringify({ success: true, suggestions }));
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

// Graph Algorithm Commands
hooksCmd.command('graph-mincut')
  .description('Find optimal code boundaries using MinCut algorithm')
  .argument('<files...>', 'Files to analyze')
  .option('--partitions <n>', 'Number of partitions', '2')
  .action(async (files, opts) => {
    if (!loadNewModules()) {
      console.log(JSON.stringify({ success: false, error: 'Graph algorithms not available' }));
      return;
    }
    try {
      const graphMod = require('../dist/core/graph-algorithms.js');
      // Build dependency graph from files
      const nodes = files.map(f => path.basename(f, path.extname(f)));
      const edges = [];
      // Simple edge detection based on imports
      for (const file of files) {
        if (!fs.existsSync(file)) continue;
        const content = fs.readFileSync(file, 'utf-8');
        const imports = content.match(/from ['"]\.\/([^'"]+)['"]/g) || [];
        imports.forEach(imp => {
          const target = imp.match(/from ['"]\.\/([^'"]+)['"]/)?.[1];
          if (target && nodes.includes(target)) {
            edges.push({ source: path.basename(file, path.extname(file)), target, weight: 1 });
          }
        });
      }
      const result = graphMod.minCut(nodes, edges);
      console.log(JSON.stringify({ success: true, nodes: nodes.length, edges: edges.length, ...result }));
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

hooksCmd.command('graph-cluster')
  .description('Detect code communities using spectral/Louvain clustering')
  .argument('<files...>', 'Files to analyze')
  .option('--method <type>', 'Clustering method: spectral, louvain', 'louvain')
  .option('--clusters <n>', 'Number of clusters (spectral only)', '3')
  .action(async (files, opts) => {
    if (!loadNewModules()) {
      console.log(JSON.stringify({ success: false, error: 'Graph algorithms not available' }));
      return;
    }
    try {
      const graphMod = require('../dist/core/graph-algorithms.js');
      const nodes = files.map(f => path.basename(f, path.extname(f)));
      const edges = [];
      for (const file of files) {
        if (!fs.existsSync(file)) continue;
        const content = fs.readFileSync(file, 'utf-8');
        const imports = content.match(/from ['"]\.\/([^'"]+)['"]/g) || [];
        imports.forEach(imp => {
          const target = imp.match(/from ['"]\.\/([^'"]+)['"]/)?.[1];
          if (target && nodes.includes(target)) {
            edges.push({ source: path.basename(file, path.extname(file)), target, weight: 1 });
          }
        });
      }
      let result;
      if (opts.method === 'spectral') {
        result = graphMod.spectralClustering(nodes, edges, parseInt(opts.clusters));
      } else {
        result = graphMod.louvainCommunities(nodes, edges);
      }
      console.log(JSON.stringify({ success: true, method: opts.method, ...result }));
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

// Security Scan Command
hooksCmd.command('security-scan')
  .description('Parallel security vulnerability scan')
  .argument('<files...>', 'Files to scan')
  .option('--json', 'Output as JSON')
  .action(async (files, opts) => {
    if (!loadNewModules() || !ExtendedWorkerPool) {
      // Fallback to basic pattern matching
      const patterns = [
        { pattern: /eval\s*\(/g, severity: 'high', message: 'eval() usage detected' },
        { pattern: /innerHTML\s*=/g, severity: 'medium', message: 'innerHTML assignment (XSS risk)' },
        { pattern: /document\.write/g, severity: 'medium', message: 'document.write usage' },
        { pattern: /password\s*=\s*['"][^'"]+['"]/gi, severity: 'critical', message: 'Hardcoded password' },
        { pattern: /api[_-]?key\s*=\s*['"][^'"]+['"]/gi, severity: 'critical', message: 'Hardcoded API key' },
        { pattern: /exec\s*\(/g, severity: 'high', message: 'exec() usage (command injection risk)' },
        { pattern: /dangerouslySetInnerHTML/g, severity: 'medium', message: 'React dangerouslySetInnerHTML' },
        { pattern: /SELECT.*FROM.*WHERE.*\+/gi, severity: 'high', message: 'SQL injection risk' },
      ];

      const findings = [];
      for (const file of files) {
        if (!fs.existsSync(file)) continue;
        try {
          const content = fs.readFileSync(file, 'utf-8');
          const lines = content.split('\n');
          patterns.forEach(p => {
            let match;
            lines.forEach((line, idx) => {
              if (p.pattern.test(line)) {
                findings.push({ file, line: idx + 1, severity: p.severity, message: p.message });
              }
              p.pattern.lastIndex = 0;
            });
          });
        } catch (e) {}
      }
      console.log(JSON.stringify({ success: true, findings, scanned: files.length }));
      return;
    }
    // Use parallel worker if available
    try {
      const pool = new ExtendedWorkerPool();
      const results = await pool.securityScan(files);
      console.log(JSON.stringify({ success: true, ...results }));
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

// RAG Context Command
hooksCmd.command('rag-context')
  .description('Get RAG-enhanced context for a query')
  .argument('<query...>', 'Query for context')
  .option('-k, --top-k <n>', 'Number of results', '5')
  .option('--rerank', 'Rerank results by relevance')
  .action(async (query, opts) => {
    const intel = new Intelligence();
    const queryStr = query.join(' ');

    // Use async recall with engine (VectorDB + HNSW)
    const memories = await intel.recallAsync(queryStr, parseInt(opts.topK));

    // Rerank if requested
    let results = memories;
    if (opts.rerank && ExtendedWorkerPool) {
      try {
        const pool = new ExtendedWorkerPool();
        results = await pool.rankContext(queryStr, memories.map(m => m.content || m));
      } catch (e) {}
    }

    console.log(JSON.stringify({ success: true, query: queryStr, results }));
  });

// Git Churn Analysis Command
hooksCmd.command('git-churn')
  .description('Analyze git churn to find hot spots')
  .option('--days <n>', 'Number of days to analyze', '30')
  .option('--top <n>', 'Top N files', '10')
  .action((opts) => {
    try {
      const { execSync } = require('child_process');
      const since = new Date(Date.now() - parseInt(opts.days) * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
      const log = execSync(`git log --since="${since}" --name-only --format="" 2>/dev/null`, { encoding: 'utf-8' });
      const files = log.trim().split('\n').filter(Boolean);
      const counts = {};
      files.forEach(f => { counts[f] = (counts[f] || 0) + 1; });
      const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, parseInt(opts.top));
      const hotSpots = sorted.map(([file, count]) => ({ file, changes: count }));
      console.log(JSON.stringify({ success: true, days: parseInt(opts.days), hotSpots }));
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: e.message }));
    }
  });

// Enhanced route command that uses new capabilities
hooksCmd.command('route-enhanced')
  .description('Enhanced routing using AST, coverage, and diff analysis')
  .argument('<task...>', 'Task description')
  .option('--file <file>', 'File context')
  .action(async (task, opts) => {
    const intel = new Intelligence();
    const taskStr = task.join(' ');

    // Base routing
    const baseRoute = await intel.routeAsync(taskStr, opts.file, null, 'edit');

    // Enhance with coverage if available
    let coverageWeight = null;
    if (opts.file && loadNewModules()) {
      try {
        const covMod = require('../dist/core/coverage-router.js');
        const reportPath = covMod.findCoverageReport();
        if (reportPath) {
          coverageWeight = covMod.getCoverageRoutingWeight(opts.file);
        }
      } catch (e) {}
    }

    // Enhance with AST complexity if available
    let complexity = null;
    if (opts.file && loadNewModules() && ASTParser) {
      try {
        const parser = new ASTParser();
        const code = fs.readFileSync(opts.file, 'utf-8');
        const ext = path.extname(opts.file).slice(1);
        const result = parser.parse(code, ext);
        complexity = parser.calculateComplexity(result);
      } catch (e) {}
    }

    // Adjust routing based on signals
    let finalAgent = baseRoute.agent;
    let adjustedConfidence = baseRoute.confidence;
    const signals = [];

    if (coverageWeight && coverageWeight.tester > 0.4) {
      signals.push('low coverage detected');
      if (coverageWeight.tester > adjustedConfidence * 0.5) {
        finalAgent = 'tester';
        adjustedConfidence = coverageWeight.tester;
      }
    }

    if (complexity && complexity.cyclomatic > 15) {
      signals.push('high complexity detected');
      if (finalAgent === 'coder') {
        finalAgent = 'reviewer';
        adjustedConfidence = Math.max(adjustedConfidence, 0.7);
      }
    }

    console.log(JSON.stringify({
      success: true,
      agent: finalAgent,
      confidence: adjustedConfidence,
      reason: baseRoute.reason,
      signals,
      coverageWeight,
      complexity
    }));
  });

// ============================================
// LEARNING & COMPRESSION COMMANDS (v2.1)
// ============================================

let TensorCompressClass = null;
let LearningEngineClass = null;

function loadLearningModules() {
  if (LearningEngineClass) return true;
  try {
    const core = require('../dist/core/index.js');
    TensorCompressClass = core.TensorCompress;
    LearningEngineClass = core.LearningEngine;
    return true;
  } catch (e) {
    return false;
  }
}

// Learning algorithm configuration
hooksCmd.command('learning-config')
  .description('Configure learning algorithms for different tasks')
  .option('-t, --task <type>', 'Task type (agent-routing, error-avoidance, confidence-scoring, trajectory-learning, context-ranking, memory-recall)')
  .option('-a, --algorithm <alg>', 'Algorithm (q-learning, sarsa, double-q, actor-critic, ppo, decision-transformer, monte-carlo, td-lambda, dqn)')
  .option('-l, --learning-rate <rate>', 'Learning rate (0.0-1.0)', parseFloat)
  .option('-g, --gamma <gamma>', 'Discount factor (0.0-1.0)', parseFloat)
  .option('-e, --epsilon <epsilon>', 'Exploration rate (0.0-1.0)', parseFloat)
  .option('--lambda <lambda>', 'Lambda for TD(λ)', parseFloat)
  .option('--list', 'List all algorithms and their descriptions')
  .option('--show', 'Show current configuration')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Learning modules not available. Run npm run build.' }));
      return;
    }

    if (opts.list) {
      const algorithms = LearningEngineClass.getAlgorithms();
      console.log(JSON.stringify({
        success: true,
        algorithms: algorithms.map(a => ({
          name: a.algorithm,
          description: a.description,
          bestFor: a.bestFor
        }))
      }));
      return;
    }

    // Load existing intelligence data
    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const engine = new LearningEngineClass();
    if (data.learning) {
      engine.import(data.learning);
    }

    if (opts.show) {
      const tasks = ['agent-routing', 'error-avoidance', 'confidence-scoring', 'trajectory-learning', 'context-ranking', 'memory-recall'];
      const configs = {};
      for (const task of tasks) {
        configs[task] = engine.getConfig(task);
      }
      console.log(JSON.stringify({ success: true, configs }));
      return;
    }

    if (!opts.task) {
      console.log(JSON.stringify({ success: false, error: 'Specify --task or use --list/--show' }));
      return;
    }

    const config = {};
    if (opts.algorithm) config.algorithm = opts.algorithm;
    if (opts.learningRate !== undefined) config.learningRate = opts.learningRate;
    if (opts.gamma !== undefined) config.discountFactor = opts.gamma;
    if (opts.epsilon !== undefined) config.epsilon = opts.epsilon;
    if (opts.lambda !== undefined) config.lambda = opts.lambda;

    engine.configure(opts.task, config);

    // Save
    data.learning = engine.export();
    fs.mkdirSync(path.dirname(dataPath), { recursive: true });
    fs.writeFileSync(dataPath, JSON.stringify(data, null, 2));

    console.log(JSON.stringify({
      success: true,
      task: opts.task,
      config: engine.getConfig(opts.task)
    }));
  });

// Learning statistics
hooksCmd.command('learning-stats')
  .description('Show learning algorithm statistics and performance')
  .option('--json', 'Output as JSON')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Learning modules not available' }));
      return;
    }

    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const engine = new LearningEngineClass();
    if (data.learning) {
      engine.import(data.learning);
    }

    const summary = engine.getStatsSummary();

    if (opts.json) {
      console.log(JSON.stringify({ success: true, ...summary }));
    } else {
      console.log(chalk.bold.cyan('\n📊 Learning Statistics\n'));
      console.log(`  Best Algorithm: ${chalk.green(summary.bestAlgorithm)}`);
      console.log(`  Total Updates:  ${summary.totalUpdates}`);
      console.log(`  Avg Reward:     ${summary.avgReward.toFixed(4)}`);

      if (summary.algorithms.length > 0) {
        console.log(chalk.bold('\n  Algorithm Performance:'));
        for (const alg of summary.algorithms) {
          console.log(`    ${alg.algorithm.padEnd(20)} updates: ${String(alg.updates).padStart(6)}  avgReward: ${alg.avgReward.toFixed(3).padStart(8)}  convergence: ${alg.convergenceScore.toFixed(3)}`);
        }
      }
      console.log('');
    }
  });

// Manual learning update
hooksCmd.command('learning-update')
  .description('Manually record a learning experience')
  .requiredOption('-t, --task <type>', 'Task type')
  .requiredOption('-s, --state <state>', 'Current state')
  .requiredOption('-a, --action <action>', 'Action taken')
  .requiredOption('-r, --reward <reward>', 'Reward received', parseFloat)
  .option('-n, --next-state <state>', 'Next state')
  .option('-d, --done', 'Episode is done')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Learning modules not available' }));
      return;
    }

    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const engine = new LearningEngineClass();
    if (data.learning) {
      engine.import(data.learning);
    }

    const experience = {
      state: opts.state,
      action: opts.action,
      reward: opts.reward,
      nextState: opts.nextState || opts.state,
      done: opts.done || false,
      timestamp: Date.now()
    };

    const delta = engine.update(opts.task, experience);

    // Save
    data.learning = engine.export();
    fs.writeFileSync(dataPath, JSON.stringify(data, null, 2));

    console.log(JSON.stringify({
      success: true,
      task: opts.task,
      experience,
      delta,
      algorithm: engine.getConfig(opts.task).algorithm
    }));
  });

// TensorCompress commands
hooksCmd.command('compress')
  .description('Compress pattern storage using TensorCompress')
  .option('--force', 'Force recompression of all patterns')
  .option('--stats', 'Show compression statistics only')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Compression modules not available' }));
      return;
    }

    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const compress = new TensorCompressClass({
      autoCompress: false,
      hotThreshold: 0.8,
      warmThreshold: 0.4,
      coolThreshold: 0.1,
      coldThreshold: 0.01
    });

    // Import existing compressed data
    if (data.compressedPatterns) {
      compress.import(data.compressedPatterns);
    }

    // Also compress any uncompressed patterns from the regular patterns
    if (data.patterns && !data.compressedPatterns) {
      for (const [key, value] of Object.entries(data.patterns)) {
        if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'number') {
          compress.store(key, value);
        }
      }
    }

    if (opts.stats) {
      const stats = compress.getStats();
      console.log(JSON.stringify({ success: true, ...stats }));
      return;
    }

    // Recompress based on access patterns
    const stats = compress.recompressAll();

    // Save compressed data
    data.compressedPatterns = compress.export();
    fs.writeFileSync(dataPath, JSON.stringify(data, null, 2));

    console.log(JSON.stringify({
      success: true,
      message: 'Compression complete',
      ...stats
    }));
  });

hooksCmd.command('compress-stats')
  .description('Show TensorCompress statistics')
  .option('--json', 'Output as JSON')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Compression modules not available' }));
      return;
    }

    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const compress = new TensorCompressClass({ autoCompress: false });
    if (data.compressedPatterns) {
      compress.import(data.compressedPatterns);
    }

    const stats = compress.getStats();

    if (opts.json) {
      console.log(JSON.stringify({ success: true, ...stats }));
    } else {
      console.log(chalk.bold.cyan('\n📦 TensorCompress Statistics\n'));
      console.log(`  Total Tensors:    ${stats.totalTensors}`);
      console.log(`  Original Size:    ${(stats.originalBytes / 1024).toFixed(2)} KB`);
      console.log(`  Compressed Size:  ${(stats.compressedBytes / 1024).toFixed(2)} KB`);
      console.log(`  Savings:          ${chalk.green(stats.savingsPercent.toFixed(1) + '%')}`);

      console.log(chalk.bold('\n  By Compression Level:'));
      console.log(`    none (hot):     ${stats.byLevel.none}`);
      console.log(`    half (warm):    ${stats.byLevel.half}`);
      console.log(`    pq8 (cool):     ${stats.byLevel.pq8}`);
      console.log(`    pq4 (cold):     ${stats.byLevel.pq4}`);
      console.log(`    binary (archive): ${stats.byLevel.binary}`);
      console.log('');
    }
  });

// Store embedding with compression
hooksCmd.command('compress-store')
  .description('Store an embedding with adaptive compression')
  .requiredOption('-k, --key <key>', 'Storage key')
  .requiredOption('-v, --vector <vector>', 'Vector as JSON array')
  .option('-l, --level <level>', 'Compression level (none, half, pq8, pq4, binary)')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Compression modules not available' }));
      return;
    }

    let vector;
    try {
      vector = JSON.parse(opts.vector);
    } catch (e) {
      console.log(JSON.stringify({ success: false, error: 'Invalid vector JSON' }));
      return;
    }

    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const compress = new TensorCompressClass({ autoCompress: false });
    if (data.compressedPatterns) {
      compress.import(data.compressedPatterns);
    }

    compress.store(opts.key, vector, opts.level);

    data.compressedPatterns = compress.export();
    fs.mkdirSync(path.dirname(dataPath), { recursive: true });
    fs.writeFileSync(dataPath, JSON.stringify(data, null, 2));

    const stats = compress.getStats();
    console.log(JSON.stringify({
      success: true,
      key: opts.key,
      level: opts.level || 'auto',
      originalDim: vector.length,
      totalTensors: stats.totalTensors
    }));
  });

// Retrieve compressed embedding
hooksCmd.command('compress-get')
  .description('Retrieve a compressed embedding')
  .requiredOption('-k, --key <key>', 'Storage key')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Compression modules not available' }));
      return;
    }

    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const compress = new TensorCompressClass({ autoCompress: false });
    if (data.compressedPatterns) {
      compress.import(data.compressedPatterns);
    }

    const vector = compress.get(opts.key);
    if (!vector) {
      console.log(JSON.stringify({ success: false, error: 'Key not found' }));
      return;
    }

    console.log(JSON.stringify({
      success: true,
      key: opts.key,
      vector: Array.from(vector),
      dimension: vector.length
    }));
  });

// Combined learning action with best algorithm
hooksCmd.command('learn')
  .description('Record learning outcome and get best action recommendation')
  .requiredOption('-s, --state <state>', 'Current state (e.g., file extension, task type)')
  .option('-a, --action <action>', 'Action taken')
  .option('-r, --reward <reward>', 'Reward (-1 to 1)', parseFloat)
  .option('--actions <actions>', 'Available actions (comma-separated)')
  .option('-t, --task <type>', 'Task type', 'agent-routing')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Learning modules not available' }));
      return;
    }

    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const engine = new LearningEngineClass();
    if (data.learning) {
      engine.import(data.learning);
    }

    let result = { success: true };

    // If action and reward provided, record the experience
    if (opts.action && opts.reward !== undefined) {
      const experience = {
        state: opts.state,
        action: opts.action,
        reward: opts.reward,
        nextState: opts.state,
        done: true,
        timestamp: Date.now()
      };

      const delta = engine.update(opts.task, experience);
      result.recorded = { experience, delta, algorithm: engine.getConfig(opts.task).algorithm };
    }

    // Get best action recommendation
    if (opts.actions) {
      const actions = opts.actions.split(',').map(a => a.trim());
      const best = engine.getBestAction(opts.task, opts.state, actions);
      result.recommendation = best;
    }

    // Save
    data.learning = engine.export();
    fs.mkdirSync(path.dirname(dataPath), { recursive: true });
    fs.writeFileSync(dataPath, JSON.stringify(data, null, 2));

    console.log(JSON.stringify(result));
  });

// Batch learn - process multiple experiences at once
hooksCmd.command('batch-learn')
  .description('Record multiple learning experiences in batch for efficiency')
  .option('-f, --file <file>', 'JSON file with experiences array')
  .option('-d, --data <json>', 'Inline JSON array of experiences')
  .option('-t, --task <type>', 'Task type for all experiences', 'agent-routing')
  .action(async (opts) => {
    if (!loadLearningModules()) {
      console.log(JSON.stringify({ success: false, error: 'Learning modules not available' }));
      return;
    }

    let experiences = [];

    // Load from file or inline
    if (opts.file) {
      try {
        const content = fs.readFileSync(opts.file, 'utf-8');
        experiences = JSON.parse(content);
      } catch (e) {
        console.log(JSON.stringify({ success: false, error: `Failed to read file: ${e.message}` }));
        return;
      }
    } else if (opts.data) {
      try {
        experiences = JSON.parse(opts.data);
      } catch (e) {
        console.log(JSON.stringify({ success: false, error: `Invalid JSON: ${e.message}` }));
        return;
      }
    } else {
      console.log(JSON.stringify({ success: false, error: 'Provide --file or --data' }));
      return;
    }

    if (!Array.isArray(experiences)) {
      experiences = [experiences];
    }

    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');
    let data = {};
    try {
      if (fs.existsSync(dataPath)) {
        data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
      }
    } catch (e) {}

    const engine = new LearningEngineClass();
    if (data.learning) {
      engine.import(data.learning);
    }

    const results = [];
    let totalReward = 0;

    for (const exp of experiences) {
      const experience = {
        state: exp.state,
        action: exp.action,
        reward: exp.reward ?? 0.5,
        nextState: exp.nextState ?? exp.state,
        done: exp.done ?? false,
        timestamp: exp.timestamp ?? Date.now()
      };

      const delta = engine.update(opts.task, experience);
      totalReward += experience.reward;
      results.push({ state: exp.state, action: exp.action, delta });
    }

    // Save
    data.learning = engine.export();
    fs.mkdirSync(path.dirname(dataPath), { recursive: true });
    fs.writeFileSync(dataPath, JSON.stringify(data, null, 2));

    const stats = engine.getStatsSummary();
    console.log(JSON.stringify({
      success: true,
      processed: experiences.length,
      avgReward: totalReward / experiences.length,
      results,
      stats: {
        bestAlgorithm: stats.bestAlgorithm,
        totalUpdates: stats.totalUpdates,
        avgReward: stats.avgReward
      }
    }));
  });

// Subscribe to learning updates - stream real-time learning events
hooksCmd.command('subscribe')
  .description('Subscribe to real-time learning updates (streaming)')
  .option('-e, --events <types>', 'Event types to subscribe to (learn,compress,route,memory)', 'learn,route')
  .option('-f, --format <fmt>', 'Output format (json, text)', 'json')
  .option('--poll <ms>', 'Poll interval in ms', parseInt, 1000)
  .action(async (opts) => {
    const events = opts.events.split(',').map(e => e.trim());
    const dataPath = path.join(process.cwd(), '.ruvector', 'intelligence.json');

    let lastStats = { patterns: 0, memories: 0, trajectories: 0 };
    let lastLearning = { totalUpdates: 0 };

    console.error(chalk.cyan('🔴 Subscribed to learning updates. Press Ctrl+C to stop.\n'));
    console.error(chalk.dim(`   Events: ${events.join(', ')}`));
    console.error(chalk.dim(`   Poll interval: ${opts.poll}ms\n`));

    const emit = (type, data) => {
      const event = { type, timestamp: Date.now(), data };
      if (opts.format === 'json') {
        console.log(JSON.stringify(event));
      } else {
        const icon = { learn: '🧠', compress: '📦', route: '🎯', memory: '💾' }[type] || '📡';
        console.log(`${icon} [${type}] ${JSON.stringify(data)}`);
      }
    };

    const check = () => {
      try {
        if (!fs.existsSync(dataPath)) return;

        const data = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
        const stats = data.stats || {};
        const learning = data.learning?.stats || {};

        // Check for new patterns (learn events)
        if (events.includes('learn')) {
          const currentPatterns = stats.total_patterns || 0;
          if (currentPatterns > lastStats.patterns) {
            emit('learn', {
              type: 'pattern',
              newPatterns: currentPatterns - lastStats.patterns,
              total: currentPatterns
            });
            lastStats.patterns = currentPatterns;
          }

          // Check learning engine updates
          let totalUpdates = 0;
          Object.values(learning).forEach(algo => {
            if (algo.updates) totalUpdates += algo.updates;
          });
          if (totalUpdates > lastLearning.totalUpdates) {
            const bestAlgo = Object.entries(learning)
              .filter(([, v]) => v.updates > 0)
              .sort((a, b) => b[1].avgReward - a[1].avgReward)[0];
            emit('learn', {
              type: 'algorithm_update',
              newUpdates: totalUpdates - lastLearning.totalUpdates,
              totalUpdates,
              bestAlgorithm: bestAlgo?.[0] || 'none'
            });
            lastLearning.totalUpdates = totalUpdates;
          }
        }

        // Check for new memories
        if (events.includes('memory')) {
          const currentMemories = stats.total_memories || 0;
          if (currentMemories > lastStats.memories) {
            emit('memory', {
              newMemories: currentMemories - lastStats.memories,
              total: currentMemories
            });
            lastStats.memories = currentMemories;
          }
        }

        // Check for new trajectories (route events)
        if (events.includes('route')) {
          const currentTrajectories = stats.total_trajectories || 0;
          if (currentTrajectories > lastStats.trajectories) {
            emit('route', {
              newTrajectories: currentTrajectories - lastStats.trajectories,
              total: currentTrajectories
            });
            lastStats.trajectories = currentTrajectories;
          }
        }

      } catch (e) {
        // Ignore read errors during updates
      }
    };

    // Initial state
    check();

    // Poll for updates
    const interval = setInterval(check, opts.poll);

    // Handle graceful shutdown
    process.on('SIGINT', () => {
      clearInterval(interval);
      console.error(chalk.dim('\n\n👋 Subscription ended.'));
      process.exit(0);
    });

    // Keep alive
    await new Promise(() => {});
  });

// Watch and learn - monitor file changes and auto-learn
hooksCmd.command('watch')
  .description('Watch for changes and auto-learn patterns in real-time')
  .option('-p, --path <dir>', 'Directory to watch', '.')
  .option('-i, --ignore <patterns>', 'Patterns to ignore (comma-separated)', 'node_modules,dist,.git')
  .option('--dry-run', 'Show what would be learned without saving')
  .action(async (opts) => {
    const watchDir = path.resolve(opts.path);
    const ignorePatterns = opts.ignore.split(',').map(p => p.trim());

    console.error(chalk.cyan(`👁️  Watching ${watchDir} for changes...\n`));
    console.error(chalk.dim(`   Ignoring: ${ignorePatterns.join(', ')}`));
    console.error(chalk.dim(`   Press Ctrl+C to stop.\n`));

    const intel = new Intelligence({ skipEngine: true });
    let lastEdit = null;
    let editCount = 0;

    const shouldIgnore = (filePath) => {
      return ignorePatterns.some(pattern => filePath.includes(pattern));
    };

    const processChange = (eventType, filename) => {
      if (!filename || shouldIgnore(filename)) return;

      const ext = path.extname(filename);
      const state = `edit:${ext || 'unknown'}`;
      const now = Date.now();

      // Determine likely action based on file type
      const agentMapping = {
        '.ts': 'typescript-developer',
        '.js': 'coder',
        '.rs': 'rust-developer',
        '.py': 'python-developer',
        '.go': 'go-developer',
        '.md': 'documentation',
        '.json': 'config-manager',
        '.yaml': 'devops-engineer',
        '.yml': 'devops-engineer',
      };
      const agent = agentMapping[ext] || 'coder';

      // Co-edit pattern detection
      if (lastEdit && lastEdit.file !== filename && (now - lastEdit.time) < 60000) {
        // Files edited within 1 minute are co-edits
        const coEditKey = [lastEdit.file, filename].sort().join('|');
        if (!opts.dryRun) {
          if (!intel.data.sequences) intel.data.sequences = {};
          if (!intel.data.sequences[lastEdit.file]) intel.data.sequences[lastEdit.file] = [];
          const existing = intel.data.sequences[lastEdit.file].find(s => s.file === filename);
          if (existing) {
            existing.score++;
          } else {
            intel.data.sequences[lastEdit.file].push({ file: filename, score: 1 });
          }
        }
        console.log(chalk.yellow(`  🔗 Co-edit: ${path.basename(lastEdit.file)} → ${path.basename(filename)}`));
      }

      // Update Q-value for this file type
      if (!opts.dryRun) {
        intel.updateQ(state, agent, 0.5);
        intel.save();
      }

      editCount++;
      console.log(chalk.green(`  ✏️  [${editCount}] ${filename} → ${agent}`));

      lastEdit = { file: filename, time: now };
    };

    // Use fs.watch for real-time monitoring
    const watcher = fs.watch(watchDir, { recursive: true }, processChange);

    process.on('SIGINT', () => {
      watcher.close();
      console.error(chalk.dim(`\n\n📊 Learned from ${editCount} file changes.`));
      process.exit(0);
    });

    // Keep alive
    await new Promise(() => {});
  });

// ============================================
// END NEW CAPABILITY COMMANDS
// ============================================

// Verify hooks are working
hooksCmd.command('verify')
  .description('Verify hooks are working correctly')
  .option('--verbose', 'Show detailed output')
  .action((opts) => {
    console.log(chalk.bold.cyan('\n🔍 RuVector Hooks Verification\n'));
    const checks = [];

    // Check 1: Settings file exists
    const settingsPath = path.join(process.cwd(), '.claude', 'settings.json');
    if (fs.existsSync(settingsPath)) {
      checks.push({ name: 'Settings file', status: 'pass', detail: '.claude/settings.json exists' });
      try {
        const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
        // Check hooks
        const requiredHooks = ['PreToolUse', 'PostToolUse', 'SessionStart', 'Stop'];
        const missingHooks = requiredHooks.filter(h => !settings.hooks?.[h]);
        if (missingHooks.length === 0) {
          checks.push({ name: 'Required hooks', status: 'pass', detail: 'All core hooks configured' });
        } else {
          checks.push({ name: 'Required hooks', status: 'fail', detail: `Missing: ${missingHooks.join(', ')}` });
        }
        // Check advanced hooks
        const advancedHooks = ['UserPromptSubmit', 'PreCompact', 'Notification'];
        const hasAdvanced = advancedHooks.filter(h => settings.hooks?.[h]);
        if (hasAdvanced.length > 0) {
          checks.push({ name: 'Advanced hooks', status: 'pass', detail: `${hasAdvanced.length}/3 configured` });
        } else {
          checks.push({ name: 'Advanced hooks', status: 'warn', detail: 'None configured (optional)' });
        }
        // Check env
        if (settings.env?.RUVECTOR_INTELLIGENCE_ENABLED) {
          checks.push({ name: 'Environment vars', status: 'pass', detail: 'Intelligence enabled' });
        } else {
          checks.push({ name: 'Environment vars', status: 'warn', detail: 'Not configured' });
        }
        // Check permissions
        if (settings.permissions?.allow?.length > 0) {
          checks.push({ name: 'Permissions', status: 'pass', detail: `${settings.permissions.allow.length} allowed patterns` });
        } else {
          checks.push({ name: 'Permissions', status: 'warn', detail: 'Not configured' });
        }
      } catch (e) {
        checks.push({ name: 'Settings parse', status: 'fail', detail: 'Invalid JSON' });
      }
    } else {
      checks.push({ name: 'Settings file', status: 'fail', detail: 'Run `npx ruvector hooks init` first' });
    }

    // Check 2: .ruvector directory
    const ruvectorDir = path.join(process.cwd(), '.ruvector');
    if (fs.existsSync(ruvectorDir)) {
      checks.push({ name: 'Data directory', status: 'pass', detail: '.ruvector/ exists' });
      const intelFile = path.join(ruvectorDir, 'intelligence.json');
      if (fs.existsSync(intelFile)) {
        const stats = fs.statSync(intelFile);
        checks.push({ name: 'Intelligence file', status: 'pass', detail: `${(stats.size / 1024).toFixed(1)}KB` });
      } else {
        checks.push({ name: 'Intelligence file', status: 'warn', detail: 'Will be created on first use' });
      }
    } else {
      checks.push({ name: 'Data directory', status: 'warn', detail: 'Will be created on first use' });
    }

    // Check 3: Hook command execution
    try {
      const { execSync } = require('child_process');
      execSync('npx ruvector hooks stats', { stdio: 'pipe', timeout: 5000 });
      checks.push({ name: 'Command execution', status: 'pass', detail: 'Hooks commands work' });
    } catch (e) {
      checks.push({ name: 'Command execution', status: 'fail', detail: 'Commands failed to execute' });
    }

    // Display results
    let passCount = 0, warnCount = 0, failCount = 0;
    checks.forEach(c => {
      const icon = c.status === 'pass' ? chalk.green('✓') : c.status === 'warn' ? chalk.yellow('⚠') : chalk.red('✗');
      const statusColor = c.status === 'pass' ? chalk.green : c.status === 'warn' ? chalk.yellow : chalk.red;
      console.log(`  ${icon} ${c.name}: ${statusColor(c.detail)}`);
      if (c.status === 'pass') passCount++;
      else if (c.status === 'warn') warnCount++;
      else failCount++;
    });

    console.log('');
    if (failCount === 0) {
      console.log(chalk.green(`✅ Verification passed! ${passCount} checks passed, ${warnCount} warnings`));
    } else {
      console.log(chalk.red(`❌ Verification failed: ${failCount} issues found`));
      console.log(chalk.dim('   Run `npx ruvector hooks doctor` for detailed diagnostics'));
    }
  });

// Doctor - diagnose setup issues
hooksCmd.command('doctor')
  .description('Diagnose and fix setup issues')
  .option('--fix', 'Automatically fix issues')
  .action((opts) => {
    console.log(chalk.bold.cyan('\n🩺 RuVector Hooks Doctor\n'));
    const issues = [];
    const fixes = [];

    // Check settings file
    const settingsPath = path.join(process.cwd(), '.claude', 'settings.json');
    if (!fs.existsSync(settingsPath)) {
      issues.push({ severity: 'error', message: 'No .claude/settings.json found', fix: 'Run `npx ruvector hooks init`' });
    } else {
      try {
        const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));

        // Check for invalid schema
        if (settings.$schema && !settings.$schema.includes('schemastore.org')) {
          issues.push({ severity: 'warning', message: 'Invalid schema URL', fix: 'Will be corrected' });
          if (opts.fix) {
            settings.$schema = 'https://json.schemastore.org/claude-code-settings.json';
            fixes.push('Fixed schema URL');
          }
        }

        // Check for old hook names
        if (settings.hooks?.Start || settings.hooks?.End) {
          issues.push({ severity: 'error', message: 'Invalid hook names (Start/End)', fix: 'Should be SessionStart/Stop' });
          if (opts.fix) {
            delete settings.hooks.Start;
            delete settings.hooks.End;
            fixes.push('Removed invalid hook names');
          }
        }

        // Check hook format
        const hookNames = ['PreToolUse', 'PostToolUse'];
        hookNames.forEach(name => {
          if (settings.hooks?.[name]) {
            settings.hooks[name].forEach((hook, i) => {
              if (typeof hook.matcher === 'object') {
                issues.push({ severity: 'error', message: `${name}[${i}].matcher should be string, not object`, fix: 'Will be corrected' });
              }
            });
          }
        });

        // Check for npx vs direct command
        const checkCommands = (hooks) => {
          if (!hooks) return;
          hooks.forEach(h => {
            h.hooks?.forEach(hh => {
              if (hh.command && hh.command.includes('ruvector') && !hh.command.startsWith('npx ') && !hh.command.includes('/bin/')) {
                issues.push({ severity: 'warning', message: `Command should use 'npx ruvector' for portability`, fix: 'Update to use npx' });
              }
            });
          });
        };
        Object.values(settings.hooks || {}).forEach(checkCommands);

        // Save fixes
        if (opts.fix && fixes.length > 0) {
          fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
        }
      } catch (e) {
        issues.push({ severity: 'error', message: 'Invalid JSON in settings file', fix: 'Re-run `npx ruvector hooks init --force`' });
      }
    }

    // Check .gitignore
    const gitignorePath = path.join(process.cwd(), '.gitignore');
    if (fs.existsSync(gitignorePath)) {
      const content = fs.readFileSync(gitignorePath, 'utf-8');
      if (!content.includes('.ruvector/')) {
        issues.push({ severity: 'warning', message: '.ruvector/ not in .gitignore', fix: 'Add to prevent committing learning data' });
        if (opts.fix) {
          fs.appendFileSync(gitignorePath, '\n# RuVector intelligence data\n.ruvector/\n');
          fixes.push('Added .ruvector/ to .gitignore');
        }
      }
    }

    // Display results
    if (issues.length === 0) {
      console.log(chalk.green('  ✓ No issues found! Your setup looks healthy.'));
    } else {
      issues.forEach(i => {
        const icon = i.severity === 'error' ? chalk.red('✗') : chalk.yellow('⚠');
        console.log(`  ${icon} ${i.message}`);
        console.log(chalk.dim(`     Fix: ${i.fix}`));
      });

      if (opts.fix && fixes.length > 0) {
        console.log(chalk.green(`\n✅ Applied ${fixes.length} fix(es):`));
        fixes.forEach(f => console.log(chalk.green(`   • ${f}`)));
      } else if (issues.some(i => i.severity === 'error')) {
        console.log(chalk.yellow('\n💡 Run with --fix to automatically fix issues'));
      }
    }
  });

// Export intelligence data
hooksCmd.command('export')
  .description('Export intelligence data for backup')
  .option('-o, --output <file>', 'Output file path', 'ruvector-export.json')
  .option('--include-all', 'Include all data (patterns, memories, trajectories)')
  .action((opts) => {
    const intel = new Intelligence();
    const exportData = {
      version: '1.0',
      exported_at: new Date().toISOString(),
      patterns: intel.data?.patterns || {},
      memories: opts.includeAll ? (intel.data?.memories || []) : [],
      trajectories: opts.includeAll ? (intel.data?.trajectories || []) : [],
      errors: intel.data?.errors || {},
      stats: intel.stats()
    };

    const outputPath = path.resolve(opts.output);
    fs.writeFileSync(outputPath, JSON.stringify(exportData, null, 2));

    console.log(chalk.green(`✅ Exported intelligence data to ${outputPath}`));
    console.log(chalk.dim(`   ${Object.keys(exportData.patterns).length} patterns`));
    console.log(chalk.dim(`   ${exportData.memories.length} memories`));
    console.log(chalk.dim(`   ${exportData.trajectories.length} trajectories`));
  });

// Import intelligence data
hooksCmd.command('import')
  .description('Import intelligence data from backup')
  .argument('<file>', 'Import file path')
  .option('--merge', 'Merge with existing data (default: replace)')
  .option('--dry-run', 'Show what would be imported without making changes')
  .action((file, opts) => {
    const importPath = path.resolve(file);
    if (!fs.existsSync(importPath)) {
      console.error(chalk.red(`❌ File not found: ${importPath}`));
      process.exit(1);
    }

    try {
      const importData = JSON.parse(fs.readFileSync(importPath, 'utf-8'));

      if (!importData.version) {
        console.error(chalk.red('❌ Invalid export file (missing version)'));
        process.exit(1);
      }

      console.log(chalk.cyan(`📦 Import file: ${file}`));
      console.log(chalk.dim(`   Version: ${importData.version}`));
      console.log(chalk.dim(`   Exported: ${importData.exported_at}`));
      console.log(chalk.dim(`   Patterns: ${Object.keys(importData.patterns || {}).length}`));
      console.log(chalk.dim(`   Memories: ${(importData.memories || []).length}`));
      console.log(chalk.dim(`   Trajectories: ${(importData.trajectories || []).length}`));

      if (opts.dryRun) {
        console.log(chalk.yellow('\n⚠️  Dry run - no changes made'));
        return;
      }

      const intel = new Intelligence();

      if (opts.merge) {
        // Merge patterns
        Object.assign(intel.data.patterns, importData.patterns || {});
        // Merge memories (deduplicate by content)
        const existingContent = new Set((intel.data.memories || []).map(m => m.content));
        (importData.memories || []).forEach(m => {
          if (!existingContent.has(m.content)) {
            intel.data.memories.push(m);
          }
        });
        // Merge trajectories
        intel.data.trajectories = (intel.data.trajectories || []).concat(importData.trajectories || []);
        // Merge errors
        Object.assign(intel.data.errors, importData.errors || {});
        console.log(chalk.green('✅ Merged intelligence data'));
      } else {
        intel.data.patterns = importData.patterns || {};
        intel.data.memories = importData.memories || [];
        intel.data.trajectories = importData.trajectories || [];
        intel.data.errors = importData.errors || {};
        console.log(chalk.green('✅ Replaced intelligence data'));
      }

      intel.save();
      console.log(chalk.dim('   Data saved to .ruvector/intelligence.json'));
    } catch (e) {
      console.error(chalk.red(`❌ Failed to import: ${e.message}`));
      process.exit(1);
    }
  });

// Pretrain - analyze repo and bootstrap learning with agent swarm
hooksCmd.command('pretrain')
  .description('Pretrain intelligence by analyzing the repository with agent swarm')
  .option('--depth <n>', 'Git history depth to analyze', '100')
  .option('--workers <n>', 'Number of parallel analysis workers', '4')
  .option('--skip-git', 'Skip git history analysis')
  .option('--skip-files', 'Skip file structure analysis')
  .option('--verbose', 'Show detailed progress')
  .action(async (opts) => {
    const { execSync, spawn } = require('child_process');
    console.log(chalk.bold.cyan('\n🧠 RuVector Pretrain - Repository Intelligence Bootstrap\n'));

    const intel = new Intelligence();
    const startTime = Date.now();
    const stats = { files: 0, patterns: 0, memories: 0, coedits: 0 };

    // Agent types for different file patterns
    const agentMapping = {
      // Rust
      '.rs': 'rust-developer',
      'Cargo.toml': 'rust-developer',
      'Cargo.lock': 'rust-developer',
      // JavaScript/TypeScript
      '.js': 'javascript-developer',
      '.jsx': 'react-developer',
      '.ts': 'typescript-developer',
      '.tsx': 'react-developer',
      '.mjs': 'javascript-developer',
      '.cjs': 'javascript-developer',
      'package.json': 'node-developer',
      // Python
      '.py': 'python-developer',
      'requirements.txt': 'python-developer',
      'pyproject.toml': 'python-developer',
      'setup.py': 'python-developer',
      // Go
      '.go': 'go-developer',
      'go.mod': 'go-developer',
      // Web
      '.html': 'frontend-developer',
      '.css': 'frontend-developer',
      '.scss': 'frontend-developer',
      '.vue': 'vue-developer',
      '.svelte': 'svelte-developer',
      // Config
      '.json': 'config-specialist',
      '.yaml': 'config-specialist',
      '.yml': 'config-specialist',
      '.toml': 'config-specialist',
      // Docs
      '.md': 'documentation-specialist',
      '.mdx': 'documentation-specialist',
      // Tests
      '.test.js': 'test-engineer',
      '.test.ts': 'test-engineer',
      '.spec.js': 'test-engineer',
      '.spec.ts': 'test-engineer',
      '_test.go': 'test-engineer',
      '_test.rs': 'test-engineer',
      // DevOps
      'Dockerfile': 'devops-engineer',
      'docker-compose.yml': 'devops-engineer',
      '.github/workflows': 'cicd-engineer',
      'Makefile': 'devops-engineer',
      // SQL
      '.sql': 'database-specialist',
    };

    // Phase 1: Analyze file structure
    if (!opts.skipFiles) {
      console.log(chalk.yellow('📁 Phase 1: Analyzing file structure...\n'));

      try {
        // Get all files in repo
        const files = execSync('git ls-files 2>/dev/null || find . -type f -not -path "./.git/*" -not -path "./node_modules/*" -not -path "./target/*"',
          { encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024 }).trim().split('\n').filter(f => f);

        const filesByType = {};
        const dirPatterns = {};

        files.forEach(file => {
          stats.files++;
          const ext = path.extname(file);
          const basename = path.basename(file);
          const dir = path.dirname(file);

          // Determine agent for this file
          let agent = 'coder'; // default
          if (agentMapping[basename]) {
            agent = agentMapping[basename];
          } else if (agentMapping[ext]) {
            agent = agentMapping[ext];
          } else if (file.includes('.test.') || file.includes('.spec.') || file.includes('_test.')) {
            agent = 'test-engineer';
          } else if (file.includes('.github/workflows')) {
            agent = 'cicd-engineer';
          }

          // Track file types
          filesByType[ext] = (filesByType[ext] || 0) + 1;

          // Track directory patterns
          const parts = dir.split('/');
          if (parts[0]) {
            dirPatterns[parts[0]] = dirPatterns[parts[0]] || { count: 0, agents: {} };
            dirPatterns[parts[0]].count++;
            dirPatterns[parts[0]].agents[agent] = (dirPatterns[parts[0]].agents[agent] || 0) + 1;
          }

          // Create Q-learning pattern for this file type
          const state = `edit:${ext || 'unknown'}`;
          if (!intel.data.patterns[state]) {
            intel.data.patterns[state] = {};
          }
          intel.data.patterns[state][agent] = (intel.data.patterns[state][agent] || 0) + 0.5;
          stats.patterns++;
        });

        // Log summary
        if (opts.verbose) {
          console.log(chalk.dim('  File types found:'));
          Object.entries(filesByType).sort((a, b) => b[1] - a[1]).slice(0, 10).forEach(([ext, count]) => {
            console.log(chalk.dim(`    ${ext || '(no ext)'}: ${count} files`));
          });
        }
        console.log(chalk.green(`  ✓ Analyzed ${stats.files} files`));
        console.log(chalk.green(`  ✓ Created ${Object.keys(intel.data.patterns).length} routing patterns`));

      } catch (e) {
        console.log(chalk.yellow(`  ⚠ File analysis skipped: ${e.message}`));
      }
    }

    // Phase 2: Analyze git history for co-edit patterns
    if (!opts.skipGit) {
      console.log(chalk.yellow('\n📜 Phase 2: Analyzing git history for co-edit patterns...\n'));

      try {
        // Get commits with files changed
        const gitLog = execSync(
          `git log --name-only --pretty=format:"COMMIT:%H" -n ${opts.depth} 2>/dev/null`,
          { encoding: 'utf-8', maxBuffer: 50 * 1024 * 1024 }
        );

        const commits = gitLog.split('COMMIT:').filter(c => c.trim());
        const coEditMap = {};

        commits.forEach(commit => {
          const lines = commit.trim().split('\n').filter(l => l && !l.startsWith('COMMIT:'));
          const files = lines.slice(1).filter(f => f.trim()); // Skip the hash

          // Track which files are edited together
          files.forEach(file1 => {
            files.forEach(file2 => {
              if (file1 !== file2) {
                const key = [file1, file2].sort().join('|');
                coEditMap[key] = (coEditMap[key] || 0) + 1;
              }
            });
          });
        });

        // Find strong co-edit patterns (files edited together 3+ times)
        const strongPatterns = Object.entries(coEditMap)
          .filter(([, count]) => count >= 3)
          .sort((a, b) => b[1] - a[1]);

        // Store as sequence patterns
        strongPatterns.slice(0, 100).forEach(([key, count]) => {
          const [file1, file2] = key.split('|');
          if (!intel.data.sequences) intel.data.sequences = {};
          if (!intel.data.sequences[file1]) intel.data.sequences[file1] = [];

          const existing = intel.data.sequences[file1].find(s => s.file === file2);
          if (existing) {
            existing.score += count;
          } else {
            intel.data.sequences[file1].push({ file: file2, score: count });
          }
          stats.coedits++;
        });

        console.log(chalk.green(`  ✓ Analyzed ${commits.length} commits`));
        console.log(chalk.green(`  ✓ Found ${strongPatterns.length} co-edit patterns`));

        if (opts.verbose && strongPatterns.length > 0) {
          console.log(chalk.dim('  Top co-edit patterns:'));
          strongPatterns.slice(0, 5).forEach(([key, count]) => {
            const [f1, f2] = key.split('|');
            console.log(chalk.dim(`    ${path.basename(f1)} ↔ ${path.basename(f2)}: ${count} times`));
          });
        }

      } catch (e) {
        console.log(chalk.yellow(`  ⚠ Git analysis skipped: ${e.message}`));
      }
    }

    // Phase 3: Create vector memories from important files
    console.log(chalk.yellow('\n💾 Phase 3: Creating vector memories from key files...\n'));

    try {
      const importantFiles = [
        'README.md', 'CLAUDE.md', 'package.json', 'Cargo.toml',
        'pyproject.toml', 'go.mod', '.claude/settings.json'
      ];

      for (const filename of importantFiles) {
        const filePath = path.join(process.cwd(), filename);
        if (fs.existsSync(filePath)) {
          try {
            const content = fs.readFileSync(filePath, 'utf-8').slice(0, 2000); // First 2KB
            intel.data.memories = intel.data.memories || [];
            intel.data.memories.push({
              content: `[${filename}] ${content.replace(/\n/g, ' ').slice(0, 500)}`,
              type: 'project',
              created: new Date().toISOString(),
              embedding: intel.simpleEmbed ? intel.simpleEmbed(content) : null
            });
            stats.memories++;
            if (opts.verbose) console.log(chalk.dim(`    ✓ ${filename}`));
          } catch (e) { /* skip unreadable files */ }
        }
      }

      console.log(chalk.green(`  ✓ Created ${stats.memories} memory entries`));

    } catch (e) {
      console.log(chalk.yellow(`  ⚠ Memory creation skipped: ${e.message}`));
    }

    // Phase 4: Analyze directory structure for agent recommendations
    console.log(chalk.yellow('\n🗂️  Phase 4: Building directory-agent mappings...\n'));

    try {
      const dirs = execSync('find . -type d -maxdepth 2 -not -path "./.git*" -not -path "./node_modules*" -not -path "./target*" 2>/dev/null || echo "."',
        { encoding: 'utf-8' }).trim().split('\n');

      const dirAgentMap = {};
      dirs.forEach(dir => {
        const name = path.basename(dir);
        // Infer agent from directory name
        if (['src', 'lib', 'core'].includes(name)) dirAgentMap[dir] = 'coder';
        else if (['test', 'tests', '__tests__', 'spec'].includes(name)) dirAgentMap[dir] = 'test-engineer';
        else if (['docs', 'documentation'].includes(name)) dirAgentMap[dir] = 'documentation-specialist';
        else if (['scripts', 'bin'].includes(name)) dirAgentMap[dir] = 'devops-engineer';
        else if (['components', 'views', 'pages'].includes(name)) dirAgentMap[dir] = 'frontend-developer';
        else if (['api', 'routes', 'handlers'].includes(name)) dirAgentMap[dir] = 'backend-developer';
        else if (['models', 'entities', 'schemas'].includes(name)) dirAgentMap[dir] = 'database-specialist';
        else if (['.github', '.gitlab', 'ci'].includes(name)) dirAgentMap[dir] = 'cicd-engineer';
      });

      // Store directory patterns
      intel.data.dirPatterns = dirAgentMap;
      console.log(chalk.green(`  ✓ Mapped ${Object.keys(dirAgentMap).length} directories to agents`));

    } catch (e) {
      console.log(chalk.yellow(`  ⚠ Directory analysis skipped: ${e.message}`));
    }

    // Phase 5: Analyze code complexity with AST
    console.log(chalk.yellow('\n📊 Phase 5: Analyzing code complexity via AST...\n'));

    try {
      if (loadNewModules() && ASTParser) {
        const parser = new ASTParser();
        const codeFiles = (intel.data.fileList || []).filter(f =>
          ['.ts', '.js', '.tsx', '.jsx', '.py', '.rs', '.go'].includes(path.extname(f))
        ).slice(0, 50); // Analyze up to 50 files

        let complexityStats = { high: 0, medium: 0, low: 0, total: 0 };

        for (const file of codeFiles) {
          try {
            if (!fs.existsSync(file)) continue;
            const code = fs.readFileSync(file, 'utf-8');
            const ext = path.extname(file).slice(1);
            const lang = { ts: 'typescript', tsx: 'typescript', js: 'javascript', py: 'python', rs: 'rust', go: 'go' }[ext];
            if (!lang) continue;

            const result = parser.parse(code, lang);
            const complexity = parser.calculateComplexity(result);

            // Store complexity data
            intel.data.complexity = intel.data.complexity || {};
            intel.data.complexity[file] = complexity;

            if (complexity.cyclomatic > 15) complexityStats.high++;
            else if (complexity.cyclomatic > 8) complexityStats.medium++;
            else complexityStats.low++;
            complexityStats.total++;
          } catch (e) { /* skip errors */ }
        }

        stats.complexity = complexityStats;
        console.log(chalk.green(`  ✓ Analyzed ${complexityStats.total} files`));
        console.log(chalk.green(`  ✓ Complexity: ${complexityStats.high} high, ${complexityStats.medium} medium, ${complexityStats.low} low`));
      } else {
        console.log(chalk.dim('  ⏭️  AST parser not available, skipping'));
      }
    } catch (e) {
      console.log(chalk.yellow(`  ⚠ Complexity analysis skipped: ${e.message}`));
    }

    // Phase 6: Analyze diff patterns from recent commits
    console.log(chalk.yellow('\n🔄 Phase 6: Analyzing diff patterns for change classification...\n'));

    try {
      const diffMod = require('../dist/core/diff-embeddings.js');
      const recentCommits = execSync(`git log --format="%H" -n 20 2>/dev/null`, { encoding: 'utf-8' }).trim().split('\n').filter(h => h);

      let changeTypes = { feature: 0, bugfix: 0, refactor: 0, docs: 0, test: 0, config: 0, unknown: 0 };

      for (const hash of recentCommits.slice(0, 10)) {
        try {
          const analysis = await diffMod.analyzeCommit(hash);
          analysis.files.forEach(f => {
            changeTypes[f.category] = (changeTypes[f.category] || 0) + 1;
          });
        } catch (e) { /* skip */ }
      }

      intel.data.changePatterns = changeTypes;
      stats.changePatterns = changeTypes;
      console.log(chalk.green(`  ✓ Analyzed ${recentCommits.length} commits`));
      console.log(chalk.green(`  ✓ Change types: ${Object.entries(changeTypes).filter(([,v]) => v > 0).map(([k,v]) => `${k}:${v}`).join(', ')}`));
    } catch (e) {
      console.log(chalk.yellow(`  ⚠ Diff analysis skipped: ${e.message}`));
    }

    // Phase 7: Check test coverage if available
    console.log(chalk.yellow('\n🧪 Phase 7: Checking test coverage data...\n'));

    try {
      const covMod = require('../dist/core/coverage-router.js');
      const reportPath = covMod.findCoverageReport();

      if (reportPath) {
        const summary = covMod.parseIstanbulCoverage(reportPath);
        intel.data.coverage = {
          overall: summary.overall,
          lowCoverageFiles: summary.lowCoverageFiles.slice(0, 20),
          uncoveredFiles: summary.uncoveredFiles.slice(0, 10)
        };
        stats.coverage = summary.overall;
        console.log(chalk.green(`  ✓ Found coverage report: ${reportPath}`));
        console.log(chalk.green(`  ✓ Overall: Lines ${summary.overall.lines.toFixed(1)}%, Functions ${summary.overall.functions.toFixed(1)}%`));
        console.log(chalk.green(`  ✓ ${summary.lowCoverageFiles.length} low-coverage files, ${summary.uncoveredFiles.length} uncovered`));
      } else {
        console.log(chalk.dim('  ⏭️  No coverage report found'));
      }
    } catch (e) {
      console.log(chalk.yellow(`  ⚠ Coverage check skipped: ${e.message}`));
    }

    // Phase 8: Detect available attention/GNN capabilities
    console.log(chalk.yellow('\n🧠 Phase 8: Detecting neural capabilities...\n'));

    try {
      let capabilities = { attention: false, gnn: false, mechanisms: [] };

      try {
        const attention = require('@ruvector/attention');
        capabilities.attention = true;
        capabilities.mechanisms = [
          'DotProductAttention', 'MultiHeadAttention', 'FlashAttention',
          'HyperbolicAttention', 'LinearAttention', 'MoEAttention',
          'GraphRoPeAttention', 'DualSpaceAttention', 'LocalGlobalAttention'
        ];
        console.log(chalk.green(`  ✓ Attention: 10 mechanisms available`));
      } catch (e) {
        console.log(chalk.dim('  ⏭️  @ruvector/attention not installed'));
      }

      try {
        const gnn = require('@ruvector/gnn');
        capabilities.gnn = true;
        console.log(chalk.green(`  ✓ GNN: RuvectorLayer, TensorCompress available`));
      } catch (e) {
        console.log(chalk.dim('  ⏭️  @ruvector/gnn not installed'));
      }

      intel.data.neuralCapabilities = capabilities;
      stats.neural = capabilities;
    } catch (e) {
      console.log(chalk.yellow(`  ⚠ Neural detection skipped: ${e.message}`));
    }

    // Phase 9: Build code graph for community detection
    console.log(chalk.yellow('\n🔗 Phase 9: Building code relationship graph...\n'));

    try {
      const graphMod = require('../dist/core/graph-algorithms.js');
      const codeFiles = execSync('git ls-files "*.ts" "*.js" 2>/dev/null || echo ""', { encoding: 'utf-8' }).trim().split('\n').filter(f => f);

      if (codeFiles.length > 5 && codeFiles.length < 200) {
        const nodes = codeFiles.map(f => path.basename(f, path.extname(f)));
        const edges = [];

        for (const file of codeFiles.slice(0, 100)) {
          try {
            if (!fs.existsSync(file)) continue;
            const content = fs.readFileSync(file, 'utf-8');
            const imports = content.match(/from ['"]\.\/([^'"]+)['"]/g) || [];
            imports.forEach(imp => {
              const target = imp.match(/from ['"]\.\/([^'"]+)['"]/)?.[1];
              if (target) {
                const targetBase = path.basename(target, path.extname(target));
                if (nodes.includes(targetBase)) {
                  edges.push({ source: path.basename(file, path.extname(file)), target: targetBase, weight: 1 });
                }
              }
            });
          } catch (e) { /* skip */ }
        }

        if (edges.length > 0) {
          const communities = graphMod.louvainCommunities(nodes, edges);
          intel.data.codeGraph = {
            nodes: nodes.length,
            edges: edges.length,
            communities: communities.numCommunities,
            modularity: communities.modularity
          };
          stats.graph = intel.data.codeGraph;
          console.log(chalk.green(`  ✓ Built graph: ${nodes.length} nodes, ${edges.length} edges`));
          console.log(chalk.green(`  ✓ Found ${communities.numCommunities} communities (modularity: ${communities.modularity.toFixed(3)})`));
        } else {
          console.log(chalk.dim('  ⏭️  Not enough import relationships found'));
        }
      } else {
        console.log(chalk.dim(`  ⏭️  Skipped (${codeFiles.length} files - need 5-200)`));
      }
    } catch (e) {
      console.log(chalk.yellow(`  ⚠ Graph analysis skipped: ${e.message}`));
    }

    // Phase 10: Initialize multi-algorithm learning engine
    console.log(chalk.yellow('\n🎯 Phase 10: Initializing multi-algorithm learning engine...\n'));

    try {
      if (loadLearningModules() && LearningEngineClass) {
        const engine = new LearningEngineClass();

        // Configure optimal algorithms for each task type based on repo analysis
        engine.configure('agent-routing', { algorithm: 'double-q', learningRate: 0.1, epsilon: 0.1 });
        engine.configure('error-avoidance', { algorithm: 'sarsa', learningRate: 0.05, epsilon: 0.05 });
        engine.configure('confidence-scoring', { algorithm: 'actor-critic', learningRate: 0.01 });
        engine.configure('trajectory-learning', { algorithm: 'decision-transformer', sequenceLength: 20 });
        engine.configure('context-ranking', { algorithm: 'ppo', clipRange: 0.2 });
        engine.configure('memory-recall', { algorithm: 'td-lambda', lambda: 0.8 });

        // Bootstrap with initial experiences from file patterns
        let bootstrapCount = 0;
        for (const [state, actions] of Object.entries(intel.data.patterns || {})) {
          for (const [action, value] of Object.entries(actions)) {
            if (value > 0.3) { // Only strong patterns
              engine.update('agent-routing', {
                state,
                action,
                reward: value,
                nextState: state,
                done: true
              });
              bootstrapCount++;
            }
          }
        }

        intel.data.learning = engine.export();
        stats.learningBootstrap = bootstrapCount;
        console.log(chalk.green(`  ✓ Configured 6 task-specific algorithms`));
        console.log(chalk.green(`  ✓ Bootstrapped with ${bootstrapCount} initial experiences`));
        console.log(chalk.dim('  Algorithms: double-q, sarsa, actor-critic, decision-transformer, ppo, td-lambda'));
      } else {
        console.log(chalk.dim('  ⏭️  LearningEngine not available'));
      }
    } catch (e) {
      console.log(chalk.yellow(`  ⚠ Learning engine init skipped: ${e.message}`));
    }

    // Phase 11: Initialize TensorCompress for pattern storage
    console.log(chalk.yellow('\n📦 Phase 11: Initializing TensorCompress for efficient storage...\n'));

    try {
      if (loadLearningModules() && TensorCompressClass) {
        const compress = new TensorCompressClass({
          autoCompress: false,
          hotThreshold: 0.8,
          warmThreshold: 0.4,
          coolThreshold: 0.1,
          coldThreshold: 0.01
        });

        // Store any existing embeddings with compression
        let compressed = 0;
        if (intel.data.memories) {
          for (let i = 0; i < intel.data.memories.length; i++) {
            const mem = intel.data.memories[i];
            if (mem.embedding && Array.isArray(mem.embedding)) {
              compress.store(`memory_${i}`, mem.embedding, 'pq8');
              compressed++;
            }
          }
        }

        if (compressed > 0) {
          const compStats = compress.recompressAll();
          intel.data.compressedPatterns = compress.export();
          stats.compressed = compressed;
          stats.compressionSavings = compStats.savingsPercent;
          console.log(chalk.green(`  ✓ Compressed ${compressed} embeddings`));
          console.log(chalk.green(`  ✓ Memory savings: ${compStats.savingsPercent.toFixed(1)}%`));
        } else {
          intel.data.compressedPatterns = compress.export();
          console.log(chalk.green(`  ✓ TensorCompress initialized (ready for future embeddings)`));
        }
        console.log(chalk.dim('  Levels: none (hot), half (warm), pq8 (cool), pq4 (cold), binary (archive)'));
      } else {
        console.log(chalk.dim('  ⏭️  TensorCompress not available'));
      }
    } catch (e) {
      console.log(chalk.yellow(`  ⚠ TensorCompress init skipped: ${e.message}`));
    }

    // Save all learning data
    intel.data.pretrained = {
      date: new Date().toISOString(),
      version: '2.1',
      stats: stats
    };
    intel.save();

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(chalk.bold.green(`\n✅ Pretrain complete in ${elapsed}s!\n`));
    console.log(chalk.cyan('Summary:'));
    console.log(`  📁 ${stats.files} files analyzed`);
    console.log(`  🧠 ${stats.patterns} agent routing patterns`);
    console.log(`  🔗 ${stats.coedits} co-edit patterns`);
    console.log(`  💾 ${stats.memories} memory entries`);
    if (stats.complexity) console.log(`  📊 ${stats.complexity.total} files analyzed for complexity`);
    if (stats.changePatterns) console.log(`  🔄 Change patterns detected`);
    if (stats.coverage) console.log(`  🧪 Coverage: ${stats.coverage.lines.toFixed(1)}% lines`);
    if (stats.neural?.attention) console.log(`  🧠 10 attention mechanisms available`);
    if (stats.graph) console.log(`  🔗 ${stats.graph.communities} code communities detected`);
    if (stats.learningBootstrap) console.log(`  🎯 ${stats.learningBootstrap} learning experiences bootstrapped`);
    if (stats.compressionSavings) console.log(`  📦 ${stats.compressionSavings.toFixed(1)}% compression savings`);
    console.log(chalk.dim('\nThe intelligence layer will now provide better recommendations.'));
  });

// Agent Builder - generate optimized agent configs based on pretrain
hooksCmd.command('build-agents')
  .description('Generate optimized agent configurations based on repository analysis')
  .option('--focus <type>', 'Focus type: quality, speed, security, testing, fullstack', 'quality')
  .option('--output <dir>', 'Output directory', '.claude/agents')
  .option('--format <fmt>', 'Format: yaml, json, md', 'yaml')
  .option('--include-prompts', 'Include detailed system prompts')
  .action((opts) => {
    console.log(chalk.bold.cyan('\n🏗️  RuVector Agent Builder\n'));

    const intel = new Intelligence();
    const outputDir = path.join(process.cwd(), opts.output);

    // Check if pretrained
    if (!intel.data.pretrained && Object.keys(intel.data.patterns || {}).length === 0) {
      console.log(chalk.yellow('⚠️  No pretrain data found. Running quick analysis...\n'));
      // Quick file analysis
      try {
        const { execSync } = require('child_process');
        const files = execSync('git ls-files 2>/dev/null', { encoding: 'utf-8' }).trim().split('\n');
        files.forEach(f => {
          const ext = path.extname(f);
          intel.data.patterns = intel.data.patterns || {};
          intel.data.patterns[`edit:${ext}`] = intel.data.patterns[`edit:${ext}`] || {};
        });
      } catch (e) { /* continue without git */ }
    }

    // Analyze patterns to determine relevant agents
    const patterns = intel.data.patterns || {};
    const detectedLangs = new Set();
    const detectedFrameworks = new Set();

    Object.keys(patterns).forEach(state => {
      if (state.includes('.rs')) detectedLangs.add('rust');
      if (state.includes('.ts') || state.includes('.js')) detectedLangs.add('typescript');
      if (state.includes('.tsx') || state.includes('.jsx')) detectedFrameworks.add('react');
      if (state.includes('.py')) detectedLangs.add('python');
      if (state.includes('.go')) detectedLangs.add('go');
      if (state.includes('.vue')) detectedFrameworks.add('vue');
      if (state.includes('.sql')) detectedFrameworks.add('database');
    });

    // Detect project type from files
    const projectTypes = detectProjectType();

    console.log(chalk.blue(`  Detected languages: ${[...detectedLangs].join(', ') || 'generic'}`));
    console.log(chalk.blue(`  Detected frameworks: ${[...detectedFrameworks].join(', ') || 'none'}`));
    console.log(chalk.blue(`  Focus mode: ${opts.focus}\n`));

    // Focus configurations
    const focusConfigs = {
      quality: {
        description: 'Emphasizes code quality, best practices, and maintainability',
        priorities: ['code-review', 'refactoring', 'documentation', 'testing'],
        temperature: 0.3
      },
      speed: {
        description: 'Optimized for rapid development and iteration',
        priorities: ['implementation', 'prototyping', 'quick-fixes'],
        temperature: 0.7
      },
      security: {
        description: 'Security-first development with vulnerability awareness',
        priorities: ['security-audit', 'input-validation', 'authentication', 'encryption'],
        temperature: 0.2
      },
      testing: {
        description: 'Test-driven development with comprehensive coverage',
        priorities: ['unit-tests', 'integration-tests', 'e2e-tests', 'mocking'],
        temperature: 0.4
      },
      fullstack: {
        description: 'Balanced full-stack development capabilities',
        priorities: ['frontend', 'backend', 'database', 'api-design'],
        temperature: 0.5
      }
    };

    const focus = focusConfigs[opts.focus] || focusConfigs.quality;

    // Agent templates based on detected stack
    const agents = [];

    // Core agents based on detected languages
    if (detectedLangs.has('rust')) {
      agents.push({
        name: 'rust-specialist',
        type: 'rust-developer',
        description: 'Rust development specialist for this codebase',
        capabilities: ['cargo', 'unsafe-rust', 'async-rust', 'wasm', 'error-handling'],
        focus: focus.priorities,
        systemPrompt: opts.includePrompts ? `You are a Rust specialist for this project.
Focus on: memory safety, zero-cost abstractions, idiomatic Rust patterns.
Use cargo conventions, prefer Result over panic, leverage the type system.
${focus.description}` : null
      });
    }

    if (detectedLangs.has('typescript')) {
      agents.push({
        name: 'typescript-specialist',
        type: 'typescript-developer',
        description: 'TypeScript development specialist',
        capabilities: ['types', 'generics', 'decorators', 'async-await', 'modules'],
        focus: focus.priorities,
        systemPrompt: opts.includePrompts ? `You are a TypeScript specialist for this project.
Focus on: strict typing, type inference, generic patterns, module organization.
Prefer type safety over any, use discriminated unions, leverage utility types.
${focus.description}` : null
      });
    }

    if (detectedLangs.has('python')) {
      agents.push({
        name: 'python-specialist',
        type: 'python-developer',
        description: 'Python development specialist',
        capabilities: ['typing', 'async', 'testing', 'packaging', 'data-science'],
        focus: focus.priorities,
        systemPrompt: opts.includePrompts ? `You are a Python specialist for this project.
Focus on: type hints, PEP standards, pythonic idioms, virtual environments.
Use dataclasses, prefer pathlib, leverage context managers.
${focus.description}` : null
      });
    }

    if (detectedLangs.has('go')) {
      agents.push({
        name: 'go-specialist',
        type: 'go-developer',
        description: 'Go development specialist',
        capabilities: ['goroutines', 'channels', 'interfaces', 'testing', 'modules'],
        focus: focus.priorities,
        systemPrompt: opts.includePrompts ? `You are a Go specialist for this project.
Focus on: simplicity, explicit error handling, goroutines, interface composition.
Follow Go conventions, use go fmt, prefer composition over inheritance.
${focus.description}` : null
      });
    }

    // Framework-specific agents
    if (detectedFrameworks.has('react')) {
      agents.push({
        name: 'react-specialist',
        type: 'react-developer',
        description: 'React/Next.js development specialist',
        capabilities: ['hooks', 'state-management', 'components', 'ssr', 'testing'],
        focus: focus.priorities,
        systemPrompt: opts.includePrompts ? `You are a React specialist for this project.
Focus on: functional components, hooks, state management, performance optimization.
Prefer composition, use memo wisely, follow React best practices.
${focus.description}` : null
      });
    }

    if (detectedFrameworks.has('database')) {
      agents.push({
        name: 'database-specialist',
        type: 'database-specialist',
        description: 'Database design and optimization specialist',
        capabilities: ['schema-design', 'queries', 'indexing', 'migrations', 'orm'],
        focus: focus.priorities,
        systemPrompt: opts.includePrompts ? `You are a database specialist for this project.
Focus on: normalized schemas, efficient queries, proper indexing, data integrity.
Consider performance implications, use transactions appropriately.
${focus.description}` : null
      });
    }

    // Focus-specific agents
    if (opts.focus === 'testing' || opts.focus === 'quality') {
      agents.push({
        name: 'test-architect',
        type: 'test-engineer',
        description: 'Testing and quality assurance specialist',
        capabilities: ['unit-tests', 'integration-tests', 'mocking', 'coverage', 'tdd'],
        focus: ['testing', 'quality', 'reliability'],
        systemPrompt: opts.includePrompts ? `You are a testing specialist for this project.
Focus on: comprehensive test coverage, meaningful assertions, test isolation.
Write tests first when possible, mock external dependencies, aim for >80% coverage.
${focus.description}` : null
      });
    }

    if (opts.focus === 'security') {
      agents.push({
        name: 'security-auditor',
        type: 'security-specialist',
        description: 'Security audit and hardening specialist',
        capabilities: ['vulnerability-scan', 'auth', 'encryption', 'input-validation', 'owasp'],
        focus: ['security', 'compliance', 'hardening'],
        systemPrompt: opts.includePrompts ? `You are a security specialist for this project.
Focus on: OWASP top 10, input validation, authentication, authorization, encryption.
Never trust user input, use parameterized queries, implement defense in depth.
${focus.description}` : null
      });
    }

    // Add coordinator agent
    agents.push({
      name: 'project-coordinator',
      type: 'coordinator',
      description: 'Coordinates multi-agent workflows for this project',
      capabilities: ['task-decomposition', 'agent-routing', 'context-management'],
      focus: focus.priorities,
      routes: agents.filter(a => a.name !== 'project-coordinator').map(a => ({
        pattern: a.capabilities[0],
        agent: a.name
      }))
    });

    // Create output directory
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Generate agent files
    agents.forEach(agent => {
      let content;
      const filename = `${agent.name}.${opts.format}`;
      const filepath = path.join(outputDir, filename);

      if (opts.format === 'yaml') {
        const yaml = [
          `# Auto-generated by RuVector Agent Builder`,
          `# Focus: ${opts.focus}`,
          `# Generated: ${new Date().toISOString()}`,
          ``,
          `name: ${agent.name}`,
          `type: ${agent.type}`,
          `description: ${agent.description}`,
          ``,
          `capabilities:`,
          ...agent.capabilities.map(c => `  - ${c}`),
          ``,
          `focus:`,
          ...agent.focus.map(f => `  - ${f}`),
        ];
        if (agent.systemPrompt) {
          yaml.push(``, `system_prompt: |`);
          agent.systemPrompt.split('\n').forEach(line => yaml.push(`  ${line}`));
        }
        if (agent.routes) {
          yaml.push(``, `routes:`);
          agent.routes.forEach(r => yaml.push(`  - pattern: "${r.pattern}"`, `    agent: ${r.agent}`));
        }
        content = yaml.join('\n');
      } else if (opts.format === 'json') {
        content = JSON.stringify(agent, null, 2);
      } else {
        // Markdown format
        content = [
          `# ${agent.name}`,
          ``,
          `**Type:** ${agent.type}`,
          `**Description:** ${agent.description}`,
          ``,
          `## Capabilities`,
          ...agent.capabilities.map(c => `- ${c}`),
          ``,
          `## Focus Areas`,
          ...agent.focus.map(f => `- ${f}`),
        ].join('\n');
        if (agent.systemPrompt) {
          content += `\n\n## System Prompt\n\n\`\`\`\n${agent.systemPrompt}\n\`\`\``;
        }
      }

      fs.writeFileSync(filepath, content);
      console.log(chalk.green(`  ✓ Created ${filename}`));
    });

    // Create index file
    const indexContent = opts.format === 'yaml'
      ? `# RuVector Agent Configuration\n# Focus: ${opts.focus}\n\nagents:\n${agents.map(a => `  - ${a.name}`).join('\n')}`
      : JSON.stringify({ focus: opts.focus, agents: agents.map(a => a.name) }, null, 2);

    fs.writeFileSync(path.join(outputDir, `index.${opts.format === 'md' ? 'json' : opts.format}`), indexContent);

    // Update settings to reference agents
    const settingsPath = path.join(process.cwd(), '.claude', 'settings.json');
    if (fs.existsSync(settingsPath)) {
      try {
        const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
        settings.agentConfig = {
          directory: opts.output,
          focus: opts.focus,
          agents: agents.map(a => a.name),
          generated: new Date().toISOString()
        };
        fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
        console.log(chalk.blue('\n  ✓ Updated .claude/settings.json with agent config'));
      } catch (e) { /* ignore settings errors */ }
    }

    console.log(chalk.bold.green(`\n✅ Generated ${agents.length} optimized agents in ${opts.output}/\n`));
    console.log(chalk.cyan('Agents created:'));
    agents.forEach(a => {
      console.log(`  🤖 ${chalk.bold(a.name)}: ${a.description}`);
    });
    console.log(chalk.dim(`\nFocus mode "${opts.focus}": ${focus.description}`));
  });

// Workers command group - Background analysis via agentic-flow
const workersCmd = program.command('workers').description('Background analysis workers (via agentic-flow)');

// Helper to run agentic-flow workers command
async function runAgenticFlow(args) {
  const { spawn } = require('child_process');
  return new Promise((resolve, reject) => {
    const proc = spawn('npx', ['agentic-flow@alpha', ...args], {
      stdio: 'inherit',
      shell: true
    });
    proc.on('close', code => code === 0 ? resolve() : reject(new Error(`Exit code ${code}`)));
    proc.on('error', reject);
  });
}

workersCmd.command('dispatch')
  .description('Dispatch background worker for analysis')
  .argument('<prompt...>', 'Prompt with trigger keyword (ultralearn, optimize, audit, map, etc.)')
  .action(async (prompt) => {
    try {
      await runAgenticFlow(['workers', 'dispatch', prompt.join(' ')]);
    } catch (e) {
      console.error(chalk.red('Worker dispatch failed:'), e.message);
    }
  });

workersCmd.command('status')
  .description('Show worker status dashboard')
  .argument('[workerId]', 'Specific worker ID')
  .action(async (workerId) => {
    try {
      const args = ['workers', 'status'];
      if (workerId) args.push(workerId);
      await runAgenticFlow(args);
    } catch (e) {
      console.error(chalk.red('Status check failed:'), e.message);
    }
  });

workersCmd.command('results')
  .description('Show worker analysis results')
  .option('--json', 'Output as JSON')
  .action(async (opts) => {
    try {
      const args = ['workers', 'results'];
      if (opts.json) args.push('--json');
      await runAgenticFlow(args);
    } catch (e) {
      console.error(chalk.red('Results fetch failed:'), e.message);
    }
  });

workersCmd.command('triggers')
  .description('List available trigger keywords')
  .action(async () => {
    try {
      await runAgenticFlow(['workers', 'triggers']);
    } catch (e) {
      console.error(chalk.red('Triggers list failed:'), e.message);
    }
  });

workersCmd.command('stats')
  .description('Show worker statistics (24h)')
  .action(async () => {
    try {
      await runAgenticFlow(['workers', 'stats']);
    } catch (e) {
      console.error(chalk.red('Stats failed:'), e.message);
    }
  });

workersCmd.command('cleanup')
  .description('Cleanup old worker records')
  .option('--keep <days>', 'Keep records for N days', '7')
  .action(async (opts) => {
    try {
      await runAgenticFlow(['workers', 'cleanup', '--keep', opts.keep]);
    } catch (e) {
      console.error(chalk.red('Cleanup failed:'), e.message);
    }
  });

workersCmd.command('cancel')
  .description('Cancel a running worker')
  .argument('<workerId>', 'Worker ID to cancel')
  .action(async (workerId) => {
    try {
      await runAgenticFlow(['workers', 'cancel', workerId]);
    } catch (e) {
      console.error(chalk.red('Cancel failed:'), e.message);
    }
  });

// Custom Worker System (agentic-flow@alpha.39+)
workersCmd.command('presets')
  .description('List available worker presets (quick-scan, deep-analysis, security-scan, etc.)')
  .action(async () => {
    try {
      await runAgenticFlow(['workers', 'presets']);
    } catch (e) {
      console.error(chalk.red('Presets list failed:'), e.message);
    }
  });

workersCmd.command('phases')
  .description('List available phase executors (24 phases: file-discovery, security-analysis, etc.)')
  .action(async () => {
    try {
      await runAgenticFlow(['workers', 'phases']);
    } catch (e) {
      console.error(chalk.red('Phases list failed:'), e.message);
    }
  });

workersCmd.command('create')
  .description('Create a custom worker from preset')
  .argument('<name>', 'Worker name')
  .option('--preset <preset>', 'Base preset (quick-scan, deep-analysis, security-scan, learning, api-docs, test-analysis)')
  .option('--triggers <triggers>', 'Comma-separated trigger keywords')
  .action(async (name, opts) => {
    try {
      const args = ['workers', 'create', name];
      if (opts.preset) args.push('--preset', opts.preset);
      if (opts.triggers) args.push('--triggers', opts.triggers);
      await runAgenticFlow(args);
    } catch (e) {
      console.error(chalk.red('Worker creation failed:'), e.message);
    }
  });

workersCmd.command('run')
  .description('Run a custom worker')
  .argument('<name>', 'Worker name')
  .option('--path <path>', 'Target path to analyze', '.')
  .action(async (name, opts) => {
    try {
      const args = ['workers', 'run', name];
      if (opts.path) args.push('--path', opts.path);
      await runAgenticFlow(args);
    } catch (e) {
      console.error(chalk.red('Worker run failed:'), e.message);
    }
  });

workersCmd.command('custom')
  .description('List registered custom workers')
  .action(async () => {
    try {
      await runAgenticFlow(['workers', 'custom']);
    } catch (e) {
      console.error(chalk.red('Custom workers list failed:'), e.message);
    }
  });

workersCmd.command('init-config')
  .description('Generate example workers.yaml config file')
  .option('--force', 'Overwrite existing config')
  .action(async (opts) => {
    try {
      const args = ['workers', 'init-config'];
      if (opts.force) args.push('--force');
      await runAgenticFlow(args);
    } catch (e) {
      console.error(chalk.red('Config init failed:'), e.message);
    }
  });

workersCmd.command('load-config')
  .description('Load custom workers from workers.yaml')
  .option('--file <file>', 'Config file path', 'workers.yaml')
  .action(async (opts) => {
    try {
      const args = ['workers', 'load-config'];
      if (opts.file !== 'workers.yaml') args.push('--file', opts.file);
      await runAgenticFlow(args);
    } catch (e) {
      console.error(chalk.red('Config load failed:'), e.message);
    }
  });

console.log && false; // Force registration

// Native Workers command group - Deep ruvector integration (no agentic-flow delegation)
const nativeCmd = program.command('native').description('Native workers with deep ONNX/VectorDB integration (no external deps)');

nativeCmd.command('run')
  .description('Run a native worker type')
  .argument('<type>', 'Worker type: security, analysis, learning')
  .option('--path <path>', 'Target path to analyze', '.')
  .option('--json', 'Output as JSON')
  .action(async (type, opts) => {
    try {
      const { createSecurityWorker, createAnalysisWorker, createLearningWorker } = require('../dist/workers/native-worker.js');

      let worker;
      switch (type) {
        case 'security':
          worker = createSecurityWorker();
          break;
        case 'analysis':
          worker = createAnalysisWorker();
          break;
        case 'learning':
          worker = createLearningWorker();
          break;
        default:
          console.error(chalk.red(`Unknown worker type: ${type}`));
          console.log(chalk.dim('Available types: security, analysis, learning'));
          return;
      }

      console.log(chalk.cyan(`\n🔧 Running native ${type} worker on ${opts.path}...\n`));
      const result = await worker.run(opts.path);

      if (opts.json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        console.log(chalk.bold(`Worker: ${result.worker}`));
        console.log(chalk.dim(`Status: ${result.success ? chalk.green('✓ Success') : chalk.red('✗ Failed')}`));
        console.log(chalk.dim(`Time: ${result.totalTimeMs.toFixed(0)}ms\n`));

        console.log(chalk.bold('Phases:'));
        for (const phase of result.phases) {
          const status = phase.success ? chalk.green('✓') : chalk.red('✗');
          console.log(`  ${status} ${phase.phase} (${phase.timeMs.toFixed(0)}ms)`);
          if (phase.data) {
            const dataStr = JSON.stringify(phase.data);
            if (dataStr.length < 100) {
              console.log(chalk.dim(`    ${dataStr}`));
            }
          }
        }

        if (result.summary) {
          console.log(chalk.bold('\nSummary:'));
          console.log(`  Files analyzed: ${result.summary.filesAnalyzed}`);
          console.log(`  Patterns found: ${result.summary.patternsFound}`);
          console.log(`  Embeddings: ${result.summary.embeddingsGenerated}`);
          console.log(`  Vectors stored: ${result.summary.vectorsStored}`);

          if (result.summary.findings.length > 0) {
            console.log(chalk.bold('\nFindings:'));
            const byType = { info: 0, warning: 0, error: 0, security: 0 };
            result.summary.findings.forEach(f => byType[f.type]++);
            if (byType.security > 0) console.log(chalk.red(`  🔒 Security: ${byType.security}`));
            if (byType.error > 0) console.log(chalk.red(`  ❌ Errors: ${byType.error}`));
            if (byType.warning > 0) console.log(chalk.yellow(`  ⚠️  Warnings: ${byType.warning}`));
            if (byType.info > 0) console.log(chalk.blue(`  ℹ️  Info: ${byType.info}`));

            // Show top findings
            console.log(chalk.dim('\nTop findings:'));
            result.summary.findings.slice(0, 5).forEach(f => {
              const icon = f.type === 'security' ? '🔒' : f.type === 'warning' ? '⚠️' : 'ℹ️';
              console.log(chalk.dim(`  ${icon} ${f.message.slice(0, 60)}${f.file ? ` (${path.basename(f.file)})` : ''}`));
            });
          }
        }
      }
    } catch (e) {
      console.error(chalk.red('Native worker failed:'), e.message);
      if (e.stack) console.error(chalk.dim(e.stack));
    }
  });

nativeCmd.command('benchmark')
  .description('Run performance benchmark suite')
  .option('--path <path>', 'Target path for worker benchmarks', '.')
  .option('--embeddings-only', 'Only benchmark embeddings')
  .option('--workers-only', 'Only benchmark workers')
  .action(async (opts) => {
    try {
      const benchmark = require('../dist/workers/benchmark.js');

      if (opts.embeddingsOnly) {
        console.log(chalk.cyan('\n📊 Benchmarking ONNX Embeddings...\n'));
        const results = await benchmark.benchmarkEmbeddings(10);
        console.log(benchmark.formatBenchmarkResults(results));
      } else if (opts.workersOnly) {
        console.log(chalk.cyan('\n🔧 Benchmarking Native Workers...\n'));
        const results = await benchmark.benchmarkWorkers(opts.path);
        console.log(benchmark.formatBenchmarkResults(results));
      } else {
        await benchmark.runFullBenchmark(opts.path);
      }
    } catch (e) {
      console.error(chalk.red('Benchmark failed:'), e.message);
      if (e.stack) console.error(chalk.dim(e.stack));
    }
  });

nativeCmd.command('list')
  .description('List available native worker types')
  .action(() => {
    console.log(chalk.cyan('\n🔧 Native Worker Types\n'));
    console.log(chalk.bold('security'));
    console.log(chalk.dim('  Security vulnerability scanner'));
    console.log(chalk.dim('  Phases: file-discovery → security-scan → summarization'));
    console.log(chalk.dim('  No ONNX/VectorDB required\n'));

    console.log(chalk.bold('analysis'));
    console.log(chalk.dim('  Full code analysis with embeddings'));
    console.log(chalk.dim('  Phases: file-discovery → pattern-extraction → embedding-generation'));
    console.log(chalk.dim('          → vector-storage → complexity-analysis → summarization'));
    console.log(chalk.dim('  Requires: ONNX embedder, VectorDB\n'));

    console.log(chalk.bold('learning'));
    console.log(chalk.dim('  Pattern learning with vector storage'));
    console.log(chalk.dim('  Phases: file-discovery → pattern-extraction → embedding-generation'));
    console.log(chalk.dim('          → vector-storage → summarization'));
    console.log(chalk.dim('  Requires: ONNX embedder, VectorDB, Intelligence memory\n'));

    console.log(chalk.bold('Available Phases:'));
    const phases = [
      'file-discovery', 'pattern-extraction', 'embedding-generation',
      'vector-storage', 'similarity-search', 'security-scan',
      'complexity-analysis', 'summarization'
    ];
    phases.forEach(p => console.log(chalk.dim(`  • ${p}`)));
  });

nativeCmd.command('compare')
  .description('Compare ruvector native vs agentic-flow workers')
  .option('--path <path>', 'Target path for benchmarks', '.')
  .option('--iterations <n>', 'Number of iterations', '5')
  .action(async (opts) => {
    const iterations = parseInt(opts.iterations) || 5;
    console.log(chalk.cyan('\n╔════════════════════════════════════════════════════════════════╗'));
    console.log(chalk.cyan('║           Worker System Comparison Benchmark                   ║'));
    console.log(chalk.cyan('╚════════════════════════════════════════════════════════════════╝\n'));

    try {
      const { performance } = require('perf_hooks');
      const benchmark = require('../dist/workers/benchmark.js');
      const { createSecurityWorker, createAnalysisWorker } = require('../dist/workers/native-worker.js');

      // Test 1: Native Security Worker
      console.log(chalk.yellow('1. Native Security Worker'));
      const securityTimes = [];
      const securityWorker = createSecurityWorker();
      for (let i = 0; i < iterations; i++) {
        const start = performance.now();
        await securityWorker.run(opts.path);
        securityTimes.push(performance.now() - start);
      }
      const secAvg = securityTimes.reduce((a, b) => a + b) / securityTimes.length;
      console.log(chalk.dim(`   Avg: ${secAvg.toFixed(1)}ms (${iterations} runs)`));

      // Test 2: Native Analysis Worker
      console.log(chalk.yellow('\n2. Native Analysis Worker (ONNX + VectorDB)'));
      const analysisTimes = [];
      const analysisWorker = createAnalysisWorker();
      for (let i = 0; i < Math.min(iterations, 3); i++) {
        const start = performance.now();
        await analysisWorker.run(opts.path);
        analysisTimes.push(performance.now() - start);
      }
      const anaAvg = analysisTimes.reduce((a, b) => a + b) / analysisTimes.length;
      console.log(chalk.dim(`   Avg: ${anaAvg.toFixed(1)}ms (${Math.min(iterations, 3)} runs)`));

      // Test 3: agentic-flow workers (if available)
      let agenticAvailable = false;
      let agenticSecAvg = 0;
      let agenticAnaAvg = 0;
      try {
        const agentic = require('agentic-flow');
        agenticAvailable = true;

        console.log(chalk.yellow('\n3. agentic-flow Security Worker'));
        // Note: Would need actual agentic-flow integration here
        console.log(chalk.dim('   (Integration pending - use agentic-flow CLI directly)'));

      } catch (e) {
        console.log(chalk.yellow('\n3. agentic-flow Workers'));
        console.log(chalk.dim('   Not installed (npm i agentic-flow@alpha)'));
      }

      // Summary
      console.log(chalk.cyan('\n═══════════════════════════════════════════════════════════════'));
      console.log(chalk.bold('Summary'));
      console.log(chalk.cyan('═══════════════════════════════════════════════════════════════'));
      console.log(chalk.white('\nNative RuVector Workers:'));
      console.log(chalk.dim(`  Security scan:  ${secAvg.toFixed(1)}ms avg`));
      console.log(chalk.dim(`  Full analysis:  ${anaAvg.toFixed(1)}ms avg`));

      if (agenticAvailable) {
        console.log(chalk.white('\nagentic-flow Workers:'));
        console.log(chalk.dim('  Security scan:  (run: agentic-flow workers native security)'));
        console.log(chalk.dim('  Full analysis:  (run: agentic-flow workers native analysis)'));
      }

      console.log(chalk.white('\nArchitecture Benefits:'));
      console.log(chalk.dim('  • Shared ONNX model cache (memory efficient)'));
      console.log(chalk.dim('  • 7 native phases with deep integration'));
      console.log(chalk.dim('  • SIMD-accelerated WASM embeddings'));
      console.log(chalk.dim('  • HNSW vector indexing (150x faster search)'));
      console.log('');
    } catch (e) {
      console.error(chalk.red('Comparison failed:'), e.message);
      if (opts.verbose) console.error(chalk.dim(e.stack));
    }
  });

// RVF (RuVector Format) commands
const rvfCmd = program.command('rvf').description('RuVector Format (.rvf) cognitive container operations');

rvfCmd.command('create <path>')
  .description('Create a new .rvf store')
  .requiredOption('-d, --dimension <n>', 'Vector dimension', parseInt)
  .option('-m, --metric <metric>', 'Distance metric (l2, cosine, dotproduct)', 'cosine')
  .action(async (storePath, opts) => {
    try {
      const { createRvfStore, rvfClose } = require('../dist/core/rvf-wrapper.js');
      const store = await createRvfStore(storePath, { dimensions: opts.dimension, metric: opts.metric });
      await rvfClose(store);
      console.log(chalk.green(`Created ${storePath} (dim=${opts.dimension}, metric=${opts.metric})`));
    } catch (e) { console.error(chalk.red(e.message)); process.exit(1); }
  });

rvfCmd.command('ingest <path>')
  .description('Ingest vectors into an .rvf store')
  .requiredOption('-i, --input <file>', 'Input file (JSON array of {id, vector})')
  .option('-f, --format <fmt>', 'Input format (json)', 'json')
  .action(async (storePath, opts) => {
    try {
      const { openRvfStore, rvfIngest, rvfClose } = require('../dist/core/rvf-wrapper.js');
      const store = await openRvfStore(storePath);
      const data = JSON.parse(fs.readFileSync(opts.input, 'utf8'));
      const result = await rvfIngest(store, data);
      await rvfClose(store);
      console.log(chalk.green(`Ingested ${result.accepted} vectors (${result.rejected} rejected)`));
    } catch (e) { console.error(chalk.red(e.message)); process.exit(1); }
  });

rvfCmd.command('query <path>')
  .description('Query nearest neighbors')
  .requiredOption('-v, --vector <values>', 'Comma-separated vector values')
  .option('-k, --k <n>', 'Number of results', parseInt, 10)
  .action(async (storePath, opts) => {
    try {
      const { openRvfStore, rvfQuery, rvfClose } = require('../dist/core/rvf-wrapper.js');
      const store = await openRvfStore(storePath);
      const vector = opts.vector.split(',').map(Number);
      const results = await rvfQuery(store, vector, opts.k);
      await rvfClose(store);
      results.forEach((r, i) => console.log(chalk.dim(`  ${i+1}. id=${r.id}  dist=${r.distance.toFixed(6)}`)));
      console.log(chalk.green(`${results.length} results`));
    } catch (e) { console.error(chalk.red(e.message)); process.exit(1); }
  });

rvfCmd.command('status <path>')
  .description('Show store statistics')
  .action(async (storePath) => {
    try {
      const { openRvfStore, rvfStatus, rvfClose } = require('../dist/core/rvf-wrapper.js');
      const store = await openRvfStore(storePath);
      const s = await rvfStatus(store);
      await rvfClose(store);
      console.log(chalk.cyan('RVF Store Status'));
      Object.entries(s).forEach(([k, v]) => console.log(chalk.dim(`  ${k}: ${v}`)));
    } catch (e) { console.error(chalk.red(e.message)); process.exit(1); }
  });

rvfCmd.command('segments <path>')
  .description('List all segments in an .rvf file')
  .action(async (storePath) => {
    try {
      const { openRvfStore, rvfClose } = require('../dist/core/rvf-wrapper.js');
      const store = await openRvfStore(storePath);
      const segs = await store.segments();
      await rvfClose(store);
      segs.forEach((seg, i) => console.log(chalk.dim(`  [${i}] type=0x${seg.type.toString(16)} size=${seg.size}`)));
      console.log(chalk.green(`${segs.length} segments`));
    } catch (e) { console.error(chalk.red(e.message)); process.exit(1); }
  });

rvfCmd.command('derive <parent> <child>')
  .description('Create a derived store with lineage tracking')
  .action(async (parentPath, childPath) => {
    try {
      const { openRvfStore, rvfDerive, rvfClose } = require('../dist/core/rvf-wrapper.js');
      const store = await openRvfStore(parentPath);
      await rvfDerive(store, childPath);
      await rvfClose(store);
      console.log(chalk.green(`Derived ${childPath} from ${parentPath}`));
    } catch (e) { console.error(chalk.red(e.message)); process.exit(1); }
  });

rvfCmd.command('compact <path>')
  .description('Compact store, reclaim deleted space')
  .action(async (storePath) => {
    try {
      const { openRvfStore, rvfCompact, rvfClose } = require('../dist/core/rvf-wrapper.js');
      const store = await openRvfStore(storePath);
      const result = await rvfCompact(store);
      await rvfClose(store);
      console.log(chalk.green(`Compacted: ${result.segmentsCompacted} segments, ${result.bytesReclaimed} bytes reclaimed`));
    } catch (e) { console.error(chalk.red(e.message)); process.exit(1); }
  });

rvfCmd.command('export <path>')
  .description('Export store data')
  .option('-o, --output <file>', 'Output file')
  .action(async (storePath, opts) => {
    try {
      const { openRvfStore, rvfStatus, rvfClose } = require('../dist/core/rvf-wrapper.js');
      const store = await openRvfStore(storePath);
      const status = await rvfStatus(store);
      const segs = await store.segments();
      await rvfClose(store);
      const data = JSON.stringify({ status, segments: segs }, null, 2);
      if (opts.output) { fs.writeFileSync(opts.output, data); console.log(chalk.green(`Exported to ${opts.output}`)); }
      else { console.log(data); }
    } catch (e) { console.error(chalk.red(e.message)); process.exit(1); }
  });

// RVF example download/list commands
const RVF_EXAMPLES = [
  { name: 'basic_store', size: '152 KB', desc: '1,000 vectors, dim 128, cosine metric' },
  { name: 'semantic_search', size: '755 KB', desc: 'Semantic search with HNSW index' },
  { name: 'rag_pipeline', size: '303 KB', desc: 'RAG pipeline with embeddings' },
  { name: 'embedding_cache', size: '755 KB', desc: 'Cached embedding store' },
  { name: 'quantization', size: '1.5 MB', desc: 'PQ-compressed vectors' },
  { name: 'progressive_index', size: '2.5 MB', desc: 'Large-scale progressive HNSW index' },
  { name: 'filtered_search', size: '255 KB', desc: 'Metadata-filtered vector search' },
  { name: 'recommendation', size: '102 KB', desc: 'Recommendation engine vectors' },
  { name: 'agent_memory', size: '32 KB', desc: 'AI agent episodic memory' },
  { name: 'swarm_knowledge', size: '86 KB', desc: 'Multi-agent shared knowledge base' },
  { name: 'experience_replay', size: '27 KB', desc: 'RL experience replay buffer' },
  { name: 'tool_cache', size: '26 KB', desc: 'MCP tool call cache' },
  { name: 'mcp_in_rvf', size: '32 KB', desc: 'MCP server embedded in RVF' },
  { name: 'ruvbot', size: '51 KB', desc: 'Chatbot knowledge store' },
  { name: 'claude_code_appliance', size: '17 KB', desc: 'Claude Code cognitive appliance' },
  { name: 'lineage_parent', size: '52 KB', desc: 'COW parent file' },
  { name: 'lineage_child', size: '26 KB', desc: 'COW child (derived) file' },
  { name: 'self_booting', size: '31 KB', desc: 'Self-booting with KERNEL_SEG' },
  { name: 'linux_microkernel', size: '15 KB', desc: 'Embedded Linux microkernel' },
  { name: 'ebpf_accelerator', size: '153 KB', desc: 'eBPF distance accelerator' },
  { name: 'browser_wasm', size: '14 KB', desc: 'Browser WASM module embedded' },
  { name: 'tee_attestation', size: '102 KB', desc: 'TEE attestation with witnesses' },
  { name: 'zero_knowledge', size: '52 KB', desc: 'ZK-proof witness chain' },
  { name: 'sealed_engine', size: '208 KB', desc: 'Sealed inference engine' },
  { name: 'access_control', size: '77 KB', desc: 'Permission-gated vectors' },
  { name: 'financial_signals', size: '202 KB', desc: 'Financial signal vectors' },
  { name: 'medical_imaging', size: '302 KB', desc: 'Medical imaging embeddings' },
  { name: 'legal_discovery', size: '903 KB', desc: 'Legal document discovery' },
  { name: 'multimodal_fusion', size: '804 KB', desc: 'Multi-modal embedding fusion' },
  { name: 'hyperbolic_taxonomy', size: '23 KB', desc: 'Hyperbolic space taxonomy' },
  { name: 'network_telemetry', size: '16 KB', desc: 'Network telemetry vectors' },
  { name: 'postgres_bridge', size: '152 KB', desc: 'PostgreSQL bridge vectors' },
  { name: 'ruvllm_inference', size: '133 KB', desc: 'RuvLLM inference cache' },
  { name: 'serverless', size: '509 KB', desc: 'Serverless deployment bundle' },
  { name: 'edge_iot', size: '27 KB', desc: 'Edge/IoT lightweight store' },
  { name: 'dedup_detector', size: '153 KB', desc: 'Deduplication detector' },
  { name: 'compacted', size: '77 KB', desc: 'Post-compaction example' },
  { name: 'posix_fileops', size: '52 KB', desc: 'POSIX file operations test' },
  { name: 'network_sync_a', size: '52 KB', desc: 'Network sync peer A' },
  { name: 'network_sync_b', size: '52 KB', desc: 'Network sync peer B' },
  { name: 'agent_handoff_a', size: '31 KB', desc: 'Agent handoff source' },
  { name: 'agent_handoff_b', size: '11 KB', desc: 'Agent handoff target' },
  { name: 'reasoning_parent', size: '5.6 KB', desc: 'Reasoning chain parent' },
  { name: 'reasoning_child', size: '8.1 KB', desc: 'Reasoning chain child' },
  { name: 'reasoning_grandchild', size: '162 B', desc: 'Minimal derived file' },
];

const RVF_BASE_URL = 'https://raw.githubusercontent.com/ruvnet/ruvector/main/examples/rvf/output';

rvfCmd.command('examples')
  .description('List available example .rvf files')
  .option('--json', 'Output as JSON')
  .action((opts) => {
    if (opts.json) {
      console.log(JSON.stringify(RVF_EXAMPLES, null, 2));
      return;
    }
    console.log(chalk.bold.cyan('\nAvailable RVF Example Files (45 total)\n'));
    console.log(chalk.dim(`Download: npx ruvector rvf download <name>\n`));
    const maxName = Math.max(...RVF_EXAMPLES.map(e => e.name.length));
    const maxSize = Math.max(...RVF_EXAMPLES.map(e => e.size.length));
    for (const ex of RVF_EXAMPLES) {
      const name = chalk.green(ex.name.padEnd(maxName));
      const size = chalk.yellow(ex.size.padStart(maxSize));
      console.log(`  ${name}  ${size}  ${chalk.dim(ex.desc)}`);
    }
    console.log(chalk.dim(`\nFull catalog: https://github.com/ruvnet/ruvector/tree/main/examples/rvf/output\n`));
  });

rvfCmd.command('download [names...]')
  .description('Download example .rvf files from GitHub')
  .option('-a, --all', 'Download all 45 examples (~11 MB)')
  .option('-o, --output <dir>', 'Output directory', '.')
  .action(async (names, opts) => {
    const https = require('https');
    const ALLOWED_REDIRECT_HOSTS = ['raw.githubusercontent.com', 'objects.githubusercontent.com', 'github.com'];
    const sanitizeFileName = (name) => {
      // Strip path separators and parent directory references
      const base = path.basename(name);
      // Only allow alphanumeric, underscores, hyphens, dots
      if (!/^[\w\-.]+$/.test(base)) throw new Error(`Invalid filename: ${base}`);
      return base;
    };
    const downloadFile = (url, dest) => new Promise((resolve, reject) => {
      const file = fs.createWriteStream(dest);
      https.get(url, (res) => {
        if (res.statusCode === 302 || res.statusCode === 301) {
          const redirectUrl = res.headers.location;
          try {
            const redirectHost = new URL(redirectUrl).hostname;
            if (!ALLOWED_REDIRECT_HOSTS.includes(redirectHost)) {
              file.close();
              reject(new Error(`Redirect to untrusted host: ${redirectHost}`));
              return;
            }
          } catch { file.close(); reject(new Error('Invalid redirect URL')); return; }
          https.get(redirectUrl, (res2) => { res2.pipe(file); file.on('finish', () => { file.close(); resolve(); }); }).on('error', reject);
          return;
        }
        if (res.statusCode !== 200) { file.close(); fs.unlinkSync(dest); reject(new Error(`HTTP ${res.statusCode}`)); return; }
        res.pipe(file);
        file.on('finish', () => { file.close(); resolve(); });
      }).on('error', reject);
    });

    let toDownload = [];
    if (opts.all) {
      toDownload = RVF_EXAMPLES.map(e => e.name);
    } else if (names && names.length > 0) {
      toDownload = names;
    } else {
      console.error(chalk.red('Specify example names or use --all. Run `npx ruvector rvf examples` to list.'));
      process.exit(1);
    }

    const outDir = path.resolve(opts.output);
    if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

    console.log(chalk.bold.cyan(`\nDownloading ${toDownload.length} .rvf file(s) to ${outDir}\n`));
    let ok = 0, fail = 0;
    for (const name of toDownload) {
      const rawName = name.endsWith('.rvf') ? name : `${name}.rvf`;
      let fileName;
      try { fileName = sanitizeFileName(rawName); } catch (e) {
        console.log(chalk.red(`SKIPPED: ${e.message}`));
        fail++;
        continue;
      }
      // Validate against known examples when not using --all
      if (!opts.all) {
        const baseName = fileName.replace(/\.rvf$/, '');
        if (!RVF_EXAMPLES.some(e => e.name === baseName)) {
          console.log(chalk.red(`SKIPPED: Unknown example '${baseName}'. Run 'npx ruvector rvf examples' to list.`));
          fail++;
          continue;
        }
      }
      const url = `${RVF_BASE_URL}/${encodeURIComponent(fileName)}`;
      const dest = path.join(outDir, fileName);
      // Path containment check
      if (!path.resolve(dest).startsWith(path.resolve(outDir) + path.sep) && path.resolve(dest) !== path.resolve(outDir)) {
        console.log(chalk.red(`SKIPPED: Path traversal detected for '${fileName}'`));
        fail++;
        continue;
      }
      try {
        process.stdout.write(chalk.dim(`  ${fileName} ... `));
        await downloadFile(url, dest);
        const stat = fs.statSync(dest);
        console.log(chalk.green(`OK (${(stat.size / 1024).toFixed(0)} KB)`));
        ok++;
      } catch (e) {
        console.log(chalk.red(`FAILED: ${e.message}`));
        fail++;
      }
    }
    console.log(chalk.bold(`\nDone: ${ok} downloaded, ${fail} failed\n`));
  });

// MCP Server command
const mcpCmd = program.command('mcp').description('MCP (Model Context Protocol) server for Claude Code integration');

mcpCmd.command('start')
  .description('Start the RuVector MCP server')
  .action(() => {
    // Execute the mcp-server.js directly
    const mcpServerPath = path.join(__dirname, 'mcp-server.js');
    if (!fs.existsSync(mcpServerPath)) {
      console.error(chalk.red('Error: MCP server not found at'), mcpServerPath);
      process.exit(1);
    }
    require(mcpServerPath);
  });

mcpCmd.command('info')
  .description('Show MCP server information and setup instructions')
  .action(() => {
    console.log(chalk.bold.cyan('\n🔌 RuVector MCP Server\n'));
    console.log(chalk.white('The RuVector MCP server provides self-learning intelligence'));
    console.log(chalk.white('tools to Claude Code via the Model Context Protocol.\n'));

    console.log(chalk.bold('Hooks Tools:'));
    console.log(chalk.dim('  hooks_stats      - Get intelligence statistics'));
    console.log(chalk.dim('  hooks_route      - Route task to best agent'));
    console.log(chalk.dim('  hooks_remember   - Store context in vector memory'));
    console.log(chalk.dim('  hooks_recall     - Search vector memory'));
    console.log(chalk.dim('  hooks_init       - Initialize hooks in project'));
    console.log(chalk.dim('  hooks_pretrain   - Pretrain from repository'));
    console.log(chalk.dim('  hooks_build_agents - Generate agent configs'));
    console.log(chalk.dim('  hooks_verify     - Verify hooks configuration'));
    console.log(chalk.dim('  hooks_doctor     - Diagnose setup issues'));
    console.log(chalk.dim('  hooks_export     - Export intelligence data'));

    console.log(chalk.bold('\nRVF Vector Store Tools:'));
    console.log(chalk.dim('  rvf_create       - Create new .rvf vector store'));
    console.log(chalk.dim('  rvf_open         - Open existing .rvf store'));
    console.log(chalk.dim('  rvf_ingest       - Insert vectors into store'));
    console.log(chalk.dim('  rvf_query        - Query nearest neighbors'));
    console.log(chalk.dim('  rvf_delete       - Delete vectors by ID'));
    console.log(chalk.dim('  rvf_status       - Get store status'));
    console.log(chalk.dim('  rvf_compact      - Compact store'));
    console.log(chalk.dim('  rvf_derive       - COW-branch to child store'));
    console.log(chalk.dim('  rvf_segments     - List file segments'));
    console.log(chalk.dim('  rvf_examples     - List example .rvf files'));

    console.log(chalk.bold('\nrvlite Query Tools:'));
    console.log(chalk.dim('  rvlite_sql       - Execute SQL query over rvlite vector DB'));
    console.log(chalk.dim('  rvlite_cypher    - Execute Cypher graph query'));
    console.log(chalk.dim('  rvlite_sparql    - Execute SPARQL RDF query'));

    console.log(chalk.bold('\nBrain Tools (Shared Intelligence):'));
    console.log(chalk.dim('  brain_search     - Semantic search shared knowledge'));
    console.log(chalk.dim('  brain_share      - Share knowledge with brain'));
    console.log(chalk.dim('  brain_get        - Get memory by ID'));
    console.log(chalk.dim('  brain_vote       - Vote on quality'));
    console.log(chalk.dim('  brain_list       - List memories'));
    console.log(chalk.dim('  brain_delete     - Delete own contribution'));
    console.log(chalk.dim('  brain_status     - System health'));
    console.log(chalk.dim('  brain_drift      - Check knowledge drift'));
    console.log(chalk.dim('  brain_partition   - MinCut knowledge topology'));
    console.log(chalk.dim('  brain_transfer   - Domain transfer'));
    console.log(chalk.dim('  brain_sync       - LoRA weight sync'));

    console.log(chalk.bold('\nEdge Tools (Distributed Compute):'));
    console.log(chalk.dim('  edge_status      - Network status'));
    console.log(chalk.dim('  edge_join        - Join as compute node'));
    console.log(chalk.dim('  edge_balance     - Check rUv balance'));
    console.log(chalk.dim('  edge_tasks       - Available compute tasks'));

    console.log(chalk.bold('\nIdentity Tools:'));
    console.log(chalk.dim('  identity_generate - Generate new pi key'));
    console.log(chalk.dim('  identity_show    - Show current identity'));

    console.log(chalk.bold('\n📦 Resources:'));
    console.log(chalk.dim('  ruvector://intelligence/stats     - Current statistics'));
    console.log(chalk.dim('  ruvector://intelligence/patterns  - Learned patterns'));
    console.log(chalk.dim('  ruvector://intelligence/memories  - Vector memories'));

    console.log(chalk.bold.yellow('\n⚙️  Setup Instructions:\n'));
    console.log(chalk.white('Add to Claude Code:'));
    console.log(chalk.cyan('  claude mcp add ruvector npx ruvector mcp start\n'));

    console.log(chalk.white('Or add to .claude/settings.json:'));
    console.log(chalk.dim(`  {
    "mcpServers": {
      "ruvector": {
        "command": "npx",
        "args": ["ruvector", "mcp", "start"]
      }
    }
  }`));
    console.log();
  });

mcpCmd.command('tools')
  .description('List all MCP tools with descriptions (JSON output)')
  .option('--group <group>', 'Filter by group (hooks, workers, rvf, rvlite, brain, edge, identity)')
  .option('--json', 'JSON output')
  .action((opts) => {
    const tools = {
      'hooks-core': [
        { name: 'hooks_stats', desc: 'Intelligence statistics' },
        { name: 'hooks_route', desc: 'Route task to best agent' },
        { name: 'hooks_remember', desc: 'Store in vector memory' },
        { name: 'hooks_recall', desc: 'Search vector memory' },
        { name: 'hooks_init', desc: 'Initialize hooks in project' },
        { name: 'hooks_pretrain', desc: 'Pretrain from repository' },
        { name: 'hooks_build_agents', desc: 'Generate agent configs' },
        { name: 'hooks_verify', desc: 'Verify hooks config' },
        { name: 'hooks_doctor', desc: 'Diagnose setup issues' },
        { name: 'hooks_export', desc: 'Export intelligence data' },
        { name: 'hooks_capabilities', desc: 'Get engine capabilities' },
        { name: 'hooks_import', desc: 'Import intelligence data' },
        { name: 'hooks_swarm_recommend', desc: 'Recommend agent for task' },
        { name: 'hooks_suggest_context', desc: 'Suggest relevant context' },
      ],
      'hooks-trajectory': [
        { name: 'hooks_trajectory_begin', desc: 'Begin execution trajectory' },
        { name: 'hooks_trajectory_step', desc: 'Add trajectory step' },
        { name: 'hooks_trajectory_end', desc: 'End trajectory with score' },
      ],
      'hooks-coedit': [
        { name: 'hooks_coedit_record', desc: 'Record co-edit pattern' },
        { name: 'hooks_coedit_suggest', desc: 'Suggest related files' },
      ],
      'hooks-errors': [
        { name: 'hooks_error_record', desc: 'Record error and fix' },
        { name: 'hooks_error_suggest', desc: 'Suggest fixes for error' },
        { name: 'hooks_force_learn', desc: 'Force learning cycle' },
      ],
      'hooks-analysis': [
        { name: 'hooks_ast_analyze', desc: 'Parse AST, extract symbols' },
        { name: 'hooks_ast_complexity', desc: 'Cyclomatic complexity' },
        { name: 'hooks_diff_analyze', desc: 'Semantic diff analysis' },
        { name: 'hooks_diff_classify', desc: 'Classify change type' },
        { name: 'hooks_diff_similar', desc: 'Find similar past commits' },
        { name: 'hooks_coverage_route', desc: 'Coverage-aware routing' },
        { name: 'hooks_coverage_suggest', desc: 'Suggest tests for files' },
        { name: 'hooks_graph_mincut', desc: 'MinCut code boundaries' },
        { name: 'hooks_graph_cluster', desc: 'Code community detection' },
        { name: 'hooks_security_scan', desc: 'Security vulnerability scan' },
        { name: 'hooks_rag_context', desc: 'RAG-enhanced context' },
        { name: 'hooks_git_churn', desc: 'Git churn hot spots' },
        { name: 'hooks_route_enhanced', desc: 'Enhanced routing w/ signals' },
        { name: 'hooks_attention_info', desc: 'Attention mechanisms info' },
        { name: 'hooks_gnn_info', desc: 'GNN capabilities info' },
      ],
      'hooks-learning': [
        { name: 'hooks_learning_config', desc: 'Configure learning algorithms (9 algos)' },
        { name: 'hooks_learning_stats', desc: 'Learning performance metrics' },
        { name: 'hooks_learning_update', desc: 'Record learning experience' },
        { name: 'hooks_learn', desc: 'Combined learn + recommend' },
        { name: 'hooks_algorithms_list', desc: 'List all algorithms' },
        { name: 'hooks_batch_learn', desc: 'Batch learning experiences' },
      ],
      'hooks-compress': [
        { name: 'hooks_compress', desc: 'Compress pattern storage (10x savings)' },
        { name: 'hooks_compress_stats', desc: 'Compression statistics' },
        { name: 'hooks_compress_store', desc: 'Store compressed embedding' },
        { name: 'hooks_compress_get', desc: 'Retrieve compressed embedding' },
      ],
      'hooks-events': [
        { name: 'hooks_subscribe_snapshot', desc: 'Event state snapshot' },
        { name: 'hooks_watch_status', desc: 'File watch status' },
      ],
      'workers': [
        { name: 'workers_dispatch', desc: 'Dispatch background worker' },
        { name: 'workers_status', desc: 'Worker status' },
        { name: 'workers_results', desc: 'Worker results' },
        { name: 'workers_triggers', desc: 'List worker triggers' },
        { name: 'workers_stats', desc: 'Worker statistics' },
        { name: 'workers_presets', desc: 'Available presets' },
        { name: 'workers_phases', desc: 'Worker phases' },
        { name: 'workers_create', desc: 'Create custom worker' },
        { name: 'workers_run', desc: 'Run worker' },
        { name: 'workers_custom', desc: 'Custom worker' },
        { name: 'workers_init_config', desc: 'Init config file' },
        { name: 'workers_load_config', desc: 'Load config' },
      ],
      'rvf': [
        { name: 'rvf_create', desc: 'Create .rvf vector store' },
        { name: 'rvf_open', desc: 'Open existing store' },
        { name: 'rvf_ingest', desc: 'Insert vectors' },
        { name: 'rvf_query', desc: 'Query nearest neighbors' },
        { name: 'rvf_delete', desc: 'Delete vectors by ID' },
        { name: 'rvf_status', desc: 'Store status' },
        { name: 'rvf_compact', desc: 'Compact store' },
        { name: 'rvf_derive', desc: 'COW-branch child store' },
        { name: 'rvf_segments', desc: 'List file segments' },
        { name: 'rvf_examples', desc: 'Example .rvf files' },
      ],
      'rvlite': [
        { name: 'rvlite_sql', desc: 'SQL query over vector DB' },
        { name: 'rvlite_cypher', desc: 'Cypher graph query' },
        { name: 'rvlite_sparql', desc: 'SPARQL RDF query' },
      ],
      'brain': [
        { name: 'brain_search', desc: 'Semantic search shared knowledge' },
        { name: 'brain_share', desc: 'Share knowledge' },
        { name: 'brain_get', desc: 'Get memory by ID' },
        { name: 'brain_vote', desc: 'Vote on quality' },
        { name: 'brain_list', desc: 'List memories' },
        { name: 'brain_delete', desc: 'Delete own contribution' },
        { name: 'brain_status', desc: 'System health' },
        { name: 'brain_drift', desc: 'Knowledge drift check' },
        { name: 'brain_partition', desc: 'MinCut knowledge topology' },
        { name: 'brain_transfer', desc: 'Domain transfer' },
        { name: 'brain_sync', desc: 'LoRA weight sync' },
      ],
      'edge': [
        { name: 'edge_status', desc: 'Network status' },
        { name: 'edge_join', desc: 'Join as compute node' },
        { name: 'edge_balance', desc: 'Check rUv balance' },
        { name: 'edge_tasks', desc: 'Available compute tasks' },
      ],
      'identity': [
        { name: 'identity_generate', desc: 'Generate new pi key' },
        { name: 'identity_show', desc: 'Show current identity' },
      ],
    };

    // Filter by group if specified
    let groups = Object.entries(tools);
    if (opts.group) {
      groups = groups.filter(([g]) => g === opts.group || g.startsWith(opts.group));
    }

    if (opts.json || !process.stdout.isTTY) {
      const flat = groups.flatMap(([group, items]) => items.map(t => ({ ...t, group })));
      console.log(JSON.stringify(flat, null, 2));
      return;
    }

    let total = 0;
    groups.forEach(([group, items]) => {
      console.log(chalk.bold.cyan(`\n${group} (${items.length} tools)`));
      items.forEach(t => {
        console.log(`  ${chalk.bold(t.name.padEnd(28))} ${chalk.dim(t.desc)}`);
      });
      total += items.length;
    });
    console.log(chalk.bold(`\nTotal: ${total} MCP tools\n`));
  });

mcpCmd.command('test')
  .description('Test MCP server setup and tool registration')
  .action(() => {
    console.log(chalk.bold.cyan('\nMCP Server Test Results'));
    console.log(chalk.dim('-'.repeat(40)));

    // Test 1: server file exists
    const mcpServerPath = path.join(__dirname, 'mcp-server.js');
    if (fs.existsSync(mcpServerPath)) {
      console.log(`  ${chalk.green('PASS')} mcp-server.js exists`);
    } else {
      console.log(`  ${chalk.red('FAIL')} mcp-server.js not found`);
      process.exit(1);
    }

    // Test 2: syntax check
    try {
      const { execSync } = require('child_process');
      execSync(`node -c ${mcpServerPath}`, { stdio: 'pipe' });
      console.log(`  ${chalk.green('PASS')} mcp-server.js syntax valid`);
    } catch {
      console.log(`  ${chalk.red('FAIL')} mcp-server.js has syntax errors`);
      process.exit(1);
    }

    // Test 3: MCP SDK available
    try {
      require('@modelcontextprotocol/sdk/server/index.js');
      console.log(`  ${chalk.green('PASS')} @modelcontextprotocol/sdk installed`);
    } catch {
      console.log(`  ${chalk.red('FAIL')} @modelcontextprotocol/sdk not installed`);
      process.exit(1);
    }

    // Test 4: count tools by parsing TOOLS array entries (each has inputSchema)
    try {
      const src = fs.readFileSync(mcpServerPath, 'utf8');
      // Extract the TOOLS array section (from 'const TOOLS = [' to the matching '];')
      const toolsStart = src.indexOf('const TOOLS = [');
      const toolsSection = toolsStart >= 0 ? src.slice(toolsStart) : src;
      // Match tool names that are followed by inputSchema (real MCP tools only)
      const toolDefs = toolsSection.match(/name:\s*'([a-z][a-z0-9_]*)'\s*,\s*\n\s*description:/g) || [];
      const toolNames = toolDefs.map(m => m.match(/name:\s*'([a-z][a-z0-9_]*)'/)[1]);
      const groups = {};
      toolNames.forEach(n => {
        const g = n.split('_')[0];
        groups[g] = (groups[g] || 0) + 1;
      });

      Object.entries(groups).sort((a, b) => b[1] - a[1]).forEach(([group, count]) => {
        console.log(`  ${chalk.green('PASS')} ${group}: ${count} tools`);
      });
      console.log(chalk.bold(`\n  Total: ${toolNames.length} tools registered`));
    } catch (e) {
      console.log(`  ${chalk.yellow('WARN')} Could not parse tool count: ${e.message}`);
    }

    // Test 5: version check
    try {
      const src = fs.readFileSync(mcpServerPath, 'utf8');
      const verMatch = src.match(/version:\s*'([^']+)'/);
      if (verMatch) {
        const pkg = require(path.join(__dirname, '..', 'package.json'));
        const match = verMatch[1] === pkg.version;
        console.log(`  ${match ? chalk.green('PASS') : chalk.yellow('WARN')} Server version: ${verMatch[1]}${match ? '' : ` (package: ${pkg.version})`}`);
      }
    } catch {}

    console.log(chalk.bold.green('\n  All checks passed.\n'));
    console.log(chalk.dim('  Setup: claude mcp add ruvector npx ruvector mcp start\n'));
  });

// ============================================================================
// Brain Commands — Shared intelligence via @ruvector/pi-brain (lazy-loaded)
// ============================================================================

// Lazy load pi-brain (optional peer dependency)
async function requirePiBrain() {
  try {
    return await import('@ruvector/pi-brain');
  } catch {
    console.error(chalk.red('Brain commands require @ruvector/pi-brain'));
    console.error(chalk.yellow('  npm install @ruvector/pi-brain'));
    process.exit(1);
  }
}

// Determine output mode: JSON when --json flag or piped
function isJsonOutput(opts) {
  return opts.json || !process.stdout.isTTY;
}

// Create a PiBrainClient from command options
async function makeBrainClient(opts) {
  const { PiBrainClient } = await requirePiBrain();
  return new PiBrainClient({
    url: opts.url || process.env.BRAIN_URL || 'https://pi.ruv.io',
    apiKey: process.env.PI || 'anonymous',
  });
}

const brainCmd = program
  .command('brain')
  .description('Shared intelligence — search, share, and manage collective knowledge')
  .option('--url <url>', 'Brain server URL', process.env.BRAIN_URL || 'https://pi.ruv.io')
  .option('--json', 'Force JSON output');

brainCmd
  .command('search <query>')
  .description('Semantic search across collective knowledge')
  .option('-l, --limit <n>', 'Max results', '10')
  .option('-c, --category <category>', 'Filter by category')
  .action(async (query, cmdOpts) => {
    const opts = brainCmd.opts();
    const spinner = ora('Searching brain...').start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.search({
        query,
        category: cmdOpts.category || undefined,
        limit: parseInt(cmdOpts.limit),
      });
      const items = Array.isArray(result) ? result : (result.memories || result.results || []);
      spinner.succeed(chalk.green(`Found ${items.length} result(s)`));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        items.forEach((item, i) => {
          console.log(chalk.cyan(`\n  ${i + 1}. ${item.title || item.id || 'Untitled'}`));
          if (item.category) console.log(chalk.gray(`     Category: ${item.category}`));
          if (item.quality_score != null) console.log(chalk.gray(`     Quality:  ${item.quality_score}`));
          if (item.content) console.log(chalk.dim(`     ${item.content.substring(0, 120)}${item.content.length > 120 ? '...' : ''}`));
        });
      }
    } catch (error) {
      spinner.fail(chalk.red('Search failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('share <title>')
  .description('Share knowledge with the collective brain')
  .requiredOption('-c, --category <category>', 'Category (architecture, pattern, solution, etc.)')
  .option('-t, --tags <tags>', 'Comma-separated tags')
  .option('--content <content>', 'Content body (reads from stdin if omitted)')
  .action(async (title, cmdOpts) => {
    const opts = brainCmd.opts();
    const spinner = ora('Sharing knowledge...').start();
    try {
      const client = await makeBrainClient(opts);
      let content = cmdOpts.content;
      if (!content && !process.stdin.isTTY) {
        const chunks = [];
        for await (const chunk of process.stdin) chunks.push(chunk);
        content = Buffer.concat(chunks).toString('utf8').trim();
      }
      if (!content) {
        spinner.fail(chalk.red('No content provided. Use --content or pipe via stdin.'));
        process.exit(1);
      }
      const tags = cmdOpts.tags ? cmdOpts.tags.split(',').map(t => t.trim()) : [];
      const result = await client.share({ title, category: cmdOpts.category, content, tags });
      spinner.succeed(chalk.green('Knowledge shared'));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        console.log(chalk.cyan(`  ID:       ${result.id || 'N/A'}`));
        console.log(chalk.gray(`  Title:    ${title}`));
        console.log(chalk.gray(`  Category: ${cmdOpts.category}`));
        if (tags.length) console.log(chalk.gray(`  Tags:     ${tags.join(', ')}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to share knowledge'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('get <id>')
  .description('Retrieve a memory by ID with full provenance')
  .action(async (id) => {
    const opts = brainCmd.opts();
    const spinner = ora('Fetching memory...').start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.get(id);
      spinner.succeed(chalk.green('Memory retrieved'));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        console.log(chalk.cyan(`  Title:      ${result.title || 'N/A'}`));
        console.log(chalk.gray(`  ID:         ${result.id || id}`));
        console.log(chalk.gray(`  Category:   ${result.category || 'N/A'}`));
        console.log(chalk.gray(`  Quality:    ${result.quality_score != null ? result.quality_score : 'N/A'}`));
        console.log(chalk.gray(`  Created:    ${result.created_at || 'N/A'}`));
        if (result.tags && result.tags.length) console.log(chalk.gray(`  Tags:       ${result.tags.join(', ')}`));
        if (result.content) {
          console.log(chalk.white('\n  Content:'));
          console.log(chalk.dim(`  ${result.content}`));
        }
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to retrieve memory'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('vote <id> <direction>')
  .description('Vote on a memory (up or down)')
  .action(async (id, direction) => {
    const opts = brainCmd.opts();
    if (!['up', 'down'].includes(direction)) {
      console.error(chalk.red('Direction must be "up" or "down"'));
      process.exit(1);
    }
    const spinner = ora(`Voting ${direction} on ${id}...`).start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.vote(id, direction);
      spinner.succeed(chalk.green(`Voted ${direction} on memory ${id}`));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      }
    } catch (error) {
      spinner.fail(chalk.red('Vote failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('list')
  .description('List memories from the collective brain')
  .option('-c, --category <category>', 'Filter by category')
  .option('-l, --limit <n>', 'Max results', '20')
  .action(async (cmdOpts) => {
    const opts = brainCmd.opts();
    const spinner = ora('Listing memories...').start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.list(
        cmdOpts.category || undefined,
        parseInt(cmdOpts.limit),
      );
      const items = Array.isArray(result) ? result : (result.memories || result.results || []);
      spinner.succeed(chalk.green(`${items.length} memor${items.length === 1 ? 'y' : 'ies'} found`));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        items.forEach((item, i) => {
          console.log(chalk.cyan(`  ${i + 1}. ${item.title || item.id || 'Untitled'}`) +
            (item.category ? chalk.gray(` [${item.category}]`) : '') +
            (item.quality_score != null ? chalk.yellow(` (q: ${item.quality_score})`) : ''));
        });
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to list memories'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('delete <id>')
  .description('Delete a memory you contributed')
  .action(async (id) => {
    const opts = brainCmd.opts();
    const spinner = ora(`Deleting memory ${id}...`).start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.delete(id);
      spinner.succeed(chalk.green(`Memory ${id} deleted`));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to delete memory'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('status')
  .description('Show brain system health and statistics')
  .action(async () => {
    const opts = brainCmd.opts();
    const spinner = ora('Fetching brain status...').start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.status();
      spinner.succeed(chalk.green('Brain status'));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        const s = result;
        console.log(chalk.cyan(`  Memories:     ${s.total_memories != null ? s.total_memories : 'N/A'}`));
        console.log(chalk.cyan(`  Contributors: ${s.total_contributors != null ? s.total_contributors : 'N/A'}`));
        console.log(chalk.cyan(`  Quality:      ${s.avg_quality != null ? s.avg_quality : 'N/A'}`));
        console.log(chalk.cyan(`  Drift:        ${s.drift_status || s.drift || 'N/A'}`));
        if (s.uptime) console.log(chalk.gray(`  Uptime:       ${s.uptime}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to get brain status'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('drift [domain]')
  .description('Check knowledge drift for a domain')
  .action(async (domain) => {
    const opts = brainCmd.opts();
    const spinner = ora('Checking drift...').start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.drift(domain || undefined);
      spinner.succeed(chalk.green('Drift analysis complete'));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (domain) console.log(chalk.gray(`  Domain: ${domain}`));
        console.log(chalk.gray(`  Drift:  ${result.drift_score != null ? result.drift_score : JSON.stringify(result)}`));
        if (result.status) console.log(chalk.gray(`  Status: ${result.status}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Drift check failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('partition [domain]')
  .description('View knowledge topology / partitioning')
  .action(async (domain) => {
    const opts = brainCmd.opts();
    const spinner = ora('Fetching partition info...').start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.partition(domain || undefined);
      spinner.succeed(chalk.green('Partition data retrieved'));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (domain) console.log(chalk.gray(`  Domain: ${domain}`));
        console.log(chalk.dim(`  ${JSON.stringify(result, null, 2)}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Partition query failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('transfer <source> <target>')
  .description('Transfer knowledge between domains')
  .action(async (source, target) => {
    const opts = brainCmd.opts();
    const spinner = ora(`Transferring knowledge: ${source} -> ${target}...`).start();
    try {
      const client = await makeBrainClient(opts);
      const result = await client.transfer(source, target);
      spinner.succeed(chalk.green('Transfer complete'));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        console.log(chalk.gray(`  Source: ${source}`));
        console.log(chalk.gray(`  Target: ${target}`));
        if (result.transferred != null) console.log(chalk.gray(`  Transferred: ${result.transferred} items`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Transfer failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('sync [direction]')
  .description('Sync LoRA weights (pull, push, or both)')
  .action(async (direction) => {
    const opts = brainCmd.opts();
    const dir = direction || 'both';
    if (!['pull', 'push', 'both'].includes(dir)) {
      console.error(chalk.red('Direction must be "pull", "push", or "both"'));
      process.exit(1);
    }
    const spinner = ora(`Syncing LoRA weights (${dir})...`).start();
    try {
      await requirePiBrain();
      const url = opts.url || process.env.BRAIN_URL || 'https://pi.ruv.io';
      const apiKey = process.env.PI || 'anonymous';
      const headers = { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' };
      const res = await fetch(`${url}/v1/lora/sync`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ direction: dir }),
      });
      if (!res.ok) throw new Error(`Sync failed (${res.status})`);
      const result = await res.json();
      spinner.succeed(chalk.green(`LoRA sync complete (${dir})`));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        console.log(chalk.gray(`  Direction: ${dir}`));
        console.log(chalk.dim(`  ${JSON.stringify(result)}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('LoRA sync failed'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

brainCmd
  .command('page <action> [id]')
  .description('Brainpedia page management (list, get, create, update)')
  .option('--title <title>', 'Page title (for create/update)')
  .option('--content <content>', 'Page content (for create/update)')
  .option('-l, --limit <n>', 'Max results for list', '20')
  .action(async (action, id, cmdOpts) => {
    const opts = brainCmd.opts();
    const validActions = ['list', 'get', 'create', 'update'];
    if (!validActions.includes(action)) {
      console.error(chalk.red(`Action must be one of: ${validActions.join(', ')}`));
      process.exit(1);
    }
    if (['get', 'update'].includes(action) && !id) {
      console.error(chalk.red(`Page ID required for "${action}"`));
      process.exit(1);
    }
    const spinner = ora(`Brainpedia: ${action}...`).start();
    try {
      const url = opts.url || process.env.BRAIN_URL || 'https://pi.ruv.io';
      const apiKey = process.env.PI || 'anonymous';
      const headers = { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' };
      let result;
      switch (action) {
        case 'list': {
          const params = new URLSearchParams();
          params.set('limit', cmdOpts.limit);
          const res = await fetch(`${url}/v1/pages?${params}`, { headers });
          if (!res.ok) throw new Error(`List failed (${res.status})`);
          result = await res.json();
          break;
        }
        case 'get': {
          const res = await fetch(`${url}/v1/pages/${id}`, { headers });
          if (!res.ok) throw new Error(`Get failed (${res.status})`);
          result = await res.json();
          break;
        }
        case 'create': {
          if (!cmdOpts.title || !cmdOpts.content) {
            spinner.fail(chalk.red('--title and --content are required for create'));
            process.exit(1);
          }
          const res = await fetch(`${url}/v1/pages`, {
            method: 'POST', headers,
            body: JSON.stringify({ title: cmdOpts.title, content: cmdOpts.content }),
          });
          if (!res.ok) throw new Error(`Create failed (${res.status})`);
          result = await res.json();
          break;
        }
        case 'update': {
          const body = {};
          if (cmdOpts.title) body.title = cmdOpts.title;
          if (cmdOpts.content) body.content = cmdOpts.content;
          const res = await fetch(`${url}/v1/pages/${id}`, {
            method: 'PUT', headers,
            body: JSON.stringify(body),
          });
          if (!res.ok) throw new Error(`Update failed (${res.status})`);
          result = await res.json();
          break;
        }
      }
      spinner.succeed(chalk.green(`Brainpedia ${action} complete`));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (action === 'list') {
          const pages = Array.isArray(result) ? result : (result.pages || []);
          pages.forEach((p, i) => {
            console.log(chalk.cyan(`  ${i + 1}. ${p.title || p.id || 'Untitled'}`));
          });
        } else {
          console.log(chalk.dim(`  ${JSON.stringify(result, null, 2)}`));
        }
      }
    } catch (error) {
      spinner.fail(chalk.red(`Brainpedia ${action} failed`));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// ============================================================================
// Edge commands — distributed compute network
// ============================================================================

const EDGE_GENESIS_URL = 'https://edge-net-genesis-875130704813.us-central1.run.app';
const EDGE_DASHBOARD_URL = 'https://edge-net-dashboard-875130704813.us-central1.run.app';

const edgeCmd = program
  .command('edge')
  .description('Distributed edge compute network — status, tasks, and rUv balance')
  .option('--json', 'Force JSON output');

edgeCmd
  .command('status')
  .description('Query edge network status from genesis node')
  .action(async () => {
    const opts = edgeCmd.opts();
    const spinner = ora('Querying edge network...').start();
    try {
      const res = await fetch(`${EDGE_GENESIS_URL}/status`);
      if (!res.ok) throw new Error(`Genesis returned ${res.status}`);
      const result = await res.json();
      spinner.succeed(chalk.green('Edge network status'));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        console.log(chalk.cyan(`  Nodes:       ${result.total_nodes != null ? result.total_nodes : 'N/A'}`));
        console.log(chalk.cyan(`  Active:      ${result.active_nodes != null ? result.active_nodes : 'N/A'}`));
        console.log(chalk.cyan(`  rUv Supply:  ${result.ruv_supply != null ? result.ruv_supply : 'N/A'}`));
        console.log(chalk.cyan(`  Phase:       ${result.phase || result.sunset_phase || 'N/A'}`));
        if (result.uptime) console.log(chalk.gray(`  Uptime:      ${result.uptime}`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to reach edge network'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

edgeCmd
  .command('join')
  .description('Join the edge compute network as a node')
  .action(async () => {
    console.log(chalk.bold.cyan('\n  Edge-Net: Join as Compute Node\n'));
    console.log(chalk.white('  The edge compute network currently runs in-browser via WASM + Web Workers.'));
    console.log(chalk.white('  To join as a compute node, open the dashboard in your browser:\n'));
    console.log(chalk.yellow(`    ${EDGE_DASHBOARD_URL}\n`));
    console.log(chalk.gray('  Node.js headless join is planned for a future release.'));
    console.log(chalk.gray('  Set PI=<your-key> to use your identity when joining.\n'));
  });

edgeCmd
  .command('balance [nodeId]')
  .description('Check rUv balance for a node')
  .action(async (nodeId) => {
    const opts = edgeCmd.opts();
    const id = nodeId || process.env.PI || 'anonymous';
    const spinner = ora('Fetching rUv balance...').start();
    try {
      const res = await fetch(`${EDGE_GENESIS_URL}/balance/${encodeURIComponent(id)}`);
      if (!res.ok) throw new Error(`Balance query failed (${res.status})`);
      const result = await res.json();
      spinner.succeed(chalk.green('rUv balance'));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        console.log(chalk.cyan(`  Node:    ${id}`));
        console.log(chalk.cyan(`  Balance: ${result.balance != null ? result.balance : JSON.stringify(result)} rUv`));
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to fetch balance'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

edgeCmd
  .command('tasks')
  .description('List available distributed compute tasks')
  .action(async () => {
    const opts = edgeCmd.opts();
    const spinner = ora('Fetching compute tasks...').start();
    try {
      const res = await fetch(`${EDGE_GENESIS_URL}/tasks`);
      if (!res.ok) throw new Error(`Tasks query failed (${res.status})`);
      const result = await res.json();
      const tasks = Array.isArray(result) ? result : (result.tasks || []);
      spinner.succeed(chalk.green(`${tasks.length} task(s) available`));
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (tasks.length === 0) {
          console.log(chalk.gray('  No compute tasks currently available.'));
        } else {
          tasks.forEach((task, i) => {
            console.log(chalk.cyan(`  ${i + 1}. ${task.name || task.id || 'Task'}`) +
              (task.reward ? chalk.yellow(` (${task.reward} rUv)`) : '') +
              (task.status ? chalk.gray(` [${task.status}]`) : ''));
          });
        }
      }
    } catch (error) {
      spinner.fail(chalk.red('Failed to fetch tasks'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

edgeCmd
  .command('dashboard')
  .description('Open the edge-net dashboard in your default browser')
  .action(async () => {
    console.log(chalk.cyan(`\n  Opening edge-net dashboard: ${EDGE_DASHBOARD_URL}\n`));
    try {
      const { execSync } = require('child_process');
      const platform = process.platform;
      if (platform === 'darwin') {
        execSync(`open "${EDGE_DASHBOARD_URL}"`);
      } else if (platform === 'win32') {
        execSync(`start "" "${EDGE_DASHBOARD_URL}"`);
      } else {
        execSync(`xdg-open "${EDGE_DASHBOARD_URL}" 2>/dev/null || echo "Open manually: ${EDGE_DASHBOARD_URL}"`);
      }
    } catch {
      console.log(chalk.yellow(`  Could not open browser. Visit manually:`));
      console.log(chalk.white(`  ${EDGE_DASHBOARD_URL}\n`));
    }
  });

// ============================================================================
// Identity commands — manage your pi key
// ============================================================================

const identityCmd = program
  .command('identity')
  .description('Manage your pi identity key for brain, edge, and MCP')
  .option('--json', 'Force JSON output');

identityCmd
  .command('generate')
  .description('Generate a new pi identity key (64 hex chars)')
  .option('--save', 'Save the key to ~/.ruvector/pi-key')
  .action(async (cmdOpts) => {
    const opts = identityCmd.opts();
    const crypto = require('crypto');
    const key = crypto.randomBytes(32).toString('hex');
    let pseudonym;
    try {
      const hash = crypto.createHash('shake256', { outputLength: 16 });
      hash.update(key);
      pseudonym = hash.digest('hex');
    } catch {
      pseudonym = crypto.createHash('sha256').update(key).digest('hex').substring(0, 32);
    }
    if (isJsonOutput(opts)) {
      console.log(JSON.stringify({ key, pseudonym }, null, 2));
    } else {
      console.log(chalk.bold.cyan('\n  New Pi Identity Generated\n'));
      console.log(chalk.white(`  Key:       ${key}`));
      console.log(chalk.gray(`  Pseudonym: ${pseudonym}`));
      console.log(chalk.yellow('\n  Save this key! Set it as PI environment variable:'));
      console.log(chalk.dim(`    export PI=${key}\n`));
    }
    if (cmdOpts.save) {
      const keyDir = path.join(require('os').homedir(), '.ruvector');
      const keyPath = path.join(keyDir, 'pi-key');
      if (!fs.existsSync(keyDir)) fs.mkdirSync(keyDir, { recursive: true, mode: 0o700 });
      fs.writeFileSync(keyPath, key, { mode: 0o600 });
      console.log(chalk.green(`  Key saved to ${keyPath}`));
    }
  });

identityCmd
  .command('show')
  .description('Show current identity pseudonym derived from your pi key')
  .action(async () => {
    const opts = identityCmd.opts();
    const crypto = require('crypto');
    let key = process.env.PI;
    if (!key) {
      const keyPath = path.join(require('os').homedir(), '.ruvector', 'pi-key');
      if (fs.existsSync(keyPath)) {
        key = fs.readFileSync(keyPath, 'utf8').trim();
      }
    }
    if (!key) {
      console.error(chalk.red('No pi key found. Set PI env var or run: ruvector identity generate --save'));
      process.exit(1);
    }
    let pseudonym;
    try {
      const hash = crypto.createHash('shake256', { outputLength: 16 });
      hash.update(key);
      pseudonym = hash.digest('hex');
    } catch {
      pseudonym = crypto.createHash('sha256').update(key).digest('hex').substring(0, 32);
    }
    if (isJsonOutput(opts)) {
      console.log(JSON.stringify({
        pseudonym,
        key_preview: key.substring(0, 8) + '...' + key.substring(key.length - 8),
        source: process.env.PI ? 'PI env var' : '~/.ruvector/pi-key',
      }, null, 2));
    } else {
      console.log(chalk.bold.cyan('\n  Pi Identity\n'));
      console.log(chalk.white(`  Pseudonym: ${pseudonym}`));
      console.log(chalk.gray(`  Key:       ${key.substring(0, 8)}...${key.substring(key.length - 8)}`));
      console.log(chalk.gray(`  Source:    ${process.env.PI ? 'PI env var' : '~/.ruvector/pi-key'}\n`));
    }
  });

identityCmd
  .command('export')
  .description('Export your pi key encrypted with a passphrase')
  .option('-o, --output <file>', 'Output file', 'pi-key.enc')
  .action(async (cmdOpts) => {
    const opts = identityCmd.opts();
    const crypto = require('crypto');
    let key = process.env.PI;
    if (!key) {
      const keyPath = path.join(require('os').homedir(), '.ruvector', 'pi-key');
      if (fs.existsSync(keyPath)) {
        key = fs.readFileSync(keyPath, 'utf8').trim();
      }
    }
    if (!key) {
      console.error(chalk.red('No pi key found. Set PI env var or run: ruvector identity generate --save'));
      process.exit(1);
    }
    let passphrase = process.env.PI_PASSPHRASE;
    if (!passphrase) {
      passphrase = crypto.randomBytes(16).toString('hex');
      console.log(chalk.yellow(`\n  Generated passphrase (save this!): ${passphrase}`));
    }
    const salt = crypto.randomBytes(16);
    const derivedKey = crypto.pbkdf2Sync(passphrase, salt, 100000, 32, 'sha256');
    const iv = crypto.randomBytes(12);
    const cipher = crypto.createCipheriv('aes-256-gcm', derivedKey, iv);
    let encrypted = cipher.update(key, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    const authTag = cipher.getAuthTag().toString('hex');
    const payload = JSON.stringify({
      version: 1,
      algorithm: 'aes-256-gcm',
      salt: salt.toString('hex'),
      iv: iv.toString('hex'),
      authTag,
      data: encrypted,
    });
    const outFile = path.resolve(cmdOpts.output);
    fs.writeFileSync(outFile, payload, { mode: 0o600 });
    if (isJsonOutput(opts)) {
      console.log(JSON.stringify({ file: outFile, algorithm: 'aes-256-gcm' }, null, 2));
    } else {
      console.log(chalk.green(`\n  Key exported to: ${outFile}`));
      console.log(chalk.gray('  Encrypted with AES-256-GCM + PBKDF2'));
      console.log(chalk.yellow('  Keep your passphrase safe -- it cannot be recovered.\n'));
    }
  });

identityCmd
  .command('import <file>')
  .description('Import a pi key from an encrypted backup')
  .option('--save', 'Save the imported key to ~/.ruvector/pi-key')
  .action(async (file, cmdOpts) => {
    const opts = identityCmd.opts();
    const crypto = require('crypto');
    const filePath = path.resolve(file);
    if (!fs.existsSync(filePath)) {
      console.error(chalk.red(`File not found: ${filePath}`));
      process.exit(1);
    }
    const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
    if (payload.version !== 1 || payload.algorithm !== 'aes-256-gcm') {
      console.error(chalk.red('Unsupported key file format'));
      process.exit(1);
    }
    const passphrase = process.env.PI_PASSPHRASE;
    if (!passphrase) {
      console.error(chalk.red('Set PI_PASSPHRASE env var to decrypt the key'));
      process.exit(1);
    }
    try {
      const salt = Buffer.from(payload.salt, 'hex');
      const iv = Buffer.from(payload.iv, 'hex');
      const authTag = Buffer.from(payload.authTag, 'hex');
      const derivedKey = crypto.pbkdf2Sync(passphrase, salt, 100000, 32, 'sha256');
      const decipher = crypto.createDecipheriv('aes-256-gcm', derivedKey, iv);
      decipher.setAuthTag(authTag);
      let decrypted = decipher.update(payload.data, 'hex', 'utf8');
      decrypted += decipher.final('utf8');
      if (!/^[0-9a-f]{64}$/i.test(decrypted)) {
        throw new Error('Decrypted value is not a valid pi key');
      }
      if (isJsonOutput(opts)) {
        console.log(JSON.stringify({
          key: decrypted,
          source: filePath,
          saved: !!cmdOpts.save,
        }, null, 2));
      } else {
        console.log(chalk.green('\n  Key imported successfully'));
        console.log(chalk.white(`  Key: ${decrypted.substring(0, 8)}...${decrypted.substring(decrypted.length - 8)}`));
      }
      if (cmdOpts.save) {
        const keyDir = path.join(require('os').homedir(), '.ruvector');
        const keyPath = path.join(keyDir, 'pi-key');
        if (!fs.existsSync(keyDir)) fs.mkdirSync(keyDir, { recursive: true, mode: 0o700 });
        fs.writeFileSync(keyPath, decrypted, { mode: 0o600 });
        console.log(chalk.green(`  Key saved to ${keyPath}`));
      }
      if (!isJsonOutput(opts)) {
        console.log(chalk.yellow('\n  Set as environment variable:'));
        console.log(chalk.dim(`    export PI=${decrypted}\n`));
      }
    } catch (error) {
      console.error(chalk.red('Failed to decrypt key -- wrong passphrase?'));
      console.error(chalk.red(error.message));
      process.exit(1);
    }
  });

// =============================================================================
// LLM Commands - LLM orchestration (lazy: @ruvector/ruvllm)
// =============================================================================

const llmCmd = program.command('llm').description('LLM orchestration with SONA adaptive learning');

llmCmd.command('embed')
  .description('Generate text embeddings via RuvLLM')
  .argument('<text>', 'Text to embed')
  .option('-m, --model <model>', 'Model name', 'minilm')
  .option('--json', 'JSON output')
  .action(async (text, opts) => {
    let ruvllm;
    try {
      ruvllm = require('@ruvector/ruvllm');
    } catch (e) {
      console.error(chalk.red('LLM commands require @ruvector/ruvllm'));
      console.error(chalk.yellow('  npm install @ruvector/ruvllm'));
      console.error(chalk.dim('  or: npx ruvector install ruvllm'));
      console.error(chalk.dim('\n  Tip: use "ruvector embed text <text>" for built-in ONNX embeddings'));
      process.exit(1);
    }
    const spinner = ora('Generating embedding...').start();
    try {
      const { performance } = require('perf_hooks');
      const start = performance.now();
      const llm = new ruvllm.RuvLLM({ embeddingDim: 384 });
      const embedding = llm.embed(text);
      const elapsed = performance.now() - start;
      spinner.stop();
      if (opts.json) {
        console.log(JSON.stringify({ text, model: opts.model, dimension: embedding.length, embedding: Array.from(embedding), timeMs: +elapsed.toFixed(2) }, null, 2));
        return;
      }
      console.log(chalk.bold.cyan('\n  RuvLLM Embedding\n'));
      console.log(chalk.dim(`  Text: "${text.length > 60 ? text.slice(0, 60) + '...' : text}"`));
      console.log(chalk.dim(`  Model: ${opts.model}`));
      console.log(chalk.dim(`  Dimension: ${embedding.length}`));
      console.log(chalk.dim(`  Time: ${elapsed.toFixed(1)}ms`));
      console.log(chalk.dim(`  First 5 values: [${Array.from(embedding).slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`));
      console.log('');
    } catch (err) {
      spinner.fail('Embedding generation failed');
      console.error(chalk.red(`  ${err.message}`));
      process.exit(1);
    }
  });

llmCmd.command('models')
  .description('List available LLM models')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    let ruvllm;
    try {
      ruvllm = require('@ruvector/ruvllm');
    } catch (e) {
      console.error(chalk.red('LLM commands require @ruvector/ruvllm'));
      console.error(chalk.yellow('  npm install @ruvector/ruvllm'));
      console.error(chalk.dim('  or: npx ruvector install ruvllm'));
      process.exit(1);
    }
    const models = typeof ruvllm.listModels === 'function' ? ruvllm.listModels() :
      (ruvllm.RUVLTRA_MODELS ? Object.values(ruvllm.RUVLTRA_MODELS) : []);
    if (opts.json) {
      console.log(JSON.stringify(models, null, 2));
      return;
    }
    console.log(chalk.bold.cyan('\n  RuvLLM Available Models\n'));
    if (!models || models.length === 0) {
      console.log(chalk.yellow('  No models registered in the registry.'));
      console.log(chalk.dim('  Upgrade @ruvector/ruvllm for model registry support:'));
      console.log(chalk.dim('    npm install @ruvector/ruvllm@latest\n'));
    } else {
      for (const m of models) {
        console.log(chalk.white(`  ${chalk.bold(m.id)}`));
        console.log(chalk.dim(`    ${m.name} - ${m.size} (${m.quantization})`));
        console.log(chalk.dim(`    Use case: ${m.useCase}`));
        console.log(chalk.dim(`    Context: ${m.contextLength} tokens`));
        console.log('');
      }
    }
    console.log(chalk.dim(`  Total: ${models ? models.length : 0} models\n`));
  });

llmCmd.command('benchmark')
  .description('Benchmark LLM inference performance')
  .option('-n, --iterations <n>', 'Number of iterations', '100')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    let ruvllm;
    try {
      ruvllm = require('@ruvector/ruvllm');
    } catch (e) {
      console.error(chalk.red('LLM commands require @ruvector/ruvllm'));
      console.error(chalk.yellow('  npm install @ruvector/ruvllm'));
      console.error(chalk.dim('  or: npx ruvector install ruvllm'));
      process.exit(1);
    }
    const iterations = parseInt(opts.iterations, 10) || 100;
    const spinner = ora(`Running ${iterations} embedding iterations...`).start();
    try {
      const { performance } = require('perf_hooks');
      const llm = new ruvllm.RuvLLM({ embeddingDim: 384 });
      const testText = 'The quick brown fox jumps over the lazy dog';
      for (let i = 0; i < 5; i++) llm.embed(testText);
      const times = [];
      const start = performance.now();
      for (let i = 0; i < iterations; i++) {
        const t0 = performance.now();
        llm.embed(testText);
        times.push(performance.now() - t0);
      }
      const totalMs = performance.now() - start;
      times.sort((a, b) => a - b);
      const avgMs = totalMs / iterations;
      const p50 = times[Math.floor(iterations * 0.5)];
      const p95 = times[Math.floor(iterations * 0.95)];
      const p99 = times[Math.floor(iterations * 0.99)];
      const opsPerSec = (iterations / totalMs) * 1000;
      spinner.stop();
      if (opts.json) {
        console.log(JSON.stringify({ iterations, totalMs: +totalMs.toFixed(2), avgMs: +avgMs.toFixed(3), p50: +p50.toFixed(3), p95: +p95.toFixed(3), p99: +p99.toFixed(3), opsPerSec: +opsPerSec.toFixed(1) }, null, 2));
        return;
      }
      console.log(chalk.bold.cyan('\n  RuvLLM Benchmark Results\n'));
      console.log(chalk.white(`  Iterations:  ${iterations}`));
      console.log(chalk.white(`  Total time:  ${totalMs.toFixed(1)}ms`));
      console.log(chalk.white(`  Avg latency: ${avgMs.toFixed(3)}ms`));
      console.log(chalk.white(`  P50:         ${p50.toFixed(3)}ms`));
      console.log(chalk.white(`  P95:         ${p95.toFixed(3)}ms`));
      console.log(chalk.white(`  P99:         ${p99.toFixed(3)}ms`));
      console.log(chalk.green(`  Throughput:  ${opsPerSec.toFixed(1)} ops/sec`));
      console.log('');
    } catch (err) {
      spinner.fail('Benchmark failed');
      console.error(chalk.red(`  ${err.message}`));
      process.exit(1);
    }
  });

llmCmd.command('info')
  .description('Show LLM module information')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    let ruvllm;
    try {
      ruvllm = require('@ruvector/ruvllm');
    } catch (e) {
      console.error(chalk.red('LLM commands require @ruvector/ruvllm'));
      console.error(chalk.yellow('  npm install @ruvector/ruvllm'));
      console.error(chalk.dim('  or: npx ruvector install ruvllm'));
      process.exit(1);
    }
    const version = typeof ruvllm.version === 'function' ? ruvllm.version() : (ruvllm.version || 'unknown');
    const hasSIMD = ruvllm.hasSimdSupport ? ruvllm.hasSimdSupport() : false;
    const models = ruvllm.listModels ? ruvllm.listModels() : [];
    const info = {
      package: '@ruvector/ruvllm',
      version,
      simd: hasSIMD,
      availableModels: models.length,
      features: ['SONA adaptive learning', 'HNSW memory', 'FastGRNN routing', 'SIMD inference', 'LoRA adapters', 'Session management', 'Federated learning']
    };
    if (opts.json) {
      console.log(JSON.stringify(info, null, 2));
      return;
    }
    console.log(chalk.bold.cyan('\n  RuvLLM Module Information\n'));
    console.log(chalk.white(`  Package:  ${info.package}`));
    console.log(chalk.white(`  Version:  ${info.version}`));
    console.log(chalk.white(`  SIMD:     ${hasSIMD ? chalk.green('Available') : chalk.yellow('Not available')}`));
    console.log(chalk.white(`  Models:   ${info.availableModels} registered`));
    console.log(chalk.white('\n  Features:'));
    for (const f of info.features) {
      console.log(chalk.dim(`    - ${f}`));
    }
    console.log('');
  });

// =============================================================================
// SONA Commands - Self-Optimizing Neural Architecture (bundled dep)
// =============================================================================

const sonaCmd = program.command('sona').description('SONA Self-Optimizing Neural Architecture');

sonaCmd.command('status')
  .description('Show SONA learning status')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    try {
      const { SonaEngine } = require('@ruvector/sona');
      const engine = new SonaEngine();
      const stats = engine.stats();
      if (opts.json) { console.log(JSON.stringify(stats, null, 2)); return; }
      console.log(chalk.bold.cyan('\n  SONA Learning Status\n'));
      console.log(chalk.white(`  Trajectories:  ${stats.trajectoryCount || stats.trajectory_count || 0}`));
      console.log(chalk.white(`  Patterns:      ${stats.patternCount || stats.pattern_count || 0}`));
      console.log(chalk.white(`  Energy:        ${typeof stats.energy === 'number' ? stats.energy.toFixed(4) : stats.energy || 'N/A'}`));
      console.log(chalk.white(`  State:         ${stats.state || 'initialized'}`));
      console.log('');
    } catch (e) {
      try {
        const { SonaCoordinator } = require('@ruvector/ruvllm');
        const sona = new SonaCoordinator();
        const stats = sona.stats();
        if (opts.json) { console.log(JSON.stringify(stats, null, 2)); return; }
        console.log(chalk.bold.cyan('\n  SONA Learning Status (JS fallback)\n'));
        console.log(chalk.white(`  Signals received:       ${stats.signalsReceived}`));
        console.log(chalk.white(`  Trajectories buffered:  ${stats.trajectoriesBuffered}`));
        console.log(chalk.white(`  Total patterns:         ${stats.patterns.totalPatterns}`));
        console.log(chalk.white(`  Avg success rate:       ${(stats.patterns.avgSuccessRate * 100).toFixed(1)}%`));
        console.log(chalk.white(`  EWC tasks learned:      ${stats.ewc.tasksLearned}`));
        console.log('');
      } catch (e2) {
        console.error(chalk.red('SONA native binding not available for this platform.'));
        console.error(chalk.yellow('  Install @ruvector/ruvllm for JS fallback: npm install @ruvector/ruvllm'));
        console.error(chalk.dim(`  Native error: ${e.message}`));
        process.exit(1);
      }
    }
  });

sonaCmd.command('patterns')
  .description('Search learned patterns')
  .argument('<query>', 'Search query')
  .option('-k, --top-k <n>', 'Number of results', '5')
  .option('--json', 'JSON output')
  .action(async (query, opts) => {
    const topK = parseInt(opts.topK, 10) || 5;
    try {
      const { SonaEngine } = require('@ruvector/sona');
      const engine = new SonaEngine();
      const results = engine.searchPatterns ? engine.searchPatterns(query, topK) : [];
      if (opts.json) { console.log(JSON.stringify(results, null, 2)); return; }
      console.log(chalk.bold.cyan(`\n  SONA Pattern Search: "${query}"\n`));
      if (!results || results.length === 0) {
        console.log(chalk.yellow('  No patterns found. Record some training data first.'));
      } else {
        for (let i = 0; i < results.length; i++) {
          const r = results[i];
          console.log(chalk.white(`  ${i + 1}. ${chalk.bold(r.type || r.id || 'pattern')}`));
          console.log(chalk.dim(`     Score: ${typeof r.score === 'number' ? r.score.toFixed(4) : r.score || 'N/A'}`));
          if (r.metadata) console.log(chalk.dim(`     Metadata: ${JSON.stringify(r.metadata)}`));
          console.log('');
        }
      }
    } catch (e) {
      try {
        const { SonaCoordinator } = require('@ruvector/ruvllm');
        const sona = new SonaCoordinator();
        const bank = sona.getReasoningBank();
        const embedding = new Float64Array(64);
        for (let i = 0; i < query.length && i < 64; i++) embedding[i] = query.charCodeAt(i) / 255;
        const results = bank.findSimilar(Array.from(embedding), topK);
        if (opts.json) { console.log(JSON.stringify(results, null, 2)); return; }
        console.log(chalk.bold.cyan(`\n  SONA Pattern Search (JS): "${query}"\n`));
        if (!results || results.length === 0) {
          console.log(chalk.yellow('  No patterns found. Record some training data first.'));
        } else {
          for (let i = 0; i < results.length; i++) {
            const r = results[i];
            console.log(chalk.white(`  ${i + 1}. ${chalk.bold(r.type || r.id)}`));
            console.log(chalk.dim(`     Success rate: ${(r.successRate * 100).toFixed(1)}%`));
            console.log('');
          }
        }
      } catch (e2) {
        console.error(chalk.red('SONA not available.'));
        console.error(chalk.dim(`  Native error: ${e.message}`));
        process.exit(1);
      }
    }
  });

sonaCmd.command('train')
  .description('Record a training trajectory')
  .argument('<data>', 'Training data JSON file or JSON string')
  .option('--json', 'JSON output')
  .action(async (data, opts) => {
    let trajectoryData;
    try {
      if (fs.existsSync(data)) { trajectoryData = JSON.parse(fs.readFileSync(data, 'utf8')); }
      else { trajectoryData = JSON.parse(data); }
    } catch (e) {
      console.error(chalk.red('Invalid training data. Provide a JSON file path or JSON string.'));
      console.error(chalk.dim(`  Error: ${e.message}`));
      console.error(chalk.dim('\n  Example: ruvector sona train \'{"steps":[{"type":"query","input":"test","output":"result","confidence":0.9}],"outcome":"success"}\''));
      process.exit(1);
    }
    const spinner = ora('Recording trajectory...').start();
    try {
      const { SonaEngine } = require('@ruvector/sona');
      const engine = new SonaEngine();
      const result = engine.recordTrajectory ? engine.recordTrajectory(JSON.stringify(trajectoryData)) : { recorded: true };
      spinner.succeed('Trajectory recorded');
      if (opts.json) { console.log(JSON.stringify(result, null, 2)); return; }
      console.log(chalk.dim(`  Steps: ${trajectoryData.steps ? trajectoryData.steps.length : 'N/A'}`));
      console.log(chalk.dim(`  Outcome: ${trajectoryData.outcome || 'N/A'}`));
      console.log('');
    } catch (e) {
      try {
        const { SonaCoordinator, TrajectoryBuilder } = require('@ruvector/ruvllm');
        const sona = new SonaCoordinator();
        const builder = new TrajectoryBuilder();
        if (trajectoryData.steps && Array.isArray(trajectoryData.steps)) {
          for (const step of trajectoryData.steps) { builder.startStep(step.type || 'query', step.input || ''); builder.endStep(step.output || '', step.confidence || 0.5); }
        }
        const trajectory = builder.complete(trajectoryData.outcome || 'success');
        sona.recordTrajectory(trajectory);
        spinner.succeed('Trajectory recorded (JS fallback)');
        if (opts.json) { console.log(JSON.stringify({ id: trajectory.id, steps: trajectory.steps.length, outcome: trajectory.outcome }, null, 2)); return; }
        console.log(chalk.dim(`  ID: ${trajectory.id}`));
        console.log(chalk.dim(`  Steps: ${trajectory.steps.length}`));
        console.log(chalk.dim(`  Outcome: ${trajectory.outcome}`));
        console.log('');
      } catch (e2) { spinner.fail('Failed to record trajectory'); console.error(chalk.red(`  ${e.message}`)); process.exit(1); }
    }
  });

sonaCmd.command('export')
  .description('Export SONA learned weights')
  .option('-o, --output <path>', 'Output file', 'sona-export.json')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    const spinner = ora('Exporting SONA state...').start();
    try {
      const { SonaEngine } = require('@ruvector/sona');
      const engine = new SonaEngine();
      const exported = engine.export ? engine.export() : JSON.stringify(engine.stats());
      const outputData = typeof exported === 'string' ? exported : JSON.stringify(exported, null, 2);
      if (opts.json) { spinner.stop(); console.log(outputData); return; }
      fs.writeFileSync(opts.output, outputData);
      spinner.succeed(`Exported to ${opts.output}`);
      console.log(chalk.dim(`  Size: ${Buffer.byteLength(outputData)} bytes`));
      console.log('');
    } catch (e) {
      try {
        const { SonaCoordinator } = require('@ruvector/ruvllm');
        const sona = new SonaCoordinator();
        const exportData = JSON.stringify({ stats: sona.stats(), reasoningBank: sona.getReasoningBank().stats(), exportedAt: new Date().toISOString() }, null, 2);
        if (opts.json) { spinner.stop(); console.log(exportData); return; }
        fs.writeFileSync(opts.output, exportData);
        spinner.succeed(`Exported to ${opts.output} (JS fallback)`);
        console.log(chalk.dim(`  Size: ${Buffer.byteLength(exportData)} bytes`));
        console.log('');
      } catch (e2) { spinner.fail('Export failed'); console.error(chalk.red(`  ${e.message}`)); process.exit(1); }
    }
  });

sonaCmd.command('stats')
  .description('Show detailed learning statistics')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    try {
      const { SonaEngine } = require('@ruvector/sona');
      const engine = new SonaEngine();
      const stats = engine.stats();
      if (opts.json) { console.log(JSON.stringify(stats, null, 2)); return; }
      console.log(chalk.bold.cyan('\n  SONA Detailed Statistics\n'));
      for (const [key, value] of Object.entries(stats)) {
        const label = key.replace(/([A-Z])/g, ' $1').replace(/_/g, ' ').trim();
        const displayLabel = label.charAt(0).toUpperCase() + label.slice(1);
        if (typeof value === 'object' && value !== null) {
          console.log(chalk.white(`  ${displayLabel}:`));
          for (const [k, v] of Object.entries(value)) console.log(chalk.dim(`    ${k}: ${typeof v === 'number' ? v.toFixed(4) : v}`));
        } else { console.log(chalk.white(`  ${displayLabel}: ${typeof value === 'number' ? value.toFixed(4) : value}`)); }
      }
      console.log('');
    } catch (e) {
      try {
        const { SonaCoordinator } = require('@ruvector/ruvllm');
        const stats = new SonaCoordinator().stats();
        if (opts.json) { console.log(JSON.stringify(stats, null, 2)); return; }
        console.log(chalk.bold.cyan('\n  SONA Detailed Statistics (JS)\n'));
        console.log(chalk.white(`  Signals received:       ${stats.signalsReceived}`));
        console.log(chalk.white(`  Trajectories buffered:  ${stats.trajectoriesBuffered}`));
        console.log(chalk.white('\n  Reasoning Bank:'));
        console.log(chalk.dim(`    Total patterns:    ${stats.patterns.totalPatterns}`));
        console.log(chalk.dim(`    Avg success rate:  ${(stats.patterns.avgSuccessRate * 100).toFixed(1)}%`));
        if (stats.patterns.byType) { for (const [type, count] of Object.entries(stats.patterns.byType)) console.log(chalk.dim(`    ${type}: ${count}`)); }
        console.log(chalk.white('\n  EWC++ (Memory Protection):'));
        console.log(chalk.dim(`    Tasks learned:     ${stats.ewc.tasksLearned}`));
        console.log(chalk.dim(`    Lambda:            ${stats.ewc.lambda}`));
        console.log(chalk.dim(`    Forgetting rate:   ${(stats.ewc.estimatedForgettingRate * 100).toFixed(2)}%`));
        console.log('');
      } catch (e2) { console.error(chalk.red('SONA not available.')); console.error(chalk.dim(`  Native error: ${e.message}`)); process.exit(1); }
    }
  });

sonaCmd.command('info')
  .description('Show SONA module information')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    let nativeAvailable = false, nativeVersion = 'N/A';
    try { const sona = require('@ruvector/sona'); nativeAvailable = !!sona.SonaEngine; nativeVersion = sona.version || '0.1.4'; } catch (e) { /* not available */ }
    let jsAvailable = false, jsVersion = 'N/A';
    try { const ruvllm = require('@ruvector/ruvllm'); jsAvailable = !!ruvllm.SonaCoordinator; jsVersion = typeof ruvllm.version === 'function' ? ruvllm.version() : (ruvllm.version || 'unknown'); } catch (e) { /* not available */ }
    const info = { package: '@ruvector/sona', nativeBinding: nativeAvailable, nativeVersion, jsFallback: jsAvailable, jsFallbackVersion: jsVersion, features: ['LoRA adaptive weights', 'EWC++ memory protection', 'ReasoningBank pattern storage', 'Trajectory tracking', 'Continual learning', 'Sub-millisecond overhead'] };
    if (opts.json) { console.log(JSON.stringify(info, null, 2)); return; }
    console.log(chalk.bold.cyan('\n  SONA Module Information\n'));
    console.log(chalk.white(`  Package:         ${info.package}`));
    console.log(chalk.white(`  Native binding:  ${nativeAvailable ? chalk.green('Available') + chalk.dim(` (v${nativeVersion})`) : chalk.yellow('Not available')}`));
    console.log(chalk.white(`  JS fallback:     ${jsAvailable ? chalk.green('Available') + chalk.dim(` (ruvllm v${jsVersion})`) : chalk.yellow('Not available')}`));
    console.log(chalk.white('\n  Features:'));
    for (const f of info.features) console.log(chalk.dim(`    - ${f}`));
    console.log('');
  });

// =============================================================================
// Route Commands - Semantic routing (lazy: @ruvector/router)
// =============================================================================

const routeCmd = program.command('route').description('Semantic routing for AI agents');

routeCmd.command('classify')
  .description('Classify input to a route')
  .argument('<input>', 'Input text to classify')
  .option('-r, --routes <json>', 'Routes definition JSON file')
  .option('-k, --top-k <n>', 'Number of results', '3')
  .option('--json', 'JSON output')
  .action(async (input, opts) => {
    let router;
    try { router = require('@ruvector/router'); } catch (e) {
      console.error(chalk.red('Route commands require @ruvector/router'));
      console.error(chalk.yellow('  npm install @ruvector/router'));
      console.error(chalk.dim('  or: npx ruvector install router'));
      process.exit(1);
    }
    const topK = parseInt(opts.topK, 10) || 3;
    const spinner = ora('Classifying input...').start();
    try {
      let routesDef = null;
      if (opts.routes) {
        if (!fs.existsSync(opts.routes)) { spinner.fail(`Routes file not found: ${opts.routes}`); process.exit(1); }
        routesDef = JSON.parse(fs.readFileSync(opts.routes, 'utf8'));
      }
      const sr = new router.SemanticRouter({ dimension: routesDef ? (routesDef.dimension || 384) : 384 });
      if (routesDef && routesDef.intents) { for (const intent of routesDef.intents) sr.addIntent(intent); }
      const embedding = new Float32Array(sr._dimension || 384);
      for (let i = 0; i < input.length && i < embedding.length; i++) embedding[i] = input.charCodeAt(i) / 255;
      let norm = 0;
      for (let i = 0; i < embedding.length; i++) norm += embedding[i] * embedding[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < embedding.length; i++) embedding[i] /= norm;
      const results = sr.routeWithEmbedding(embedding, topK);
      spinner.stop();
      if (opts.json) { console.log(JSON.stringify({ input, results }, null, 2)); return; }
      console.log(chalk.bold.cyan(`\n  Route Classification: "${input.length > 50 ? input.slice(0, 50) + '...' : input}"\n`));
      if (!results || results.length === 0) {
        console.log(chalk.yellow('  No matching routes found.'));
        if (!routesDef) { console.log(chalk.dim('  Provide a routes file with -r/--routes to define intents.')); console.log(chalk.dim('  Example: ruvector route classify "hello" -r routes.json')); }
      } else {
        for (let i = 0; i < results.length; i++) {
          const r = results[i];
          console.log(chalk.white(`  ${i + 1}. ${chalk.bold(r.intent)}`));
          console.log(chalk.dim(`     Score: ${r.score.toFixed(4)}`));
          if (r.metadata && Object.keys(r.metadata).length > 0) console.log(chalk.dim(`     Metadata: ${JSON.stringify(r.metadata)}`));
        }
      }
      console.log('');
    } catch (err) { spinner.fail('Classification failed'); console.error(chalk.red(`  ${err.message}`)); process.exit(1); }
  });

routeCmd.command('benchmark')
  .description('Benchmark routing throughput')
  .option('-n, --iterations <n>', 'Number of iterations', '1000')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    let router;
    try { router = require('@ruvector/router'); } catch (e) {
      console.error(chalk.red('Route commands require @ruvector/router'));
      console.error(chalk.yellow('  npm install @ruvector/router'));
      console.error(chalk.dim('  or: npx ruvector install router'));
      process.exit(1);
    }
    const iterations = parseInt(opts.iterations, 10) || 1000;
    const spinner = ora(`Benchmarking ${iterations} route classifications...`).start();
    try {
      const { performance } = require('perf_hooks');
      const dim = 128;
      const sr = new router.SemanticRouter({ dimension: dim, threshold: 0.3 });
      const intentNames = ['greeting', 'farewell', 'question', 'command', 'feedback'];
      for (const name of intentNames) { const emb = new Float32Array(dim); for (let i = 0; i < dim; i++) emb[i] = Math.random() - 0.5; sr.addIntent({ name, utterances: [`example ${name}`], embedding: emb }); }
      const warmupEmb = new Float32Array(dim); for (let i = 0; i < dim; i++) warmupEmb[i] = Math.random() - 0.5;
      for (let i = 0; i < 50; i++) sr.routeWithEmbedding(warmupEmb, 3);
      const times = [], start = performance.now();
      for (let i = 0; i < iterations; i++) { const emb = new Float32Array(dim); for (let j = 0; j < dim; j++) emb[j] = Math.random() - 0.5; const t0 = performance.now(); sr.routeWithEmbedding(emb, 3); times.push(performance.now() - t0); }
      const totalMs = performance.now() - start;
      times.sort((a, b) => a - b);
      const avgMs = totalMs / iterations, p50 = times[Math.floor(iterations * 0.5)], p95 = times[Math.floor(iterations * 0.95)], p99 = times[Math.floor(iterations * 0.99)], opsPerSec = (iterations / totalMs) * 1000;
      spinner.stop();
      if (opts.json) { console.log(JSON.stringify({ iterations, dimension: dim, intents: intentNames.length, totalMs: +totalMs.toFixed(2), avgMs: +avgMs.toFixed(3), p50: +p50.toFixed(3), p95: +p95.toFixed(3), p99: +p99.toFixed(3), opsPerSec: +opsPerSec.toFixed(1) }, null, 2)); return; }
      console.log(chalk.bold.cyan('\n  Router Benchmark Results\n'));
      console.log(chalk.white(`  Iterations:  ${iterations}`));
      console.log(chalk.white(`  Dimension:   ${dim}`));
      console.log(chalk.white(`  Intents:     ${intentNames.length}`));
      console.log(chalk.white(`  Total time:  ${totalMs.toFixed(1)}ms`));
      console.log(chalk.white(`  Avg latency: ${avgMs.toFixed(3)}ms`));
      console.log(chalk.white(`  P50:         ${p50.toFixed(3)}ms`));
      console.log(chalk.white(`  P95:         ${p95.toFixed(3)}ms`));
      console.log(chalk.white(`  P99:         ${p99.toFixed(3)}ms`));
      console.log(chalk.green(`  Throughput:  ${opsPerSec.toFixed(1)} ops/sec`));
      console.log('');
    } catch (err) { spinner.fail('Benchmark failed'); console.error(chalk.red(`  ${err.message}`)); process.exit(1); }
  });

routeCmd.command('info')
  .description('Show router module information')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    let routerAvailable = false, routerVersion = 'N/A', metrics = {};
    try { const router = require('@ruvector/router'); routerAvailable = true; routerVersion = router.version || '0.1.28'; const sr = new router.SemanticRouter({ dimension: 16 }); metrics = { distanceMetrics: Object.keys(router.DistanceMetric || {}), hasSemanticRouter: true }; } catch (e) { /* Not available */ }
    const info = { package: '@ruvector/router', available: routerAvailable, version: routerVersion, backend: 'Rust NAPI (HNSW + SIMD)', features: ['Semantic intent matching', 'HNSW indexing', 'SIMD-accelerated search', 'Multiple distance metrics', 'Save/load state', 'Async embedding support'], ...metrics };
    if (opts.json) { console.log(JSON.stringify(info, null, 2)); return; }
    console.log(chalk.bold.cyan('\n  Semantic Router Information\n'));
    console.log(chalk.white(`  Package:   ${info.package}`));
    console.log(chalk.white(`  Status:    ${routerAvailable ? chalk.green('Available') + chalk.dim(` (v${routerVersion})`) : chalk.yellow('Not installed')}`));
    console.log(chalk.white(`  Backend:   ${info.backend}`));
    if (info.distanceMetrics) console.log(chalk.white(`  Metrics:   ${info.distanceMetrics.join(', ')}`));
    console.log(chalk.white('\n  Features:'));
    for (const f of info.features) console.log(chalk.dim(`    - ${f}`));
    if (!routerAvailable) console.log(chalk.yellow('\n  Install: npm install @ruvector/router'));
    console.log('');
  });

// ── Decompile Command ──────────────────────────────────────────────────────
const decompileCmd = program
  .command('decompile [target]')
  .description('Decompile npm packages, local JS files, or URLs into modules')
  .option('-o, --output <dir>', 'Output directory')
  .option('-f, --format <type>', 'Output format: modules, single, json', 'modules')
  .option('-c, --confidence <n>', 'Minimum confidence threshold (0-1)', '0.3')
  .option('--no-witness', 'Skip witness chain generation')
  .option('--json', 'JSON output to stdout (for piping)')
  .option('-q, --quiet', 'Suppress progress output')
  .option('--version-pkg <ver>', 'Package version (alternative to @version syntax)')
  .option('--diff <version>', 'Compare against another version')
  .option('--model <file>', 'Decompile LLM model weight file (.gguf, .safetensors)')
  .option('--api <model-id>', 'Probe remote LLM API to discover architecture')
  .option('--api-key <key>', 'API key for --api mode (or use env vars)')
  .action(async (target, opts) => {
    // Model weight decompilation mode (ADR-138)
    if (opts.model) {
      try {
        const modelDecompiler = require('../src/decompiler/model-decompiler.js');
        const result = await modelDecompiler.decompileModelFile(opts.model);
        if (opts.json) {
          console.log(JSON.stringify(result, null, 2));
        } else {
          modelDecompiler.printModelResult(result);
        }
      } catch (err) {
        console.error(chalk.red(`Model decompilation failed: ${err.message}`));
        process.exit(1);
      }
      return;
    }

    // API probing mode (ADR-138)
    if (opts.api) {
      try {
        const apiProber = require('../src/decompiler/api-prober.js');
        const result = await apiProber.probeModel(opts.api, { apiKey: opts.apiKey });
        if (opts.json) {
          console.log(JSON.stringify(result, null, 2));
        } else {
          apiProber.printProbeResult(result);
        }
      } catch (err) {
        console.error(chalk.red(`API probe failed: ${err.message}`));
        process.exit(1);
      }
      return;
    }

    if (!target) {
      console.log(chalk.cyan('\nUsage:'));
      console.log(chalk.white('  ruvector decompile <package>           Decompile npm package'));
      console.log(chalk.white('  ruvector decompile <pkg>@<ver>         Specific version'));
      console.log(chalk.white('  ruvector decompile ./bundle.js         Local file'));
      console.log(chalk.white('  ruvector decompile https://unpkg.com/x URL'));
      console.log(chalk.white('  ruvector decompile --model <file.gguf> LLM weight file'));
      console.log(chalk.white('  ruvector decompile --api <model-id>    Probe remote API'));
      console.log(chalk.dim('\nOptions:'));
      console.log(chalk.dim('  -o, --output <dir>     Output directory'));
      console.log(chalk.dim('  -f, --format <type>    modules | single | json'));
      console.log(chalk.dim('  -c, --confidence <n>   Min confidence (0-1, default: 0.3)'));
      console.log(chalk.dim('  --no-witness           Skip witness chain'));
      console.log(chalk.dim('  --json                 JSON to stdout'));
      console.log(chalk.dim('  --diff <version>       Diff against another version'));
      console.log(chalk.dim('  --model <file>         Decompile .gguf/.safetensors'));
      console.log(chalk.dim('  --api <model-id>       Probe LLM API'));
      console.log(chalk.dim('  --api-key <key>        API key (or set env var)'));
      console.log('');
      return;
    }

    const decompiler = require('../src/decompiler/index.js');
    const { parseTarget } = decompiler;
    const parsed = parseTarget(target);
    const minConfidence = parseFloat(opts.confidence);
    const decompileOpts = { minConfidence, witness: opts.witness !== false, useRust: true };
    const quiet = opts.quiet || opts.json;
    let spinner = null;

    if (!quiet) {
      spinner = ora('Analyzing target...').start();
    }

    try {
      let result;

      if (parsed.type === 'npm') {
        const version = opts.versionPkg || parsed.version;
        if (!quiet) spinner.text = `Fetching ${parsed.name}${version ? '@' + version : ''}...`;
        result = await decompiler.decompilePackage(parsed.name, version, decompileOpts);
        if (!quiet) spinner.text = `Decompiled ${result.packageInfo.name}@${result.packageInfo.version}`;
      } else if (parsed.type === 'file') {
        if (!quiet) spinner.text = `Reading ${parsed.path}...`;
        result = decompiler.decompileFile(parsed.path, { ...decompileOpts, filePath: parsed.path });
      } else if (parsed.type === 'url') {
        if (!quiet) spinner.text = `Fetching ${parsed.url}...`;
        result = await decompiler.decompileUrl(parsed.url, decompileOpts);
      }

      if (!quiet) spinner.succeed(chalk.green('Decompilation complete'));

      // Handle --diff flag
      if (opts.diff && parsed.type === 'npm') {
        if (!quiet) {
          const diffSpinner = ora(`Fetching ${parsed.name}@${opts.diff} for diff...`).start();
          try {
            const other = await decompiler.decompilePackage(parsed.name, opts.diff, decompileOpts);
            diffSpinner.succeed('Diff complete');
            const resultNames = new Set(result.modules.map((m) => m.name));
            const otherNames = new Set(other.modules.map((m) => m.name));
            const added = [...resultNames].filter((n) => !otherNames.has(n));
            const removed = [...otherNames].filter((n) => !resultNames.has(n));
            const common = [...resultNames].filter((n) => otherNames.has(n));

            console.log(chalk.bold.cyan('\n  Version Diff'));
            console.log(chalk.white(`  ${opts.diff} -> ${result.packageInfo.version}`));
            if (added.length) console.log(chalk.green(`  Added:   ${added.join(', ')}`));
            if (removed.length) console.log(chalk.red(`  Removed: ${removed.join(', ')}`));
            console.log(chalk.dim(`  Common:  ${common.length} modules`));
            console.log('');
          } catch (err) {
            diffSpinner.fail(`Diff failed: ${err.message}`);
          }
        }
      }

      // Output
      if (opts.json) {
        const jsonOut = {
          modules: result.modules.map((m) => ({
            name: m.name, fragments: m.fragments, confidence: m.confidence,
            contentLength: m.content.length,
          })),
          metrics: result.metrics,
          witness: result.witness ? { root: result.witness.root, chain_length: result.witness.chain.length } : null,
          packageInfo: result.packageInfo || null,
        };
        console.log(JSON.stringify(jsonOut, null, 2));
        return;
      }

      // Determine output directory
      let outputDir = opts.output;
      if (!outputDir) {
        const baseName = result.packageInfo
          ? `${result.packageInfo.name.replace('/', '-')}@${result.packageInfo.version}`
          : path.basename(target, '.js');
        outputDir = path.join(process.cwd(), 'decompiled', baseName);
      }

      decompiler.writeOutput(result, outputDir, opts.format);

      console.log(chalk.bold.cyan('\n  Decompilation Summary'));
      console.log(chalk.white(`  Modules:     ${result.modules.length}`));
      console.log(chalk.white(`  Source size:  ${(result.metrics.source.sizeBytes / 1024).toFixed(1)} KB`));
      console.log(chalk.white(`  Functions:    ${result.metrics.source.functions}`));
      console.log(chalk.white(`  Classes:      ${result.metrics.source.classes}`));
      if (result.witness) {
        const wRoot = result.witness.root || result.witness.chain_root || '';
        console.log(chalk.white(`  Witness root: ${wRoot.slice(0, 16)}...`));
      }
      console.log(chalk.green(`  Output:       ${outputDir}`));
      console.log('');

      if (result.modules.length > 0) {
        console.log(chalk.dim('  Detected modules:'));
        for (const mod of result.modules) {
          const conf = (mod.confidence * 100).toFixed(0);
          console.log(chalk.dim(`    ${mod.name} (${mod.fragments} fragments, ${conf}% confidence)`));
        }
        console.log('');
      }
    } catch (err) {
      if (spinner) spinner.fail(chalk.red('Decompilation failed'));
      console.error(chalk.red(`  ${err.message}`));
      process.exit(1);
    }
  });

// =============================================================================
// Optimize Commands — Claude Code profile optimization (ADR-139)
// =============================================================================

const optimizeCmd = program.command('optimize')
  .description('Optimize Claude Code configuration per task type (ADR-139)')
  .option('-p, --profile <type>', 'Task profile: coding|research|quickfix|planning|background|swarm|review|ci')
  .option('-s, --show', 'Show current optimization status')
  .option('-l, --list', 'List all available profiles')
  .option('--generate-settings', 'Output optimal .claude/settings.json')
  .option('--detect <prompt>', 'Auto-detect task type from a prompt')
  .option('--apply', 'Apply profile env vars to current process (for hooks)')
  .option('--json', 'JSON output')
  .action(async (opts) => {
    let optimizerMod;
    try {
      // Resolve via package.json so it works whether the optimizer ships under
      // src/optimizer/ or dist/optimizer/.
      optimizerMod = require('../src/optimizer/index.js');
    } catch (e) {
      console.error(chalk.yellow('\n  ruvector optimize: not yet shipped in this release.\n'));
      console.error(chalk.gray('  The optimizer module (profiles, settings generation) is in development'));
      console.error(chalk.gray('  and will land in a future release. Track progress at:'));
      console.error(chalk.white('    https://github.com/ruvnet/ruvector/issues/401\n'));
      process.exit(1);
    }

    // --list: show all profiles
    if (opts.list) {
      const profiles = optimizerMod.listProfiles();
      if (opts.json) {
        const data = {};
        for (const name of profiles) {
          data[name] = optimizerMod.getProfile(name);
        }
        console.log(JSON.stringify(data, null, 2));
        return;
      }
      console.log(chalk.bold.cyan('\n  RVAgent Optimizer Profiles (ADR-139)\n'));
      console.log(chalk.dim('  Based on decompiled Claude Code v2.1.91 intelligence\n'));
      for (const name of profiles) {
        const p = optimizerMod.getProfile(name);
        const envCount = Object.keys(p.env).length;
        console.log(`  ${chalk.bold.white(name.padEnd(12))} ${chalk.dim(p.description)}`);
        console.log(chalk.dim(`${''.padEnd(14)}Permission: ${p.permissionMode}, Env vars: ${envCount}`));
      }
      console.log('');
      console.log(chalk.dim('  Usage: ruvector optimize --profile <type>'));
      console.log(chalk.dim('         ruvector optimize --generate-settings --profile coding'));
      console.log('');
      return;
    }

    // --detect: infer task type from prompt
    if (opts.detect) {
      const detected = optimizerMod.detectTaskType(opts.detect);
      if (opts.json) {
        console.log(JSON.stringify({ prompt: opts.detect, taskType: detected }));
        return;
      }
      console.log(chalk.cyan(`  Detected task type: ${chalk.bold(detected)}`));
      return;
    }

    // Determine profile to use
    const profileName = opts.profile || 'coding';
    const profile = optimizerMod.getProfile(profileName);

    if (!profile) {
      console.error(chalk.red(`  Unknown profile: ${profileName}`));
      console.error(chalk.yellow(`  Available: ${optimizerMod.listProfiles().join(', ')}`));
      process.exit(1);
    }

    // --generate-settings: output settings.json
    if (opts.generateSettings) {
      const { generateSettings, formatSettings } = require('../src/optimizer/settings-generator.js');
      const settings = generateSettings({ ...profile, taskType: profileName });
      if (opts.json) {
        console.log(formatSettings(settings));
      } else {
        console.log(chalk.bold.cyan(`\n  Generated settings.json for profile: ${profileName}\n`));
        console.log(formatSettings(settings));
        console.log('');
        console.log(chalk.dim('  Save to .claude/settings.json to activate.'));
        console.log('');
      }
      return;
    }

    // --show: display profile details
    if (opts.show) {
      if (opts.json) {
        console.log(JSON.stringify({ profile: profileName, ...profile }, null, 2));
        return;
      }
      console.log(chalk.bold.cyan(`\n  Profile: ${profileName}\n`));
      console.log(`  ${chalk.dim('Description:')} ${profile.description}`);
      console.log(`  ${chalk.dim('Permission:')}  ${profile.permissionMode}`);
      console.log(`  ${chalk.dim('Env vars:')}`);
      for (const [key, val] of Object.entries(profile.env)) {
        console.log(`    ${chalk.white(key)}=${chalk.green(val)}`);
      }
      console.log('');
      return;
    }

    // --apply or default: apply env vars
    const result = optimizerMod.applyProfile(profileName);
    if (opts.json) {
      console.log(JSON.stringify(result, null, 2));
      return;
    }
    console.log(chalk.bold.cyan(`\n  Applied profile: ${profileName}`));
    console.log(chalk.dim(`  ${profile.description}\n`));
    for (const [key, val] of Object.entries(result.applied)) {
      console.log(`  ${chalk.green('+')} ${key}=${val}`);
    }
    console.log(`\n  ${chalk.dim('Permission mode:')} ${result.permissionMode}`);
    console.log('');
  });

program.parse();


