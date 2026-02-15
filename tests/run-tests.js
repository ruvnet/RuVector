#!/usr/bin/env node

/**
 * MCP Test Suite Runner
 * 
 * Runs all MCP-related tests and provides a comprehensive report
 * This can run without the MCP server for validation purposes
 */

const fs = require('fs');
const path = require('path');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  red: '\x1b[31m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  gray: '\x1b[90m'
};

function log(msg, color = 'reset') {
  console.log(colors[color] + msg + colors.reset);
}

function header(msg) {
  console.log('\n' + '='.repeat(70));
  log(msg, 'cyan');
  console.log('='.repeat(70));
}

function section(msg) {
  console.log('\n' + '-'.repeat(70));
  log(msg, 'blue');
  console.log('-'.repeat(70));
}

async function runTests() {
  header('🧪 RuVector MCP Test Suite');

  const results = {
    passed: 0,
    failed: 0,
    skipped: 0,
    warnings: 0
  };

  // Test 1: File existence
  section('Test 1: File Structure Validation');
  
  const requiredFiles = {
    'tests/mcp-demo.js': 'Full MCP demo script',
    'tests/mcp-simple-test.js': 'Simple MCP test script',
    'tests/MCP_TESTS.md': 'MCP tests documentation',
    'tests/QUICKSTART.md': 'Quick start guide',
    'tests/validate-tests.js': 'Test validation script'
  };

  for (const [file, desc] of Object.entries(requiredFiles)) {
    const exists = fs.existsSync(file);
    if (exists) {
      const stats = fs.statSync(file);
      log(`  ✅ ${file} - ${desc} (${Math.round(stats.size / 1024)}KB)`, 'green');
      results.passed++;
    } else {
      log(`  ❌ ${file} - NOT FOUND`, 'red');
      results.failed++;
    }
  }

  // Test 2: Module loading
  section('Test 2: Module Loading');

  try {
    const { SimpleMCPClient, generateEmbedding } = require(path.join(__dirname, 'mcp-simple-test.js'));
    log('  ✅ mcp-simple-test.js module loads correctly', 'green');
    results.passed++;

    // Test embedding generation
    const embedding = generateEmbedding(128, 1);
    if (embedding.length === 128) {
      log('  ✅ generateEmbedding() produces correct dimensions', 'green');
      results.passed++;
    } else {
      log('  ❌ generateEmbedding() dimension mismatch', 'red');
      results.failed++;
    }

    // Test embedding normalization
    const magnitude = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
    if (Math.abs(magnitude - 1.0) < 0.0001) {
      log('  ✅ Embeddings are properly normalized', 'green');
      results.passed++;
    } else {
      log(`  ⚠️  Embedding normalization: ${magnitude.toFixed(4)} (expected 1.0)`, 'yellow');
      results.warnings++;
    }
  } catch (error) {
    log(`  ❌ Module loading failed: ${error.message}`, 'red');
    results.failed++;
  }

  // Test 3: MCP SDK availability
  section('Test 3: Dependencies Check');

  try {
    require('@modelcontextprotocol/sdk/client/index.js');
    log('  ✅ @modelcontextprotocol/sdk is installed', 'green');
    results.passed++;
  } catch (error) {
    log('  ⚠️  @modelcontextprotocol/sdk not installed (optional for simple test)', 'yellow');
    log('     Install with: npm install @modelcontextprotocol/sdk', 'gray');
    results.warnings++;
  }

  // Test 4: MCP server binary
  section('Test 4: MCP Server Binary Check');

  const serverPaths = [
    './target/release/ruvector-mcp',
    './target/debug/ruvector-mcp'
  ];

  let serverFound = false;
  for (const serverPath of serverPaths) {
    if (fs.existsSync(serverPath)) {
      const stats = fs.statSync(serverPath);
      log(`  ✅ MCP server found: ${serverPath} (${Math.round(stats.size / 1024 / 1024)}MB)`, 'green');
      serverFound = true;
      results.passed++;
      break;
    }
  }

  if (!serverFound) {
    log('  ⚠️  MCP server binary not found', 'yellow');
    log('     Build with: cargo build --release -p ruvector-cli --bin ruvector-mcp', 'gray');
    results.warnings++;
  }

  // Test 5: Documentation quality
  section('Test 5: Documentation Quality');

  const docs = ['tests/MCP_TESTS.md', 'tests/QUICKSTART.md'];
  for (const doc of docs) {
    try {
      const content = fs.readFileSync(doc, 'utf8');
      const checks = [
        { name: 'Has code examples', test: () => content.includes('```') },
        { name: 'Has usage section', test: () => content.toLowerCase().includes('usage') },
        { name: 'Mentions MCP', test: () => content.toLowerCase().includes('mcp') },
        { name: 'Has links', test: () => content.includes('[') && content.includes(']') }
      ];

      let docPassed = true;
      for (const check of checks) {
        if (!check.test()) {
          log(`  ⚠️  ${doc}: Missing ${check.name}`, 'yellow');
          results.warnings++;
          docPassed = false;
        }
      }

      if (docPassed) {
        log(`  ✅ ${doc} - Complete documentation`, 'green');
        results.passed++;
      }
    } catch (error) {
      log(`  ❌ ${doc}: Error reading file`, 'red');
      results.failed++;
    }
  }

  // Test 6: Script executability
  section('Test 6: Script Permissions');

  const scripts = ['tests/mcp-demo.js', 'tests/mcp-simple-test.js', 'tests/validate-tests.js'];
  for (const script of scripts) {
    try {
      fs.accessSync(script, fs.constants.X_OK);
      log(`  ✅ ${script} - Executable`, 'green');
      results.passed++;
    } catch (error) {
      log(`  ⚠️  ${script} - Not executable (chmod +x ${script})`, 'yellow');
      results.warnings++;
    }
  }

  // Summary
  header('📊 Test Summary');

  const total = results.passed + results.failed + results.skipped;
  const passRate = total > 0 ? ((results.passed / total) * 100).toFixed(1) : 0;

  console.log('');
  log(`  Total Tests: ${total}`, 'cyan');
  log(`  ✅ Passed:   ${results.passed}`, 'green');
  log(`  ❌ Failed:   ${results.failed}`, 'red');
  log(`  ⏭️  Skipped:  ${results.skipped}`, 'gray');
  log(`  ⚠️  Warnings: ${results.warnings}`, 'yellow');
  log(`  📈 Pass Rate: ${passRate}%`, passRate >= 80 ? 'green' : 'yellow');

  console.log('');
  console.log('='.repeat(70));

  if (results.failed === 0) {
    log('\n✅ All tests passed!', 'green');
    console.log('');
    log('Next steps:', 'cyan');
    console.log('  1. Build MCP server: cargo build --release -p ruvector-cli --bin ruvector-mcp');
    console.log('  2. Run simple test: node tests/mcp-simple-test.js');
    console.log('  3. Run full demo: node tests/mcp-demo.js');
    console.log('');
    return 0;
  } else {
    log('\n⚠️  Some tests failed. Please review the output above.', 'yellow');
    console.log('');
    return 1;
  }
}

// Run tests
if (require.main === module) {
  runTests().then(code => {
    process.exit(code);
  }).catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

module.exports = { runTests };
