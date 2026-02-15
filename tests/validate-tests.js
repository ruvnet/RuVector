#!/usr/bin/env node

/**
 * MCP Test Validation Script
 * 
 * This script validates that the MCP test scripts are properly structured
 * without requiring the actual MCP server to be running.
 */

const fs = require('fs');
const path = require('path');

console.log('🔍 Validating MCP Test Scripts\n');
console.log('='.repeat(70) + '\n');

const testsDir = __dirname;
const testFiles = [
  'mcp-demo.js',
  'mcp-simple-test.js'
];

const docFiles = [
  'MCP_TESTS.md',
  'QUICKSTART.md'
];

let allValid = true;

// Check test files
console.log('📝 Checking test script files:\n');
for (const file of testFiles) {
  const filePath = path.join(testsDir, file);
  
  try {
    // Check if file exists
    if (!fs.existsSync(filePath)) {
      console.log(`   ❌ ${file} - NOT FOUND`);
      allValid = false;
      continue;
    }

    // Check if file is readable
    const content = fs.readFileSync(filePath, 'utf8');
    
    // Check file size
    const stats = fs.statSync(filePath);
    
    // Check for key components
    const checks = {
      'Has shebang': content.startsWith('#!/usr/bin/env node'),
      'Has exports': content.includes('module.exports'),
      'Has main function': content.includes('async function') || content.includes('function'),
      'Has MCP references': content.includes('MCP') || content.includes('mcp'),
      'Proper size': stats.size > 1000
    };

    const allChecksPassed = Object.values(checks).every(v => v);
    
    if (allChecksPassed) {
      console.log(`   ✅ ${file} - VALID (${Math.round(stats.size / 1024)}KB)`);
    } else {
      console.log(`   ⚠️  ${file} - WARNINGS:`);
      for (const [check, passed] of Object.entries(checks)) {
        if (!passed) {
          console.log(`      - ${check}: FAILED`);
        }
      }
    }

    // Try to parse as JavaScript
    require(filePath);
    
  } catch (error) {
    console.log(`   ❌ ${file} - ERROR: ${error.message}`);
    allValid = false;
  }
}

// Check documentation files
console.log('\n📚 Checking documentation files:\n');
for (const file of docFiles) {
  const filePath = path.join(testsDir, file);
  
  try {
    if (!fs.existsSync(filePath)) {
      console.log(`   ❌ ${file} - NOT FOUND`);
      allValid = false;
      continue;
    }

    const content = fs.readFileSync(filePath, 'utf8');
    const stats = fs.statSync(filePath);
    
    const checks = {
      'Has headers': content.includes('#'),
      'Has code blocks': content.includes('```'),
      'Mentions MCP': content.toLowerCase().includes('mcp'),
      'Has usage examples': content.includes('Usage') || content.includes('usage'),
      'Proper size': stats.size > 500
    };

    const allChecksPassed = Object.values(checks).every(v => v);
    
    if (allChecksPassed) {
      console.log(`   ✅ ${file} - VALID (${Math.round(stats.size / 1024)}KB)`);
    } else {
      console.log(`   ⚠️  ${file} - WARNINGS:`);
      for (const [check, passed] of Object.entries(checks)) {
        if (!passed) {
          console.log(`      - ${check}: FAILED`);
        }
      }
    }
    
  } catch (error) {
    console.log(`   ❌ ${file} - ERROR: ${error.message}`);
    allValid = false;
  }
}

// Check for executable permissions
console.log('\n🔐 Checking file permissions:\n');
for (const file of testFiles) {
  const filePath = path.join(testsDir, file);
  
  try {
    fs.accessSync(filePath, fs.constants.X_OK);
    console.log(`   ✅ ${file} - EXECUTABLE`);
  } catch (error) {
    console.log(`   ⚠️  ${file} - NOT EXECUTABLE (run: chmod +x ${file})`);
  }
}

// Check for required dependencies
console.log('\n📦 Checking dependencies:\n');

const requiredModules = [
  '@modelcontextprotocol/sdk',
  'readline',
  'child_process',
  'path',
  'fs'
];

for (const module of requiredModules) {
  try {
    // Built-in modules don't need to be installed
    if (['readline', 'child_process', 'path', 'fs', 'os'].includes(module)) {
      console.log(`   ✅ ${module} - BUILT-IN`);
      continue;
    }

    // Try to resolve the module
    require.resolve(module);
    console.log(`   ✅ ${module} - INSTALLED`);
  } catch (error) {
    console.log(`   ⚠️  ${module} - NOT FOUND (run: npm install ${module})`);
  }
}

// Summary
console.log('\n' + '='.repeat(70));
if (allValid) {
  console.log('\n✅ All validation checks passed!\n');
  console.log('Next steps:');
  console.log('  1. Build MCP server: cargo build --release -p ruvector-cli --bin ruvector-mcp');
  console.log('  2. Run tests: node tests/mcp-simple-test.js\n');
  process.exit(0);
} else {
  console.log('\n⚠️  Some validation checks failed. Please review the output above.\n');
  process.exit(1);
}
