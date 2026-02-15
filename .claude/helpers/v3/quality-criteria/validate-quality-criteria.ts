#!/usr/bin/env npx ts-node
/**
 * Quality Criteria Output Validator
 * Validates HTML output from qe-quality-criteria-recommender agent
 *
 * Usage: npx ts-node scripts/validate-quality-criteria.ts <path-to-html>
 */

import * as fs from 'fs';
import * as path from 'path';

interface ValidationResult {
  passed: boolean;
  gate: string;
  message: string;
  count?: number;
}

interface ValidationReport {
  file: string;
  timestamp: string;
  passed: boolean;
  gatesPassed: number;
  gatesTotal: number;
  results: ValidationResult[];
}

const QUALITY_GATES = [
  {
    name: 'No confidence percentages',
    pattern: /\d+%\s*(confidence|confident)/gi,
    shouldFind: false,
    message: 'Found confidence percentages - use evidence types instead',
  },
  {
    name: 'Evidence types present',
    pattern: /<span class="evidence-type (direct|inferred|claimed)">/gi,
    shouldFind: true,
    minCount: 1,
    message: 'Missing evidence type classifications',
  },
  {
    name: 'Coverage explicitly defined',
    pattern: /\d+\s*of\s*10\s*HTSM\s*Categories/i,
    shouldFind: true,
    message: 'Coverage must show "X of 10 HTSM Categories" format',
  },
  {
    name: 'File:line traceability',
    pattern: /<code>[^<]+\.(ts|js|md|json|html):\d+-\d+<\/code>/gi,
    shouldFind: true,
    minCount: 1,
    message: 'Evidence must have file:line references',
  },
  {
    name: 'No "Lines X-Y" without file',
    pattern: /Lines?\s+\d+[-–]\d+(?![^<]*\.(ts|js|md))/gi,
    shouldFind: false,
    message: 'Found "Lines X-Y" without file path',
  },
  {
    name: 'Correct footer text',
    pattern: /AI Semantic Understanding/i,
    shouldFind: true,
    message: 'Footer must say "AI Semantic Understanding" not "Heuristic analysis"',
  },
  {
    name: 'No heuristic analysis mention',
    pattern: /heuristic\s+analysis|keyword\s+matching/gi,
    shouldFind: false,
    message: 'Found "heuristic analysis" or "keyword matching" - forbidden',
  },
  {
    name: 'Light body background',
    pattern: /body\s*\{[^}]*background:\s*linear-gradient\([^)]*#f5f7fa/i,
    shouldFind: true,
    message: 'Body must have light gradient background (#f5f7fa)',
  },
  {
    name: 'Dark header background',
    pattern: /\.header\s*\{[^}]*background:\s*linear-gradient\([^)]*var\(--primary\)/i,
    shouldFind: true,
    message: 'Header must have dark gradient background (var(--primary))',
  },
  {
    name: 'Reasoning column present',
    pattern: /<th[^>]*>Reasoning<\/th>/i,
    shouldFind: true,
    message: 'Evidence tables must have "Reasoning" column',
  },
];

function validateFile(filePath: string): ValidationReport {
  const content = fs.readFileSync(filePath, 'utf-8');
  const results: ValidationResult[] = [];

  for (const gate of QUALITY_GATES) {
    const matches = content.match(gate.pattern);
    const count = matches?.length || 0;

    let passed: boolean;
    if (gate.shouldFind) {
      passed = gate.minCount ? count >= gate.minCount : count > 0;
    } else {
      passed = count === 0;
    }

    results.push({
      passed,
      gate: gate.name,
      message: passed ? 'OK' : gate.message,
      count,
    });
  }

  const gatesPassed = results.filter(r => r.passed).length;

  return {
    file: path.basename(filePath),
    timestamp: new Date().toISOString(),
    passed: gatesPassed === results.length,
    gatesPassed,
    gatesTotal: results.length,
    results,
  };
}

function printReport(report: ValidationReport): void {
  console.log('\n' + '='.repeat(60));
  console.log('Quality Criteria Validation Report');
  console.log('='.repeat(60));
  console.log(`File: ${report.file}`);
  console.log(`Time: ${report.timestamp}`);
  console.log(`Result: ${report.passed ? '✅ PASSED' : '❌ FAILED'}`);
  console.log(`Gates: ${report.gatesPassed}/${report.gatesTotal}`);
  console.log('-'.repeat(60));

  for (const result of report.results) {
    const icon = result.passed ? '✅' : '❌';
    const countStr = result.count !== undefined ? ` (${result.count} found)` : '';
    console.log(`${icon} ${result.gate}${countStr}`);
    if (!result.passed) {
      console.log(`   → ${result.message}`);
    }
  }

  console.log('='.repeat(60) + '\n');
}

// Main execution
const args = process.argv.slice(2);
if (args.length === 0) {
  console.error('Usage: npx ts-node scripts/validate-quality-criteria.ts <path-to-html>');
  process.exit(1);
}

const filePath = args[0];
if (!fs.existsSync(filePath)) {
  console.error(`File not found: ${filePath}`);
  process.exit(1);
}

const report = validateFile(filePath);
printReport(report);

// Exit with appropriate code
process.exit(report.passed ? 0 : 1);
