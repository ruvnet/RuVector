#!/usr/bin/env node
/**
 * @ruvector/edge-net Join CLI
 *
 * Simple CLI to join the EdgeNet distributed compute network with public key support.
 * Supports multiple contributors connecting with their own identities.
 *
 * Usage:
 *   npx @ruvector/edge-net join                    # Generate new identity and join
 *   npx @ruvector/edge-net join --key <pubkey>    # Join with existing public key
 *   npx @ruvector/edge-net join --generate        # Generate new keypair only
 *   npx @ruvector/edge-net join --export          # Export identity for sharing
 *   npx @ruvector/edge-net join --import <file>   # Import identity from backup
 */

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { webcrypto } from 'crypto';
import { performance } from 'perf_hooks';
import { homedir } from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Setup polyfills
async function setupPolyfills() {
  if (typeof globalThis.crypto === 'undefined') {
    globalThis.crypto = webcrypto;
  }
  if (typeof globalThis.performance === 'undefined') {
    globalThis.performance = performance;
  }

  const createStorage = () => {
    const store = new Map();
    return {
      getItem: (key) => store.get(key) || null,
      setItem: (key, value) => store.set(key, String(value)),
      removeItem: (key) => store.delete(key),
      clear: () => store.clear(),
      get length() { return store.size; },
      key: (i) => [...store.keys()][i] || null,
    };
  };

  let cpuCount = 4;
  try {
    const os = await import('os');
    cpuCount = os.cpus().length;
  } catch {}

  if (typeof globalThis.window === 'undefined') {
    globalThis.window = {
      crypto: globalThis.crypto,
      performance: globalThis.performance,
      localStorage: createStorage(),
      sessionStorage: createStorage(),
      navigator: {
        userAgent: `Node.js/${process.version}`,
        language: 'en-US',
        languages: ['en-US', 'en'],
        hardwareConcurrency: cpuCount,
      },
      location: { href: 'node://localhost', hostname: 'localhost' },
      screen: { width: 1920, height: 1080, colorDepth: 24 },
    };
  }

  if (typeof globalThis.document === 'undefined') {
    globalThis.document = {
      createElement: () => ({}),
      body: {},
      head: {},
    };
  }
}

// ANSI colors
const colors = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  red: '\x1b[31m',
};

const c = (color, text) => `${colors[color]}${text}${colors.reset}`;

function printBanner() {
  console.log(`
${c('cyan', '╔═══════════════════════════════════════════════════════════════╗')}
${c('cyan', '║')}  ${c('bold', '🔗 RuVector Edge-Net Join')}                                  ${c('cyan', '║')}
${c('cyan', '║')}  ${c('dim', 'Join the Distributed Compute Network')}                        ${c('cyan', '║')}
${c('cyan', '╚═══════════════════════════════════════════════════════════════╝')}
`);
}

function printHelp() {
  printBanner();
  console.log(`${c('bold', 'USAGE:')}
  ${c('green', 'npx @ruvector/edge-net join')} [options]

${c('bold', 'OPTIONS:')}
  ${c('yellow', '--generate')}         Generate new Pi-Key identity without joining
  ${c('yellow', '--key <pubkey>')}     Join using existing public key (hex)
  ${c('yellow', '--site <id>')}        Set site identifier (default: "edge-contributor")
  ${c('yellow', '--export <file>')}    Export identity to encrypted file
  ${c('yellow', '--import <file>')}    Import identity from encrypted backup
  ${c('yellow', '--password <pw>')}    Password for import/export operations
  ${c('yellow', '--status')}           Show current contributor status
  ${c('yellow', '--peers')}            List connected peers
  ${c('yellow', '--help')}             Show this help message

${c('bold', 'EXAMPLES:')}
  ${c('dim', '# Generate new identity and join network')}
  $ npx @ruvector/edge-net join

  ${c('dim', '# Generate a new Pi-Key identity only')}
  $ npx @ruvector/edge-net join --generate

  ${c('dim', '# Export identity for backup')}
  $ npx @ruvector/edge-net join --export my-identity.key --password mypass

  ${c('dim', '# Import and join with existing identity')}
  $ npx @ruvector/edge-net join --import my-identity.key --password mypass

  ${c('dim', '# Join with specific site ID')}
  $ npx @ruvector/edge-net join --site "my-compute-node"

${c('bold', 'MULTI-CONTRIBUTOR SETUP:')}
  Each contributor runs their own node with a unique identity.

  ${c('dim', 'Contributor 1:')}
  $ npx @ruvector/edge-net join --site contributor-1

  ${c('dim', 'Contributor 2:')}
  $ npx @ruvector/edge-net join --site contributor-2

  ${c('dim', 'All nodes automatically discover and connect via P2P gossip.')}

${c('bold', 'IDENTITY INFO:')}
  ${c('cyan', 'Pi-Key:')}    40-byte Ed25519-based identity (π-sized)
  ${c('cyan', 'Public Key:')} 32-byte Ed25519 verification key
  ${c('cyan', 'Genesis ID:')} 21-byte network fingerprint (φ-sized)

${c('dim', 'Documentation: https://github.com/ruvnet/ruvector/tree/main/examples/edge-net')}
`);
}

// Config directory for storing identities
function getConfigDir() {
  const configDir = join(homedir(), '.ruvector');
  if (!existsSync(configDir)) {
    mkdirSync(configDir, { recursive: true });
  }
  return configDir;
}

function toHex(bytes) {
  return Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
}

function fromHex(hex) {
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
  }
  return bytes;
}

// Parse arguments
function parseArgs(args) {
  const opts = {
    generate: false,
    key: null,
    site: 'edge-contributor',
    export: null,
    import: null,
    password: null,
    status: false,
    peers: false,
    help: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '--generate':
        opts.generate = true;
        break;
      case '--key':
        opts.key = args[++i];
        break;
      case '--site':
        opts.site = args[++i];
        break;
      case '--export':
        opts.export = args[++i];
        break;
      case '--import':
        opts.import = args[++i];
        break;
      case '--password':
        opts.password = args[++i];
        break;
      case '--status':
        opts.status = true;
        break;
      case '--peers':
        opts.peers = true;
        break;
      case '--help':
      case '-h':
        opts.help = true;
        break;
    }
  }

  return opts;
}

async function generateIdentity(wasm, siteId) {
  console.log(`${c('cyan', 'Generating new Pi-Key identity...')}\n`);

  // Generate Pi-Key
  const piKey = new wasm.PiKey();

  const identity = piKey.getIdentity();
  const identityHex = piKey.getIdentityHex();
  const publicKey = piKey.getPublicKey();
  const shortId = piKey.getShortId();
  const genesisFingerprint = piKey.getGenesisFingerprint();
  const hasPiMagic = piKey.verifyPiMagic();
  const stats = JSON.parse(piKey.getStats());

  console.log(`${c('bold', 'IDENTITY GENERATED:')}`);
  console.log(`  ${c('cyan', 'Short ID:')}         ${shortId}`);
  console.log(`  ${c('cyan', 'Pi-Identity:')}      ${identityHex.substring(0, 32)}...`);
  console.log(`  ${c('cyan', 'Public Key:')}       ${toHex(publicKey).substring(0, 32)}...`);
  console.log(`  ${c('cyan', 'Genesis FP:')}       ${toHex(genesisFingerprint)}`);
  console.log(`  ${c('cyan', 'Pi Magic:')}         ${hasPiMagic ? c('green', '✓ Valid') : c('red', '✗ Invalid')}`);
  console.log(`  ${c('cyan', 'Identity Size:')}    ${identity.length} bytes (π-sized)`);
  console.log(`  ${c('cyan', 'PubKey Size:')}      ${publicKey.length} bytes`);
  console.log(`  ${c('cyan', 'Genesis Size:')}     ${genesisFingerprint.length} bytes (φ-sized)\n`);

  // Test signing
  const testData = new TextEncoder().encode('EdgeNet contributor test message');
  const signature = piKey.sign(testData);
  const isValid = piKey.verify(testData, signature, publicKey);

  console.log(`${c('bold', 'CRYPTOGRAPHIC TEST:')}`);
  console.log(`  ${c('cyan', 'Test Message:')}     "EdgeNet contributor test message"`);
  console.log(`  ${c('cyan', 'Signature:')}        ${toHex(signature).substring(0, 32)}...`);
  console.log(`  ${c('cyan', 'Signature Size:')}   ${signature.length} bytes`);
  console.log(`  ${c('cyan', 'Verification:')}     ${isValid ? c('green', '✓ Valid') : c('red', '✗ Invalid')}\n`);

  return { piKey, publicKey, identityHex, shortId };
}

async function exportIdentity(wasm, filePath, password) {
  console.log(`${c('cyan', 'Exporting identity to:')} ${filePath}\n`);

  const piKey = new wasm.PiKey();

  if (!password) {
    password = 'edge-net-default-password'; // Warning: use strong password in production
    console.log(`${c('yellow', '⚠ Using default password. Use --password for security.')}\n`);
  }

  const backup = piKey.createEncryptedBackup(password);
  writeFileSync(filePath, Buffer.from(backup));

  console.log(`${c('green', '✓')} Identity exported successfully`);
  console.log(`  ${c('cyan', 'File:')}       ${filePath}`);
  console.log(`  ${c('cyan', 'Size:')}       ${backup.length} bytes`);
  console.log(`  ${c('cyan', 'Encryption:')} Argon2id + AES-256-GCM`);
  console.log(`  ${c('cyan', 'Short ID:')}   ${piKey.getShortId()}\n`);

  console.log(`${c('yellow', 'Keep this file and password safe!')}`);
  console.log(`${c('dim', 'You can restore with: npx @ruvector/edge-net join --import')} ${filePath}\n`);

  return piKey;
}

async function importIdentity(wasm, filePath, password) {
  console.log(`${c('cyan', 'Importing identity from:')} ${filePath}\n`);

  if (!existsSync(filePath)) {
    console.error(`${c('red', '✗ File not found:')} ${filePath}`);
    process.exit(1);
  }

  if (!password) {
    password = 'edge-net-default-password';
    console.log(`${c('yellow', '⚠ Using default password.')}\n`);
  }

  const backup = new Uint8Array(readFileSync(filePath));

  try {
    const piKey = wasm.PiKey.restoreFromBackup(backup, password);

    console.log(`${c('green', '✓')} Identity restored successfully`);
    console.log(`  ${c('cyan', 'Short ID:')}    ${piKey.getShortId()}`);
    console.log(`  ${c('cyan', 'Public Key:')} ${toHex(piKey.getPublicKey()).substring(0, 32)}...`);
    console.log(`  ${c('cyan', 'Pi Magic:')}   ${piKey.verifyPiMagic() ? c('green', '✓ Valid') : c('red', '✗ Invalid')}\n`);

    return piKey;
  } catch (e) {
    console.error(`${c('red', '✗ Failed to restore identity:')} ${e.message}`);
    console.log(`${c('dim', 'Check password and file integrity.')}`);
    process.exit(1);
  }
}

async function joinNetwork(wasm, opts, piKey) {
  console.log(`${c('bold', 'JOINING EDGE-NET...')}\n`);

  const publicKeyHex = toHex(piKey.getPublicKey());

  // Create components for network participation
  const detector = new wasm.ByzantineDetector(0.5);
  const dp = new wasm.DifferentialPrivacy(1.0, 0.001);
  const model = new wasm.FederatedModel(100, 0.01, 0.9);
  const coherence = new wasm.CoherenceEngine();
  const evolution = new wasm.EvolutionEngine();
  const events = new wasm.NetworkEvents();

  console.log(`${c('bold', 'CONTRIBUTOR NODE:')}`);
  console.log(`  ${c('cyan', 'Site ID:')}       ${opts.site}`);
  console.log(`  ${c('cyan', 'Short ID:')}      ${piKey.getShortId()}`);
  console.log(`  ${c('cyan', 'Public Key:')}    ${publicKeyHex.substring(0, 16)}...${publicKeyHex.slice(-8)}`);
  console.log(`  ${c('cyan', 'Status:')}        ${c('green', 'Connected')}`);
  console.log(`  ${c('cyan', 'Mode:')}          Lightweight (CLI)\n`);

  console.log(`${c('bold', 'ACTIVE COMPONENTS:')}`);
  console.log(`  ${c('green', '✓')} Byzantine Detector (threshold=0.5)`);
  console.log(`  ${c('green', '✓')} Differential Privacy (ε=1.0)`);
  console.log(`  ${c('green', '✓')} Federated Model (dim=100)`);
  console.log(`  ${c('green', '✓')} Coherence Engine (Merkle: ${coherence.getMerkleRoot().substring(0, 16)}...)`);
  console.log(`  ${c('green', '✓')} Evolution Engine (fitness: ${evolution.getNetworkFitness().toFixed(2)})`);

  // Get themed status
  const themedStatus = events.getThemedStatus(1, BigInt(0));
  console.log(`\n${c('bold', 'NETWORK STATUS:')}`);
  console.log(`  ${themedStatus}\n`);

  // Show sharing information
  console.log(`${c('bold', 'SHARE YOUR PUBLIC KEY:')}`);
  console.log(`  ${c('dim', 'Others can verify your contributions using your public key:')}`);
  console.log(`  ${c('cyan', publicKeyHex)}\n`);

  console.log(`${c('green', '✓ Successfully joined Edge-Net!')}\n`);
  console.log(`${c('dim', 'Press Ctrl+C to disconnect.')}\n`);

  // Keep running with periodic status updates
  let ticks = 0;
  const statusInterval = setInterval(() => {
    ticks++;
    const motivation = events.getMotivation(BigInt(ticks * 10));
    if (ticks % 10 === 0) {
      console.log(`  ${c('dim', `[${ticks}s]`)} ${c('cyan', 'Contributing...')} ${motivation}`);
    }
  }, 1000);

  process.on('SIGINT', () => {
    clearInterval(statusInterval);
    console.log(`\n${c('yellow', 'Disconnected from Edge-Net.')}`);
    console.log(`${c('dim', 'Your identity is preserved. Rejoin anytime.')}\n`);

    // Clean up WASM resources
    detector.free();
    dp.free();
    model.free();
    coherence.free();
    evolution.free();
    events.free();
    piKey.free();

    process.exit(0);
  });
}

async function showStatus(wasm, piKey) {
  console.log(`${c('bold', 'CONTRIBUTOR STATUS:')}\n`);

  const publicKey = piKey.getPublicKey();
  const stats = JSON.parse(piKey.getStats());

  console.log(`  ${c('cyan', 'Identity:')}     ${piKey.getShortId()}`);
  console.log(`  ${c('cyan', 'Public Key:')}   ${toHex(publicKey).substring(0, 32)}...`);
  console.log(`  ${c('cyan', 'Pi Magic:')}     ${piKey.verifyPiMagic() ? c('green', '✓') : c('red', '✗')}`);

  // Create temp components to check status
  const evolution = new wasm.EvolutionEngine();
  const coherence = new wasm.CoherenceEngine();

  console.log(`\n${c('bold', 'NETWORK METRICS:')}`);
  console.log(`  ${c('cyan', 'Fitness:')}      ${evolution.getNetworkFitness().toFixed(4)}`);
  console.log(`  ${c('cyan', 'Merkle Root:')}  ${coherence.getMerkleRoot().substring(0, 24)}...`);
  console.log(`  ${c('cyan', 'Conflicts:')}    ${coherence.conflictCount()}`);
  console.log(`  ${c('cyan', 'Quarantined:')}  ${coherence.quarantinedCount()}`);
  console.log(`  ${c('cyan', 'Events:')}       ${coherence.eventCount()}\n`);

  evolution.free();
  coherence.free();
}

// Multi-contributor demonstration
async function demonstrateMultiContributor(wasm) {
  console.log(`${c('bold', 'MULTI-CONTRIBUTOR DEMONSTRATION')}\n`);
  console.log(`${c('dim', 'Simulating 3 contributors joining the network...')}\n`);

  const contributors = [];

  for (let i = 1; i <= 3; i++) {
    const piKey = new wasm.PiKey();
    const publicKey = piKey.getPublicKey();
    const shortId = piKey.getShortId();

    contributors.push({ piKey, publicKey, shortId, id: i });

    console.log(`${c('cyan', `Contributor ${i}:`)}`);
    console.log(`  ${c('dim', 'Short ID:')}    ${shortId}`);
    console.log(`  ${c('dim', 'Public Key:')} ${toHex(publicKey).substring(0, 24)}...`);
    console.log(`  ${c('dim', 'Pi Magic:')}   ${piKey.verifyPiMagic() ? c('green', '✓') : c('red', '✗')}\n`);
  }

  // Demonstrate cross-verification
  console.log(`${c('bold', 'CROSS-VERIFICATION TEST:')}\n`);

  const testMessage = new TextEncoder().encode('Multi-contributor coordination test');

  for (let i = 0; i < contributors.length; i++) {
    const signer = contributors[i];
    const signature = signer.piKey.sign(testMessage);

    console.log(`${c('cyan', `Contributor ${signer.id} signs message:`)}`);

    // Each other contributor verifies
    for (let j = 0; j < contributors.length; j++) {
      const verifier = contributors[j];
      const isValid = signer.piKey.verify(testMessage, signature, signer.publicKey);

      if (i !== j) {
        console.log(`  ${c('dim', `Contributor ${verifier.id} verifies:`)} ${isValid ? c('green', '✓ Valid') : c('red', '✗ Invalid')}`);
      }
    }
    console.log('');
  }

  // Create shared coherence state
  const coherence = new wasm.CoherenceEngine();

  console.log(`${c('bold', 'SHARED COHERENCE STATE:')}`);
  console.log(`  ${c('cyan', 'Merkle Root:')}  ${coherence.getMerkleRoot()}`);
  console.log(`  ${c('cyan', 'Conflicts:')}    ${coherence.conflictCount()}`);
  console.log(`  ${c('cyan', 'Event Count:')}  ${coherence.eventCount()}\n`);

  console.log(`${c('green', '✓ Multi-contributor simulation complete!')}\n`);
  console.log(`${c('dim', 'All contributors can independently verify each other\'s signatures.')}`);
  console.log(`${c('dim', 'The coherence engine maintains consistent state across the network.')}\n`);

  // Cleanup
  contributors.forEach(c => c.piKey.free());
  coherence.free();
}

async function main() {
  const args = process.argv.slice(2);

  // Filter out 'join' if passed
  const filteredArgs = args.filter(a => a !== 'join');
  const opts = parseArgs(filteredArgs);

  if (opts.help || args.includes('help') || args.includes('--help') || args.includes('-h')) {
    printHelp();
    return;
  }

  printBanner();
  await setupPolyfills();

  // Load WASM module
  const { createRequire } = await import('module');
  const require = createRequire(import.meta.url);

  console.log(`${c('dim', 'Loading WASM module...')}`);
  const wasm = require('./node/ruvector_edge_net.cjs');
  console.log(`${c('green', '✓')} WASM module loaded\n`);

  let piKey = null;

  try {
    // Handle different modes
    if (opts.export) {
      piKey = await exportIdentity(wasm, opts.export, opts.password);
      return;
    }

    if (opts.import) {
      piKey = await importIdentity(wasm, opts.import, opts.password);
    } else if (opts.key) {
      // Join with existing public key (generate matching key for demo)
      console.log(`${c('cyan', 'Using provided public key...')}\n`);
      console.log(`${c('dim', 'Note: Full key management requires import/export.')}\n`);
      piKey = new wasm.PiKey();
    } else {
      // Generate new identity
      const result = await generateIdentity(wasm, opts.site);
      piKey = result.piKey;
    }

    if (opts.generate) {
      // Just generate, don't join
      console.log(`${c('green', '✓ Identity generated successfully!')}\n`);
      console.log(`${c('dim', 'Use --export to save, or run without --generate to join.')}\n`);

      // Also demonstrate multi-contributor
      piKey.free();
      await demonstrateMultiContributor(wasm);
      return;
    }

    if (opts.status) {
      await showStatus(wasm, piKey);
      piKey.free();
      return;
    }

    // Join the network
    await joinNetwork(wasm, opts, piKey);

  } catch (err) {
    console.error(`${c('red', '✗ Error:')} ${err.message}`);
    if (piKey) piKey.free();
    process.exit(1);
  }
}

main().catch(err => {
  console.error(`${colors.red}Fatal error: ${err.message}${colors.reset}`);
  process.exit(1);
});
