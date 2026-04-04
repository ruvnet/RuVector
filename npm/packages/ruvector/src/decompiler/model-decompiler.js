'use strict';

/**
 * LLM model weight decompiler for Node.js.
 * Parses GGUF and Safetensors files to reconstruct architecture info.
 * See ADR-138.
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

// ── GGUF constants ───────────────────────────────────────────────────────

const GGUF_MAGIC = 0x46554747;

const QUANT_TYPES = {
  0: { name: 'F32', bpw: 32 }, 1: { name: 'F16', bpw: 16 },
  2: { name: 'Q4_0', bpw: 4.5 }, 3: { name: 'Q4_1', bpw: 5 },
  6: { name: 'Q5_0', bpw: 5.5 }, 7: { name: 'Q5_1', bpw: 6 },
  8: { name: 'Q8_0', bpw: 8.5 }, 9: { name: 'Q8_1', bpw: 9 },
  10: { name: 'Q2_K', bpw: 2.56 }, 11: { name: 'Q3_K', bpw: 3.44 },
  12: { name: 'Q4_K', bpw: 4.5 }, 13: { name: 'Q5_K', bpw: 5.5 },
  14: { name: 'Q6_K', bpw: 6.56 }, 15: { name: 'Q8_K', bpw: 8.5 },
  29: { name: 'BF16', bpw: 16 },
};

// ── Main entry ───────────────────────────────────────────────────────────

async function decompileModelFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.gguf') return decompileGguf(filePath);
  if (ext === '.safetensors') return decompileSafetensors(filePath);
  throw new Error(`Unsupported model format: ${ext} (expected .gguf or .safetensors)`);
}

// ── GGUF decompiler ──────────────────────────────────────────────────────

function decompileGguf(filePath) {
  const fd = fs.openSync(filePath, 'r');
  const stat = fs.fstatSync(fd);
  let pos = 0;

  function readBuf(n) {
    const buf = Buffer.alloc(n);
    fs.readSync(fd, buf, 0, n, pos);
    pos += n;
    return buf;
  }
  function readU32() { return readBuf(4).readUInt32LE(0); }
  function readU64() { return Number(readBuf(8).readBigUInt64LE(0)); }
  function readF32() { return readBuf(4).readFloatLE(0); }
  function readF64() { return readBuf(8).readDoubleLE(0); }
  function readStr() {
    const len = readU64();
    if (len > 65536) throw new Error(`String too long: ${len}`);
    return readBuf(len).toString('utf8');
  }

  function readValue() {
    const type = readU32();
    switch (type) {
      case 0: return readBuf(1).readUInt8(0);
      case 1: return readBuf(1).readInt8(0);
      case 2: return readBuf(2).readUInt16LE(0);
      case 3: return readBuf(2).readInt16LE(0);
      case 4: return readU32();
      case 5: return readBuf(4).readInt32LE(0);
      case 6: return readF32();
      case 7: return readBuf(1).readUInt8(0) !== 0;
      case 8: return readStr();
      case 9: { // Array
        const elemType = readU32();
        const count = readU64();
        const arr = [];
        for (let i = 0; i < Math.min(count, 10000); i++) {
          if (elemType === 8) arr.push(readStr());
          else if (elemType === 4) arr.push(readU32());
          else if (elemType === 0) arr.push(readBuf(1).readUInt8(0));
          else if (elemType === 5) arr.push(readBuf(4).readInt32LE(0));
          else if (elemType === 6) arr.push(readF32());
          else if (elemType === 10) arr.push(readU64());
          else readBuf(elemType <= 1 ? 1 : elemType <= 3 ? 2 : elemType <= 6 ? 4 : 8);
        }
        // Skip remaining if array was truncated
        if (count > 10000) {
          // Cannot reliably skip variable-size elements, just return what we have
        }
        return arr;
      }
      case 10: return readU64();
      case 11: return Number(readBuf(8).readBigInt64LE(0));
      case 12: return readF64();
      default: throw new Error(`Unknown value type: ${type}`);
    }
  }

  // Parse header
  const magic = readU32();
  if (magic !== GGUF_MAGIC) throw new Error(`Not a GGUF file (magic: 0x${magic.toString(16)})`);
  const version = readU32();
  const tensorCount = readU64();
  const metadataCount = readU64();

  // Parse metadata
  const metadata = {};
  for (let i = 0; i < metadataCount; i++) {
    const key = readStr();
    metadata[key] = readValue();
  }

  // Parse tensor infos
  const tensors = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = readStr();
    const nDims = readU32();
    const shape = [];
    for (let d = 0; d < nDims; d++) shape.push(readU64());
    const quantType = readU32();
    const offset = readU64();
    const qt = QUANT_TYPES[quantType] || { name: `Unknown(${quantType})`, bpw: 0 };
    tensors.push({ name, shape, quantType, quantName: qt.name, bpw: qt.bpw, offset });
  }

  fs.closeSync(fd);

  return buildResult({
    format: `GGUF v${version}`,
    metadata,
    tensors,
    fileSize: stat.size,
    filePath,
  });
}

// ── Safetensors decompiler ───────────────────────────────────────────────

function decompileSafetensors(filePath) {
  const fd = fs.openSync(filePath, 'r');
  const stat = fs.fstatSync(fd);
  const lenBuf = Buffer.alloc(8);
  fs.readSync(fd, lenBuf, 0, 8, 0);
  const headerLen = Number(lenBuf.readBigUInt64LE(0));
  if (headerLen > 100 * 1024 * 1024) throw new Error(`Header too large: ${headerLen}`);

  const headerBuf = Buffer.alloc(headerLen);
  fs.readSync(fd, headerBuf, 0, headerLen, 8);
  fs.closeSync(fd);

  const header = JSON.parse(headerBuf.toString('utf8'));
  const metadata = {};
  const tensors = [];

  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') {
      Object.assign(metadata, info);
      continue;
    }
    if (!info || !info.dtype) continue;
    const dtypeMap = { F32: 32, F16: 16, BF16: 16, F64: 64, I8: 8, I16: 16, I32: 32, I64: 64 };
    tensors.push({
      name,
      shape: info.shape || [],
      quantName: info.dtype,
      bpw: dtypeMap[info.dtype] || 32,
      offset: info.data_offsets ? info.data_offsets[0] : 0,
    });
  }

  tensors.sort((a, b) => a.offset - b.offset);

  return buildResult({
    format: 'Safetensors',
    metadata,
    tensors,
    fileSize: stat.size,
    filePath,
  });
}

// ── Architecture inference ───────────────────────────────────────────────

function buildResult({ format, metadata, tensors, fileSize, filePath }) {
  const arch = inferArchitecture(metadata, tensors);
  const quant = detectQuantization(tensors, arch);
  const layers = extractLayers(tensors, arch);

  // Witness: SHA3 not available in Node crypto, use SHA256
  const hash = crypto.createHash('sha256').update(filePath).digest('hex');

  return {
    format,
    architecture: arch,
    layers: layers.slice(0, 50), // Limit output
    tokenizer: extractTokenizer(metadata),
    quantization: quant,
    witness: { source_hash: hash, chain_root: hash.slice(0, 32) },
    metadata: flattenMetadata(metadata),
    fileSize,
  };
}

function inferArchitecture(metadata, tensors) {
  const archKey = metadata['general.architecture'] || '';
  const prefix = archKey ? `${archKey}.` : '';
  const hiddenSize = Number(metadata[`${prefix}embedding_length`]) || inferHiddenSize(tensors);
  const numLayers = Number(metadata[`${prefix}block_count`]) || inferNumLayers(tensors);
  const numHeads = Number(metadata[`${prefix}attention.head_count`]) || inferNumHeads(hiddenSize);
  const numKvHeads = Number(metadata[`${prefix}attention.head_count_kv`]) || inferKvHeads(tensors, hiddenSize, numHeads);
  const ffnSize = Number(metadata[`${prefix}feed_forward_length`]) || inferFfnSize(tensors);
  const vocabSize = inferVocabSize(tensors);
  const maxSeqLen = Number(metadata[`${prefix}context_length`]) || 0;
  const totalParams = tensors.reduce((sum, t) => sum + t.shape.reduce((a, b) => a * b, 1), 0);

  return {
    name: archKey || 'unknown',
    hidden_size: hiddenSize,
    num_layers: numLayers,
    num_heads: numHeads,
    num_kv_heads: numKvHeads,
    intermediate_size: ffnSize,
    vocab_size: vocabSize,
    max_sequence_length: maxSeqLen,
    total_params: totalParams,
    estimated_size_mb: (totalParams * 2) / (1024 * 1024),
  };
}

function inferHiddenSize(tensors) {
  for (const t of tensors) {
    if ((t.name.includes('embed') || t.name.includes('token_embd')) && t.shape.length === 2) {
      return t.shape[1];
    }
  }
  return 0;
}

function inferNumLayers(tensors) {
  let max = -1;
  for (const t of tensors) {
    const m = t.name.match(/(?:blk|layers|h)\.\s*(\d+)\./);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return max >= 0 ? max + 1 : 0;
}

function inferNumHeads(hiddenSize) {
  if (!hiddenSize) return 0;
  for (const hd of [128, 64, 96, 256]) {
    if (hiddenSize % hd === 0) return hiddenSize / hd;
  }
  return 0;
}

function inferKvHeads(tensors, hiddenSize, numHeads) {
  if (!hiddenSize || !numHeads) return numHeads;
  const headDim = hiddenSize / numHeads;
  for (const t of tensors) {
    if ((t.name.includes('attn_k') || t.name.includes('k_proj')) && t.shape.length === 2) {
      if (headDim > 0 && t.shape[0] % headDim === 0) return t.shape[0] / headDim;
    }
  }
  return numHeads;
}

function inferFfnSize(tensors) {
  for (const t of tensors) {
    if ((t.name.includes('ffn_up') || t.name.includes('up_proj') ||
         t.name.includes('ffn_gate') || t.name.includes('gate_proj')) && t.shape.length === 2) {
      return t.shape[0];
    }
  }
  return 0;
}

function inferVocabSize(tensors) {
  for (const t of tensors) {
    if ((t.name.includes('embed') || t.name.includes('token_embd')) && t.shape.length === 2) {
      return t.shape[0];
    }
  }
  return 0;
}

function detectQuantization(tensors, arch) {
  const counts = {};
  for (const t of tensors) {
    if (t.name.includes('norm') || t.name.includes('embed') || t.name.includes('embd')) continue;
    counts[t.quantName] = (counts[t.quantName] || 0) + 1;
  }
  let method = 'Unknown';
  let maxCount = 0;
  for (const [name, count] of Object.entries(counts)) {
    if (count > maxCount) { method = name; maxCount = count; }
  }
  const bpw = (QUANT_TYPES[Object.keys(QUANT_TYPES).find(k => QUANT_TYPES[k].name === method)] || {}).bpw || 0;
  const totalBits = tensors.reduce((s, t) => s + t.shape.reduce((a, b) => a * b, 1) * t.bpw, 0);
  const quantizedMb = totalBits / 8 / (1024 * 1024);

  return {
    method,
    bits_per_weight: bpw,
    original_size_mb: arch.estimated_size_mb,
    quantized_size_mb: quantizedMb,
    compression_ratio: quantizedMb > 0 ? arch.estimated_size_mb / quantizedMb : 1,
  };
}

function extractLayers(tensors, arch) {
  const layers = [];
  // Just collect unique layer indices
  const seen = new Set();
  for (const t of tensors) {
    const m = t.name.match(/(?:blk|layers|h)\.\s*(\d+)\./);
    if (m && !seen.has(m[1])) {
      seen.add(m[1]);
      const idx = parseInt(m[1], 10);
      const blockTensors = tensors.filter(tt => {
        const mm = tt.name.match(/(?:blk|layers|h)\.\s*(\d+)\./);
        return mm && parseInt(mm[1], 10) === idx;
      });
      layers.push({
        index: idx,
        tensor_count: blockTensors.length,
        param_count: blockTensors.reduce((s, tt) => s + tt.shape.reduce((a, b) => a * b, 1), 0),
        quantization: blockTensors[0]?.quantName,
      });
    }
  }
  return layers;
}

function extractTokenizer(metadata) {
  const tokens = metadata['tokenizer.ggml.tokens'];
  if (!Array.isArray(tokens)) return null;
  const special = [];
  for (const key of ['tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id',
                      'tokenizer.ggml.padding_token_id', 'tokenizer.ggml.unknown_token_id']) {
    if (metadata[key] != null) {
      special.push({ name: key.replace('tokenizer.ggml.', ''), id: metadata[key] });
    }
  }
  return {
    vocab_size: tokens.length,
    special_tokens: special,
    sample_tokens: tokens.slice(0, 20).map((t, i) => ({ id: i, text: String(t) })),
  };
}

function flattenMetadata(metadata) {
  const flat = {};
  for (const [k, v] of Object.entries(metadata)) {
    if (Array.isArray(v)) flat[k] = `[${v.length} elements]`;
    else if (typeof v === 'object' && v !== null) flat[k] = JSON.stringify(v);
    else flat[k] = String(v);
  }
  return flat;
}

// ── Pretty printer ───────────────────────────────────────────────────────

function printModelResult(result) {
  const _chalk = require('chalk');
  const chalk = _chalk.default || _chalk;
  const a = result.architecture;

  console.log(chalk.bold.cyan('\n  LLM Model Decompilation'));
  console.log(chalk.white(`  Format:       ${result.format}`));
  console.log(chalk.white(`  Architecture: ${a.name}`));
  console.log(chalk.white(`  Parameters:   ${formatNumber(a.total_params)} (${formatSize(a.total_params)})`));
  console.log('');
  console.log(chalk.white(`  Hidden size:     ${a.hidden_size}`));
  console.log(chalk.white(`  Layers:          ${a.num_layers}`));
  console.log(chalk.white(`  Attention heads: ${a.num_heads}`));
  if (a.num_kv_heads !== a.num_heads) {
    const ratio = a.num_heads / a.num_kv_heads;
    console.log(chalk.white(`  KV heads:        ${a.num_kv_heads} (GQA ${ratio}:1)`));
  }
  console.log(chalk.white(`  FFN size:        ${a.intermediate_size}`));
  console.log(chalk.white(`  Vocab size:      ${a.vocab_size}`));
  if (a.max_sequence_length > 0) {
    console.log(chalk.white(`  Max seq length:  ${a.max_sequence_length}`));
  }

  if (result.quantization) {
    const q = result.quantization;
    console.log('');
    console.log(chalk.white(`  Quantization:  ${q.method}`));
    console.log(chalk.white(`  Original size: ${q.original_size_mb.toFixed(0)} MB (FP16)`));
    console.log(chalk.white(`  Quantized:     ${q.quantized_size_mb.toFixed(0)} MB`));
    console.log(chalk.white(`  Compression:   ${q.compression_ratio.toFixed(1)}x`));
  }

  if (result.tokenizer) {
    console.log('');
    console.log(chalk.white(`  Tokenizer:`));
    console.log(chalk.white(`    Vocab: ${formatNumber(result.tokenizer.vocab_size)} tokens`));
    if (result.tokenizer.special_tokens.length > 0) {
      const specials = result.tokenizer.special_tokens.map(s => `${s.name}(${s.id})`).join(', ');
      console.log(chalk.white(`    Special: ${specials}`));
    }
  }

  console.log('');
  console.log(chalk.dim(`  Witness: ${result.witness.source_hash.slice(0, 16)}...`));
  console.log('');
}

function formatNumber(n) {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function formatSize(params) {
  const mb = (params * 2) / (1024 * 1024);
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB FP16`;
  return `${mb.toFixed(0)} MB FP16`;
}

module.exports = { decompileModelFile, decompileGguf, decompileSafetensors, printModelResult };
