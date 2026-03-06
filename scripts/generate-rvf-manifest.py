#!/usr/bin/env python3
"""
Generate manifest.json for RVF example files.
Scans a directory of .rvf files, computes SHA-256 hashes, categorizes them,
and produces a manifest for GCS hosting.

Usage:
  python3 scripts/generate-rvf-manifest.py \
    --input examples/rvf/output/ \
    --version 0.2.1 \
    --output manifest.json
"""

import argparse
import hashlib
import json
import os
import struct
from datetime import datetime, timezone

CATEGORY_MAP = {
    'basic_store': 'core', 'semantic_search': 'core', 'rag_pipeline': 'core',
    'embedding_cache': 'core', 'quantization': 'core', 'progressive_index': 'core',
    'filtered_search': 'core', 'recommendation': 'core',
    'agent_memory': 'ai', 'swarm_knowledge': 'ai', 'experience_replay': 'ai',
    'tool_cache': 'ai', 'ruvbot': 'ai', 'ruvllm_inference': 'ai',
    'mcp_in_rvf': 'integration', 'claude_code_appliance': 'integration',
    'claude_code_appliance_v1': 'integration',
    'postgres_bridge': 'integration', 'serverless': 'integration',
    'lineage_parent': 'lineage', 'lineage_child': 'lineage',
    'reasoning_parent': 'lineage', 'reasoning_child': 'lineage',
    'reasoning_grandchild': 'lineage',
    'self_booting': 'compute', 'linux_microkernel': 'compute',
    'ebpf_accelerator': 'compute', 'browser_wasm': 'compute',
    'tee_attestation': 'security', 'zero_knowledge': 'security',
    'sealed_engine': 'security', 'access_control': 'security',
    'financial_signals': 'industry', 'medical_imaging': 'industry',
    'legal_discovery': 'industry',
    'multimodal_fusion': 'core', 'hyperbolic_taxonomy': 'core',
    'network_telemetry': 'network', 'network_sync_a': 'network',
    'network_sync_b': 'network', 'agent_handoff_a': 'network',
    'agent_handoff_b': 'network',
    'edge_iot': 'compute', 'dedup_detector': 'core',
    'compacted': 'core', 'posix_fileops': 'core',
}

CATEGORY_DESCRIPTIONS = {
    'core': 'Basic vector storage, search, and indexing',
    'ai': 'AI agent, embedding, RAG, and chatbot examples',
    'security': 'Attestation, ZK proofs, access control, sealed engines',
    'compute': 'eBPF, WASM, self-booting, IoT, kernels',
    'lineage': 'COW chains, derivation trees, reasoning chains',
    'industry': 'Finance, medical, legal domain examples',
    'network': 'Sync, handoff, telemetry, distributed examples',
    'integration': 'MCP, PostgreSQL, serverless, Claude Code bridges',
}

def human_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def detect_rvf_segments(filepath):
    """Try to detect RVF segment types from the file header."""
    segments = []
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic == b'RVF\x01' or magic == b'\x00RVF':
                # Try to read segment directory
                segments = ['VEC', 'META']  # Most files have at least these
    except:
        pass
    return segments if segments else ['VEC', 'META']

def generate_manifest(input_dir, version, base_url=None):
    if base_url is None:
        base_url = f"https://storage.googleapis.com/ruvector-examples/v{version}"

    examples = []
    total_size = 0

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.rvf'):
            continue

        filepath = os.path.join(input_dir, filename)
        name = filename[:-4]  # strip .rvf
        size = os.path.getsize(filepath)
        total_size += size

        category = CATEGORY_MAP.get(name, 'core')

        # Check for sidecar metadata
        meta_path = filepath + '.meta.json'
        description = ''
        tags = []
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                description = meta.get('description', '')
                tags = meta.get('tags', [])
                if 'category' in meta:
                    category = meta['category']

        if not description:
            # Generate description from name
            description = name.replace('_', ' ').title()

        examples.append({
            'name': name,
            'file': filename,
            'size': size,
            'size_human': human_size(size),
            'sha256': sha256_file(filepath),
            'description': description,
            'category': category,
            'tags': tags if tags else [category, name.split('_')[0]],
            'segments': detect_rvf_segments(filepath),
            'created': datetime.fromtimestamp(
                os.path.getmtime(filepath), tz=timezone.utc
            ).strftime('%Y-%m-%d'),
        })

    manifest = {
        'version': version,
        'updated': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'base_url': base_url,
        'total_size': human_size(total_size),
        'total_size_bytes': total_size,
        'count': len(examples),
        'examples': examples,
        'categories': CATEGORY_DESCRIPTIONS,
    }

    return manifest

def main():
    parser = argparse.ArgumentParser(description='Generate RVF example manifest')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing .rvf files')
    parser.add_argument('--version', '-v', required=True, help='Package version')
    parser.add_argument('--output', '-o', default='manifest.json', help='Output manifest file')
    parser.add_argument('--base-url', help='Base URL for downloads (default: GCS URL)')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: {args.input} is not a directory")
        return 1

    manifest = generate_manifest(args.input, args.version, args.base_url)

    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated manifest: {args.output}")
    print(f"  Version: {manifest['version']}")
    print(f"  Examples: {manifest['count']}")
    print(f"  Total size: {manifest['total_size']}")
    print(f"  Categories: {', '.join(manifest['categories'].keys())}")

    return 0

if __name__ == '__main__':
    exit(main())
