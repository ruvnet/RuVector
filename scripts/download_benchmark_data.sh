#!/bin/bash
# Download real public genomic data for benchmarking
#
# Sources:
#   - NCBI GenBank / RefSeq (public domain sequences)
#   - ClinVar (public variant annotations)
#
# All data is freely available and redistributable.
#
# Usage:
#   chmod +x scripts/download_benchmark_data.sh
#   ./scripts/download_benchmark_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/benchmark"

mkdir -p "$DATA_DIR"

echo "=== Downloading real genomic benchmark data ==="
echo "Target directory: $DATA_DIR"
echo ""

# PhiX174 complete genome (5,386 bp) - standard Illumina sequencing control
# GenBank: NC_001422.1
echo "[1/5] PhiX174 bacteriophage genome (5,386 bp)..."
if [ ! -f "$DATA_DIR/phix174.fasta" ]; then
    curl -sL --retry 3 --max-time 30 \
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001422.1&rettype=fasta" \
        > "$DATA_DIR/phix174.fasta"
    echo "  Downloaded: $(wc -c < "$DATA_DIR/phix174.fasta") bytes"
else
    echo "  Already exists, skipping."
fi

# SARS-CoV-2 reference genome (29,903 bp) - Wuhan-Hu-1 isolate
# GenBank: NC_045512.2
echo "[2/5] SARS-CoV-2 reference genome (29,903 bp)..."
if [ ! -f "$DATA_DIR/sars_cov2.fasta" ]; then
    curl -sL --retry 3 --max-time 30 \
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_045512.2&rettype=fasta" \
        > "$DATA_DIR/sars_cov2.fasta"
    echo "  Downloaded: $(wc -c < "$DATA_DIR/sars_cov2.fasta") bytes"
else
    echo "  Already exists, skipping."
fi

# E. coli K-12 MG1655 complete genome (4,641,652 bp) - model organism
# GenBank: U00096.3
echo "[3/5] E. coli K-12 MG1655 genome (4,641,652 bp)..."
if [ ! -f "$DATA_DIR/ecoli_k12.fasta" ]; then
    curl -sL --retry 3 --max-time 120 \
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=U00096.3&rettype=fasta" \
        > "$DATA_DIR/ecoli_k12.fasta"
    echo "  Downloaded: $(wc -c < "$DATA_DIR/ecoli_k12.fasta") bytes"
else
    echo "  Already exists, skipping."
fi

# Human mitochondrial genome (16,569 bp) - revised Cambridge Reference Sequence
# GenBank: NC_012920.1
echo "[4/5] Human mitochondrial genome (16,569 bp)..."
if [ ! -f "$DATA_DIR/human_mito.fasta" ]; then
    curl -sL --retry 3 --max-time 30 \
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_012920.1&rettype=fasta" \
        > "$DATA_DIR/human_mito.fasta"
    echo "  Downloaded: $(wc -c < "$DATA_DIR/human_mito.fasta") bytes"
else
    echo "  Already exists, skipping."
fi

# Lambda phage genome (48,502 bp) - classic molecular biology reference
# GenBank: NC_001416.1
echo "[5/5] Lambda phage genome (48,502 bp)..."
if [ ! -f "$DATA_DIR/lambda_phage.fasta" ]; then
    curl -sL --retry 3 --max-time 30 \
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001416.1&rettype=fasta" \
        > "$DATA_DIR/lambda_phage.fasta"
    echo "  Downloaded: $(wc -c < "$DATA_DIR/lambda_phage.fasta") bytes"
else
    echo "  Already exists, skipping."
fi

echo ""
echo "=== Download complete ==="
echo ""
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
echo ""
echo "Sequence lengths (bases, excluding headers):"
for f in "$DATA_DIR"/*.fasta; do
    name=$(basename "$f")
    bases=$(grep -v '^>' "$f" | tr -d '\n' | wc -c)
    printf "  %-25s %s bp\n" "$name" "$bases"
done
