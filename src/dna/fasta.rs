//! FASTA/FASTQ format parser for genomic data.

use super::sequence::{PackedSequence, QualityScores, SequenceError, SequenceRead};

/// A FASTA record (header + sequence, no quality)
#[derive(Clone, Debug)]
pub struct FastaRecord {
    /// Sequence identifier
    pub id: String,
    /// Description (rest of header line)
    pub description: String,
    /// Packed DNA sequence
    pub sequence: PackedSequence,
}

/// Parse FASTA format content
pub fn parse_fasta(content: &str) -> Result<Vec<FastaRecord>, SequenceError> {
    let mut records = Vec::new();
    let mut current_id = String::new();
    let mut current_desc = String::new();
    let mut current_seq = Vec::new();

    for line in content.lines() {
        if line.starts_with('>') {
            if !current_id.is_empty() {
                let packed = PackedSequence::from_ascii(&current_seq)?;
                records.push(FastaRecord {
                    id: current_id.clone(),
                    description: current_desc.clone(),
                    sequence: packed,
                });
                current_seq.clear();
            }
            let header = &line[1..];
            let parts: Vec<&str> = header.splitn(2, char::is_whitespace).collect();
            current_id = parts[0].to_string();
            current_desc = parts.get(1).unwrap_or(&"").to_string();
        } else if !line.is_empty() {
            for &c in line.as_bytes() {
                if matches!(c, b'A' | b'C' | b'G' | b'T' | b'a' | b'c' | b'g' | b't') {
                    current_seq.push(c.to_ascii_uppercase());
                }
            }
        }
    }

    if !current_id.is_empty() {
        let packed = PackedSequence::from_ascii(&current_seq)?;
        records.push(FastaRecord {
            id: current_id,
            description: current_desc,
            sequence: packed,
        });
    }

    Ok(records)
}

/// Parse FASTQ format content (4 lines per record: @id, sequence, +, quality)
pub fn parse_fastq(content: &str) -> Result<Vec<SequenceRead>, SequenceError> {
    let mut reads = Vec::new();
    let lines: Vec<&str> = content.lines().collect();

    let mut i = 0;
    while i + 3 <= lines.len() {
        if !lines[i].starts_with('@') {
            i += 1;
            continue;
        }

        let id = lines[i][1..].to_string();
        let seq_line = lines[i + 1].as_bytes();
        let qual_line = lines[i + 3].as_bytes();

        let sequence = PackedSequence::from_ascii(seq_line)?;
        let quality = QualityScores::from_phred33(qual_line);

        reads.push(SequenceRead {
            id,
            sequence,
            quality,
        });
        i += 4;
    }

    Ok(reads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fasta() {
        let content = ">seq1 test sequence\nACGTACGT\nACGTACGT\n>seq2\nGGGGCCCC\n";
        let records = parse_fasta(content).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "seq1");
        assert_eq!(records[0].sequence.len(), 16);
        assert_eq!(records[1].sequence.len(), 8);
    }

    #[test]
    fn test_parse_fastq() {
        let content = "@read1\nACGTACGT\n+\nIIIIIIII\n@read2\nGGCCTTAA\n+\n!!!!!!!!";
        let reads = parse_fastq(content).unwrap();
        assert_eq!(reads.len(), 2);
        assert_eq!(reads[0].id, "read1");
        assert_eq!(reads[0].sequence.len(), 8);
        assert_eq!(reads[0].quality.len(), 8);
    }

    #[test]
    fn test_parse_fasta_filters_non_acgt() {
        let content = ">seq1\nACGTNNACGT\n";
        let records = parse_fasta(content).unwrap();
        assert_eq!(records[0].sequence.len(), 8);
    }

    #[test]
    fn test_parse_fasta_multiline() {
        let content = ">gene1\nACGT\nACGT\nACGT\n";
        let records = parse_fasta(content).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].sequence.len(), 12);
    }

    #[test]
    fn test_parse_fasta_description() {
        let content = ">seq1 a description here\nACGT\n";
        let records = parse_fasta(content).unwrap();
        assert_eq!(records[0].id, "seq1");
        assert_eq!(records[0].description, "a description here");
    }

    #[test]
    fn test_parse_fasta_no_description() {
        let content = ">seq1\nACGT\n";
        let records = parse_fasta(content).unwrap();
        assert_eq!(records[0].id, "seq1");
        assert_eq!(records[0].description, "");
    }

    #[test]
    fn test_parse_fasta_empty() {
        let content = "";
        let records = parse_fasta(content).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn test_parse_fastq_quality() {
        let content = "@read1\nACGTACGT\n+\nIIIIIIII\n";
        let reads = parse_fastq(content).unwrap();
        assert_eq!(reads.len(), 1);
        // 'I' = ASCII 73, PHRED = 73 - 33 = 40
        assert_eq!(reads[0].quality.phred_at(0), 40);
    }

    #[test]
    fn test_parse_fasta_case_insensitive() {
        let content = ">seq1\nacgtACGT\n";
        let records = parse_fasta(content).unwrap();
        assert_eq!(records[0].sequence.len(), 8);
    }

    #[test]
    fn test_parse_fasta_multiple_records() {
        let content = ">s1\nACGT\n>s2\nGCAT\n>s3\nTTTT\n";
        let records = parse_fasta(content).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].id, "s1");
        assert_eq!(records[1].id, "s2");
        assert_eq!(records[2].id, "s3");
    }
}
