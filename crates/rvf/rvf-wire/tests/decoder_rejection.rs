//! Invalid input must follow the public Result contract, never unwind.
use rvf_wire::index_seg_codec::read_adjacency;
use rvf_wire::vec_seg_codec::{read_vec_block, write_vec_block, BlockDirEntry, VecBlock};

fn entry(count: u32) -> BlockDirEntry {
    BlockDirEntry {
        block_offset: 0,
        vector_count: count,
        dim: 1,
        dtype: 4,
        tier: 0,
    }
}

fn encoded_ids(count: u32, restart: u16, ids: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0; count as usize];
    bytes.push(1);
    bytes.extend_from_slice(&restart.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(ids);
    bytes
}

#[test]
fn vec_zero_restart_is_an_error() {
    assert!(read_vec_block(&encoded_ids(1, 0, &[1]), &entry(1)).is_err());
}

#[test]
fn vec_truncated_delta_is_an_error() {
    assert!(read_vec_block(&encoded_ids(1, 128, &[0x80]), &entry(1)).is_err());
}

#[test]
fn vec_overflowing_delta_is_an_error() {
    let mut ids = vec![0xff; 9];
    ids.extend_from_slice(&[1, 1]); // MAX followed by delta 1.
    assert!(read_vec_block(&encoded_ids(2, 128, &ids), &entry(2)).is_err());
}

#[test]
fn vec_count_must_match_directory() {
    let mut bytes = encoded_ids(1, 128, &[1, 1]);
    bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert!(read_vec_block(&bytes, &entry(1)).is_err());
}

#[test]
fn adjacency_overflow_is_an_error() {
    let mut bytes = vec![1, 2]; // one layer, two neighbors
    bytes.extend_from_slice(&[0xff; 9]);
    bytes.extend_from_slice(&[1, 1]);
    assert!(read_adjacency(&bytes, 1).is_err());
}

#[test]
fn adjacency_truncations_are_errors() {
    for bytes in [&[][..], &[1][..], &[1, 1][..], &[0x80][..]] {
        assert!(read_adjacency(bytes, 1).is_err());
    }
}

#[test]
fn adjacency_impossible_count_is_an_error() {
    // Capacity overflow is rejected before any allocation attempt.
    assert!(read_adjacency(&[], u64::MAX).is_err());
}

#[test]
fn adjacency_impossible_layer_and_neighbor_counts_are_errors() {
    let mut max = vec![0xff; 9];
    max.push(1);
    assert!(read_adjacency(&max, 1).is_err());
    let mut neighbors = vec![1];
    neighbors.extend_from_slice(&max);
    assert!(read_adjacency(&neighbors, 1).is_err());
}

#[test]
fn vec_raw_ids_and_unknown_encoding() {
    let mut bytes = vec![10, 0, 128, 0, 1, 0, 0, 0];
    bytes.extend_from_slice(&42u64.to_le_bytes());
    assert_eq!(read_vec_block(&bytes, &entry(1)).unwrap().ids, vec![42]);
    assert!(read_vec_block(&bytes[..15], &entry(1)).is_err());
    bytes[1] = 2;
    assert!(read_vec_block(&bytes, &entry(1)).is_err());
}

#[test]
fn vec_roundtrip_crosses_delta_restart_boundary() {
    let ids: Vec<_> = (0..260).map(|n| n * 7 + 100).collect();
    let block = VecBlock {
        vector_data: vec![1; 260],
        ids: ids.clone(),
        dim: 1,
        dtype: 4,
        tier: 0,
    };
    let bytes = write_vec_block(&block);
    assert_eq!(read_vec_block(&bytes, &entry(260)).unwrap().ids, ids);
    // Each prefix must return or reject normally, without a panic.
    for len in 0..bytes.len() {
        let _ = read_vec_block(&bytes[..len], &entry(260));
    }
}

#[test]
fn individual_varint_overflow_is_an_error() {
    let mut overflow = vec![0x80; 9];
    overflow.push(2);
    assert!(rvf_wire::varint::decode_varint(&overflow).is_err());
    assert!(read_vec_block(&encoded_ids(1, 128, &overflow), &entry(1)).is_err());
    assert!(read_adjacency(&overflow, 1).is_err());
}

#[test]
fn valid_vec_and_adjacency_controls() {
    let block = VecBlock {
        vector_data: vec![10, 20],
        ids: vec![4, 9],
        dim: 1,
        dtype: 4,
        tier: 0,
    };
    let decoded = read_vec_block(&write_vec_block(&block), &entry(2)).unwrap();
    assert_eq!(decoded.ids, block.ids);
    assert_eq!(decoded.vector_data, block.vector_data);
    let nodes = read_adjacency(&[1, 2, 4, 5], 1).unwrap();
    assert_eq!(nodes[0].layers, vec![vec![4, 9]]);
    assert!(read_adjacency(&[], 0).unwrap().is_empty());
}
