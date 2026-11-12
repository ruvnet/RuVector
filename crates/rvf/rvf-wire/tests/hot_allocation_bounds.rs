//! Observe allocations with a bounded count, never an attacker-sized allocation.
use rvf_wire::hot_seg_codec::{
    read_hot_entries, read_hot_header, write_hot_seg, HotEntry, HotHeader,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static OBSERVING: Cell<bool> = const { Cell::new(false) };
    static MAX_REQUEST: Cell<usize> = const { Cell::new(0) };
}

struct ObservedAllocator;

fn record(size: usize) {
    if OBSERVING.try_with(Cell::get).unwrap_or(false) {
        let _ = MAX_REQUEST.try_with(|max| max.set(max.get().max(size)));
    }
}

// Delegate all memory management to System; observation is thread-local so the
// test runner and concurrent tests do not contaminate the measured call.
unsafe impl GlobalAlloc for ObservedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: ObservedAllocator = ObservedAllocator;

fn header(count: u32) -> HotHeader {
    HotHeader {
        vector_count: count,
        dim: 2,
        dtype: 0x04,
        neighbor_m: 1,
    }
}

fn observe(data: &[u8], header: &HotHeader) -> (Result<Vec<HotEntry>, &'static str>, usize) {
    MAX_REQUEST.with(|max| max.set(0));
    OBSERVING.with(|active| active.set(true));
    let result = read_hot_entries(data, header);
    OBSERVING.with(|active| active.set(false));
    (result, MAX_REQUEST.with(Cell::get))
}

#[test]
fn empty_payload_does_not_reserve_from_claimed_vector_count() {
    // 4096 entries is deliberately bounded (<256 KiB even on the old decoder).
    let (result, maximum) = observe(&[], &header(4096));
    assert_eq!(result, Err("hot entry truncated"));
    assert_eq!(
        maximum, 0,
        "empty payload requested {maximum} allocation bytes"
    );
}

#[test]
fn short_payload_reservation_is_bounded_by_possible_entries() {
    // Exactly one unpadded entry; the second declared entry is truncated.
    let data = [0u8; 12];
    let (result, maximum) = observe(&data, &header(4096));
    assert_eq!(result, Err("hot entry truncated"));
    assert!(
        maximum <= std::mem::size_of::<HotEntry>(),
        "one possible entry requested {maximum} allocation bytes"
    );
}

#[test]
fn round_trip_preserves_vectors_neighbors_and_alignment() {
    for (dtype, width) in [(0x00, 4), (0x01, 2), (0x04, 1)] {
        let mut header = header(2);
        header.dtype = dtype;
        let entries: Vec<_> = (0..2)
            .map(|i| HotEntry {
                vector_id: i + 10,
                vector_data: vec![i as u8 + 1; 2 * width],
                neighbor_ids: vec![i + 100],
            })
            .collect();
        let payload = write_hot_seg(&header, &entries);
        let (decoded_header, offset) = read_hot_header(&payload).unwrap();
        assert_eq!(payload.len() % 64, 0);
        assert_eq!(
            read_hot_entries(&payload[offset..], &decoded_header).unwrap(),
            entries
        );
    }
}

#[test]
fn truncated_vector_and_neighbor_payloads_return_errors() {
    assert_eq!(
        read_hot_entries(&[0; 11], &header(1)),
        Err("hot entry truncated")
    );
    let mut data = [0u8; 19];
    data[10] = 1; // neighbor count, but only seven of eight ID bytes follow.
    assert_eq!(
        read_hot_entries(&data, &header(1)),
        Err("neighbor IDs truncated")
    );
}

#[test]
fn zero_count_empty_payload_still_succeeds_without_allocating() {
    let (result, maximum) = observe(&[], &header(0));
    assert_eq!(result.unwrap(), vec![]);
    assert_eq!(maximum, 0);
}
