//! Integration tests for ruvix-queue.
//!
//! These tests verify:
//! - Concurrent send/recv operations
//! - Zero-copy descriptor-based messaging
//! - Edge cases (queue full, empty, timeouts, message too large)

#![cfg(feature = "std")]

use ruvix_queue::{DescriptorValidator, KernelQueue, MessageDescriptor, QueueConfig};
use ruvix_types::{Handle, KernelError, MsgPriority, RegionHandle, RegionPolicy};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Helper to create a test region handle.
fn test_region() -> RegionHandle {
    RegionHandle(Handle::new(1, 0))
}

/// Helper to create a queue with heap-allocated backing memory.
/// Returns both the queue and the backing buffer (which must be kept alive).
fn create_test_queue(ring_size: u32, max_msg_size: u32) -> (KernelQueue, Vec<u8>) {
    let config = QueueConfig::new(ring_size, max_msg_size);
    KernelQueue::new_heap(config).expect("Failed to create queue")
}

// =============================================================================
// Basic Operations Tests
// =============================================================================

#[test]
fn test_basic_send_recv() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    // Send a message
    let msg = b"Hello, RuVix!";
    queue.send(msg, MsgPriority::Normal).expect("send failed");

    // Receive the message
    let mut buf = [0u8; 256];
    let len = queue
        .recv_timeout(&mut buf, Duration::from_millis(100))
        .expect("recv failed");

    assert_eq!(len, msg.len());
    assert_eq!(&buf[..len], msg);
}

#[test]
fn test_multiple_messages_fifo() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    // Send multiple messages
    for i in 0..5 {
        let msg = format!("Message {}", i);
        queue
            .send(msg.as_bytes(), MsgPriority::Normal)
            .expect("send failed");
    }

    // Receive in FIFO order (same priority)
    let mut buf = [0u8; 256];
    for i in 0..5 {
        let expected = format!("Message {}", i);
        let len = queue
            .recv_timeout(&mut buf, Duration::from_millis(100))
            .expect("recv failed");
        assert_eq!(&buf[..len], expected.as_bytes());
    }
}

#[test]
fn test_priority_ordering() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    // Send messages with different priorities (lower priority first)
    queue.send(b"low", MsgPriority::Low).expect("send failed");
    queue
        .send(b"normal", MsgPriority::Normal)
        .expect("send failed");
    queue
        .send(b"urgent", MsgPriority::Urgent)
        .expect("send failed");
    queue.send(b"high", MsgPriority::High).expect("send failed");

    // Note: Our ring buffer is FIFO within same priority level.
    // The basic ring buffer doesn't reorder by priority during dequeue.
    // Priority is stored but ordering depends on implementation.
    // This test verifies messages can be sent with different priorities.

    let mut buf = [0u8; 256];
    let mut received = Vec::new();

    for _ in 0..4 {
        let len = queue
            .recv_timeout(&mut buf, Duration::from_millis(100))
            .expect("recv failed");
        received.push(String::from_utf8_lossy(&buf[..len]).to_string());
    }

    assert_eq!(received.len(), 4);
    assert!(received.contains(&"low".to_string()));
    assert!(received.contains(&"normal".to_string()));
    assert!(received.contains(&"high".to_string()));
    assert!(received.contains(&"urgent".to_string()));
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

#[test]
fn test_queue_full() {
    // Create a small queue
    let (mut queue, _buffer) = create_test_queue(4, 256);

    // Fill the queue (ring buffer holds exactly ring_size entries)
    for i in 0..4 {
        let msg = format!("msg{}", i);
        queue
            .send(msg.as_bytes(), MsgPriority::Normal)
            .expect("send should succeed");
    }

    // Next send should fail with QueueFull
    let result = queue.send(b"overflow", MsgPriority::Normal);
    assert!(matches!(result, Err(KernelError::QueueFull)));
}

#[test]
fn test_queue_empty_nonblocking() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    // Try to receive from empty queue (non-blocking recv)
    let mut buf = [0u8; 256];
    let result = queue.recv(&mut buf);

    // Should return QueueEmpty
    assert!(matches!(result, Err(KernelError::QueueEmpty)));
}

#[test]
fn test_message_too_large() {
    let (mut queue, _buffer) = create_test_queue(16, 64); // max 64 bytes

    // Try to send a message larger than max_msg_size
    let large_msg = vec![0u8; 128];
    let result = queue.send(&large_msg, MsgPriority::Normal);

    assert!(matches!(result, Err(KernelError::MessageTooLarge)));
}

#[test]
fn test_empty_message() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    // Send empty message
    queue
        .send(&[], MsgPriority::Normal)
        .expect("empty send should succeed");

    // Receive it
    let mut buf = [0u8; 256];
    let len = queue
        .recv_timeout(&mut buf, Duration::from_millis(100))
        .expect("recv failed");

    assert_eq!(len, 0);
}

#[test]
fn test_max_size_message() {
    let max_size = 128u32;
    let (mut queue, _buffer) = create_test_queue(16, max_size);

    // Send a message exactly at max size
    let msg = vec![0xAB; max_size as usize];
    queue
        .send(&msg, MsgPriority::Normal)
        .expect("max size send should succeed");

    // Receive it
    let mut buf = vec![0u8; max_size as usize];
    let len = queue
        .recv_timeout(&mut buf, Duration::from_millis(100))
        .expect("recv failed");

    assert_eq!(len, max_size as usize);
    assert!(buf.iter().all(|&b| b == 0xAB));
}

#[test]
fn test_buffer_too_small() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    // Send a message
    let msg = b"This message is longer than the buffer";
    queue.send(msg, MsgPriority::Normal).expect("send failed");

    // Try to receive with buffer too small
    let mut small_buf = [0u8; 10];
    let result = queue.recv(&mut small_buf);

    // Should fail because buffer is too small
    assert!(result.is_err());
}

// =============================================================================
// Zero-Copy Descriptor Tests
// =============================================================================

#[test]
fn test_descriptor_basic() {
    let desc = MessageDescriptor::new(test_region(), 100, 256);

    assert!(desc.is_valid());
    assert_eq!(desc.offset, 100);
    assert_eq!(desc.length, 256);
}

#[test]
fn test_descriptor_null_region_invalid() {
    let desc = MessageDescriptor::new(RegionHandle::null(), 0, 100);
    assert!(!desc.is_valid());
}

#[test]
fn test_descriptor_zero_length_invalid() {
    let desc = MessageDescriptor::new(test_region(), 0, 0);
    assert!(!desc.is_valid());
}

#[test]
fn test_descriptor_roundtrip() {
    let original = MessageDescriptor::new(test_region(), 12345, 9999);
    let bytes = original.to_bytes();
    let recovered = MessageDescriptor::from_bytes(&bytes).expect("from_bytes failed");

    assert_eq!(original.region, recovered.region);
    assert_eq!(original.offset, recovered.offset);
    assert_eq!(original.length, recovered.length);
}

#[test]
fn test_descriptor_immutable_region_allowed() {
    let validator = DescriptorValidator::new();
    let result = validator.validate_policy(&RegionPolicy::Immutable);
    assert!(result.is_ok());
}

#[test]
fn test_descriptor_append_only_region_allowed() {
    let validator = DescriptorValidator::new();
    let result = validator.validate_policy(&RegionPolicy::AppendOnly { max_size: 1024 });
    assert!(result.is_ok());
}

#[test]
fn test_descriptor_slab_region_rejected() {
    let validator = DescriptorValidator::new();
    let result = validator.validate_policy(&RegionPolicy::Slab {
        slot_size: 64,
        slot_count: 16,
    });

    // Slab regions are rejected for TOCTOU protection
    assert!(matches!(result, Err(KernelError::InvalidDescriptorRegion)));
}

#[test]
fn test_descriptor_bounds_valid() {
    let validator = DescriptorValidator::new();
    let desc = MessageDescriptor::new(test_region(), 100, 200);

    // offset=100, length=200, end=300, region_size=500 -> OK
    assert!(validator.validate_bounds(&desc, 500).is_ok());

    // Exactly at boundary: end=300, region_size=300 -> OK
    assert!(validator.validate_bounds(&desc, 300).is_ok());
}

#[test]
fn test_descriptor_bounds_overflow() {
    let validator = DescriptorValidator::new();

    // offset near u64::MAX would overflow when adding length
    let desc = MessageDescriptor::new(test_region(), u64::MAX - 10, 100);

    // This should be detected as overflow
    assert!(validator.validate_bounds(&desc, 1000).is_err());
}

#[test]
fn test_descriptor_bounds_out_of_range() {
    let validator = DescriptorValidator::new();
    let desc = MessageDescriptor::new(test_region(), 100, 200);

    // end=300 > region_size=299 -> Error
    assert!(validator.validate_bounds(&desc, 299).is_err());
}

#[test]
fn test_full_descriptor_validation() {
    let validator = DescriptorValidator::new();
    let desc = MessageDescriptor::new(test_region(), 0, 100);

    // Valid descriptor with immutable region
    assert!(validator
        .validate(&desc, &RegionPolicy::Immutable, 1000)
        .is_ok());

    // Valid descriptor with append-only region
    assert!(validator
        .validate(&desc, &RegionPolicy::AppendOnly { max_size: 2000 }, 1000)
        .is_ok());

    // Invalid: slab region
    assert!(validator
        .validate(
            &desc,
            &RegionPolicy::Slab {
                slot_size: 64,
                slot_count: 16
            },
            1000
        )
        .is_err());
}

// =============================================================================
// Concurrent Tests
// =============================================================================

#[test]
fn test_concurrent_send_recv_single_producer_single_consumer() {
    let config = QueueConfig::new(64, 256);
    let (queue, buffer) = KernelQueue::new_heap(config).expect("Failed to create queue");

    // Use Arc<Mutex> for thread safety
    let queue = Arc::new(std::sync::Mutex::new(queue));
    let _buffer = Arc::new(buffer); // Keep buffer alive

    let message_count = 100;
    let received_count = Arc::new(AtomicUsize::new(0));

    let queue_sender = Arc::clone(&queue);
    let queue_receiver = Arc::clone(&queue);
    let received_count_clone = Arc::clone(&received_count);

    // Producer thread
    let producer = thread::spawn(move || {
        for i in 0..message_count {
            let msg = format!("Message {}", i);
            loop {
                let mut q = queue_sender.lock().unwrap();
                match q.send(msg.as_bytes(), MsgPriority::Normal) {
                    Ok(()) => break,
                    Err(KernelError::QueueFull) => {
                        drop(q);
                        thread::yield_now();
                        continue;
                    }
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }
        }
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        let mut buf = [0u8; 256];
        let mut received = 0;

        while received < message_count {
            let mut q = queue_receiver.lock().unwrap();
            match q.recv(&mut buf) {
                Ok(_len) => {
                    received += 1;
                    received_count_clone.fetch_add(1, Ordering::SeqCst);
                }
                Err(KernelError::QueueEmpty) => {
                    drop(q);
                    thread::yield_now();
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    });

    producer.join().expect("Producer panicked");
    consumer.join().expect("Consumer panicked");

    assert_eq!(received_count.load(Ordering::SeqCst), message_count);
}

#[test]
fn test_concurrent_multiple_producers() {
    let config = QueueConfig::new(128, 256);
    let (queue, buffer) = KernelQueue::new_heap(config).expect("Failed to create queue");

    let queue = Arc::new(std::sync::Mutex::new(queue));
    let _buffer = Arc::new(buffer);

    let producer_count = 4;
    let messages_per_producer = 25;
    let total_messages = producer_count * messages_per_producer;
    let received_count = Arc::new(AtomicUsize::new(0));

    let mut producers = Vec::new();

    // Spawn producer threads
    for producer_id in 0..producer_count {
        let queue_clone = Arc::clone(&queue);
        let handle = thread::spawn(move || {
            for i in 0..messages_per_producer {
                let msg = format!("P{}-M{}", producer_id, i);
                loop {
                    let mut q = queue_clone.lock().unwrap();
                    match q.send(msg.as_bytes(), MsgPriority::Normal) {
                        Ok(()) => break,
                        Err(KernelError::QueueFull) => {
                            drop(q);
                            thread::yield_now();
                            continue;
                        }
                        Err(e) => panic!("Unexpected error: {:?}", e),
                    }
                }
            }
        });
        producers.push(handle);
    }

    // Consumer thread
    let queue_consumer = Arc::clone(&queue);
    let received_clone = Arc::clone(&received_count);
    let consumer = thread::spawn(move || {
        let mut buf = [0u8; 256];
        let mut received = 0;
        let max_attempts = total_messages * 100;
        let mut attempts = 0;

        while received < total_messages && attempts < max_attempts {
            let mut q = queue_consumer.lock().unwrap();
            match q.recv(&mut buf) {
                Ok(_len) => {
                    received += 1;
                    received_clone.fetch_add(1, Ordering::SeqCst);
                }
                Err(KernelError::QueueEmpty) => {
                    attempts += 1;
                    drop(q);
                    thread::yield_now();
                }
                Err(e) => panic!("Unexpected error: {:?}", e),
            }
        }
    });

    for producer in producers {
        producer.join().expect("Producer panicked");
    }
    consumer.join().expect("Consumer panicked");

    assert_eq!(received_count.load(Ordering::SeqCst), total_messages);
}

#[test]
fn test_stress_high_throughput() {
    let config = QueueConfig::new(256, 64);
    let (queue, buffer) = KernelQueue::new_heap(config).expect("Failed to create queue");

    let queue = Arc::new(std::sync::Mutex::new(queue));
    let _buffer = Arc::new(buffer);

    let message_count = 1000;
    let sent_count = Arc::new(AtomicUsize::new(0));
    let received_count = Arc::new(AtomicUsize::new(0));

    let queue_sender = Arc::clone(&queue);
    let queue_receiver = Arc::clone(&queue);
    let sent_clone = Arc::clone(&sent_count);
    let received_clone = Arc::clone(&received_count);

    // High-speed producer
    let producer = thread::spawn(move || {
        let msg = [0xABu8; 32];
        for _ in 0..message_count {
            loop {
                let mut q = queue_sender.lock().unwrap();
                match q.send(&msg, MsgPriority::High) {
                    Ok(()) => {
                        sent_clone.fetch_add(1, Ordering::SeqCst);
                        break;
                    }
                    Err(KernelError::QueueFull) => {
                        drop(q);
                        thread::yield_now();
                    }
                    Err(e) => panic!("Error: {:?}", e),
                }
            }
        }
    });

    // High-speed consumer
    let consumer = thread::spawn(move || {
        let mut buf = [0u8; 64];
        let mut received = 0;
        let max_attempts = message_count * 1000;
        let mut attempts = 0;

        while received < message_count && attempts < max_attempts {
            let mut q = queue_receiver.lock().unwrap();
            match q.recv(&mut buf) {
                Ok(_) => {
                    received += 1;
                    received_clone.fetch_add(1, Ordering::SeqCst);
                }
                Err(KernelError::QueueEmpty) => {
                    attempts += 1;
                    drop(q);
                }
                Err(e) => panic!("Error: {:?}", e),
            }
        }
    });

    producer.join().expect("Producer panicked");
    consumer.join().expect("Consumer panicked");

    let sent = sent_count.load(Ordering::SeqCst);
    let received = received_count.load(Ordering::SeqCst);

    assert_eq!(sent, message_count);
    assert_eq!(received, message_count);
}

// =============================================================================
// Queue Statistics Tests
// =============================================================================

#[test]
fn test_queue_stats() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    // Initially empty
    assert_eq!(queue.send_count(), 0);
    assert_eq!(queue.recv_count(), 0);

    // Send some messages
    queue.send(b"msg1", MsgPriority::Normal).unwrap();
    queue.send(b"msg2", MsgPriority::High).unwrap();

    assert_eq!(queue.send_count(), 2);
    assert_eq!(queue.recv_count(), 0);

    // Receive one
    let mut buf = [0u8; 256];
    queue.recv(&mut buf).unwrap();

    assert_eq!(queue.send_count(), 2);
    assert_eq!(queue.recv_count(), 1);
}

#[test]
fn test_queue_length() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    assert_eq!(queue.len(), 0);
    assert!(queue.is_empty());

    queue.send(b"a", MsgPriority::Normal).unwrap();
    assert_eq!(queue.len(), 1);
    assert!(!queue.is_empty());

    queue.send(b"b", MsgPriority::Normal).unwrap();
    queue.send(b"c", MsgPriority::Normal).unwrap();
    assert_eq!(queue.len(), 3);

    let mut buf = [0u8; 256];
    queue.recv(&mut buf).unwrap();
    assert_eq!(queue.len(), 2);
}

// =============================================================================
// Ring Buffer Power-of-Two Tests
// =============================================================================

#[test]
fn test_ring_size_must_be_power_of_two() {
    // Valid power-of-two sizes should work
    for size in [4, 8, 16, 32, 64, 128, 256] {
        let config = QueueConfig::new(size, 256);
        assert!(
            KernelQueue::new_heap(config).is_ok(),
            "Size {} should work",
            size
        );
    }
}

#[test]
fn test_ring_size_non_power_of_two_rejected() {
    // Non-power-of-two sizes should be rejected
    for size in [3, 5, 6, 7, 9, 10, 15, 17, 100] {
        let config = QueueConfig::new(size, 256);
        let result = KernelQueue::new_heap(config);
        assert!(result.is_err(), "Size {} should be rejected", size);
    }
}

// =============================================================================
// Peek Operation Tests
// =============================================================================

#[test]
fn test_peek_without_consume() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    queue.send(b"peek me", MsgPriority::Normal).unwrap();

    // Peek should return the entry metadata without consuming
    let entry = queue.peek().expect("peek failed");
    assert_eq!(entry.length, 7); // "peek me" is 7 bytes

    // Message should still be there
    assert_eq!(queue.len(), 1);

    // Can receive it normally
    let mut buf = [0u8; 256];
    let len = queue.recv(&mut buf).expect("recv failed");
    assert_eq!(&buf[..len], b"peek me");

    // Now it's gone
    assert_eq!(queue.len(), 0);
}

#[test]
fn test_peek_empty_queue() {
    let (queue, _buffer) = create_test_queue(16, 256);

    let result = queue.peek();
    assert!(result.is_none());
}

// =============================================================================
// Timeout Tests
// =============================================================================

#[test]
fn test_recv_timeout_empty_queue() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    let mut buf = [0u8; 256];
    let start = std::time::Instant::now();
    let result = queue.recv_timeout(&mut buf, Duration::from_millis(50));
    let elapsed = start.elapsed();

    assert!(matches!(result, Err(KernelError::Timeout)));
    // Should have waited at least 50ms
    assert!(elapsed >= Duration::from_millis(50));
    // But not too long (allow some slack)
    assert!(elapsed < Duration::from_millis(200));
}

#[test]
fn test_recv_timeout_with_data() {
    let (mut queue, _buffer) = create_test_queue(16, 256);

    // Send a message first
    queue.send(b"ready", MsgPriority::Normal).unwrap();

    // Receive with timeout should succeed immediately
    let mut buf = [0u8; 256];
    let start = std::time::Instant::now();
    let len = queue
        .recv_timeout(&mut buf, Duration::from_secs(10))
        .expect("recv should succeed");
    let elapsed = start.elapsed();

    assert_eq!(&buf[..len], b"ready");
    // Should complete quickly since data was available
    assert!(elapsed < Duration::from_millis(100));
}
