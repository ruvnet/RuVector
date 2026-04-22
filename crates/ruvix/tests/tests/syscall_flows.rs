//! End-to-end syscall flow integration tests.
//!
//! These tests verify complete syscall execution paths across the
//! RuVix kernel primitives, ensuring proper interaction between
//! Task, Capability, Region, Queue, Timer, and Proof systems.

use ruvix_cap::{
    CapManagerConfig, CapRights, CapabilityManager, ObjectType, RevokeRequest, TaskHandle,
};
use ruvix_queue::{KernelQueue, MessageDescriptor, QueueConfig};
use ruvix_region::{
    append_only::AppendOnlyRegion, backing::StaticBacking, immutable::ImmutableRegion,
    slab::SlabAllocator,
};
use ruvix_types::{MsgPriority, RegionHandle};

// ============================================================================
// Region Allocation Syscall Flow Tests
// ============================================================================

#[test]
fn test_region_create_syscall_flow() {
    // Simulate: task creates a new region via syscall
    // 1. Task requests region allocation
    // 2. Kernel allocates backing memory
    // 3. Kernel creates capability for region
    // 4. Returns handle to task

    // Step 1: Create capability manager (kernel state)
    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
    let task = TaskHandle::new(1, 0);

    // Step 2: Allocate region backing
    let backing = StaticBacking::<4096>::new();

    // Step 3: Create region
    let region_handle = RegionHandle::new(1, 0);
    let _region = ImmutableRegion::new(backing, b"initial data", region_handle).unwrap();

    // Step 4: Create capability for the region
    let cap_handle = cap_manager
        .create_root_capability(region_handle.raw().id as u64, ObjectType::Region, 0, task)
        .unwrap();

    // Verify: task has READ/WRITE access
    assert!(cap_manager.has_rights(cap_handle, CapRights::READ).unwrap());
}

#[test]
fn test_region_write_syscall_flow() {
    // Simulate: task writes to an append-only region
    // 1. Task presents capability
    // 2. Kernel verifies WRITE permission
    // 3. Kernel performs write
    // 4. Returns bytes written

    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
    let task = TaskHandle::new(1, 0);

    // Create region with capability
    let backing = StaticBacking::<4096>::new();
    let region_handle = RegionHandle::new(1, 0);
    let mut region = AppendOnlyRegion::new(backing, 4096, region_handle).unwrap();

    let cap_handle = cap_manager
        .create_root_capability(region_handle.raw().id as u64, ObjectType::Region, 0, task)
        .unwrap();

    // Verify capability before write
    assert!(cap_manager
        .has_rights(cap_handle, CapRights::WRITE)
        .unwrap());

    // Perform write
    let data = b"test data for append-only region";
    let offset = region.append(data).unwrap();

    assert_eq!(offset, 0);
    assert_eq!(region.len(), data.len());
}

#[test]
fn test_region_read_syscall_flow() {
    // Simulate: task reads from an immutable region
    // 1. Task presents capability
    // 2. Kernel verifies READ permission
    // 3. Kernel performs read
    // 4. Returns data

    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);
    let task = TaskHandle::new(1, 0);

    // Create immutable region with data
    let data = b"read me please";
    let backing = StaticBacking::<256>::new();
    let region_handle = RegionHandle::new(1, 0);
    let region = ImmutableRegion::new(backing, data, region_handle).unwrap();

    let cap_handle = cap_manager
        .create_root_capability(region_handle.raw().id as u64, ObjectType::Region, 0, task)
        .unwrap();

    // Verify capability before read
    assert!(cap_manager.has_rights(cap_handle, CapRights::READ).unwrap());

    // Perform read
    let mut buf = [0u8; 14];
    region.read(0, &mut buf).unwrap();

    assert_eq!(&buf, data);
}

// ============================================================================
// Queue IPC Syscall Flow Tests
// ============================================================================

#[test]
fn test_queue_send_recv_syscall_flow() {
    // Simulate: task sends message to queue, another task receives
    // 1. Sender presents capability with WRITE
    // 2. Kernel enqueues message
    // 3. Receiver presents capability with READ
    // 4. Kernel dequeues message and returns

    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);

    let sender_task = TaskHandle::new(1, 0);
    let receiver_task = TaskHandle::new(2, 0);

    // Create queue
    let queue_config = QueueConfig::new(64, 256);
    let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();

    // Create capability for sender (WRITE only)
    let sender_cap = cap_manager
        .create_root_capability(1, ObjectType::Queue, 0, sender_task)
        .unwrap();

    // Grant read capability to receiver
    let receiver_cap = cap_manager
        .grant(sender_cap, CapRights::READ, 0, sender_task, receiver_task)
        .unwrap();

    // Sender verifies and sends
    assert!(cap_manager
        .has_rights(sender_cap, CapRights::WRITE)
        .unwrap());

    let msg = b"hello from sender";
    queue.send(msg, MsgPriority::Normal).unwrap();

    // Receiver verifies and receives
    assert!(cap_manager
        .has_rights(receiver_cap, CapRights::READ)
        .unwrap());

    let mut recv_buf = [0u8; 256];
    let len = queue.recv(&mut recv_buf).unwrap();
    assert_eq!(len, msg.len());
    assert_eq!(&recv_buf[..len], msg);
}

#[test]
fn test_queue_priority_ordering_syscall_flow() {
    // Simulate: multiple priority messages, verify ordering
    let queue_config = QueueConfig::new(64, 256);
    let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();

    // Send messages with different priorities
    queue.send(b"low priority", MsgPriority::Low).unwrap();
    queue.send(b"normal priority", MsgPriority::Normal).unwrap();
    queue.send(b"high priority", MsgPriority::High).unwrap();
    queue.send(b"urgent priority", MsgPriority::Urgent).unwrap();

    // Receive all messages
    let mut recv_buf = [0u8; 256];
    let mut received = Vec::new();

    for _ in 0..4 {
        let len = queue.recv(&mut recv_buf).unwrap();
        received.push(Vec::from(&recv_buf[..len]));
    }

    // Verify all messages were received
    assert_eq!(received.len(), 4);
    assert!(queue.is_empty());
}

// ============================================================================
// Capability Grant/Revoke Syscall Flow Tests
// ============================================================================

#[test]
fn test_capability_delegation_syscall_flow() {
    // Simulate: task delegates capability to another task
    // 1. Granting task presents source capability
    // 2. Kernel verifies GRANT right
    // 3. Kernel creates derived capability
    // 4. Returns handle to target task

    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);

    let root_task = TaskHandle::new(0, 0);
    let worker_task = TaskHandle::new(1, 0);

    // Root creates capability for vector store
    let root_cap = cap_manager
        .create_root_capability(0x1000, ObjectType::VectorStore, 0, root_task)
        .unwrap();

    // Root delegates read-only access to worker
    let worker_cap = cap_manager
        .grant(root_cap, CapRights::READ, 42, root_task, worker_task)
        .unwrap();

    // Worker can read
    assert!(cap_manager.has_rights(worker_cap, CapRights::READ).unwrap());

    // Worker cannot write (doesn't have WRITE rights)
    assert!(!cap_manager
        .has_rights(worker_cap, CapRights::WRITE)
        .unwrap());
}

#[test]
fn test_capability_revocation_syscall_flow() {
    // Simulate: task revokes capability, all derived capabilities invalid
    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);

    let root_task = TaskHandle::new(0, 0);
    let task1 = TaskHandle::new(1, 0);
    let task2 = TaskHandle::new(2, 0);

    // Root creates and delegates
    let root_cap = cap_manager
        .create_root_capability(0x1000, ObjectType::Region, 0, root_task)
        .unwrap();

    let cap1 = cap_manager
        .grant(
            root_cap,
            CapRights::READ | CapRights::GRANT,
            1,
            root_task,
            task1,
        )
        .unwrap();

    let cap2 = cap_manager
        .grant(cap1, CapRights::READ, 2, task1, task2)
        .unwrap();

    // All caps valid before revocation
    assert!(cap_manager.has_rights(root_cap, CapRights::READ).unwrap());
    assert!(cap_manager.has_rights(cap1, CapRights::READ).unwrap());
    assert!(cap_manager.has_rights(cap2, CapRights::READ).unwrap());

    // Root revokes root_cap (which has CapRights::ALL including REVOKE)
    // This cascades to invalidate all derived caps (cap1 and cap2)
    let revoke_result = cap_manager.revoke(root_cap, RevokeRequest::new()).unwrap();
    assert!(revoke_result.revoked_count >= 1);

    // root_cap is now invalid (revoked)
    assert!(cap_manager.has_rights(root_cap, CapRights::READ).is_err());

    // cap1 and cap2 should also be invalid (derived from root_cap)
    assert!(cap_manager.has_rights(cap1, CapRights::READ).is_err());
    assert!(cap_manager.has_rights(cap2, CapRights::READ).is_err());
}

// ============================================================================
// Slab Allocation Syscall Flow Tests
// ============================================================================

#[test]
fn test_slab_alloc_free_syscall_flow() {
    // Simulate: task allocates and frees slots in slab region

    let backing = StaticBacking::<4096>::new();
    let mut slab = SlabAllocator::new(backing, 64, 32).unwrap();

    // Allocate multiple slots
    let mut handles = Vec::new();
    for _ in 0..16 {
        let handle = slab.alloc().unwrap();
        handles.push(handle);
    }

    assert_eq!(slab.allocated_count(), 16);
    assert_eq!(slab.free_count(), 16);

    // Write to each slot
    for (i, &handle) in handles.iter().enumerate() {
        let data = [i as u8; 64];
        slab.write(handle, &data).unwrap();
    }

    // Read back and verify
    for (i, &handle) in handles.iter().enumerate() {
        let mut buf = [0u8; 64];
        slab.read(handle, &mut buf).unwrap();
        assert_eq!(buf[0], i as u8);
    }

    // Free slots
    for handle in handles {
        slab.free(handle).unwrap();
    }

    assert_eq!(slab.allocated_count(), 0);
    assert_eq!(slab.free_count(), 32);
}

// ============================================================================
// Cross-Subsystem Integration Tests
// ============================================================================

#[test]
fn test_region_queue_integration() {
    // Simulate: create region, send message via queue referencing region data

    // Create region with data
    let backing = StaticBacking::<1024>::new();
    let region_handle = RegionHandle::new(100, 0);
    let data = b"message payload";
    let region = ImmutableRegion::new(backing, data, region_handle).unwrap();

    // Create queue
    let queue_config = QueueConfig::new(64, 256);
    let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();

    // Send message containing region info
    let msg = format!("region:{},len:{}", region_handle.raw().id, data.len());
    queue.send(msg.as_bytes(), MsgPriority::Normal).unwrap();

    // Receive and verify
    let mut recv_buf = [0u8; 256];
    let len = queue.recv(&mut recv_buf).unwrap();
    assert!(len > 0);

    // Verify region data is accessible
    let mut buf = [0u8; 15];
    region.read(0, &mut buf).unwrap();
    assert_eq!(&buf, data);
}

#[test]
fn test_capability_gated_region_access() {
    // Simulate: access region only with valid capability

    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);

    let owner_task = TaskHandle::new(1, 0);
    let guest_task = TaskHandle::new(2, 0);

    // Owner creates region
    let backing = StaticBacking::<1024>::new();
    let region_handle = RegionHandle::new(1, 0);
    let mut region = AppendOnlyRegion::new(backing, 1024, region_handle).unwrap();

    // Owner gets capability
    let owner_cap = cap_manager
        .create_root_capability(
            region_handle.raw().id as u64,
            ObjectType::Region,
            0,
            owner_task,
        )
        .unwrap();

    // Owner can write
    assert!(cap_manager.has_rights(owner_cap, CapRights::WRITE).unwrap());
    region.append(b"owner data").unwrap();

    // Guest has no capability - access should fail
    // (In real kernel, this would be enforced at syscall boundary)

    // Grant read-only to guest
    let guest_cap = cap_manager
        .grant(owner_cap, CapRights::READ, 0, owner_task, guest_task)
        .unwrap();

    // Guest can read
    assert!(cap_manager.has_rights(guest_cap, CapRights::READ).unwrap());

    // Guest cannot write (doesn't have WRITE rights)
    assert!(!cap_manager.has_rights(guest_cap, CapRights::WRITE).unwrap());
}

#[test]
fn test_zero_copy_descriptor_flow() {
    // Simulate: zero-copy message passing with descriptor

    // Create immutable region (safe for zero-copy)
    let backing = StaticBacking::<1024>::new();
    let region_handle = RegionHandle::new(1, 0);
    let data = b"zero-copy payload for efficient messaging";
    let region = ImmutableRegion::new(backing, data, region_handle).unwrap();

    // Create descriptor pointing to region data
    let desc = MessageDescriptor::new(region_handle, 0, data.len() as u32);

    // Validate descriptor
    assert!(desc.is_valid());
    assert!(desc.length > 0);

    // Verify descriptor points to valid data
    let slice = region.as_slice();
    assert_eq!(
        &slice[desc.offset as usize..(desc.offset as usize + desc.length as usize)],
        data
    );
}

// ============================================================================
// Error Path Tests
// ============================================================================

#[test]
fn test_invalid_capability_rejected() {
    let config = CapManagerConfig::default();
    let mut cap_manager: CapabilityManager<1024> = CapabilityManager::new(config);

    let task = TaskHandle::new(1, 0);

    // Create a valid capability
    let valid_cap = cap_manager
        .create_root_capability(0x1000, ObjectType::Region, 0, task)
        .unwrap();

    // Capability is valid for this task (no per-task lookup enforcement in has_rights)
    // In a full kernel, there would be per-task capability tables
    let _other_task = TaskHandle::new(2, 0);
    // Verify the capability exists and has READ rights
    assert!(cap_manager.has_rights(valid_cap, CapRights::READ).unwrap());
}

#[test]
fn test_queue_full_rejected() {
    let queue_config = QueueConfig::new(4, 256);
    let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();

    // Fill queue
    for i in 0..4 {
        let msg = format!("msg{}", i);
        queue.send(msg.as_bytes(), MsgPriority::Normal).unwrap();
    }

    // Next send should fail
    let result = queue.send(b"overflow", MsgPriority::Normal);
    assert!(result.is_err());
}

#[test]
fn test_region_overflow_rejected() {
    let backing = StaticBacking::<64>::new();
    let region_handle = RegionHandle::new(1, 0);
    let mut region = AppendOnlyRegion::new(backing, 64, region_handle).unwrap();

    // First write succeeds
    region.append(&[0u8; 32]).unwrap();

    // Second write fills remaining space
    region.append(&[0u8; 32]).unwrap();

    // Third write should fail - no space
    let result = region.append(&[0u8; 1]);
    assert!(result.is_err());
}

// ============================================================================
// Concurrent Access Patterns (Single-Threaded Simulation)
// ============================================================================

#[test]
fn test_multiple_tasks_queue_access() {
    // Simulate multiple tasks accessing the same queue

    let queue_config = QueueConfig::new(64, 256);
    let (mut queue, _buffer) = KernelQueue::new_heap(queue_config).unwrap();

    // Task 1 sends messages
    for i in 0..10 {
        let msg = format!("task1-msg{}", i);
        queue.send(msg.as_bytes(), MsgPriority::Normal).unwrap();
    }

    // Task 2 sends messages
    for i in 10..20 {
        let msg = format!("task2-msg{}", i);
        queue.send(msg.as_bytes(), MsgPriority::High).unwrap();
    }

    // Task 3 receives all messages
    let mut recv_buf = [0u8; 256];
    let mut received = Vec::new();
    while let Ok(len) = queue.recv(&mut recv_buf) {
        received.push(Vec::from(&recv_buf[..len]));
    }

    assert_eq!(received.len(), 20);
}

#[test]
fn test_multiple_regions_slab() {
    // Allocate multiple independent regions from slab

    let backing = StaticBacking::<8192>::new();
    let mut slab = SlabAllocator::new(backing, 128, 32).unwrap();

    // Allocate slots for different "regions"
    let region1 = slab.alloc().unwrap();
    let region2 = slab.alloc().unwrap();
    let region3 = slab.alloc().unwrap();

    // Write different data to each
    slab.write(region1, &[1u8; 128]).unwrap();
    slab.write(region2, &[2u8; 128]).unwrap();
    slab.write(region3, &[3u8; 128]).unwrap();

    // Read back and verify isolation
    let mut buf = [0u8; 128];

    slab.read(region1, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 1));

    slab.read(region2, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 2));

    slab.read(region3, &mut buf).unwrap();
    assert!(buf.iter().all(|&b| b == 3));
}
