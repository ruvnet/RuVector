//! Integration tests for ruvector-raft
//!
//! These tests exercise the public API of the Raft consensus implementation
//! end-to-end, covering node lifecycle, leader election mechanics, log
//! replication, state transitions, and RPC message handling.

use ruvector_raft::election::{ElectionState, ElectionTimer, VoteTracker, VoteValidator};
use ruvector_raft::log::{LogEntry, RaftLog, Snapshot};
use ruvector_raft::rpc::{
    AppendEntriesRequest, AppendEntriesResponse, RaftMessage, RequestVoteRequest,
    RequestVoteResponse,
};
use ruvector_raft::state::{LeaderState, PersistentState, RaftState, VolatileState};
use ruvector_raft::{RaftNode, RaftNodeConfig};
use std::thread;
use std::time::Duration;

// ---------------------------------------------------------------------------
// 1. Node creation and initialisation
// ---------------------------------------------------------------------------

#[test]
fn test_node_creation_defaults() {
    let config = RaftNodeConfig::new(
        "node-1".to_string(),
        vec![
            "node-1".to_string(),
            "node-2".to_string(),
            "node-3".to_string(),
        ],
    );

    let node = RaftNode::new(config);

    assert_eq!(node.current_state(), RaftState::Follower);
    assert_eq!(node.current_term(), 0);
    assert!(node.current_leader().is_none());
}

#[test]
fn test_node_creation_single_member_cluster() {
    let config = RaftNodeConfig::new("solo".to_string(), vec!["solo".to_string()]);
    let node = RaftNode::new(config);

    assert_eq!(node.current_state(), RaftState::Follower);
    assert_eq!(node.current_term(), 0);
}

#[test]
fn test_node_creation_five_member_cluster() {
    let members: Vec<String> = (1..=5).map(|i| format!("node-{}", i)).collect();
    let config = RaftNodeConfig::new("node-1".to_string(), members.clone());
    let node = RaftNode::new(config);

    assert_eq!(node.current_state(), RaftState::Follower);
    assert_eq!(node.current_term(), 0);
    assert!(node.current_leader().is_none());
}

#[test]
fn test_node_config_custom_timeouts() {
    let mut config = RaftNodeConfig::new(
        "n1".to_string(),
        vec!["n1".to_string(), "n2".to_string(), "n3".to_string()],
    );
    config.election_timeout_min = 500;
    config.election_timeout_max = 1000;
    config.heartbeat_interval = 100;
    config.max_entries_per_message = 50;
    config.snapshot_chunk_size = 128 * 1024;

    // Verify the config is accepted without panics
    let _node = RaftNode::new(config);
}

// ---------------------------------------------------------------------------
// 2. Leader election mechanics (synchronous components)
// ---------------------------------------------------------------------------

#[test]
fn test_leader_election_single_node_wins_immediately() {
    // A cluster of 1 node: quorum is 1, so self-vote wins.
    let mut state = ElectionState::new(1, 150, 300);
    state.start_election(1, &"solo".to_string());

    // The self-vote alone satisfies quorum for cluster-size 1.
    assert!(state.votes.has_quorum());
}

#[test]
fn test_leader_election_three_node_quorum() {
    let mut state = ElectionState::new(3, 150, 300);
    state.start_election(1, &"node-1".to_string());

    // After self-vote: 1 of 2 needed => no quorum yet
    assert!(!state.votes.has_quorum());

    // Second vote reaches quorum
    let won = state.record_vote("node-2".to_string());
    assert!(won);
    assert!(state.votes.has_quorum());
}

#[test]
fn test_leader_election_five_node_quorum() {
    let mut state = ElectionState::new(5, 150, 300);
    state.start_election(1, &"node-1".to_string());

    assert!(!state.votes.has_quorum()); // 1/3

    let won = state.record_vote("node-2".to_string());
    assert!(!won); // 2/3

    let won = state.record_vote("node-3".to_string());
    assert!(won); // 3/3 = quorum
}

#[test]
fn test_duplicate_votes_not_counted() {
    let mut tracker = VoteTracker::new(3);
    tracker.record_vote("node-1".to_string());
    tracker.record_vote("node-1".to_string()); // duplicate
    tracker.record_vote("node-1".to_string()); // duplicate

    assert_eq!(tracker.vote_count(), 1);
    assert!(!tracker.has_quorum());
}

#[test]
fn test_election_timer_fires_after_timeout() {
    let timer = ElectionTimer::new(10, 20);
    assert!(!timer.is_elapsed());

    // Sleep beyond the maximum timeout
    thread::sleep(Duration::from_millis(50));
    assert!(timer.is_elapsed());
}

#[test]
fn test_election_timer_reset_clears_timeout() {
    let mut timer = ElectionTimer::new(10, 20);
    thread::sleep(Duration::from_millis(50));
    assert!(timer.is_elapsed());

    timer.reset();
    assert!(!timer.is_elapsed());
}

#[test]
fn test_election_should_start_after_timeout() {
    let state = ElectionState::new(3, 10, 20);
    thread::sleep(Duration::from_millis(50));
    assert!(state.should_start_election());
}

#[test]
fn test_election_should_not_start_before_timeout() {
    let state = ElectionState::new(3, 5000, 10000);
    assert!(!state.should_start_election());
}

// ---------------------------------------------------------------------------
// 3. Vote validation (Raft safety invariants)
// ---------------------------------------------------------------------------

#[test]
fn test_vote_granted_when_candidate_up_to_date_and_no_prior_vote() {
    assert!(VoteValidator::should_grant_vote(
        1,     // receiver term
        &None, // no prior vote
        5,     // receiver last log index
        1,     // receiver last log term
        &"candidate-A".to_string(),
        2, // candidate term (higher)
        5, // candidate last log index
        1, // candidate last log term
    ));
}

#[test]
fn test_vote_denied_when_candidate_term_is_stale() {
    assert!(!VoteValidator::should_grant_vote(
        3,     // receiver term
        &None, // no prior vote
        5,
        2,
        &"candidate-A".to_string(),
        2, // candidate term is lower
        10,
        3,
    ));
}

#[test]
fn test_vote_denied_when_already_voted_for_different_candidate() {
    assert!(!VoteValidator::should_grant_vote(
        1,
        &Some("candidate-B".to_string()), // already voted for B
        5,
        1,
        &"candidate-A".to_string(), // A is requesting
        1,
        5,
        1,
    ));
}

#[test]
fn test_vote_granted_when_re_voting_for_same_candidate() {
    assert!(VoteValidator::should_grant_vote(
        1,
        &Some("candidate-A".to_string()),
        5,
        1,
        &"candidate-A".to_string(),
        1,
        5,
        1,
    ));
}

#[test]
fn test_vote_denied_when_candidate_log_is_behind() {
    // Candidate has older term on last entry
    assert!(!VoteValidator::should_grant_vote(
        2,
        &None,
        10,
        3, // receiver's last log term is 3
        &"candidate".to_string(),
        2,
        10,
        2, // candidate's last log term is only 2
    ));
}

#[test]
fn test_vote_denied_when_candidate_log_is_shorter_same_term() {
    assert!(!VoteValidator::should_grant_vote(
        1,
        &None,
        10,
        1, // same last log term
        &"candidate".to_string(),
        1,
        5, // shorter log
        1,
    ));
}

// ---------------------------------------------------------------------------
// 4. Log entry append and commit
// ---------------------------------------------------------------------------

#[test]
fn test_log_append_sequential_entries() {
    let mut log = RaftLog::new();
    assert!(log.is_empty());

    let idx1 = log.append(1, b"set x=1".to_vec());
    let idx2 = log.append(1, b"set y=2".to_vec());
    let idx3 = log.append(2, b"set z=3".to_vec());

    assert_eq!(idx1, 1);
    assert_eq!(idx2, 2);
    assert_eq!(idx3, 3);
    assert_eq!(log.last_index(), 3);
    assert_eq!(log.last_term(), 2);
    assert_eq!(log.len(), 3);
    assert!(!log.is_empty());
}

#[test]
fn test_log_get_entry_by_index() {
    let mut log = RaftLog::new();
    log.append(1, b"first".to_vec());
    log.append(1, b"second".to_vec());
    log.append(2, b"third".to_vec());

    let entry = log.get(2).expect("entry at index 2 should exist");
    assert_eq!(entry.term, 1);
    assert_eq!(entry.index, 2);
    assert_eq!(entry.command, b"second");

    assert!(log.get(0).is_none(), "index 0 should return None");
    assert!(log.get(99).is_none(), "out-of-range index should return None");
}

#[test]
fn test_log_entries_from_range() {
    let mut log = RaftLog::new();
    for i in 1..=5 {
        log.append(1, format!("cmd{}", i).into_bytes());
    }

    let entries = log.entries_from(3);
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].index, 3);
    assert_eq!(entries[2].index, 5);
}

#[test]
fn test_log_append_entries_for_replication() {
    let mut log = RaftLog::new();
    log.append(1, b"existing".to_vec());

    let new_entries = vec![
        LogEntry::new(1, 2, b"replicated-1".to_vec()),
        LogEntry::new(2, 3, b"replicated-2".to_vec()),
    ];
    log.append_entries(new_entries).unwrap();

    assert_eq!(log.last_index(), 3);
    assert_eq!(log.last_term(), 2);
}

#[test]
fn test_log_append_entries_rejects_non_sequential() {
    let mut log = RaftLog::new();
    log.append(1, b"existing".to_vec());

    // Index 5 is non-sequential (expected 2)
    let bad_entries = vec![LogEntry::new(1, 5, b"bad".to_vec())];
    let result = log.append_entries(bad_entries);

    assert!(result.is_err(), "non-sequential append should fail");
}

#[test]
fn test_log_truncate_removes_entries_from_index() {
    let mut log = RaftLog::new();
    for i in 1..=5 {
        log.append(1, format!("cmd{}", i).into_bytes());
    }

    log.truncate_from(3).unwrap();
    assert_eq!(log.last_index(), 2);
    assert!(log.get(3).is_none());
    assert!(log.get(4).is_none());
}

#[test]
fn test_log_matches_checks_term_at_index() {
    let mut log = RaftLog::new();
    log.append(1, b"a".to_vec());
    log.append(1, b"b".to_vec());
    log.append(2, b"c".to_vec());

    assert!(log.matches(0, 0), "index 0 always matches");
    assert!(log.matches(1, 1));
    assert!(log.matches(3, 2));
    assert!(!log.matches(3, 1), "wrong term should not match");
    assert!(!log.matches(10, 1), "missing index should not match");
}

// ---------------------------------------------------------------------------
// 5. State transitions (Follower -> Candidate -> Leader)
// ---------------------------------------------------------------------------

#[test]
fn test_state_is_follower_by_default() {
    let state = RaftState::Follower;
    assert!(state.is_follower());
    assert!(!state.is_candidate());
    assert!(!state.is_leader());
}

#[test]
fn test_state_transition_follower_to_candidate() {
    let mut state = RaftState::Follower;
    assert!(state.is_follower());

    state = RaftState::Candidate;
    assert!(state.is_candidate());
    assert!(!state.is_follower());
}

#[test]
fn test_state_transition_candidate_to_leader() {
    let mut state = RaftState::Candidate;
    assert!(state.is_candidate());

    state = RaftState::Leader;
    assert!(state.is_leader());
}

#[test]
fn test_state_transition_leader_step_down_to_follower() {
    let mut state = RaftState::Leader;
    assert!(state.is_leader());

    // On discovering a higher term, leader must step down
    state = RaftState::Follower;
    assert!(state.is_follower());
}

#[test]
fn test_persistent_state_term_increment_clears_vote() {
    let mut ps = PersistentState::new();

    ps.vote_for("candidate-A".to_string());
    assert!(ps.voted_for.is_some());

    ps.increment_term();
    assert_eq!(ps.current_term, 1);
    assert!(
        ps.voted_for.is_none(),
        "incrementing term must clear voted_for"
    );
}

#[test]
fn test_persistent_state_update_term_clears_vote() {
    let mut ps = PersistentState::new();
    ps.vote_for("candidate-A".to_string());

    let updated = ps.update_term(5);
    assert!(updated);
    assert_eq!(ps.current_term, 5);
    assert!(ps.voted_for.is_none());
}

#[test]
fn test_persistent_state_update_term_noop_on_lower() {
    let mut ps = PersistentState::new();
    ps.update_term(5);

    let updated = ps.update_term(3);
    assert!(!updated, "should not update to lower term");
    assert_eq!(ps.current_term, 5);
}

#[test]
fn test_persistent_state_can_vote_for() {
    let mut ps = PersistentState::new();
    assert!(ps.can_vote_for(&"anyone".to_string()));

    ps.vote_for("nodeA".to_string());
    assert!(ps.can_vote_for(&"nodeA".to_string()));
    assert!(!ps.can_vote_for(&"nodeB".to_string()));
}

#[test]
fn test_volatile_state_commit_index_monotonic() {
    let mut vs = VolatileState::new();
    assert_eq!(vs.commit_index, 0);

    vs.update_commit_index(10);
    assert_eq!(vs.commit_index, 10);

    // Attempting to lower commit index should have no effect
    vs.update_commit_index(5);
    assert_eq!(vs.commit_index, 10, "commit index must not decrease");
}

#[test]
fn test_volatile_state_pending_entries() {
    let mut vs = VolatileState::new();
    vs.update_commit_index(10);
    assert_eq!(vs.pending_entries(), 10);

    vs.apply_entries(7);
    assert_eq!(vs.pending_entries(), 3);
    assert_eq!(vs.last_applied, 7);
}

// ---------------------------------------------------------------------------
// 6. Leader state tracking
// ---------------------------------------------------------------------------

#[test]
fn test_leader_state_initialisation() {
    let members = vec!["n2".to_string(), "n3".to_string()];
    let ls = LeaderState::new(&members, 10);

    // next_index initialised to last_log_index + 1
    assert_eq!(ls.get_next_index(&"n2".to_string()), Some(11));
    assert_eq!(ls.get_next_index(&"n3".to_string()), Some(11));

    // match_index initialised to 0
    assert_eq!(ls.get_match_index(&"n2".to_string()), Some(0));
    assert_eq!(ls.get_match_index(&"n3".to_string()), Some(0));
}

#[test]
fn test_leader_state_replication_update() {
    let members = vec!["n2".to_string(), "n3".to_string()];
    let mut ls = LeaderState::new(&members, 10);

    ls.update_replication(&"n2".to_string(), 8);
    assert_eq!(ls.get_match_index(&"n2".to_string()), Some(8));
    assert_eq!(ls.get_next_index(&"n2".to_string()), Some(9));
}

#[test]
fn test_leader_state_decrement_next_index() {
    let members = vec!["n2".to_string()];
    let mut ls = LeaderState::new(&members, 10);

    ls.decrement_next_index(&"n2".to_string());
    assert_eq!(ls.get_next_index(&"n2".to_string()), Some(10));

    // Keep decrementing
    for _ in 0..20 {
        ls.decrement_next_index(&"n2".to_string());
    }
    // Should not go below 1
    assert_eq!(ls.get_next_index(&"n2".to_string()), Some(1));
}

#[test]
fn test_leader_state_commit_index_calculation() {
    let members = vec![
        "n1".to_string(),
        "n2".to_string(),
        "n3".to_string(),
        "n4".to_string(),
    ];
    let mut ls = LeaderState::new(&members, 10);

    ls.update_replication(&"n1".to_string(), 10);
    ls.update_replication(&"n2".to_string(), 8);
    ls.update_replication(&"n3".to_string(), 6);
    ls.update_replication(&"n4".to_string(), 4);

    // Sorted: [4, 6, 8, 10] => median at index 2 => 8
    let commit = ls.calculate_commit_index();
    assert_eq!(commit, 8);
}

// ---------------------------------------------------------------------------
// 7. Snapshot and log compaction
// ---------------------------------------------------------------------------

#[test]
fn test_snapshot_creation_compacts_log() {
    let mut log = RaftLog::new();
    for i in 1..=10 {
        log.append(1, format!("cmd{}", i).into_bytes());
    }

    let snapshot = log
        .create_snapshot(5, b"state-at-5".to_vec(), vec!["n1".to_string()])
        .unwrap();

    assert_eq!(snapshot.last_included_index, 5);
    assert_eq!(snapshot.last_included_term, 1);
    assert_eq!(log.base_index(), 5);
    assert_eq!(log.len(), 5, "entries 6-10 should remain");
    assert!(log.get(3).is_none(), "entries before snapshot should be gone");
    assert!(log.get(6).is_some(), "entries after snapshot should remain");
}

#[test]
fn test_snapshot_install_replaces_log() {
    let mut log = RaftLog::new();
    for i in 1..=5 {
        log.append(1, format!("cmd{}", i).into_bytes());
    }

    let snapshot = Snapshot {
        last_included_index: 10,
        last_included_term: 3,
        data: b"full-state".to_vec(),
        configuration: vec!["n1".to_string(), "n2".to_string()],
    };

    log.install_snapshot(snapshot).unwrap();

    assert_eq!(log.base_index(), 10);
    assert_eq!(log.base_term(), 3);
    assert_eq!(log.len(), 0, "all entries should be cleared");
    assert_eq!(log.last_index(), 10, "last index should be snapshot index");
    assert_eq!(log.last_term(), 3, "last term should be snapshot term");
}

#[test]
fn test_log_term_at_after_snapshot() {
    let mut log = RaftLog::new();
    for i in 1..=10 {
        log.append(1, format!("cmd{}", i).into_bytes());
    }
    log.create_snapshot(5, b"state".to_vec(), vec![]).unwrap();

    // base_index term should be available
    assert_eq!(log.term_at(5), Some(1));
    // Below base_index should return None
    assert_eq!(log.term_at(3), None);
    // Entries after snapshot should still be accessible
    assert_eq!(log.term_at(6), Some(1));
}

// ---------------------------------------------------------------------------
// 8. RPC message construction and serialisation round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_append_entries_heartbeat_construction() {
    let req = AppendEntriesRequest::heartbeat(3, "leader-1".to_string(), 42);

    assert!(req.is_heartbeat());
    assert_eq!(req.term, 3);
    assert_eq!(req.leader_id, "leader-1");
    assert_eq!(req.leader_commit, 42);
    assert!(req.entries.is_empty());
}

#[test]
fn test_append_entries_request_serialisation_roundtrip() {
    let entries = vec![
        LogEntry::new(2, 5, b"write-x".to_vec()),
        LogEntry::new(2, 6, b"write-y".to_vec()),
    ];

    let original = AppendEntriesRequest::new(
        2,
        "leader-1".to_string(),
        4,
        1,
        entries,
        3,
    );

    let bytes = original.to_bytes().unwrap();
    let decoded = AppendEntriesRequest::from_bytes(&bytes).unwrap();

    assert_eq!(original.term, decoded.term);
    assert_eq!(original.leader_id, decoded.leader_id);
    assert_eq!(original.prev_log_index, decoded.prev_log_index);
    assert_eq!(original.prev_log_term, decoded.prev_log_term);
    assert_eq!(original.entries.len(), decoded.entries.len());
    assert_eq!(original.leader_commit, decoded.leader_commit);
}

#[test]
fn test_append_entries_response_success_and_failure() {
    let success = AppendEntriesResponse::success(3, 15);
    assert!(success.success);
    assert_eq!(success.match_index, Some(15));
    assert!(success.conflict_index.is_none());

    let failure = AppendEntriesResponse::failure(3, Some(10), Some(2));
    assert!(!failure.success);
    assert_eq!(failure.conflict_index, Some(10));
    assert_eq!(failure.conflict_term, Some(2));
}

#[test]
fn test_request_vote_serialisation_roundtrip() {
    let original = RequestVoteRequest::new(5, "candidate-X".to_string(), 20, 4);

    let bytes = original.to_bytes().unwrap();
    let decoded = RequestVoteRequest::from_bytes(&bytes).unwrap();

    assert_eq!(original.term, decoded.term);
    assert_eq!(original.candidate_id, decoded.candidate_id);
    assert_eq!(original.last_log_index, decoded.last_log_index);
    assert_eq!(original.last_log_term, decoded.last_log_term);
}

#[test]
fn test_request_vote_response_granted_and_denied() {
    let granted = RequestVoteResponse::granted(5);
    assert!(granted.vote_granted);
    assert_eq!(granted.term, 5);

    let denied = RequestVoteResponse::denied(5);
    assert!(!denied.vote_granted);
    assert_eq!(denied.term, 5);
}

#[test]
fn test_raft_message_envelope_term_extraction() {
    let msg = RaftMessage::AppendEntriesRequest(AppendEntriesRequest::heartbeat(
        7,
        "leader".to_string(),
        0,
    ));
    assert_eq!(msg.term(), 7);

    let msg = RaftMessage::RequestVoteRequest(RequestVoteRequest::new(
        3,
        "candidate".to_string(),
        0,
        0,
    ));
    assert_eq!(msg.term(), 3);

    let msg = RaftMessage::AppendEntriesResponse(AppendEntriesResponse::success(4, 10));
    assert_eq!(msg.term(), 4);

    let msg = RaftMessage::RequestVoteResponse(RequestVoteResponse::denied(9));
    assert_eq!(msg.term(), 9);
}

#[test]
fn test_raft_message_serialisation_roundtrip() {
    let original = RaftMessage::RequestVoteRequest(RequestVoteRequest::new(
        10,
        "c1".to_string(),
        50,
        8,
    ));

    let bytes = original.to_bytes().unwrap();
    let decoded = RaftMessage::from_bytes(&bytes).unwrap();

    assert_eq!(decoded.term(), 10);
}

// ---------------------------------------------------------------------------
// 9. Persistent state serialisation round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_persistent_state_serialisation_roundtrip() {
    let mut ps = PersistentState::new();
    ps.update_term(7);
    ps.vote_for("node-42".to_string());
    ps.log.append(5, b"some-command".to_vec());
    ps.log.append(7, b"another-command".to_vec());

    let bytes = ps.to_bytes().unwrap();
    let restored = PersistentState::from_bytes(&bytes).unwrap();

    assert_eq!(restored.current_term, 7);
    assert_eq!(restored.voted_for, Some("node-42".to_string()));
    assert_eq!(restored.log.last_index(), 2);
    assert_eq!(restored.log.last_term(), 7);
}

// ---------------------------------------------------------------------------
// 10. End-to-end simulated election flow (synchronous)
// ---------------------------------------------------------------------------

/// Simulates a 3-node election without real networking.
/// Walks through the Raft election protocol step by step.
#[test]
fn test_simulated_election_flow_three_nodes() {
    // --- Setup ---
    let mut node1_ps = PersistentState::new();
    let mut node2_ps = PersistentState::new();
    let node3_ps = PersistentState::new();

    let mut node1_election = ElectionState::new(3, 150, 300);

    // --- Node 1 starts election ---
    let mut node1_state = RaftState::Candidate;
    node1_ps.increment_term();
    node1_ps.vote_for("node-1".to_string());
    node1_election.start_election(node1_ps.current_term, &"node-1".to_string());

    assert_eq!(node1_state, RaftState::Candidate);
    assert_eq!(node1_ps.current_term, 1);
    assert!(!node1_election.votes.has_quorum()); // only self-vote

    // --- Node 1 sends RequestVote to node-2 and node-3 ---
    let vote_req = RequestVoteRequest::new(
        node1_ps.current_term,
        "node-1".to_string(),
        node1_ps.log.last_index(),
        node1_ps.log.last_term(),
    );

    // --- Node 2 evaluates vote ---
    let should_grant = VoteValidator::should_grant_vote(
        node2_ps.current_term,
        &node2_ps.voted_for,
        node2_ps.log.last_index(),
        node2_ps.log.last_term(),
        &vote_req.candidate_id,
        vote_req.term,
        vote_req.last_log_index,
        vote_req.last_log_term,
    );
    assert!(should_grant, "node-2 should grant vote to node-1");
    node2_ps.update_term(vote_req.term);
    node2_ps.vote_for("node-1".to_string());

    let node2_resp = RequestVoteResponse::granted(node2_ps.current_term);

    // --- Node 3 evaluates vote ---
    let should_grant = VoteValidator::should_grant_vote(
        node3_ps.current_term,
        &node3_ps.voted_for,
        node3_ps.log.last_index(),
        node3_ps.log.last_term(),
        &vote_req.candidate_id,
        vote_req.term,
        vote_req.last_log_index,
        vote_req.last_log_term,
    );
    assert!(should_grant, "node-3 should grant vote to node-1");

    // --- Node 1 processes node-2's vote response ---
    if node2_resp.vote_granted {
        let won = node1_election.record_vote("node-2".to_string());
        assert!(won, "node-1 should win election with 2 votes in 3-node cluster");
    }

    // --- Node 1 becomes leader ---
    node1_state = RaftState::Leader;
    assert!(node1_state.is_leader());
}

/// Simulates log replication from leader to a follower.
#[test]
fn test_simulated_log_replication() {
    // Leader has entries
    let mut leader_log = RaftLog::new();
    leader_log.append(1, b"cmd-1".to_vec());
    leader_log.append(1, b"cmd-2".to_vec());
    leader_log.append(2, b"cmd-3".to_vec());

    // Follower has only the first entry
    let mut follower_log = RaftLog::new();
    follower_log.append(1, b"cmd-1".to_vec());

    // Leader sends entries 2 and 3
    let entries_to_send = leader_log.entries_from(2);
    assert_eq!(entries_to_send.len(), 2);

    // Verify follower's log matches at prev_log_index=1, prev_log_term=1
    assert!(follower_log.matches(1, 1));

    // Follower appends the entries
    follower_log.append_entries(entries_to_send).unwrap();

    assert_eq!(follower_log.last_index(), 3);
    assert_eq!(follower_log.last_term(), 2);

    // Verify follower's log now matches leader's log
    for i in 1..=3 {
        let leader_entry = leader_log.get(i).unwrap();
        let follower_entry = follower_log.get(i).unwrap();
        assert_eq!(leader_entry.term, follower_entry.term);
        assert_eq!(leader_entry.command, follower_entry.command);
    }
}

/// Tests that a follower correctly handles a conflicting log scenario.
#[test]
fn test_simulated_log_conflict_resolution() {
    // Follower has divergent entries from an old leader
    let mut follower_log = RaftLog::new();
    follower_log.append(1, b"cmd-1".to_vec());
    follower_log.append(1, b"cmd-2-old".to_vec());
    follower_log.append(1, b"cmd-3-old".to_vec());

    // New leader says: prev_log_index=1, prev_log_term=1, new entries for index 2+
    assert!(follower_log.matches(1, 1));

    // Truncate conflicting entries
    follower_log.truncate_from(2).unwrap();
    assert_eq!(follower_log.last_index(), 1);

    // Append corrected entries from new leader
    let corrected = vec![
        LogEntry::new(2, 2, b"cmd-2-new".to_vec()),
        LogEntry::new(2, 3, b"cmd-3-new".to_vec()),
    ];
    follower_log.append_entries(corrected).unwrap();

    assert_eq!(follower_log.last_index(), 3);
    assert_eq!(follower_log.last_term(), 2);
    assert_eq!(follower_log.get(2).unwrap().command, b"cmd-2-new");
    assert_eq!(follower_log.get(3).unwrap().command, b"cmd-3-new");
}

// ---------------------------------------------------------------------------
// 11. Async node operations
// ---------------------------------------------------------------------------

/// The submit_command path sends an InternalMessage through an unbounded
/// channel and then awaits a response on a oneshot-like channel. The node's
/// run-loop must be active to consume the internal message and send the
/// response. Because the RaftNode future is !Send (parking_lot guards),
/// we use a LocalSet to run both the node and the test on the same thread.
#[tokio::test]
async fn test_submit_command_rejected_when_not_leader() {
    let local = tokio::task::LocalSet::new();

    local
        .run_until(async {
            let config = RaftNodeConfig::new(
                "node-1".to_string(),
                vec![
                    "node-1".to_string(),
                    "node-2".to_string(),
                    "node-3".to_string(),
                ],
            );
            let node = std::sync::Arc::new(RaftNode::new(config));

            // Start the node's event loop on a local task (no Send required).
            let node_bg = node.clone();
            tokio::task::spawn_local(async move {
                node_bg.start().await;
            });

            // Yield once so the run-loop task is polled and ready.
            tokio::task::yield_now().await;

            // Node starts as Follower, so submitting a command should fail.
            let result = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                node.submit_command(b"test-data".to_vec()),
            )
            .await;

            match result {
                Ok(Ok(_)) => panic!("follower should reject client commands"),
                Ok(Err(_)) => { /* expected: NotLeader error */ }
                Err(_) => panic!("submit_command timed out"),
            }
        })
        .await;
}
