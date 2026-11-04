'use strict';
// Raft safety regressions. Run after `npm run build`: node --test test/*.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const { RaftNode, RaftEvent, RaftState } = require('../dist/index.js');

function waitFor(emitter, event, ms = 2000) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(`timeout waiting for ${event}`)), ms);
    emitter.once(event, (e) => { clearTimeout(t); resolve(e); });
  });
}

test('leader does not commit an entry that no follower has acknowledged', async () => {
  const node = new RaftNode({ nodeId: 'a', peers: ['b', 'c'], electionTimeout: [1, 2], heartbeatInterval: 10_000 });
  node.setTransport({
    requestVote: async (_peer, req) => ({ term: req.term, voteGranted: true }),
    appendEntries: async () => { throw new Error('unreachable'); },
  });
  const elected = waitFor(node, RaftEvent.LeaderElected);
  node.start();
  await elected;
  await node.propose('x');
  assert.equal(node.commitIndex, 0, 'entry was committed with only the leader holding it');
  node.stop();
});

test('commit needs a true majority: 5 nodes, 1 follower ack is not enough, 2 is', () => {
  const s = new RaftState('a', ['b', 'c', 'd', 'e']);
  return (async () => {
    await s.setTerm(1);
    await s.log.appendCommand(1, 'x');
    s.initLeaderState();
    s.setMatchIndex('b', 1);
    assert.equal(s.updateCommitIndex(), false, 'committed with 2/5');
    assert.equal(s.commitIndex, 0);
    s.setMatchIndex('c', 1);
    assert.equal(s.updateCommitIndex(), true, 'did not commit with 3/5');
    assert.equal(s.commitIndex, 1);
  })();
});

test('single-node cluster still commits its own entries', async () => {
  const node = new RaftNode({ nodeId: 'solo', peers: [], electionTimeout: [1, 2], heartbeatInterval: 10_000 });
  node.setTransport({ requestVote: async () => ({ term: 0, voteGranted: false }), appendEntries: async () => ({ term: 0, success: true }) });
  const elected = waitFor(node, RaftEvent.LeaderElected);
  node.start();
  await elected;
  await node.propose('x');
  assert.equal(node.commitIndex, 1);
  node.stop();
});

test('vote grants from an earlier election do not count toward a later term', async () => {
  // Peer b answers term-1's RequestVote only after the node has moved on to term 2.
  const pending = [];
  const node = new RaftNode({ nodeId: 'a', peers: ['b', 'c'], electionTimeout: [1, 2], heartbeatInterval: 10_000 });
  node.setTransport({
    requestVote: (_peer, req) => new Promise((resolve) => pending.push({ req, resolve })),
    appendEntries: async () => { throw new Error('unreachable'); },
  });
  node.start();
  // Wait until term 2's election is in flight (term-1 requests still unresolved).
  while (node.currentTerm < 2) await new Promise((r) => setTimeout(r, 1));
  node.stop(); // freeze further elections; node stays Candidate in term 2
  const stale = pending.filter((p) => p.req.term === 1);
  assert.ok(stale.length >= 1);
  // A grant in term 1 (response term 1 < current term 2) must not elect us in term 2.
  for (const p of stale) p.resolve({ term: 1, voteGranted: true });
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(node.isLeader, false, 'became leader in term 2 on term-1 votes');
});

test('follower reports matchIndex as prevLogIndex + entries, not its whole log', async () => {
  const node = new RaftNode({ nodeId: 'f', peers: ['l'] });
  // Follower holds stale uncommitted entries 1..3 from term 1.
  node.loadState({ currentTerm: 1, votedFor: null, log: [1, 2, 3].map((i) => ({ term: 1, index: i, command: i, timestamp: 0 })) });
  // Term-2 leader sends only entry 1 (matching) — entries 2..3 are untouched but unverified.
  const res = await node.handleAppendEntries({
    term: 2, leaderId: 'l', prevLogIndex: 0, prevLogTerm: 0,
    entries: [{ term: 1, index: 1, command: 1, timestamp: 0 }], leaderCommit: 3,
  });
  assert.equal(res.success, true);
  assert.equal(res.matchIndex, 1);
  assert.equal(node.commitIndex, 1, 'follower committed entries the leader never verified');
  node.stop();
});

test('3-node in-memory cluster elects one leader and replicates a committed entry', async () => {
  const ids = ['n1', 'n2', 'n3'];
  const nodes = new Map();
  const applied = new Map(ids.map((id) => [id, []]));
  for (const id of ids) {
    const n = new RaftNode({ nodeId: id, peers: ids.filter((p) => p !== id), electionTimeout: [20, 60], heartbeatInterval: 5 });
    n.setStateMachine({ apply: async (c) => { applied.get(id).push(c); } });
    nodes.set(id, n);
  }
  for (const n of nodes.values()) {
    n.setTransport({
      requestVote: (peer, req) => nodes.get(peer).handleRequestVote(req),
      appendEntries: (peer, req) => nodes.get(peer).handleAppendEntries(req),
    });
  }
  for (const n of nodes.values()) n.start();
  let leader;
  for (let i = 0; i < 400 && !leader; i++) {
    await new Promise((r) => setTimeout(r, 5));
    leader = [...nodes.values()].find((n) => n.isLeader);
  }
  assert.ok(leader, 'no leader elected');
  await leader.propose('cmd-1');
  for (let i = 0; i < 200 && ids.some((id) => applied.get(id).length === 0); i++) {
    await new Promise((r) => setTimeout(r, 5));
  }
  for (const n of nodes.values()) n.stop();
  const term = leader.currentTerm;
  assert.equal([...nodes.values()].filter((n) => n.isLeader && n.currentTerm === term).length, 1);
  for (const id of ids) assert.deepEqual(applied.get(id), ['cmd-1'], `${id} did not apply`);
});
