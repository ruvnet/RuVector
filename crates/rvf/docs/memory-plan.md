 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Plan: Close RVF N-API Binding Gaps (TDD-First)        

 Context
                                                       
 RVF's Rust runtime supports 20 segment types but the
 N-API layer only exposes vector CRUD + lineage +
 kernel/eBPF. Claude-flow uses SQLite (memory.db) for
 metadata, patterns, and audit trails alongside RVF
 (memory.rvf) for vectors. The v3 rvf-backend.ts keeps
  all entry data in in-memory Maps — lost on restart.

 Goal: Add missing N-API bindings so RVF replaces
 SQLite entirely. TDD-first: write failing tests,
 implement minimally, refactor.

 Orchestration: Use agentic-qe for test
 generation/quality gates and claude-flow for
 multi-agent coordination.

 ---
 DeepWiki Review — Resolved Issues
 #: 1
 Claim: kv_meta conflicts with MetadataStore
 Verdict: WRONG
 Evidence: MetadataStore (filter.rs:65) is per-vector
   field/value pairs for query filters. kv_meta
    is store-level KV in META_SEG. Completely
   separate purposes. Add doc comment to
   clarify.
 ────────────────────────────────────────
 #: 2
 Claim: freeze() N-API impossible with &mut self
 Verdict: WRONG
 Evidence: ingest_batch() (lib.rs:413-425) already
 uses
    &self → guard.as_mut() for mutable access
   through Mutex<Option<RvfStore>>. freeze()
   uses identical pattern.
 ────────────────────────────────────────
 #: 3
 Claim: Boot loading order risk
 Verdict: VALID (minor)
 Evidence: META_SEG loading goes AFTER FileIdentity
   restoration (line 1268) and BEFORE
   seg_writer init (line 1270). Only needs
   self.segment_dir and self.file, both already
    populated at that point.
 ────────────────────────────────────────
 #: 4
 Claim: JsCowStats field mismatch
 Verdict: VALID (minor)
 Evidence: pending_writes is usize in Rust CowStats
   (cow.rs:346), cast to u32 in JsCowStats. All
    6 other fields match exactly.
 ────────────────────────────────────────
 #: 5
 Claim: MembershipFilter methods don't exist
 Verdict: WRONG
 Evidence: Verified in membership.rs:
   contains(u64)->bool :105, add(&mut, u64)
   :56, remove(&mut, u64) :72,
   member_count()->u64 :113. All exist.
 ────────────────────────────────────────
 #: 6
 Claim: Witness chain methods unavailable
 Verdict: WRONG
 Evidence: last_witness_hash(&self)->&[u8;32] :1166,
   query_audited(&mut
   self,...)->Result<Vec<SearchResult>> :368.
   Both verified.
 Corrections applied to plan below: boot loading
 placement clarified, pending_writes cast noted.

 ---
 Execution Strategy

 Agent Orchestration
 Agent: qe-tdd-red
 Role: Write failing Rust tests
 Phase: 1
 ────────────────────────────────────────
 Agent: coder
 Role: Implement Rust code (GREEN)
 Phase: 2
 ────────────────────────────────────────
 Agent: qe-tdd-refactor
 Role: Refactor and harden
 Phase: 3
 ────────────────────────────────────────
 Agent: coder (parallel)
 Role: N-API bindings + TS SDK
 Phase: 4-5
 ────────────────────────────────────────
 Agent: qe-test-architect
 Role: Integration test design
 Phase: 6
 ────────────────────────────────────────
 Agent: production-validator
 Role: End-to-end verification
 Phase: 7
 All agents spawned via Claude Code Task tool with
 run_in_background: true.

 ---
 Phase 1: RED — Failing Rust Tests for META_SEG KV API

 File: crates/rvf/rvf-runtime/src/store.rs (in mod
 tests)

 The TDD plan specifies 12 tests using existing
 TempDir + RvfOptions pattern:

 test_put_get_meta_round_trip          - put then get
 returns same bytes
 test_get_meta_returns_none_missing    - get
 nonexistent key returns None
 test_put_meta_overwrites              - second put
 replaces value
 test_delete_meta_removes_key          - delete
 returns true, get returns None
 test_delete_meta_missing_returns_false - delete
 nonexistent returns false
 test_list_meta_keys                   - returns all
 inserted keys
 test_meta_persists_through_reopen     - close,
 reopen, get still works
 test_meta_survives_compaction         - compact
 preserves META_SEG
 test_readonly_rejects_put_meta        -
 open_readonly, put_meta returns Err
 test_empty_value_allowed              - put_meta("k",
  b"") works
 test_large_value_64kb                 - 64KB value
 round-trips
 test_multiple_puts_latest_on_reopen   - 3 puts,
 reopen loads all 3 keys

 Run: cd crates/rvf/rvf-runtime && cargo test -- meta
 — all 12 must FAIL.

 Phase 2: GREEN — Minimal META_SEG Implementation

 File: crates/rvf/rvf-runtime/src/store.rs

 Step 2a: Add field to RvfStore struct (after line 68)

 /// Store-level key-value metadata persisted in
 META_SEG segments.
 /// NOTE: This is separate from `metadata:
 MetadataStore` (filter.rs) which stores
 /// per-vector field/value pairs for query filtering.
  kv_meta is store-global KV.
 kv_meta: std::collections::HashMap<String, Vec<u8>>,

 Initialize as HashMap::new() in create(), open(),
 open_readonly(), branch(), derive().

 Step 2b: Serialization functions (after
 simple_shake256_256)

 Wire format v1:
 version(u8=1) | count(u32 LE) | [key_len(u16 LE)
 key(UTF-8) value_len(u32 LE) value(bytes)]*

 - serialize_kv_meta(map) -> Vec<u8>
 - deserialize_kv_meta(payload) ->
 Option<HashMap<String, Vec<u8>>>

 Step 2c: Public API (after close(), before kernel
 section ~line 644)

 pub fn put_meta(&mut self, key: &str, value: &[u8])
 -> Result<(), RvfError>
   // check read_only, insert into kv_meta, call
 write_meta_snapshot()

 pub fn get_meta(&self, key: &str) -> Option<&[u8]>
   // kv_meta.get(key).map(|v| v.as_slice())

 pub fn delete_meta(&mut self, key: &str) ->
 Result<bool, RvfError>
   // check read_only, remove from kv_meta,
 write_meta_snapshot if changed

 pub fn list_meta_keys(&self) -> Vec<&str>
   // kv_meta.keys().map(String::as_str).collect()

 Internal helper:
 fn write_meta_snapshot(&mut self) -> Result<(),
 RvfError>
   // serialize kv_meta → write_meta_seg() → push to
 segment_dir → fsync → epoch++ → write_manifest

 Follows the embed_kernel() pattern exactly.

 Step 2d: Boot loading (after FileIdentity restore at
 line 1268, before seg_writer init at line 1270)

 Insert META_SEG loading here — self.segment_dir and
 self.file are already populated:

 // Load store-level KV metadata from latest META_SEG
 let meta_entries: Vec<_> = self.segment_dir.iter()
     .filter(|&&(_, _, _, seg_type)| seg_type ==
 SegmentType::Meta as u8)
     .collect();
 if let Some(&&(_, offset, _, _)) =
 meta_entries.iter().max_by_key(|&&&(seg_id, _, _, _)|
  seg_id) {
     let mut reader = BufReader::new(&self.file);
     let (_hdr, payload) =
 read_path::read_segment_payload(&mut reader,
 offset)?;
     if let Some(map) = deserialize_kv_meta(&payload)
 {
         self.kv_meta = map;
     }
 }

 Boot order (verified safe): manifest parse → deletion
  bitmap → VEC_SEG → FileIdentity → META_SEG KV →
 seg_writer init

 Step 2e: Run tests

 cd crates/rvf/rvf-runtime && cargo test -- meta — all
  12 must PASS.

 Existing functions reused (NO new file I/O code):
 Function: SegmentWriter::write_meta_seg()
 Location: write_path.rs:100
 Purpose: Writes META_SEG payload to file
 ────────────────────────────────────────
 Function: read_path::read_segment_payload()
 Location: read_path.rs:293
 Purpose: Generic segment reader
 ────────────────────────────────────────
 Function: scan_preservable_segments()
 Location: store.rs:1406
 Purpose: Already preserves META_SEG during compaction
 ────────────────────────────────────────
 Function: map_rvf_err()
 Location: lib.rs:25 (N-API)
 Purpose: Converts RvfError to napi::Error
 Phase 3: REFACTOR

 - Add key length validation (reject keys > 65535
 bytes, u16 limit)
 - Add total payload size limit (16 MB)
 - Consider witness integration for
 put_meta/delete_meta
 - Add flush_meta() method (see Scalability Note
 below)

 Scalability Note: Snapshot-per-write

 write_meta_snapshot() serializes the ENTIRE kv_meta
 HashMap on every put_meta() call. With 1,000 entries
 at ~3KB each, every write triggers ~3MB disk I/O.

 Accepted for Phase 2 — fine for initial target
 (~hundreds of keys).

 Mitigation (Phase 3): Add flush_meta() public method.
  put_meta()/delete_meta() update the in-memory
 HashMap only (no disk write). Callers explicitly
 flush via:
 - store.flush_meta() — explicit
 - store.compact() — already writes everything
 - store.close() — auto-flush before close (drop
 guard)

 rvf-backend.ts (Phase 7) batches writes in-memory and
  calls flush_meta() on shutdown()/compact()/periodic
 timer.

 Future (deferred): Incremental delta model — each
 META_SEG holds a single delta, boot replays all
 deltas, compaction merges into snapshot.

 ---
 Phase 4: RED/GREEN — N-API Bindings

 File: crates/rvf/rvf-node/src/lib.rs

 N-API struct is RvfDatabase (not RvfStore). Pattern:
 self.inner.lock() -> guard.as_ref()/as_mut().

 4A. META_SEG KV Methods (4 methods)

 #[napi]
 pub fn put_meta(&self, key: String, value: Buffer) ->
  Result<()>
   // lock, as_mut, store.put_meta(&key,
 &value).map_err(map_rvf_err)

 #[napi]
 pub fn get_meta(&self, key: String) ->
 Result<Option<Buffer>>
   // lock, as_ref,
 store.get_meta(&key).map(Buffer::from)

 #[napi]
 pub fn delete_meta(&self, key: String) ->
 Result<bool>
   // lock, as_mut,
 store.delete_meta(&key).map_err(map_rvf_err)

 #[napi]
 pub fn list_meta_keys(&self) -> Result<Vec<String>>
   // lock, as_ref, store.list_meta_keys().map(|s|
 s.to_string()).collect()

 4B. COW Branching Methods (5 methods)

 #[napi]
 pub fn branch(&self, child_path: String) ->
 Result<RvfDatabase>
   // lock, as_ref,
 store.branch(Path::new(&child_path)), wrap in
 RvfDatabase

 #[napi]
 pub fn freeze(&self) -> Result<()>
   // lock, as_mut, store.freeze()

 #[napi]
 pub fn cow_stats(&self) -> Result<Option<JsCowStats>>
   // lock, as_ref, store.cow_stats().map(|s|
 JsCowStats::from(s))

 #[napi]
 pub fn is_cow_child(&self) -> Result<bool>
   // lock, as_ref, store.is_cow_child()

 #[napi]
 pub fn parent_store_path(&self) ->
 Result<Option<String>>
   // lock, as_ref, store.parent_path().map(|p|
 p.to_string_lossy().into())

 New TS-facing type:
 #[napi(object)]
 pub struct JsCowStats {
   pub cluster_count: u32,
   pub local_cluster_count: u32,
   pub cluster_size: u32,
   pub vectors_per_cluster: u32,
   pub frozen: bool,
   pub snapshot_epoch: u32,
   pub pending_writes: u32,  // usize in Rust, safe
 cast — count won't exceed u32
 }

 4C. Membership Filter Methods (4 methods)

 Option handling: store.membership_filter() returns
 None on non-branched (root) stores.
 Method: membership_contains
 When filter is None: Return false (no filter = all
   visible)
 ────────────────────────────────────────
 Method: membership_count
 When filter is None: Return null (JS null, not 0)
 ────────────────────────────────────────
 Method: membership_add
 When filter is None: Return error: "no membership
   filter (not a branch)"
 ────────────────────────────────────────
 Method: membership_remove
 When filter is None: Return error: "no membership
   filter (not a branch)"
 #[napi]
 pub fn membership_contains(&self, id: i64) ->
 Result<bool>
   // lock, as_ref
   // match store.membership_filter() {
   //   Some(f) => Ok(f.contains(id as u64)),
   //   None => Ok(false),  // no filter = all visible
   // }

 #[napi]
 pub fn membership_add(&self, id: i64) -> Result<()>
   // lock, as_mut
   // match store.membership_filter_mut() {
   //   Some(f) => { f.add(id as u64); Ok(()) },
   //   None => Err(napi::Error::from_reason("No
 membership filter (not a COW branch)")),
   // }

 #[napi]
 pub fn membership_remove(&self, id: i64) ->
 Result<()>
   // lock, as_mut, same pattern as membership_add

 #[napi]
 pub fn membership_count(&self) -> Result<Option<f64>>
   // lock, as_ref
   // Ok(store.membership_filter().map(|f|
 f.member_count() as f64))

 Note: Using i64 instead of BigInt for N-API
 simplicity (JS numbers up to 2^53).

 4D. Witness Chain Methods (2 methods)

 #[napi]
 pub fn last_witness_hash(&self) -> Result<Buffer>
   // lock, as_ref,
 Buffer::from(store.last_witness_hash().as_slice())

 #[napi]
 pub fn query_audited(&self, vector: Vec<f64>, k: u32,
  ef_search: Option<u32>) -> Result<Vec<SearchResult>>
   // lock, as_mut (query_audited needs &mut self!),
 convert f64->f32
   // build QueryOptions { ef_search:
 ef_search.unwrap_or(100), ..Default::default() }
   // call store.query_audited(&f32_vec, k as usize,
 &opts)

 Total: 15 new #[napi] methods + 1 #[napi(object)]
 struct.

 Phase 5: TypeScript SDK Updates

 Files: npm/packages/rvf/src/backend.ts,
 npm/packages/rvf/src/database.ts

 5A. Backend interface additions (14 methods)

 Add to RvfBackend interface and implement in
 NodeBackend:
 - putMeta(key, value), getMeta(key), deleteMeta(key),
  listMetaKeys()
 - branch(childPath), freeze(), cowStats(),
 isCowChild(), parentStorePath()
 - membershipContains(id), membershipAdd(id),
 membershipRemove(id), membershipCount()
 - lastWitnessHash(), queryAudited(vector, k)

 WasmBackend stubs throw "not supported in WASM".

 5B. RvfDatabase class additions (14 methods)

 Same methods as public API on RvfDatabase class,
 delegating to backend.

 5C. Type exports

 export interface CowStats {
   clusterCount: number;
   localClusterCount: number;
   clusterSize: number;
   vectorsPerCluster: number;
   frozen: boolean;
   snapshotEpoch: number;
   pendingWrites: number;
 }

 Phase 6: MCP Server Tools

 File: npm/packages/rvf-mcp-server/src/server.ts

 Add 7 new tools following existing
 this.mcp.tool(name, desc, zodSchema, handler)
 pattern:
 Tool: put_meta
 Params: storeId, key, value
 Returns: void
 ────────────────────────────────────────
 Tool: get_meta
 Params: storeId, key
 Returns: value or null
 ────────────────────────────────────────
 Tool: list_meta_keys
 Params: storeId
 Returns: string[]
 ────────────────────────────────────────
 Tool: branch
 Params: storeId, childPath
 Returns: new storeId
 ────────────────────────────────────────
 Tool: freeze
 Params: storeId
 Returns: void
 ────────────────────────────────────────
 Tool: cow_stats
 Params: storeId
 Returns: CowStats or null
 ────────────────────────────────────────
 Tool: witness_hash
 Params: storeId
 Returns: hex string
 Phase 7: Update rvf-backend.ts

 File: /Users/danielalberttis/Desktop/Projects/claude-
 flow/v3/@claude-flow/memory/src/rvf-backend.ts

 This is the critical consumer. Current state: keeps
 all entry data in in-memory Maps (lost on restart).

 Changes:

 1. Replace this.entries Map with META_SEG
 persistence:
   - On store(): call db.putMeta(entry.id,
 JSON.stringify(entry)) after vector ingest
   - On get(): call db.getMeta(id) instead of
 this.entries.get(id)
   - On initialize(): load all entries from
 db.listMetaKeys() + db.getMeta() into cache
   - On delete(): call db.deleteMeta(id) alongside
 vector delete
 2. Replace this.namespaceIndex / this.keyIndex with
 META_SEG keys:
   - Store index maps as meta keys: ns::{namespace} →
 JSON array of entry IDs
   - Store key index as: key::{namespace}:{key} →
 entry ID
 3. Add COW branching support:
   - branch() method calls db.branch() (not just
 db.derive())
   - freeze() method calls db.freeze()
 4. Add witness chain support:
   - queryAudited() passes through to
 db.queryAudited()
   - Expose lastWitnessHash() for external
 verification

 Result:

 - No more in-memory-only data — everything persists
 in .rvf
 - No more JSON sidecar (rvf.metadata.json becomes
 unnecessary)
 - No more SQLite (memory.db becomes unnecessary)

 Phase 8: Build & Deploy

 # 1. Run Rust tests
 cd crates/rvf/rvf-runtime && cargo test -- meta

 # 2. Build native binary
 cd crates/rvf/rvf-node && cargo build --release

 # 3. Copy binary
 cp target/release/librvf_node.dylib \
    /path/to/node_modules/@ruvector/rvf-node/rvf-node.
 darwin-arm64.node

 # 4. Copy to nested location if exists
 cp target/release/librvf_node.dylib \
    /path/to/node_modules/@ruvector/rvf/node_modules/@
 ruvector/rvf-node/rvf-node.darwin-arm64.node

 # 5. Smoke test
 node -e "
   const n = require('@ruvector/rvf-node');
   const db =
 n.RvfDatabase.create('/tmp/meta_test.rvf',
 {dimension:4});
   db.putMeta('hello', Buffer.from('world'));
   console.log(db.getMeta('hello').toString()); //
 'world'
   console.log(db.isCowChild()); // false
   db.close();
 "

 ---
 Files Modified (Summary)
 #: 1
 File: crates/rvf/rvf-runtime/src/store.rs
 What Changes: Add kv_meta field, 4 public methods,
   serialization, boot loading, 12 tests
 ────────────────────────────────────────
 #: 2
 File: crates/rvf/rvf-node/src/lib.rs
 What Changes: Add 15 #[napi] methods + JsCowStats
   struct
 ────────────────────────────────────────
 #: 3
 File: npm/packages/rvf/src/backend.ts
 What Changes: Add 14 methods to interface +
   NodeBackend
 ────────────────────────────────────────
 #: 4
 File: npm/packages/rvf/src/database.ts
 What Changes: Add 14 public methods
 ────────────────────────────────────────
 #: 5
 File: npm/packages/rvf-mcp-server/src/server.ts
 What Changes: Add 7 MCP tools
 ────────────────────────────────────────
 #: 6
 File: claude-flow/v3/.../rvf-backend.ts
 What Changes: Replace in-memory Maps with META_SEG
   persistence
 Verification

 TDD (throughout)

 - RED: 12 Rust tests fail → GREEN: all pass →
 REFACTOR: harden
 - N-API smoke test after each binding added

 Integration (MCP)

 1. rvf_create_store → put_meta → get_meta (roundtrip)
 2. rvf_create_store → rvf_ingest → query_audited →
 witness_hash (chain advanced)
 3. rvf_create_store → branch → verify child cow_stats

 End-to-End (claude-flow)

 1. Restart claude-flow MCP
 2. memory_store → memory_retrieve → verify
 .swarm/memory.rvf persists entry data
 3. Restart again → verify data survives (no more
 in-memory loss)
 4. Confirm memory.db and rvf.metadata.json are no
 longer needed

 Execution Order

 1. Phase 1-3 (Rust META_SEG TDD cycle) — the
 bottleneck
 2. Phase 4 (N-API bindings) — depends on Phase 2
 3. Phase 8 build immediately after Phase 4
 4. Phase 5 (TS SDK) — can parallelize interface work
 with Phase 4
 5. Phase 6 (MCP tools) — depends on Phase 5
 6. Phase 7 (rvf-backend.ts) — depends on Phase 5
 7. Verification throughout