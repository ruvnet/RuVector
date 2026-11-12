import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';
import { IDBFactory } from 'fake-indexeddb';

// Exercise the exported class in isolation from the generated WASM engine.
// IndexedDB transactions use the spec-compatible test implementation;
// independent browser replay verifies the same schedules against Chromium.
function fixture() {
  const source = fs.readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');
  const start = source.indexOf('export class BrowserWriterLease');
  const end = source.indexOf('// ── Epoch Sync', start);
  assert.ok(start >= 0 && end > start);
  const classSource = source.slice(start, end).replace('export class', 'class');
  const script = ts.transpileModule(classSource, {compilerOptions: {target: ts.ScriptTarget.ES2020}}).outputText;
  const callbacks = [];
  const context = vm.createContext({indexedDB: new IDBFactory(), Date, Math, Promise,
    setTimeout, setInterval: fn => (callbacks.push(fn), callbacks.length), clearInterval: () => {},
    addEventListener: () => {}, removeEventListener: () => {}});
  vm.runInContext(script + ';globalThis.Lease = BrowserWriterLease;', context);
  return {L: context.Lease, callbacks};
}
async function read(L, key) {
  const db = await L.openDb();
  try { return await L.idbGet(db, key); } finally { db.close(); }
}
async function expire(L, key) {
  const db = await L.openDb();
  try { await L.idbPut(db, {...await L.idbGet(db, key), ts: 0}); }
  finally { db.close(); }
}

test('only one simultaneous contender acquires a store', async () => {
  const {L} = fixture(); const first = new L(); const second = new L();
  const original = L.idbPut;
  // Delay a contender after its absence read, a legal asynchronous schedule.
  L.idbPut = async (db, record) => {
    if (record.holder === second.holderId()) await new Promise(r => setTimeout(r, 50));
    return original.call(L, db, record);
  };
  const acquired = await Promise.all([first.acquire('store', 200), second.acquire('store', 200)]);
  assert.equal(acquired.filter(Boolean).length, 1);
  const winner = acquired[0] ? first : second;
  assert.equal((await read(L, 'store')).holder, winner.holderId());
  await first.release(); await second.release();
});

test('an expired owner cannot release its successor lease', async () => {
  const {L} = fixture(); const old = new L(); const next = new L();
  assert.equal(await old.acquire('store', 200), true); await expire(L, 'store');
  assert.equal(await next.acquire('store', 200), true);
  await old.release();
  assert.equal((await read(L, 'store'))?.holder, next.holderId());
  await next.release(); assert.equal(await read(L, 'store'), undefined);
});

test('an expired owner heartbeat cannot overwrite its successor lease', async () => {
  const {L, callbacks} = fixture(); const old = new L(); const next = new L();
  assert.equal(await old.acquire('store', 200), true); const oldBeat = callbacks[0];
  await expire(L, 'store'); assert.equal(await next.acquire('store', 200), true);
  await oldBeat(); assert.equal((await read(L, 'store')).holder, next.holderId());
  await old.release(); await next.release();
});

test('ordinary acquire, heartbeat and release preserve the owner', async () => {
  const {L, callbacks} = fixture(); const owner = new L();
  assert.equal(await owner.acquire('store', 200), true);
  await callbacks[0](); assert.equal((await read(L, 'store')).holder, owner.holderId());
  await owner.release(); assert.equal(await read(L, 'store'), undefined);
  assert.equal(await owner.acquire('store', 200), true); await owner.release();
});


test('aborted lease writes cannot be acknowledged as acquired', async () => {
  const {L} = fixture(); const open = L.openDb;
  L.openDb = async () => {
    const db = await open.call(L); const transaction = db.transaction.bind(db);
    db.transaction = (...args) => {
      const tx = transaction(...args); const objectStore = tx.objectStore.bind(tx);
      tx.objectStore = (...storeArgs) => {
        const store = objectStore(...storeArgs); const put = store.put.bind(store);
        store.put = (...values) => {
          const request = put(...values);
          request.addEventListener('success', () => tx.abort());
          return request;
        };
        return store;
      };
      return tx;
    };
    return db;
  };
  const owner = new L();
  await assert.rejects(owner.acquire('store', 200), /Lease transaction|Abort/);
  assert.equal(await read(L, 'store'), undefined);
});


test('one instance serializes release and reacquisition without losing either store', async () => {
  const {L} = fixture(); const owner = new L();
  assert.equal(await owner.acquire('old', 200), true);
  assert.equal(await owner.acquire('different', 200), false);
  const open = L.openDb; let resume; let reached;
  const blocked = new Promise(r => {reached = r;});
  const gate = new Promise(r => {resume = r;});
  let first = true;
  L.openDb = async () => {
    if (first) {first = false; reached(); await gate;}
    return open.call(L);
  };
  const releasing = owner.release(); await blocked;
  const acquiring = owner.acquire('new', 200); resume();
  await releasing; assert.equal(await acquiring, true);
  assert.equal(await read(L, 'old'), undefined);
  assert.equal((await read(L, 'new')).holder, owner.holderId());
  await owner.release(); assert.equal(await read(L, 'new'), undefined);
});
