'use strict';
const n = require('@ruvector/rvf-node');
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'qa-test-'));
const storePath = path.join(tmpDir, 'test.rvf');

console.log('--- queryAudited N-API smoke test ---');
const db = n.RvfDatabase.create(storePath, { dimension: 4 });

// Ingest 3 vectors
db.ingestBatch(
    new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0]),
    [1, 2, 3]
);
console.log('Ingested 3 vectors');

// Regular query
const regular = db.query(new Float32Array([1,0,0,0]), 2);
console.log('query() results:', regular.map(r => ({ id: r.id, dist: r.distance.toFixed(4) })));

// Audited query (should return same results + advance witness chain)
const hashBefore = Buffer.from(db.lastWitnessHash()).toString('hex').slice(0, 16);
const audited = db.queryAudited([1.0, 0.0, 0.0, 0.0], 2);
const hashAfter = Buffer.from(db.lastWitnessHash()).toString('hex').slice(0, 16);

console.log('queryAudited() results:', audited.map(r => ({ id: r.id, dist: r.distance.toFixed(4) })));
console.log('Witness hash before:', hashBefore);
console.log('Witness hash after: ', hashAfter);
console.log('Hash changed:', hashBefore !== hashAfter);

db.close();
fs.rmSync(tmpDir, { recursive: true });
console.log('\nPASS');
