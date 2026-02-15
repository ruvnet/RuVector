'use strict';
const n = require('@ruvector/rvf-node');
const fs = require('fs');
const os = require('os');
const path = require('path');

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'raw-meta-'));
const storePath = path.join(tmpDir, 'test.rvf');

console.log('S1: create + putMeta');
const db = n.RvfDatabase.create(storePath, { dimension: 4 });
db.putMeta('e:x1', Buffer.from('{"id":"x1","key":"k1"}'));
db.putMeta('e:x2', Buffer.from('{"id":"x2","key":"k2"}'));
console.log('listMetaKeys:', db.listMetaKeys());
console.log('getMeta e:x1:', db.getMeta('e:x1').toString());
console.log('deleteMeta e:x2:', db.deleteMeta('e:x2'));
console.log('listMetaKeys after delete:', db.listMetaKeys());
db.close();

console.log('\nS2: reopen');
const db2 = n.RvfDatabase.open(storePath);
console.log('listMetaKeys:', db2.listMetaKeys());
console.log('getMeta e:x1:', db2.getMeta('e:x1').toString());
console.log('getMeta e:x2:', db2.getMeta('e:x2'));
db2.close();

fs.rmSync(tmpDir, { recursive: true });
console.log('\nPASS');
