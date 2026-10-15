import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
const cli = fileURLToPath(new URL('../bin/cli.js', import.meta.url));
function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvlite-literal-ids-'));
  t.after(() => fs.rmSync(dir, {recursive:true,force:true}));
  return path.join(dir, 'db.json');
}
function run(db,args) {
  return spawnSync(process.execPath,[cli,...args,'--path',db], {
    encoding:'utf8', timeout:10000,env:{...process.env,NO_COLOR:'1'}
  });
}
for (const id of ['__proto__','constructor','toString','ordinary']) {
  test(`literal ${id} persists and can be fetched/deleted`,t=>{
    const db=fixture(t);
    assert.equal(run(db,['insert','[1,0]','--id',id]).status,0);
    const data=JSON.parse(fs.readFileSync(db));
    assert.equal(Object.hasOwn(data.vectors,id),true);
    const got=run(db,['get',id]);
    assert.equal(got.status,0,got.stderr);
    assert.deepEqual(JSON.parse(got.stdout).vector,[1,0]);
    assert.equal(run(db,['delete',id]).status,0);
    assert.equal(Object.hasOwn(JSON.parse(fs.readFileSync(db)).vectors,id),false);
    assert.equal(run(db,['get',id]).status,1);
  });
}
for(const id of ['__proto__','constructor','toString']) {
  for(const command of ['get','delete']) {
    test(`${command} does not invent missing inherited ID ${id}`,t=>{
      const db=fixture(t);
      assert.equal(run(db,['insert','[1,0]','--id','keeper']).status,0);
      const bytes=fs.readFileSync(db);
      assert.equal(run(db,[command,id]).status,1);
      assert.deepEqual(fs.readFileSync(db),bytes);
    });
  }
}
test('batch insertion persists a literal prototype-named ID',t=>{
  const db=fixture(t),input=path.join(path.dirname(db),'batch.json');
  fs.writeFileSync(input,JSON.stringify([{id:'__proto__',vector:[1,0]},{id:'keeper',vector:[0,1]}]));
  assert.equal(run(db,['batch-insert',input]).status,0);
  const data=JSON.parse(fs.readFileSync(db));
  assert.equal(Object.hasOwn(data.vectors,'__proto__'),true);
  assert.deepEqual(data.vectors.keeper.vector,[0,1]);
});
test('replacing a literal ID keeps the current last-write behavior',t=>{
  const db=fixture(t);
  assert.equal(run(db,['insert','[1,0]','--id','constructor']).status,0);
  assert.equal(run(db,['insert','[0,1]','--id','constructor']).status,0);
  const got=run(db,['get','constructor']);
  assert.equal(got.status,0);assert.deepEqual(JSON.parse(got.stdout).vector,[0,1]);
});

test('automatic IDs do not overwrite explicit generated-looking IDs',t=>{
  const db=fixture(t);
  assert.equal(run(db,['insert','[1,0]','--id','vec_1']).status,0);
  assert.equal(run(db,['insert','[0,1]']).status,0);
  const data=JSON.parse(fs.readFileSync(db));
  assert.deepEqual(Object.keys(data.vectors).sort(),['vec_1','vec_2']);
  assert.deepEqual(data.vectors.vec_1.vector,[1,0]);
  assert.deepEqual(data.vectors.vec_2.vector,[0,1]);
});
test('mixed batch custom and automatic IDs remain distinct',t=>{
  const db=fixture(t), input=path.join(path.dirname(db),'batch.json');
  fs.writeFileSync(input,JSON.stringify([{id:'vec_1',vector:[1,0]},{vector:[0,1]}]));
  assert.equal(run(db,['batch-insert',input]).status,0);
  const data=JSON.parse(fs.readFileSync(db));
  assert.deepEqual(Object.keys(data.vectors).sort(),['vec_1','vec_2']);
  assert.deepEqual(data.vectors.vec_1.vector,[1,0]);
});
test('allocation skips occupied IDs and preserves available gaps',t=>{
  const db=fixture(t);
  assert.equal(run(db,['insert','[1,0]','--id','vec_1']).status,0);
  assert.equal(run(db,['insert','[1,1]','--id','vec_3']).status,0);
  assert.equal(run(db,['insert','[0,1]']).status,0);
  assert.equal(run(db,['insert','[0,2]']).status,0);
  const data=JSON.parse(fs.readFileSync(db));
  assert.deepEqual(Object.keys(data.vectors).sort(),['vec_1','vec_2','vec_3','vec_4']);
  assert.deepEqual(data.vectors.vec_1.vector,[1,0]);
  assert.deepEqual(data.vectors.vec_3.vector,[1,1]);
});
test('ordinary automatic allocation remains monotonic',t=>{
  const db=fixture(t);
  assert.equal(run(db,['insert','[1,0]']).status,0);
  assert.equal(run(db,['insert','[0,1]']).status,0);
  assert.deepEqual(Object.keys(JSON.parse(fs.readFileSync(db)).vectors).sort(),['vec_1','vec_2']);
});


test('an exhausted counter refuses automatic allocation without overwriting',t=>{
  const db=fixture(t);
  fs.writeFileSync(db,JSON.stringify({vectors:{keeper:{vector:[1,0],norm:1}},graph:{nodes:{},edges:{}},triples:[],nextId:Number.MAX_SAFE_INTEGER+1}));
  const original=fs.readFileSync(db);
  assert.equal(run(db,['insert','[0,1]']).status,1);
  assert.deepEqual(fs.readFileSync(db),original);
  assert.equal(run(db,['insert','[0,1]','--id','explicit']).status,0);
  assert.equal(Object.hasOwn(JSON.parse(fs.readFileSync(db)).vectors,'explicit'),true);
});


test('automatic batch IDs also avoid later caller-supplied IDs',t=>{
  const db=fixture(t), input=path.join(path.dirname(db),'batch.json');
  fs.writeFileSync(input,JSON.stringify([{vector:[1,0]},{id:'vec_1',vector:[0,1]}]));
  assert.equal(run(db,['batch-insert',input]).status,0);
  const data=JSON.parse(fs.readFileSync(db));
  assert.deepEqual(Object.keys(data.vectors).sort(),['vec_1','vec_2']);
  assert.deepEqual(data.vectors.vec_2.vector,[1,0]);
  assert.deepEqual(data.vectors.vec_1.vector,[0,1]);
});
test('explicit duplicate IDs in a batch retain last-write replacement',t=>{
  const db=fixture(t), input=path.join(path.dirname(db),'batch.json');
  fs.writeFileSync(input,JSON.stringify([{id:'same',vector:[1,0]},{id:'same',vector:[0,1]}]));
  assert.equal(run(db,['batch-insert',input]).status,0);
  const data=JSON.parse(fs.readFileSync(db));
  assert.deepEqual(Object.keys(data.vectors),['same']);
  assert.deepEqual(data.vectors.same.vector,[0,1]);
});
