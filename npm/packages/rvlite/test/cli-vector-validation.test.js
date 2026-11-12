import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const cli=fileURLToPath(new URL('../bin/cli.js',import.meta.url));
const run=args=>spawnSync(process.execPath,[cli,...args],{encoding:'utf8',timeout:10000});
for(const [command,vector] of [...['[1]','[1,2,3]','["1",2]','[null,2]','[]','[1e999,2]','{}'].map(v=>['insert',v]),...['[1]','["1",2]','[null,2]'].map(v=>['search',v])]) {
 test(`${command} refuses invalid ${vector}`,()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rvlite-shape-'));
 try{const db=path.join(dir,'db.json');assert.equal(run(['init','-d','2','-p',db]).status,0);assert.equal(run(['insert','[1,2]','-p',db]).status,0);const before=fs.readFileSync(db);const r=run([command,vector,'-p',db]);assert.notEqual(r.status,0,r.stdout);assert.deepEqual(fs.readFileSync(db),before);}finally{fs.rmSync(dir,{recursive:true,force:true})}
 });
}
test('valid configured and unconfigured vectors remain supported',()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rvlite-control-'));
 try{const db=path.join(dir,'db.json');assert.equal(run(['init','-d','2','-p',db]).status,0);assert.equal(run(['insert','[1,2]','-p',db]).status,0);assert.equal(run(['search','[1,2]','-p',db]).status,0);assert.equal(run(['insert','[1,2,3]','-p',path.join(dir,'unconfigured.json')]).status,0);}finally{fs.rmSync(dir,{recursive:true,force:true})}
});
for(const dimensions of ['0','-1','2.5','2oops','Infinity']) {
 test(`init refuses invalid dimensions ${dimensions}`,()=>{
 const dir=fs.mkdtempSync(path.join(os.tmpdir(),'rvlite-config-'));
 try{const db=path.join(dir,'db.json');const r=run(['init','-d',dimensions,'-p',db]);assert.notEqual(r.status,0,r.stdout);assert.equal(fs.existsSync(db),false);}finally{fs.rmSync(dir,{recursive:true,force:true})}
 });
}
