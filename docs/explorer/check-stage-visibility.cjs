// The stage visibility observer must honour the newest entry in a batched callback.
// explorer-shell.js moves #stage into the workspace at load, so one delivery can carry [hidden, visible];
// reading only the first entry left the stage marked hidden and the canvas never drew.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(__dirname+'/index.html','utf8');
const line=html.split('\n').find(l=>l.includes('new IntersectionObserver(')&&l.includes('stageVisible'));
assert(line,'stage IntersectionObserver not found');
let cb;const ctx={stageVisible:true,$:()=>({}),IntersectionObserver:class{constructor(f){cb=f}observe(){}}};
vm.createContext(ctx);vm.runInContext('var stageVisible=true;'+line,ctx);
assert.equal(typeof cb,'function','observer callback not registered');
cb([{isIntersecting:false},{isIntersecting:true}]);assert.equal(vm.runInContext('stageVisible',ctx),true,'batched hidden→visible must end visible');
cb([{isIntersecting:true},{isIntersecting:false}]);assert.equal(vm.runInContext('stageVisible',ctx),false,'batched visible→hidden must end hidden');
console.log('PASS: stage visibility follows the newest observer entry');
