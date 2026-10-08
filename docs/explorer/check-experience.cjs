// DOM interaction regression check. Run with jsdom 26 on NODE_PATH.
const fs=require('node:fs'),assert=require('node:assert/strict');
const {JSDOM}=require('jsdom');
const page=fs.readFileSync(__dirname+'/index.html','utf8');
const dom=new JSDOM(page,{url:'https://ruvnet.github.io/RuVector/explorer/',runScripts:'outside-only',pretendToBeVisual:true});
const w=dom.window,d=w.document,read=n=>fs.readFileSync(__dirname+'/'+n,'utf8'),strip=s=>s.replace(/^import .*\n/gm,'').replace(/export /g,'');
w.matchMedia=()=>({matches:false,addEventListener(){}});w.IntersectionObserver=class{observe(){}};
w.requestAnimationFrame=()=>0;w.setInterval=()=>0;w.scrollTo=()=>{};w.Element.prototype.scrollIntoView=function(){};
w.HTMLDialogElement.prototype.showModal=function(){this.open=true};w.HTMLDialogElement.prototype.close=function(){this.open=false};
const errors=[];w.addEventListener('error',e=>errors.push(e.error));
const state={schema:1,busy:false,dataset:'clusters',seed:214,vectors:3000,dimensions:32,index:'hnsw',target:.95,run:1,query:{recall:1,wasmRecall:1,evals:120},evidence:{count:20,learnedRecall:.96,staticRecall:1,staticEvals:500,learnedEvals:120,memoryEvals:490,memoryRecall:1,controllerEvals:130,controllerRecall:.94},memories:[{slot:0,sequence:1,hits:2,answers:[7,9],recall:.9,recalled:true,position:1,retention:'FIFO',source:'Synthetic clusters dataset'}],benchmark:null};
let launched=null,trained=false;
w.RuVectorExplorer={snapshot:()=>structuredClone(state),preset:async id=>{launched=id;return true},train:()=>{trained=true;return true}};
w.eval(strip(read('capability-math.js'))+'\nwindow.testMath={makeDataset,distance,nearest,softmax,quantizationMSE,poincareDistance,decay,hybridRank,permitted,compact,exactMinCut,cutEdges,typedDecision};');
w.eval('(()=>{const {makeDataset,nearest,compact,typedDecision}=window.testMath;'+strip(read('experience-labs.js'))+';window.testLabs={evaluateLab,labScene,labMetricHTML,labChart,focusedModes,labNames,labPoints};})()');
w.eval('(()=>{const {makeDataset,distance,nearest,softmax,quantizationMSE,poincareDistance,decay,hybridRank,permitted,compact,exactMinCut,cutEdges,typedDecision}=window.testMath;const {evaluateLab,labScene,labMetricHTML,labChart,focusedModes}=window.testLabs;'+strip(read('capability-lab.js'))+'})()');
w.eval('(()=>{'+read('explorer-shell.js')+'})()');
w.eval('(()=>{const {evaluateLab,labScene,labMetricHTML,focusedModes,labNames,labPoints}=window.testLabs;'+strip(read('experience.js'))+'})()');
const click=s=>{const el=d.querySelector(s);assert(el,'Missing '+s);el.click();};
const change=(s,value,type='change')=>{const el=d.querySelector(s);el.value=value;el.dispatchEvent(new w.Event(type,{bubbles:true}));};
async function run(){
  assert(d.querySelector('.rx-launcher'));assert(d.querySelector('#rx-live-metrics').textContent.includes('96.0%'));
  click('[data-rx-scenario="memory"]');await new Promise(setImmediate);assert.equal(launched,'memory');assert.equal(d.body.dataset.rxSection,'learn');
  click('#rx-train');assert(trained);assert(d.querySelector('#rx-memory-detail').textContent.includes('#7'));
  click('[data-rx-scenario="compress"]');assert.equal(d.body.dataset.rxSection,'capabilities');assert(d.querySelector('#rvcl-svg').textContent.includes('QUANTIZED'));
  change('#rvcl-bits','2');assert(d.querySelector('#rvcl-metrics').textContent.includes('270 B'));
  click('.rvcl-card[data-rvcl-mode="compress"]');assert(d.querySelector('#rx-detail-dialog').open);assert.equal(d.querySelector('#rx-guide-watch').getAttribute('aria-selected'),'true');
  click('[data-guide-tab="change"]');change('[data-guide-bits]','8');assert(d.querySelector('.rx-guide-measure').textContent.includes('1080 B'));
  click('[data-guide-tab="measure"]');assert(d.querySelector('#rx-guide-panel').textContent.includes('not a native RuVector'));
  d.querySelector('[data-guide-tab="measure"]').dispatchEvent(new w.KeyboardEvent('keydown',{key:'ArrowRight',bubbles:true}));assert.equal(d.querySelector('#rx-guide-build').getAttribute('aria-selected'),'true');
  click('[data-rx-close]');assert(!d.querySelector('#rx-detail-dialog').open);
  click('.rvcl-tabs [data-rvcl-mode="compact"]');change('#rvcl-strength','100','input');assert(d.querySelector('#rvcl-metrics').textContent.includes('360 / 360'));
  click('.rvcl-card[data-rvcl-mode="route"]');change('[data-guide-strength]','100','input');assert(d.querySelector('.rx-guide-measure').textContent.includes('95%'));
  click('[data-rx-close]');click('#rx-top-search');assert.equal(d.body.dataset.rxSection,'search');
  assert.equal(errors.length,0,errors.map(e=>e.stack).join('\n'));
  console.log('PASS: launcher routing, engine bridge calls, live memory inspection, three exhibits, modal controls, measurement state, keyboard tabs, close and preserved search');dom.window.close();
}
run().catch(e=>{console.error(e);dom.window.close();process.exitCode=1});
