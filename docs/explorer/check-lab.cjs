const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map(),buttons=[];const el=id=>{if(!elements.has(id))elements.set(id,{dataset:{},classList:{toggle(){}},textContent:'',innerHTML:'',onclick:null,oninput:null,onchange:null});return elements.get(id)};
const root=el('root');root.querySelectorAll=()=>buttons;root.addEventListener=(type,f)=>root[type]=f;
const modes=['contrast','retrieval','attention','hyperbolic','graph','mincut','hybrid','memory','sona','temporal','compact','compress','route','access','rvf','federation'];
for(const mode of modes)buttons.push({dataset:{rvclMode:mode},classList:{contains:()=>false},setAttribute(k,v){this[k]=v}});
const ctx={document:{getElementById:id=>id==='rv-capability-lab'?root:el(id),hidden:false},matchMedia:()=>({matches:false,addEventListener(){}}),IntersectionObserver:class{observe(){}},requestAnimationFrame(){},console,Math,Set};vm.createContext(ctx);
const math=fs.readFileSync(__dirname+'/capability-math.js','utf8').replace(/export /g,'');const source=fs.readFileSync(__dirname+'/capability-lab.js','utf8').replace(/^import .*\n/,'');vm.runInContext(math+'\n'+source,ctx);
assert(el('rvcl-geometry').innerHTML.includes('<circle'));assert(el('rvcl-metrics').innerHTML.includes('360'));
for(const b of buttons){root.click({target:{closest:()=>b}});assert.equal(b['aria-pressed'],'true');assert(el('rvcl-disclosure').textContent.length>20);assert(el('rvcl-geometry').innerHTML.length>2000)}
el('rvcl-query').onclick();root.click({target:{closest:()=>buttons[7]}});assert(el('rvcl-metrics').innerHTML.includes('<strong>1</strong>'));
root.click({target:{closest:()=>buttons[11]}});el('rvcl-bits').onchange({target:{value:'2'}});assert(el('rvcl-metrics').innerHTML.includes('6 bits'));
root.click({target:{closest:()=>buttons[5]}});assert(el('rvcl-metrics').innerHTML.includes('<strong>3</strong>'));
root.click({target:{closest:()=>buttons[10]}});el('rvcl-policy').onchange({target:{value:'lfu'}});assert(el('rvcl-metrics').innerHTML.includes('LFU'));
el('rvcl-play').onclick();assert.equal(el('rvcl-play').textContent,'Resume motion');el('rvcl-tour').onclick();assert.equal(el('rvcl-tour').textContent,'Stop cinematic tour');assert.equal(el('rvcl-play').textContent,'Pause motion');
console.log('PASS: 16 SVG scenes, source boundaries, exact query metrics, memory count, quantization, cut, compaction, pause and tour');
