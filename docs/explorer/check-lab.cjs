const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map(),buttons=[];const el=id=>{if(!elements.has(id))elements.set(id,{classList:{toggle(){}},textContent:'',innerHTML:'',onclick:null,oninput:null,onchange:null});return elements.get(id)};
const root=el('root');root.querySelectorAll=()=>buttons;root.addEventListener=(type,f)=>root[type]=f;
for(const mode of ['contrast','retrieval','graph','memory','compress','route'])buttons.push({dataset:{rvclMode:mode},setAttribute(k,v){this[k]=v}});
const ctx={document:{getElementById:id=>id==='rv-capability-lab'?root:el(id),hidden:false},matchMedia:()=>({matches:false,addEventListener(){}}),IntersectionObserver:class{observe(){}},requestAnimationFrame(){},console,Math,Set};vm.createContext(ctx);vm.runInContext(fs.readFileSync(__dirname+'/capability-lab.js','utf8'),ctx);
assert(el('rvcl-geometry').innerHTML.includes('<circle'));assert(el('rvcl-metrics').innerHTML.includes('360'));
for(const b of buttons){root.click({target:{closest:()=>b}});assert.equal(b['aria-pressed'],'true');assert(el('rvcl-disclosure').textContent.length>20)}
el('rvcl-query').onclick();root.click({target:{closest:()=>buttons[3]}});assert(el('rvcl-metrics').innerHTML.includes('<strong>1</strong>'));
root.click({target:{closest:()=>buttons[4]}});el('rvcl-bits').onchange({target:{value:'2'}});assert(el('rvcl-metrics').innerHTML.includes('6 bits'));
el('rvcl-play').onclick();assert.equal(el('rvcl-play').textContent,'Resume motion');el('rvcl-tour').onclick();assert.equal(el('rvcl-tour').textContent,'Stop guided tour');console.log('PASS: six modes, geometry, exact query metrics, memory count, quantization control, pause and tour');
