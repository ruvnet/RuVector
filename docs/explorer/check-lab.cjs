const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map(),buttons=[];const el=id=>{if(!elements.has(id))elements.set(id,{dataset:{},parentElement:{},firstChild:{},classList:{toggle(){}},textContent:'',innerHTML:'',onclick:null,oninput:null,onchange:null});return elements.get(id)};
const views=['canopy','space','hyper','trace'].map(v=>({dataset:{rvclView:v},setAttribute(k,x){this[k]=x}}));const root=el('root');root.querySelectorAll=selector=>selector.includes('data-rvcl-view')?views:buttons;root.addEventListener=(type,f)=>root[type]=f;
const modes=['contrast','retrieval','attention','hyperbolic','graph','mincut','hybrid','memory','sona','temporal','compact','compress','route','access','rvf','federation'];
for(const mode of modes)buttons.push({dataset:{rvclMode:mode},classList:{contains:()=>false},setAttribute(k,v){this[k]=v}});
const ctx={document:{getElementById:id=>id==='rv-capability-lab'?root:el(id),hidden:false},matchMedia:()=>({matches:false,addEventListener(){}}),IntersectionObserver:class{observe(){}},requestAnimationFrame(){},console,Math,Set};vm.createContext(ctx);
const math=fs.readFileSync(__dirname+'/capability-math.js','utf8').replace(/export /g,'');const labs=fs.readFileSync(__dirname+'/experience-labs.js','utf8').replace(/^import .*\n/gm,'').replace(/export /g,'');const source=fs.readFileSync(__dirname+'/capability-lab.js','utf8').replace(/^import .*\n/gm,'');vm.runInContext(math+'\nconst {evaluateLab,labScene,labMetricHTML,labChart,focusedModes}=(()=>{'+labs+';return {evaluateLab,labScene,labMetricHTML,labChart,focusedModes};})();\n'+source,ctx);
assert(el('rvcl-geometry').innerHTML.includes('<circle'));assert(el('rvcl-metrics').innerHTML.includes('360'));
for(const b of buttons){root.click({target:{closest:s=>s==='[data-rvcl-mode]'?b:null}});assert.equal(b['aria-pressed'],'true');assert(el('rvcl-disclosure').textContent.length>20);assert(el('rvcl-geometry').innerHTML.length>2000)}
el('rvcl-query').onclick();root.click({target:{closest:s=>s==='[data-rvcl-mode]'?buttons[7]:null}});assert(el('rvcl-metrics').innerHTML.includes('<strong>1</strong>'));
root.click({target:{closest:s=>s==='[data-rvcl-mode]'?buttons[11]:null}});el('rvcl-bits').onchange({target:{value:'2'}});assert(el('rvcl-metrics').innerHTML.includes('2 bits'));assert(el('rvcl-geometry').innerHTML.includes('QUANTIZED'));
root.click({target:{closest:s=>s==='[data-rvcl-mode]'?buttons[5]:null}});assert(el('rvcl-metrics').innerHTML.includes('<strong>3</strong>'));
root.click({target:{closest:s=>s==='[data-rvcl-mode]'?buttons[10]:null}});el('rvcl-policy').onchange({target:{value:'lfu'}});assert(el('rvcl-metrics').innerHTML.includes('LFU'));
el('rvcl-play').onclick();assert.equal(el('rvcl-play').textContent,'Resume motion');el('rvcl-tour').onclick();assert.equal(el('rvcl-tour').textContent,'Stop cinematic tour');assert.equal(el('rvcl-play').textContent,'Pause motion');
for(const v of views){root.click({target:{closest:s=>s==='[data-rvcl-view]'?v:null}});assert.equal(v['aria-pressed'],'true');assert(el('rvcl-geometry').innerHTML.length>2000)}
el('rvcl-replay').oninput({target:{value:'35'}});assert(el('rvcl-geometry').innerHTML.length>2000);
console.log('PASS: 16 SVG scenes, source boundaries, exact query metrics, memory count, quantization, cut, compaction, pause and tour');
