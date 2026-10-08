import {evaluateLab,labScene,labMetricHTML,focusedModes,labNames,labPoints} from './experience-labs.js?v=20261008';

const $=id=>document.getElementById(id);
const navigate=id=>window.dispatchEvent(new CustomEvent('ruvector:navigate',{detail:id}));
const motion=()=>matchMedia('(prefers-reduced-motion:reduce)').matches?'auto':'smooth';
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent=n=>n==null?'Waiting':(n*100).toFixed(1)+'%';
const number=n=>n==null?'Waiting':Math.round(n).toLocaleString('en-US');
const snapshot=()=>window.RuVectorExplorer?.snapshot();
const scenarios=[
  {id:'search',name:'Semantic search',hint:'Explore measured WASM search on synthetic clustered vectors.'},
  {id:'memory',name:'Agent memory',hint:'Follow trajectory recall and inspect stored query memories.'},
  {id:'route',name:'Ticket routing',hint:'Explore a synthetic routing fixture and its confidence gate. No text encoder runs here.'},
  {id:'compress',name:'Compression',hint:'Measure scalar rounding, coordinate payload and top 10 overlap.'}
];
const launcher=document.createElement('section');launcher.className='rx-launcher';launcher.setAttribute('aria-label','Experiment launcher');
launcher.innerHTML=`<span class="lbl">RUVECTOR / EXPERIMENTS</span><h2>What would you like to explore?</h2><p>Start with a question. Follow the vectors, change one constraint, and inspect what actually happened.</p><form class="rx-launch-form"><input id="rx-intent" aria-label="Choose an experiment" placeholder="Try agent memory, ticket routing or compression" maxlength="180" autocomplete="off"><button type="submit">Explore ↗</button></form><div class="rx-presets">${scenarios.map(s=>`<button data-rx-scenario="${s.id}" aria-pressed="false">${s.name}</button>`).join('')}</div><p id="rx-launch-status" role="status">Four starting points. Computation stays in this browser.</p>`;
document.querySelector('.rx-topnav').after(launcher);
let launchBusy=false;
async function launch(id){
  if(launchBusy)return;
  const s=scenarios.find(s=>s.id===id);if(!s)return;
  $('rx-intent').value=s.name;
  if(id==='search'||id==='memory'){
    if(!window.RuVectorExplorer||snapshot()?.busy){$('rx-launch-status').textContent='The vector index is preparing. Try this preset when loading finishes.';return;}
    launchBusy=true;launcher.setAttribute('aria-busy','true');$('rx-launch-status').textContent='Preparing '+s.name.toLowerCase()+'…';
    try{navigate(id==='memory'?'learn':'search');const ok=await window.RuVectorExplorer.preset(id);$('rx-launch-status').textContent=ok?s.hint:'The engine is busy. Try again after the current operation.';if(!ok)return;}
    catch{ $('rx-launch-status').textContent='The experiment could not start. Use Search to inspect the engine status.';return; }
    finally{launchBusy=false;launcher.removeAttribute('aria-busy');}
  }else{
    navigate('capabilities');document.querySelector(`.rvcl-tabs [data-rvcl-mode="${id}"]`)?.click();$('rx-launch-status').textContent=s.hint;
  }
  launcher.querySelectorAll('[data-rx-scenario]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.rxScenario===id)));
  (id==='search'||id==='memory'?$('stage'):$('rvcl-svg')).scrollIntoView({behavior:motion(),block:'center'});
  refreshEvidence();
}
launcher.addEventListener('click',e=>{const b=e.target.closest('[data-rx-scenario]');if(b)launch(b.dataset.rxScenario);});
launcher.querySelector('form').addEventListener('submit',e=>{
  e.preventDefault();const q=$('rx-intent').value.trim().toLowerCase();
  const id=/memory|agent|trajectory|learn/.test(q)?'memory':/ticket|route|routing|decision/.test(q)?'route':/compress|quant|precision|bits/.test(q)?'compress':/search|semantic|vector|retriev/.test(q)?'search':null;
  if(id)launch(id);else $('rx-launch-status').textContent='Choose agent memory, semantic search, ticket routing or compression. This launcher selects local presets.';
});

const evidence=document.createElement('section');evidence.className='rx-evidence';evidence.setAttribute('aria-label','Learning evidence and memory inspector');
evidence.innerHTML=`<span class="lbl">LIVE ENGINE / EVIDENCE</span><h3>What changed, and why?</h3><div id="rx-live-metrics" class="rx-evidence-grid"></div><p id="rx-evidence-note">Waiting for the first query.</p><div class="rx-actions"><button id="rx-train">Run 25 learning queries</button><button id="rx-export-evidence">Export evidence JSON</button><button id="rx-retention">Explore retention ↗</button></div><details><summary>Inspect stored trajectory memories</summary><label class="lbl" for="rx-memory-choice">Choose a memory</label><select id="rx-memory-choice" aria-label="Stored trajectory memory"></select><div id="rx-memory-detail" class="rx-memory-detail"></div><p class="rx-snapshot-note">Current engine uses FIFO retention. A recalled memory supplies entry points; its individual causal benefit is not measured.</p></details>`;
$('rx-panel-learn').prepend(evidence);
const benchActions=document.createElement('div');benchActions.className='rx-evidence';
benchActions.innerHTML='<span class="lbl">REPRODUCIBLE EVIDENCE</span><h3>Keep the experiment with the result.</h3><p>Export the actual synthetic queries, configuration, package versions, recall and repeated pass timings. Timings describe this browser and device.</p><div class="rx-actions"><button id="rx-export-benchmark">Export benchmark JSON</button></div><p id="rx-benchmark-status" role="status">Run the benchmark to create an evidence record.</p>';
$('rx-panel-bench').append(benchActions);
let latest=null,selectedSlot=null,lastMemorySignature='';
function metric(label,value,note){return `<div><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`;}
function refreshEvidence(){
  const s=snapshot();if(!s)return;latest=s;const e=s.evidence,change=v=>e.staticEvals?percent(1-v/e.staticEvals):'Waiting';
  evidence.dataset.missed=String(e.learnedRecall!=null&&e.learnedRecall<s.target);
  $('rx-live-metrics').innerHTML=metric('LEARNED RECALL',percent(e.learnedRecall),'Target '+percent(s.target))+metric('STATIC RECALL',percent(e.staticRecall),'Same '+e.count+' queries')+metric('DISTANCE WORK SAVED',change(e.learnedEvals),'Versus configured static breadth')+metric('QUERIES IN WINDOW',e.count,'Current index and learning epoch');
  $('rx-evidence-note').textContent=e.count?`Controller only: ${change(e.controllerEvals)} work saved at ${percent(e.controllerRecall)} recall. Memory only: ${change(e.memoryEvals)} saved at ${percent(e.memoryRecall)} recall. ${e.learnedRecall<s.target?'The learned run is below target. ':'Recall and work must be assessed together. '}These contributions are separate trials and are not additive. This loop adjusts search breadth and recalls trajectories; SONA weights are not trained.`:'Run a query to collect evidence. Existing stored learning state is retained.';
  $('rx-train').disabled=s.busy;$('rx-export-evidence').disabled=!s.query;
  $('rx-export-benchmark').disabled=!s.benchmark;
  $('rx-benchmark-status').textContent=s.benchmark?`${s.benchmark.queries.length} queries · ${s.benchmark.rows.length} engines · median of repeated pass means. Ready to export.`:'Run the benchmark to create an evidence record.';
  const sig=s.memories.map(m=>`${m.slot}:${m.sequence}:${m.hits}:${m.recalled}`).join('|');
  if(sig!==lastMemorySignature){
    lastMemorySignature=sig;const select=$('rx-memory-choice');select.replaceChildren();
    s.memories.slice().reverse().forEach(m=>{const o=document.createElement('option');o.value=m.slot;o.textContent=`${m.sequence==null?'Stored slot '+m.slot:'Trajectory '+m.sequence}${m.recalled?' · recalled now':''} · ${m.hits} recalls`;select.append(o);});
    if(selectedSlot==null||!s.memories.some(m=>m.slot===selectedSlot))selectedSlot=s.memories.at(-1)?.slot??null;
    select.value=selectedSlot??'';select.disabled=!s.memories.length;
  }
  renderMemory();
}
function renderMemory(){
  const m=latest?.memories.find(m=>m.slot===selectedSlot),d=$('rx-memory-detail');
  if(!m){d.textContent='No trajectory memories stored yet.';return;}
  d.innerHTML=`<div><span>SOURCE</span>${escape(m.source)}</div><div><span>RETURNED ANSWERS</span>${m.answers.map(id=>'#'+id).join(', ')}</div><div><span>RECALL AT CREATION</span>${m.recall==null?'Unavailable for restored records':percent(m.recall)}</div><div><span>REUSED</span>${m.hits} times${m.recalled?' · selected for current query':''}</div><div><span>RETENTION</span>FIFO · position ${m.position} of ${latest.memories.length}</div><div><span>EVICTION ORDER</span>${m.position===1?'Oldest · next at capacity':'Evicted after '+(m.position-1)+' older entries'}</div>`;
}
$('rx-memory-choice').onchange=e=>{selectedSlot=Number(e.target.value);renderMemory();};
$('rx-train').onclick=()=>{if(window.RuVectorExplorer?.train())$('rx-evidence-note').textContent='Running 25 queries. The evidence updates when the batch completes.';};
$('rx-retention').onclick=()=>{navigate('capabilities');document.querySelector('.rvcl-tabs [data-rvcl-mode="compact"]').click();$('rvcl-svg').scrollIntoView({behavior:motion(),block:'center'});};
function download(data,name){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('rx-export-evidence').onclick=()=>{const s=snapshot();if(s?.query)download({...s,recordedAt:new Date().toISOString(),scope:'Synthetic browser learning demo; SONA not executed'},'ruvector-learning-evidence.json');};
$('rx-export-benchmark').onclick=()=>{const s=snapshot();if(s?.benchmark)download(s.benchmark,'ruvector-benchmark-evidence.json');};
window.addEventListener('ruvector:query',refreshEvidence);
setInterval(()=>{if(!document.hidden)refreshEvidence();},1500);
refreshEvidence();

// The guide uses the same fixture functions as the main exhibits, with its own state.
let guide=null;
const stepNames={search:['QUERY','DESCEND','EXPLORE','RERANK','RESULT'],learn:['RECALL','SEARCH','SCORE','STORE','ADAPT'],bench:['DATASET','WARM UP','REPEAT','RECALL','EXPORT'],capabilities:['INPUT','MECHANISM','OUTCOME','EVIDENCE','SOURCE']};
function flowScene(section,step){const names=stepNames[section]||stepNames.search;return names.map((name,i)=>{const x=160+i*218,col=i===step?'#ffad82':'#7ef0cf';return `${i?`<path d="M${x-153} 270H${x-66}" fill="none" stroke="#7ef0cf" stroke-opacity=".4"/>`:''}<circle cx="${x}" cy="270" r="54" fill="#0b1623" stroke="${col}"/><circle class="rx-orbit" cx="${x}" cy="270" r="66" fill="none" stroke="${col}" stroke-dasharray="4 12" opacity=".4"/><text x="${x}" y="276" text-anchor="middle" fill="${col}" font-family="monospace" font-size="18">0${i+1}</text><text x="${x}" y="371" text-anchor="middle" fill="${col}" font-family="monospace" font-size="16">${name}</text>`;}).join('');}
window.addEventListener('ruvector:guide',e=>{
  guide={...e.detail,tab:'watch',step:0,bits:4,strength:.7,policy:'lru',query:[.2,.1,.3],selected:null};
  if(guide.section!=='capabilities')guide.mode=null;
  renderGuide();
});
function guideResult(){return focusedModes.includes(guide.mode)?evaluateLab(guide.mode,guide):null;}
function renderGuide(){
  const g=guide;if(!g)return;const r=guideResult(),content=$('rx-dialog-content');
  content.innerHTML=`<span class="lbl">${r?'INTERACTIVE / LOCAL COMPUTATION':'EXPLORER / GUIDED EXPERIMENT'}</span><h2 id="rx-dialog-title">${escape(r?labNames[g.mode]:g.title)}</h2><div class="rx-guide-tabs" role="tablist" aria-label="Experiment guide">${['watch','change','measure','build'].map(t=>`<button role="tab" id="rx-guide-${t}" aria-controls="rx-guide-panel" aria-selected="${g.tab===t}" tabindex="${g.tab===t?0:-1}" data-guide-tab="${t}">${t[0].toUpperCase()+t.slice(1)}</button>`).join('')}</div><section id="rx-guide-panel" role="tabpanel" aria-labelledby="rx-guide-${g.tab}"></section>`;
  renderGuidePanel();
}
function renderGuidePanel(){
  const g=guide,r=guideResult(),p=$('rx-guide-panel');
  if(g.tab==='watch'||g.tab==='change'){
    const controls=r?`<div class="rx-guide-controls">${g.mode==='compress'?`<label>Precision <select data-guide-bits><option value="8" ${g.bits===8?'selected':''}>8 bits</option><option value="4" ${g.bits===4?'selected':''}>4 bits</option><option value="2" ${g.bits===2?'selected':''}>2 bits</option></select></label>`:`<label>${g.mode==='compact'?'Memory budget':'Confidence gate'} <input data-guide-strength type="range" min="1" max="100" value="${Math.round(g.strength*100)}" aria-label="${g.mode==='compact'?'Memory budget':'Confidence gate'}"></label>`}${g.mode==='compact'?`<label>Retention <select data-guide-policy><option value="lru" ${g.policy==='lru'?'selected':''}>LRU</option><option value="lfu" ${g.policy==='lfu'?'selected':''}>LFU</option></select></label>`:''}<button data-guide-query>New query</button></div>`:`<div class="rx-guide-controls"><button data-guide-step>Next stage</button><button data-guide-run>${g.section==='bench'?'Run benchmark':'Run a query'}</button></div>`;
    p.innerHTML=`<svg class="rx-guide-scene" viewBox="0 0 1200 580" role="img" aria-label="${escape(r?labNames[g.mode]:'Experiment stages')}">${r?labScene(r,{selected:g.selected}):flowScene(g.section,g.step)}</svg>${controls}<div class="rx-guide-copy"><p>${escape(r?descriptions[g.mode]:g.body)}</p><p>${escape(r?'Adjust a control or select a vector. Measurements update from the same fixture.':g.items.join(' '))}</p></div><div class="rx-guide-measure" style="margin-top:16px">${r?labMetricHTML(r):engineMetrics(g.section)}</div>`;
  }else if(g.tab==='measure'){
    p.innerHTML=`<div class="rx-guide-measure">${r?labMetricHTML(r):engineMetrics(g.section)}</div><p class="rx-guide-boundary">${escape(r?'Measured with a deterministic 360 vector JavaScript fixture. This is not a native RuVector crate benchmark.':g.section==='capabilities'?'This exhibit is an architecture illustration. No executed result is available.':'Snapshot of the current browser run. JS trace, WASM search and learning trials are distinct computations.')}</p><p>${r?'Change one parameter, keep the query fixed, and compare the result.':escape(g.body)}</p><div class="rx-actions"><button data-guide-refresh>Refresh measurements</button><button data-guide-export>Export this evidence</button></div>`;
  }else{
    const source=g.source||(g.section==='capabilities'?$('rvcl-source')?.href:'https://github.com/ruvnet/RuVector');
    const config=r?{mode:g.mode,seed:731,count:360,bits:g.bits,policy:g.policy,strength:g.strength,query:g.query}:{dataset:latest?.dataset,seed:latest?.seed,index:latest?.index,vectors:latest?.vectors,target:latest?.target};
    p.innerHTML=`<p>Use this configuration to reproduce the exhibit. Connect the production implementation appropriate to your workload.</p><pre class="rx-guide-build">${escape(JSON.stringify(config,null,2))}</pre><a href="${escape(source)}" target="_blank" rel="noopener">Open source implementation ↗</a><p class="rx-guide-boundary">${escape(r?'Fixture arithmetic runs locally. A production integration needs its own dataset, native adapter and validation.':g.section==='capabilities'?$('rvcl-disclosure')?.textContent||'Architecture illustration.':g.section==='learn'?'The browser loop stores query trajectories and adjusts breadth. It does not train SONA.':'The WASM engine uses synthetic vectors and a scalar build. Browser measurements do not establish production performance.')}</p>`;
  }
}
const descriptions={compress:'Keep the query fixed while changing precision. The two fields compare original and rounded coordinates; top 10 overlap exposes lost answers.',compact:'Capacity is a decision. Bright cells survive; dim cells are eviction candidates. Select a cell to inspect why it survives or leaves.',route:'Raise the gate to demand stronger evidence. Grey examples abstain. Coverage measures how often the fixture routes; it does not measure accuracy.'};
function engineMetrics(section){
  const s=snapshot();if(!s||section==='capabilities')return metric('EVIDENCE TYPE','ILLUSTRATION','No engine measurement in this exhibit');
  if(section==='learn')return metric('LEARNED RECALL',percent(s.evidence.learnedRecall),'Target '+percent(s.target))+metric('STATIC RECALL',percent(s.evidence.staticRecall),'Same query window')+metric('LEARNED EVALS',number(s.evidence.learnedEvals),'Mean distance calculations')+metric('STORED MEMORIES',s.memories.length,'FIFO trajectory memory');
  if(section==='bench')return metric('QUERIES',s.benchmark?.queries.length??'Not run','Identical query set across engines')+metric('ENGINES',s.benchmark?.rows.length??'Not run','Measured in this browser')+metric('TIMING','PASS MEDIAN','Median of repeated pass means')+metric('DATASET',s.dataset,s.vectors+' synthetic vectors');
  return metric('WASM RECALL',percent(s.query?.wasmRecall),'Exact baseline comparison')+metric('JS TRACE RECALL',percent(s.query?.recall),'Separate trace engine')+metric('DISTANCE EVALS',number(s.query?.evals),'JS trace only')+metric('INDEX',s.index.toUpperCase(),s.vectors+' synthetic vectors');
}
$('rx-detail-dialog').addEventListener('click',e=>{
  if(!guide)return;const tab=e.target.closest('[data-guide-tab]');
  if(tab){guide.tab=tab.dataset.guideTab;renderGuide();$('rx-guide-'+guide.tab).focus();return;}
  if(e.target.closest('[data-guide-step]')){guide.step=(guide.step+1)%5;renderGuidePanel();}
  if(e.target.closest('[data-guide-run]')){(guide.section==='bench'?$('benchBtn'):$('runBtn')).click();renderGuidePanel();}
  if(e.target.closest('[data-guide-refresh]'))renderGuidePanel();
  if(e.target.closest('[data-guide-query]')){guide.selected=((guide.selected??18)+71)%360;guide.query=labPoints[guide.selected].v.slice();renderGuidePanel();}
  const point=e.target.closest('[data-rvcl-point]');if(point){guide.selected=Number(point.dataset.rvclPoint);guide.query=labPoints[guide.selected].v.slice();renderGuidePanel();}
  if(e.target.closest('[data-guide-export]')){const r=guideResult();download(r?{schema:1,seed:731,kind:'synthetic local JavaScript fixture',mode:guide.mode,bits:guide.bits,policy:guide.policy,strength:guide.strength,query:guide.query,metrics:r.metrics}:snapshot(),'ruvector-experiment.json');}
});
$('rx-detail-dialog').addEventListener('change',e=>{if(!guide)return;if(e.target.matches('[data-guide-bits]'))guide.bits=Number(e.target.value);else if(e.target.matches('[data-guide-policy]'))guide.policy=e.target.value;else return;renderGuidePanel();});
$('rx-detail-dialog').addEventListener('input',e=>{
  if(!guide||!e.target.matches('[data-guide-strength]'))return;guide.strength=Number(e.target.value)/100;
  const r=guideResult();$('rx-guide-panel').querySelector('svg').innerHTML=labScene(r,{selected:guide.selected});$('rx-guide-panel').querySelector('.rx-guide-measure').innerHTML=labMetricHTML(r);
});
$('rx-detail-dialog').addEventListener('keydown',e=>{
  const b=e.target.closest('[data-guide-tab]');if(!b)return;
  const keys=['ArrowLeft','ArrowRight','Home','End'];if(!keys.includes(e.key))return;e.preventDefault();e.stopPropagation();
  const tabs=['watch','change','measure','build'],i=tabs.indexOf(guide.tab);guide.tab=tabs[e.key==='Home'?0:e.key==='End'?3:(i+(e.key==='ArrowRight'?1:3))%4];renderGuide();$('rx-guide-'+guide.tab).focus();
});
