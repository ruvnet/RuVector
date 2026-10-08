import {makeDataset,nearest,compact,typedDecision} from './capability-math.js?v=20261008';

export const focusedModes=['compact','compress','route'];
export const labNames={compact:'Memory retention',compress:'Precision laboratory',route:'Decision boundaries'};
export const labPoints=makeDataset();
const colors=['#7ef0cf','#7ba5ff','#ff966a','#c899ff'];
const pct=n=>(n*100).toFixed(0)+'%';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,Number(x)||a));

// Every displayed result is calculated from a deterministic, public 3D fixture.
// This module deliberately has no network, storage, model or native-crate dependency.
export function evaluateLab(mode,{points=labPoints,query=[.2,.1,.3],bits=8,strength=.7,policy='lru'}={}){
  bits=[2,4,8].includes(Number(bits))?Number(bits):8;
  strength=clamp(strength,.01,1);policy=policy==='lfu'?'lfu':'lru';
  const truth=nearest(points,query),truthIds=new Set(truth.map(p=>p.id));
  const overlap=rows=>rows.filter(p=>truthIds.has(p.id)).length/(truth.length||1);
  if(mode==='compress'){
    const levels=2**bits-1,rounded=points.map(p=>({...p,v:p.v.map(v=>Math.round((v+1)*levels/2)*2/levels-1)}));
    const ranked=nearest(rounded,query),mse=points.reduce((s,p,i)=>s+p.v.reduce((t,v,j)=>t+(v-rounded[i].v[j])**2,0),0)/(points.length*3);
    return {mode,bits,query,points,rounded,truth,ranked,recall:overlap(ranked),mse,bytes:Math.ceil(points.length*3*bits/8),fullBytes:points.length*3*4,
      metrics:[['TOP 10 OVERLAP',pct(overlap(ranked)),'Against exact original coordinates'],['COORDINATE PAYLOAD',Math.ceil(points.length*3*bits/8)+' B','Theoretical packed size; overhead excluded'],['RECONSTRUCTION ERROR',mse.toFixed(6),'Mean squared coordinate error'],['PRECISION',bits+' bits', (32/bits)+'× coordinate compression']]};
  }
  if(mode==='compact'){
    const capacity=Math.round(16+strength*(points.length-16)),kept=compact(points,capacity,policy),keptIds=new Set(kept.map(p=>p.id)),ranked=nearest(kept,query);
    return {mode,query,points,truth,ranked,kept,keptIds,capacity,policy,recall:overlap(ranked),
      metrics:[['ANSWERS RETAINED',pct(overlap(ranked)),'Original top 10 still retrievable'],['MEMORY BUDGET',capacity+' / '+points.length,'Adjust capacity to expose the tradeoff'],['EVICTED',points.length-capacity,'Preview only; no stored data deleted'],['RETENTION POLICY',policy.toUpperCase(),policy==='lru'?'Recent access wins; ID breaks ties':'Frequent access wins; ID breaks ties']]};
  }
  const threshold=.25+strength*.7,decision=typedDecision(query,.24,threshold);
  const classified=points.map(p=>({...p,decision:typedDecision(p.v,.24,threshold)}));
  const accepted=classified.filter(p=>!p.decision.abstain).length;
  return {mode:'route',points,query,threshold,decision,classified,
    metrics:[['CURRENT DECISION',decision.abstain?'ABSTAIN':'ROUTE '+(decision.index+1),'Synthetic centroid classifier'],['TOP SCORE',pct(decision.confidence),'Normalized similarity; not calibrated'],['CONFIDENCE GATE',pct(threshold),'Raise the gate to abstain more often'],['FIXTURE COVERAGE',pct(accepted/points.length),accepted+' of '+points.length+' points accepted']]};
}

const text=(x,y,s,color='#9badc3',size=12)=>`<text x="${x}" y="${y}" fill="${color}" font-size="${size}" font-family="ui-monospace,monospace">${s}</text>`;
const dot=(x,y,r,color,opacity=1,extra='')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" opacity="${opacity}" ${extra}/>`;
const line=(x1,y1,x2,y2,color,opacity=.35)=>`<path d="M${x1} ${y1}L${x2} ${y2}" stroke="${color}" opacity="${opacity}" fill="none"/>`;
const cloudPos=(p,cx)=>[cx+p.v[0]*160,285+p.v[1]*135];
const frame=(x,label)=>`<rect x="${x}" y="105" width="460" height="365" rx="12" fill="#080f19" stroke="#263950"/>${text(x+22,137,label,'#c4d4e8')}`;
const ring=(x,y,r,c)=>`<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="${c}" stroke-opacity=".35"/><circle class="rx-orbit" cx="${x}" cy="${y}" r="${r+9}" fill="none" stroke="${c}" stroke-dasharray="4 14" stroke-opacity=".3"/>`;

export function labScene(r,{selected=null}={}){
  let svg='';
  if(r.mode==='compress'){
    svg=frame(70,'ORIGINAL / FLOAT32')+frame(670,'QUANTIZED / '+r.bits+' BITS');
    const ids=new Set(r.ranked.map(p=>p.id)),truth=new Set(r.truth.map(p=>p.id));
    for(let i=0;i<r.points.length;i++){
      const p=r.points[i],a=cloudPos(p,300),b=cloudPos(r.rounded[i],900),hot=truth.has(p.id);
      svg+=dot(...a,hot?4:2,hot?'#fff':colors[p.c],hot?1:.45,`data-rvcl-point="${p.id}"`);
      svg+=dot(...b,ids.has(p.id)?4:2,ids.has(p.id)?'#fff':colors[p.c],ids.has(p.id)?1:.45,`data-rvcl-point="${p.id}"`);
      if(hot)svg+=line(...a,...b,ids.has(p.id)?'#7ef0cf':'#ff966a',.12);
    }
    for(const cx of [300,900]){const q=cloudPos({v:r.query},cx);svg+=ring(...q,13,'#ff966a');}
    svg+=text(556,264,r.bits+' BIT','#ff966a',15)+text(556,288,'→','#7ef0cf',28)+text(70,510,'White: top 10 results. Coral links: original answers lost after rounding.');
  }else if(r.mode==='compact'){
    svg+=text(70,100,'360 MEMORIES / SELECT A CELL TO INSPECT','#c4d4e8');
    const truth=new Set(r.truth.map(p=>p.id));
    r.points.forEach((p,i)=>{
      const x=70+(i%24)*27,y=125+Math.floor(i/24)*23,keep=r.keptIds.has(p.id),hot=truth.has(p.id);
      svg+=`<rect data-rvcl-point="${p.id}" x="${x}" y="${y}" width="19" height="15" rx="3" fill="${keep?colors[p.c]:'#243144'}" opacity="${keep?.85:.45}" stroke="${selected===p.id?'white':hot?'#ff966a':'transparent'}" stroke-width="2"/>`;
    });
    const p=r.points.find(p=>p.id===selected)||r.truth[0],keep=r.keptIds.has(p.id);
    svg+=`<rect x="790" y="125" width="335" height="323" rx="12" fill="#0b131f" stroke="#31455d"/>`;
    svg+=text(820,167,'MEMORY #'+p.id,'#e3efff',19)+text(820,205,keep?'RETAINED':'EVICTION CANDIDATE',keep?'#7ef0cf':'#ff966a',14);
    svg+=text(820,246,'Age: '+p.age+' hours')+text(820,276,'Access count: '+p.hits)+text(820,306,'Source: seeded fixture 731')+text(820,346,r.policy==='lru'?'Reason: rank by age':'Reason: rank by access count')+text(820,386,'Capacity: '+r.capacity+' entries');
    svg+=text(70,510,'Bright: retained. Dim: evicted. Coral outline: an original top 10 answer.');
  }else{
    const cx=350,cy=280;
    for(let i=0;i<4;i++){const a=i*Math.PI/2,x=cx+Math.cos(a)*168,y=cy+Math.sin(a)*142;svg+=ring(x,y,62,colors[i])+text(x-29,y-70,'ROUTE '+(i+1),colors[i]);}
    r.classified.forEach(p=>{const [x,y]=cloudPos(p,cx);svg+=dot(x,y,p.decision.abstain?2:3,p.decision.abstain?'#5a6679':colors[p.decision.index],p.decision.abstain?.35:.8,`data-rvcl-point="${p.id}"`);});
    const q=cloudPos({v:r.query},cx);svg+=ring(...q,16,'#fff')+dot(...q,4,'#fff');
    svg+=text(730,136,'ROUTE SCORES','#c4d4e8');
    r.decision.scores.forEach((v,i)=>{const y=185+i*60;svg+=text(730,y,'R'+(i+1),colors[i])+`<rect x="785" y="${y-14}" width="${v*290}" height="18" rx="3" fill="${colors[i]}"/>`+text(1090,y,pct(v),colors[i]);});
    svg+=line(785+r.threshold*290,154,785+r.threshold*290,400,'#ff966a',.9)+text(730,440,r.decision.abstain?'ABSTAIN / REQUIRE MORE EVIDENCE':'ACCEPT / ROUTE '+(r.decision.index+1),r.decision.abstain?'#ff966a':'#7ef0cf',17);
    svg+=text(70,510,'Colored: accepted. Grey: abstained. Click a vector to query its decision.');
  }
  return svg;
}

export function labMetricHTML(r){return r.metrics.map(([label,value,note])=>`<div><span>${label}</span><strong>${value}</strong><small>${note}</small></div>`).join('');}

export function labChart(r){
  if(r.mode==='route')return r.decision.scores.map((v,i)=>text(8,30+i*38,'R'+(i+1))+`<rect x="50" y="${16+i*38}" width="${v*340}" height="18" fill="${colors[i]}"/>`+text(410,30+i*38,pct(v))).join('')+line(50+r.threshold*340,10,50+r.threshold*340,170,'#ff966a',1);
  const ids=new Set(r.ranked.map(p=>p.id));
  return r.truth.map((p,i)=>{const x=30+(i%5)*93,y=45+Math.floor(i/5)*70;return dot(x,y,10,ids.has(p.id)?'#7ef0cf':'#ff966a')+text(x-10,y+30,'#'+p.id);}).join('');
}
