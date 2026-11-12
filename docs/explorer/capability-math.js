export function makeDataset(count=360,seed=731){const rnd=()=>((seed=(seed*1664525+1013904223)>>>0)/4294967296);return Array.from({length:count},(_,id)=>({id,c:id%4,v:[rnd()*2-1,rnd()*2-1,rnd()*2-1],age:id%72,hits:1+id%17,mask:1<<(id%4),lexical:rnd()}));}
export const distance=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
export function nearest(points,q,k=10){return points.map(p=>({...p,d:distance(p.v,q)})).sort((a,b)=>a.d-b.d||a.id-b.id).slice(0,k);}
export function softmax(values,temperature=1){const t=Math.max(.01,temperature),m=Math.max(...values),e=values.map(v=>Math.exp((v-m)/t)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);}
export function quantizationMSE(points,bits){const levels=2**bits-1;return points.reduce((s,p)=>s+p.v.reduce((a,v)=>a+(v-(Math.round((v+1)*levels/2)*2/levels-1))**2,0),0)/(points.length*3);}
export function poincareDistance(a,b){const aa=a.reduce((s,x)=>s+x*x,0),bb=b.reduce((s,x)=>s+x*x,0);if(aa>=1||bb>=1)throw new RangeError('Points must be inside the unit ball');return Math.acosh(1+2*distance(a,b)**2/((1-aa)*(1-bb)));}
export const decay=(age,halfLife)=>2**(-age/Math.max(.01,halfLife));
export function hybridRank(points,q,alpha=.5){return points.map(p=>({...p,score:alpha/(1+distance(p.v,q))+(1-alpha)*p.lexical})).sort((a,b)=>b.score-a.score||a.id-b.id);}
export function permitted(points,mask){return points.filter(p=>(p.mask&mask)===p.mask);}
export function compact(points,capacity,policy='lru'){return [...points].sort((a,b)=>policy==='lfu'?b.hits-a.hits||a.id-b.id:a.age-b.age||a.id-b.id).slice(0,capacity);}
export const cutEdges=[[0,1,4],[0,2,5],[1,2,3],[1,3,4],[2,3,5],[3,4,1],[2,5,2],[4,5,5],[4,6,4],[5,6,3],[5,7,5],[6,7,4]];
export function exactMinCut(edges=cutEdges,n=8){let best={weight:Infinity,mask:0,edges:[]};for(let mask=1;mask<(1<<n)-1;mask++){if(!(mask&1))continue;const crossed=edges.filter(([a,b])=>Boolean(mask&(1<<a))!==Boolean(mask&(1<<b))),weight=crossed.reduce((s,e)=>s+e[2],0);if(weight<best.weight)best={weight,mask,edges:crossed};}return best;}
export function typedDecision(q,temperature=.4,threshold=.6){const centroids=[[.8,0,0],[0,.8,0],[-.8,0,0],[0,-.8,0]],scores=softmax(centroids.map(c=>-distance(q,c)),temperature),confidence=Math.max(...scores),index=scores.indexOf(confidence);return{scores,confidence,index,abstain:confidence<threshold};}
