export const UI_URI = 'ui://ruvector/console/v1.html';

export const consoleHtml = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:dark;font:14px/1.45 system-ui,sans-serif;background:#0b0f19;color:#eef1fa}
*{box-sizing:border-box}body{margin:0;padding:18px;max-width:900px}header{display:flex;align-items:center;gap:12px;margin-bottom:18px}
.mark{width:35px;height:35px;border-radius:11px;background:linear-gradient(135deg,#8a5dff,#16dbc2);box-shadow:0 0 24px #7860ff55}
h1{font-size:18px;margin:0}small{color:#9aa5be}h2{font-size:13px;text-transform:uppercase;letter-spacing:.11em;color:#aab5ca}
.badge{margin-left:auto;color:#53e0b5;border:1px solid #328c75;border-radius:20px;padding:4px 9px;font-size:11px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{background:#151c2c;border:1px solid #2b354a;border-radius:16px;padding:16px}
label{display:block;color:#b9c5d9;font-size:12px;margin:11px 0 5px}select,input,textarea{width:100%;background:#0d1422;border:1px solid #39445b;border-radius:9px;padding:10px;color:#f0f4ff;font:inherit}
textarea{min-height:90px;resize:vertical}button{background:#7c5df0;color:white;border:0;border-radius:9px;padding:10px 14px;font:inherit;font-weight:650;cursor:pointer;margin-top:12px}
button:disabled{opacity:.55;cursor:wait}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0d1422;border-radius:9px;padding:12px;min-height:54px;margin:10px 0 0;font-size:12px}
ul{list-style:none;margin:0;padding:0}li{display:flex;justify-content:space-between;border-top:1px solid #2b354a;padding:10px 0;gap:12px}
.on{color:#53e0b5}.off{color:#e7b875}.wide{grid-column:1/-1}@media(max-width:640px){body{padding:12px}.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style></head><body>
<header><div class="mark" aria-hidden="true"></div><div><h1>RuVector Console</h1><small>Tenant scoped decisions and vector search</small></div><span class="badge">MCP + WASM</span></header>
<div class="grid">
<section class="panel"><h2>Context</h2><label for="tenant">Tenant</label><select id="tenant"></select><p id="role"><small>Loading memberships…</small></p><h2>Capability map</h2><ul id="capabilities"></ul></section>
<section class="panel"><h2>Exact vector search</h2><form id="search"><label for="collection">Collection</label><input id="collection" value="default" maxlength="64" required><label for="query">Query vector as JSON</label><textarea id="query" spellcheck="false">[1,0,0]</textarea><label for="k">Results</label><input id="k" type="number" value="5" min="1" max="20"><button id="submit" type="submit">Search</button></form></section>
<section class="panel wide"><h2>Result</h2><pre id="result" role="status">Connect to load your tenant context.</pre></section>
</div>
<script>
(()=>{const $=id=>document.getElementById(id);let seq=0;const pending=new Map();
function request(method,params){return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});parent.postMessage({jsonrpc:'2.0',id,method,params},'*')})}
window.addEventListener('message',e=>{if(e.source!==parent||e.data?.jsonrpc!=='2.0')return;const m=e.data;if(typeof m.id==='number'&&pending.has(m.id)){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}},{passive:true});
const ready=request('ui/initialize',{appInfo:{name:'ruvector-console',version:'0.1.0'},appCapabilities:{},protocolVersion:'2026-01-26'}).then(()=>parent.postMessage({jsonrpc:'2.0',method:'ui/notifications/initialized',params:{}},'*'));
async function call(name,args){await ready;const response=await request('tools/call',{name,arguments:args});if(response?.isError)throw Error(response.content?.[0]?.text||'Tool failed');return response.structuredContent||response}
function output(value){$('result').textContent=JSON.stringify(value,null,2)}
ready.then(async()=>{const context=await call('list_tenants',{});const tenants=context.tenants||[];const select=$('tenant');select.replaceChildren();for(const t of tenants){const option=document.createElement('option');option.value=t.tenant_id;option.textContent=t.tenant_id;select.append(option)}
$('role').textContent=tenants.length?('Role: '+tenants[0].role):'No tenant membership';select.addEventListener('change',()=>{$('role').textContent='Role: '+(tenants.find(t=>t.tenant_id===select.value)?.role||'none')});
const catalog=await call('capability_catalog',{});const list=$('capabilities');for(const item of catalog.capabilities||[]){const li=document.createElement('li');const name=document.createElement('span');name.textContent=item.name;const status=document.createElement('span');status.className=item.available?'on':'off';status.textContent=item.available?'ready':'pending';li.append(name,status);list.append(li)}output({tenants:tenants.length,ready:catalog.capabilities?.filter(c=>c.available).map(c=>c.name)})
}).catch(e=>output({error:String(e)}));
$('search').addEventListener('submit',async e=>{e.preventDefault();const button=$('submit');button.disabled=true;try{const result=await call('search_vectors',{tenant_id:$('tenant').value,collection:$('collection').value,query:JSON.parse($('query').value),k:Number($('k').value)});output(result)}catch(error){output({error:String(error)})}finally{button.disabled=false}});
})();
</script></body></html>`;
