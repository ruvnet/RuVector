/** A standalone, offline report viewer. Registry strings are rendered only as text. */
export function reportHtml(report) {
  if (report?.schemaVersion !== 1 || !Array.isArray(report.packages) || !Array.isArray(report.graph?.edges) || !Array.isArray(report.graph?.runtimeCycles)) throw new TypeError('Invalid constellation report');
  const data = JSON.stringify(report).replace(/</g, '\\u003c').replace(/>/g, '\\u003e').replace(/&/g, '\\u0026').replace(/\u2028/g, '\\u2028').replace(/\u2029/g, '\\u2029');
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RuvNet npm constellation</title>
<style>
:root{font-family:system-ui,sans-serif;color:#172b3b;background:#f4f7fb;font-size:16px}*{box-sizing:border-box}body{margin:0}main{max-width:1120px;margin:auto;padding:40px 24px}header{margin-bottom:28px}h1{font-size:clamp(1.8rem,4vw,2.6rem);letter-spacing:-.04em;margin:8px 0}h2{font-size:1.2rem}p{line-height:1.6}.eyebrow{color:#23634e;font-size:.8rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase}.muted{color:#526475}.scope{background:#e8f1fc;border-left:4px solid #3976b9;padding:14px 18px}.stats{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}.stat{background:white;border:1px solid #dce4ee;border-radius:12px;min-width:150px;padding:16px;flex:1}.stat strong{display:block;font-size:1.7rem}.panel{background:white;border:1px solid #dce4ee;border-radius:14px;padding:24px;margin:20px 0;overflow:hidden}.controls{display:grid;grid-template-columns:1fr 1fr;gap:18px}label{font-weight:650;display:block}input,select{display:block;margin-top:8px;width:100%;padding:12px;font:inherit;border:1px solid #9aabba;border-radius:7px;background:white;color:inherit}button{font:inherit;color:#175f9c;background:none;border:0;text-decoration:underline;cursor:pointer;padding:5px 0;text-align:left;overflow-wrap:anywhere}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #167d65;outline-offset:3px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;text-align:left}caption{text-align:left;font-weight:700;padding:12px 0}th,td{padding:12px 10px;border-bottom:1px solid #e6edf3;vertical-align:top;overflow-wrap:anywhere}th{font-size:.85rem;color:#526475}code{font-size:.88rem;overflow-wrap:anywhere}.badge{display:inline-block;background:#edf4f8;border-radius:5px;padding:3px 7px;font-size:.8rem}.selected{overflow-wrap:anywhere}.empty{color:#526475;padding:16px 0}footer{font-size:.85rem;color:#526475;margin-top:30px}#description{white-space:pre-wrap}#status{min-height:1.5em}@media(max-width:640px){main{padding:24px 14px}.controls{grid-template-columns:1fr}.panel{padding:16px}}
</style></head><body><main>
<header><div class="eyebrow">RuvNet ecosystem / observed dependencies</div><h1>Npm constellation</h1><p class="muted" id="observed"></p></header>
<div class="scope">This snapshot covers public maintainer search and latest package manifests. Advisory checks cover direct package versions only. Private packages, transitive dependencies, source security, tarball integrity verification and publish permissions are not certified.</div>
<section class="stats" aria-label="Snapshot totals"><div class="stat"><strong id="package-total"></strong>Packages</div><div class="stat"><strong id="edge-total"></strong>Typed edges</div><div class="stat"><strong id="cycle-total"></strong>Runtime cycle groups</div><div class="stat"><strong id="scan-status"></strong>Scan status</div></section>
<section class="panel" aria-labelledby="explore-heading"><h2 id="explore-heading">Explore a package</h2><p class="muted">Search, select a package, then follow its direct neighbors. Dependency types remain visible; optional and peer edges are separate.</p><div class="controls"><label for="search">Search packages<input type="search" id="search" placeholder="Package name or description" autocomplete="off"></label><label for="packages">Selected package<select id="packages"></select></label></div><p id="status" role="status" aria-live="polite"></p></section>
<section class="panel" aria-labelledby="selected-name"><h2 id="selected-name" class="selected"></h2><p id="description"></p><p id="details" class="muted"></p><p id="findings"></p><p id="cycles"></p><div class="table-wrap" id="outgoing"></div><div class="table-wrap" id="incoming"></div></section>
<footer>Offline snapshot viewer. No external scripts, network requests, package installation or executable actions. Listed ranges are declarations, not verified installed resolutions.</footer>
</main><script type="application/json" id="snapshot">${data}</script><script>
'use strict';
const report=JSON.parse(document.getElementById('snapshot').textContent);
const byId=id=>document.getElementById(id);
const text=(id,value)=>{byId(id).textContent=String(value??'');};
const packages=report.packages;
const names=new Map(packages.map(p=>[p.name,p]));
const edges=report.graph.edges;
const cycles=report.graph.runtimeCycles;
text('package-total',packages.length);text('edge-total',edges.length);text('cycle-total',cycles.length);text('scan-status',report.complete?'Complete':'Incomplete');
text('observed','Observed '+(report.observedAt??'unknown')+' · '+(report.maintainer??'unknown maintainer')+' · Advisory status: '+(report.advisoryStatus??'unknown'));
let selected=packages[0]?.name??null;
function cell(row,value){const td=document.createElement('td');td.textContent=String(value??'');row.append(td);return td;}
function renderEdges(id,title,items,incoming){
 const host=byId(id);host.replaceChildren();
 if(!items.length){const p=document.createElement('p');p.className='empty';p.textContent=title+': no observed direct ecosystem edges.';host.append(p);return;}
 const table=document.createElement('table');const caption=document.createElement('caption');caption.textContent=title+' ('+items.length+')';table.append(caption);
 const head=document.createElement('thead');const hr=document.createElement('tr');for(const value of ['Neighbor','Edge type','Declared range','Optional']){const th=document.createElement('th');th.scope='col';th.textContent=value;hr.append(th);}head.append(hr);table.append(head);
 const body=document.createElement('tbody');
 for(const edge of items){const row=document.createElement('tr');const name=incoming?edge.source:edge.target;const td=cell(row,'');const button=document.createElement('button');button.type='button';button.textContent=name;button.setAttribute('aria-label','Explore '+name);button.addEventListener('click',()=>{byId('search').value='';selected=name;filter();byId('packages').focus();});td.append(button);cell(row,edge.kind);cell(row,edge.range);cell(row,edge.optional?'Yes':'No');body.append(row);}
 table.append(body);host.append(table);
}
function render(){
 const pkg=names.get(selected);
 text('selected-name',pkg?.name??'No package selected');text('description',pkg?.description??'');text('details',pkg?'Version '+pkg.version+' · License '+(pkg.license??'unknown'):'');
 text('findings',pkg?'Findings: '+((pkg.findings??[]).join(', ')||'None recorded'):'');
 const groups=pkg?cycles.filter(c=>c.includes(pkg.name)):[];
 text('cycles',groups.length?'Runtime cycle groups containing this package: '+groups.map(c=>c.join(', ')).join('; '):pkg?'No observed runtime cycle contains this package.':'');
 renderEdges('outgoing','Depends on',pkg?edges.filter(e=>e.source===pkg.name):[],false);renderEdges('incoming','Used by',pkg?edges.filter(e=>e.target===pkg.name):[],true);
}
function filter(){
 const query=byId('search').value.toLowerCase();const matches=packages.filter(p=>String(p.name).toLowerCase().includes(query)||String(p.description??'').toLowerCase().includes(query));
 if(!matches.some(p=>p.name===selected))selected=matches[0]?.name??null;
 const select=byId('packages');select.replaceChildren();for(const pkg of matches){const option=document.createElement('option');option.value=pkg.name;option.textContent=pkg.name;option.selected=pkg.name===selected;select.append(option);}select.disabled=!matches.length;text('status',matches.length+' of '+packages.length+' packages match');render();
}
byId('search').addEventListener('input',filter);byId('packages').addEventListener('change',event=>{selected=event.target.value;render();});filter();
</script></body></html>`;
}
