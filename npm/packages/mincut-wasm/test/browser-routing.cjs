// A real Chromium module Worker, served from loopback with no external requests.
const { chromium } = require('playwright');
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const server = http.createServer((req,res) => {
  const pathname = new URL(req.url,'http://localhost').pathname;
  if(pathname==='/'){res.setHeader('content-type','text/html');res.end('<!doctype html><title>Routing test</title>');return;}
  const file=path.resolve(root,'.'+pathname);
  if(!file.startsWith(root+path.sep)){res.writeHead(403).end();return;}
  fs.readFile(file,(error,body)=>{
    if(error){res.writeHead(404).end();return;}
    res.setHeader('content-type',file.endsWith('.wasm')?'application/wasm':'text/javascript');res.end(body);
  });
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  let browser;
  try {
    browser=await chromium.launch({headless:true});
    const page=await browser.newPage();
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const result=await page.evaluate(async()=>{
      const {RoutingWorker}=await import('/routing/client.mjs');
      const r=await RoutingWorker.create(4,new Uint32Array([0,1,1,2,1,3,3,2]),new Uint32Array([1,1,2,2]),new Uint32Array([0,1]));
      try {
        await r.prepare(new Uint32Array([0,2]));
        const route=await r.route(0,2);
        await r.setCoordinates(new Float64Array([0,179.9,0,-179.8,90,0,-90,0]));
        const snap=await r.nearest(0,-180,50000);
        await r.update(new Uint32Array([2]),new Uint32Array([0xffffffff]));
        const unreachable=await r.route(0,2);
        const abort=new AbortController();
        const pending=r.route(0,2,{signal:abort.signal});abort.abort();
        let cancelled=false;try{await pending;}catch{cancelled=true;}
        return {route,snap,unreachable,cancelled};
      }finally{r.close();}
    });
    assert.equal(result.route.cost,5);assert.deepEqual(result.route.arcs,[0,2,3]);
    assert.equal(result.snap.node,0);assert.equal(result.unreachable,undefined);assert.equal(result.cancelled,true);
    assert.deepEqual(errors,[]);console.log('Chromium WASM Worker: route, turn, closure, map, cancellation passed');
  }finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve));}
})().catch(e=>{console.error(e);process.exitCode=1;});
