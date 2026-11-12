const assert = require('node:assert/strict');
const http = require('node:http');
const zlib = require('node:zlib');
const axios = require('axios');
const compression = require('compression');
const qs = require('qs');
const proxyaddr = require('proxy-addr');
const {Address4, Address6} = require('ip-address');
const copy = require('fast-copy').default;
const {McpServer} = require('@modelcontextprotocol/sdk/server/mcp.js');
const {Client} = require('@modelcontextprotocol/sdk/client/index.js');
const {InMemoryTransport} = require('@modelcontextprotocol/sdk/inMemory.js');
const {z} = require('zod');
(async () => {
 assert.deepEqual(qs.parse('a[b]=1&__proto__[polluted]=yes'), {a:{b:'1'}});
 assert.equal({}.polluted, undefined);
 const source = {nested:{v:3}, array:[1,2]};
 const cloned = copy(source); cloned.nested.v=4;
 assert.equal(source.nested.v,3); assert.notEqual(cloned.array,source.array);
 assert.equal(new Address4('127.0.0.1').correctForm(),'127.0.0.1');
 assert.equal(new Address6('::1').correctForm(),'::1');
 const trusted=proxyaddr.compile('loopback');
 assert.equal(trusted('127.0.0.1'),true); assert.equal(trusted('8.8.8.8'),false);
 const middleware=compression({threshold:0});
 const payload='RuVector compression and axios contract '.repeat(200);
 const server=http.createServer((req,res)=>middleware(req,res,()=>{res.setHeader('content-type','text/plain');res.end(payload);}));
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 try {
  const url='http://127.0.0.1:'+server.address().port;
  const result=await axios.get(url,{proxy:false,timeout:3000});
  assert.equal(result.data,payload);
  const raw=await axios.get(url,{proxy:false,timeout:3000,decompress:false,responseType:'arraybuffer',headers:{'accept-encoding':'gzip'}});
  assert.equal(raw.headers['content-encoding'],'gzip');
  assert.equal(zlib.gunzipSync(raw.data).toString(),payload);
 } finally { await new Promise(resolve=>server.close(resolve)); }
 const sdkServer=new McpServer({name:'ruvector-contract',version:'1.0.0'});
 sdkServer.registerTool('echo',{inputSchema:{text:z.string().max(32)}},async({text})=>({content:[{type:'text',text}]}));
 const sdkClient=new Client({name:'contract-client',version:'1.0.0'});
 const [a,b]=InMemoryTransport.createLinkedPair();
 try {
  await sdkServer.connect(a); await sdkClient.connect(b);
  const tools=await sdkClient.listTools(); assert.equal(tools.tools[0].name,'echo');
  const answer=await sdkClient.callTool({name:'echo',arguments:{text:'native-safe'}});
  assert.equal(answer.content[0].text,'native-safe');
  const invalid=await sdkClient.callTool({name:'echo',arguments:{text:'x'.repeat(33)}});
  assert.equal(invalid.isError,true);
 } finally {await sdkClient.close(); await sdkServer.close();}
 console.log('PASS: 7 dependency consumers; MCP tool roundtrip and validation, HTTP compression/axios, query pollution, copy isolation, IP/proxy trust');
})().catch(error=>{console.error(error);process.exitCode=1;});
