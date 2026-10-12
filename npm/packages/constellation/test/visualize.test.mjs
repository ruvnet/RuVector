import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { reportHtml } from '../src/visualize.mjs';
const report = () => ({schemaVersion:1,observedAt:'2026-10-11',maintainer:'ruvnet',complete:true,advisoryStatus:'queried-direct-package-versions',packages:[{name:'a',version:'1',description:'first',findings:[]},{name:'b',version:'2',findings:[]}],graph:{edges:[{source:'a',target:'b',kind:'optionalDependencies',range:'^2',optional:true}],runtimeCycles:[]}});
test('embeds a lossless escaped snapshot with no external resources',()=>{
 const fixture=report();fixture.packages[0].name='</script><script>alert("x")</script>';
 fixture.packages[0].description='<img src=x onerror=alert(1)>&\u2028\u2029';
 const html=reportHtml(fixture);
 assert.equal((html.match(/<script\b/g)||[]).length,2);
 assert.equal((html.match(/<\/script>/g)||[]).length,2);
 assert(!html.includes('<img'));
 assert(!html.includes('alert("x")'));
 const payload=html.match(/id="snapshot">([\s\S]*?)<\/script>/)[1];
 assert(!/[<>&\u2028\u2029]/.test(payload));
 assert.deepEqual(JSON.parse(payload),fixture);
 assert(!/\b(?:src|href)=["']https?:/i.test(html));
 assert(!html.includes('innerHTML'));
});
test('renderer executes, filters packages, and follows typed neighbor links without HTML sinks',()=>{
 const html=reportHtml(report());const elements=new Map();
 class Element{
  constructor(tag){this.tag=tag;this.children=[];this.listeners={};this.value='';this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(){}
  addEventListener(type,fn){this.listeners[type]=fn;}
  focus(){}
 }
 const document={getElementById(id){if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);},createElement:tag=>new Element(tag)};
 document.getElementById('snapshot').textContent=JSON.stringify(report());
 const script=html.match(/<script>\n([\s\S]*?)<\/script>/)[1];vm.runInNewContext(script,{document});
 assert.equal(elements.get('package-total').textContent,'2');assert.equal(elements.get('selected-name').textContent,'a');
 const table=elements.get('outgoing').children[0];const row=table.children[2].children[0];
 assert.equal(row.children[1].textContent,'optionalDependencies');assert.equal(row.children[3].textContent,'Yes');
 row.children[0].children[0].listeners.click();assert.equal(elements.get('selected-name').textContent,'b');
 elements.get('search').value='missing';elements.get('search').listeners.input();assert.equal(elements.get('selected-name').textContent,'No package selected');assert.equal(elements.get('packages').disabled,true);
});
test('rejects incomplete report shapes and retains explicit audit scope',()=>{
 assert.throws(()=>reportHtml({packages:[]}),TypeError);
 const html=reportHtml(report());assert(html.includes('direct package versions only'));assert(html.includes('not certified'));
});
