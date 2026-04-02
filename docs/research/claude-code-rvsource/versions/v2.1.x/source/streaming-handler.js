while(($=EM7(K))!==-1)yield K.slice(0,$),K=K.slice($)}if(K.length>0)yield K}class hM7{constructor(){this.event=null,this.data=[],this.chunks=[]}decode(q){if(q.endsWith("\r"))q=q.substring(0,q.length-1);if(!q){if(!this.event&&!this.data.length)return null;let Y={event:this.event,data:this.data.join(`
`),raw:this.chunks};return this.event=null,this.data=[],this.chunks=[],Y}if(this.chunks.push(q),q.startsWith(":"))return null;let[K,_,z]=rO5(q,":");if(z.startsWith(" "))z=z.substring(1);if(K==="event")this.event=z;else if(K==="data")this.data.push(z);return null}}function rO5(q,K){let _=q.indexOf(K);if(_!==-1)return[q.substring(0,_),K,q.substring(_+K.length)];return[q,"",""]}var hI6,rv;var Fa8=L(()=>{Dl();FW();Ba8();U96();w_8();FW();rv=class rv{constructor(q,K,_){this.iterator=q,hI6.set(this,void 0),this.controller=K,J4(this,hI6,_,"f")}static fromSSEResponse(q,K,_){let z=!1,Y=_?UW(_):console;async function*$(){if(z)throw new mq("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");z=!0;let O=!1;try{for await(let A of nO5(q,K)){if(A.event==="completion")try{yield JSON.parse(A.data)}catch(w){throw Y.error("Could not parse message into JSON:",A.data),Y.error("From chunk:",A.raw),w}if(A.event==="message_start"||A.event==="message_delta"||A.event==="message_stop"||A.event==="content_block_start"||A.event==="content_block_delta"||A.event==="content_block_stop")try{yield JSON.parse(A.data)}catch(w){throw Y.error("Could not parse message into JSON:",A.data),Y.error("From chunk:",A.raw),w}if(A.event==="ping")continue;if(A.event==="error")throw new nq(void 0,Y_8(A.data)??A.data,void 0,q.headers)}O=!0}catch(A){if(fl(A))return;throw A}finally{if(!O)K.abort()}}return new rv($,K,_)}static fromReadableStream(q,K,_){let z=!1;async function*Y(){let O=new Ze,A=yI6(q);for await(let w of A)for(let j of O.decode(w))yield j;for(let w of O.flush())yield w}async function*$(){if(z)throw new mq("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");z=!0;let O=!1

if(q==="abort"){let z=K[0];if(!x1(this,c96,"f")&&!_?.length)Promise.reject(z);x1(this,pI6,"f").call(this,z),x1(this,gI6,"f").call(this,z),this._emit("end");return}if(q==="error"){let z=K[0];if(!x1(this,c96,"f")&&!_?.length)Promise.reject(z);x1(this,pI6,"f").call(this,z),x1(this,gI6,"f").call(this,z),this._emit("end")}}_emitFinal(){if(this.receivedMessages.at(-1))this._emit("finalMessage",x1(this,ZR,"m",zs8).call(this))}async _fromReadableStream(q,K){let _=K?.signal,z;if(_){if(_.aborted)this.controller.abort();z=this.controller.abort.bind(this.controller),_.addEventListener("abort",z)}try{x1(this,ZR,"m",Ys8).call(this),this._connected(null);let Y=rv.fromReadableStream(q,this.controller);for await(let $ of Y)x1(this,ZR,"m",$s8).call(this,$);if(Y.controller.signal?.aborted)throw new c_;x1(this,ZR,"m",Os8).call(this)}finally{if(_&&z)_.removeEventListener("abort",z)}}[(Te=new WeakMap,VP6=new WeakMap,mI6=new WeakMap,G_8=new WeakMap,pI6=new WeakMap,BI6=new WeakMap,v_8=new WeakMap,gI6=new WeakMap,Gl=new WeakMap,FI6=new WeakMap,T_8=new WeakMap,k_8=new WeakMap,c96=new WeakMap,V_8=new WeakMap,N_8=new WeakMap,UI6=new WeakMap,y_8=new WeakMap,ZR=new WeakSet,zs8=function(){if(this.receivedMessages.length===0)throw new mq("stream ended without producing a Message with role=assistant");return this.receivedMessages.at(-1)},BM7=function(){if(this.receivedMessages.length===0)throw new mq("stream ended without producing a Message with role=assistant");let K=this.receivedMessages.at(-1).content.filter((_)=>_.type==="text").map((_)=>_.text);if(K.length===0)throw new mq("stream ended without producing a content block with type=text");return K.join(" ")},Ys8=function(){if(this.ended)return;J4(this,Te,void 0,"f")},$s8=function(K){if(this.ended)return;let _=x1(this,ZR,"m",gM7).call(this,K);switch(this._emit("streamEvent",K,_),K.type){case"content_block_delta":{let z=_.content.at(-1);switch(K.delta.type){case"text_delta":{if(z.type==="text")this._emit("text",K.delta.text,z.text||"")

break}case"citations_delta":{if(z.type==="text")this._emit("citation",K.delta.citation,z.citations??[]);break}case"input_json_delta":{if(UM7(z)&&z.input)this._emit("inputJson",K.delta.partial_json,z.input);break}case"thinking_delta":{if(z.type==="thinking")this._emit("thinking",K.delta.thinking,z.thinking);break}case"signature_delta":{if(z.type==="thinking")this._emit("signature",z.signature);break}case"compaction_delta":{if(z.type==="compaction"&&z.content)this._emit("compaction",z.content);break}default:QM7(K.delta)}break}case"message_stop":{this._addMessageParam(_),this._addMessage(ea8(_,x1(this,VP6,"f"),{logger:x1(this,UI6,"f")}),!0);break}case"content_block_stop":{this._emit("contentBlock",_.content.at(-1));break}case"message_start":{J4(this,Te,_,"f");break}case"content_block_start":case"message_delta":break}},Os8=function(){if(this.ended)throw new mq("stream has ended, this shouldn't happen");let K=x1(this,Te,"f");if(!K)throw new mq("request ended without sending any chunks");return J4(this,Te,void 0,"f"),ea8(K,x1(this,VP6,"f"),{logger:x1(this,UI6,"f")})},gM7=function(K){let _=x1(this,Te,"f");if(K.type==="message_start"){if(_)throw new mq(`Unexpected event order, got ${K.type} before receiving "message_stop"`);return K.message}if(!_)throw new mq(`Unexpected event order, got ${K.type} before "message_start"`);switch(K.type){case"message_stop":return _;case"message_delta":if(_.container=K.delta.container,_.stop_reason=K.delta.stop_reason,_.stop_sequence=K.delta.stop_sequence,_.usage.output_tokens=K.usage.output_tokens,_.context_management=K.context_management,K.usage.input_tokens!=null)_.usage.input_tokens=K.usage.input_tokens;if(K.usage.cache_creation_input_tokens!=null)_.usage.cache_creation_input_tokens=K.usage.cache_creation_input_tokens;if(K.usage.cache_read_input_tokens!=null)_.usage.cache_read_input_tokens=K.usage.cache_read_input_tokens;if(K.usage.server_tool_use!=null)_.usage.server_tool_use=K.usage.server_tool_use;if(K.usage.iterations!=null)_.usage.iterations=K.usage.iterations;return _

case"content_block_start":return _.content.push(K.content_block),_;case"content_block_delta":{let z=_.content.at(K.index);switch(K.delta.type){case"text_delta":{if(z?.type==="text")_.content[K.index]={...z,text:(z.text||"")+K.delta.text};break}case"citations_delta":{if(z?.type==="text")_.content[K.index]={...z,citations:[...z.citations??[],K.delta.citation]};break}case"input_json_delta":{if(z&&UM7(z)){let Y=z[FM7]||"";Y+=K.delta.partial_json;let $={...z};if(Object.defineProperty($,FM7,{value:Y,enumerable:!1,writable:!0}),Y)try{$.input=f_8(Y)}catch(O){let A=new mq(`Unable to parse tool parameter JSON from model. Please retry your request or adjust your prompt. Error: ${O}. JSON: ${Y}`);x1(this,y_8,"f").call(this,A)}_.content[K.index]=$}break}case"thinking_delta":{if(z?.type==="thinking")_.content[K.index]={...z,thinking:z.thinking+K.delta.thinking};break}case"signature_delta":{if(z?.type==="thinking")_.content[K.index]={...z,signature:K.delta.signature};break}case"compaction_delta":{if(z?.type==="compaction")_.content[K.index]={...z,content:(z.content||"")+K.delta.content};break}default:QM7(K.delta)}return _}case"content_block_stop":return _}},Symbol.asyncIterator)](){let q=[],K=[],_=!1;return this.on("streamEvent",(z)=>{let Y=K.shift();if(Y)Y.resolve(z);else q.push(z)}),this.on("end",()=>{_=!0;for(let z of K)z.resolve(void 0);K.length=0}),this.on("abort",(z)=>{_=!0;for(let Y of K)Y.reject(z);K.length=0}),this.on("error",(z)=>{_=!0;for(let Y of K)Y.reject(z);K.length=0}),{next:async()=>{if(!q.length){if(_)return{value:void 0,done:!0};return new Promise((Y,$)=>K.push({resolve:Y,reject:$})).then((Y)=>Y?{value:Y,done:!1}:{value:void 0,done:!0})}return{value:q.shift(),done:!1}},return:async()=>{return this.abort(),{value:void 0,done:!0}}}}toReadableStream(){return new rv(this[Symbol.asyncIterator].bind(this),this.controller).toReadableStream()}}});var NP6;var E_8=L(()=>{NP6=class NP6 extends Error{constructor(q){let K=typeof q==="string"?q:q.map((_)=>{if(_.type==="text")return _.text;return`[${_.type}]`}).join(" ")

if(q==="abort"){let z=K[0];if(!x1(this,n96,"f")&&!_?.length)Promise.reject(z);x1(this,aI6,"f").call(this,z),x1(this,tI6,"f").call(this,z),this._emit("end");return}if(q==="error"){let z=K[0];if(!x1(this,n96,"f")&&!_?.length)Promise.reject(z);x1(this,aI6,"f").call(this,z),x1(this,tI6,"f").call(this,z),this._emit("end")}}_emitFinal(){if(this.receivedMessages.at(-1))this._emit("finalMessage",x1(this,GR,"m",Gs8).call(this))}async _fromReadableStream(q,K){let _=K?.signal,z;if(_){if(_.aborted)this.controller.abort();z=this.controller.abort.bind(this.controller),_.addEventListener("abort",z)}try{x1(this,GR,"m",Ts8).call(this),this._connected(null);let Y=rv.fromReadableStream(q,this.controller);for await(let $ of Y)x1(this,GR,"m",ks8).call(this,$);if(Y.controller.signal?.aborted)throw new c_;x1(this,GR,"m",Vs8).call(this)}finally{if(_&&z)_.removeEventListener("abort",z)}}[(ye=new WeakMap,hP6=new WeakMap,oI6=new WeakMap,L_8=new WeakMap,aI6=new WeakMap,sI6=new WeakMap,h_8=new WeakMap,tI6=new WeakMap,Tl=new WeakMap,eI6=new WeakMap,R_8=new WeakMap,S_8=new WeakMap,n96=new WeakMap,C_8=new WeakMap,b_8=new WeakMap,qu6=new WeakMap,vs8=new WeakMap,GR=new WeakSet,Gs8=function(){if(this.receivedMessages.length===0)throw new mq("stream ended without producing a Message with role=assistant");return this.receivedMessages.at(-1)},sM7=function(){if(this.receivedMessages.length===0)throw new mq("stream ended without producing a Message with role=assistant");let K=this.receivedMessages.at(-1).content.filter((_)=>_.type==="text").map((_)=>_.text);if(K.length===0)throw new mq("stream ended without producing a content block with type=text");return K.join(" ")},Ts8=function(){if(this.ended)return;J4(this,ye,void 0,"f")},ks8=function(K){if(this.ended)return;let _=x1(this,GR,"m",tM7).call(this,K);switch(this._emit("streamEvent",K,_),K.type){case"content_block_delta":{let z=_.content.at(-1);switch(K.delta.type){case"text_delta":{if(z.type==="text")this._emit("text",K.delta.text,z.text||"")

break}case"citations_delta":{if(z.type==="text")this._emit("citation",K.delta.citation,z.citations??[]);break}case"input_json_delta":{if(qX7(z)&&z.input)this._emit("inputJson",K.delta.partial_json,z.input);break}case"thinking_delta":{if(z.type==="thinking")this._emit("thinking",K.delta.thinking,z.thinking);break}case"signature_delta":{if(z.type==="thinking")this._emit("signature",z.signature);break}default:KX7(K.delta)}break}case"message_stop":{this._addMessageParam(_),this._addMessage(Ds8(_,x1(this,hP6,"f"),{logger:x1(this,qu6,"f")}),!0);break}case"content_block_stop":{this._emit("contentBlock",_.content.at(-1));break}case"message_start":{J4(this,ye,_,"f");break}case"content_block_start":case"message_delta":break}},Vs8=function(){if(this.ended)throw new mq("stream has ended, this shouldn't happen");let K=x1(this,ye,"f");if(!K)throw new mq("request ended without sending any chunks");return J4(this,ye,void 0,"f"),Ds8(K,x1(this,hP6,"f"),{logger:x1(this,qu6,"f")})},tM7=function(K){let _=x1(this,ye,"f");if(K.type==="message_start"){if(_)throw new mq(`Unexpected event order, got ${K.type} before receiving "message_stop"`);return K.message}if(!_)throw new mq(`Unexpected event order, got ${K.type} before "message_start"`);switch(K.type){case"message_stop":return _;case"message_delta":if(_.stop_reason=K.delta.stop_reason,_.stop_sequence=K.delta.stop_sequence,_.usage.output_tokens=K.usage.output_tokens,K.usage.input_tokens!=null)_.usage.input_tokens=K.usage.input_tokens;if(K.usage.cache_creation_input_tokens!=null)_.usage.cache_creation_input_tokens=K.usage.cache_creation_input_tokens;if(K.usage.cache_read_input_tokens!=null)_.usage.cache_read_input_tokens=K.usage.cache_read_input_tokens;if(K.usage.server_tool_use!=null)_.usage.server_tool_use=K.usage.server_tool_use;return _;case"content_block_start":return _.content.push({...K.content_block}),_;case"content_block_delta":{let z=_.content.at(K.index);switch(K.delta.type){case"text_delta":{if(z?.type==="text")_.content[K.index]={...z,text:(z.text||"")+K.delta.text}

break}case"citations_delta":{if(z?.type==="text")_.content[K.index]={...z,citations:[...z.citations??[],K.delta.citation]};break}case"input_json_delta":{if(z&&qX7(z)){let Y=z[eM7]||"";Y+=K.delta.partial_json;let $={...z};if(Object.defineProperty($,eM7,{value:Y,enumerable:!1,writable:!0}),Y)$.input=f_8(Y);_.content[K.index]=$}break}case"thinking_delta":{if(z?.type==="thinking")_.content[K.index]={...z,thinking:z.thinking+K.delta.thinking};break}case"signature_delta":{if(z?.type==="thinking")_.content[K.index]={...z,signature:K.delta.signature};break}default:KX7(K.delta)}return _}case"content_block_stop":return _}},Symbol.asyncIterator)](){let q=[],K=[],_=!1;return this.on("streamEvent",(z)=>{let Y=K.shift();if(Y)Y.resolve(z);else q.push(z)}),this.on("end",()=>{_=!0;for(let z of K)z.resolve(void 0);K.length=0}),this.on("abort",(z)=>{_=!0;for(let Y of K)Y.reject(z);K.length=0}),this.on("error",(z)=>{_=!0;for(let Y of K)Y.reject(z);K.length=0}),{next:async()=>{if(!q.length){if(_)return{value:void 0,done:!0};return new Promise((Y,$)=>K.push({resolve:Y,reject:$})).then((Y)=>Y?{value:Y,done:!1}:{value:void 0,done:!0})}return{value:q.shift(),done:!1}},return:async()=>{return this.abort(),{value:void 0,done:!0}}}}toReadableStream(){return new rv(this[Symbol.asyncIterator].bind(this),this.controller).toReadableStream()}}});var _u6;var Ns8=L(()=>{uB();NE();js8();ve();Ge();_u6=class _u6 extends AH{create(q,K){return this._client.post("/v1/messages/batches",{body:q,...K})}retrieve(q,K){return this._client.get(jj`/v1/messages/batches/${q}`,K)}list(q={},K){return this._client.getAPIList("/v1/messages/batches",qI,{query:q,...K})}delete(q,K){return this._client.delete(jj`/v1/messages/batches/${q}`,K)}cancel(q,K){return this._client.post(jj`/v1/messages/batches/${q}/cancel`,K)}async results(q,K){let _=await this.retrieve(q);if(!_.results_url)throw new mq(`No batch \`results_url\`; Has it finished processing? ${_.processing_status} - ${_.id}`)

return!0})),z=F$Y(_.map((O)=>O.message.content),t77),Y=g0K(process.env.CLAUDE_CODE_TEST_FIXTURES_ROOT??Z8(),`fixtures/${z.map((O)=>I0K("sha1").update(g6(O)).digest("hex").slice(0,6)).join("-")}.json`);try{let O=l8(await m0K(Y,{encoding:"utf8"}));return O.output.forEach(g$Y),O.output.map((A,w)=>x0K(A,Q$Y,w,p$Y()))}catch(O){if(d1(O)!=="ENOENT")throw O}if(Y7.isCI&&!c6(process.env.VCR_RECORD))throw Error(`Anthropic API fixture missing: ${Y}. Re-run tests with VCR_RECORD=1, then commit the result. Input messages:
${g6(z,null,2)}`);let $=await K();if(Y7.isCI&&!c6(process.env.VCR_RECORD))return $;return await u0K(B0K(Y),{recursive:!0}),await p0K(Y,g6({input:z,output:$.map((O,A)=>x0K(O,t77,A))},null,2),{encoding:"utf8"}),$}function g$Y(q){if(q.type==="stream_event")return;let K=q.message.model,_=q.message.usage,z=x86(K,_);Fh6(z,_,K)}function F$Y(q,K){return q.map((_)=>{if(typeof _==="string")return K(_);return _.map((z)=>{switch(z.type){case"tool_result":if(typeof z.content==="string")return{...z,content:K(z.content)};if(Array.isArray(z.content))return{...z,content:z.content.map((Y)=>{switch(Y.type){case"text":return{...Y,text:K(Y.text)};case"image":return Y;default:return}})};return z;case"text":return{...z,text:K(z.text)};case"tool_use":return{...z,input:YF8(z.input,K)};case"image":return z;default:return}})})}function YF8(q,K){return SC(q,(_,z)=>{if(Array.isArray(_))return _.map((Y)=>YF8(Y,K));if(HD6(_))return YF8(_,K);return K(_,z,q)})}function U$Y(q,K,_,z){return{uuid:z??`UUID-${_}`,requestId:"REQUEST_ID",timestamp:q.timestamp,message:{...q.message,content:q.message.content.map((Y)=>{switch(Y.type){case"text":return{...Y,text:K(Y.text),citations:Y.citations||[]};case"tool_use":return{...Y,input:YF8(Y.input,K)};default:return Y}}).filter(Boolean)},type:"assistant"}}function x0K(q,K,_,z){if(q.type==="assistant")return U$Y(q,K,_,z);else return q}function t77(q){if(typeof q!=="string")return q

for(let A of O){if(A.type==="tool_use")_.add(A.id);if(A.type==="tool_result")z.add(A.tool_use_id)}}let Y=new Set([..._].filter(($)=>!z.has($)&&!K?.has($)));if(Y.size===0)return q;return q.filter(($)=>{if($.type!=="assistant")return!0;let O=$.message.content;if(!Array.isArray(O))return!0;let A=[];for(let w of O)if(w.type==="tool_use")A.push(w.id);if(A.length===0)return!0;return!A.every((w)=>Y.has(w))})}function KS6(q){if(q.type!=="assistant")return null;if(Array.isArray(q.message.content))return q.message.content.filter((K)=>K.type==="text").map((K)=>K.type==="text"?K.text:"").join(`
`).trim()||null;return null}function yQ(q){if(q.type!=="user")return null;let K=q.message.content;return Hd(K)}function H47(q){let K=yQ(q);if(K===null)return null;let _=qK(K,"bash-input");if(_)return{text:_,mode:"bash"};let z=qK(K,zG);if(z){let Y=qK(K,gO8)??"";return{text:`${z} ${Y}`,mode:"prompt"}}return{text:dL7(K),mode:"prompt"}}function Z3(q,K=""){return q.filter((_)=>_.type==="text").map((_)=>_.text).join(K)}function Hd(q){if(typeof q==="string")return q;if(Array.isArray(q))return Z3(q,`
`).trim()||null;return null}function fS6(q,K,_,z,Y,$,O,A,w){if(q.type!=="stream_event"&&q.type!=="stream_request_start"){if(q.type==="tombstone"){$?.(q.message);return}if(q.type==="tool_use_summary")return;if(q.type==="assistant"){let j=q.message.content.find((H)=>H.type==="thinking");if(j&&j.type==="thinking")O?.(()=>({thinking:j.thinking,isStreaming:!1,streamingEndedAt:Date.now()}))}w?.(()=>null),K(q);return}if(q.type==="stream_request_start"){z("requesting");return}if(q.event.type==="message_start"){if(q.ttftMs!=null)A?.({ttftMs:q.ttftMs})}if(q.event.type==="message_stop"){z("tool-use"),Y(()=>[]);return}switch(q.event.type){case"content_block_start":switch(w?.(()=>null),q.event.content_block.type){case"thinking":case"redacted_thinking":z("thinking");return;case"text":z("responding");return;case"tool_use":{z("tool-input");let j=q.event.content_block,H=q.event.index;Y((J)=>[...J,{index:H,contentBlock:j,unparsedToolInput:""}])

return}case"server_tool_use":case"web_search_tool_result":case"code_execution_tool_result":case"mcp_tool_use":case"mcp_tool_result":case"container_upload":case"web_fetch_tool_result":case"bash_code_execution_tool_result":case"text_editor_code_execution_tool_result":case"tool_search_tool_result":case"compaction":z("tool-input");return}return;case"content_block_delta":switch(q.event.delta.type){case"text_delta":{let j=q.event.delta.text;_(j),w?.((H)=>(H??"")+j);return}case"input_json_delta":{let j=q.event.delta.partial_json,H=q.event.index;_(j),Y((J)=>{let M=J.find((X)=>X.index===H);if(!M)return J;return[...J.filter((X)=>X!==M),{...M,unparsedToolInput:M.unparsedToolInput+j}]});return}case"thinking_delta":_(q.event.delta.thinking);return;case"signature_delta":return;default:return}case"content_block_stop":return;case"message_delta":z("responding");return;default:z("responding");return}}function Nv(q){return`<system-reminder>
${q}
</system-reminder>`}function V9(q){return q.map((K)=>{if(typeof K.message.content==="string")return{...K,message:{...K.message,content:Nv(K.message.content)}};else if(Array.isArray(K.message.content)){let _=K.message.content.map((z)=>{if(z.type==="text")return{...z,text:Nv(z.text)};return z});return{...K,message:{...K.message,content:_}}}return K})}function A2Y(q){if(q.isSubAgent)return D2Y(q);if(q.reminderType==="sparse")return W2Y(q);return M2Y(q)}function J2Y(){let q=QB8();switch(q){case"trim":return w2Y;case"cut":return j2Y;case"cap":return H2Y;case null:return mvK;default:return mvK}}function M2Y(q){if(q.isSubAgent)return[];if(s2())return P2Y(q)

if(oO8(b8,$.querySource),K8=b8.max_tokens,g3("query_api_request_sent"),!$.agentId)vM("api_request_sent");let E1=Dq();e=E1==="firstParty"&&OM()||E1==="anthropicAws"&&!process.env.ANTHROPIC_AWS_BASE_URL?ql8():void 0;let _7=await r6.beta.messages.create({...b8,stream:!0},{signal:Y,...e&&{headers:{[cG6]:e}}}).withResponse();return g3("query_response_headers_received"),H6=_7.request_id,a=_7.response,_7.data},{model:$.model,fallbackModel:$.fallbackModel,thinkingConfig:_,...gK()?{fastMode:g}:!1,signal:Y,querySource:$.querySource}),b6;do if(b6=await m6.next(),!("controller"in b6.value))yield b6.value;while(!b6.done);G6=b6.value,X6.length=0,v6=0,x6=void 0,R6.length=0,W6=wf,Z6=null,k6=!1;let T6=c6(process.env.CLAUDE_ENABLE_STREAM_WATCHDOG),s=parseInt(process.env.CLAUDE_STREAM_IDLE_TIMEOUT_MS||"",10)||90000,$6=s/2,h6=!1,P6=null,V6=null,S6=null;e6(),Vg8("api_call");try{let r6=!0,R8=null,C8=30000,b8=0,E1=0;for await(let D1 of G6){e6();let M7=Date.now();if(R8!==null){let N7=M7-R8;if(N7>C8)E1++,b8+=N7,N(`Streaming stall detected: ${(N7/1000).toFixed(1)}s gap between events (stall #${E1})`,{level:"warn"}),d("tengu_streaming_stall",{stall_duration_ms:N7,stall_count:E1,total_stall_time_ms:b8,event_type:D1.type,model:$.model,request_id:H6??"unknown"})}if(R8=M7,r6){if(N("Stream started - received first chunk"),g3("query_first_chunk_received"),!$.agentId)vM("first_chunk");tfK(),r6=!1}switch(D1.type){case"message_start":{x6=D1.message,v6=Date.now()-z6,W6=x56(W6,D1.message?.usage);break}case"content_block_start":switch(D1.content_block.type){case"tool_use":R6[D1.index]={...D1.content_block,input:""};break;case"server_tool_use":if(R6[D1.index]={...D1.content_block,input:""},D1.content_block.name==="advisor")k6=!0,N("[AdvisorTool] Advisor tool called"),d("tengu_advisor_tool_call",{model:$.model,advisor_model:H??"unknown"});break;case"text":R6[D1.index]={...D1.content_block,text:""};break;case"thinking":R6[D1.index]={...D1.content_block,thinking:"",signature:""};break

default:if(R6[D1.index]={...D1.content_block},D1.content_block.type==="advisor_tool_result")k6=!1,N("[AdvisorTool] Advisor tool result received");break}break;case"content_block_delta":{let N7=R6[D1.index],P1=D1.delta;if(!N7)throw d("tengu_streaming_error",{error_type:"content_block_not_found_delta",part_type:D1.type,part_index:D1.index}),RangeError("Content block not found");switch(P1.type){case"citations_delta":break;case"input_json_delta":if(N7.type!=="tool_use"&&N7.type!=="server_tool_use")throw d("tengu_streaming_error",{error_type:"content_block_type_mismatch_input_json",expected_type:"tool_use",actual_type:N7.type}),Error("Content block is not a input_json block");if(typeof N7.input!=="string")throw d("tengu_streaming_error",{error_type:"content_block_input_not_string",input_type:typeof N7.input}),Error("Content block input is not a string");N7.input+=P1.partial_json;break;case"text_delta":if(N7.type!=="text")throw d("tengu_streaming_error",{error_type:"content_block_type_mismatch_text",expected_type:"text",actual_type:N7.type}),Error("Content block is not a text block");N7.text+=P1.text;break;case"signature_delta":if(N7.type!=="thinking")throw d("tengu_streaming_error",{error_type:"content_block_type_mismatch_thinking_signature",expected_type:"thinking",actual_type:N7.type}),Error("Content block is not a thinking block");N7.signature=P1.signature;break;case"thinking_delta":if(N7.type!=="thinking")throw d("tengu_streaming_error",{error_type:"content_block_type_mismatch_thinking_delta",expected_type:"thinking",actual_type:N7.type}),Error("Content block is not a thinking block");N7.thinking+=P1.thinking;break}break}case"content_block_stop":{let N7=R6[D1.index];if(!N7)throw d("tengu_streaming_error",{error_type:"content_block_not_found_stop",part_type:D1.type,part_index:D1.index}),RangeError("Content block not found");if(!x6)throw d("tengu_streaming_error",{error_type:"partial_message_not_found",part_type:D1.type}),Error("Message not found")

let P1={message:{...x6,content:UF8([N7],z,$.agentId)},requestId:H6??void 0,type:"assistant",uuid:ql8(),timestamp:new Date().toISOString(),...!1,...H&&{advisorModel:H}};X6.push(P1),yield P1;break}case"message_delta":{W6=x56(W6,D1.usage),Z6=D1.delta.stop_reason;let N7=X6.at(-1);if(N7)N7.message.usage=W6,N7.message.stop_reason=Z6;let P1=x86(A,W6);N6+=Fh6(P1,W6,$.model);let D7=WTK(D1.delta.stop_reason,$.model);if(D7)yield D7;if(Z6==="max_tokens")d("tengu_max_tokens_reached",{max_tokens:K8}),yield U9({content:`${MW}: Claude's response exceeded the ${K8} output token maximum. To configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.`,apiError:"max_output_tokens",error:"max_output_tokens"});if(Z6==="model_context_window_exceeded")d("tengu_context_window_exceeded",{max_tokens:K8,output_tokens:W6.output_tokens}),yield U9({content:`${MW}: The model has reached its context window limit.`,apiError:"max_output_tokens",error:"max_output_tokens"});break}case"message_stop":break}yield{type:"stream_event",event:D1,...D1.type==="message_start"?{ttftMs:v6}:void 0}}if(q8(),h6){let D1=P6!==null?Math.round(performance.now()-P6):-1;throw a8("info","cli_stream_loop_exited_after_watchdog_clean"),d("tengu_stream_loop_exited_after_watchdog",{request_id:H6??"unknown",exit_delay_ms:D1,exit_path:"clean",model:$.model}),P6=null,Error("Stream idle timeout - no chunks received")}if(!x6||X6.length===0&&!Z6)throw N(!x6?"Stream completed without receiving message_start event - triggering non-streaming fallback":"Stream completed with message_start but no content blocks completed - triggering non-streaming fallback",{level:"error"}),d("tengu_stream_no_events",{model:$.model,request_id:H6??"unknown"}),Error("Stream ended without receiving any events");if(E1>0)N(`Streaming completed with ${E1} stall(s), total stall time: ${(b8/1000).toFixed(1)}s`,{level:"warn"}),d("tengu_streaming_stall_summary",{stall_count:E1,total_stall_time_ms:b8,model:$.model,request_id:H6??"unknown"});let _7=a

if(_=_.replace("/ws/","/session/"),!_.endsWith("/events"))_=_.endsWith("/")?_+"events":_+"/events";return`${K}//${q.host}${_}${q.search}`}var eBY=100,qgY=15000,KgY=3000,FK8;var Q$7=L(()=>{VK();_8();w$();tL();F$7();U$7();FK8=class FK8 extends gK8{postUrl;uploader;streamEventBuffer=[];streamEventTimer=null;constructor(q,K={},_,z,Y){super(q,K,_,z,Y);let{maxConsecutiveFailures:$,onBatchDropped:O}=Y??{};this.postUrl=_gY(q),this.uploader=new MM6({maxBatchSize:500,maxQueueSize:1e5,baseDelayMs:500,maxDelayMs:8000,jitterMs:1000,maxConsecutiveFailures:$,onBatchDropped:(A,w)=>{a8("error","cli_hybrid_batch_dropped_max_failures",{batchSize:A,failures:w}),O?.(A,w)},send:(A)=>this.postOnce(A)}),N(`HybridTransport: POST URL = ${this.postUrl}`),a8("info","cli_hybrid_transport_initialized")}async write(q){if(q.type==="stream_event"){if(this.streamEventBuffer.push(q),!this.streamEventTimer)this.streamEventTimer=setTimeout(()=>this.flushStreamEvents(),eBY);return}return await this.uploader.enqueue([...this.takeStreamEvents(),q]),this.uploader.flush()}async writeBatch(q){return await this.uploader.enqueue([...this.takeStreamEvents(),...q]),this.uploader.flush()}get droppedBatchCount(){return this.uploader.droppedBatchCount}flush(){return this.uploader.enqueue(this.takeStreamEvents()),this.uploader.flush()}takeStreamEvents(){if(this.streamEventTimer)clearTimeout(this.streamEventTimer),this.streamEventTimer=null;let q=this.streamEventBuffer;return this.streamEventBuffer=[],q}flushStreamEvents(){this.streamEventTimer=null,this.uploader.enqueue(this.takeStreamEvents())}close(){if(this.streamEventTimer)clearTimeout(this.streamEventTimer),this.streamEventTimer=null;this.streamEventBuffer=[];let q=this.uploader,K;Promise.race([q.flush(),new Promise((_)=>{K=setTimeout(_,KgY)})]).finally(()=>{clearTimeout(K),q.close()}),super.close()}async postOnce(q){let K=FD();if(!K){N("HybridTransport: No session token available for POST"),a8("warn","cli_hybrid_post_no_token");return}let _={Authorization:`Bearer ${K}`,"Content-Type":"application/json"},z

try{z=await O1.post(this.postUrl,{events:q},{headers:_,validateStatus:()=>!0,timeout:qgY})}catch(Y){throw N(`HybridTransport: POST error: ${Y.message}`),a8("warn","cli_hybrid_post_network_error"),Y}if(z.status>=200&&z.status<300){N(`HybridTransport: POST success count=${q.length}`);return}if(z.status>=400&&z.status<500&&z.status!==429){N(`HybridTransport: POST returned ${z.status} (permanent), dropping`),a8("warn","cli_hybrid_post_client_error",{status:z.status});return}throw N(`HybridTransport: POST returned ${z.status} (retryable)`),a8("warn","cli_hybrid_post_retryable_error",{status:z.status}),Error(`POST failed with ${z.status}`)}}});class d$7{inflight=null;pending=null;closed=!1;config;constructor(q){this.config=q}enqueue(q){if(this.closed)return;this.pending=this.pending?JnK(this.pending,q):q,this.drain()}close(){this.closed=!0,this.pending=null}async drain(){if(this.inflight||this.closed)return;if(!this.pending)return;let q=this.pending;this.pending=null,this.inflight=this.sendWithRetry(q).then(()=>{if(this.inflight=null,this.pending&&!this.closed)this.drain()})}async sendWithRetry(q){let K=q,_=0;while(!this.closed){if(await this.config.send(K))return;if(_++,await C7(this.retryDelay(_)),this.pending&&!this.closed)K=JnK(K,this.pending),this.pending=null}}retryDelay(q){let K=Math.min(this.config.baseDelayMs*2**(q-1),this.config.maxDelayMs),_=Math.random()*this.config.jitterMs;return K+_}}function JnK(q,K){let _={...q};for(let[z,Y]of Object.entries(K))if((z==="external_metadata"||z==="internal_metadata")&&_[z]&&typeof _[z]==="object"&&typeof Y==="object"&&Y!==null)_[z]={..._[z],...Y};else _[z]=Y;return _}var MnK=()=>{};import{randomUUID as XnK}from"crypto";function PnK(){return!0}function OgY(){return{byMessage:new Map,scopeToMessage:new Map}}function Ll8(q){return`${q.session_id}:${q.parent_tool_use_id??""}`}function AgY(q,K){let _=[],z=new Map;for(let Y of q)switch(Y.event.type){case"message_start":{let $=Y.event.message.id,O=K.scopeToMessage.get(Ll8(Y));if(O)K.byMessage.delete(O)

K.scopeToMessage.set(Ll8(Y),$),K.byMessage.set($,[]),_.push(Y);break}case"content_block_delta":{if(Y.event.delta.type!=="text_delta"){_.push(Y);break}let $=K.scopeToMessage.get(Ll8(Y)),O=$?K.byMessage.get($):void 0;if(!O){_.push(Y);break}let A=O[Y.event.index]??=[];A.push(Y.event.delta.text);let w=z.get(A);if(w){w.event.delta.text=A.join("");break}let j={type:"stream_event",uuid:Y.uuid,session_id:Y.session_id,parent_tool_use_id:Y.parent_tool_use_id,event:{type:"content_block_delta",index:Y.event.index,delta:{type:"text_delta",text:A.join("")}}};z.set(A,j),_.push(j);break}default:_.push(Y)}return _}function wgY(q,K){q.byMessage.delete(K.message.id);let _=Ll8(K);if(q.scopeToMessage.get(_)===K.message.id)q.scopeToMessage.delete(_)}class UK8{workerEpoch=0;heartbeatIntervalMs;heartbeatJitterFraction;heartbeatTimer=null;heartbeatInFlight=!1;closed=!1;consecutiveAuthFailures=0;currentState=null;sessionBaseUrl;sessionId;http=v4q({keepAlive:!0});streamEventBuffer=[];streamEventTimer=null;streamTextAccumulator=OgY();workerState;eventUploader;internalEventUploader;deliveryUploader;onEpochMismatch;getAuthHeaders;constructor(q,K,_){if(this.onEpochMismatch=_?.onEpochMismatch??(()=>{process.exit(1)}),this.heartbeatIntervalMs=_?.heartbeatIntervalMs??zgY,this.heartbeatJitterFraction=_?.heartbeatJitterFraction??0,this.getAuthHeaders=_?.getAuthHeaders??Vy6,K.protocol!=="http:"&&K.protocol!=="https:")throw Error(`CCRClient: Expected http(s) URL, got ${K.protocol}`);let z=K.pathname.replace(/\/$/,"");this.sessionBaseUrl=`${K.protocol}//${K.host}${z}`,this.sessionId=z.split("/").pop()||"",this.workerState=new d$7({send:(Y)=>this.request("put","/worker",{worker_epoch:this.workerEpoch,...Y},"PUT worker").then(($)=>$.ok),baseDelayMs:500,maxDelayMs:30000,jitterMs:500}),this.eventUploader=new MM6({maxBatchSize:100,maxBatchBytes:10485760,maxQueueSize:1e5,send:async(Y)=>{let $=await this.request("post","/worker/events",{worker_epoch:this.workerEpoch,events:Y},"client events")

this.currentState=q,this.workerState.enqueue({worker_status:q,requires_action_details:K?{tool_name:K.tool_name,action_description:K.action_description,request_id:K.request_id}:null})}reportMetadata(q){this.workerState.enqueue({external_metadata:q})}handleEpochMismatch(){N("CCRClient: Epoch mismatch (409), shutting down",{level:"error"}),a8("error","cli_worker_epoch_mismatch"),this.onEpochMismatch()}startHeartbeat(){this.stopHeartbeat();let q=()=>{let _=this.heartbeatIntervalMs*this.heartbeatJitterFraction*(2*Math.random()-1);this.heartbeatTimer=setTimeout(K,this.heartbeatIntervalMs+_)},K=()=>{if(this.sendHeartbeat(),this.heartbeatTimer===null)return;q()};q()}stopHeartbeat(){if(this.heartbeatTimer)clearTimeout(this.heartbeatTimer),this.heartbeatTimer=null}async sendHeartbeat(){if(this.heartbeatInFlight)return;this.heartbeatInFlight=!0;try{if((await this.request("post","/worker/heartbeat",{session_id:this.sessionId,worker_epoch:this.workerEpoch},"Heartbeat",{timeout:5000})).ok)N("CCRClient: Heartbeat sent")}finally{this.heartbeatInFlight=!1}}async writeEvent(q){if(q.type==="stream_event"){if(this.streamEventBuffer.push(q),!this.streamEventTimer)this.streamEventTimer=setTimeout(()=>void this.flushStreamEventBuffer(),YgY);return}if(await this.flushStreamEventBuffer(),q.type==="assistant")wgY(this.streamTextAccumulator,q);await this.eventUploader.enqueue(this.toClientEvent(q))}toClientEvent(q){let K=q;return{payload:{...K,uuid:typeof K.uuid==="string"?K.uuid:XnK()}}}async flushStreamEventBuffer(){if(this.streamEventTimer)clearTimeout(this.streamEventTimer),this.streamEventTimer=null;if(this.streamEventBuffer.length===0)return;let q=this.streamEventBuffer;this.streamEventBuffer=[];let K=AgY(q,this.streamTextAccumulator);await this.eventUploader.enqueue(K.map((_)=>({payload:_,ephemeral:!0})))}async writeInternalEvent(q,K,{isCompaction:_=!1,agentId:z}={}){let Y={payload:{type:q,...K,uuid:typeof K.uuid==="string"?K.uuid:XnK()},..._&&{is_compaction:!0},...z&&{agent_id:z}}

return{type:"system",subtype:"informational",content:K?q.errors?.join(", ")||"Unknown error":"Session completed successfully",level:K?"warning":"info",uuid:q.uuid,timestamp:new Date().toISOString()}}function NnY(q){return{type:"system",subtype:"informational",content:`Remote session initialized (model: ${q.model})`,level:"info",uuid:q.uuid,timestamp:new Date().toISOString()}}function ynY(q){if(!q.status)return null;return{type:"system",subtype:"informational",content:q.status==="compacting"?"Compacting conversation…":`Status: ${q.status}`,level:"info",uuid:q.uuid,timestamp:new Date().toISOString()}}function EnY(q){return{type:"system",subtype:"informational",content:`Tool ${q.tool_name} running for ${q.elapsed_time_seconds}s…`,level:"info",uuid:q.uuid,timestamp:new Date().toISOString(),toolUseID:q.tool_use_id}}function LnY(q){return{type:"system",subtype:"compact_boundary",content:"Conversation compacted",level:"info",uuid:q.uuid,timestamp:new Date().toISOString(),compactMetadata:M_7(q.compact_metadata)}}function uM6(q,K){switch(q.type){case"assistant":return{type:"message",message:TnY(q)};case"user":{let _=q.message?.content,z=Array.isArray(_)&&_.some((Y)=>Y.type==="tool_result");if(K?.convertToolResults&&z)return{type:"message",message:n8({content:_,toolUseResult:q.tool_use_result,uuid:q.uuid,timestamp:q.timestamp})};if(K?.convertUserTextMessages&&!z){if(typeof _==="string"||Array.isArray(_))return{type:"message",message:n8({content:_,toolUseResult:q.tool_use_result,uuid:q.uuid,timestamp:q.timestamp})}}return{type:"ignored"}}case"stream_event":return{type:"stream_event",event:knY(q)};case"result":if(q.subtype!=="success")return{type:"message",message:VnY(q)};return{type:"ignored"};case"system":if(q.subtype==="init")return{type:"message",message:NnY(q)};if(q.subtype==="status"){let _=ynY(q);return _?{type:"message",message:_}:{type:"ignored"}}if(q.subtype==="compact_boundary")return{type:"message",message:LnY(q)};return N(`[sdkMessageAdapter] Ignoring system message subtype: ${q.subtype}`),{type:"ignored"}

filter to type=="text")
echo "$response" | jq -r '.content[] | select(.type == "text") | .text'
\`\`\`


---

## Streaming (SSE)

\`\`\`bash
curl https://api.anthropic.com/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: $ANTHROPIC_API_KEY" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "{{OPUS_ID}}",
    "max_tokens": 64000,
    "stream": true,
    "messages": [{"role": "user", "content": "Write a haiku"}]
  }'
\`\`\`

The response is a stream of Server-Sent Events:

\`\`\`
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}

event: message_stop
data: {"type":"message_stop"}
\`\`\`

---

## Tool Use

\`\`\`bash
curl https://api.anthropic.com/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: $ANTHROPIC_API_KEY" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "{{OPUS_ID}}",
    "max_tokens": 16000,
    "tools": [{
      "name": "get_weather",
      "description": "Get current weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
      }
    }],
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}]
  }'
\`\`\`

When Claude responds with a \`tool_use\` block, send the result back:

\`\`\`bash
curl https://api.anthropic.com/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: $ANTHROPIC_API_KEY" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "{{OPUS_ID}}",
    "max_tokens": 16000,
    "tools": [{
      "name": "get_weather",
      "description": "Get current weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"]
      }
    }],
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"},
      {"role": "assistant", "content": [
        {"type": "text", "text": "Let me check the weather."},
        {"type": "tool_use", "id": "toolu_abc123", "name": "get_weather", "input": {"location": "Paris"}}
      ]},
      {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_abc123", "content": "72°F and sunny"}
      ]}
    ]
  }'
\`\`\`

---

## Prompt Caching

Put \`cache_control\` on the last block of the stable prefix. See \`shared/prompt-caching.md\` for placement patterns and the silent-invalidator audit checklist.

\`\`\`bash
curl https://api.anthropic.com/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: $ANTHROPIC_API_KEY" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "{{OPUS_ID}}",
    "max_tokens": 16000,
    "system": [
      {"type": "text", "text": "<large shared prompt...>", "cache_control": {"type": "ephemeral"}}
    ],
    "messages": [{"role": "user", "content": "Summarize the key points"}]
  }'
\`\`\`

For 1-hour TTL: \`"cache_control": {"type": "ephemeral", "ttl": "1h"}\`. Top-level \`"cache_control"\` on the request body auto-places on the last cacheable block. Verify hits via the response \`usage.cache_creation_input_tokens\` / \`usage.cache_read_input_tokens\` fields.

---

## Extended Thinking

> **Opus 4.6 and Sonnet 4.6:** Use adaptive thinking. \`budget_tokens\` is deprecated on both Opus 4.6 and Sonnet 4.6.
> **Older models:** Use \`"type": "enabled"\` with \`"budget_tokens": N\` (must be < \`max_tokens\`, min 1024).

\`\`\`bash
# Opus 4.6: adaptive thinking (recommended)
curl https://api.anthropic.com/v1/messages \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: $ANTHROPIC_API_KEY" \\
  -H "anthropic-version: 2023-06-01" \\
  -d '{
    "model": "{{OPUS_ID}}",
    "max_tokens": 16000,
    "thinking": {
      "type": "adaptive"
    },
    "output_config": {
      "effort": "high"
    },
    "messages": [{"role": "user", "content": "Solve this step by step..."}]
  }'
\`\`\`

---

## Required Headers

| Header              | Value              | Description                |
| ------------------- | ------------------ | -------------------------- |
| \`Content-Type\`      | \`application/json\` | Required                   |
| \`x-api-key\`         | Your API key       | Authentication             |
| \`anthropic-version\` | \`2023-06-01\`       | API version                |
| \`anthropic-beta\`    | Beta feature IDs   | Required for beta features |
`

var U45=`# Streaming — Python

## Quick Start

\`\`\`python
with client.messages.stream(
    model="{{OPUS_ID}}",
    max_tokens=64000,
    messages=[{"role": "user", "content": "Write a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
\`\`\`

### Async

\`\`\`python
async with async_client.messages.stream(
    model="{{OPUS_ID}}",
    max_tokens=64000,
    messages=[{"role": "user", "content": "Write a story"}]
) as stream:
    async for text in stream.text_stream:
        print(text, end="", flush=True)
\`\`\`

---

## Handling Different Content Types

Claude may return text, thinking blocks, or tool use. Handle each appropriately:

> **Opus 4.6:** Use \`thinking: {type: "adaptive"}\`. On older models, use \`thinking: {type: "enabled", budget_tokens: N}\` instead.

\`\`\`python
with client.messages.stream(
    model="{{OPUS_ID}}",
    max_tokens=64000,
    thinking={"type": "adaptive"},
    messages=[{"role": "user", "content": "Analyze this problem"}]
) as stream:
    for event in stream:
        if event.type == "content_block_start":
            if event.content_block.type == "thinking":
                print("\\n[Thinking...]")
            elif event.content_block.type == "text":
                print("\\n[Response:]")

        elif event.type == "content_block_delta":
            if event.delta.type == "thinking_delta":
                print(event.delta.thinking, end="", flush=True)
            elif event.delta.type == "text_delta":
                print(event.delta.text, end="", flush=True)
\`\`\`

---

## Streaming with Tool Use

The Python tool runner currently returns complete messages. Use streaming for individual API calls within a manual loop if you need per-token streaming with tools:

\`\`\`python
with client.messages.stream(
    model="{{OPUS_ID}}",
    max_tokens=64000,
    tools=tools,
    messages=messages
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    response = stream.get_final_message()
    # Continue with tool execution if response.stop_reason == "tool_use"
\`\`\`

---

## Getting the Final Message

\`\`\`python
with client.messages.stream(
    model="{{OPUS_ID}}",
    max_tokens=64000,
    messages=[{"role": "user", "content": "Hello"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

    # Get full message after streaming
    final_message = stream.get_final_message()
    print(f"\\n\\nTokens used: {final_message.usage.output_tokens}")
\`\`\`

---

## Streaming with Progress Updates

\`\`\`python
def stream_with_progress(client, **kwargs):
    """Stream a response with progress updates."""
    total_tokens = 0
    content_parts = []

    with client.messages.stream(**kwargs) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    text = event.delta.text
                    content_parts.append(text)
                    print(text, end="", flush=True)

            elif event.type == "message_delta":
                if event.usage and event.usage.output_tokens is not None:
                    total_tokens = event.usage.output_tokens

        final_message = stream.get_final_message()

    print(f"\\n\\n[Tokens used: {total_tokens}]")
    return "".join(content_parts)
\`\`\`

---

## Error Handling in Streams

\`\`\`python
try:
    with client.messages.stream(
        model="{{OPUS_ID}}",
        max_tokens=64000,
        messages=[{"role": "user", "content": "Write a story"}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
except anthropic.APIConnectionError:
    print("\\nConnection lost. Please retry.")
except anthropic.RateLimitError:
    print("\\nRate limited. Please wait and retry.")
except anthropic.APIStatusError as e:
    print(f"\\nAPI error: {e.status_code}")
\`\`\`

---

## Stream Event Types

| Event Type            | Description                 | When it fires                     |
| --------------------- | --------------------------- | --------------------------------- |
| \`message_start\`       | Contains message metadata   | Once at the beginning             |
| \`content_block_start\` | New content block beginning | When a text/tool_use block starts |
| \`content_block_delta\` | Incremental content update  | For each token/chunk              |
| \`content_block_stop\`  | Content block complete      | When a block finishes             |
| \`message_delta\`       | Message-level updates       | Contains \`stop_reason\`, usage     |
| \`message_stop\`        | Message complete            | Once at the end                   |

## Best Practices

1. **Always flush output** — Use \`flush=True\` to show tokens immediately
2. **Handle partial responses** — If the stream is interrupted, you may have incomplete content
3. **Track token usage** — The \`message_delta\` event contains usage information
4. **Use timeouts** — Set appropriate timeouts for your application
5. **Default to streaming** — Use \`.get_final_message()\` to get the complete response even when streaming, giving you timeout protection without needing to handle individual events
`

for await (const event of stream) {
  if (
    event.type === "content_block_delta" &&
    event.delta.type === "text_delta"
  ) {
    process.stdout.write(event.delta.text);
  }
}
\`\`\`

---

## Handling Different Content Types

> **Opus 4.6:** Use \`thinking: {type: "adaptive"}\`. On older models, use \`thinking: {type: "enabled", budget_tokens: N}\` instead.

\`\`\`typescript
const stream = client.messages.stream({
  model: "{{OPUS_ID}}",
  max_tokens: 64000,
  thinking: { type: "adaptive" },
  messages: [{ role: "user", content: "Analyze this problem" }],
});

for await (const event of stream) {
  switch (event.type) {
    case "content_block_start":
      switch (event.content_block.type) {
        case "thinking":
          console.log("\\n[Thinking...]");
          break;
        case "text":
          console.log("\\n[Response:]");
          break;
      }
      break;
    case "content_block_delta":
      switch (event.delta.type) {
        case "thinking_delta":
          process.stdout.write(event.delta.thinking);
          break;
        case "text_delta":
          process.stdout.write(event.delta.text);
          break;
      }
      break;
  }
}
\`\`\`

---

## Streaming with Tool Use (Tool Runner)

Use the tool runner with \`stream: true\`. The outer loop iterates over tool runner iterations (messages), the inner loop processes stream events:

\`\`\`typescript
import Anthropic from "@anthropic-ai/sdk";
import { betaZodTool } from "@anthropic-ai/sdk/helpers/beta/zod";
import { z } from "zod";

const client = new Anthropic();

const getWeather = betaZodTool({
  name: "get_weather",
  description: "Get current weather for a location",
  inputSchema: z.object({
    location: z.string().describe("City and state, e.g., San Francisco, CA"),
  }),
  run: async ({ location }) => \`72°F and sunny in \${location}\`,
})

const runner = client.beta.messages.toolRunner({
  model: "{{OPUS_ID}}",
  max_tokens: 64000,
  tools: [getWeather],
  messages: [
    { role: "user", content: "What's the weather in Paris and London?" },
  ],
  stream: true,
});

// Outer loop: each tool runner iteration
for await (const messageStream of runner) {
  // Inner loop: stream events for this iteration
  for await (const event of messageStream) {
    switch (event.type) {
      case "content_block_delta":
        switch (event.delta.type) {
          case "text_delta":
            process.stdout.write(event.delta.text);
            break;
          case "input_json_delta":
            // Tool input being streamed
            break;
        }
        break;
    }
  }
}
\`\`\`

---

## Getting the Final Message

\`\`\`typescript
const stream = client.messages.stream({
  model: "{{OPUS_ID}}",
  max_tokens: 64000,
  messages: [{ role: "user", content: "Hello" }],
});

for await (const event of stream) {
  // Process events...
}

const finalMessage = await stream.finalMessage();
console.log(\`Tokens used: \${finalMessage.usage.output_tokens}\`)

\`\`\`

---

## Stream Event Types

| Event Type            | Description                 | When it fires                     |
| --------------------- | --------------------------- | --------------------------------- |
| \`message_start\`       | Contains message metadata   | Once at the beginning             |
| \`content_block_start\` | New content block beginning | When a text/tool_use block starts |
| \`content_block_delta\` | Incremental content update  | For each token/chunk              |
| \`content_block_stop\`  | Content block complete      | When a block finishes             |
| \`message_delta\`       | Message-level updates       | Contains \`stop_reason\`, usage     |
| \`message_stop\`        | Message complete            | Once at the end                   |

## Best Practices

1. **Always flush output** — Use \`process.stdout.write()\` for immediate display
2. **Handle partial responses** — If the stream is interrupted, you may have incomplete content
3. **Track token usage** — The \`message_delta\` event contains usage information
4. **Use \`finalMessage()\`** — Get the complete \`Anthropic.Message\` object even when streaming. Don't wrap \`.on()\` events in \`new Promise()\` — \`finalMessage()\` handles all completion/error/abort states internally
5. **Buffer for web UIs** — Consider buffering a few tokens before rendering to avoid excessive DOM updates
6. **Use \`stream.on("text", ...)\` for deltas** — The \`text\` event provides just the delta string, simpler than manually filtering \`content_block_delta\` events
7. **For agentic loops with streaming** — See the [Streaming Manual Loop](./tool-use.md#streaming-manual-loop) section in tool-use.md for combining \`stream()\` + \`finalMessage()\` with a tool-use loop

## Raw SSE Format

If using raw HTTP (not SDKs), the stream returns Server-Sent Events:

\`\`\`
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","type":"message",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}

event: message_stop
data: {"type":"message_stop"}
\`\`\`
`

break;case"stream_event":if(b6.event.type==="message_start")x6=wf,x6=x56(x6,b6.event.message.usage);if(b6.event.type==="message_delta"){if(x6=x56(x6,b6.event.usage),b6.event.delta.stop_reason!=null)I6=b6.event.delta.stop_reason}if(b6.event.type==="message_stop")this.totalUsage=Ug8(this.totalUsage,x6);if(v)yield{type:"stream_event",event:b6.event,session_id:N8(),parent_tool_use_id:null,uuid:pc()};break;case"attachment":if(this.mutableMessages.push(b6),R)e.push(b6),l();if(b6.attachment.type==="structured_output")N6=b6.attachment.data;else if(b6.attachment.type==="hook_deferred_tool")Z6={id:b6.attachment.toolUseID,name:b6.attachment.toolName,input:b6.attachment.toolInput};else if(b6.attachment.type==="max_turns_reached"){X8={turnCount:b6.attachment.turnCount,maxTurns:b6.attachment.maxTurns};continue}else if(Z&&b6.attachment.type==="queued_command"){let T6=b6.attachment;yield{type:"user",message:{role:"user",content:T6.prompt},session_id:N8(),parent_tool_use_id:null,uuid:T6.source_uuid||b6.uuid,timestamp:b6.timestamp,isReplay:!0,...T6.fileAttachments?.length&&{file_attachments:T6.fileAttachments}}}break;case"stream_request_start":break;case"system":{let T6=this.config.snipReplay?.(b6,this.mutableMessages);if(T6!==void 0){if(T6.executed)this.mutableMessages.length=0,this.mutableMessages.push(...T6.messages);break}if(this.mutableMessages.push(b6),b6.subtype==="compact_boundary"&&b6.compactMetadata){let s=this.mutableMessages.length-1;if(s>0)this.mutableMessages.splice(0,s);let $6=e.length-1;if($6>0)e.splice(0,$6),a=e.length;yield{type:"system",subtype:"compact_boundary",session_id:N8(),uuid:b6.uuid,compact_metadata:bd8(b6.compactMetadata)}}if(b6.subtype==="api_error")yield{type:"system",subtype:"api_retry",attempt:b6.retryAttempt,max_retries:b6.maxRetries,retry_delay_ms:b6.retryInMs,error_status:b6.error.status??null,error:PTK(b6.error),session_id:N8(),uuid:b6.uuid};break}case"tool_use_summary":yield{type:"tool_use_summary",summary:b6.summary,preceding_tool_use_ids:b6.precedingToolUseIds,session_id:N8(),uuid:b6.uuid}