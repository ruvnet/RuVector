return{type:"tool_result",tool_use_id:Y.id,content:A}}catch(O){return{type:"tool_result",tool_use_id:Y.id,content:O instanceof NP6?O.content:`Error: ${O instanceof Error?O.message:String(O)}`,is_error:!0}}}))}}var dI6,yP6,l96,EP,cI6,yE,vl,ke,lI6,nM7,As8,nI6;var ws8=L(()=>{Dl();E_8();FW();NE();xI6();nI6=class nI6{constructor(q,K,_){dI6.add(this),this.client=q,yP6.set(this,!1),l96.set(this,!1),EP.set(this,void 0),cI6.set(this,void 0),yE.set(this,void 0),vl.set(this,void 0),ke.set(this,void 0),lI6.set(this,0),J4(this,EP,{params:{...K,messages:structuredClone(K.messages)}},"f");let Y=["BetaToolRunner",...oa8(K.tools,K.messages)].join(", ");J4(this,cI6,{..._,headers:x3([{"x-stainless-helper":Y},_?.headers])},"f"),J4(this,ke,iM7(),"f")}async*[(yP6=new WeakMap,l96=new WeakMap,EP=new WeakMap,cI6=new WeakMap,yE=new WeakMap,vl=new WeakMap,ke=new WeakMap,lI6=new WeakMap,dI6=new WeakSet,nM7=async function(){let K=x1(this,EP,"f").params.compactionControl;if(!K||!K.enabled)return!1;let _=0;if(x1(this,yE,"f")!==void 0)try{let w=await x1(this,yE,"f");_=w.usage.input_tokens+(w.usage.cache_creation_input_tokens??0)+(w.usage.cache_read_input_tokens??0)+w.usage.output_tokens}catch{return!1}let z=K.contextTokenThreshold??cM7;if(_<z)return!1;let Y=K.model??x1(this,EP,"f").params.model,$=K.summaryPrompt??lM7,O=x1(this,EP,"f").params.messages;if(O[O.length-1].role==="assistant"){let w=O[O.length-1];if(Array.isArray(w.content)){let j=w.content.filter((H)=>H.type!=="tool_use");if(j.length===0)O.pop();else w.content=j}}let A=await this.client.beta.messages.create({model:Y,messages:[...O,{role:"user",content:[{type:"text",text:$}]}],max_tokens:x1(this,EP,"f").params.max_tokens},{headers:{"x-stainless-helper":"compaction"}});if(A.content[0]?.type!=="text")throw new mq("Expected text response for compaction");return x1(this,EP,"f").params.messages=[{role:"user",content:A.content}],!0},Symbol.asyncIterator)](){var q;if(x1(this,yP6,"f"))throw new mq("Cannot iterate over a consumed stream")

J4(this,yP6,!0,"f"),J4(this,l96,!0,"f"),J4(this,vl,void 0,"f");try{while(!0){let K;try{if(x1(this,EP,"f").params.max_iterations&&x1(this,lI6,"f")>=x1(this,EP,"f").params.max_iterations)break;J4(this,l96,!1,"f"),J4(this,vl,void 0,"f"),J4(this,lI6,(q=x1(this,lI6,"f"),q++,q),"f"),J4(this,yE,void 0,"f");let{max_iterations:_,compactionControl:z,...Y}=x1(this,EP,"f").params;if(Y.stream)K=this.client.beta.messages.stream({...Y},x1(this,cI6,"f")),J4(this,yE,K.finalMessage(),"f"),x1(this,yE,"f").catch(()=>{}),yield K;else J4(this,yE,this.client.beta.messages.create({...Y,stream:!1},x1(this,cI6,"f")),"f"),yield x1(this,yE,"f");if(!await x1(this,dI6,"m",nM7).call(this)){if(!x1(this,l96,"f")){let{role:A,content:w}=await x1(this,yE,"f");x1(this,EP,"f").params.messages.push({role:A,content:w})}let O=await x1(this,dI6,"m",As8).call(this,x1(this,EP,"f").params.messages.at(-1));if(O)x1(this,EP,"f").params.messages.push(O);else if(!x1(this,l96,"f"))break}}finally{if(K)K.abort()}}if(!x1(this,yE,"f"))throw new mq("ToolRunner concluded without a message from the server");x1(this,ke,"f").resolve(await x1(this,yE,"f"))}catch(K){throw J4(this,yP6,!1,"f"),x1(this,ke,"f").promise.catch(()=>{}),x1(this,ke,"f").reject(K),J4(this,ke,iM7(),"f"),K}}setMessagesParams(q){if(typeof q==="function")x1(this,EP,"f").params=q(x1(this,EP,"f").params);else x1(this,EP,"f").params=q;J4(this,l96,!0,"f"),J4(this,vl,void 0,"f")}async generateToolResponse(){let q=await x1(this,yE,"f")??this.params.messages.at(-1);if(!q)return null;return x1(this,dI6,"m",As8).call(this,q)}done(){return x1(this,ke,"f").promise}async runUntilDone(){if(!x1(this,yP6,"f"))for await(let q of this);return this.done()}get params(){return x1(this,EP,"f").params}pushMessages(...q){this.setMessagesParams((K)=>({...K,messages:[...K.messages,...q]}))}then(q,K){return this.runUntilDone().then(q,K)}};As8=async function(K){if(x1(this,vl,"f")!==void 0)return x1(this,vl,"f");return J4(this,vl,JA5(x1(this,EP,"f").params,K),"f"),x1(this,vl,"f")}});var EP6;var js8=L(()=>{FW();Ba8()

return K.split(`
`).filter((_)=>_.startsWith("worktree ")).map((_)=>_.slice(9).normalize("NFC"))}catch{return[]}}var gR5;var wA8=L(()=>{gR5=BR5(pR5)});import{open as Ch7,readdir as zm$,realpath as Ym$,stat as $m$}from"fs/promises";import{join as bh7}from"path";function xh7(q){if(typeof q!=="string")return null;return FR5.test(q)?q:null}function Ih7(q){if(!q.includes("\\"))return q;try{return JSON.parse(`"${q}"`)}catch{return q}}function P66(q,K){let _=[`"${K}":"`,`"${K}": "`];for(let z of _){let Y=q.indexOf(z);if(Y<0)continue;let $=Y+z.length,O=$;while(O<q.length){if(q[O]==="\\"){O+=2;continue}if(q[O]==='"')return Ih7(q.slice($,O));O++}}return}function OT(q,K){let _=[`"${K}":"`,`"${K}": "`],z;for(let Y of _){let $=0;while(!0){let O=q.indexOf(Y,$);if(O<0)break;let A=O+Y.length,w=A;while(w<q.length){if(q[w]==="\\"){w+=2;continue}if(q[w]==='"'){z=Ih7(q.slice(A,w));break}w++}$=w+1}}return z}async function uh7(q,K,_){try{let z=await Ch7(q,"r");try{let Y=await z.read(_,0,X66,0);if(Y.bytesRead===0)return{head:"",tail:""};let $=_.toString("utf8",0,Y.bytesRead),O=Math.max(0,K-X66),A=$;if(O>0){let w=await z.read(_,0,X66,O);A=_.toString("utf8",0,w.bytesRead)}return{head:$,tail:A}}finally{await z.close()}}catch{return{head:"",tail:""}}}function UR5(q){return Math.abs(Q_6(q)).toString(36)}function XX(q){let K=q.replace(/[^a-zA-Z0-9]/g,"-");if(K.length<=_51)return K;let _=typeof Bun<"u"?Bun.hash(q).toString(36):UR5(q);return`${K.slice(0,_51)}-${_}`}function JA8(){return bh7(q7(),"projects")}function mh7(q){return bh7(JA8(),XX(q))}function cR5(){return dR5??=Buffer.from('"compact_boundary"')}function ph7(q){try{let K=JSON.parse(q);if(K.type!=="system"||K.subtype!=="compact_boundary")return null;return{hasPreservedSegment:Boolean(K.compactMetadata?.preservedSegment)}}catch{return null}}function i_6(q,K,_,z){let Y=z-_;if(Y<=0)return;if(q.len+Y>q.buf.length){let $=Buffer.allocUnsafe(Math.min(Math.max(q.buf.length*2,q.len+Y),q.cap))

CP_=new Set(["chrome_bridge_connection_succeeded","chrome_bridge_connection_failed","chrome_bridge_disconnected","chrome_bridge_tool_call_completed","chrome_bridge_tool_call_error","chrome_bridge_tool_call_started","chrome_bridge_tool_call_timeout","tengu_api_error","tengu_api_success","tengu_brief_mode_enabled","tengu_brief_mode_toggled","tengu_brief_send","tengu_cancel","tengu_compact_failed","tengu_exit","tengu_flicker","tengu_init","tengu_model_fallback_triggered","tengu_oauth_error","tengu_oauth_success","tengu_oauth_token_refresh_failure","tengu_oauth_token_refresh_success","tengu_oauth_token_refresh_lock_acquiring","tengu_oauth_token_refresh_lock_acquired","tengu_oauth_token_refresh_starting","tengu_oauth_token_refresh_completed","tengu_oauth_token_refresh_lock_releasing","tengu_oauth_token_refresh_lock_released","tengu_query_error","tengu_session_file_read","tengu_started","tengu_tool_use_error","tengu_tool_use_granted_in_prompt_permanent","tengu_tool_use_granted_in_prompt_temporary","tengu_tool_use_rejected_in_prompt","tengu_tool_use_success","tengu_uncaught_exception","tengu_unhandled_rejection","tengu_voice_recording_started","tengu_voice_toggled","tengu_team_mem_sync_pull","tengu_team_mem_sync_push","tengu_team_mem_sync_started","tengu_team_mem_entries_capped"]),bP_=["arch","clientType","errorType","http_status_range","http_status","kairosActive","model","platform","provider","skillMode","subscriptionType","toolName","userBucket","userType","version","versionBase"];tl6=[];IP_=$1(async()=>{if(c16())return S08=!1,!1;try{return S08=!0,!0}catch(q){return j6(q),S08=!1,!1}});mP_=$1(()=>{let q=qC(),K=yP_("sha256").update(q).digest("hex");return parseInt(K.slice(0,8),16)%uP_})});function BP_(){if(vv6("datadog"))return!1;if(yE1!==void 0)return yE1;try{return J$(Esq)}catch{return!1}}function Lsq(q,K){let _=qE1(q);if(_===0)return;let z=_!==null?{...K,sample_rate:_}:K;if(BP_())ysq(q,q_8(z))

if(!_)return null;let{gapMinutes:z,config:Y}=_,$=wi_(q),O=Math.max(1,Y.keepRecent),A=new Set($.slice(-O)),w=new Set($.filter((J)=>!A.has(J)));if(w.size===0)return null;let j=0,H=q.map((J)=>{if(J.type!=="user"||!Array.isArray(J.message.content))return J;let M=!1,X=J.message.content.map((P)=>{if(P.type==="tool_result"&&w.has(P.tool_use_id)&&P.content!==jM4)return j+=WM4(P),M=!0,{...P,content:jM4};return P});if(!M)return J;return{...J,message:{...J.message,content:X}}});if(j===0)return null;return d("tengu_time_based_microcompact",{gapMinutes:Math.round(z),gapThresholdMinutes:Y.gapThresholdMinutes,toolsCleared:w.size,toolsKept:A.size,keepRecent:Y.keepRecent,tokensSaved:j}),N(`[TIME-BASED MC] gap ${Math.round(z)}min > ${Y.gapThresholdMinutes}min, cleared ${w.size} tool results (~${j} tokens), kept last ${A.size}`),TV6(),_o(),{messages:H}}var jM4="[Old tool result content cleared]",JM4=2000,Ai_,HM4=null,kV6=null,lu1=null;var aC=L(()=>{ZY();Y2();bX();PV6();_8();dq();Vq6();r8();k8();yq6();UN();QN8();wM4();Ai_=new Set([pq,...Mw6,$9,Z_,gL,mj,N4,xK])});function DM4(q){let K=Buffer.from(q,"base64"),_="";for(let z of K)_+=String.fromCharCode(z^Mi_);return _.split(",")}function fi_(){let q=process.env.ANTHROPIC_BASE_URL;if(!q)return null;try{return new URL(q).hostname.toLowerCase()}catch{return null}}function Zi_(){if(OM())return null;let q=fi_(),K=Hu6(),_=K==="Asia/Shanghai"||K==="Asia/Urumqi";if(!q)return{known:!1,labKw:!1,cnTZ:_,host:null};return{known:Wi_().some((z)=>q===z||q.endsWith("."+z)),labKw:Di_().some((z)=>q.includes(z)),cnTZ:_,host:q}}function Gi_(q,K){if(!q&&!K)return"'";if(q&&!K)return"’";if(!q&&K)return"ʼ";return"ʹ"}function fM4(q){let K=Zi_(),_=Gi_(K?.known??!1,K?.labKw??!1),z=K?.cnTZ?q.replace(/-/g,"/"):q

$.set(w,{count:j.count+1,totalTokens:j.totalTokens+O})}}}break}case"image":case"server_tool_use":case"web_search_tool_result":case"search_result":case"document":case"thinking":case"redacted_thinking":case"code_execution_tool_result":case"mcp_tool_use":case"mcp_tool_result":case"container_upload":case"web_fetch_tool_result":case"bash_code_execution_tool_result":case"text_editor_code_execution_tool_result":case"tool_search_tool_result":case"compaction":_.other+=O;break}}function _P4(q,K,_){q.set(K,(q.get(K)||0)+_)}function YP4(q){let K={total_tokens:q.total,human_message_tokens:q.humanMessages,assistant_message_tokens:q.assistantMessages,local_command_output_tokens:q.localCommandOutputs,other_tokens:q.other};q.attachments.forEach((z,Y)=>{K[`attachment_${Y}_count`]=z}),q.toolRequests.forEach((z,Y)=>{K[`tool_request_${Y}_tokens`]=z}),q.toolResults.forEach((z,Y)=>{K[`tool_result_${Y}_tokens`]=z});let _=[...q.duplicateFileReads.values()].reduce((z,Y)=>z+Y.tokens,0);if(K.duplicate_read_tokens=_,K.duplicate_read_file_count=q.duplicateFileReads.size,q.total>0){K.human_message_percent=Math.round(q.humanMessages/q.total*100),K.assistant_message_percent=Math.round(q.assistantMessages/q.total*100),K.local_command_output_percent=Math.round(q.localCommandOutputs/q.total*100),K.duplicate_read_percent=Math.round(_/q.total*100);let z=[...q.toolRequests.values()].reduce(($,O)=>$+O,0),Y=[...q.toolResults.values()].reduce(($,O)=>$+O,0);K.tool_request_percent=Math.round(z/q.total*100),K.tool_result_percent=Math.round(Y/q.total*100),q.toolRequests.forEach(($,O)=>{K[`tool_request_${O}_percent`]=Math.round($/q.total*100)}),q.toolResults.forEach(($,O)=>{K[`tool_result_${O}_percent`]=Math.round($/q.total*100)})}return K}var $P4=L(()=>{UN();a1();r8()});function km1(q){if(aZ6())return fOq(q);return q}function bV6(q){return q||aZ6()}function OP4(q){return aZ6()&&q.status===429}var io6=L(()=>{ov();pX1()});function xq6(q){if(!q||typeof q!=="object")return null;let K=q,_=5,z=0

if(K[47]!==$||K[48]!==V||K[49]!==W||K[50]!==k||K[51]!==z||K[52]!==_||K[53]!==H||K[54]!==X||K[55]!==O||K[56]!==w)R=_.message.content.map((p,C)=>j9.createElement(wlz,{key:C,message:_,addMargin:$,tools:O,progressMessagesForMessage:H,param:p,style:X,verbose:w,imageIndex:V[C],isUserContinuation:k,lookups:z,isTranscriptMode:W})),K[47]=$,K[48]=V,K[49]=W,K[50]=k,K[51]=z,K[52]=_,K[53]=H,K[54]=X,K[55]=O,K[56]=w,K[57]=R;else R=K[57];let b;if(K[58]!==E||K[59]!==R)b=j9.createElement(u,{flexDirection:"column",width:E},R),K[58]=E,K[59]=R,K[60]=b;else b=K[60];let I=b,m;if(K[61]!==I||K[62]!==y)m=y?j9.createElement(TS4,null,I):I,K[61]=I,K[62]=y,K[63]=m;else m=K[63];return m}case"system":{if(_.subtype==="compact_boundary"){if(E4())return null;let y;if(K[64]===Symbol.for("react.memo_cache_sentinel"))y=j9.createElement(s3K,null),K[64]=y;else y=K[64];return y}if(_.subtype==="microcompact_boundary")return null;if(_.subtype==="local_command"){let y;if(K[68]!==_.content)y={type:"text",text:_.content},K[68]=_.content,K[69]=y;else y=K[69];let E;if(K[70]!==$||K[71]!==W||K[72]!==y||K[73]!==w)E=j9.createElement(rj6,{addMargin:$,param:y,verbose:w,isTranscriptMode:W}),K[70]=$,K[71]=W,K[72]=y,K[73]=w,K[74]=E;else E=K[74];return E}let V;if(K[75]!==$||K[76]!==W||K[77]!==_||K[78]!==w)V=j9.createElement(A9K,{message:_,addMargin:$,verbose:w,isTranscriptMode:W}),K[75]=$,K[76]=W,K[77]=_,K[78]=w,K[79]=V;else V=K[79];return V}case"grouped_tool_use":{let V;if(K[80]!==j||K[81]!==z||K[82]!==_||K[83]!==J||K[84]!==O)V=j9.createElement(e3K,{message:_,tools:O,lookups:z,inProgressToolUseIDs:j,shouldAnimate:J}),K[80]=j,K[81]=z,K[82]=_,K[83]=J,K[84]=O,K[85]=V;else V=K[85];return V}case"collapsed_read_search":{let V=w||W,y;if(K[86]!==j||K[87]!==f||K[88]!==z||K[89]!==_||K[90]!==J||K[91]!==V||K[92]!==O)y=j9.createElement(A0,null,j9.createElement(o3K,{message:_,inProgressToolUseIDs:j,shouldAnimate:J,verbose:V,tools:O,lookups:z,isActiveGroup:f})),K[86]=j,K[87]=f,K[88]=z,K[89]=_,K[90]=J,K[91]=V,K[92]=O,K[93]=y;else y=K[93]

try{return await Rh4([K],Y)}catch($){return j6($),null}}function F_Y(q){let K=0;for(let _ of q){if(_.type!=="assistant")continue;let z=_.message?.content;if(!Array.isArray(z))continue;for(let Y of z){if(Y.type!=="tool_use"||!g_Y.has(Y.name))continue;if(u17(Y.name,Y.input))K++}}return K}async function U_Y(){try{let q=kY(),K=(await u_Y(q)).size,z=(await Bh7(q,K)).postBoundaryBuf,Y=zn(z),$=Y.findLastIndex((A)=>A.type==="system"&&("subtype"in A)&&A.subtype==="compact_boundary"),O=$>=0?Y.slice($+1):Y;return{promptCount:p_Y(O),memoryAccessCount:F_Y(O)}}catch{return{promptCount:0,memoryAccessCount:0}}}async function OfK(q){if(JP6()==="remote"){let X=process.env.CLAUDE_CODE_REMOTE_SESSION_ID;if(X){let P=process.env.SESSION_INGRESS_URL;if(!cL8(X,P))return VM(X,P)}return""}let K=v7();if(K.attribution?.pr)return K.attribution.pr;if(K.includeCoAuthoredBy===!1)return"";let _=`\uD83E\uDD16 Generated with [Claude Code](${_26})`,z=q();if(N(`PR Attribution: appState.attribution exists: ${!!z.attribution}`),z.attribution){let X=z.attribution.fileStates,W=X instanceof Map?X.size:Object.keys(X).length;N(`PR Attribution: fileStates count: ${W}`)}let[Y,{promptCount:$,memoryAccessCount:O},A]=await Promise.all([B_Y(z),U_Y(),Lh4()]),w=Y?.summary.claudePercent??0;N(`PR Attribution: claudePercent: ${w}, promptCount: ${$}, memoryAccessCount: ${O}`);let j=cY(D5()),H=A?j:hh4(j);if(w===0&&$===0&&O===0)return N("PR Attribution: returning default (no data)"),_;let J=O>0?`, ${O} ${O===1?"memory":"memories"} recalled`:"",M=`\uD83E\uDD16 Generated with [Claude Code](${_26}) (${w}% ${$}-shotted by ${H}${J})`;return N(`PR Attribution: returning summary: ${M}`),M}var g_Y;var Mg8=L(()=>{T8();O$();ZY();Y2();bX();No();_8();mA();h8();dq();m17();t4();W66();i1();dH6();g_Y=new Set([pq,$9,Z_,N4,xK])});function Xg8(){return Ch6()}function Pg8(){return sm8()}function d_Y(){if(c6(process.env.CLAUDE_CODE_DISABLE_BACKGROUND_TASKS))return null

let _=[];_.push("=".repeat(80)),_.push(`QUERY PROFILING REPORT - Query #${sfK}`),_.push("=".repeat(80)),_.push("");let z=K[0]?.startTime??0,Y=z,$=0,O=0;for(let j of K){let H=j.startTime-z,J=j.startTime-Y;if(_.push(Kz8(H,J,j.name,j77.get(j.name),10,9,lzY(J,j.name))),j.name==="query_api_request_sent")$=H;if(j.name==="query_first_chunk_received")O=H;Y=j.startTime}let A=K[K.length-1],w=A?A.startTime-z:0;if(_.push(""),_.push("-".repeat(80)),O>0){let j=$,H=O-$,J=(j/O*100).toFixed(1),M=(H/O*100).toFixed(1);_.push(`Total TTFT: ${YI(O)}ms`),_.push(`  - Pre-request overhead: ${YI(j)}ms (${J}%)`),_.push(`  - Network latency: ${YI(H)}ms (${M}%)`)}else _.push(`Total time: ${YI(w)}ms`);return _.push(izY(K,z)),_.push("=".repeat(80)),_.join(`
`)}function izY(q,K){let _=[{name:"Context loading",start:"query_context_loading_start",end:"query_context_loading_end"},{name:"Microcompact",start:"query_microcompact_start",end:"query_microcompact_end"},{name:"Autocompact",start:"query_autocompact_start",end:"query_autocompact_end"},{name:"Query setup",start:"query_setup_start",end:"query_setup_end"},{name:"Tool schemas",start:"query_tool_schema_build_start",end:"query_tool_schema_build_end"},{name:"Message normalization",start:"query_message_normalization_start",end:"query_message_normalization_end"},{name:"Client creation",start:"query_client_creation_start",end:"query_client_creation_end"},{name:"Network TTFB",start:"query_api_request_sent",end:"query_first_chunk_received"},{name:"Tool execution",start:"query_tool_execution_start",end:"query_tool_execution_end"}],z=new Map(q.map((O)=>[O.name,O.startTime-K])),Y=[];Y.push(""),Y.push("PHASE BREAKDOWN:");for(let O of _){let A=z.get(O.start),w=z.get(O.end);if(A!==void 0&&w!==void 0){let j=w-A,H="█".repeat(Math.min(Math.ceil(j/10),50));Y.push(`  ${O.name.padEnd(22)} ${YI(j).padStart(10)}ms ${H}`)}}let $=z.get("query_api_request_sent");if($!==void 0)Y.push(""),Y.push(`  ${"Total pre-API overhead".padEnd(22)} ${YI($).padStart(10)}ms`);return Y.join(`
`)}function Eg8(){if(!b78)return

let F=[...e2(k)],U=V,c=A.startsWith("agent:")||A.startsWith("repl_main_thread");F=await Wb4(F,v.contentReplacementState,c?(T6)=>void aH6(T6,v.agentId).catch(j6):void 0,new Set(v.options.tools.filter((T6)=>!Number.isFinite(T6.maxResultSizeChars)).map((T6)=>T6.name)));let K6=0;g3("query_microcompact_start"),F=(await H.microcompact(F,v,A)).messages;let q6=void 0;g3("query_microcompact_end");let t=tK(IZK(_,Y));g3("query_autocompact_start");let{compactionResult:n,consecutiveFailures:z6,consecutiveRapidRefills:M6,rapidRefillBreakerTripped:J6}=await H.autocompact(F,v,{systemPrompt:_,userContext:z,systemContext:Y,toolUseContext:v,forkContextMessages:F},A,U,K6);if(g3("query_autocompact_end"),J6)return d("tengu_auto_compact_rapid_refill_breaker",{consecutiveRapidRefills:U?.consecutiveRapidRefills??0,turnsSincePreviousCompact:U?.turnCounter??-1,queryChainId:g,queryDepth:C.depth}),yield U9({content:SZK,error:"invalid_request"}),{reason:"rapid_refill_breaker"};if(n){let{preCompactTokenCount:T6,postCompactTokenCount:s,truePostCompactTokenCount:$6,compactionUsage:h6}=n;if(d("tengu_auto_compact_succeeded",{originalMessageCount:k.length,compactedMessageCount:n.summaryMessages.length+n.attachments.length+n.hookResults.length,preCompactTokenCount:T6,postCompactTokenCount:s,truePostCompactTokenCount:$6,compactionInputTokens:h6?.input_tokens,compactionOutputTokens:h6?.output_tokens,compactionCacheReadTokens:h6?.cache_read_input_tokens??0,compactionCacheCreationTokens:h6?.cache_creation_input_tokens??0,compactionTotalTokens:h6?h6.input_tokens+(h6.cache_creation_input_tokens??0)+(h6.cache_read_input_tokens??0)+h6.output_tokens:0,queryChainId:g,queryDepth:C.depth}),q.taskBudget){let V6=qy8(F);X=Math.max(0,(X??q.taskBudget.total)-V6)}U={compacted:!0,turnId:H.uuid(),turnCounter:0,consecutiveFailures:0,consecutiveRapidRefills:M6};let P6=xa(n);for(let V6 of P6)yield V6;F=P6}else if(z6!==void 0)U={...U??{compacted:!1,turnId:"",turnCounter:0},consecutiveFailures:z6};v={...v,messages:F,turnStartIndex:LYY(F)};let G6=[],H6=[],e=[],a=!1

if(q&&q.trim()!=="")K+=`

Additional Instructions:
${q}`;return K+=oZK,K}function gYY(q){let K=q;K=K.replace(/<analysis>[\s\S]*?<\/analysis>/,"");let _=K.match(/<summary>([\s\S]*?)<\/summary>/);if(_){let z=_[1]||"";K=K.replace(/<summary>[\s\S]*?<\/summary>/,`Summary:
${z.trim()}`)}return K=K.replace(/\n\n+/g,`

`),K.trim()}function B78(q,K,_,z,Y){let O=`This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

${gYY(q)}`;if(_)O+=`

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: ${_}`;if(z)O+=`

Recent messages are preserved verbatim.`;if(K)return`${O}
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.`;return O}var BYY,oZK

let M=L8("tengu_compact_cache_prefix",!0),X=sZK(Y),P=n8({content:X}),W=q,D=_,f,G,Z=0;for(;;){if(f=await O0K({messages:W,summaryRequest:P,appState:j,context:K,preCompactTokenCount:w,cacheSafeParams:D,stripNonEssential:A}),G=KS6(f),!G?.startsWith(Dp))break;Z++;let n=Z<=K0K?_0K(W,f):null;if(!n)throw d("tengu_compact_failed",{reason:"prompt_too_long",preCompactTokenCount:w,promptCacheSharingEnabled:M,ptlAttempts:Z}),Error(z0K);d("tengu_compact_ptl_retry",{attempt:Z,droppedMessages:W.length-n.length,remainingMessages:n.length}),W=n,D={...D,forkContextMessages:n}}if(!G)throw N(`Compact failed: no summary text in response. Response: ${g6(f)}`,{level:"error"}),d("tengu_compact_failed",{reason:"no_summary",preCompactTokenCount:w,promptCacheSharingEnabled:M}),Error("Failed to generate conversation summary - response did not contain valid text content");else if(Ba(G))throw d("tengu_compact_failed",{reason:"api_error",preCompactTokenCount:w,promptCacheSharingEnabled:M}),Error(G);let v=Am1(K.readFileState);K.readFileState.clear(),K.loadedNestedMemoryPaths?.clear(),lo6(K.memorySelector);let[k,V]=await Promise.all([A0K(v,K,q0K),H0K(K)]),y=[...k,...V],E=dg8(K.agentId);if(E)y.push(E);let R=await j0K(K);if(R)y.push(R);let b=w0K(K.agentId);if(b)y.push(b);for(let n of cg8(K.options.tools,K.options.mainLoopModel,[],{callSite:"compact_full"}))y.push(P4(n));for(let n of lg8(K,[]))y.push(P4(n));for(let n of ng8(K.options.mcpClients,K.options.tools,K.options.mainLoopModel,[]))y.push(P4(n));K.onCompactProgress?.({type:"hooks_start",hookType:"session_start"});let I=await Kf("compact",{model:K.options.mainLoopModel}),m=F78($?"auto":"manual",w??0,q.at(-1)?.uuid),p=Rd(q);if(p.size>0)m.compactMetadata.preCompactDiscoveredTools=[...p].sort();let C=kY(),F=[n8({content:B78(G,z,C,void 0,!1),isCompactSummary:!0,isVisibleInTranscriptOnly:!0})],U=cN([f]),c=LV6([m,...F,...y,...I]),K6=tC(f),o=O?.querySource??K.options.querySource??"unknown"

d("tengu_compact",{preCompactTokenCount:w,stripNonEssential:A,postCompactTokenCount:U,truePostCompactTokenCount:c,autoCompactThreshold:O?.autoCompactThreshold??-1,willRetriggerNextTurn:O!==void 0&&c>=O.autoCompactThreshold,isAutoCompact:$,querySource:o,queryChainId:K.queryTracking?.chainId??"",queryDepth:K.queryTracking?.depth??-1,isRecompactionInChain:O?.isRecompactionInChain??!1,turnsSincePreviousCompact:O?.turnsSincePreviousCompact??-1,previousCompactTurnId:O?.previousCompactTurnId??"",compactionInputTokens:K6?.input_tokens,compactionOutputTokens:K6?.output_tokens,compactionCacheReadTokens:K6?.cache_read_input_tokens??0,compactionCacheCreationTokens:K6?.cache_creation_input_tokens??0,compactionTotalTokens:K6?K6.input_tokens+(K6.cache_creation_input_tokens??0)+(K6.cache_read_input_tokens??0)+K6.output_tokens:0,promptCacheSharingEnabled:M,...(()=>{try{return YP4(zP4(q))}catch(n){return j6(n),{}}})()}),Je(),ig8(),K.onCompactProgress?.({type:"hooks_start",hookType:"post_compact"});let q6=await rg8({trigger:$?"auto":"manual",compactSummary:G},K.abortController.signal),t=[J,q6.userDisplayMessage].filter(Boolean).join(`
`);return{boundaryMarker:m,summaryMessages:F,attachments:y,hookResults:I,userDisplayMessage:t||void 0,preCompactTokenCount:w,postCompactTokenCount:U,truePostCompactTokenCount:c,compactionUsage:K6}}catch(w){if(!$)$0K(w,K);throw w}finally{K.setStreamMode?.("requesting"),K.setResponseLength?.(()=>0),K.onCompactProgress?.({type:"compact_end"}),K.setSDKStatus?.(null)}}async function Y0K(q,K,_,z,Y,$="from"){try{let O=$==="up_to"?q.slice(0,K):q.slice(K),A=$==="up_to"?q.slice(K).filter((q6)=>q6.type!=="progress"&&!pJ(q6)&&!(q6.type==="user"&&q6.isCompactSummary)):q.slice(0,K).filter((q6)=>q6.type!=="progress");if(O.length===0)throw Error($==="up_to"?"Nothing to summarize before the selected message.":"Nothing to summarize after the selected message.");let w=SZ(q);_.onCompactProgress?.({type:"hooks_start",hookType:"pre_compact"}),_.setSDKStatus?.("compacting")

let j=await _S6({trigger:"manual",customInstructions:null},_.abortController.signal),H;if(j.newCustomInstructions&&Y)H=`${j.newCustomInstructions}

User context: ${Y}`;else if(j.newCustomInstructions)H=j.newCustomInstructions;else if(Y)H=`User context: ${Y}`;_.setStreamMode?.("requesting"),_.setResponseLength?.(()=>0),_.onCompactProgress?.({type:"compact_start"});let J=aZK(H,$),M=n8({content:J}),X={preCompactTokenCount:w,direction:$,messagesSummarized:O.length},P=$==="up_to"?O:q,W=$==="up_to"?{...z,forkContextMessages:O}:z,D,f,G=0;for(;;){if(D=await O0K({messages:P,summaryRequest:M,appState:_.getAppState(),context:_,preCompactTokenCount:w,cacheSafeParams:W}),f=KS6(D),!f?.startsWith(Dp))break;G++;let q6=G<=K0K?_0K(P,D):null;if(!q6)throw d("tengu_partial_compact_failed",{reason:"prompt_too_long",...X,ptlAttempts:G}),Error(z0K);d("tengu_compact_ptl_retry",{attempt:G,droppedMessages:P.length-q6.length,remainingMessages:q6.length,path:"partial"}),P=q6,W={...W,forkContextMessages:q6}}if(!f)throw d("tengu_partial_compact_failed",{reason:"no_summary",...X}),Error("Failed to generate conversation summary - response did not contain valid text content");else if(Ba(f))throw d("tengu_partial_compact_failed",{reason:"api_error",...X}),Error(f);let Z=Am1(_.readFileState);_.readFileState.clear(),_.loadedNestedMemoryPaths?.clear(),lo6(_.memorySelector);let[v,k]=await Promise.all([A0K(Z,_,q0K,A),H0K(_)]),V=[...v,...k],y=dg8(_.agentId);if(y)V.push(y);let E=await j0K(_);if(E)V.push(E);let R=w0K(_.agentId);if(R)V.push(R);for(let q6 of cg8(_.options.tools,_.options.mainLoopModel,A,{callSite:"compact_partial"}))V.push(P4(q6));for(let q6 of lg8(_,A))V.push(P4(q6));for(let q6 of ng8(_.options.mcpClients,_.options.tools,_.options.mainLoopModel,A))V.push(P4(q6));_.onCompactProgress?.({type:"hooks_start",hookType:"session_start"});let b=await Kf("compact",{model:_.options.mainLoopModel}),I=cN([D]),m=tC(D)

d("tengu_partial_compact",{preCompactTokenCount:w,postCompactTokenCount:I,messagesKept:A.length,messagesSummarized:O.length,direction:$,hasUserFeedback:!!Y,trigger:"message_selector",compactionInputTokens:m?.input_tokens,compactionOutputTokens:m?.output_tokens,compactionCacheReadTokens:m?.cache_read_input_tokens??0,compactionCacheCreationTokens:m?.cache_creation_input_tokens??0});let p=$==="up_to"?q.slice(0,K).findLast((q6)=>q6.type!=="progress")?.uuid:A.at(-1)?.uuid,C=F78("manual",w??0,p,Y,O.length),g=Rd(q);if(g.size>0)C.compactMetadata.preCompactDiscoveredTools=[...g].sort();let F=kY(),c=[n8({content:B78(f,!1,F,void 0,!1),isCompactSummary:!0,...A.length>0?{summarizeMetadata:{messagesSummarized:O.length,userContext:Y,direction:$}}:{isVisibleInTranscriptOnly:!0}})];Je(),ig8(),_.onCompactProgress?.({type:"hooks_start",hookType:"post_compact"});let K6=await rg8({trigger:"manual",compactSummary:f},_.abortController.signal),o=$==="up_to"?c.at(-1)?.uuid??C.uuid:C.uuid;return{boundaryMarker:b77(C,o,A),summaryMessages:c,messagesToKeep:A,attachments:V,hookResults:b,userDisplayMessage:K6.userDisplayMessage,preCompactTokenCount:w,postCompactTokenCount:I,compactionUsage:m}}catch(O){throw $0K(O,_),O}finally{_.setStreamMode?.("requesting"),_.setResponseLength?.(()=>0),_.onCompactProgress?.({type:"compact_end"}),_.setSDKStatus?.(null)}}function $0K(q,K){if(!Ee(q,ba)&&!Ee(q,qS6))K.addNotification?.({key:"error-compacting-conversation",text:"Error compacting conversation",priority:"immediate",color:"error"})}function rYY(){return async()=>({behavior:"deny",message:"Tool use is not allowed during compaction",decisionReason:{type:"other",reason:"compaction agent should only produce text summary"}})}async function O0K({messages:q,summaryRequest:K,appState:_,context:z,preCompactTokenCount:Y,cacheSafeParams:$,stripNonEssential:O=!1}){let A=!O&&L8("tengu_compact_cache_prefix",!0),w=pfK()?setInterval((j)=>{mfK(),j?.("compacting")},30000,z.setSDKStatus):void 0

return P4({type:"plan_mode",reminderType:"full",isSubAgent:!!q.agentId,planFilePath:_,planExists:z})}async function H0K(q){let K=q.getAppState();return Object.values(K.tasks).filter((z)=>z.type==="local_agent").flatMap((z)=>{if(z.retrieved||z.status==="pending"||z.agentId===q.agentId)return[];return[P4({type:"task_status",taskId:z.agentId,taskType:"local_agent",description:z.description,status:z.status,deltaSummary:z.status==="running"?z.progress?.summary??null:z.error??null,outputFilePath:aY(z.agentId)})]})}function oYY(q){let K=new Set;for(let z of q){if(z.type!=="user"||!Array.isArray(z.message.content))continue;for(let Y of z.message.content)if(Y.type==="tool_result"&&typeof Y.content==="string"&&l08(Y.content))K.add(Y.tool_use_id)}let _=new Set;for(let z of q){if(z.type!=="assistant"||!Array.isArray(z.message.content))continue;for(let Y of z.message.content){if(Y.type!=="tool_use"||Y.name!==pq||K.has(Y.id))continue;let $=Y.input;if($&&typeof $==="object"&&"file_path"in $&&typeof $.file_path==="string")_.add(Rq($.file_path))}}return _}function aYY(q,K){if(L3(q)<=K)return q;let _=K*4-eZK.length;return q.slice(0,_)+eZK}function sYY(q,K){let _=Rq(q);try{let z=Rq(PW(K));if(_===z)return!0}catch{}try{if(new Set(cZK.map((Y)=>Rq(CO6(Y)))).has(_))return!0}catch{}return!1}var q0K=5,FYY=50000,UYY=5000,QYY=5000,dYY=25000,cYY=2,qS6="Not enough messages to compact.",K0K=3,tZK="[earlier conversation truncated for compaction retry]",z0K="Conversation too long. Press esc twice to go up a few messages and try again.",ba="API Error: Request was aborted.",eR6="Compaction interrupted · This may be due to network issues — please try again.",eZK=`

[... skill content truncated for compaction; use Read on the skill path if you need the full text]`;var Ia=L(()=>{Xm();ov();T8();T8();hd();ZY();lP();$y8();qP();k1();jD();$P4();_8();E8();jk();qv();B$();h8();lZK();a1();i_();lH();nR6();sK6();t4();r8();Fj();CZ();eC();l1();k8();d2();Kb();yq6();$o();S77();UN();C77()});function pp(q){let K=q===void 0||q.startsWith("repl_main_thread")||q==="sdk"

function P0K(){return`IMPORTANT: This message and these instructions are NOT part of the actual user conversation. Do NOT include any references to "note-taking", "session notes extraction", or these update instructions in the notes content.

Based on the user conversation above (EXCLUDING this note-taking instruction message as well as system prompt, claude.md entries, or any past session summaries), update the session notes file.

The file {{notesPath}} has already been read for you. Here are its current contents:
<current_notes_content>
{{currentNotes}}
</current_notes_content>

Your ONLY task is to use the Edit tool to update the notes file, then stop. You can make multiple edits (update every section as needed) - make all Edit tool calls in parallel in a single message. Do not call any other tools.

CRITICAL RULES FOR EDITING:
- The file must maintain its exact structure with all sections, headers, and italic descriptions intact
-- NEVER modify, delete, or add section headers (the lines starting with '#' like # Task specification)
-- NEVER modify or delete the italic _section description_ lines (these are the lines in italics immediately following each header - they start and end with underscores)
-- The italic _section descriptions_ are TEMPLATE INSTRUCTIONS that must be preserved exactly as-is - they guide what content belongs in each section
-- ONLY update the actual content that appears BELOW the italic _section descriptions_ within each existing section
-- Do NOT add any new sections, summaries, or information outside the existing structure
- Do NOT reference this note-taking process or instructions anywhere in the notes
- It's OK to skip updating a section if there are no substantial new insights to add. Do not add filler content like "No info yet", just leave sections blank/unedited if appropriate.
- Write DETAILED, INFO-DENSE content for each section - include specifics like file paths, function names, error messages, exact commands, technical details, etc.
- For "Key results", include the complete, exact output the user requested (e.g., full table, full answer, etc.)
- Do not include information that's already in the CLAUDE.md files included in the context
- Keep each section under ~${og8} tokens/words - if a section is approaching this limit, condense it by cycling out less important details while preserving the most critical information
- Focus on actionable, specific information that would help someone understand or recreate the work discussed in the conversation
- IMPORTANT: Always update "Current State" to reflect the most recent work - this is critical for continuity after compaction

Use the Edit tool with file_path: {{notesPath}}

STRUCTURE PRESERVATION REMINDER:
Each section has TWO parts that must be preserved exactly as they appear in the current file:
1. The section header (line starting with #)
2. The italic description line (the _italicized text_ immediately after the header - this is a template instruction)

You ONLY update the actual content that comes AFTER these two preserved lines. The italic description lines starting and ending with underscores are part of the template structure, NOT content to be edited or removed.

REMEMBER: Use the Edit tool in parallel and stop. Do not continue after the edits. Only include insights from the actual user conversation, never from these note-taking instructions. Do not delete or change section headers or italic _section descriptions_.`}async function m77(){let q=f0K(q7(),"session-memory","config","template.md")

M+=`

Some session memory sections were truncated for length. The full session memory can be viewed at: ${f}`}let X=[n8({content:M,isCompactSummary:!0,isVisibleInTranscriptOnly:!0})],P=dg8($),W=P?[P]:[],D=Vo6(X);return{boundaryMarker:b77(w,X.at(-1).uuid,_),summaryMessages:X,attachments:W,hookResults:z,messagesToKeep:_,preCompactTokenCount:A,postCompactTokenCount:D,truePostCompactTokenCount:D}}async function tg8(q,K,_,z){if(!sg8())return null;await $$Y(),await VX4();let Y=vX4(),$=await _y8();if(!$)return d("tengu_sm_compact_no_session_memory",{}),null;if(await Z0K($))return d("tengu_sm_compact_empty_template",{}),null;try{let O;if(Y){if(O=q.findIndex((P)=>P.uuid===Y),O===-1)return d("tengu_sm_compact_summarized_id_not_found",{}),null}else O=q.length-1,d("tengu_sm_compact_resumed_session",{});let A=w$Y(q,O),w=q.slice(A).filter((P)=>!pJ(P)),j=await Kf("compact",{model:D5()}),H=kY(),J=j$Y(q,$,w,j,H,K,z),M=xa(J),X=Vo6(M);if(_!==void 0&&X>=_)return d("tengu_sm_compact_threshold_exceeded",{postCompactTokenCount:X,autoCompactThreshold:_}),null;return{...J,postCompactTokenCount:X,truePostCompactTokenCount:X}}catch(O){return d("tengu_sm_compact_error",{}),null}}var ag8,g77,T0K=!1;var eg8=L(()=>{_8();d8();E8();a1();dq();Nz();sK6();t4();CZ();eC();l1();k8();p77();SV6();Ia();aC();C77();ag8={minTokens:1e4,minTextBlockMessages:5,maxTokens:40000},g77={...ag8}});function y0K(q){let K=q.trim().toLowerCase(),_;if(K.endsWith("m"))_=parseFloat(K)*1e6;else if(K.endsWith("k"))_=parseFloat(K)*1000;else{let z=parseInt(K,10);_=z>=100&&z<=1000?z*1000:z}if(!Number.isFinite(_)||_<F77||_>N0K)return;return Math.round(_)}function I56(q,K){let _=QT(q,gW());if(process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW){let z=pU("CLAUDE_CODE_AUTO_COMPACT_WINDOW",process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW,F77,N0K);if(z.status!=="invalid"){let Y=Math.max(F77,z.effective);return{window:Math.min(_,Y),configured:Y,source:"env"}}}if(K!==void 0)return{window:Math.min(_,K),configured:K,source:"settings"}

return{window:_,configured:_,source:"model"}}function Sd(q,K){let _=Math.min(U78(q),H$Y),z=f0()?K:void 0,{window:Y}=I56(q,z);return Y-_}function P$Y(){return Date.now()-DR()>=X$Y}function a68(q,K){let _=Sd(q,K),z=_-Q77,Y=process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE;if(Y){let $=parseFloat(Y);if(!isNaN($)&&$>0&&$<=100){let O=Math.floor(_*($/100));return Math.min(O,z)}}return z}function oH6(q,K,_){let z=f0(),Y=z?_:void 0,$=a68(K,Y),O=z?$:Sd(K,Y),A=Math.max(0,Math.round((O-q)/O*100)),w=O-J$Y,j=O-M$Y,H=q>=w,J=q>=j,M=z&&q>=$,P=Sd(K,Y)-d77,W=process.env.CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE,D=W?parseInt(W,10):NaN,f=!isNaN(D)&&D>0?D:P,G=q>=f;return{percentLeft:A,isAboveWarningThreshold:H,isAboveErrorThreshold:J,isAboveAutoCompactThreshold:M,isAtBlockingLimit:G}}function f0(){if(c6(process.env.DISABLE_COMPACT))return!1;if(c6(process.env.DISABLE_AUTO_COMPACT))return!1;return w8().autoCompactEnabled}async function W$Y(q,K,_,z,Y=0){if(z==="session_memory"||z==="compact")return!1;if(!f0())return!1;let $=SZ(q)-Y,O=a68(K,_),A=Sd(K,_);N(`autocompact: tokens=${$} threshold=${O} effectiveWindow=${A}${Y>0?` snipFreed=${Y}`:""}`);let{isAboveAutoCompactThreshold:w}=oH6($,K,_);return w}async function LZK(q,K,_,z,Y,$){if(c6(process.env.DISABLE_COMPACT))return{wasCompacted:!1};if(Y?.consecutiveFailures!==void 0&&Y.consecutiveFailures>=V0K)return{wasCompacted:!1};let O=K.options.mainLoopModel,A=K.getAppState().autoCompactWindow;if(!await W$Y(q,O,A,z,$))return{wasCompacted:!1};let H=Y?.compacted===!0&&Y.turnCounter<U77?(Y?.consecutiveRapidRefills??0)+1:0;if(H>=E0K)return N(`autocompact: rapid-refill breaker tripped — ${H} consecutive refills within <${U77} turns each (last was ${Y?.turnCounter} turns)`,{level:"warn"}),{wasCompacted:!1,rapidRefillBreakerTripped:!0};let J={isRecompactionInChain:Y?.compacted===!0,turnsSincePreviousCompact:Y?.turnCounter??-1,previousCompactTurnId:Y?.turnId,autoCompactThreshold:a68(O,A),querySource:z},M=await tg8(q,K.agentId,J.autoCompactThreshold,!1)

if(w)return N(`Auto tool search enabled: ${j}`+(Y?` [source: ${Y}]`:"")),O(!0,A,"auto_above_threshold",H),!0;return N(`Auto tool search disabled: ${j}`+(Y?` [source: ${Y}]`:"")),O(!1,A,"auto_below_threshold",H),!1}case"standard":return O(!1,A,"standard_mode"),!1}}function gp(q){return typeof q==="object"&&q!==null&&"type"in q&&q.type==="tool_reference"}function I$Y(q){return gp(q)&&"tool_name"in q&&typeof q.tool_name==="string"}function u$Y(q){return typeof q==="object"&&q!==null&&"type"in q&&q.type==="tool_result"&&"content"in q&&Array.isArray(q.content)}function Rd(q){let K=new Set,_=0;for(let z of q){if(z.type==="system"&&z.subtype==="compact_boundary"){let $=z.compactMetadata?.preCompactDiscoveredTools;if($){for(let O of $)K.add(O);_+=$.length}continue}if(z.type!=="user")continue;let Y=z.message?.content;if(!Array.isArray(Y))continue;for(let $ of Y)if(u$Y($)){for(let O of $.content)if(I$Y(O))K.add(O.tool_name)}}if(K.size>0)N(`Dynamic tool loading: found ${K.size} discovered tools in message history`+(_>0?` (${_} carried from compact boundary)`:""));return K}function i78(){return L8("tengu_glacier_2xr",!1)}function s77(q,K,_){let z=new Set,Y=0,$=0,O=new Set;for(let M of K){if(M.type!=="attachment")continue;if(Y++,O.add(M.attachment.type),M.attachment.type!=="deferred_tools_delta")continue;$++;for(let X of M.attachment.addedNames)z.add(X);for(let X of M.attachment.removedNames)z.delete(X)}let A=q.filter(oC),w=new Set(A.map((M)=>M.name)),j=new Set(q.map((M)=>M.name)),H=A.filter((M)=>!z.has(M.name)),J=[];for(let M of z){if(w.has(M))continue;if(!j.has(M))J.push(M)}if(H.length===0&&J.length===0)return null

return[n8({content:Nv(z.join(" ")),isMeta:!0})]}case"async_hook_response":{let _=q.response,z=[];if(_.systemMessage)z.push(n8({content:_.systemMessage,isMeta:!0}));if(_.hookSpecificOutput&&"additionalContext"in _.hookSpecificOutput&&_.hookSpecificOutput.additionalContext)z.push(n8({content:_.hookSpecificOutput.additionalContext,isMeta:!0}));return V9(z)}case"token_usage":return[n8({content:Nv(`Token usage: ${q.used}/${q.total}; ${q.remaining} remaining`),isMeta:!0})];case"budget_usd":return[n8({content:Nv(`USD budget: $${q.used}/$${q.total}; $${q.remaining} remaining`),isMeta:!0})];case"output_token_usage":{let _=q.budget!==null?`${pK(q.turn)} / ${pK(q.budget)}`:pK(q.turn);return[n8({content:Nv(`Output tokens — turn: ${_} · session: ${pK(q.session)}`),isMeta:!0})]}case"hook_blocking_error":return[n8({content:Nv(`${q.hookName} hook blocking error from command: "${q.blockingError.command}": ${q.blockingError.blockingError}`),isMeta:!0})];case"hook_success":if(q.hookEvent!=="SessionStart"&&q.hookEvent!=="UserPromptSubmit")return[];if(q.content==="")return[];return[n8({content:Nv(`${q.hookName} hook success: ${q.content}`),isMeta:!0})];case"hook_additional_context":{if(q.content.length===0)return[];return[n8({content:Nv(`${q.hookName} hook additional context: ${q.content.join(`
`)}`),isMeta:!0})]}case"hook_stopped_continuation":return[n8({content:Nv(`${q.hookName} hook stopped continuation: ${q.message}`),isMeta:!0})];case"compaction_reminder":return V9([n8({content:"Auto-compact is enabled. When the context window is nearly full, older messages will be automatically summarized so you can continue working seamlessly. There is no need to stop or rush — you have unlimited context through automatic compaction.",isMeta:!0})]);case"context_efficiency":return[];case"date_change":return V9([n8({content:`The date has changed. Today's date is now ${q.newDate}. DO NOT mention this to the user explicitly because they are already aware.`,isMeta:!0})])

if(K[30]!==R||K[31]!==_)q6=()=>_(`Auto-compact window unchanged: ${R}`),K[30]=R,K[31]=_,K[32]=q6;else q6=K[32];let t,n;if(K[33]===Symbol.for("react.memo_cache_sentinel"))t=LA.createElement(T,null,"This command configures when auto-compaction happens. The actual threshold is the minimum of this setting and your model's context window."),n=!J&&LA.createElement(T,{color:"warning"},"Auto-compact is currently disabled (see /config)"),K[33]=t,K[34]=n;else t=K[33],n=K[34];let z6;if(K[35]!==K6||K[36]!==X)z6=X?LA.createElement(T,{color:"warning"},"CLAUDE_CODE_AUTO_COMPACT_WINDOW is set and takes precedence. Unset it to change this setting here."):LA.createElement(u,null,LA.createElement(T,null,"Select auto-compact window: "),LA.createElement(T,{bold:!0,color:"suggestion"},K6)),K[35]=K6,K[36]=X,K[37]=z6;else z6=K[37];let M6;if(K[38]===Symbol.for("react.memo_cache_sentinel"))M6=LA.createElement(u,{flexDirection:"column",marginTop:1},LA.createElement(T,{bold:!0},"Long context that holds up"),LA.createElement(T,null,"Both Opus 4.6 and Sonnet 4.6 achieve state-of-the-art scores on long-context retrieval benchmarks at 1M tokens — Opus 4.6 scores 78.3% on MRCR v2, the highest among frontier models at that length. Opus 4.6 includes 1M context at standard pricing; Sonnet 4.6 1M is available with overages."),LA.createElement(T,{dimColor:!0},"Learn more: ",SJY)),K[38]=M6;else M6=K[38];let J6;if(K[39]!==z6)J6=LA.createElement(u,{flexDirection:"column",gap:1},t,n,z6,M6),K[39]=z6,K[40]=J6;else J6=K[40];let G6;if(K[41]!==o||K[42]!==q6||K[43]!==J6)G6=LA.createElement(h1,{title:"Auto-compact",subtitle:o,onCancel:q6,inputGuide:bJY},J6),K[41]=o,K[42]=q6,K[43]=J6,K[44]=G6;else G6=K[44];return G6}function bJY(){return LA.createElement(T,{dimColor:!0},"↑/↓ to change · Enter to apply · Esc to cancel")}function xJY(q){return q.autoCompactWindow}var LA,rK7,SJY="https://claude.com/blog/1m-context-ga",lK7=1e5,nK7=1e5,iK7=1e6,mS6=0,IJY=async(q,K,_)=>{let z=_?.trim()||"";if(z){let Y=Bq8(z,K)

return{...r6,model:q8}}return{...S6,model:q8}})}function f8(P6){S8((V6)=>({...V6,verbose:P6})),J({...w8(),verbose:P6}),n((V6)=>({...V6,verbose:P6})),M6((V6)=>{if("verbose"in V6){let{verbose:S6,...q8}=V6;return q8}return{...V6,verbose:P6}})}let k6=[{id:"autoCompactEnabled",label:"Auto-compact",value:H.autoCompactEnabled,type:"boolean",onChange(P6){S8((V6)=>({...V6,autoCompactEnabled:P6})),J({...w8(),autoCompactEnabled:P6}),d("tengu_auto_compact_setting_changed",{enabled:P6})}},{id:"spinnerTipsEnabled",label:"Show tips",value:X?.spinnerTipsEnabled??!0,type:"boolean",onChange(P6){P7("localSettings",{spinnerTipsEnabled:P6}),P((V6)=>({...V6,spinnerTipsEnabled:P6})),d("tengu_tips_setting_changed",{enabled:P6})}},{id:"prefersReducedMotion",label:"Reduce motion",value:X?.prefersReducedMotion??!1,type:"boolean",onChange(P6){P7("localSettings",{prefersReducedMotion:P6}),P((V6)=>({...V6,prefersReducedMotion:P6})),n((V6)=>({...V6,settings:{...V6.settings,prefersReducedMotion:P6}})),d("tengu_reduce_motion_setting_changed",{enabled:P6})}},{id:"thinkingEnabled",label:"Thinking mode",value:c??!0,type:"boolean",onChange(P6){n((V6)=>({...V6,thinkingEnabled:P6})),P7("userSettings",{alwaysThinkingEnabled:P6?void 0:!1}),d("tengu_thinking_toggled",{enabled:P6})}},...gK()&&AM()?[{id:"fastMode",label:`Fast mode (${wu} only)`,value:!!K6,type:"boolean",onChange(P6){if(yY6(),P7("userSettings",{fastMode:P6?!0:void 0}),P6)n((V6)=>({...V6,mainLoopModel:$Q6(),mainLoopModelForSession:null,fastMode:!0})),M6((V6)=>({...V6,model:$Q6(),"Fast mode":"ON"}))

if(K[49]!==_.error||K[50]!==_.status)o=_.status==="failed"&&_.error&&Aw.default.createElement(u,{flexDirection:"column",marginTop:1},Aw.default.createElement(T,{bold:!0,color:"error"},"Error"),Aw.default.createElement(T,{color:"error",wrap:"wrap"},_.error)),K[49]=_.error,K[50]=_.status,K[51]=o;else o=K[51];let q6;if(K[52]!==z||K[53]!==g||K[54]!==F||K[55]!==U||K[56]!==K6||K[57]!==o||K[58]!==R)q6=Aw.default.createElement(h1,{title:R,subtitle:g,onCancel:z,color:"background",inputGuide:F},U,K6,o),K[52]=z,K[53]=g,K[54]=F,K[55]=U,K[56]=K6,K[57]=o,K[58]=R,K[59]=q6;else q6=K[59];let t;if(K[60]!==P||K[61]!==q6)t=Aw.default.createElement(u,{flexDirection:"column",tabIndex:0,autoFocus:!0,onKeyDown:P},q6),K[60]=P,K[61]=q6,K[62]=t;else t=K[62];return t}var Aw;var JuK=L(()=>{t6();RL6();i6();Kq();aq();Of();I7();Ra();IK();x4();dK();A_7();k36();Aw=w6(D6(),1)});import{randomUUID as CyY}from"crypto";function Cd8(q){return q.flatMap((K)=>{switch(K.type){case"assistant":return[{type:"assistant",message:K.message,uuid:K.uuid,requestId:void 0,timestamp:new Date().toISOString()}];case"user":return[{type:"user",message:K.message,uuid:K.uuid??CyY(),timestamp:K.timestamp??new Date().toISOString(),isMeta:K.isSynthetic}];case"system":if(K.subtype==="compact_boundary")return[{type:"system",content:"Conversation compacted",level:"info",subtype:"compact_boundary",compactMetadata:M_7(K.compact_metadata),uuid:K.uuid,timestamp:new Date().toISOString()}];return[];default:return[]}})}function bd8(q){let K=q.preservedSegment;return{trigger:q.trigger,pre_tokens:q.preTokens,...K&&{preserved_segment:{head_uuid:K.headUuid,anchor_uuid:K.anchorUuid,tail_uuid:K.tailUuid}}}}function M_7(q){let K=q.preserved_segment;return{trigger:q.trigger,preTokens:q.pre_tokens,...K&&{preservedSegment:{headUuid:K.head_uuid,anchorUuid:K.anchor_uuid,tailUuid:K.tail_uuid}}}}function X_7(q){return q.flatMap((K)=>{switch(K.type){case"assistant":return[{type:"assistant",message:byY(K),session_id:N8(),parent_tool_use_id:null,uuid:K.uuid,error:K.error}]

case"user":return[{type:"user",message:K.message,session_id:N8(),parent_tool_use_id:null,uuid:K.uuid,timestamp:K.timestamp,isSynthetic:K.isMeta||K.isVisibleInTranscriptOnly,...K.toolUseResult!==void 0&&{tool_use_result:K.toolUseResult}}];case"system":if(K.subtype==="compact_boundary"&&K.compactMetadata)return[{type:"system",subtype:"compact_boundary",session_id:N8(),uuid:K.uuid,compact_metadata:bd8(K.compactMetadata)}];if(K.subtype==="local_command"&&(K.content.includes(`<${iW}>`)||K.content.includes(`<${F_6}>`)))return[P_7(K.content,K.uuid)];return[];default:return[]}})}function P_7(q,K){let _=YA(q).replace(/<local-command-stdout>([\s\S]*?)<\/local-command-stdout>/,"$1").replace(/<local-command-stderr>([\s\S]*?)<\/local-command-stderr>/,"$1").trim();return{type:"assistant",message:Wv({content:_}).message,parent_tool_use_id:null,session_id:N8(),uuid:K}}function MuK(q){if(!q)return;return{status:q.status,...q.resetsAt!==void 0&&{resetsAt:q.resetsAt},...q.rateLimitType!==void 0&&{rateLimitType:q.rateLimitType},...q.utilization!==void 0&&{utilization:q.utilization},...q.overageStatus!==void 0&&{overageStatus:q.overageStatus},...q.overageResetsAt!==void 0&&{overageResetsAt:q.overageResetsAt},...q.overageDisabledReason!==void 0&&{overageDisabledReason:q.overageDisabledReason},...q.isUsingOverage!==void 0&&{isUsingOverage:q.isUsingOverage},...q.surpassedThreshold!==void 0&&{surpassedThreshold:q.surpassedThreshold}}}function byY(q){let K=q.message.content;if(!Array.isArray(K))return q.message;let _=K.map((z)=>{if(z.type!=="tool_use")return z;if(z.name===UX){let Y=KP();if(Y)return{...z,input:{...z.input,plan:Y}}}return z});return{...q.message,content:_}}var LC6=L(()=>{T8();O$();SE();a1();lH()});function xyY(q,K){if(q===UX)return"Review the plan in Claude Code on the web";if(!K||typeof K!=="object")return q;if(q===OO&&"questions"in K){let _=K.questions

if(F<0||F>=R)break;if(I<0)I=F;let U=F+J+36;if(U+j<=R&&q.compare(w,0,j,U,U+j)===0)if(m<0)m=F;else(p??=[m]).push(F);C=F+J}let g=p?mxY(q,W,p):m>=0?m:I;if(g>=0){let F=g+J,U=q.toString("latin1",F,F+36);P.set(U,M.length),M.push(W,R,b)}else X.push(W,R)}else X.push(W,R);W=R}let f=-1;for(let E=M.length-3;E>=0;E-=3){let R=q.indexOf(O,M[E]);if(R===-1||R>=M[E+1]){f=E;break}}if(f<0)return q;let G=new Set,Z=new Set,v=0,k=f;while(k!==void 0){if(G.has(k))break;G.add(k),Z.add(M[k]),v+=M[k+1]-M[k];let E=M[k+2];if(E<0)break;let R=q.toString("latin1",E,E+36);k=P.get(R)}if(D-v<D>>1)return q;let V=[],y=0;for(let E=0;E<M.length;E+=3){let R=M[E];while(y<X.length&&X[y]<R)V.push(q.subarray(X[y],X[y+1])),y+=2;if(Z.has(R))V.push(q.subarray(R,M[E+1]))}while(y<X.length)V.push(q.subarray(X[y],X[y+1])),y+=2;return Buffer.concat(V)}function BxY(q,K,_,z){let O=Buffer.from('{"type":"attribution-snapshot"'),A=Buffer.from('"compact_boundary"'),w=Buffer.allocUnsafe(1048576),j=Buffer.allocUnsafe(O.length),H=oz7(q,"r"),J=-1,M=0,X=-1,P=0,W=(D,f,G,Z)=>{if(G>=O.length&&D.compare(O,0,O.length,f,f+O.length)===0){J=Z,M=G;return}let v=D.toString("utf8",f,f+G);if(D.includes(A,f)&&D.indexOf(A,f)<f+G){let V=l8(v);if(V?.type==="system"&&V.subtype==="compact_boundary"){if(!V.compactMetadata?.preservedSegment)z(),J=-1,M=0}}let k=l8(v);if(k)_(k)};try{while(P<K){let D=FC6(H,w,0,1048576,P);if(D===0)break;let f=0;for(let G=0;G<D;G++)if(w[G]===10){if(X>=0){let Z=P+G-X,v=Math.min(O.length,Z);if(FC6(H,j,0,v,X),v===O.length&&j.compare(O,0,O.length,0,O.length)===0)J=X,M=Z;else{let k=Buffer.allocUnsafe(Z);FC6(H,k,0,Z,X),W(k,0,Z,X)}X=-1}else if(G>f)W(w,f,G-f,P+f);f=G+1}if(f<D&&X<0)X=P+f;P+=D}if(X>=0){let D=K-X,f=Buffer.allocUnsafe(D);FC6(H,f,0,D,X),W(f,0,D,X)}}finally{rz7(H)}return{lastAttributionOffset:J,lastAttributionLength:M}}function gxY(q,K,_){if(K<0||_<=0)return null;let z=oz7(q,"r");try{let Y=Buffer.allocUnsafe(_)

N(`[presence] pulse → ${K}`),O1.post(K,{client_id:uBY,connected_at:p$7},{headers:{...Yb6.getAuthHeaders(),"anthropic-version":"2023-06-01","anthropic-client-platform":"cli"},timeout:YnK,validateStatus:()=>!0}).then((_)=>{if(_.status>=400)N(`[presence] pulse got ${_.status}`)},()=>{})}var YnK=5000,uBY,Yb6=null,m$7=null,p$7=null,B$7=0;var $nK=L(()=>{VK();sr8();T8();_8();l1();uBY=cx6()});import{readFile as pBY,stat as BBY}from"fs/promises";async function AnK(q,K,_){let[z,Y]=await Promise.all([K.readMain(),K.readSubagents()]),$=new Set;for(let j of z??[]){let H=j.payload.uuid;if(typeof H==="string")$.add(H)}for(let j of Y??[]){let H=j.payload.uuid;if(typeof H==="string")$.add(H)}N(`[persistence-sync] Server has ${$.size} events since compaction`);let O=(j)=>{N(`[persistence-sync] Write failed: ${j}`)},A=await OnK(Df(N8()),$);for(let j of A)q("transcript",j,{...pJ(j)&&{isCompaction:!0}}).catch(O);let w=0;for(let j of _){let H=await OnK(fW(j),$);for(let J of H)q("transcript",J,{...pJ(J)&&{isCompaction:!0},agentId:j}).catch(O);w+=H.length}return N(`[persistence-sync] Uploaded ${A.length} main + ${w} subagent entries`),{uploadedMain:A.length,uploadedSubagents:w}}async function OnK(q,K){let _;try{_=(await BBY(q)).size}catch(A){if(K7(A))return[];throw A}if(_>fD6)return N(`[persistence-sync] Skipping ${q} — ${_} bytes exceeds ${fD6} threshold`),[];let z;try{z=await pBY(q,"utf8")}catch(A){if(K7(A))return[];throw A}let Y=z.split(`
`).filter(Boolean),$=[],O=-1;for(let A=0;A<Y.length;A++){let w=Y[A];if(!w)continue;let j;try{j=l8(w)}catch{continue}if(!FBY(j))continue;if($.push(j),pJ(j))O=$.length-1}return $.slice(O+1).filter((A)=>!K.has(A.uuid))}function FBY(q){return typeof q==="object"&&q!==null&&"type"in q&&gBY.has(q.type)&&"uuid"in q&&typeof q.uuid==="string"}var gBY;var wnK=L(()=>{T8();_8();E8();a1();t4();W66();r8();gBY=new Set(["user","assistant","attachment","system"])});function kl8(q){if(q===null||typeof q!=="object")return q;let K=q;if("requestId"in K&&!("request_id"in K))K.request_id=K.requestId,delete K.requestId

d("tengu_post_compact_survey_event",{event_type:"responded",appearance_id:q,response:K,session_memory_compaction_enabled:_}),QO("feedback_survey",{event_type:"responded",appearance_id:q,response:K,survey_type:"post_compact"})}function DrY(q){let K=sg8();d("tengu_post_compact_survey_event",{event_type:"appeared",appearance_id:q,session_memory_compaction_enabled:K}),QO("feedback_survey",{event_type:"appeared",appearance_id:q,survey_type:"post_compact"})}var i36,jrY=3000,HrY="tengu_post_compact_survey",JrY=0.2;var S65=L(()=>{t6();l16();l1();k8();eg8();d8();a1();vm();Yi8();i36=w6(D6(),1)});function C65(q){let K=Y6(11),{onSelect:_,inputValue:z,setInputValue:Y}=q,$;if(K[0]!==_)$=(M)=>_(ZrY[M]),K[0]=_,K[1]=$;else $=K[1];let O;if(K[2]!==z||K[3]!==Y||K[4]!==$)O={inputValue:z,setInputValue:Y,isValidDigit:GrY,onDigit:$},K[2]=z,K[3]=Y,K[4]=$,K[5]=O;else O=K[5];xb6(O);let A;if(K[6]===Symbol.for("react.memo_cache_sentinel"))A=U0.default.createElement(u,null,U0.default.createElement(T,{color:"ansi:cyan"},C9," "),U0.default.createElement(T,{bold:!0},"Can Anthropic look at your session transcript to help us improve Claude Code?")),K[6]=A;else A=K[6];let w;if(K[7]===Symbol.for("react.memo_cache_sentinel"))w=U0.default.createElement(u,{marginLeft:2},U0.default.createElement(T,{dimColor:!0},"Learn more: https://code.claude.com/docs/en/data-usage#session-quality-surveys")),K[7]=w;else w=K[7];let j;if(K[8]===Symbol.for("react.memo_cache_sentinel"))j=U0.default.createElement(u,{width:10},U0.default.createElement(T,null,U0.default.createElement(T,{color:"ansi:cyan"},"1"),": Yes")),K[8]=j;else j=K[8];let H;if(K[9]===Symbol.for("react.memo_cache_sentinel"))H=U0.default.createElement(u,{width:10},U0.default.createElement(T,null,U0.default.createElement(T,{color:"ansi:cyan"},"2"),": No")),K[9]=H;else H=K[9];let J

a plain \`string\` implicitly converts.

---

## Context Editing / Compaction (Beta)

**Beta-namespace prefix is inconsistent** (source-verified against \`src/Anthropic/Models/Beta/Messages/*.cs\` @ 12.9.0). No prefix: \`MessageCreateParams\`, \`MessageCountTokensParams\`, \`Role\`. **Everything else has the \`Beta\` prefix**: \`BetaMessageParam\`, \`BetaMessage\`, \`BetaContentBlock\`, \`BetaToolUseBlock\`, all block param types. The unprefixed \`Role\` WILL collide with \`Anthropic.Models.Messages.Role\` if you import both namespaces (CS0104). Safest: import only Beta; if mixing, alias the beta \`Role\`:

\`\`\`csharp
using Anthropic.Models.Beta.Messages;
using NonBeta = Anthropic.Models.Messages;  // only if you also need non-beta types
// Now: MessageCreateParams, BetaMessageParam, Role (beta's), NonBeta.Role (if needed)
\`\`\`


\`BetaMessage.Content\` is \`IReadOnlyList<BetaContentBlock>\` — a 15-variant discriminated union. Narrow with \`TryPick*\`. **Response \`BetaContentBlock\` is NOT assignable to param \`BetaContentBlockParam\`** — there's no \`.ToParam()\` in C#. Round-trip by converting each block:

\`\`\`csharp
using Anthropic.Models.Beta.Messages;

var betaParams = new MessageCreateParams   // no Beta prefix — one of only 2 unprefixed
{
    Model = Model.ClaudeOpus4_6,
    MaxTokens = 16000,
    Betas = ["compact-2026-01-12"],
    ContextManagement = new BetaContextManagementConfig
    {
        Edits = [new BetaCompact20260112Edit()],
    },
    Messages = messages,
};
BetaMessage resp = await client.Beta.Messages.Create(betaParams);

foreach (BetaContentBlock block in resp.Content)
{
    if (block.TryPickCompaction(out BetaCompactionBlock? compaction))
    {
        // Content is nullable — compaction can fail server-side
        Console.WriteLine($"compaction summary: {compaction.Content}")

use \`anthropic.File()\` to attach a filename + content-type for the multipart encoding.

\`\`\`go
f, _ := os.Open("./upload_me.txt")
defer f.Close()

meta, err := client.Beta.Files.Upload(ctx, anthropic.BetaFileUploadParams{
    File:  anthropic.File(f, "upload_me.txt", "text/plain"),
    Betas: []anthropic.AnthropicBeta{anthropic.AnthropicBetaFilesAPI2025_04_14},
})
// meta.ID is the file_id to reference in subsequent message requests
\`\`\`

Other \`Beta.Files\` methods: \`List\`, \`Delete\`, \`Download\`, \`GetMetadata\`.

---

## Context Editing / Compaction (Beta)

Use \`Beta.Messages.New\` with \`ContextManagement\` on \`BetaMessageNewParams\`. There is no \`NewBetaAssistantMessage\` — use \`.ToParam()\` for the round-trip.

\`\`\`go
params := anthropic.BetaMessageNewParams{
    Model:     anthropic.ModelClaudeOpus4_6,  // also supported: ModelClaudeSonnet4_6
    MaxTokens: 16000,
    Betas:     []anthropic.AnthropicBeta{"compact-2026-01-12"},
    ContextManagement: anthropic.BetaContextManagementConfigParam{
        Edits: []anthropic.BetaContextManagementConfigEditUnionParam{
            {OfCompact20260112: &anthropic.BetaCompact20260112EditParam{}},
        },
    },
    Messages: []anthropic.BetaMessageParam{ /* ... */ },
}

resp, err := client.Beta.Messages.New(ctx, params)
if err != nil {
    log.Fatal(err)
}

// Round-trip: append response to history via .ToParam()
params.Messages = append(params.Messages, resp.ToParam())

// Read compaction blocks from the response
for _, block := range resp.Content {
    if c, ok := block.AsAny().(anthropic.BetaCompactionBlock); ok {
        fmt.Println("compaction summary:", c.Content)
    }
}
\`\`\`

Other edit types: \`BetaClearToolUses20250919EditParam\`, \`BetaClearThinking20251015EditParam\`.
`;var N45=()=>{}

you must pass it back on subsequent requests — append \`response.content\`, not just the text.

\`\`\`python
import anthropic

client = anthropic.Anthropic()
messages = []

def chat(user_message: str) -> str:
    messages.append({"role": "user", "content": user_message})

    response = client.beta.messages.create(
        betas=["compact-2026-01-12"],
        model="{{OPUS_ID}}",
        max_tokens=16000,
        messages=messages,
        context_management={
            "edits": [{"type": "compact_20260112"}]
        }
    )

    # Append full content — compaction blocks must be preserved
    messages.append({"role": "assistant", "content": response.content})

    return next(block.text for block in response.content if block.type == "text")

# Compaction triggers automatically when context grows large
print(chat("Help me build a Python web scraper"))
print(chat("Add support for JavaScript-rendered pages"))
print(chat("Now add rate limiting and error handling"))
\`\`\`

---

## Stop Reasons

The \`stop_reason\` field in the response indicates why the model stopped generating:

| Value | Meaning |
|-------|---------|
| \`end_turn\` | Claude finished its response naturally |
| \`max_tokens\` | Hit the \`max_tokens\` limit — increase it or use streaming |
| \`stop_sequence\` | Hit a custom stop sequence |
| \`tool_use\` | Claude wants to call a tool — execute it and continue |
| \`pause_turn\` | Model paused and can be resumed (agentic flows) |
| \`refusal\` | Claude refused for safety reasons — output may not match your schema |

---

## Cost Optimization Strategies

### 1. Use Prompt Caching for Repeated Context

\`\`\`python
# Automatic caching (simplest — caches the last cacheable block)
response = client.messages.create(
    model="{{OPUS_ID}}",
    max_tokens=16000,
    cache_control={"type": "ephemeral"},
    system=large_document_text,  # e.g., 50KB of context
    messages=[{"role": "user", "content": "Summarize the key points"}]
)

# First request: full cost
# Subsequent requests: ~90% cheaper for cached portion
\`\`\`

### 2. Choose the Right Model

\`\`\`python
# Default to Opus for most tasks
response = client.messages.create(
    model="{{OPUS_ID}}",  # $5.00/$25.00 per 1M tokens
    max_tokens=16000,
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)

# Use Sonnet for high-volume production workloads
standard_response = client.messages.create(
    model="{{SONNET_ID}}",  # $3.00/$15.00 per 1M tokens
    max_tokens=16000,
    messages=[{"role": "user", "content": "Summarize this document"}]
)

# Use Haiku only for simple, speed-critical tasks
simple_response = client.messages.create(
    model="{{HAIKU_ID}}",  # $1.00/$5.00 per 1M tokens
    max_tokens=256,
    messages=[{"role": "user", "content": "Classify this as positive or negative"}]
)
\`\`\`

### 3. Use Token Counting Before Requests

\`\`\`python
count_response = client.messages.count_tokens(
    model="{{OPUS_ID}}",
    messages=messages,
    system=system
)

estimated_input_cost = count_response.input_tokens * 0.000005  # $5/1M tokens
print(f"Estimated input cost: \${estimated_input_cost:.4f}")
\`\`\`

---

## Retry with Exponential Backoff

> **Note:** The Anthropic SDK automatically retries rate limit (429) and server errors (5xx) with exponential backoff. You can configure this with \`max_retries\` (default: 2). Only implement custom retry logic if you need behavior beyond what the SDK provides.

\`\`\`python
import time
import random
import anthropic

def call_with_retry(
    client: anthropic.Anthropic,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    **kwargs
):
    """Call the API with exponential backoff retry."""
    last_exception = None

    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            last_exception = e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_exception = e
            else:
                raise  # Client errors (4xx except 429) should not be retried

        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
        print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s")
        time.sleep(delay)

    raise last_exception
\`\`\`
`

Claude reads the full file when the task calls for it. |

Both patterns keep the fixed context small and load detail on demand.

---

## Long-Running Agents: Managing Context

| Pattern | When to use it | What to expect |
| --- | --- | --- |
| **Context editing** | Context grows stale over many turns (old tool results, completed thinking). | Tool results and thinking blocks are cleared based on configurable thresholds. Keeps the transcript lean without summarizing. |
| **Compaction** | Conversation likely to reach or exceed the context window limit. | Earlier context is summarized into a compaction block server-side. See \`SKILL.md\` §Compaction for the critical \`response.content\` handling. |
| **Memory** | State must persist across sessions (not just within one conversation). | Claude reads/writes files in a memory directory. Survives process restarts. |

**Choosing between them:** Context editing and compaction operate within a session — editing prunes stale turns, compaction summarizes when you're near the limit. Memory is for cross-session persistence. Many long-running agents use all three.

---

## Caching for Agents

**Read \`prompt-caching.md\` first.** It covers the prefix-match invariant, breakpoint placement, the silent-invalidator audit, and why changing tools or models mid-session breaks the cache. This section covers only the agent-specific workarounds for those constraints.

| Constraint (from \`prompt-caching.md\`) | Agent-specific workaround |
| --- | --- |
| Editing the system prompt mid-session invalidates the cache. | Append a \`<system-reminder>\` block in the \`messages\` array instead. The cached prefix stays intact. Claude Code uses this for time updates and mode transitions. |
| Switching models mid-session invalidates the cache. | Spawn a **subagent** with the cheaper model for the sub-task

adding, removing, or reordering a tool invalidates the entire cache. Same for switching models (caches are model-scoped). If you need "modes", don't swap the tool set — give Claude a tool that records the mode transition, or pass the mode as message content. Serialize tools deterministically (sort by name).

**Fork operations must reuse the parent's exact prefix.** Side computations (summarization, compaction, sub-agents) often spin up a separate API call. If the fork rebuilds \`system\` / \`tools\` / \`model\` with any difference, it misses the parent's cache entirely. Copy the parent's \`system\`, \`tools\`, and \`model\` verbatim, then append fork-specific content at the end.

---

## Silent invalidators

When reviewing code, grep for these inside anything that feeds the prompt prefix:

| Pattern | Why it breaks caching |
|---|---|
| \`datetime.now()\` / \`Date.now()\` / \`time.time()\` in system prompt | Prefix changes every request |
| \`uuid4()\` / \`crypto.randomUUID()\` / request IDs early in content | Same — every request is unique |
| \`json.dumps(d)\` without \`sort_keys=True\` / iterating a \`set\` | Non-deterministic serialization → prefix bytes differ |
| f-string interpolating session/user ID into system prompt | Per-user prefix; no cross-user sharing |
| Conditional system sections (\`if flag: system += ...\`) | Every flag combination is a distinct prefix |
| \`tools=build_tools(user)\` where set varies per user | Tools render at position 0

try {
  const response = await client.messages.create({...});
} catch (error) {
  if (error instanceof Anthropic.BadRequestError) {
    console.error("Bad request:", error.message);
  } else if (error instanceof Anthropic.AuthenticationError) {
    console.error("Invalid API key");
  } else if (error instanceof Anthropic.RateLimitError) {
    console.error("Rate limited - retry later");
  } else if (error instanceof Anthropic.APIError) {
    console.error(\`API error \${error.status}:\`, error.message);
  }
}
\`\`\`

All classes extend \`Anthropic.APIError\` with a typed \`status\` field. Check from most specific to least specific. See [shared/error-codes.md](../../shared/error-codes.md) for the full error code reference.

---

## Multi-Turn Conversations

The API is stateless — send the full conversation history each time. Use \`Anthropic.MessageParam[]\` to type the messages array:

\`\`\`typescript
const messages: Anthropic.MessageParam[] = [
  { role: "user", content: "My name is Alice." },
  { role: "assistant", content: "Hello Alice! Nice to meet you." },
  { role: "user", content: "What's my name?" },
];

const response = await client.messages.create({
  model: "{{OPUS_ID}}",
  max_tokens: 16000,
  messages: messages,
});
\`\`\`

**Rules:**

- Consecutive same-role messages are allowed — the API combines them into a single turn
- First message must be \`user\`
- Use SDK types (\`Anthropic.MessageParam\`, \`Anthropic.Message\`, \`Anthropic.Tool\`, etc.) for all API data structures — don't redefine equivalent interfaces

---

### Compaction (long conversations)

> **Beta, Opus 4.6 and Sonnet 4.6.** When conversations approach the 200K context window, compaction automatically summarizes earlier context server-side. The API returns a \`compaction\` block; you must pass it back on subsequent requests — append \`response.content\`, not just the text.

\`\`\`typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();
const messages: Anthropic.Beta.BetaMessageParam[] = []

async function chat(userMessage: string): Promise<string> {
  messages.push({ role: "user", content: userMessage });

  const response = await client.beta.messages.create({
    betas: ["compact-2026-01-12"],
    model: "{{OPUS_ID}}",
    max_tokens: 16000,
    messages,
    context_management: {
      edits: [{ type: "compact_20260112" }],
    },
  });

  // Append full content — compaction blocks must be preserved
  messages.push({ role: "assistant", content: response.content });

  const textBlock = response.content.find(
    (b): b is Anthropic.Beta.BetaTextBlock => b.type === "text",
  );
  return textBlock?.text ?? "";
}

// Compaction triggers automatically when context grows large
console.log(await chat("Help me build a Python web scraper"));
console.log(await chat("Add support for JavaScript-rendered pages"));
console.log(await chat("Now add rate limiting and error handling"))