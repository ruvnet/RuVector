// ===================================================================
// Module: streaming-handler
// Source: @anthropic-ai/claude-code@2.1.91
// Confidence: 0.04
// Fragments: 24
// Extracted: 2026-04-03T03:17:18.022Z
// ===================================================================

while(($=EM7(K))!==-1)yield K.slice(0,$),K=K.slice($)}if(K.length>0)yield K}class hM7{constructor(){this.event=null,this.data=[],this.chunks=[]}decode(q){if(q.endsWith("\r"))q=q.substring(0,q.length-1);if(!q){if(!this.event&&!this.data.length)return null;let Y={event:this.event,data:this.data.join(`
`),raw:this.chunks};return this.event=null,this.data=[],this.chunks=[],Y}if(this.chunks.push(q),q.startsWith(":"))return null;let[K,_,z]=rO5(q,":");if(z.startsWith(" "))z=z.substring(1);if(K==="event")this.event=z;else if(K==="data")this.data.push(z);return null}}function rO5(q,K){let _=q.indexOf(K);if(_!==-1)return[q.substring(0,_),K,q.substring(_+K.length)];return[q,"",""]}var hI6,rv;var Fa8=L(()=>{Dl();FW();Ba8();U96();w_8();FW();rv=class rv{constructor(q,K,_){this.iterator=q,hI6.set(this,void 0),this.controller=K,J4(this,hI6,_,"f")}static fromSSEResponse(q,K,_){let z=!1,Y=_?UW(_):console;async function*$(){if(z)throw new mq("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");z=!0;let O=!1;try{for await(let A of nO5(q,K)){if(A.event==="completion")try{yield JSON.parse(A.data)}catch(w){throw Y.error("Could not parse message into JSON:",A.data),Y.error("From chunk:",A.raw),w}if(A.event==="message_start"||A.event==="message_delta"||A.event==="message_stop"||A.event==="content_block_start"||A.event==="content_block_delta"||A.event==="content_block_stop")try{yield JSON.parse(A.data)}catch(w){throw Y.error("Could not parse message into JSON:",A.data),Y.error("From chunk:",A.raw),w}if(A.event==="ping")continue;if(A.event==="error")throw new nq(void 0,Y_8(A.data)??A.data,void 0,q.headers)}O=!0}catch(A){if(fl(A))return;throw A}finally{if(!O)K.abort()}}return new rv($,K,_)}static fromReadableStream(q,K,_){let z=!1;async function*Y(){let O=new Ze,A=yI6(q);for await(let w of A)for(let j of O.decode(w))yield j;for(let w of O.flush())yield w}async function*$(){if(z)throw new mq("Cannot iterate over a consumed stream, use `.tee()` to split the stream.");z=!0;let O=!1;

if(q==="abort"){let z=K[0];if(!x1(this,c96,"f")&&!_?.length)Promise.reject(z);x1(this,pI6,"f").call(this,z),x1(this,gI6,"f").call(this,z),this._emit("end");return}if(q==="error"){let z=K[0];if(!x1(this,c96,"f")&&!_?.length)Promise.reject(z);x1(this,pI6,"f").call(this,z),x1(this,gI6,"f").call(this,z),this._emit("end")}}_emitFinal(){if(this.receivedMessages.at(-1))this._emit("finalMessage",x1(this,ZR,"m",zs8).call(this))}async _fromReadableStream(q,K){let _=K?.signal,z;if(_){if(_.aborted)this.controller.abort();z=this.controller.abort.bind(this.controller),_.addEventListener("abort",z)}try{x1(this,ZR,"m",Ys8).call(this),this._connected(null);let Y=rv.fromReadableStream(q,this.controller);for await(let $ of Y)x1(this,ZR,"m",$s8).call(this,$);if(Y.controller.signal?.aborted)throw new c_;x1(this,ZR,"m",Os8).call(this)}finally{if(_&&z)_.removeEventListener("abort",z)}}[(Te=new WeakMap,VP6=new WeakMap,mI6=new WeakMap,G_8=new WeakMap,pI6=new WeakMap,BI6=new WeakMap,v_8=new WeakMap,gI6=new WeakMap,Gl=new WeakMap,FI6=new WeakMap,T_8=new WeakMap,k_8=new WeakMap,c96=new WeakMap,V_8=new WeakMap,N_8=new WeakMap,UI6=new WeakMap,y_8=new WeakMap,ZR=new WeakSet,zs8=function(){if(this.receivedMessages.length===0)throw new mq("stream ended without producing a Message with role=assistant");return this.receivedMessages.at(-1)},BM7=function(){if(this.receivedMessages.length===0)throw new mq("stream ended without producing a Message with role=assistant");let K=this.receivedMessages.at(-1).content.filter((_)=>_.type==="text").map((_)=>_.text);if(K.length===0)throw new mq("stream ended without producing a content block with type=text");return K.join(" ")},Ys8=function(){if(this.ended)return;J4(this,Te,void 0,"f")},$s8=function(K){if(this.ended)return;let _=x1(this,ZR,"m",gM7).call(this,K);switch(this._emit("streamEvent",K,_),K.type){case"content_block_delta":{let z=_.content.at(-1);switch(K.delta.type){case"text_delta":{if(z.type==="text")this._emit("text",K.delta.text,z.text||"");

case"content_block_start":return _.content.push(K.content_block),_;case"content_block_delta":{let z=_.content.at(K.index);switch(K.delta.type){case"text_delta":{if(z?.type==="text")_.content[K.index]={...z,text:(z.text||"")+K.delta.text};break}case"citations_delta":{if(z?.type==="text")_.content[K.index]={...z,citations:[...z.citations??[],K.delta.citation]};break}case"input_json_delta":{if(z&&UM7(z)){let Y=z[FM7]||"";Y+=K.delta.partial_json;let $={...z};if(Object.defineProperty($,FM7,{value:Y,enumerable:!1,writable:!0}),Y)try{$.input=f_8(Y)}catch(O){let A=new mq(`Unable to parse tool parameter JSON from model. Please retry your request or adjust your prompt. Error: ${O}. JSON: ${Y}`);x1(this,y_8,"f").call(this,A)}_.content[K.index]=$}break}case"thinking_delta":{if(z?.type==="thinking")_.content[K.index]={...z,thinking:z.thinking+K.delta.thinking};break}case"signature_delta":{if(z?.type==="thinking")_.content[K.index]={...z,signature:K.delta.signature};break}case"compaction_delta":{if(z?.type==="compaction")_.content[K.index]={...z,content:(z.content||"")+K.delta.content};break}default:QM7(K.delta)}return _}case"content_block_stop":return _}},Symbol.asyncIterator)](){let q=[],K=[],_=!1;return this.on("streamEvent",(z)=>{let Y=K.shift();if(Y)Y.resolve(z);else q.push(z)}),this.on("end",()=>{_=!0;for(let z of K)z.resolve(void 0);K.length=0}),this.on("abort",(z)=>{_=!0;for(let Y of K)Y.reject(z);K.length=0}),this.on("error",(z)=>{_=!0;for(let Y of K)Y.reject(z);K.length=0}),{next:async()=>{if(!q.length){if(_)return{value:void 0,done:!0};return new Promise((Y,$)=>K.push({resolve:Y,reject:$})).then((Y)=>Y?{value:Y,done:!1}:{value:void 0,done:!0})}return{value:q.shift(),done:!1}},return:async()=>{return this.abort(),{value:void 0,done:!0}}}}toReadableStream(){return new rv(this[Symbol.asyncIterator].bind(this),this.controller).toReadableStream()}}});var NP6;var E_8=L(()=>{NP6=class NP6 extends Error{constructor(q){let K=typeof q==="string"?q:q.map((_)=>{if(_.type==="text")return _.text;return`[${_.type}]`}).join(" ");

if(q==="abort"){let z=K[0];if(!x1(this,n96,"f")&&!_?.length)Promise.reject(z);x1(this,aI6,"f").call(this,z),x1(this,tI6,"f").call(this,z),this._emit("end");return}if(q==="error"){let z=K[0];if(!x1(this,n96,"f")&&!_?.length)Promise.reject(z);x1(this,aI6,"f").call(this,z),x1(this,tI6,"f").call(this,z),this._emit("end")}}_emitFinal(){if(this.receivedMessages.at(-1))this._emit("finalMessage",x1(this,GR,"m",Gs8).call(this))}async _fromReadableStream(q,K){let _=K?.signal,z;if(_){if(_.aborted)this.controller.abort();z=this.controller.abort.bind(this.controller),_.addEventListener("abort",z)}try{x1(this,GR,"m",Ts8).call(this),this._connected(null);let Y=rv.fromReadableStream(q,this.controller);for await(let $ of Y)x1(this,GR,"m",ks8).call(this,$);if(Y.controller.signal?.aborted)throw new c_;x1(this,GR,"m",Vs8).call(this)}finally{if(_&&z)_.removeEventListener("abort",z)}}[(ye=new WeakMap,hP6=new WeakMap,oI6=new WeakMap,L_8=new WeakMap,aI6=new WeakMap,sI6=new WeakMap,h_8=new WeakMap,tI6=new WeakMap,Tl=new WeakMap,eI6=new WeakMap,R_8=new WeakMap,S_8=new WeakMap,n96=new WeakMap,C_8=new WeakMap,b_8=new WeakMap,qu6=new WeakMap,vs8=new WeakMap,GR=new WeakSet,Gs8=function(){if(this.receivedMessages.length===0)throw new mq("stream ended without producing a Message with role=assistant");return this.receivedMessages.at(-1)},sM7=function(){if(this.receivedMessages.length===0)throw new mq("stream ended without producing a Message with role=assistant");let K=this.receivedMessages.at(-1).content.filter((_)=>_.type==="text").map((_)=>_.text);if(K.length===0)throw new mq("stream ended without producing a content block with type=text");return K.join(" ")},Ts8=function(){if(this.ended)return;J4(this,ye,void 0,"f")},ks8=function(K){if(this.ended)return;let _=x1(this,GR,"m",tM7).call(this,K);switch(this._emit("streamEvent",K,_),K.type){case"content_block_delta":{let z=_.content.at(-1);switch(K.delta.type){case"text_delta":{if(z.type==="text")this._emit("text",K.delta.text,z.text||"");

break}case"citations_delta":{if(z?.type==="text")_.content[K.index]={...z,citations:[...z.citations??[],K.delta.citation]};break}case"input_json_delta":{if(z&&qX7(z)){let Y=z[eM7]||"";Y+=K.delta.partial_json;let $={...z};if(Object.defineProperty($,eM7,{value:Y,enumerable:!1,writable:!0}),Y)$.input=f_8(Y);_.content[K.index]=$}break}case"thinking_delta":{if(z?.type==="thinking")_.content[K.index]={...z,thinking:z.thinking+K.delta.thinking};break}case"signature_delta":{if(z?.type==="thinking")_.content[K.index]={...z,signature:K.delta.signature};break}default:KX7(K.delta)}return _}case"content_block_stop":return _}},Symbol.asyncIterator)](){let q=[],K=[],_=!1;return this.on("streamEvent",(z)=>{let Y=K.shift();if(Y)Y.resolve(z);else q.push(z)}),this.on("end",()=>{_=!0;for(let z of K)z.resolve(void 0);K.length=0}),this.on("abort",(z)=>{_=!0;for(let Y of K)Y.reject(z);K.length=0}),this.on("error",(z)=>{_=!0;for(let Y of K)Y.reject(z);K.length=0}),{next:async()=>{if(!q.length){if(_)return{value:void 0,done:!0};return new Promise((Y,$)=>K.push({resolve:Y,reject:$})).then((Y)=>Y?{value:Y,done:!1}:{value:void 0,done:!0})}return{value:q.shift(),done:!1}},return:async()=>{return this.abort(),{value:void 0,done:!0}}}}toReadableStream(){return new rv(this[Symbol.asyncIterator].bind(this),this.controller).toReadableStream()}}});var _u6;var Ns8=L(()=>{uB();NE();js8();ve();Ge();_u6=class _u6 extends AH{create(q,K){return this._client.post("/v1/messages/batches",{body:q,...K})}retrieve(q,K){return this._client.get(jj`/v1/messages/batches/${q}`,K)}list(q={},K){return this._client.getAPIList("/v1/messages/batches",qI,{query:q,...K})}delete(q,K){return this._client.delete(jj`/v1/messages/batches/${q}`,K)}cancel(q,K){return this._client.post(jj`/v1/messages/batches/${q}/cancel`,K)}async results(q,K){let _=await this.retrieve(q);if(!_.results_url)throw new mq(`No batch \`results_url\`; Has it finished processing? ${_.processing_status} - ${_.id}`);

else j=`Received RST_STREAM with code ${q.rstCode} triggered by internal client error: ${this.internalError.message}`;break;default:w=_w.Status.INTERNAL,j=`Received RST_STREAM with code ${q.rstCode}`}this.endCall({code:w,details:j,metadata:new Wa.Metadata,rstCode:q.rstCode})})}),q.on("error",(A)=>{if(A.code!=="ERR_HTTP2_STREAM_ERROR")this.trace("Node error event: message="+A.message+" code="+A.code+" errno="+VCz(A.errno)+" syscall="+A.syscall),this.internalError=A;this.callEventTracker.onStreamEnd(!1)})}getDeadlineInfo(){return[`remote_addr=${this.getPeer()}`]}onDisconnect(){this.connectionDropped=!0,setImmediate(()=>{this.endCall({code:_w.Status.UNAVAILABLE,details:"Connection dropped",metadata:new Wa.Metadata})})}outputStatus(){if(!this.statusOutput)this.statusOutput=!0,this.trace("ended with status: code="+this.finalStatus.code+' details="'+this.finalStatus.details+'"'),this.callEventTracker.onCallEnd(this.finalStatus),process.nextTick(()=>{this.listener.onReceiveStatus(this.finalStatus)}),this.http2Stream.resume()}trace(q){vCz.trace(TCz.LogVerbosity.DEBUG,kCz,"["+this.callId+"] "+q)}endCall(q){if(this.finalStatus===null||this.finalStatus.code===_w.Status.OK)this.finalStatus=q,this.maybeOutputStatus();this.destroyHttp2Stream()}maybeOutputStatus(){if(this.finalStatus!==null){if(this.finalStatus.code!==_w.Status.OK||this.readsClosed&&this.unpushedReadMessages.length===0&&!this.isReadFilterPending&&!this.isPushPending)this.outputStatus()}}push(q){this.trace("pushing to reader message of length "+(q instanceof Buffer?q.length:null)),this.canPush=!1,this.isPushPending=!0,process.nextTick(()=>{if(this.isPushPending=!1,this.statusOutput)return;this.listener.onReceiveMessage(q),this.maybeOutputStatus()})}tryPush(q){if(this.canPush)this.http2Stream.pause(),this.push(q);else this.trace("unpushedReadMessages.push message of length "+q.length),this.unpushedReadMessages.push(q)}handleTrailers(q){this.serverEndedCall=!0,this.callEventTracker.onStreamEnd(!0);let K="";for(let $ of Object.keys(q))K+="\t\t"+$+": "+q[$]+`
`;

if(this.channelzEnabled)this.streamTracker.addCallStarted(),A={addMessageSent:()=>{var j;this.messagesSent+=1,this.lastMessageSentTimestamp=new Date,(j=Y.addMessageSent)===null||j===void 0||j.call(Y)},addMessageReceived:()=>{var j;this.messagesReceived+=1,this.lastMessageReceivedTimestamp=new Date,(j=Y.addMessageReceived)===null||j===void 0||j.call(Y)},onCallEnd:(j)=>{var H;(H=Y.onCallEnd)===null||H===void 0||H.call(Y,j),this.removeActiveCall(w)},onStreamEnd:(j)=>{var H;if(j)this.streamTracker.addCallSucceeded();else this.streamTracker.addCallFailed();(H=Y.onStreamEnd)===null||H===void 0||H.call(Y,j)}};else A={addMessageSent:()=>{var j;(j=Y.addMessageSent)===null||j===void 0||j.call(Y)},addMessageReceived:()=>{var j;(j=Y.addMessageReceived)===null||j===void 0||j.call(Y)},onCallEnd:(j)=>{var H;(H=Y.onCallEnd)===null||H===void 0||H.call(Y,j),this.removeActiveCall(w)},onStreamEnd:(j)=>{var H;(H=Y.onStreamEnd)===null||H===void 0||H.call(Y,j)}};return w=new hCz.Http2SubchannelCall(O,A,z,this,(0,RCz.getNextCallNumber)()),this.addActiveCall(w),w}getChannelzRef(){return this.channelzRef}getPeerName(){return this.subchannelAddressString}getOptions(){return this.options}getAuthContext(){return this.authContext}shutdown(){this.session.close(),(0,tC8.unregisterChannelzRef)(this.channelzRef)}}class ws4{constructor(q){this.channelTarget=q,this.session=null,this.isShutdown=!1}trace(q){CE6.trace(Ce6.LogVerbosity.DEBUG,Xn1,(0,Mn1.uriToString)(this.channelTarget)+" "+q)}createSession(q,K,_){if(this.isShutdown)return Promise.reject();if(q.socket.closed)return Promise.reject("Connection closed before starting HTTP/2 handshake");return new Promise((z,Y)=>{var $,O,A,w,j,H,J;let M=null,X=this.channelTarget;if("grpc.http_connect_target"in _){let E=(0,Mn1.parseUri)(_["grpc.http_connect_target"]);if(E)X=E,M=(0,Mn1.uriToString)(E)}let P=q.secure?"https":"http",W=(0,ECz.getDefaultAuthority)(X),D=()=>{var E;

if(this.stream=q,this.callEventTracker=_,this.handler=z,this.listener=null,this.deadlineTimer=null,this.deadline=1/0,this.maxSendMessageSize=Zy.DEFAULT_MAX_SEND_MESSAGE_LENGTH,this.maxReceiveMessageSize=Zy.DEFAULT_MAX_RECEIVE_MESSAGE_LENGTH,this.cancelled=!1,this.metadataSent=!1,this.wantTrailers=!1,this.cancelNotified=!1,this.incomingEncoding="identity",this.readQueue=[],this.isReadPending=!1,this.receivedHalfClose=!1,this.streamEnded=!1,this.metricsRecorder=new Zt4.PerRequestMetricRecorder,this.stream.once("error",(J)=>{}),this.stream.once("close",()=>{var J;if(qj6("Request to method "+((J=this.handler)===null||J===void 0?void 0:J.path)+" stream closed with rstCode "+this.stream.rstCode),this.callEventTracker&&!this.streamEnded)this.streamEnded=!0,this.callEventTracker.onStreamEnd(!1),this.callEventTracker.onCallEnd({code:Zy.Status.CANCELLED,details:"Stream closed before sending status",metadata:null});this.notifyOnCancel()}),this.stream.on("data",(J)=>{this.handleDataFrame(J)}),this.stream.pause(),this.stream.on("end",()=>{this.handleEndEvent()}),"grpc.max_send_message_length"in Y)this.maxSendMessageSize=Y["grpc.max_send_message_length"];if("grpc.max_receive_message_length"in Y)this.maxReceiveMessageSize=Y["grpc.max_receive_message_length"];this.host=($=K[":authority"])!==null&&$!==void 0?$:K.host,this.decoder=new Jxz.StreamDecoder(this.maxReceiveMessageSize);let A=Mb8.Metadata.fromHttp2Headers(K);if(kt4.isTracerEnabled(Vt4))qj6("Request to "+this.handler.path+" received headers "+JSON.stringify(A.toJSON()));let w=A.get(Rn1);if(w.length>0)this.handleTimeoutHeader(w[0]);let j=A.get(Sn1);if(j.length>0)this.incomingEncoding=j[0];A.remove(Rn1),A.remove(Sn1),A.remove(ht4),A.remove(xE6.constants.HTTP2_HEADER_ACCEPT_ENCODING),A.remove(xE6.constants.HTTP2_HEADER_TE),A.remove(xE6.constants.HTTP2_HEADER_CONTENT_TYPE),this.metadata=A;let H=(O=q.session)===null||O===void 0?void 0:O.socket;

if(this.metadataSent)return;this.metadataSent=!0;let K=q?q.toHttp2Headers():null,_=Object.assign(Object.assign(Object.assign({},Tt4),Dxz),K);this.stream.respond(_,fxz)}sendMessage(q,K){if(this.checkCancelled())return;let _;try{_=this.serializeMessage(q)}catch(z){this.sendStatus({code:Zy.Status.INTERNAL,details:`Error serializing response: ${(0,Dt4.getErrorMessage)(z)}`,metadata:null});return}if(this.maxSendMessageSize!==-1&&_.length-5>this.maxSendMessageSize){this.sendStatus({code:Zy.Status.RESOURCE_EXHAUSTED,details:`Sent message larger than max (${_.length} vs. ${this.maxSendMessageSize})`,metadata:null});return}this.maybeSendMetadata(),qj6("Request to "+this.handler.path+" sent data frame of size "+_.length),this.stream.write(_,(z)=>{var Y;if(z){this.sendStatus({code:Zy.Status.INTERNAL,details:`Error writing message: ${(0,Dt4.getErrorMessage)(z)}`,metadata:null});return}(Y=this.callEventTracker)===null||Y===void 0||Y.addMessageSent(),K()})}sendStatus(q){var K,_,z;if(this.checkCancelled())return;qj6("Request to method "+((K=this.handler)===null||K===void 0?void 0:K.path)+" ended with status code: "+Zy.Status[q.code]+" details: "+q.details);let Y=(z=(_=q.metadata)===null||_===void 0?void 0:_.clone())!==null&&z!==void 0?z:new Mb8.Metadata;if(this.shouldSendMetrics)Y.set(Zt4.GRPC_METRICS_HEADER,this.metricsRecorder.serialize());if(this.metadataSent)if(!this.wantTrailers)this.wantTrailers=!0,this.stream.once("wantTrailers",()=>{if(this.callEventTracker&&!this.streamEnded)this.streamEnded=!0,this.callEventTracker.onStreamEnd(!0),this.callEventTracker.onCallEnd(q);let $=Object.assign({[vt4]:q.code,[Gt4]:encodeURI(q.details)},Y.toHttp2Headers());this.stream.sendTrailers($),this.notifyOnCancel()}),this.stream.end();else this.notifyOnCancel();else{if(this.callEventTracker&&!this.streamEnded)this.streamEnded=!0,this.callEventTracker.onStreamEnd(!0),this.callEventTracker.onCallEnd(q);let $=Object.assign(Object.assign({[vt4]:q.code,[Gt4]:encodeURI(q.details)},Tt4),Y.toHttp2Headers());

if(this.started===!0)throw Error("server is already started");this.started=!0}tryShutdown(Y){var $;let O=(j)=>{(0,HW.unregisterChannelzRef)(this.channelzRef),Y(j)},A=0;function w(){if(A--,A===0)O()}this.shutdown=!0;for(let[j,H]of this.http2Servers.entries()){A++;let J=H.channelzRef.name;this.trace("Waiting for server "+J+" to close"),this.closeServer(j,()=>{this.trace("Server "+J+" finished closing"),w()});for(let M of H.sessions.keys()){A++;let X=($=M.socket)===null||$===void 0?void 0:$.remoteAddress;this.trace("Waiting for session "+X+" to close"),this.closeSession(M,()=>{this.trace("Session "+X+" finished closing"),w()})}}if(A===0)O()}addHttp2Port(){throw Error("Not yet implemented")}getChannelzRef(){return this.channelzRef}_verifyContentType(Y,$){let O=$[Gy.constants.HTTP2_HEADER_CONTENT_TYPE];if(typeof O!=="string"||!O.startsWith("application/grpc"))return Y.respond({[Gy.constants.HTTP2_HEADER_STATUS]:Gy.constants.HTTP_STATUS_UNSUPPORTED_MEDIA_TYPE},{endStream:!0}),!1;return!0}_retrieveHandler(Y){mt4("Received call to method "+Y+" at address "+this.serverAddressString);let $=this.handlers.get(Y);if($===void 0)return mt4("No handler registered for method "+Y+". Sending UNIMPLEMENTED status."),null;return $}_respondWithError(Y,$,O=null){var A,w;let j=Object.assign({"grpc-status":(A=Y.code)!==null&&A!==void 0?A:bM.Status.INTERNAL,"grpc-message":Y.details,[Gy.constants.HTTP2_HEADER_STATUS]:Gy.constants.HTTP_STATUS_OK,[Gy.constants.HTTP2_HEADER_CONTENT_TYPE]:"application/grpc+proto"},(w=Y.metadata)===null||w===void 0?void 0:w.toHttp2Headers());$.respond(j,{endStream:!0}),this.callTracker.addCallFailed(),O===null||O===void 0||O.streamTracker.addCallFailed()}_channelzHandler(Y,$,O){this.onStreamOpened($);let A=this.sessions.get($.session);if(this.callTracker.addCallStarted(),A===null||A===void 0||A.streamTracker.addCallStarted(),!this._verifyContentType($,O)){this.callTracker.addCallFailed(),A===null||A===void 0||A.streamTracker.addCallFailed();return}let w=O[It4],j=this._retrieveHandler(w);

if(!j){this._respondWithError(un1(w),$,A);return}let H={addMessageSent:()=>{if(A)A.messagesSent+=1,A.lastMessageSentTimestamp=new Date},addMessageReceived:()=>{if(A)A.messagesReceived+=1,A.lastMessageReceivedTimestamp=new Date},onCallEnd:(M)=>{if(M.code===bM.Status.OK)this.callTracker.addCallSucceeded();else this.callTracker.addCallFailed()},onStreamEnd:(M)=>{if(A)if(M)A.streamTracker.addCallSucceeded();else A.streamTracker.addCallFailed()}},J=(0,bt4.getServerInterceptingCall)([...Y,...this.interceptors],$,O,H,j,this.options);if(!this._runHandlerForCall(J,j))this.callTracker.addCallFailed(),A===null||A===void 0||A.streamTracker.addCallFailed(),J.sendStatus({code:bM.Status.INTERNAL,details:`Unknown handler type: ${j.type}`})}_streamHandler(Y,$,O){if(this.onStreamOpened($),this._verifyContentType($,O)!==!0)return;let A=O[It4],w=this._retrieveHandler(A);if(!w){this._respondWithError(un1(A),$,null);return}let j=(0,bt4.getServerInterceptingCall)([...Y,...this.interceptors],$,O,null,w,this.options);if(!this._runHandlerForCall(j,w))j.sendStatus({code:bM.Status.INTERNAL,details:`Unknown handler type: ${w.type}`})}_runHandlerForCall(Y,$){let{type:O}=$;if(O==="unary")xxz(Y,$);else if(O==="clientStream")Ixz(Y,$);else if(O==="serverStream")uxz(Y,$);else if(O==="bidi")mxz(Y,$);else return!1;return!0}_setupHandlers(Y,$){if(Y===null)return;let O=Y.address(),A="null";if(O)if(typeof O==="string")A=O;else A=O.address+":"+O.port;this.serverAddressString=A;let w=this.channelzEnabled?this._channelzHandler:this._streamHandler,j=this.channelzEnabled?this._channelzSessionHandler(Y):this._sessionHandler(Y);Y.on("stream",w.bind(this,$)),Y.on("session",j)}_sessionHandler(Y){return($)=>{var O,A;(O=this.http2Servers.get(Y))===null||O===void 0||O.sessions.add($);let w=null,j=null,H=null,J=!1,M=this.enableIdleTimeout($);if(this.maxConnectionAgeMs!==IE6){let f=this.maxConnectionAgeMs/10,G=Math.random()*f*2-f;w=setTimeout(()=>{var Z,v;

else this.keepaliveTrace("Received ping response"),k()}))y="Ping returned false"}catch(E){y=(E instanceof Error?E.message:"")||"Unknown error"}if(y){this.keepaliveTrace("Ping send failed: "+y),this.channelzTrace.addTrace("CT_INFO","Connection dropped due to ping send error: "+y),D=!0,$.close();return}J.keepAlivesSent+=1,W=setTimeout(()=>{G(),this.keepaliveTrace("Ping timeout passed without response"),this.channelzTrace.addTrace("CT_INFO","Connection dropped by keepalive timeout from "+M),D=!0,$.close()},this.keepaliveTimeoutMs),(V=W.unref)===null||V===void 0||V.call(W)},k(),$.on("close",()=>{var V;if(!D)this.channelzTrace.addTrace("CT_INFO","Connection dropped by client "+M);if(this.sessionChildrenTracker.unrefChild(H),(0,HW.unregisterChannelzRef)(H),X)clearTimeout(X);if(P)clearTimeout(P);if(G(),f!==null)clearTimeout(f.timeout),this.sessionIdleTimeouts.delete($);(V=this.http2Servers.get(Y))===null||V===void 0||V.sessions.delete($),this.sessions.delete($)})}}enableIdleTimeout(Y){var $,O;if(this.sessionIdleTimeout>=xt4)return null;let A={activeStreams:0,lastIdle:Date.now(),onClose:this.onStreamClose.bind(this,Y),timeout:setTimeout(this.onIdleTimeout,this.sessionIdleTimeout,this,Y)};(O=($=A.timeout).unref)===null||O===void 0||O.call($),this.sessionIdleTimeouts.set(Y,A);let{socket:w}=Y;return this.trace("Enable idle timeout for "+w.remoteAddress+":"+w.remotePort),A}onIdleTimeout(Y,$){let{socket:O}=$,A=Y.sessionIdleTimeouts.get($);if(A!==void 0&&A.activeStreams===0)if(Date.now()-A.lastIdle>=Y.sessionIdleTimeout)Y.trace("Session idle timeout triggered for "+(O===null||O===void 0?void 0:O.remoteAddress)+":"+(O===null||O===void 0?void 0:O.remotePort)+" last idle at "+A.lastIdle),Y.closeSession($);else A.timeout.refresh()}onStreamOpened(Y){let $=Y.session,O=this.sessionIdleTimeouts.get($);if(O)O.activeStreams+=1,Y.once("close",O.onClose)}onStreamClose(Y){var $,O;let A=this.sessionIdleTimeouts.get(Y);

if(A){if(A.activeStreams-=1,A.activeStreams===0)A.lastIdle=Date.now(),A.timeout.refresh(),this.trace("Session onStreamClose"+(($=Y.socket)===null||$===void 0?void 0:$.remoteAddress)+":"+((O=Y.socket)===null||O===void 0?void 0:O.remotePort)+" at "+A.lastIdle)}}},(()=>{let z=typeof Symbol==="function"&&Symbol.metadata?Object.create(null):void 0;if(_=[Sxz("Calling start() is no longer necessary. It can be safely omitted.")],yxz(q,null,_,{kind:"method",name:"start",static:!1,private:!1,access:{has:(Y)=>("start"in Y),get:(Y)=>Y.start},metadata:z},null,K),z)Object.defineProperty(q,Symbol.metadata,{enumerable:!0,configurable:!0,writable:!0,value:z})})(),q})();PK6.Server=bxz;async function xxz(q,K){let _;function z(O,A,w,j){if(O){q.sendStatus((0,mE6.serverErrorToStatus)(O,w));return}q.sendMessage(A,()=>{q.sendStatus({code:bM.Status.OK,details:"OK",metadata:w!==null&&w!==void 0?w:null})})}let Y,$=null;q.start({onReceiveMetadata(O){Y=O,q.startRead()},onReceiveMessage(O){if($){q.sendStatus({code:bM.Status.UNIMPLEMENTED,details:`Received a second request message for server streaming method ${K.path}`,metadata:null});return}$=O,q.startRead()},onReceiveHalfClose(){if(!$){q.sendStatus({code:bM.Status.UNIMPLEMENTED,details:`Received no request message for server streaming method ${K.path}`,metadata:null});return}_=new mE6.ServerWritableStreamImpl(K.path,q,Y,$);try{K.func(_,z)}catch(O){q.sendStatus({code:bM.Status.UNKNOWN,details:`Server method handler threw error ${O.message}`,metadata:null})}},onCancel(){if(_)_.cancelled=!0,_.emit("cancelled","cancelled")}})}function Ixz(q,K){let _;function z(Y,$,O,A){if(Y){q.sendStatus((0,mE6.serverErrorToStatus)(Y,O));return}q.sendMessage($,()=>{q.sendStatus({code:bM.Status.OK,details:"OK",metadata:O!==null&&O!==void 0?O:null})})}q.start({onReceiveMetadata(Y){_=new mE6.ServerDuplexStreamImpl(K.path,q,Y);

let P1={message:{...x6,content:UF8([N7],z,$.agentId)},requestId:H6??void 0,type:"assistant",uuid:ql8(),timestamp:new Date().toISOString(),...!1,...H&&{advisorModel:H}};X6.push(P1),yield P1;break}case"message_delta":{W6=x56(W6,D1.usage),Z6=D1.delta.stop_reason;let N7=X6.at(-1);if(N7)N7.message.usage=W6,N7.message.stop_reason=Z6;let P1=x86(A,W6);N6+=Fh6(P1,W6,$.model);let D7=WTK(D1.delta.stop_reason,$.model);if(D7)yield D7;if(Z6==="max_tokens")d("tengu_max_tokens_reached",{max_tokens:K8}),yield U9({content:`${MW}: Claude's response exceeded the ${K8} output token maximum. To configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable.`,apiError:"max_output_tokens",error:"max_output_tokens"});if(Z6==="model_context_window_exceeded")d("tengu_context_window_exceeded",{max_tokens:K8,output_tokens:W6.output_tokens}),yield U9({content:`${MW}: The model has reached its context window limit.`,apiError:"max_output_tokens",error:"max_output_tokens"});break}case"message_stop":break}yield{type:"stream_event",event:D1,...D1.type==="message_start"?{ttftMs:v6}:void 0}}if(q8(),h6){let D1=P6!==null?Math.round(performance.now()-P6):-1;throw a8("info","cli_stream_loop_exited_after_watchdog_clean"),d("tengu_stream_loop_exited_after_watchdog",{request_id:H6??"unknown",exit_delay_ms:D1,exit_path:"clean",model:$.model}),P6=null,Error("Stream idle timeout - no chunks received")}if(!x6||X6.length===0&&!Z6)throw N(!x6?"Stream completed without receiving message_start event - triggering non-streaming fallback":"Stream completed with message_start but no content blocks completed - triggering non-streaming fallback",{level:"error"}),d("tengu_stream_no_events",{model:$.model,request_id:H6??"unknown"}),Error("Stream ended without receiving any events");if(E1>0)N(`Streaming completed with ${E1} stall(s), total stall time: ${(b8/1000).toFixed(1)}s`,{level:"warn"}),d("tengu_streaming_stall_summary",{stall_count:E1,total_stall_time_ms:b8,model:$.model,request_id:H6??"unknown"});let _7=a;

if(_7)Jg1(_7.headers),s6=_7.headers}catch(r6){if(q8(),h6&&P6!==null){let E1=Math.round(performance.now()-P6);a8("info","cli_stream_loop_exited_after_watchdog_error"),d("tengu_stream_loop_exited_after_watchdog",{request_id:H6??"unknown",exit_delay_ms:E1,exit_path:"error",error_name:r6 instanceof Error?r6.name:"unknown",model:$.model})}if(r6 instanceof c_)if(Y.aborted){if(N(`Streaming aborted by user: ${F6(r6)}`),k6)d("tengu_advisor_tool_interrupted",{model:$.model,advisor_model:H??"unknown"});throw r6}else throw N(`Streaming timeout (SDK abort): ${r6.message}`,{level:"error"}),new IB({message:"Request timed out"});if(c6(process.env.CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK)||L8("tengu_disable_streaming_to_non_streaming_fallback",!1))throw N(`Error streaming (non-streaming fallback disabled): ${F6(r6)}`,{level:"error"}),d("tengu_streaming_fallback_to_non_streaming",{model:$.model,error:r6 instanceof Error?r6.name:String(r6),attemptNumber:M6,maxOutputTokens:K8,thinkingType:_.type,fallback_disabled:!0,request_id:H6??"unknown",fallback_cause:h6?"watchdog":"other"}),r6;if(N(`Error streaming, falling back to non-streaming mode: ${F6(r6)}`,{level:"error"}),I6=!0,$.onStreamingFallback)$.onStreamingFallback();d("tengu_streaming_fallback_to_non_streaming",{model:$.model,error:r6 instanceof Error?r6.name:String(r6),attemptNumber:M6,maxOutputTokens:K8,thinkingType:_.type,fallback_disabled:!1,request_id:H6??"unknown",fallback_cause:h6?"watchdog":"other"}),a8("info","cli_nonstreaming_fallback_started"),d("tengu_nonstreaming_fallback_started",{request_id:H6??"unknown",model:$.model,fallback_cause:h6?"watchdog":"other"});

let C8=yield*odK({model:$.model,source:$.querySource},{model:$.model,fallbackModel:$.fallbackModel,thinkingConfig:_,...gK()&&{fastMode:g},signal:Y,initialConsecutive529Errors:fw6(r6)?1:0,querySource:$.querySource},O6,(E1,_7,D1)=>{M6=E1,K8=D1},(E1)=>oO8(E1,$.querySource),H6),b8={message:{...C8,content:UF8(C8.content,z,$.agentId)},requestId:H6??void 0,type:"assistant",uuid:ql8(),timestamp:new Date().toISOString(),...!1,...H&&{advisorModel:H}};X6.push(b8),l6=b8,yield b8}finally{q8()}}catch(m6){if(m6 instanceof Zw6)throw m6;if(!I6&&m6 instanceof Pm&&m6.originalError instanceof nq&&m6.originalError.status===404){let T6=m6.originalError.requestID??"unknown";if(N("Streaming endpoint returned 404, falling back to non-streaming mode",{level:"warn"}),I6=!0,$.onStreamingFallback)$.onStreamingFallback();d("tengu_streaming_fallback_to_non_streaming",{model:$.model,error:"404_stream_creation",attemptNumber:M6,maxOutputTokens:K8,thinkingType:_.type,request_id:T6,fallback_cause:"404_stream_creation"});try{let s=yield*odK({model:$.model,source:$.querySource},{model:$.model,fallbackModel:$.fallbackModel,thinkingConfig:_,...gK()&&{fastMode:g},signal:Y},O6,(h6,P6,V6)=>{M6=h6,K8=V6},(h6)=>oO8(h6,$.querySource),T6),$6={message:{...s,content:UF8(s.content,z,$.agentId)},requestId:H6??void 0,type:"assistant",uuid:ql8(),timestamp:new Date().toISOString(),...!1,...H&&{advisorModel:H}};X6.push($6),l6=$6,yield $6}catch(s){if(s instanceof Zw6)throw s;N(`Non-streaming fallback also failed: ${F6(s)}`,{level:"error"});let $6=s,h6=$.model;if(s instanceof Pm)$6=s.originalError,h6=s.retryContext.model;if($6 instanceof nq)tE8($6);let P6=H6||($6 instanceof nq?$6.requestID:void 0)||($6 instanceof nq?$6.error?.request_id:void 0);if(h77({error:$6,model:h6,messageCount:y.length,messageTokens:cN(y),durationMs:Date.now()-z6,durationMsIncludingRetries:Date.now()-n,attempt:M6,requestId:P6,clientRequestId:e,didFallBackToNonStreaming:I6,queryTracking:$.queryTracking,querySource:$.querySource,llmSpan:t,fastMode:f8,previousRequestId:O}),$6 instanceof c_){_6();

if(_=_.replace("/ws/","/session/"),!_.endsWith("/events"))_=_.endsWith("/")?_+"events":_+"/events";return`${K}//${q.host}${_}${q.search}`}var eBY=100,qgY=15000,KgY=3000,FK8;var Q$7=L(()=>{VK();_8();w$();tL();F$7();U$7();FK8=class FK8 extends gK8{postUrl;uploader;streamEventBuffer=[];streamEventTimer=null;constructor(q,K={},_,z,Y){super(q,K,_,z,Y);let{maxConsecutiveFailures:$,onBatchDropped:O}=Y??{};this.postUrl=_gY(q),this.uploader=new MM6({maxBatchSize:500,maxQueueSize:1e5,baseDelayMs:500,maxDelayMs:8000,jitterMs:1000,maxConsecutiveFailures:$,onBatchDropped:(A,w)=>{a8("error","cli_hybrid_batch_dropped_max_failures",{batchSize:A,failures:w}),O?.(A,w)},send:(A)=>this.postOnce(A)}),N(`HybridTransport: POST URL = ${this.postUrl}`),a8("info","cli_hybrid_transport_initialized")}async write(q){if(q.type==="stream_event"){if(this.streamEventBuffer.push(q),!this.streamEventTimer)this.streamEventTimer=setTimeout(()=>this.flushStreamEvents(),eBY);return}return await this.uploader.enqueue([...this.takeStreamEvents(),q]),this.uploader.flush()}async writeBatch(q){return await this.uploader.enqueue([...this.takeStreamEvents(),...q]),this.uploader.flush()}get droppedBatchCount(){return this.uploader.droppedBatchCount}flush(){return this.uploader.enqueue(this.takeStreamEvents()),this.uploader.flush()}takeStreamEvents(){if(this.streamEventTimer)clearTimeout(this.streamEventTimer),this.streamEventTimer=null;let q=this.streamEventBuffer;return this.streamEventBuffer=[],q}flushStreamEvents(){this.streamEventTimer=null,this.uploader.enqueue(this.takeStreamEvents())}close(){if(this.streamEventTimer)clearTimeout(this.streamEventTimer),this.streamEventTimer=null;this.streamEventBuffer=[];let q=this.uploader,K;Promise.race([q.flush(),new Promise((_)=>{K=setTimeout(_,KgY)})]).finally(()=>{clearTimeout(K),q.close()}),super.close()}async postOnce(q){let K=FD();if(!K){N("HybridTransport: No session token available for POST"),a8("warn","cli_hybrid_post_no_token");return}let _={Authorization:`Bearer ${K}`,"Content-Type":"application/json"},z;

this.currentState=q,this.workerState.enqueue({worker_status:q,requires_action_details:K?{tool_name:K.tool_name,action_description:K.action_description,request_id:K.request_id}:null})}reportMetadata(q){this.workerState.enqueue({external_metadata:q})}handleEpochMismatch(){N("CCRClient: Epoch mismatch (409), shutting down",{level:"error"}),a8("error","cli_worker_epoch_mismatch"),this.onEpochMismatch()}startHeartbeat(){this.stopHeartbeat();let q=()=>{let _=this.heartbeatIntervalMs*this.heartbeatJitterFraction*(2*Math.random()-1);this.heartbeatTimer=setTimeout(K,this.heartbeatIntervalMs+_)},K=()=>{if(this.sendHeartbeat(),this.heartbeatTimer===null)return;q()};q()}stopHeartbeat(){if(this.heartbeatTimer)clearTimeout(this.heartbeatTimer),this.heartbeatTimer=null}async sendHeartbeat(){if(this.heartbeatInFlight)return;this.heartbeatInFlight=!0;try{if((await this.request("post","/worker/heartbeat",{session_id:this.sessionId,worker_epoch:this.workerEpoch},"Heartbeat",{timeout:5000})).ok)N("CCRClient: Heartbeat sent")}finally{this.heartbeatInFlight=!1}}async writeEvent(q){if(q.type==="stream_event"){if(this.streamEventBuffer.push(q),!this.streamEventTimer)this.streamEventTimer=setTimeout(()=>void this.flushStreamEventBuffer(),YgY);return}if(await this.flushStreamEventBuffer(),q.type==="assistant")wgY(this.streamTextAccumulator,q);await this.eventUploader.enqueue(this.toClientEvent(q))}toClientEvent(q){let K=q;return{payload:{...K,uuid:typeof K.uuid==="string"?K.uuid:XnK()}}}async flushStreamEventBuffer(){if(this.streamEventTimer)clearTimeout(this.streamEventTimer),this.streamEventTimer=null;if(this.streamEventBuffer.length===0)return;let q=this.streamEventBuffer;this.streamEventBuffer=[];let K=AgY(q,this.streamTextAccumulator);await this.eventUploader.enqueue(K.map((_)=>({payload:_,ephemeral:!0})))}async writeInternalEvent(q,K,{isCompaction:_=!1,agentId:z}={}){let Y={payload:{type:q,...K,uuid:typeof K.uuid==="string"?K.uuid:XnK()},..._&&{is_compaction:!0},...z&&{agent_id:z}};

await this.internalEventUploader.enqueue(Y)}flushInternalEvents(){return this.internalEventUploader.flush()}async flush(){return await this.flushStreamEventBuffer(),this.eventUploader.flush()}async readInternalEvents(){return this.paginatedGet("/worker/internal-events",{},"internal_events")}async readSubagentInternalEvents(){return this.paginatedGet("/worker/internal-events",{subagents:"true"},"subagent_events")}async paginatedGet(q,K,_){let z=this.getAuthHeaders();if(Object.keys(z).length===0)return null;let Y=[],$;do{let O=new URL(`${this.sessionBaseUrl}${q}`);for(let[w,j]of Object.entries(K))O.searchParams.set(w,j);if($)O.searchParams.set("cursor",$);let A=await this.getWithRetry(O.toString(),z,_);if(!A)return null;Y.push(...A.data??[]),$=A.next_cursor}while($);return N(`CCRClient: Read ${Y.length} internal events from ${q}${K.subagents?" (subagents)":""}`),Y}async getWithRetry(q,K,_){for(let z=1;z<=10;z++){let Y;try{Y=await this.http.get(q,{headers:{...K,"anthropic-version":"2023-06-01","User-Agent":M$()},validateStatus:PnK,timeout:30000})}catch($){if(N(`CCRClient: GET ${q} failed (attempt ${z}/10): ${F6($)}`,{level:"warn"}),z<10){let O=Math.min(500*2**(z-1),30000)+Math.random()*500;await C7(O)}continue}if(Y.status>=200&&Y.status<300)return Y.data;if(Y.status===409)this.handleEpochMismatch();if(N(`CCRClient: GET ${q} returned ${Y.status} (attempt ${z}/10)`,{level:"warn"}),z<10){let $=Math.min(500*2**(z-1),30000)+Math.random()*500;await C7($)}}return N("CCRClient: GET retries exhausted",{level:"error"}),a8("error","cli_worker_get_retries_exhausted",{context:_}),null}reportDelivery(q,K){this.deliveryUploader.enqueue({eventId:q,status:K})}getWorkerEpoch(){return this.workerEpoch}get internalEventsPending(){return this.internalEventUploader.pendingCount}close(){if(this.closed=!0,this.stopHeartbeat(),h78(),this.streamEventTimer)clearTimeout(this.streamEventTimer),this.streamEventTimer=null;

var T45=`# Claude API — C#

> **Note:** The C# SDK is the official Anthropic SDK for C#. Tool use is supported via the Messages API. A class-annotation-based tool runner is not available; use raw tool definitions with JSON schema. The SDK also supports Microsoft.Extensions.AI IChatClient integration with function invocation.

## Installation

\`\`\`bash
dotnet add package Anthropic
\`\`\`

## Client Initialization

\`\`\`csharp
using Anthropic;

// Default (uses ANTHROPIC_API_KEY env var)
AnthropicClient client = new();

// Explicit API key (use environment variables — never hardcode keys)
AnthropicClient client = new() {
    ApiKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY")
};
\`\`\`

---

## Basic Message Request

\`\`\`csharp
using Anthropic.Models.Messages;

var parameters = new MessageCreateParams
{
    Model = Model.ClaudeOpus4_6,
    MaxTokens = 16000,
    Messages = [new() { Role = Role.User, Content = "What is the capital of France?" }]
};
var response = await client.Messages.Create(parameters);

// ContentBlock is a union wrapper. .Value unwraps to the variant object,
// then OfType<T> filters to the type you want. Or use the TryPick* idiom
// shown in the Thinking section below.
foreach (var text in response.Content.Select(b => b.Value).OfType<TextBlock>())
{
    Console.WriteLine(text.Text);
}
\`\`\`

---

## Streaming

\`\`\`csharp
using Anthropic.Models.Messages;

var parameters = new MessageCreateParams
{
    Model = Model.ClaudeOpus4_6,
    MaxTokens = 64000,
    Messages = [new() { Role = Role.User, Content = "Write a haiku" }]
};

await foreach (RawMessageStreamEvent streamEvent in client.Messages.CreateStreaming(parameters))
{
    if (streamEvent.TryPickContentBlockDelta(out var delta) &&
        delta.Delta.TryPickText(out var text))
    {
        Console.Write(text.Text);

}
}
\`\`\`

**\`RawMessageStreamEvent\` TryPick methods** (naming drops the \`Message\`/\`Raw\` prefix): \`TryPickStart\`, \`TryPickDelta\`, \`TryPickStop\`, \`TryPickContentBlockStart\`, \`TryPickContentBlockDelta\`, \`TryPickContentBlockStop\`. There is no \`TryPickMessageStop\` — use \`TryPickStop\`.

---

## Thinking

**Adaptive thinking is the recommended mode for Claude 4.6+ models.** Claude decides dynamically when and how much to think.

\`\`\`csharp
using Anthropic.Models.Messages;

var response = await client.Messages.Create(new MessageCreateParams
{
    Model = Model.ClaudeOpus4_6,
    MaxTokens = 16000,
    // ThinkingConfigParam? implicitly converts from the concrete variant classes —
    // no wrapper needed.
    Thinking = new ThinkingConfigAdaptive(),
    Messages =
    [
        new() { Role = Role.User, Content = "Solve: 27 * 453" },
    ],
});

// ThinkingBlock(s) precede TextBlock in Content. TryPick* narrows the union.
foreach (var block in response.Content)
{
    if (block.TryPickThinking(out ThinkingBlock? t))
    {
        Console.WriteLine($"[thinking] {t.Thinking}");
    }
    else if (block.TryPickText(out TextBlock? text))
    {
        Console.WriteLine(text.Text);
    }
}
\`\`\`

> **Deprecated:** \`new ThinkingConfigEnabled { BudgetTokens = N }\` (fixed-budget extended thinking) still works on Claude 4.6 but is deprecated. Use adaptive thinking above.

Alternative to \`TryPick*\`: \`.Select(b => b.Value).OfType<ThinkingBlock>()\` (same LINQ pattern as the Basic Message example).

---

## Tool Use

### Defining a tool

\`Tool\` (NOT \`ToolParam\`) with an \`InputSchema\` record. \`InputSchema.Type\` is auto-set to \`"object"\` by the constructor — don't set it. \`ToolUnion\` has an implicit conversion from \`Tool\`, triggered by the collection expression \`[...]\`.

\`\`\`csharp
using System.Text.Json;
using Anthropic.Models.Messages;

var L45=`# Claude API — Java

> **Note:** The Java SDK supports the Claude API and beta tool use with annotated classes. Agent SDK is not yet available for Java.

## Installation

Maven:

\`\`\`xml
<dependency>
    <groupId>com.anthropic</groupId>
    <artifactId>anthropic-java</artifactId>
    <version>2.17.0</version>
</dependency>
\`\`\`

Gradle:

\`\`\`groovy
implementation("com.anthropic:anthropic-java:2.17.0")
\`\`\`

## Client Initialization

\`\`\`java
import com.anthropic.client.AnthropicClient;
import com.anthropic.client.okhttp.AnthropicOkHttpClient;

// Default (reads ANTHROPIC_API_KEY from environment)
AnthropicClient client = AnthropicOkHttpClient.fromEnv();

// Explicit API key
AnthropicClient client = AnthropicOkHttpClient.builder()
    .apiKey("your-api-key")
    .build();
\`\`\`

---

## Basic Message Request

\`\`\`java
import com.anthropic.models.messages.MessageCreateParams;
import com.anthropic.models.messages.Message;
import com.anthropic.models.messages.Model;

MessageCreateParams params = MessageCreateParams.builder()
    .model(Model.CLAUDE_OPUS_4_6)
    .maxTokens(16000L)
    .addUserMessage("What is the capital of France?")
    .build();

Message response = client.messages().create(params);
response.content().stream()
    .flatMap(block -> block.text().stream())
    .forEach(textBlock -> System.out.println(textBlock.text()));
\`\`\`

---

## Streaming

\`\`\`java
import com.anthropic.core.http.StreamResponse;
import com.anthropic.models.messages.RawMessageStreamEvent;

MessageCreateParams params = MessageCreateParams.builder()
    .model(Model.CLAUDE_OPUS_4_6)
    .maxTokens(64000L)
    .addUserMessage("Write a haiku")
    .build();

try (StreamResponse<RawMessageStreamEvent> streamResponse = client.messages().createStreaming(params)) {
    streamResponse.stream()
        .flatMap(event -> event.contentBlockDelta().stream())
        .flatMap(deltaEvent -> deltaEvent.delta().text().stream())
        .forEach(textDelta -> System.out.print(textDelta.text()));

maximum 24 hours
- Results available for 29 days after creation
- 50% cost reduction on all token usage
- All Messages API features supported (vision, tools, caching, etc.)

---

## Create a Batch

\`\`\`python
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

client = anthropic.Anthropic()

message_batch = client.messages.batches.create(
    requests=[
        Request(
            custom_id="request-1",
            params=MessageCreateParamsNonStreaming(
                model="{{OPUS_ID}}",
                max_tokens=16000,
                messages=[{"role": "user", "content": "Summarize climate change impacts"}]
            )
        ),
        Request(
            custom_id="request-2",
            params=MessageCreateParamsNonStreaming(
                model="{{OPUS_ID}}",
                max_tokens=16000,
                messages=[{"role": "user", "content": "Explain quantum computing basics"}]
            )
        ),
    ]
)

print(f"Batch ID: {message_batch.id}")
print(f"Status: {message_batch.processing_status}")
\`\`\`

---

## Poll for Completion

\`\`\`python
import time

while True:
    batch = client.messages.batches.retrieve(message_batch.id)
    if batch.processing_status == "ended":
        break
    print(f"Status: {batch.processing_status}, processing: {batch.request_counts.processing}")
    time.sleep(60)

print("Batch complete!")
print(f"Succeeded: {batch.request_counts.succeeded}")
print(f"Errored: {batch.request_counts.errored}")
\`\`\`

---

## Retrieve Results

> **Note:** Examples below use \`match/case\` syntax, requiring Python 3.10+. For earlier versions, use \`if/elif\` chains instead.

\`\`\`python
for result in client.messages.batches.results(message_batch.id):
    match result.result.type:
        case "succeeded":
            msg = result.result.message
            text = next((b.text for b in msg.content if b.type == "text"), "")
            print(f"[{result.custom_id}] {text[:100]}")
        case "errored":
            if result.result.error.type == "invalid_request":
                print(f"[{result.custom_id}] Validation error - fix request and retry")
            else:
                print(f"[{result.custom_id}] Server error - safe to retry")
        case "canceled":
            print(f"[{result.custom_id}] Canceled")
        case "expired":
            print(f"[{result.custom_id}] Expired - resubmit")
\`\`\`

---

## Cancel a Batch

\`\`\`python
cancelled = client.messages.batches.cancel(message_batch.id)
print(f"Status: {cancelled.processing_status}")  # "canceling"
\`\`\`

---

## Batch with Prompt Caching

\`\`\`python
shared_system = [
    {"type": "text", "text": "You are a literary analyst."},
    {
        "type": "text",
        "text": large_document_text,  # Shared across all requests
        "cache_control": {"type": "ephemeral"}
    }
]

message_batch = client.messages.batches.create(
    requests=[
        Request(
            custom_id=f"analysis-{i}",
            params=MessageCreateParamsNonStreaming(
                model="{{OPUS_ID}}",
                max_tokens=16000,
                system=shared_system,
                messages=[{"role": "user", "content": question}]
            )
        )
        for i, question in enumerate(questions)
    ]
)
\`\`\`

---

## Full End-to-End Example

\`\`\`python
import anthropic
import time
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

client = anthropic.Anthropic()

# 1. Prepare requests
items_to_classify = [
    "The product quality is excellent!",
    "Terrible customer service, never again.",
    "It's okay, nothing special.",
]

requests = [
    Request(
        custom_id=f"classify-{i}",
        params=MessageCreateParamsNonStreaming(
            model="{{HAIKU_ID}}",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": f"Classify as positive/negative/neutral (one word): {text}"
            }]
        )
    )
    for i, text in enumerate(items_to_classify)
]

# 2. Create batch
batch = client.messages.batches.create(requests=requests)
print(f"Created batch: {batch.id}")

# 3. Wait for completion
while True:
    batch = client.messages.batches.retrieve(batch.id)
    if batch.processing_status == "ended":
        break
    time.sleep(10)

# 4. Collect results
results = {}
for result in client.messages.batches.results(batch.id):
    if result.result.type == "succeeded":
        msg = result.result.message
        results[result.custom_id] = next((b.text for b in msg.content if b.type == "text"), "")

for custom_id, classification in sorted(results.items()):
    print(f"{custom_id}: {classification}")
\`\`\`
`;

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