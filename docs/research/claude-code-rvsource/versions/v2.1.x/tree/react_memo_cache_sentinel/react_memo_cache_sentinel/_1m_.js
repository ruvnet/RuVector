// Module: _1m_

/* original: rS8 */ var __esModule_u=B((Mc4)=>{
  Object.defineProperty(Mc4,"__esModule",{
    value:!0
  });
  Mc4.getOtlpEncoder=Mc4.encodeAsString=Mc4.encodeAsLongBits=Mc4.toLongBits=Mc4.hrTimeToNanos=void 0;
  var $Tz=jz(),wc1=Ac4();
  function jc1(q){
    let K=BigInt(1e9);
    return BigInt(Math.trunc(q[0]))*K+BigInt(Math.trunc(q[1]))
  }Mc4.hrTimeToNanos=jc1;
  function jc4(q){
    let K=Number(BigInt.asUintN(32,q)),_=Number(BigInt.asUintN(32,q>>BigInt(32)));
    return{
      low:K,high:_
    }
  }Mc4.toLongBits=jc4;
  function Hc1(q){
    let K=jc1(q);
    return jc4(K)
  }Mc4.encodeAsLongBits=Hc1;
  function Hc4(q){
    return jc1(q).toString()
  }Mc4.encodeAsString=Hc4;
  var OTz=typeof BigInt<"u"?Hc4:$Tz.hrTimeToNanoseconds;
  function wc4(q){
    return q
  }function Jc4(q){
    if(q===void 0)return;
    return(0,wc1.hexToBinary)(q)
  }var ATz={
    encodeHrTime:Hc1,encodeSpanContext:wc1.hexToBinary,encodeOptionalSpanContext:Jc4
  };
  function wTz(q){
    if(q===void 0)return ATz;
    let K=q.useLongBits??!0,_=q.useHex??!1;
    return{
      encodeHrTime:K?Hc1:OTz,encodeSpanContext:_?wc4:wc1.hexToBinary,encodeOptionalSpanContext:_?wc4:Jc4
    }
  }Mc4.getOtlpEncoder=wTz
} /* confidence: 65% */

/* original: OTz */ var composed_value=typeof BigInt<"u"?Hc4:$Tz.hrTimeToNanoseconds; /* confidence: 30% */

/* original: GTz */ var composed_value=rS8(),aS8=oS8(); /* confidence: 30% */

/* original: et6 */ var composed_value=oS8(),nTz=rS8(),iTz=256,rTz=512; /* confidence: 30% */

/* original: J */ let composed_value=(0,et6.createResource)(O),M={
  resource:composed_value,scopeSpans:w,schemaUrl:composed_value.schemaUrl
}; /* confidence: 30% */

/* original: oc4 */ var Tokenizer=B((ic4)=>{
  Object.defineProperty(ic4,"__esModule",{
    value:!0
  });
  ic4.ProtobufTraceSerializer=void 0;
  var nc4=iS8(),Kkz=Zc1(),_kz=nc4.opentelemetry.proto.collector.trace.v1.ExportTraceServiceResponse,zkz=nc4.opentelemetry.proto.collector.trace.v1.ExportTraceServiceRequest;
  ic4.ProtobufTraceSerializer={
    serializeRequest:(q)=>{
      let K=(0,Kkz.createExportTraceServiceRequest)(q);
      return zkz.encode(K).finish()
    },deserializeResponse:(q)=>{
      return _kz.decode(q)
    }
  }
} /* confidence: 36% */

/* original: Al4 */ var Tokenizer=B(($l4)=>{
  Object.defineProperty($l4,"__esModule",{
    value:!0
  });
  $l4.JsonTraceSerializer=void 0;
  var Mkz=Zc1();
  $l4.JsonTraceSerializer={
    serializeRequest:(q)=>{
      let K=(0,Mkz.createExportTraceServiceRequest)(q,{
        useHex:!0,useLongBits:!1
      });
      return new TextEncoder().encode(JSON.stringify(K))
    },deserializeResponse:(q)=>{
      if(q.length===0)return{
        
      };
      return JSON.parse(new TextDecoder().decode(q))
    }
  }
} /* confidence: 36% */

/* original: q */ let composed_value=D5(),K=u86(composed_value)!==null,_=Eh4()||K?tW1(composed_value):"Claude Opus 4.6",z=`\uD83E\uDD16 Generated with [Claude Code](${_26})`,Y=`Co-Authored-By: ${_} <noreply@anthropic.com>`,$=v7(); /* confidence: 30% */

/* original: jP6 */ function utility_fn(){
  return G8.modelStrings
} /* confidence: 40% */

/* original: _X9 */ function helper_fn(){
  if(jP6()!==null)return;
   /* confidence: 35% */

/* original: e9 */ function helper_fn(){
  let q=jP6();
   /* confidence: 35% */

/* original: LY6 */ function helper_fn(q){
  return q===e9().opus40||q===e9().opus41||q===e9().opus45||q===e9().opus46
} /* confidence: 35% */

/* original: u86 */ function Opus46_1m_Opus461Mcontext_Opus(q){
  switch(q){
    case e9().opus46:return"Opus 4.6";
    case e9().opus46+"[1m]":return"Opus 4.6 (1M context)";
    case e9().opus45:return"Opus 4.5";
    case e9().opus41:return"Opus 4.1";
    case e9().opus40:return"Opus 4";
    case e9().sonnet46+"[1m]":return"Sonnet 4.6 (1M context)";
    case e9().sonnet46:return"Sonnet 4.6";
    case e9().sonnet45+"[1m]":return"Sonnet 4.5 (1M context)";
    case e9().sonnet45:return"Sonnet 4.5";
    case e9().sonnet40:return"Sonnet 4";
    case e9().sonnet40+"[1m]":return"Sonnet 4 (1M context)";
    case e9().sonnet37:return"Sonnet 3.7";
    case e9().sonnet35:return"Sonnet 3.5";
    case e9().haiku45:return"Haiku 4.5";
    case e9().haiku35:return"Haiku 3.5";
    default:return null
  } /* confidence: 65% */

/* original: GH */ function helper_fn(q){
  let K=u86(q);
   /* confidence: 35% */

/* original: tW1 */ function helper_fn(q){
  let K=u86(q);
   /* confidence: 35% */

/* original: jc1 */ function helper_fn(q){
  let K=BigInt(1e9);
  return BigInt(Math.trunc(q[0]))*K+BigInt(Math.trunc(q[1]))
} /* confidence: 35% */

/* original: Hc4 */ function helper_fn(q){
  return jc1(q).toString()
} /* confidence: 35% */

/* original: vTz */ function helper_fn(q,K){
  let _=(0,GTz.getOtlpEncoder)(K);
  return{
    resourceLogs:kTz(q,_)
  }
} /* confidence: 35% */

/* original: dc4 */ function named_entity(q,K){
  return{
    attributes:q.attributes?(0,et6.toAttributes)(q.attributes):[],name:q.name,timeUnixNano:K.encodeHrTime(q.time),droppedAttributesCount:q.droppedAttributesCount||0
  }
} /* confidence: 70% */

/* original: tXK */ function opus_Opus_Opus46mostcapablefor(q=!1){
  return{
    value:!tw()?e9().opus46:"opus",label:"Opus",description:`Opus 4.6 Â· Most capable for complex work${Oi(q)}`,descriptionForModel:"Opus 4.6 - most capable for complex work"
  } /* confidence: 65% */

/* original: qPK */ function 1m_opus1m_Opus1Mcontext_Opus46(q=!1){
  return{
    value:!tw()?e9().opus46+"[1m]":"opus[1m]",label:"Opus (1M context)",description:`Opus 4.6 for long sessions${Oi(q)}`,descriptionForModel:"Opus 4.6 with 1M context window - for long sessions with large codebases"
  } /* confidence: 65% */

