// Module: data

/* original: sqK */ var 0___T___Z_10_UTF8_AppleDTDPLIS=B((wFz)=>{
  var oqK=Ok1(),YFz=rqK();
  wFz.build=AFz;
  function $Fz(q){
    function K(_){
      return _<10?"0"+_:_
    }return q.getUTCFullYear()+"-"+K(q.getUTCMonth()+1)+"-"+K(q.getUTCDate())+"T"+K(q.getUTCHours())+":"+K(q.getUTCMinutes())+":"+K(q.getUTCSeconds())+"Z"
  }var OFz=Object.prototype.toString;
  function aqK(q){
    var K=OFz.call(q).match(/\[object (.*)\]/);
    return K?K[1]:K
  }function AFz(q,K){
    var _={
      version:"1.0",encoding:"UTF-8"
    },z={
      pubid:"-//Apple//DTD PLIST 1.0//EN",sysid:"http://www.apple.com/DTDs/PropertyList-1.0.dtd"
    },Y=YFz.create("plist");
    if(Y.dec(_.version,_.encoding,_.standalone),Y.dtd(z.pubid,z.sysid),Y.att("version","1.0"),Rr1(q,Y),!K)K={
      
    };
    return K.pretty=K.pretty!==!1,Y.end(K)
  }function Rr1(q,K){
    var _,z,Y,$=aqK(q);
    if($=="Undefined")return;
    else if(Array.isArray(q)){
      K=K.ele("array");
      for(z=0;
      z<q.length;
      z++)Rr1(q[z],K)
    }else if(Buffer.isBuffer(q))K.ele("data").raw(q.toString("base64"));
    else if($=="Object"){
      K=K.ele("dict");
      for(Y in q)if(q.hasOwnProperty(Y))K.ele("key").txt(Y),Rr1(q[Y],K)
    }else if($=="Number")_=q%1===0?"integer":"real",K.ele(_).txt(q.toString());
    else if($=="BigInt")K.ele("integer").txt(q);
    else if($=="Date")K.ele("date").txt($Fz(new Date(q)));
    else if($=="Boolean")K.ele(q?"true":"false");
    else if($=="String")K.ele("string").txt(q);
    else if($=="ArrayBuffer")K.ele("data").raw(oqK.fromByteArray(q));
    else if(q&&q.buffer&&aqK(q.buffer)=="ArrayBuffer")K.ele("data").raw(oqK.fromByteArray(new Uint8Array(q.buffer),K));
    else if($==="Null")K.ele("null").txt("")
  }
} /* confidence: 65% */

/* original: OFz */ var OFz=Object.prototype.toString;

/* original: q4K */ var composed_value=B((Sr1)=>{
  var tqK=S7K();
  Object.keys(tqK).forEach(function(q){
    Sr1[q]=tqK[q]
  });
  var eqK=sqK();
  Object.keys(eqK).forEach(function(q){
    Sr1[q]=eqK[q]
  })
} /* confidence: 30% */

/* original: O */ let composed_value=(await Promise.resolve().then(() => w6(q4K(),1))).parse(_.stdout)?.["Window Settings"]?.[K]; /* confidence: 30% */

/* original: aqK */ function helper_fn(q){
  var K=OFz.call(q).match(/\[object (.*)\]/);
  return K?K[1]:K
} /* confidence: 35% */

/* original: Rr1 */ function helper_fn(q,K){
  var _,z,Y,$=aqK(q);
   /* confidence: 35% */

/* original: MFz */ function Apple_Terminal_osascript_e_tel(){
  try{
    if(Y7.terminal!=="Apple_Terminal")return!1;
    let K=(await K1("osascript",["-e",'tell application "Terminal" to name of current settings of front window'])).stdout.trim();
    if(!K)return!1;
    let _=await K1("defaults",["export","com.apple.Terminal","-"]);
    if(_.code!==0)return!1;
    let O=(await Promise.resolve().then(() => w6(q4K(),1))).parse(_.stdout)?.["Window Settings"]?.[K];
    if(!O)return!1;
    return O.Bell===!1
  }catch(q){
    return j6(q),!1
  }
} /* confidence: 65% */

